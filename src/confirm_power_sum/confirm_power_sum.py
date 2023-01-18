import numpy as np
import os
import time
import json
import yaml
import argparse
import h5py
from datetime import datetime
from tqdm import tqdm

import torch
import torch.nn.functional as F

import util
from data import Data
from template import Template


def load_templates(template_dir, ibegin=0, iend=1000):
    """
    Load signal frequency and amplitude templates
    """
    if isinstance(ibegin, int):
        indices = range(ibegin, iend)
    elif isinstance(ibegin, np.ndarray):
        indices = ibegin
    else:
        raise TypeError('Unknown type for template indices {}'.format(type(ibegin)))

    freq_fits = []
    amps = []
    for i in indices:
        k = 10 * (i // 10_000 + 1)  # 0:10_000 -> 10k, 10_000:20_000 -> 20k
        filename = '%s/%dk/fit%06d.h5' % (template_dir, k, i)
        with h5py.File(filename, 'r') as f:
            freq_fit = f['f'][:]  # 2 x 5760
            freq_fits.append(freq_fit.reshape(1, 2, 5760))
            amp = f['p_total'][:]
            amps.append(amp.reshape(1, 2, 5760))

    freq_fits = np.concatenate(freq_fits, axis=0)  # (n_templates, 2, 5760)
    amps = np.concatenate(amps, axis=0)
    f_shifts = freq_fits - np.expand_dims(freq_fits[:, 0, 0], axis=(1, 2))

    mem = len(f_shifts) * 2 * 5760 * 4 * 2
    print('Load %d frequency templates (%.2f Mbytes)' % (len(f_shifts), mem / 1000**2))

    assert len(f_shifts) == len(indices)

    return f_shifts, amps


def create_shear_grid(fdots, H, W=5760):
    """
    Create grid for torch.grid_sample

    Args:
      fdots (array[float]): fdots in frequency bin per 120 days
                            nfdot x H x W x 2; 2 for xy
    """
    n_fdot = len(fdots)

    grid = torch.zeros((n_fdot, H, W, 2))
    grid.dtype

    x_grid = 2 / W * (torch.arange(W) + 0.5) - 1
    y0_grid = 2 / H * (torch.arange(H) + 0.5) - 1

    grid[:, :, :, 0] = x_grid.reshape(1, W)

    t = torch.arange(W) / W

    for i, fdot in enumerate(fdots):
        grid[i, :, :, 1] = y0_grid.reshape(H, 1) + (2 * fdot / H) * t.reshape(1, W)

    return grid


def confirm(x, H1_half, mask, freq, fshifts, amps, template, grid,
            freqs_confirm, fdots_confirm, fdots,
            device, *, return_all=False):
    """
    Confirm signal around (freq_confirm, fdot_confirm)

    Args:
      x (tensor): 1 x nch x 360 x 5760
      H1_half (int): Zoom in to frequency bins freq_confirm +- H1_half
      freq (array): data frequency in Hz (n_templates, )
      fshifts (tensor):
      fdots_confirm (array): freq bin / 120 days (n_templates, )
      fdots (array): fdots [freq bin / 120 days] in grid
    """
    n_templates = len(fshifts)
    assert len(amps) == n_templates
    assert len(freqs_confirm) == n_templates
    assert len(fdots_confirm) == n_templates

    H1 = 2 * H1_half

    # Prepare data in device
    x = x[0].to(device)  # 2 x H x W
    mask = mask[0].to(device)  # 2 x W

    assert x.ndim == 3
    nch, H, W = x.shape
    refine = template.refine
    H_template = template.H
    assert H == 360

    assert grid.shape[1] == (H1_half * 2 - H_template + 1) * refine

    fac = freq.mean().item() / 1000.0
    n_fdot = grid.size(0)

    # t for deDoppler time domain
    t = (torch.arange(H) / H - 0.5).reshape(1, H, 1)
    t = t.to(device)

    t_5760 = torch.arange(5760, dtype=torch.float) / 5760  # [0, 1) in 120 days
    t_5760 = t_5760.reshape(1, 5760).to(device)

    # Loop over templates
    power_sum_max = np.zeros(n_templates, dtype=np.float32)

    fdot_argmaxs = np.zeros(n_templates, dtype=np.float32)
    freq_argmaxs = np.zeros(n_templates, dtype=np.float32)
    w2_sum = np.zeros(n_templates, dtype=np.float32)
    power_sums = []
    for j in range(n_templates):
        # deDoppler shift
        fshift = fac * fshifts[j] + fdots_confirm[j] * t_5760

        x_t = torch.fft.ifft(x, dim=1)  # ifft freq to H=360 subtime
        shift = torch.exp(-1j * 2.0 * torch.pi * fshift.reshape(2, 1, W) * t)

        x_de = torch.fft.fft(x_t * shift, dim=1)  # 2 x H x 5760

        # Select frequency range
        ii = freqs_confirm[j]  # frequency bin (int)
        if ii - H1_half < 0:
            i_low = 0
            i_high = i_low + H1
        elif ii + H1_half > H:
            i_low = H - H1
            i_high = H
        else:
            i_low = ii - H1_half
            i_high = ii + H1_half

        assert 0 <= i_low and i_high <= H
        assert i_high - i_low == H1
        x_de = x_de[:, i_low:i_high]

        # Optimal weighting
        w = amps[j].reshape(2, W)  # 2 x W

        w /= torch.sum(w * mask)

        x_de *= w.sqrt().reshape(2, 1, W)

        # Apply sinc template
        p = template(x_de)  # -> (refine * (H - 1), 5760)
        # refine - H transpose already done

        # Shear fdot x freq x time 3D block
        # H_refined = refine * (H1 - 2); -2 = - H_template + 1
        H_refined, W = p.shape
        img = p.reshape(1, 1, H_refined, W).expand(n_fdot, 1, H_refined, W)

        y = F.grid_sample(img, grid, mode='nearest', padding_mode='zeros', align_corners=False)
        # -> n_fdot x 1 x H_refined x W

        y = y.sum(axis=3)
        if return_all:
            power_sums.append(y.cpu().numpy().reshape(1, n_fdot, H_refined))

        ymax, argmax = y.flatten().max(dim=0)
        argmax = argmax.item()
        assert y.shape[0] == n_fdot
        assert y.shape[2] == H_refined

        # best (fdot, freq)
        fdot_argmax, subgrid_freq_argmax = np.unravel_index(argmax, (n_fdot, H_refined))
        fdot_argmaxs[j] = fdots_confirm[j] + fdots[fdot_argmax]
        freq_argmaxs[j] = freqs_confirm[j] - H1_half + subgrid_freq_argmax / refine

        power_sum_max[j] = ymax.item()
        w2_sum[j] = torch.sum(mask * w**2).item()

    ret = {'power_sum_max': power_sum_max,  # (n_templates, )
           'fdot_argmax': fdot_argmaxs,     # (n_template, ) fdot index
           'freq_argmax': freq_argmaxs,     # (n_template, ) original freq bin
           'w2_sum': w2_sum}

    if return_all:
        # Hough map before taking max (n_templates, n_fdot, H)
        ret['power_sum'] = np.concatenate(power_sums, axis=0)

    return ret


def compute_signal_info(d):
    """
    Normalized as ~ sqrt(N_modes) S^2
      perfect_optimal = sum w SS* / sqrt(w2_sum)
      perfect_total = sum SS* / sqrt(N_modes)
    """
    if 'signal_H1' not in d:
        return None

    f0 = np.zeros(2, dtype=np.int16)

    x = torch.cat([d['signal_H1'], d['signal_L1']], dim=2)  # signal only (1, H, )

    p = x.real**2 + x.imag**2  # (1, H, W2) where W2 = W_H1 + W_L1
    p_total = p[0].sum(dim=0)  # total signal in all frequencies (W, )
    perfect_total = p_total.sum().item()
    n_modes = len(p_total)

    w = p_total / p_total.sum()  # sum(w) = 1 (W, )
    wp_sum = torch.sum(w * p_total).item()
    w2_sum = torch.sum(w**2).item()  # w2_sum = 1 / N_modes for w=1

    perfect_total /= np.sqrt(n_modes)
    perfect_optimal = wp_sum / np.sqrt(w2_sum)  # ~ sqrt(N_modes) S^2

    for ch, loc in enumerate(['H1', 'L1']):
        f0[ch] = p[0, :, 0].argmax().item()  # signal frequency at t=0 (freq bin)

    ret = {'perfect_total': perfect_total,
           'perfect_optimal': perfect_optimal,
           'f0': f0}
    return ret


def load_search_result(odir, i, idx_template, fdots_search, n_select):
    """
    i (int): data index
    """
    filename = '%s/%06d.h5' % (odir, i)
    with h5py.File(filename, 'r') as f:
        power_sum_max = f['power_sum_max'][:]  # (n_templates, )
        power_sum_argmax = f['power_sum_argmax'][:]
        w2_sum = f['w2_sum'][()]

    assert len(power_sum_max) == len(idx_template)

    sigma2 = 2.25
    sig = (power_sum_max - sigma2) / (sigma2 * np.sqrt(w2_sum))  # (n_templates)

    H = 360
    n_fdot = len(fdots_search)
    kk_sort = np.argsort(sig)[::-1][:n_select]

    fdots = []
    freqs = []
    for k in kk_sort:
        argmax = power_sum_argmax[k]
        assert 0 <= argmax < n_fdot * H
        ifdot, jfreq = np.unravel_index(argmax, (n_fdot, H))
        fdots.append(fdots_search[ifdot])
        freqs.append(jfreq)

    ret = {'idx': kk_sort,
           'sig': sig[kk_sort],
           'fdot': np.array(fdots),
           'freq': np.array(freqs)}
    return ret

#
# Main
#


def main():
    # Command-line options
    parser = argparse.ArgumentParser()
    parser.add_argument('name')
    parser.add_argument('--data', default='test')
    parser.add_argument('--settings', default='SETTINGS.json')
    parser.add_argument('--i', default=None)
    parser.add_argument('--overwrite', action='store_true')
    arg = parser.parse_args()

    overwrite = arg.overwrite

    # Data
    data_type = arg.data
    assert data_type in ['train', 'test']

    if arg.i is not None:
        ibegin, iend = util.get_range(arg.i, None)  # data indices
    elif data_type in ['train', 'test']:
        ibegin = 0
        iend = None
    else:
        ibegin, iend = 0, 1000

    # Settings
    with open(arg.settings, 'r') as f:
        settings = json.load(f)
    input_dir = settings['INPUT_DIR']
    output_dir = settings['OUTPUT_DIR']

    # Config
    filename_yml = arg.name.replace('.yml', '') + '.yml'
    name = os.path.basename(arg.name).replace('.yml', '')
    with open(filename_yml, 'r') as f:
        cfg = yaml.safe_load(f)

    # Output directory
    if not os.path.exists(output_dir):
        raise FileNotFoundError(output_dir)

    odir = '%s/%s/%s' % (output_dir, name, data_type)  # output/confirm_120_flat/test
    if not os.path.exists(odir):
        os.makedirs(odir)
        print(odir, 'created')

    device = torch.device('cuda')

    # Data
    if data_type == 'train' or data_type == 'test':
        data = Data(input_dir, output_dir, data_type, ibegin, iend)
    else:
        raise AssertionError

    # Load search info
    search_name = cfg['search']['name']
    search_dir = '%s/%s/%s' % (output_dir, search_name, data_type)
    filename = '%s/search_info.h5' % search_dir
    with h5py.File(filename, 'r') as f:
        fdots_search = f['fdots'][:]
        idx_template = f['template_indices'][:]

    # Confirmation parameters
    refine = cfg['confirm']['refine']
    print('Refine: %d' % refine)

    # Frequency slope [freq bin / 120 days]
    fdot_min, fdot_max = map(int, cfg['fdot'].split('..'))
    n_fdots = refine * (fdot_max - fdot_min) + 1
    fdots = np.linspace(fdot_min, fdot_max, n_fdots)
    print('fdots %d..%d (%d)' % (fdot_min, fdot_max, n_fdots))

    # Template indices
    template_dir = settings['TEMPLATE_DATA_DIR']
    fshifts, amps = load_templates(template_dir, idx_template)
    print('Templates %d...%d (%d)' %
          (idx_template[0], idx_template[-1] + 1, len(idx_template)))

    H_freq = cfg['confirm']['H_freq']
    H_template = cfg['confirm']['H_template']

    H1 = H_freq + H_template
    H1_half = H1 // 2
    H_refined = (2 * H1_half - H_template + 1) * refine

    # Shear grid
    fdots_refined = refine * fdots  # refined freq bin / 120 days
    grid = create_shear_grid(fdots_refined, H_refined).to(device)

    # Sinc frequency-direction weighting
    template = Template(H_template, refine)
    template.to(device)

    # Confirm top `n_select` templates
    n_select = cfg['confirm']['n_template_select']

    # Loop over data
    print('Begin', datetime.now().strftime('%Y-%m-%d %H:%M'))
    tb = time.time()

    loader = data.loader(batch_size=1, num_workers=1)
    for i, d in enumerate(tqdm(loader, ncols=78)):
        if d['realistic']:
            continue

        # Check output file
        ofilename = '%s/%06d.h5' % (odir, i + ibegin)
        if i > 10 and os.path.exists(ofilename) and not overwrite:
            raise FileExistsError(ofilename)

        # Load base search result
        search = load_search_result(search_dir, d['i'],
                                    idx_template, fdots_search, n_select)
        idx = search['idx']

        fshifts_confirm = torch.from_numpy(fshifts[idx]).to(device)
        amps_confirm = torch.from_numpy(amps[idx]).to(device)

        freqs_confirm = search['freq']
        fdots_confirm = search['fdot']

        # Compute
        #   d['x']: 1 x 2 x 360 x 5760
        ret = confirm(d['x'], H1_half, d['mask'], d['freq'],
                      fshifts_confirm, amps_confirm, template, grid,
                      freqs_confirm, fdots_confirm, fdots, device)

        # Write
        with h5py.File(ofilename, 'w') as f:
            f['template_indices'] = idx
            f['power_sum_max'] = ret['power_sum_max']
            f['fdot_argmax'] = ret['fdot_argmax'],  # (n_template, ) fdot
            f['freq_argmax'] = ret['freq_argmax']
            f['w2_sum'] = ret['w2_sum']

    # Time
    dt = time.time() - tb
    print('Time: %.2f sec' % dt)


if __name__ == '__main__':
    main()
