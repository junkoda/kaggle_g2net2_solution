"""
Grid search flat noise data
Use optimal weighting proportional to template amplitude

weighted_power_sum
  - weight normalization: sum(w) = 1
  - <wp_sum> = sigma2
  - Var wp_sum = w2_sum sigma2
  - significance = (wp_sum - sigma2) / (sqrt(w2_sum) * sigma2)
  - w2_sum = 1 / N_modes for constant weight
"""
import argparse
import numpy as np
import pandas as pd
import os
import json
import time
import yaml
import argparse
import h5py
from datetime import datetime
from tqdm import tqdm

import torch
import torch.nn.functional as F

import sim3
import util
from data import Data


def load_templates(template_dir, ibegin=0, iend=1000):
    freq_fits = []
    amps = []
    for i in range(ibegin, iend):
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

    return f_shifts, amps


def create_shear_grid(fdots, H=360, W=5760):
    """
    fdots (array[float]): fdots in frequency bin per 120 days
    
    Create grid for torch.grid_sample
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


def compute_max_power_sum(x, mask, freq, fshifts, amps, grid, device, *,
                          return_all=False):
    """
    Search fshifts templates x grid fdot shear,
    Take max over fdot x freq -> max_power_sum (n_templates,)

    x (tensor[complex64]): h  (batch_size, 2, 360, 5760)
    mask (tensor): (batch_size, 2, 5760)
    grid (tensor): n_fdot x H_in x W
    """
    # Prepare data in device
    batch_size, nch, H, W = x.shape
    assert batch_size == 1
    assert grid.shape[1] == H

    x = x[0].to(device)  # 2 x H x W
    mask = mask[0].to(device) # 2 x W

    fac = freq.mean().item() / 1000.0
    n_fdot = grid.size(0)

    t = (torch.arange(H) / H - 0.5).reshape(1, H, 1)
    t = t.to(device)

    # Reference perfect total
    n_modes = mask.sum().item()  # number of nonzero data H1 + L1

    # Loop over templates
    n_templates = len(fshifts)

    power_sum_max = np.zeros(n_templates, dtype=np.float32)
    power_sum_argmax = np.zeros(n_templates, dtype=np.int32)
    w2_sum = np.zeros(n_templates, dtype=np.float32)
    power_sums = []
    for j in range(n_templates):
        # deDoppler shift
        x_t = torch.fft.ifft(x, dim=1)  # ifft freq to H=360 subtime
        shift = torch.exp(-1j * 2.0 * torch.pi * fac * fshifts[j].reshape(2, 1, W) * t)

        x_de = torch.fft.fft(x_t * shift, dim=1)  # 2 x H x 5760

        # Optimal weighting
        w = amps[j].reshape(2, W)  # 2 x W
        w /= torch.sum(w * mask)

        # Compute power sum
        p = torch.sum(w.reshape(2, 1, W) * (x_de.real**2 + x_de.imag**2), dim=0)   #  H x W

        img = p.reshape(1, 1, H, W).expand(n_fdot, 1, H, W)
        y = F.grid_sample(img, grid, mode='nearest', padding_mode='zeros', align_corners=False)
        # -> n_fdot x 1 x H x W

        # Sum
        y = y.sum(axis=3)  # sum over all W time -> (n_fdot, 1, H)

        if return_all:
            power_sums.append(y.cpu().numpy().reshape(1, n_fdot, H))

        ymax, argmax = y.flatten().max(dim=0)
        power_sum_max[j] = ymax.item()
        power_sum_argmax[j] = argmax.item()
        w2_sum[j] = torch.sum(mask * w**2).item()

    ret = {'power_sum_max': power_sum_max,         # (n_templates, )
           'power_sum_argmax': power_sum_argmax,
           'w2_sum': w2_sum,
           'n_modes': n_modes}

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
    p_total = p[0].sum(dim=0)  # total signal in all frequencies  (W, )
    perfect_total = p_total.sum().item()
    n_modes = len(p_total)

    w = p_total / p_total.sum()  # sum(w) = 1  (W, )
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


#
# Main
#
def main():
    # Command-line options
    parser = argparse.ArgumentParser()
    parser.add_argument('name')
    parser.add_argument('--data', default='test')
    parser.add_argument('--i', default=None)
    parser.add_argument('--settings', default='SETTINGS.json')
    parser.add_argument('--overwrite', action='store_true')
    arg = parser.parse_args()

    # Allow overwriting existing output
    overwrite = arg.overwrite

    # Data
    data_type = arg.data
    assert data_type in ['train', 'test', 'sim3', 'noise']

    if arg.i is not None:
        ibegin, iend = util.get_range(arg.i, None)  # data indices
    elif data_type in ['train', 'test']:
        ibegin = 0
        iend = None
    else:
        ibegin, iend = 0, 1

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

    odir = '%s/%s/%s' % (output_dir, name, data_type)  # output/wide_120/test
    if not os.path.exists(odir):
        os.makedirs(odir)
        print(odir, 'created')

    device = torch.device('cuda')

    # Data
    if data_type == 'train' or data_type == 'test':
        data = Data(input_dir, output_dir, data_type, ibegin, iend)
    elif data_type == 'noise':
        data = sim3.Data(ibegin, iend, y=0)
    elif data_type == 'sim3':  # PyFStat data with test-like discontinous timestep
        data = sim3.Data(ibegin, iend, y=1)
    else:
        raise AssertionError

    # Frequency slope
    fdot_min, fdot_max = map(int, cfg['fdot'].split('..'))
    n_fdots = fdot_max - fdot_min + 1
    fdots = np.linspace(fdot_min, fdot_max, n_fdots)
    print('fdots %d..%d' % (fdot_min, fdot_max))

    # Template indices
    template_dir = settings['TEMPLATE_DATA_DIR']
    template_begin, template_end = map(int, cfg['templates'].split('...'))
    fshifts, amps = load_templates(template_dir, template_begin, template_end)
    fshifts = torch.from_numpy(fshifts).to(device)
    amps = torch.from_numpy(amps).to(device)
    print('templates %d...%d' % (template_begin, template_end))

    # Shear grid
    grid = create_shear_grid(fdots).to(device)

    # Write common info
    ofilename = '%s/search_info.h5' % odir
    with h5py.File(ofilename, 'w') as f:
        f['fdots'] = fdots
        f['template_indices'] = np.arange(template_begin, template_end)

    # Loop over data
    print('Begin', datetime.now().strftime('%Y-%m-%d %H:%M'))
    tb = time.time()

    updated = 0
    loader = data.loader(batch_size=1, num_workers=1)
    for i, d in enumerate(tqdm(loader, ncols=78)):
        if d['realistic']:
            continue

        updated += 1

        # Check output file
        ofilename = '%s/%06d.h5' % (odir, i + ibegin)
        if i > 10 and os.path.exists(ofilename) and not overwrite:
            raise FileExistsError(ofilename)

        # Compute
        ret = compute_max_power_sum(d['x'], d['mask'], d['freq'], fshifts, amps, grid, device)
        signal_info = compute_signal_info(d)

        # Write
        with h5py.File(ofilename, 'w') as f:
            f['power_sum_max'] = ret['power_sum_max']
            f['power_sum_argmax'] = ret['power_sum_argmax']
            f['w2_sum'] = ret['w2_sum']
            f['n_modes'] = ret['n_modes']
            if signal_info is not None:  # signal-only available in sim3 and sim5
                f['perfect_total'] = signal_info['perfect_total']
                f['perfect_optimal'] = signal_info['perfect_optimal']
                f_offset = d['f_offset'].item()
                f['f0'] = signal_info['f0'] + f_offset

        tqdm.write(ofilename)

    # Time
    dt = time.time() - tb
    dt_per_datum = dt / len(loader)
    if dt > 3600:
        print('Time: %.2f hr, %.2f sec per datum' % (dt / 3600, dt_per_datum))
    elif dt > 100:
        print('Time: %.2f min, %.2f sec per datum' % (dt / 60, dt_per_datum))
    else:
        print('Time: %.2f sec, %.2f sec per datum' % (dt, dt_per_datum))

    print('%d Done.' % updated)


if __name__ == '__main__':
    main()
