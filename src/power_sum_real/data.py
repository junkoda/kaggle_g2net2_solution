import numpy as np
import pandas as pd
import h5py
import torch


def create_array(t, h, ch, x, mask):
    """
    Assign data to appropriate array element based on time

    Args:
      t (array): GPS time in sec
      h (array[complex64]): Fourier modes (STFT) at time t
      x (array[float32]): 2 x 360 x 5760
      mask (array[float32]): number of data - 2 x 360 x 5760
    """
    t = t - 1238166018
    assert np.all(t >= 0.0)

    T = 1800
    W = 5760

    dest = np.round(t / T).astype(int)  # nearest time bin
    for i, ibin in enumerate(dest):
        if ibin >= W:
            continue

        x[ch, :, ibin] = h[:, i]
        mask[ch, ibin] = 1


def normalize_time_dependence(x, mask):
    """
    Normalize with measured sigma2_time(t)
    x -> x / sigma_time(t)

    x (array): 2 x 360 x 5760
    mask (array): 2 x 5760

    Returns:
      sigma2: sigma2_time
      w = 1/sigma2 or 0 if sigma2=0
    """
    nch, H, W = x.shape

    # Normalize by freq-averaged P(t)
    p = x.real**2 + x.imag**2
    sigma2_time = np.mean(p, axis=1, keepdims=True)
    sigma_time = np.sqrt(sigma2_time)  # 2 x 1 x 5760
    x = x / np.clip(sigma_time, 1.0e-8, None)  # clip avoids division by zero; + epsilon also works
    p /= np.clip(sigma2_time, 1.0e-8, None)

    assert np.isfinite(x).all()

    ret = {'x': x,
           'sigma2_time': sigma2_time,
           'p': p}
    return ret


def mask_anomalous_frequency(x, mask, p, *, th=5.0):
    """
    x (array[complex64]): 2 x 360 x 5760, masked inplace
    p (array[float]): 2 x 360 x 5760
    """
    nch, H, W = p.shape
    assert nch == 2 and H == 360

    n_modes = np.sum(mask, axis=1)
    count = [0, 0]

    for ch in [0, 1]:
        idx = mask[ch].astype(bool)
        sig = (np.mean(p[ch, :, idx], axis=0) - 1) * np.sqrt(n_modes[ch])

        i_anomalous = np.arange(H)[sig > th]
        for i in i_anomalous:
            if i > 0:
                x[ch, i - 1] = 0
            x[ch, i] = 0
            if i + 1 < H:
                x[ch, i + 1] = 0
            count[ch] += 1

    return np.array(count)


def normalize(x, mask):
    """
    Normalize with measured sigma2 = sigma2_time * sigma2_freq

    x (array): 2 x 360 x 5760, modified inplace
    mask (array): 2 x 5760

    Returns:
      sigma2: sigma2_time x sigma2_freq
      w = 1/sigma2 or 0 if sigma2=0
    """
    nch, H, W = x.shape

    # Normalize by freq-averaged P(t)
    p = x.real**2 + x.imag**2
    sigma2_time = np.mean(p, axis=1, keepdims=True)
    sigma_time = np.sqrt(sigma2_time)
    x /= np.clip(sigma_time, 1.0e-8, None)

    # Normalize by residual freq-dependence P(𝜔)
    p = x.real**2 + x.imag**2
    sigma2_freq = (np.sum(p, axis=2, keepdims=True) /
                   np.sum(mask, axis=1).reshape(nch, 1, 1))
    sigma_freq = np.sqrt(sigma2_freq)
    x /= np.clip(sigma_freq, 1.0e-8, None)

    assert np.isfinite(x).all()

    sigma2 = sigma2_time * sigma2_freq  # 2 x 360 x 5760

    # w = 1/sigma2 or 0 if no data
    w = np.zeros((nch, H, W), dtype=np.float32)
    idx = sigma2 > 0
    w[idx] = 1 / sigma2[idx]
    assert np.isfinite(w).all()

    ret = {'w': w, }  # 1/sigma2 real-nise weight 2 x 360 x 5760
    return ret


#
# Dataset
#
class Dataset(torch.utils.data.Dataset):
    """
    d = dataset[i]
      x: 2 x 360 x 5760
      y: target 0 or 1
      mask: 1 if data exists in time bin, might be 2 for a rare case
    """
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i: int):
        d = self.data[i]

        x = np.zeros((2, 360, 5760), dtype=np.complex64)
        mask = np.zeros((2, 5760), dtype=np.float32)

        for ch, loc in enumerate(['H1', 'L1']):
            create_array(d[loc]['t'], d[loc]['h'], ch, x, mask)

        time_normalized = normalize_time_dependence(x, mask)

        # Remove anomalous frequency
        th = self.data.mask_threshold
        count = mask_anomalous_frequency(x, mask, time_normalized['p'], th=th)

        dd = normalize(x, mask)  # normalize x inplace

        ret = {'i': d['i'],
               'x': x,       # Fourier coefficient (complex64)
               'mask': mask,
               'anomalous_count': count,
               'w': dd['w'],  # 1/sigma2 weight
               'freq': d['frequency'],
               'y': np.float(d['y']),
               'realistic': d['realistic'],
               'significance': d['significance']}

        if 'signal' in d['H1']:
            ret['signal_H1'] = d['H1']['signal']
            ret['signal_L1'] = d['L1']['signal']

        if 'fdot' in d:
            ret['fdot'] = d['fdot']
            ret['f_offset'] = d['f_offset']

        return ret


#
# Data
#
class _Data:
    """
    data = Data()
    d = data[i]
      H1 - t, h
      L1 - t, h
    """
    def dataset(self):
        return Dataset(self)

    def loader(self, batch_size=1, num_workers=1, *, shuffle=False, drop_last=False):
        ldr = torch.utils.data.DataLoader(self.dataset(), num_workers=num_workers,
                shuffle=shuffle, batch_size=batch_size, drop_last=drop_last,
                pin_memory=True)

        return ldr


class Data(_Data):
    """
    Kaggle train/test data
    """
    def __init__(self, input_dir: str, output_dir: str,
                 data_type: str, ibegin=0, iend=None, *, df_inherit=None,
                 real_only=True, mask_threshold=10):
        """
        Kaggle provided data

        data_type (str): train or test
        df (pd.DataFrame): columns: id, target, realistic, and significance
        """
        self.input_dir = input_dir
        self.mask_threshold = mask_threshold

        if data_type == 'train':
            filename = '%s/train_labels.csv' % input_dir
            df = pd.read_csv(filename)
            df = df[df.target >= 0].reset_index()  # Remove y = -1
            df['realistic'] = True
        elif data_type == 'test':
            filename = '%s/sample_submission.csv' % input_dir
            df = load_test_df(input_dir, output_dir)
        else:
            raise ValueError('data_type must be train or test: {}'.format(data_type))

        # Inherit pervious significance
        if df_inherit is not None:
            assert len(df) == len(df_inherit)
            df = df_inherit
        else:
            df['significance'] = 0.0

        # ibegin:iend
        if ibegin > 0 or iend is not None:
            if iend is None:
                iend = len(df)
            else:
                assert iend <= len(df)
            df = df.iloc[ibegin:iend]
            print('data', ibegin, iend, len(df))
            assert len(df) > 0

        if real_only:
            df = df[df.realistic]

        self.data_type = data_type
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i: int):
        r = self.df.iloc[i]
        file_id = r.id

        filename = '%s/%s/%s.hdf5' % (self.input_dir, self.data_type, file_id)
        with h5py.File(filename, 'r') as f:
            g = f[file_id]
            d = {'i': r.name,
                 'y': r.target,
                 'realistic': r.realistic,
                 'frequency': g['frequency_Hz'][:],
                 'significance': r.significance}

            for loc in ['H1', 'L1']:
                h = g[loc]['SFTs'][:] * 1e22
                t = g[loc]['timestamps_GPS'][:]
                assert h.shape[0] == 360
                assert h.shape[1] == len(t)

                d[loc] = {'h': h, 't': t}

        return d


def load_test_df(input_dir, output_dir):
    # Load test metadata with real/sim noise column
    filename = input_dir + '/sample_submission.csv'
    df = pd.read_csv(filename)

    with h5py.File(output_dir + '/test_real_noise.h5', 'r') as f:
        real_noise = f['idx'][:]
    df['realistic'] = real_noise

    return df
