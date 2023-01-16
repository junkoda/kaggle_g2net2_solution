import numpy as np
import pandas as pd
import h5py
import torch


def create_array(t, h, ch, x, mask):
    """
    Assign data to appropriate array element based on time

    Args:
      t (array): GPS time in sec
      h (array[complex64]): Fourier mode
      x (array[float32]): |h|**2 - 2 x 360 x 5760
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

        ret = {'i': d['i'],
               'x': x,       # Fourier coefficient (complex64)
               'mask': mask,
               'freq': d['frequency'],
               'y': np.float(d['y']),
               'realistic': d['realistic']}

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
                 data_type: str, ibegin=0, iend=None):
        """
        Kaggle provided data

        data_type (str): train or test
        df (pd.DataFrame): columns: id, target, realistic
        """
        self.input_dir = input_dir
        if data_type == 'train':
            filename = '%s/train_labels.csv' % input_dir
            df = pd.read_csv(filename)
            df = df[df.target >= 0].reset_index()  # Remove y = -1, truth unknown
            df['realistic'] = False
        elif data_type == 'test':
            filename = '%s/sample_submission.csv' % input_dir
            df = load_test_df(input_dir, output_dir)
        else:
            raise ValueError('data_type must be train or test: {}'.format(data_type))

        # ibegin:iend
        if ibegin > 0 or iend is not None:
            if iend is None:
                iend = len(df)
            else:
                assert iend <= len(df)
            df = df.iloc[ibegin:iend]
            print('data', ibegin, iend, len(df))
            assert len(df) > 0

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
                 'frequency': g['frequency_Hz'][:]}

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
