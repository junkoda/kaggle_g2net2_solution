"""
Data class for PyFStat result

"""
import numpy as np
import h5py
import data


def random_copy(h: int, H: int):
    """
    Return random source and destination row range

    h: src height
    H: destination height (360)
    """
    assert H == 360

    ibegin_src = 0
    iend_src = h
    ibegin_dest = np.random.randint(-16, 360 + 16 - h)  # allow curring margins 16
    iend_dest = ibegin_dest + h

    if ibegin_dest < 0:
        ibegin_src = -ibegin_dest
        ibegin_dest = 0

    assert ibegin_src <= 16

    if iend_dest > 360:
        iend_src -= (iend_dest - 360)
        iend_dest = 360

    assert ibegin_src < iend_src
    assert iend_src >= h - 16
    assert iend_dest - ibegin_dest == iend_src - ibegin_src
    assert iend_dest - ibegin_dest >= h - 16

    src = slice(ibegin_src, iend_src)
    dest = slice(ibegin_dest, iend_dest)

    return src, dest


def compose(signal, noise):
    """
    Compose signal + noise data
    """
    h = signal['H1'].shape[0]
    H = noise['H1'].shape[0]

    assert signal['L1'].shape[0] == h
    assert noise['L1'].shape[0] == H
    assert signal['H1'].shape[1] == noise['H1'].shape[1]
    assert signal['L1'].shape[1] == noise['L1'].shape[1]

    src, dest = random_copy(h, H)
    offset = dest.start - src.start

    for loc in ['H1', 'L1']:
        noise[loc][dest] += signal[loc][src]

    return noise, offset


class Data(data._Data):
    def __init__(self, ibegin, iend, y, *, depth=20, signal_only=False):      
        assert y in [0, 1]
        df = data.load_test_df()
        if ibegin > 0 or iend is not None:
            iend = iend if iend is not None else len(df)
            df = df.iloc[ibegin:iend]

        self.y = y
        self.df = df
        self.fac = 20 / depth
        self.signal_only = signal_only

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]

        # Load mock
        path = '/scratch/kaggle/grav2/sim3/depth20'
        filename = '%s/%06d.h5' % (path, i)

        t = {}
        signal = {}
        noise = {}
        fac = self.fac

        with h5py.File(filename, 'r') as f:
            d = {'y': self.y,
                 'sn': f.attrs['sn'],
                 'realistic': False, #r.real,
                 'frequency': f['frequency'][:]}

            for loc in ['H1', 'L1']:
                g = f[loc]
                signal[loc] = (fac * 1e22) * g['signal'][:]
                noise[loc] = 1e22 * g['noise'][:]
                if self.signal_only:
                    noise[loc] = np.zeros_like(noise[loc])
                t[loc] = g['time'][:]

        if self.y == 1:
            _, offset = compose(signal, noise)
            d['f_offset'] = offset

        for loc in ['H1', 'L1']:
            d[loc] = {'h': noise[loc],
                      't': t[loc]}
            if self.y == 1:
                d[loc]['signal'] = signal[loc]

        return d
