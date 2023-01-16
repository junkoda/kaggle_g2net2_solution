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


def _load_power_mean(r):
    # Frequency averaged power at each time
    di = '/kaggle/input/g2net-detecting-continuous-gravitational-waves'

    file_id = r.id
    filename = '%s/test/%s.hdf5' % (di, file_id)
    p_mean = {}
    with h5py.File(filename, 'r') as f:
        g = f[file_id]

        for loc in ['H1', 'L1']:
            a = g[loc]['SFTs'][:] * 1e22
            p = a.real**2 + a.imag**2
            p_mean[loc] = np.mean(p, axis=0)

    return p_mean


def _add_realistic_noise(noise, p_mean):
    """
    Random Gaussian with sigma(t)^2 = p_mean(t) - 2.25 is added
    noise (np.array[complex64])
    """
    n = noise.shape[1]
    assert len(p_mean) == n

    for i in range(n):
        if p_mean[i] > 2.25:
            s = np.sqrt(p_mean[i] - 2.25)
            noise[:, i].real += np.random.normal(0, s, size=360)
            noise[:, i].imag += np.random.normal(0, s, size=360)


class Data(data._Data):
    def __init__(self, ibegin=0, iend=None, *, y=1, depth=20, signal_only=False,
                 real_only=True, mask_threshold=10):
        assert y in [0, 1]
        self.mask_threshold = mask_threshold

        df = data.load_test_df()
        if ibegin > 0 or iend is not None:
            iend = iend if iend is not None else len(df)
            df = df.iloc[ibegin:iend]

        if real_only:
            df = df[df.realistic]

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
        filename = '%s/%06d.h5' % (path, r.name)

        t = {}
        signal = {}
        noise = {}
        fac = self.fac

        with h5py.File(filename, 'r') as f:
            d = {'i': r.name,
                 'y': self.y,
                 'sn': f.attrs['sn'],
                 'realistic': r.realistic,
                 'frequency': f['frequency'][:],
                 'significance': 0.0}

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

        # Add time-dependent noise
        if r.realistic:
            p_mean = _load_power_mean(r)
            for loc in ['H1', 'L1']:
                _add_realistic_noise(noise[loc], p_mean[loc])

        for loc in ['H1', 'L1']:
            d[loc] = {'h': noise[loc],
                      't': t[loc]}
            if self.y == 1:
                d[loc]['signal'] = signal[loc]

        return d
