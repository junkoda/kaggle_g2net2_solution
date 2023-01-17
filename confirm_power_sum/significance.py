"""
Read search_power_sum_optimal output and update significance
The output can be used as: search_power_sum_optimal --inherit=test.parquet
"""
import numpy as np
import os
import glob
import h5py


def load_search(name: str, data_type: str, *, max_only=True):
    # Output directory
    odir = '/kaggle/grav2/work/power_sum/experiments/%s/%s' % (name, data_type)
    if not os.path.exists(odir):
        raise FileNotFoundError(odir)

    # Output files
    filenames = glob.glob('%s/0*.h5' % odir)
    if not filenames:
        raise FileNotFoundError(odir)
    assert len(filenames) > 0
    filenames.sort()

    sigma2 = 2.25

    # Loop over power_sum output
    power_sums = []
    w2_sums = []
    sigs = []
    sig_maxs = []
    n_modess = []
    for filename in filenames:
        with h5py.File(filename, 'r') as f:
            power_sum_max = f['power_sum_max'][:]  # (n_templates, )
            w2_sum = f['w2_sum'][:]  # (n_templates, )
            n_modes = f['n_modes'][()]

        sig = (power_sum_max - sigma2) / (sigma2 * np.sqrt(w2_sum))
        assert sig.ndim == 1

        sig_max = np.max(sig)
        sigs.append(sig.reshape(1, -1))
        sig_maxs.append(sig_max)
        power_sums.append(power_sum_max.reshape(1, -1))
        w2_sums.append(w2_sum.reshape(1, -1))
        n_modess.append(n_modes)

    sig_max = np.array(sig_maxs)
    sig = np.concatenate(sigs, axis=0)
    power_sum_max = np.concatenate(power_sums, axis=0)

    ret = {'power_sum_max': power_sum_max,
           'sig': sig,
           'sig_max': sig_max,
           'w2_sum': np.concatenate(w2_sums, axis=0),
           'n_modes': np.array(n_modess)}
    return ret


def load_confirm(output_dir: str, name: str, data_type: str):
    odir = '%s/%s/%s' % (output_dir, name, data_type)

    sigma2 = 2.25
    filenames = glob.glob('%s/0*.h5' % odir)
    assert filenames
    filenames.sort()

    sig_maxs = []
    sigs = []
    power_sums = []
    w2_sums = []
    for filename in filenames:
        with h5py.File(filename, 'r') as f:
            power_sum_max = f['power_sum_max'][:]
            w2_sum = f['w2_sum'][:]

        sig = (power_sum_max - sigma2) / (sigma2 * np.sqrt(w2_sum))
        sig_maxs.append(np.max(sig))
        sigs.append(sig.reshape(1, -1))
        power_sums.append(power_sum_max.reshape(1, -1))
        w2_sums.append(w2_sum.reshape(1, -1))

    ret = {'sig_max': np.array(sig_maxs),
           'sig': np.concatenate(sigs, axis=0),
           'power_sum_max': np.concatenate(power_sums, axis=0),
           'w2_sum': np.concatenate(w2_sums, axis=0)}
    return ret
