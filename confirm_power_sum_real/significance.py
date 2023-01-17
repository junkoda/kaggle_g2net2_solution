"""
Read search_power_sum_optimal output and update significance
The output can be used as: search_power_sum_optimal --inherit=test.parquet
"""
import numpy as np
import os
import math
import glob
import h5py


def load_search(output_dir, name: str, data_type: str):
    # Output directory
    odir = '%s/%s/%s' % (output_dir, name, data_type)
    if not os.path.exists(odir):
        raise FileNotFoundError(odir)

    # Output files
    filenames = glob.glob('%s/0*.h5' % odir)
    if not filenames:
        raise FileNotFoundError(odir)
    assert len(filenames) > 0
    filenames.sort()

    # Load common info
    filename = '%s/search_info.h5' % odir
    with h5py.File(filename, 'r') as f:
        c = f['clip'][()]

    # Clipped Gaussian statistics
    m = s = 1.0
    if c > 0:
        m = 1 - math.exp(-c)
        s = math.sqrt(1 - 2 * c * math.exp(-c) - math.exp(-2 * c))

    # Loop over power_sum output
    sigs = []
    signal_levels = []
    for filename in filenames:
        with h5py.File(filename, 'r') as f:
            power_sum_max = f['power_sum_max'][:]  # (n_templates, )
            if power_sum_max.ndim == 2:
                assert power_sum_max.shape[1] == 3
                w_sum = f['w_sum'][:]
            else:
                w_sum = 1
            w2_sum = f['w2_sum'][:]  # (n_templates, ) or (n_templates, 3)
            signal_level = f['signal_level'][()]

        sig = (power_sum_max - m * w_sum) / (s * np.sqrt(w2_sum))
        sig = np.max(sig, axis=0, keepdims=True)
        sigs.append(sig)
        signal_levels.append(signal_level)

    sigs = np.concatenate(sigs, axis=0)

    ret = {'sig_max': sigs,
           'signal_level': np.array(signal_levels),
           'clip': c}
    return ret


def load_confirm(output_dir: str, name: str, data_type: str):
    # Output directory
    odir = '%s/%s/%s' % (output_dir, name, data_type)
    if not os.path.exists(odir):
        raise FileNotFoundError(odir)

    # Output files
    filenames = glob.glob('%s/0*.h5' % odir)
    if not filenames:
        raise FileNotFoundError(odir)
    assert len(filenames) > 0
    filenames.sort()

    # Clipped Gaussian statistics
    m = s = 1.0

    # Loop over power_sum output
    sig_maxs = []
    signal_levels = []
    for filename in filenames:
        with h5py.File(filename, 'r') as f:
            power_sum_max = f['power_sum_max'][:]  # (n_templates, )
            if power_sum_max.ndim == 2:
                assert power_sum_max.shape[1] == 3
                w_sum = f['w_sum'][:]
            else:
                w_sum = 1
            w2_sum = f['w2_sum'][:]  # (n_templates, ) or (n_templates, 3)
            signal_level = f['signal_level'][:]  # (3, )

        sig = (power_sum_max - m * w_sum) / (s * np.sqrt(w2_sum))
        sig_max = np.max(sig, axis=0, keepdims=True)  # (1, 3)

        sig_maxs.append(sig_max)
        signal_levels.append(signal_level.reshape(1, -1))

    sig_maxs = np.concatenate(sig_maxs, axis=0)

    ret = {'sig_max': sig_maxs,
           'signal_level': np.concatenate(signal_levels, axis=0)}
    return ret
