"""
Detect test data with real noise

Simulated noise have flat noise power spectrum 2.25e-44
"""
import numpy as np
import pandas as pd
import h5py
import json
import argparse
from tqdm.auto import tqdm


def prepare(input_dir, output_dir):
    # Data
    submit = pd.read_csv(input_dir + '/sample_submission.csv')

    # std dev of p
    m = 2.25e-44
    s = 1.1869793104951747e-45

    outs = []

    for i, r in tqdm(submit.iterrows(), total=len(submit), ncols=78):
        file_id = r['id']

        filename = '%s/%s/%s.hdf5' % (input_dir, 'test', file_id)
        with h5py.File(filename, 'r') as f:
            g = f[file_id]

            a = g['H1']['SFTs'][:, :4096].astype(np.complex128)
            b = g['L1']['SFTs'][:, :4096].astype(np.complex128)

        pa = a.real**2 + a.imag**2
        pb = b.real**2 + b.imag**2

        # Mean at each time
        pa_mean = np.mean(pa, axis=0)
        pb_mean = np.mean(pb, axis=0)

        # Outlier
        out = np.sum(np.logical_or(pa_mean < m - 5*s, pa_mean > 3.0e-44))
        out += np.sum(np.logical_or(pb_mean < m - 5*s, pb_mean > 3.0e-44))

        outs.append(out)

    outs = np.array(outs)
    idx = outs >= 2

    print('Real noise %d / %d' % (np.sum(idx), len(idx)))

    ofilename = '%s/test_real_noise.h5' % output_dir
    with h5py.File('test_real_noise.h5', 'w') as f:
        f['idx'] = idx

    print(ofilename, 'written')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--settings', default='SETTINGS.json')
    arg = parser.parse_args()

    with open(arg.settings, 'r') as f:
        cfg = json.load(f)

    prepare(input_dir=cfg['KAGGLE_INPUT_DIR'],
            output_dir=cfg['OUTPUT_DIR'])
