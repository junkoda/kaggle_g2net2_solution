"""
Create real-noise-only submission file
"""
import numpy as np
import os
import json
import argparse
from data import load_test_df

import significance


def create_submission_real(input_dir: str, output_dir: str,
                           name: str, *, flat=1, exact=True, ch=2):
    """
    Create submission file for real noise predictions only
    Flat data (80%) are set in [1000, 1000.1]

    Args:
      name: name of experiment / config yml
      flat (int): 0 or 1, value for flat data prediction
    """
    assert flat in [0, 1]
    submit = load_test_df(input_dir, output_dir)
    idx_real = submit['realistic'].values

    # Load significance
    sig = significance.load_confirm(output_dir, name, 'test')['sig_max']

    assert sig.ndim == 2 and sig.shape[1] == 3
    sig = sig[:, ch]
    ch = '_ch%d' % ch

    # Create submission.csv
    submit1 = submit[['id', 'target']].copy()

    if len(idx_real) == len(sig):
        submit1['target'].values[idx_real] = sig
    else:
        m = len(sig)
        submit1['target'].values[idx_real][:m] = sig
        print('Warning: Only %d / %d prediction assigned' % (len(sig), np.sum(idx_real)))
    #print('Real noise', np.sum(idx_real), len(sig))
    #assert np.sum(idx_real) == len(sig)

    # Set flat
    idx_flat = ~idx_real
    n = np.sum(idx_flat)
    np.random.seed(2022)

    if flat == 1:
        if exact:
            submit1['target'].values[idx_flat] = 1000.0
        else:
            submit1['target'].values[idx_flat] = np.random.uniform(1000, 1000.1, n)
        print('%d real noise test data are set to random sig ~1000' % n)
    elif flat == 0:
        if exact:
            submit1['target'].values[idx_flat] = -0.1
        else:
            submit1['target'].values[idx_flat] = -np.random.uniform(0, 0.1, n)
        print('%d real noise test data are set to random sig ~0' % n)
    else:
        raise AssertionError

    y_pred = submit1.target.values
    print('>1000', np.sum(y_pred >= 1000))
    print('<0', np.sum(y_pred < 0))
    print('Detected', np.sum(np.logical_and(7 < y_pred, y_pred < 1000)))
    print('Total', len(submit1))

    suf = '_exact' if exact else ''
    ofilename = '%s/%s/submission%d%s%s.csv.gz' % (output_dir, name, flat, suf, ch)
    submit1.to_csv(ofilename, index=False)
    print(ofilename, 'written')


def main():
    # Command-line options
    parser = argparse.ArgumentParser()
    parser.add_argument('name')
    parser.add_argument('--settings', default='SETTINGS.json')
    parser.add_argument('--flat', type=int, default=1, help='set flat to 0 or 1')
    parser.add_argument('--exact', default='true', help='set exactly 0 or 1')
    parser.add_argument('--ch', type=int, default=2)
    arg = parser.parse_args()

    name = arg.name.replace('.yml', '')

    # Settings
    with open(arg.settings, 'r') as f:
        settings = json.load(f)
    input_dir = settings['INPUT_DIR']
    output_dir = settings['OUTPUT_DIR']

    # Config
    name = os.path.basename(arg.name).replace('.yml', '')

    # Directory
    odir = '%s/%s' % (output_dir, name)
    if not os.path.exists(odir):
        raise FileNotFoundError(odir)

    exact = arg.exact.lower() == 'true'
    create_submission_real(input_dir, output_dir, name, exact=exact, ch=arg.ch)


if __name__ == '__main__':
    main()
