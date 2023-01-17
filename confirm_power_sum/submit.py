import numpy as np
import pandas as pd
import os
import json
import argparse
from sklearn.metrics import roc_auc_score

import significance
from data import load_test_df


def load_labels(input_dir):
    train_labels = pd.read_csv(input_dir + '/train_labels.csv')
    train_labels = train_labels[train_labels.target >= 0]
    assert len(train_labels) == 600

    y_true = train_labels.target.values

    return y_true


def evaluate_train(input_dir, output_dir, name):
    y_true = load_labels(input_dir)
    confirm = significance.load_confirm(output_dir, name, 'train')
    sig = confirm['sig_max']
    score = roc_auc_score(y_true, sig)

    print('%s train %.6f' % (name, score))


def create_submission0(input_dir: str, output_dir: str, name: str, *, exact=True):
    submit = load_test_df(input_dir, output_dir)
    idx_real = submit['realistic'].values
    idx_flat = ~idx_real

    confirm = significance.load_confirm(output_dir, name, 'test')
    sig = confirm['sig_max']

    submit0 = submit[['id', 'target']].copy()

    n = np.sum(idx_real)
    if len(sig) == n:
        submit0['target'].values[idx_flat] = sig
    else:
        nsig = len(sig)
        submit0['target'].values[idx_flat][:nsig] = sig
        print('Warning: Only %d / %d predictions assigned' % (nsig, n))

    if exact:
        # Set -0.1 to real noise prediction
        submit0['target'].values[idx_real] = -0.1
    else:
        # Set random [-0.1, 0] to real noise prediction
        np.random.seed(2022)
        submit0['target'].values[idx_real] = -np.random.uniform(0, 0.1, n)
    print('%d real noise test data are set to 0.' % n)

    print('target > 0', np.sum(submit0.target.values > 0))
    print('target > 0.5', np.sum(submit0.target.values > 0.501))
    print('total', len(submit0))

    suf = '_exact' if exact else ''
    ofilename = '%s/%s/submission0%s.csv.gz' % (output_dir, name, suf)
    submit0.to_csv(ofilename, index=False)
    print(ofilename, 'written')


def main():
    # Command-line options
    parser = argparse.ArgumentParser()
    parser.add_argument('name')
    parser.add_argument('--settings', default='SETTINGS.json')
    parser.add_argument('--data-type', default='test', help='train or test')
    parser.add_argument('--exact', default='true')
    arg = parser.parse_args()

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

    #evaluate_train(input_dir, output_dir, name)

    exact = arg.exact.lower() == 'true'
    create_submission0(input_dir, output_dir, name, exact=exact)

if __name__ == '__main__':
    main()
