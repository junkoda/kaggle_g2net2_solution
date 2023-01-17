"""
Last 10 min final submission
Set y_pred = 0.5 for anormlous noise, false positive candidates
Little change to the score
"""
import numpy as np
import pandas as pd
import os
import json
import yaml
import h5py
import argparse
from tqdm import tqdm

from data import Data, create_array, normalize_time_dependence, mask_anomalous_frequency
from data import load_flat_real

def remove_false_positive(data, i, r):
    #
    # Data
    #
    d = data[i]
    assert r.name == d['i']

    x = np.zeros((2, 360, 5760), dtype=np.complex64)
    mask = np.zeros((2, 5760), dtype=np.float32)

    for ch, loc in enumerate(['H1', 'L1']):
        create_array(d[loc]['t'], d[loc]['h'], ch, x, mask)

    time_normalized = normalize_time_dependence(x, mask)

    count = mask_anomalous_frequency(x, mask, time_normalized['p'])

    # 2nd loop
    time_normalized = normalize_time_dependence(x, mask)
    p = time_normalized['p']

    #
    # Plot image
    #
    #img = np.sum(p.reshape(2, 360, 120, 48), axis=3) / np.sum(mask.reshape(2, 1, 120, 48), axis=3)

    #
    # Frequency dependence
    #
    #for ch in [0, 1]:
    #    idx = mask[ch].astype(bool)
    #    p_mean = np.mean(p[ch, :, idx], axis=0)
    #    p_mean[p_mean == 0.0] = np.nan

    # std power
    stds = []
    std_maxs = []
    for ch in [0, 1]:
        idx = mask[ch].astype(bool)
        p_std = np.std(p[ch, :, idx], axis=0)
        p_std[p_std == 0.0] = np.nan
        stds.append(np.nanmean(p_std))
        std_maxs.append(np.nanmax(p_std))

    if (stds[0] > 1.1 or stds[1] > 1.1) and r.target > 0.5:
        print('%4d %.4f -> 0.5' % (r.name, r.target))
        return 0.5

    return r.target


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('name')
    parser.add_argument('--mask-threshold', type=float, default=10)
    parser.add_argument('--settings', default='SETTINGS.json')
    arg = parser.parse_args()

    # Settings
    with open(arg.settings, 'r') as f:
        settings = json.load(f)
    input_dir = settings['INPUT_DIR']
    output_dir = settings['OUTPUT_DIR']

    # Config
    filename_yml = arg.name.replace('.yml', '') + '.yml'
    name = os.path.basename(arg.name).replace('.yml', '')
    #with open(filename_yml, 'r') as f:
    #    cfg = yaml.safe_load(f)

    # Load prediction
    filename = '%s/%s/submission_prob_trivial.csv.gz' % (output_dir, name)
    submit = pd.read_csv(filename)

    idx_real = load_flat_real(output_dir)
    submit_real = submit.iloc[idx_real]

    data = Data(input_dir, output_dir, 'test',
                real_only=True, mask_threshold=arg.mask_threshold)
    n = len(data)
    print('n', n)

    y_pred = submit['target'].values
    for i in range(n):
        r = submit_real.iloc[i]
        y_pred[r.name] = remove_false_positive(data, i, r)

    ofilename = '%s/%s/submission_final.tar.gz' % (output_dir, name)
    submit.to_csv(ofilename, index=False)
    print(ofilename, 'written')