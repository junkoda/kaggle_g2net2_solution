import pandas as pd
import argparse
import os
import json
import yaml

from data import load_flat_real


def load(path, name, i, *, ch=None, exact=False):
    ex = '_exact' if exact else ''
    ch = '_ch%d' % ch if ch is not None else ''

    filename = '%s/%s/submission%d%s%s.csv.gz' % (path, name, i, ex, ch)
    print('Read', filename)

    return pd.read_csv(filename)


def main():
    # Command-line options
    parser = argparse.ArgumentParser()
    parser.add_argument('name')
    parser.add_argument('--settings', default='SETTINGS.json')
    arg = parser.parse_args()

    name = arg.name.replace('.yml', '')

    # Settings
    with open(arg.settings, 'r') as f:
        settings = json.load(f)
    output_dir = settings['OUTPUT_DIR']

    # Config
    filename_yml = arg.name.replace('.yml', '') + '.yml'
    name = os.path.basename(arg.name).replace('.yml', '')
    with open(filename_yml, 'r') as f:
        cfg = yaml.safe_load(f)

    # Output directory
    if not os.path.exists(output_dir):
        raise FileNotFoundError(output_dir)

    odir = '%s/%s' % (output_dir, name)
    if not os.path.exists(odir):
        os.mkdir(odir)
        print(odir, 'created')

    idx_real = load_flat_real(output_dir)

    # Flat
    print('Flat', cfg['flat']['name'])
    assert isinstance(cfg['flat']['exact'], bool)
    submit = load(output_dir, cfg['flat']['name'], 0,
                  exact=cfg['flat']['exact'])

    # Real
    print('Real', cfg['real']['name'])
    assert isinstance(cfg['real']['exact'], bool)
    real = load(output_dir, cfg['real']['name'], 1,
                ch=cfg['real']['ch'], exact=cfg['real']['exact'])

    # Merge flat and real
    submit.target.values[idx_real] = real.target.values[idx_real]

    print('Target value range %.2f - %.2f' % (submit.target.min(), submit.target.max()))

    ofilename = '%s/submission_significance.csv.gz' % odir
    submit.to_csv(ofilename, index=False)

    print(ofilename, 'written')


if __name__ == '__main__':
    main()
