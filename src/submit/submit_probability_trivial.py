"""
Convert `significance` (standardized power sum) to probability
using sigmoid function.

Complicated calculation does not matter at all; those are relic of failed attempt.
"""
import numpy as np
import pandas as pd
import argparse
import os
import json
import yaml
import scipy.stats
from scipy.special import expit
from scipy.optimize import bisect

from data import load_flat_real

# Fitting parameter to signal (exponental) and noise (Gumbel)
opt_gumbel = (6.1270753319524855, 0.24172757209375376)
opt_exp = (0.05388390072796784, 0.08766754432784142)
a_exp, s_exp = opt_exp


def f_gumbel(x, m, s):
    return scipy.stats.gumbel_r.pdf(x, m, s)


def f_exp(x, a, s):
    return a * np.exp(-s * x)


def get_sigmoid(scale):
    tt = 0.7310585786300049  # sigmoid(1) = 1 / (1 + exp(-1))
    assert scale > 0
    opt_exp1 = [opt_exp[0] / scale, opt_exp[1] / scale]

    # x0: where exp signal and gumbel noise become equal == detected threshold
    ff = lambda x: f_gumbel(x, *opt_gumbel) - f_exp(x, *opt_exp1)
    mu = opt_gumbel[0]
    x0 = bisect(ff, mu, 20)

    # x1: where signal / (signal + noise) become sigmoid(1) = 1/(1 + e^-1) = 0.73
    tt = 0.7310585786300049
    ff = lambda x: (1 - tt) * f_exp(x, *opt_exp1) - tt * f_gumbel(x, *opt_gumbel)
    mu = opt_gumbel[0]
    x1 = bisect(ff, mu, 20)

    # ss: sigmoid temperature
    ss = x1 - x0

    # Detected fraction model from signal exp
    th = x0
    a_exp = opt_exp1[0]
    s_exp = opt_exp1[1]
    detected_fraction_fit = 2 * a_exp / s_exp * np.exp(-s_exp * th)  # fraction / positive
    b = 0.5 * (1 - detected_fraction_fit)

    sigmoid = lambda x: b + (1 - b) * expit((x - th) / ss)

    ret = {'x0': x0,
           'ss': ss,
           'b': b,
           'f_detected': detected_fraction_fit,
           'sigmoid': sigmoid}
    return ret


def compute_prob_flat(sig):
    assert len(sig) == 6478

    sigmoid = get_sigmoid(1.0)
    f = sigmoid['sigmoid']
    print('sig', sig.shape)
    y_prob = f(sig)

    return y_prob


def compute_prob_real(sig):
    n = 1497
    assert len(sig) == n

    sigmoid = get_sigmoid(1.0)
    f = sigmoid['sigmoid']
    print('sig', sig.shape)
    y_prob = f(sig)

    return y_prob


def load(output_dir, name, i, *, ch=None, exact=True):
    ex = '_exact' if exact else ''
    ch = '_ch%d' % ch if ch is not None else ''

    filename = '%s/%s/submission%d%s%s.csv.gz' % (output_dir, name, i, ex, ch)
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
    input_dir = settings['INPUT_DIR']
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
    idx_flat = ~idx_real

    #
    # Flat
    #
    print('Flat', cfg['flat']['name'])
    assert isinstance(cfg['flat']['exact'], bool)
    submit = load(output_dir, cfg['flat']['name'], 0,
                  exact=cfg['flat']['exact'])
    sig_flat = submit['target'][idx_flat].values
    prob_flat = compute_prob_flat(sig_flat)
    print('prob flat computed', prob_flat.shape)

    #
    # Real
    #
    print('Real', cfg['real']['name'])
    assert isinstance(cfg['real']['exact'], bool)
    real = load(output_dir, cfg['real']['name'], 1,
                ch=cfg['real']['ch'], exact=cfg['real']['exact'])

    # Significance
    sig_real = real['target'][idx_real].values

    prob_real = compute_prob_real(sig_real)

    # Merge flat and real
    submit.target.values[idx_flat] = prob_flat
    submit.target.values[idx_real] = 0.999 * prob_real

    print('Target value range %.2f - %.2f' % (submit.target.min(), submit.target.max()))

    ofilename = '%s/submission_prob_trivial.csv.gz' % odir
    submit.to_csv(ofilename, index=False)

    print(ofilename, 'written')


if __name__ == '__main__':
    main()
