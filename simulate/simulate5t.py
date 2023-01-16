"""
Simulate 5t
- Save template only: frequency and amplitude, 2 x 5760 each

$ python3 simulate5t.py --i=0:1001 --odir=10k
$ python3 simulate5t.py --10k=0

Requirement:
  output directories sim5y and temp
"""
import numpy as np
import os
import h5py
import time
import shutil
import argparse
import pyfstat

from gravlib import slack
import util
from subprocess import CalledProcessError
from scipy.optimize import curve_fit

# Functions
def sinc2(x, x0):
    return np.sinc(x - x0)**2


def fit_sinc(p, ch, t, i_max):
    """
    p (np.array): normalized power 2 x M x N
    """
    i = i_max[ch, t]
    i_begin = max(i - 10, 0)
    i_end = min(i + 11, p.shape[1])
    xx = np.arange(i_begin, i_end)
    yy = p[ch, i_begin:i_end, t]
    x0 = i

    # Brute-force search
    x0s = np.linspace(i - 0.5, i + 0.5, 101)  # 𝜟x0 = 0.01
    errs = []
    for x0 in x0s:
        y = sinc2(xx, x0)
        err = np.sum((yy - y)**2)
        errs.append(err)

    ii = np.argmin(errs)
    x0 = x0s[ii]
    err = errs[ii]

    popt, _ = curve_fit(sinc2, xx, yy, p0=(x0))
    x0_opt = popt[0]

    assert abs(x0_opt - x0) < 0.01

    err_opt = np.sum((yy - sinc2(xx, x0_opt))**2)
    assert err_opt <= err

    return x0_opt, err_opt


def fit_longterm(freq, deg=6):
    """
    freq (array[float]): len 5760
    """
    assert freq.ndim == 1
    x = np.arange(120) + (0.5 * 47 / 48)
    y = np.mean(freq.reshape(120, 48), axis=1)
    poly_coef = np.polyfit(x, y, deg)  # np.array (deg + 1, )
    poly = np.poly1d(poly_coef)

    t = np.arange(48 * 120) / 48
    y_fit = poly(t)

    return y_fit, poly_coef


def fit(p, i_max, ch, freq_out, err_out):
    # For each 5760 timestep
    for t in range(p.shape[2]):
        x0, err = fit_sinc(p, ch, t, i_max)

        freq_out[ch, t] = x0
        err_out[ch, t] = err

    assert np.isfinite(freq_out[ch]).all()
    assert np.isfinite(err_out[ch]).all()

#
# Main
#
# Command-line options
parser = argparse.ArgumentParser()
parser.add_argument('--i', default='0', help='index range 5:10')
parser.add_argument('--odir', default='default', help='output directory sim5f/odir')
parser.add_argument('--k', type=int, help='shortcut for setting --i and --odir in units of 10k')
parser.add_argument('--next-k', action='store_true', help='set --k automatically')
parser.add_argument('--depth', type=float, default=20.0, help='constant depth')
parser.add_argument('--overwrite', action='store_true')
parser.add_argument('--send-slack', default='true')

arg = parser.parse_args()

k = arg.k
if arg.next_k:
    for k in range(1, 100):
        if not os.path.exists('temp/%dk' % (10 * k)):
            break

if k is not None:
    istart = 10000 * (k - 1)
    iend = 10000 * k + 1
    odir = '%dk' % (10 * k)
    print('--k=%d: --i=%d:%d --odir=%s' % (k, istart, iend, odir))
else:
    istart, iend = util.get_range(arg.i, 10)
    odir = arg.odir

# Output directories
odir_base = 'sim5t'
if not os.path.exists(odir_base):
    raise FileNotFoundError(odir_base)

if os.path.exists('temp/%s' % odir) or os.path.exists('%s/%s/fit' % (odir_base, odir)):
    if not arg.overwrite:
        raise FileExistsError(odir)

odir_base = '%s/%s' % (odir_base, odir)

for sdir in ['parameters', 'fit']:
    odir_sub = '%s/%s' % (odir_base, sdir)
    if not os.path.exists(odir_sub):
        os.makedirs(odir_sub)
        print(odir_sub, 'created')

odir_fit = '%s/fit' % odir_base

odir_temp = 'temp/%s' % odir
if not os.path.exists(odir_temp):
    os.mkdir(odir_temp)
    

#
# Set simulation parameters
#
writer_kwargs = {'tstart': 1238166018,
                 'duration': 4 * 30 * 86400,  # 120 days
                 'detectors': 'H1,L1',
                 'sqrtSX': 0.5e-23,  # noise sigma 0.5e-23 is same as train data
                 'Tsft': 1800,
                 'SFTWindowType': 'tukey',
                 'SFTWindowBeta': 1e-8,
                 'Band': None,
                 'outdir': odir_temp,
                 'label': 'temp'}

depth = arg.depth
print('Depth', depth)

signal_parameters_generator = \
    pyfstat.AllSkyInjectionParametersGenerator(
        priors={'tref': writer_kwargs['tstart'],
                'F0': 1000.0,
                'F1': 0,
                'F2': 0,
                'h0': writer_kwargs['sqrtSX'] / depth,
                **pyfstat.injection_parameters.isotropic_amplitude_priors})


#
# Run simulator
#
def simulate(i: int, *, overwrite=False):
    ofilename = '%s/fit%06d.h5' % (odir_fit, i)
    if os.path.exists(ofilename) and not overwrite:
        raise FileExistsError(ofilename)

    #
    # Create signal
    #
    params = signal_parameters_generator.draw()  # random signal parameters

    writer_kwargs_template = writer_kwargs.copy()
    #writer_kwargs_template['Band'] = None
    writer = pyfstat.Writer(**writer_kwargs_template, **params)
    writer.sqrtSX = 0.0  # set noise to 0
    writer.make_data()

    # Load signal
    freq, t, signal5 = pyfstat.utils.get_sft_as_arrays(writer.sftfilepath)

    # Checks
    #H = signal5['H1'].shape[0]
    assert signal5['H1'].shape == signal5['L1'].shape
    assert np.array_equal(t['H1'], t['L1'])

    # Save parameter file
    shutil.copy('%s/temp.cff' % odir_temp,
                '%s/parameters/%06d.cff' % (odir_base, i))

    signal = np.concatenate([np.expand_dims(signal5['H1'], 0),
                             np.expand_dims(signal5['L1'], 0)], axis=0)

    # Signal array
    signal *= 1e22
    t = t['H1']
    freq_min = freq[0] * 1800  # in units of fundamental frequency 𝛥f = 1/1800 Hz

    #
    # Fit
    #
    print('Fitting...')
    nch, M, N = signal.shape
    assert nch == 2
    freq = np.zeros((nch, N), dtype=np.float32)
    freq_long = np.zeros((nch, N), dtype=np.float32)
    err = np.zeros((2, N), dtype=np.float32)

    # Power
    p = signal.real**2 + signal.imag**2
    i_max = np.argmax(p, axis=1)

    p_total = np.sum(p, axis=1)
    p = p / p_total.reshape(2, 1, 5760)

    # Fit 2 channels
    deg = 6  # degree of fitting polynomial
    poly_coef = np.zeros((2, deg + 1))
    for ch in range(nch):
        # Fit subgrid frequency with sinc
        fit(p, i_max, ch, freq, err)

        # Fit long-terxm modulation with a polynomial
        freq_long[ch], poly_coef[ch] = fit_longterm(freq[ch])

    # Write fit
    with h5py.File(ofilename, 'w') as f:
        f['f'] = freq               # frequency in frequency bin index (2, 5760)
        f['f_long'] = freq_long     # long-term frequency evolution (2, 5760)
        f['poly_coef'] = poly_coef  # Polynomial fit coefficients (deg + 1)
        f['freq_min'] = freq_min    # base frequency in units of fundamental frequency 1/1800 Hz
        f['p_total'] = p_total      # total power (2, 5760)

    print(ofilename, 'written')



# Simulation loop
tb = time.time()
overwrite = arg.overwrite

i = istart
while i < iend:
    try:
        simulate(i, overwrite=overwrite)
    except CalledProcessError:
        print('Error')
        continue

    i += 1

dt_hrs = (time.time() - tb) / 3600
print('Done %.2f' % dt_hrs)
if arg.send_slack == 'true':
    msg = 'simulate5t finished --i=%d:%d --odir=%s %.2f hr' % (istart, iend, odir, dt_hrs)
    slack.notify(msg)
