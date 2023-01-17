"""
Create simulated signal and noise that has same timestep as test data

$ python3 simulate3.py --i=0:1001
"""
import numpy as np
import pandas as pd
import os
import glob
import h5py
import argparse
import pyfstat

from scipy import stats
import util
from subprocess import CalledProcessError


di = '/kaggle/input/g2net-detecting-continuous-gravitational-waves'
submit = pd.read_csv(di + '/sample_submission.csv')

# Command-line options
parser = argparse.ArgumentParser()
parser.add_argument('--i', default=':10', help='index range 5:10')
parser.add_argument('--odir', '-o', default='signal')
parser.add_argument('--depth', type=float, default=20.0, help='constant depth')
parser.add_argument('--overwrite', action='store_true')

arg = parser.parse_args()
istart, iend = util.get_range(arg.i, 10)

# Output dir
odir = 'sim3'
if not os.path.exists(odir):
    raise FileNotFoundError(odir)

odir = '%s/%s' % (odir, arg.odir)

if not os.path.exists(odir):
    os.mkdir(odir)
    print(odir, 'created')

# Set simulation parameters
depth = arg.depth
print('Depth', depth)

sqrtSX = 0.5e-23  # 0.5e-23 is same as train data, noise power = 2.25e-44

signal_parameters_generator = \
    pyfstat.AllSkyInjectionParametersGenerator(
        priors={'tref': 1238166018,
                'F0': {'uniform': {'low': 50.0, 'high': 500.0}},
                'F1': lambda: 10**stats.uniform(-12, 4).rvs(),
                'F2': 0,
                'h0': sqrtSX / depth,
                **pyfstat.injection_parameters.isotropic_amplitude_priors})


def simulate(i, *, overwrite=False):
    # Output
    ofilename = '%s/%06d.h5' % (odir, i)
    if os.path.exists(ofilename) and not overwrite:
        raise FileExistsError(ofilename)

    # Load timestamp from test data
    r = submit.iloc[i]
    file_id = r['id']

    filename = '%s/%s/%s.hdf5' % (di, 'test', file_id)
    with h5py.File(filename, 'r') as f:
        g = f[file_id]
        timestamps = {'H1': g['H1']['timestamps_GPS'][:],
                      'L1': g['L1']['timestamps_GPS'][:]}

    # Writer
    writer_kwargs = {'timestamps': timestamps,
                    'detectors': 'H1,L1',
                    'sqrtSX': sqrtSX,
                    'Tsft': 1800,
                    'SFTWindowType': 'tukey',
                    'SFTWindowBeta': 0.01,
                    'Band': 0.2,
                    'outdir': 'temp',
                    'label': 'temp'}

    # Random signal parameters
    params = signal_parameters_generator.draw()
    writer = pyfstat.Writer(**writer_kwargs, **params)

    # Compute SN ratio
    writer.make_data()
    snr = pyfstat.SignalToNoiseRatio.from_sfts(F0=writer.F0, sftfilepath=writer.sftfilepath)
    squared_snr = snr.compute_snr2(Alpha=writer.Alpha,
                                   Delta=writer.Delta,
                                   psi=writer.psi,
                                   phi=writer.phi,
                                   h0=writer.h0,
                                   cosi=writer.cosi)

    snrs = np.sqrt(squared_snr)
    print('SN', snrs)

    #
    # Signal only (freq1, t1, signal)
    #
    writer.sqrtSX = 0.0
    writer.make_data()
    freq1, t1, signal = pyfstat.utils.get_sft_as_arrays(writer.sftfilepath)

    signal_0 = signal['H1'][1:]
    signal_1 = signal['L1'][1:]
    H, _ = signal_0.shape  # 360 x variable n_timesteps

    a0 = signal_0.astype(np.complex128)
    p0 = a0.real**2 + a0.imag**2
    i_peak0 = np.argmax(p0, axis=0)

    a1 = signal_1.astype(np.complex128)
    p1 = a1.real**2 + a1.imag**2
    i_peak1 = np.argmax(p1, axis=0)

    # Frequency range of signal peak
    i_min = min(np.min(i_peak0), np.min(i_peak1))
    i_max = min(np.max(i_peak0), np.max(i_peak1))

    i_begin = max(0, i_min - 16)
    i_end = min(i_max + 1 + 16, 360)

    assert 0 <= i_begin and i_end <= 360

    signal = {'H1': signal_0[i_begin:i_end],
              'L1': signal_1[i_begin:i_end]}

    #
    # Noise only (freq0, t0, noise)
    #
    params['h0'] = 0
    writer.sqrtSX = 0.5e-23
    writer = pyfstat.Writer(**writer_kwargs, **params)
    writer.make_data()
    freq0, t0, noise = pyfstat.utils.get_sft_as_arrays(writer.sftfilepath)

    # Checks
    assert np.array_equal(t0['L1'], t1['L1'])
    assert np.array_equal(t0['H1'], t1['H1'])
    assert np.array_equal(freq0, freq1)
    freq = freq0[1:]

    #
    # Write
    #
    with h5py.File(ofilename, 'w') as f:
        f.attrs['sn'] = snrs
        f.attrs['depth'] = depth
        f['frequency'] = freq
        f['i_begin_signal'] = i_begin

        for loc in ['H1', 'L1']:
            g = f.create_group(loc)

            g['time'] = t0[loc]
            g['signal'] = signal[loc]
            g['noise'] = noise[loc][1:]
            #print('noise', noise[loc][1:].shape)

    print(ofilename, 'written')

    # Delete temporary files
    tmpfiles = glob.glob('temp/*.sft')
    for tmpfile in tmpfiles:
        os.remove(tmpfile)


# Simulation loop
overwrite = arg.overwrite

i = istart
while i < iend:
    try:
        simulate(i, overwrite=overwrite)
    except CalledProcessError:
        continue

    i += 1
