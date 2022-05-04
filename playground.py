import scipy.io.wavfile as wave
from window import window
from lattice import atom_grid
from frame_ops import frames, dual_frames_serial,analysis_op,recon_op
from visual import plotlist
import numpy as np
import matplotlib.pyplot as plt



def doppler(A, B, expo=2, Fs=100):

    # test signal

    n   = np.linspace(0, 1, Fs)
    return np.sin(2 * np.pi * B * np.exp(-A * np.abs(n - 1 / 2) ** expo) * n)



#### signal
fs, sig = 2000, doppler(40, 200, 2, 2000)
# fs, sig = wave.read('./sample.wav')
# sig  = sig[:1000]


sig_len = sig.__len__()
fs = fs
dt = 1/fs
plt.plot(sig)


#### window
wind_length = sig_len/10
wind = window(('bsplines',4),wind_length)

#### atom grid
grid = atom_grid(N=sig_len, a=50, b=200, alp=0.0, fs=fs, plot=1)

#### frames analysis
frms = frames(sig_len,wind,grid,dt)
decomp = analysis_op(sig,frms)

plt.figure()
plotlist(decomp,grid,fs/2)

#### synthesis
dual_frms = dual_frames_serial(frms,10)
recomp = recon_op(decomp,dual_frms)

plt.figure()
plt.plot(sig)
plt.plot(recomp.real)
