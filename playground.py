import scipy.io.wavfile as wave
from window import window
from lattice import atom_grid
from frame_ops import frames, dual_frames_serial,analysis_op,recon_op,frame_algorithm
from visual import plotlist
import numpy as np
import matplotlib.pyplot as plt



def doppler(A, B, expo=2, Fs=100):

    # test signal

    n   = np.linspace(0, 1, Fs)
    return np.sin(2 * np.pi * B * np.exp(-A * np.abs(n - 1 / 2) ** expo) * n)



#### signal
fs = 2000
sig = doppler(40, fs/10, 2, fs)
# fs, sig = wave.read('./sample.wav')
# sig  = sig[:1000]


sig_len = sig.__len__()
fs = fs
dt = 1/fs
plt.plot(sig)


#### window
wind_length = 5*sig_len
wind = window(('bsplines',4),wind_length)

#### atom grid
grid = atom_grid(N=sig_len, a=2*fs, b=np.log(2), alp=0.999, fs=fs, plot=1)

#### frames analysis
frms = frames(sig_len,wind,grid,dt)
decomp = analysis_op(sig,frms)

plt.figure()
plotlist(decomp,grid,fs/2)

# #### synthesis
# dual_frms = dual_frames_serial(frms,100)
# recomp = recon_op(decomp,dual_frms)
#
# plt.figure()
# plt.plot(sig)
# plt.plot(recomp.real)


############### frame algorithm ###########
#from scipy import linalg as LA
from scipy.sparse import linalg as LA
#eigval = LA.eigvals(np.dot(frms.T,frms.conj()))
S = np.dot(frms.T,frms.conj())
eigW = LA.eigs(S,which='LM',return_eigenvectors=0,k=1)
lmd = 2/(eigW.max().__abs__())
fa = frame_algorithm(decomp,frms,200,lmd)
plt.figure()
plt.plot(sig)
plt.plot(fa)



#norm
from numpy import linalg as npLA
maxEig = (npLA.norm(frms.conj(),'fro')**2)#/min(frms.shape)
lmd = 10/(maxEig)
fa = frame_algorithm(decomp,frms,100,lmd)
plt.figure()
plt.plot(sig)
plt.plot(fa)





