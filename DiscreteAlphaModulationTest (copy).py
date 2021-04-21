from DiscreteAlphaModulation import *


N = 200
sig  = doppler(40,10,2,N)


N,eps,alp,fs,factor= 200,5,0.5,200,1
sig  = doppler(40,10,2,N)
atoms= atom_grid(N,eps,alp,fs,factor,0)
wind = mySpline(4,10)
F  = frames(N,wind,atoms,1/fs)
#F2 = frames2(N,wind,atoms,1/fs)
ao = analysis_op(sig,F)
rc = recon_op(ao,F)
fo = frame_op(sig,F)

plt.figure()
plotlist(ao,atoms,fs/2)
