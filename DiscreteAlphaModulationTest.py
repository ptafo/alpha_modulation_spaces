from DiscreteAlphaModulation import *

wind = mySpline(4,100,1,0)[1]
sig  = doppler(40,10,2,200)

Ws,D = frame_grid(200,0.3,0,plot=1,factor=20)
tt  = [mySTFT2(sig,wind,D[w],w,1/200) for w in Ws]
plt.figure()
plotlist(tt,Ws)
plt.colorbar()




import scipy.io.wavfile as wave
(rate, SIG) = wave.read('./sample.wav')

wind = mySpline(4,100,1,0)[1]
sig  = SIG[:10000]

Ws,D = frame_grid(len(sig),0.01,0,fs=rate,plot=1,factor=1000)
tt  = [mySTFT2(sig,wind,D[w],w,1/rate) for w in Ws]
plt.figure()
plotlist(tt,Ws)
plt.colorbar()

