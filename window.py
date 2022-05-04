from __future__ import division
from scipy.interpolate import BSpline
import numpy as np
import matplotlib.pyplot as plt





def window(W, len):

    # create a window function
    # a window is defined by a function f, its support supp and its numbers of sampling points
    # input: W:tuple, len:int
    # e.g: window(lambda x:x/2,(0,1),10)


    if W[0] == 'bsplines':
        f = BSpline.basis_element(range(W[1] + 1), 0)
        supp = (0, W[1])
    elif W[0] == 'gauss':
        f = lambda x: np.exp(-np.pi * x**2 / W[1])
        supp = (-3 * W[1], 3 * W[1])
    else:
        f = W[0]
        supp = W[1]

    return f, supp, len



def myDil(wind, a):

    # dilate a window function by a given dilation parameter a and return a vector
    # input: win:window, a:double
    # e.g: myDil((lambda x:x/2,(0,1),10),1)


    f,(l,r), L = wind
    x = np.linspace(l,r,np.int((L-1)*a)+1)
    y = np.power(np.abs(a), -1 / 2) * f(x)

    return y




def myTrans(y, L, i):

    # center the vector y around the i-th element of a zeros vector with length L
    # input: y:array, L:int, i:int

    assert np.all(i >= 0) and np.all(i < L), "index out of range"
    y = np.fft.fftshift(np.pad(y, (L // 2, L // 2), mode='constant'))
    y = y[np.arange(L).reshape(-1,1) - i -1]

    return y.T



def myMod(sig, omega, dt=1):

    # Discrete modulation of a signal with  frequency parameter omega in Hertz and timestep dt in seconds
    # input: sig:array, omega:float, dt:float

    timestamp = np.arange(len(sig)) * dt

    return np.exp( 2 * np.pi * 1j * omega * timestamp)*sig




def plotFun(wind):

    # plot a window function

    plt.plot(myDil(wind, 1))



## application
# fs = 10
# dt = 1/fs
# wind = window(('bsplines',1),fs)
# plotFun(win)
# dilWin = myDil(wind,1)
# plt.plot(dilWin)
# transWin = myTrans(dilWin,200,30)
# plt.plot(transWin)
# modWin = myMod(dilWin,4,dt)
# plt.plot(modWin)


