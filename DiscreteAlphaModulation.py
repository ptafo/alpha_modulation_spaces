from scipy.interpolate import BSpline
import numpy as np
import matplotlib.pyplot as plt




def mySpline(n, L = 100, a = 1, plot=1):


    # This function returns a normalize splines of degree n and length L
    # a represent the dilation factor

    n = int(n)
    b = BSpline.basis_element(range(n+1), 0)
    x = np.linspace(0, n, np.int(L*a))
    y = np.power(np.abs(a), -1/2)*b(x)


    if plot:
        plt.plot(range(np.int(L*a)), y, 'g', lw=3)

    return b,y



def myTrans(wdw,L,ctr):

    # The function centers a vector wdw around the ctr'th element of a zero vector with length L
    # cent must be greater than 0 and lower than L

    return np.pad(wdw, (L, L))[L + len(wdw) // 2 - ctr:][:L]




def myMod(sig,omega,dt=1):

    # Discrete modulation of a signal with parameter omega and dt

    return np.exp( np.arange(1,1+len(sig)) * dt * 2 * np.pi * 1j * omega)*sig




def doppler(A, B, expo=2, Fs=100):

    # test signal

    n   = np.linspace(0, 1, Fs)
    return np.sin(2 * np.pi * B * np.exp(-A * np.abs(n - 1 / 2) ** expo) * n)



def beta_alpha(omega, alpha):

    if alpha < 0 or alpha >= 1:
        raise ValueError('Use alpha in [0,1)')

    return np.power(1 + np.abs(omega), -alpha)



def p_alpha(omega, alpha):

    if alpha < 0 or alpha >= 1:
        raise ValueError('Use alpha in [0,1)')

    return np.sign(omega) * (np.power(1 + (1 - alpha) * np.abs(omega), 1 / (1 - alpha)) - 1)



def omega_j(epsilon, j, alpha):

    if alpha < 0 or alpha >= 1:
        raise ValueError('Use alpha in [0,1)')
    if epsilon <= 0:
        raise ValueError('Use epsilon > 0')

    return p_alpha(epsilon * j, alpha)




def plotlist(coeffs,yNames = None):

    # Plot a list of coefficients

    for i, ci in enumerate(coeffs):
        plt.imshow(ci.reshape(1, -1), extent=[0, 1000, i + 0.5, i + 1.5], cmap='inferno', aspect='auto',
               vmin = min([x.min() for x in coeffs]), vmax = max([x.max() for x in coeffs]))

    plt.ylim(0.5, len(coeffs) + 0.5)
    plt.yticks(np.arange(1, len(coeffs) + 1), yNames[np.arange(1, len(coeffs) + 1)-1].round(3))
    plt.show()





def mySTFT2(sig, wind, time, frequency, Dt=1):

    return np.array([abs(sum(sig * np.conjugate(myTrans(myMod(wind, frequency, Dt), len(sig), t-1)))) for t in time])


def frame_grid(N, eps, alp, fs=None, factor=1, plot=1):

    # return the coorbit frame grid for a discrete Signal
    # N: signal length
    # fs: frequency sampling
    # factor: reduce grid density to save computation time


    if(fs==None): fs=N

    wj_max = fs/2
    J      = np.int((np.power(wj_max + 1, 1 - alp) - 1) / (1 - alp) / eps)
    wjs    = p_alpha(np.arange(J+1)*eps,alp)
    wjs    = wjs[::factor]

    Dict = {}

    for i in wjs:
        #Dict[i] = np.arange(0,N + eps * beta_alpha(i, alp),eps * beta_alpha(i, alp))[1:-1]
        #Dict[i] = Dict[i]//fs
        #Dict[i] = np.unique(np.arange(N * fs / eps / beta_alpha(i, alp)) * eps * beta_alpha(i, alp) * fs // 1)

        #Dict[i] = np.unique(np.arange(N / fs / eps / beta_alpha(i, alp)) * eps * beta_alpha(i, alp) * fs // 1)
        Dict[i] = np.unique(np.arange(N+1,step= fs * eps * beta_alpha(i, alp)).astype(int))[1:]

    # plot gitter
    if(plot):
        plt.figure()
        for u in range(len(Dict)):
            plt.scatter(list(Dict.values())[u], [list(Dict.keys())[u]] * len(list(Dict.values())[u]),c='b',s=1)

    return wjs,Dict

