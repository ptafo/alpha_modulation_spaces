from scipy.interpolate import BSpline
from scipy.sparse.linalg import cg
import numpy as np
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import multiprocessing as mp
from functools import partial
from scipy.interpolate import griddata





def mySpline(n, L = 100):


    # This function returns a normalized B-spline of degree n
    # L:int support length
    # a:int represents the dilation factor
    # expand:int length of the signal to expand to

    n = int(n)
    b = BSpline.basis_element(range(n+1), 0)

    return b, (0,n), L



def myDil(fun,a,plot=0):

    # fun: function to dilate
    # a  : dilation factor

    f,(l,r), L = fun
    x          = np.linspace(l,r,np.int(L*a))
    y = np.power(np.abs(a), -1 / 2) * f(x)

    return y




def myTrans(wdw,L,index):

    # The function centers a vector wdw around the ctr'th element of a zero vector with length L
    # ctr must be greater than 0 and lower than L

    wdw = np.fft.fftshift(np.pad(wdw,(L//2,L//2), mode='constant'))
    wdw = wdw[np.arange(L).reshape(-1,1) - index]

    return wdw.T





def myMod(sig,omega,dt=1):

    # Discrete modulation of a signal with parameter omega and dt

    return np.exp(np.arange(len(sig)) * dt * 2 * np.pi * 1j * omega)*sig




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



def transform(array, transformation='db'):
    # Apply transformation to coefficients. Copied from the ltfatpy sourcecode.

    coef = np.copy(array)
    if transformation == 'db':
        coef = 20. * np.log10(np.abs(coef) + np.finfo(np.float32).tiny)
    elif transformation == 'dbsq':
        coef = 10. * np.log10(np.abs(coef) + np.finfo(np.float32).tiny)
    elif transformation == 'linsq':
        coef = np.square(np.abs(coef))
    elif transformation == 'linabs':
        coef = np.abs(coef)
    elif transformation == 'lin':
        if not np.isrealobj(coef):
            raise ValueError("Complex valued input cannot be plotted using the"
                             " 'lin' flag. Please use the 'linsq' or 'linabs' "
                             "flag.")
        else:
            # coef is returned in the output so we make a copy to avoid
            # returning a reference to the data passed in input
            coef = coef.copy()
    else:
        raise ValueError("Please use transformations of type 'db', 'dbsq', 'linsq', 'linabs' or 'lin'.")

    return coef



def plotlist(ci,atoms,freq_max,trans= "linabs"):

    # Plot a list of coefficients

    points2 = np.array([[b, a] for a, c, d in zip(*atoms) for b in d])
    grid_x, grid_y = np.meshgrid(atoms[2][-1], np.flip(np.arange(0,freq_max,1)))
    grid_z0 = griddata(points2, transform(ci, trans), (grid_x, grid_y), method='cubic')
    plt.imshow(grid_z0, extent=[0, 1, 0, freq_max], cmap='inferno', aspect='auto',vmin=min(ci.__abs__()), vmax=max(ci.__abs__()), interpolation="spline16")
    plt.colorbar()
    plt.show()





def atom_grid(N, eps, alp, fs=None, factor=1, plot=1):

    # return the coorbit frame grid for a discrete Signal
    # N: signal length
    # fs: frequency sampling
    # factor: reduce grid density to save computation time
    # return the atom


    if(fs==None): fs=N

    wj_max = fs/2
    J      = np.int((np.power(wj_max + 1, 1 - alp) - 1) * fs / (1 - alp) / eps)
    wjs    = p_alpha(np.arange(J+1)*eps/fs,alp)
    wjs    = wjs[::factor]
    Xjks   = list(map(lambda x:np.arange(N)[::int(max(1,eps*beta_alpha(x,alp)))],wjs))
    b_wjs  = beta_alpha(wjs,alp)
    atoms  = [wjs,b_wjs,Xjks]

    if plot:
        plt.figure()
        points2 = np.array([[b, a] for a, c, d in zip(*atoms) for b in d])
        plt.plot(points2[:,0],points2[:,1],'r.',markersize=1)

    return atoms




def frames(N,wind,atoms,dt=1):

    # function to generate the frame elements
    # N: length of signal
    # window: function
    # atoms: elements
    # dt: time discretization


    return np.vstack([myTrans(myMod(myDil(wind, c), a, dt), N, b) for a, c, b in zip(*atoms)]).reshape(-1,N)





def analysis_op(sig,frms):

    # generate the analysis operator based on the given frame elements

    return np.dot(frms.conj(),sig)



def recon_op(coefs,frms):

    # generate the reconstruction operator based on the given frame elements

    return np.dot(frms.T,coefs)



def frame_op(sig,frms):

    # return the frame representation of the signal

    A = np.dot(frms.T,frms.conj())

    return np.dot(A, sig)




def dual_frames(frms, iters=None, cores = 4 ):

    A    = np.dot(frms.T, frms.conj())
    pool = mp.Pool(processes=cores)
    cg_part = partial(cg,A,maxiter=iters)
    results = pool.map(cg_part, frms[:100])
    results = np.array([r[0] for r in results])
    pool.close()
    #tmp  = [ cg(A,b,maxiter=iters)[0] for b in frms]

    return results


def dual_frames_serial(frms, iters=None):

    A    = np.dot(frms.T, frms.conj())
    tmp  = [ cg(A,b,maxiter=iters)[0] for b in frms[:100]]

    return np.array(tmp)
