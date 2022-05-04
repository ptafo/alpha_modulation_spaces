from __future__ import division
from scipy.sparse.linalg import cg
import numpy as np
import multiprocessing as mp
from functools import partial
from window import myDil,myMod,myTrans



def frames(L,wind,atoms,dt=1):

    # function to generate the frame elements
    # L: length of signal
    # wind: window function
    # atoms: atom grid
    # dt: time discretization


    return np.vstack([myTrans(myMod(myDil(wind, a), omega, dt), L, i) for omega, a, i in zip(*atoms)]).reshape(-1,L)





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

    # parallel computing of dual frames

    A    = np.dot(frms.T, frms.conj())
    pool = mp.Pool(processes=cores)
    cg_part = partial(cg,A,maxiter=iters)
    results = pool.map(cg_part, frms)
    results = np.array([r[0] for r in results])
    pool.close()

    return results


def dual_frames_serial(frms, iters=None):

    # serial computing of dual frames

    A    = np.dot(frms.T, frms.conj())
    tmp  = [cg(A,b,maxiter=iters)[0] for b in frms]

    return np.array(tmp)


# def dual_frames_helper(frm, A, iters=None):
#
#     AA = A[:, frm != 0]
#     bool = np.sum(AA != 0, axis=1) != 0
#     AAA = A[bool][:, bool]
#     frt = frm[bool]
#
#     tmp  = cg(AAA,frt ,maxiter=iters)[0]
#     slt = np.zeros(frm.shape,dtype=complex)
#     slt[bool] = np.array(tmp)
#
#     return slt
#
# def dual_frames_serial_2(frms, iters=None):
#
#     A    = np.dot(frms.T, frms.conj())
#     tmp  = [dual_frames_helper(b,A,iters) for b in frms]
#     print('done!')
#
#     return np.array(tmp)
