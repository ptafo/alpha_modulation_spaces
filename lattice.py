from __future__ import division
import numpy as np
import matplotlib.pyplot as plt




def p_alpha(b, j=1, alpha=0):

    # position map for alpha modulation spaces
    # input: b:float, j:int, alpha:float

    assert alpha >= 0 and alpha < 1,'Use alpha in [0,1)'
    assert b > 0, 'Use b > 0'

    return np.sign(j) * (np.power(1 + (1 - alpha) * b * np.abs(j), 1 / (1 - alpha)) - 1)





def s_alpha(b, j=1, alpha=0,version=0):

    # size map for alpha modulation spaces
    # input: b:float, j:int, alpha:float, version:(0:Fornasier,1:DahlkeTeschke)

    res = b * (1 + np.abs(p_alpha(b, j + np.sign(j) + (j == 0), alpha))) ** alpha, (1 + np.abs(p_alpha(b,j,alpha)))**alpha

    return res[version]





def atom_grid(N, a, b , alp, fs=None, plot=1):

    # return the coorbit frame grid for a discrete Signal
    # a frame grid consists of frequencies W, dilation parameters B (for each frequency) and time X (for each frequency)
    # N:array - signal length
    # a:int   - time sampling step
    # b:float - frequency step
    # alpha:float - modulation parameter
    # fs:int  - frequency sampling

    # test parameters
    # N, a, b, alp, fs, plot = 10000, 10, 1000, 0.2, 5000, 1


    if(fs==None): fs=N


    Js = np.floor(((1+fs/2)**(1-alp)-1) / (1-alp) / b)
    W = p_alpha(b, np.arange(Js+1), alp)
    X = list(map(lambda x: np.arange(N)[::int(max(1,a*s_alpha(b,x,alp,1)**-1))],np.arange(Js+1)))
    B = s_alpha(b,np.arange(Js+1),alp,1)**-1

    atoms = [W, B, X]

    if plot:
        plt.figure()
        points2 = np.array([[u, v] for v, c, d in zip(*atoms) for u in d])
        plt.plot(points2[:, 0], points2[:, 1], 'r.', markersize=1)



    print(f'{sum([len(listElem) for listElem in X])} points to compute')

    return atoms
