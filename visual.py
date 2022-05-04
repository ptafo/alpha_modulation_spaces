from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata






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

    # Plot a list of lists of coefficients

    points2 = np.array([[b, a] for a, c, d in zip(*atoms) for b in d])
    if(len(atoms[0])<2):
        plt.imshow(ci.reshape(1, -1).astype("float"), extent=[0, 10, 0, 1])
    else:
        grid_x, grid_y = np.meshgrid(atoms[2][-1], np.flip(np.arange(0, freq_max, 1)))
        grid_z0 = griddata(points2, transform(ci, trans), (grid_x, grid_y), method='cubic')
        plt.imshow(grid_z0, extent=[0, 1, 0, freq_max], cmap='inferno', aspect='auto', vmin=min(ci.__abs__()),
                   vmax=max(ci.__abs__()), interpolation="spline16")
    plt.colorbar()
    plt.show()



