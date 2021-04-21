from DiscreteAlphaModulation import *



def exploreSig(sig,fs, eps, alp, len_wind, factor,name = "signal"):

    N = len(sig)
    atoms = atom_grid(N, eps, alp/10, fs, factor, 0)
    wind = mySpline(4, len_wind)
    F = frames(N, wind, atoms, 1 / fs)
    ao = analysis_op(sig, F)
    plt.figure()
    plotlist(ao, atoms, fs / 2)
    plt.savefig("./sig1/{}_{}_{}_{}_{}_{}".format(name,fs, eps, alp, len_wind, factor))

if __name__ == "__main__":

    N = 2000
    fs=N
    sig = doppler(40, 200, 2, N)

    eps = [500,400,300,200,100]
    alp = [9,8,7,6,5,4,6,3,2,1,0]
    len_wind =  [10,25,50,75,100,200,300,400,500,1000]
    factor = [1]



    pool = mp.Pool(processes=60)

    results = [pool.apply_async(exploreSig, args=(sig, fs, e, a, lw, f, "sig33"))
               for e in eps
               for a in alp
               for lw in len_wind
               for f in factor]

    output = [p.get() for p in results]


#sig1 doppler(40,10 ,2,200)  fs = 200
#sig2 doppler(40,200,2,200)  fs = 200
#sig3 doppler(40,200,2,2000) fs = 2000

