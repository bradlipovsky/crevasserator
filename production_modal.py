from functools import partial
import multiprocessing
from modal import modal_elasticity_solution
import pickle
import numpy as np
from time import perf_counter

def mp_handler():
    t0 = perf_counter()
    nx = 96
    x = np.round(np.logspace(np.log10(10),np.log10(30e3),nx))
    pool = multiprocessing.Pool(processes=24,maxtasksperchild=1)
    LL = np.round(np.linspace(100,1000,96))
    for L in LL:
        sol = partial(modal_elasticity_solution,
                        verbose=0,
                        writevtk=False,
                        open_ocean_wavelength=L)
        results = pool.map(sol,x)
        output = np.zeros((nx,3))
        output[:,1:3] = np.array(results)
        output[:,0] = x

        with open(f'modal_{L}m.pickle', 'wb') as handle:
            pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print(f'TOTAL RUNTIME (L={L}m): {perf_counter()-t0}')

if __name__ == "__main__":
    mp_handler()
