from functools import partial
import multiprocessing
from modal import modal_elasticity_solution
import pickle
import numpy as np
from time import perf_counter

def mp_handler():
    t0 = perf_counter()
    nx = 4
    x = np.round(np.logspace(2,np.log10(30e3),nx))
    pool = multiprocessing.Pool(processes=2,maxtasksperchild=1)
    sol = partial(modal_elasticity_solution,verbose=1,writevtk=False)
    results = pool.map(sol,x)
    output = np.zeros((nx,3))
    output[:,1:3] = np.array(results)
    output[:,0] = x

    with open('modal.pickle', 'wb') as handle:
        pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print(f'TOTAL RUNTIME: {perf_counter()-t0}')

if __name__ == "__main__":
    mp_handler()
