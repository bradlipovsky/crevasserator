'''

This script runs the parameter space study that examines stress intensity 
factors as a function of distance from the ice front.

'''

from time import perf_counter
import numpy as np
import pickle
from os import path
import crevasse as c # main module for this project
from importlib import reload
reload(c)

def main():
    # Output filename
    filename='../output/swell-sifs.pkl'
    c.test_filename(filename)
 
    # Geometry: domain width, domain height, 
    #  crevasse location, crevasse width, crevasse height
    geom = {'W':60000,
        'H':300,
        'Lc':15,
        'Wc':1,
        'Hc': 5,
        'fl':0,
        'swell_wavelength':1340.0,
        'ice_wavelength':4610.0}

    # Materials: Youngs modulus, poisson ratio, 
    #  ice density, water density, gravity
    mats = {'E':1e10, 'nu':0.3, 'rho':910, 'rhow':1024, 'g':9.81}

    D,flexural_gravity_wavelength, lam= c.fgl(mats,geom)

    number_of_locations = 10
    Lcs_swell = np.linspace(20,2*flexural_gravity_wavelength,
                           number_of_locations)

    output={}
    number_of_processors = min(number_of_locations,48)
    for min_or_max in ('min','max'):
        for mode in ('I','II'):
            for this_run in ('bottom','surface'):
                s = '%s K%s %s'%(this_run,mode,min_or_max)
                print ('Running simulation with %s '
                       'crevasses in Mode-%s.'%(this_run,mode))

                t1_start = perf_counter() 
                output[s] = c.call_pmap(geom,
                                        mats,
                                        this_run,
                                        mode,
                                        Lcs_swell,
                                        number_of_processors,
                                        extrema=min_or_max,
                                        verbose=True)
                t1_stop = perf_counter()

                print("Elapsed time in "
                        "outer loop: %f s."%(t1_stop-t1_start))

    with open(filename,'wb') as f:
        pickle.dump(output, f)
    print('Saved %s.\n\n'%filename)

if __name__ == "__main__":
    main()
