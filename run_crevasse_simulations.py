from time import perf_counter
import numpy as np
import pickle
from os import path
import crevasse as c # main module for this project
from importlib import reload
reload(c)

# print (dolfin.__version__)

def main():
    # Output filename
    filename='swell-sifs.pkl'
    
    # Geometry: domain width, domain height, 
    #  crevasse location, crevasse width, crevasse height
    geom = {'W':60000,'H':300,'Lc':15, 'Wc':1, 'Hc': 5}

    # Materials: Youngs modulus, poisson ratio, 
    #  ice density, water density, gravity
    mats = {'E':1e10, 'nu':0.3, 'rho':910, 'rhow':1024, 'g':9.81}

    D,flexural_gravity_wavelength, lam= c.fgl(mats,geom)
    Lcs_swell = np.linspace(20,2*flexural_gravity_wavelength,10) 

    if path.exists(filename):
        print('The output filename has already been used. \n\
                To be safe, rename this file if you want to re-run.')
        val = input("Type YES to continue.")
        if val!='YES':
            return

    output={}
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
                                        mode,Lcs_swell,96,
                                        extrema=min_or_max,
                                        verbose=False)
                t1_stop = perf_counter()

                print("Elapsed time in "
                        "outer loop: %f s."%(t1_stop-t1_start))

    with open(filename,'wb') as f:
        pickle.dump(output, f)
    print('Saved %s.\n\n'%filename)

if __name__ == "__main__":
    main()
