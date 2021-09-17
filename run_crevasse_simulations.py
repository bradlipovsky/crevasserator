# The following few lines of code are unethical and shouldn't be used every again. But they work.
# import os
# os.environ['PATH'] = "/home/bradlipovsky/anaconda3/envs/fenicsproject/bin:" + os.environ['PATH']
# print(os.environ['PATH'])

from dolfin import *
from mshr import *
import matplotlib.pyplot as plt
# %matplotlib notebook
from time import perf_counter
import numpy as np
import pickle
import scipy.integrate as integrate
import scipy.special as special
from os import path
import multiprocessing
from functools import partial
import crevasse as c # main module for this project
from importlib import reload
from scipy.optimize import fminbound
reload(c)
print (dolfin.__version__)



# Geometry: domain width, domain height,  crevasse location, crevasse width, crevasse height
geom = {'W':60000,'H':300,'Lc':15, 'Wc':1, 'Hc': 5}
    
# Materials: Youngs modulus, poisson ratio, ice density, water density, gravity
mats = {'E':1e10, 'nu':0.3, 'rho':910, 'rhow':1024, 'g':9.81}

run_names = ('bottom','surface')

D = mats['E']/(1-mats['nu']**2) * geom['H']**3 / 12
flexural_gravity_wavelength = 2*pi*(D/(mats['rhow']*mats['g']))**(1/4)

# Lcs_swell = Lcs
Lcs_swell = np.linspace(10,3*flexural_gravity_wavelength,10000)
# Lcs_swell = (100,200,300,600,1200,2400,4800,9600,19200)


output=[]
for mode in ('I','II'):
    for this_run in run_names:

        filename = 'swell-sifs-%s.pkl'%this_run

        if path.exists(filename):
            print('The simulation "%s" has already been run and saved. \n\
            To be safe, rename this file if you want to re-run.'%this_run)

        else:
            print ('Running simulation %s.'%this_run)

            g = geom
            t1_start = perf_counter() 

            output.append(c.call_pmap(this_run,mode))

            t1_stop = perf_counter()    
            print("Elapsed time in outer loop: %f s."%(t1_stop-t1_start))
        
with open('swell-sifs-high-res-linear.pkl', 'wb') as f:
    pickle.dump(output, f)
print('Saved %s.\n\n'%filename)

