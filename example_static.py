'''

This script carries out a single model run. This is useful for testing
new features and for learning about how the code works. After the 
solution is calculated, it is output in a format that can be read
with ParaView.

'''

import crevasse as c # main module for this project
from dolfin import *

def main():
	# Geometry: domain width, domain height,  
	# crevasse location, crevasse width, crevasse height
    geom = {'W':60000,'H':300,'Lc':15, 'Wc':1, 'Hc': 5}

	# Materials: Youngs modulus, poisson ratio, 
	#  ice density, water density, gravity
    mats = {'E':1e10, 'nu':0.3, 'rho':910, 'rhow':1024, 'g':9.81}
    mats['mu'] = mats['E']/2./(1+mats['nu'])
    mats['lmbda'] = mats['E']*mats['nu']/(1+mats['nu'])/(1-2*mats['nu'])

    U,mesh = c.elasticity_solutions(case='full-minus-prestress',
                verbose=True,crevasse_location="surface",
                geometry=geom,swell_amplitude=0.0, swell_phase=0.0)
    
    File("output/example_mesh.xml") << mesh
    File("output/example.pvd") << U
    
    stress = c.sigma(U,mats['lmbda'],mats['mu'])
    TSpace = TensorFunctionSpace(mesh, "CG", 1)
    q = project(stress, TSpace)
    File("output/example_stress.pvd") << q

if __name__ == "__main__":
    main()
