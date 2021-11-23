import crevasse as c # main module for this project
from dolfin import *

def main():
	# Geometry: domain width, domain height,  
	# crevasse location, crevasse width, crevasse height
    geom = {'W':60000,'H':300,'Lc':15, 'Wc':1, 'Hc': 5}

	# Materials: Youngs modulus, poisson ratio, 
	#  ice density, water density, gravity
    mats = {'E':1e10, 'nu':0.3, 'rho':910, 'rhow':1024, 'g':9.81}

    U,mesh = c.elasticity_solutions(case='full-minus-prestress',
                verbose=True,crevasse_location="surface",
                geometry=geom,swell_amplitude=0.0, swell_phase=0.0)
    fFile = HDF5File(MPI.comm_world,"example_output.h5","w") 
    fFile.write(U,"/f")
    fFile.close()
    File("example_mesh.xml") << mesh

if __name__ == "__main__":
    main()
