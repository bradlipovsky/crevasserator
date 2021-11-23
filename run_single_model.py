import pickle
import crevasse as c # main module for this project

def main():
	# Geometry: domain width, domain height,  
	# crevasse location, crevasse width, crevasse height
	geom = {'W':60000,'H':300,'Lc':15, 'Wc':1, 'Hc': 5}

	# Materials: Youngs modulus, poisson ratio, 
	#  ice density, water density, gravity
	mats = {'E':1e10, 'nu':0.3, 'rho':910, 'rhow':1024, 'g':9.81}

	output = c.elasticity_solutions(case='full-minus-prestress',
		verbose=True,crevasse_location="surface",
                geometry=geom,swell_amplitude=0.0, swell_phase=0.0)

	with open('example_output.pkl', 'wb') as file:
		pickle.dump( output , file)	

if __name__ == "__main__":
	main()
