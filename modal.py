'''
Fenics script to do a harmonic analysis of surface gravity waves.
'''
import numpy as np
from time import perf_counter
import gmsh # ellpsoid with holes
import meshio

t0 = perf_counter()
lmbda,mu = 8e9,3e9
rho = 910
rhof = 1010

Wx = 50000
Hw = 500
Hi = 400
Hc = Hw - (rho/rhof) * Hi
xf = Wx/4

h_crevasse = 5
w_crevasse = 1
x_crevasse = 100

gravity = 9.8
k = 2*np.pi/200
A = 1
omega = np.sqrt(gravity*k*np.tanh(k*Hw))

max_element_size = 100

print(f'The phase velocity is {omega/k} m/s')
print(f'The wave length is {2*np.pi/k} m')
print(f'The wave period is {2*np.pi/omega} s')


gmsh.initialize()

gmsh.model.occ.addRectangle(xf, Hc, 0.0, (Wx-xf), Hi, tag=1)
gmsh.model.occ.addRectangle(0.0,0.0,0.0,Wx,Hw, tag=2)
gmsh.model.occ.addRectangle(xf+x_crevasse, Hi+Hc-h_crevasse,0.0,
                            w_crevasse,h_crevasse,tag=3)
gmsh.model.occ.cut([(2,1)],[(2,3)],tag=4)
gmsh.model.occ.fragment([(2,4)],[(2,2)])
gmsh.model.occ.synchronize()

# Label the surfaces (ice and water)
solid_tag = []
for surface in gmsh.model.getEntities(dim=2):
    com = gmsh.model.occ.getCenterOfMass(surface[0], surface[1])
    print(com)
    if np.isclose(com[0], Wx*.625, atol=max_element_size/5):
        solid_tag.append(surface[1])
    else:
        fluid_tag = surface[1]

gmsh.model.addPhysicalGroup(2,[fluid_tag],1)
gmsh.model.addPhysicalGroup(2,solid_tag,2)

# Label boundaries
ice_interface = []
for edge in gmsh.model.getEntities(dim=1):
    com = gmsh.model.occ.getCenterOfMass(edge[0], edge[1])
    if np.isclose(com[1], Hw):
        water_surface = [edge[1]]
    elif np.isclose(com[0],xf):
        ice_interface.append(edge[1])
    elif np.isclose(com[1],Hc):
        ice_interface.append(edge[1])
    elif np.isclose(com[1],0.0):
        water_bottom= [edge[1]]

# Add physical entities
gmsh.model.addPhysicalGroup(1,water_surface,3)
gmsh.model.addPhysicalGroup(1,ice_interface,4)
gmsh.model.addPhysicalGroup(1,water_bottom,5)

# Refine near the crack
distance_field = gmsh.model.mesh.field.add("Distance")
gmsh.model.mesh.field.setNumbers(distance_field, "EdgesList", [3])
threshold_field = gmsh.model.mesh.field.add("Threshold")
gmsh.model.mesh.field.setNumber(threshold_field, "IField", distance_field)
gmsh.model.mesh.field.setNumber(threshold_field, "LcMin", 0.02)
gmsh.model.mesh.field.setNumber(threshold_field, "LcMax", 40)
gmsh.model.mesh.field.setNumber(threshold_field, "DistMin", 0)
gmsh.model.mesh.field.setNumber(threshold_field, "DistMax", 100)
min_field = gmsh.model.mesh.field.add("Min")
gmsh.model.mesh.field.setNumbers(min_field, "FieldsList", [threshold_field])
gmsh.model.mesh.field.setAsBackgroundMesh(min_field)
#gmsh.option.setNumber("Mesh.MeshSizeMax", max_element_size)

gmsh.model.mesh.generate(2)
gmsh.write("mesh.msh")
gmsh.finalize()

'''
Convert the mesh
'''
def create_mesh(mesh, cell_type, prune_z=False):
    cells = mesh.get_cells_type(cell_type)
    cell_data = mesh.get_cell_data("gmsh:physical", cell_type)
    points = mesh.points[:, :2] if prune_z else mesh.points

    out_mesh = meshio.Mesh(points=points,
        cells={cell_type: cells},
        cell_data={"name_to_read": [cell_data]})

    return out_mesh

mesh_from_file = meshio.read("mesh.msh")

triangle_mesh = create_mesh(mesh_from_file, "triangle", prune_z=True)
meshio.write("mesh.xdmf", triangle_mesh)

line_mesh = create_mesh(mesh_from_file, "line", prune_z=True)
meshio.write("facet_mesh.xdmf", line_mesh)

print(f'Done with meshing at t={perf_counter()-t0} s')



'''
Solve the coupled system of PDEs.
'''
from dolfin import *
import dolfin

'''
Read mesh and mark domains/boundaries
'''
mesh = Mesh()
with XDMFFile("mesh.xdmf") as infile:
	infile.read(mesh)

mvc = MeshValueCollection("size_t", mesh, 2)
with XDMFFile("mesh.xdmf") as infile:
	infile.read(mvc)
mf = cpp.mesh.MeshFunctionSizet(mesh, mvc)

mvc2 = MeshValueCollection("size_t", mesh, 1)
with XDMFFile("facet_mesh.xdmf") as infile:
    infile.read(mvc2)
mf2 = cpp.mesh.MeshFunctionSizet(mesh, mvc2)

# Use dS when integrating over the interior boundaries
# Use ds for the exterior boundaries,
dx = Measure("dx", domain=mesh,subdomain_data=mf)
ds = Measure("ds", domain=mesh, subdomain_data=mf2)
dS = Measure("dS", domain=mesh, subdomain_data=mf2)

dXf= dx(subdomain_id=1)
dXs = dx(subdomain_id=2)
dst = ds(subdomain_id=3)
dSi = dS(subdomain_id=4)
dsb  = ds(subdomain_id=5)

'''
Set up functional spaces 
'''
V = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
Vv = VectorElement("Lagrange", mesh.ufl_cell(), 2)
W = FunctionSpace(mesh, V*Vv)

TRF = TrialFunction(W)
TTF = TestFunction(W)
(p, u) = split(TRF)
(q, v) = split(TTF)


'''
Dirichlet boundary conditions
'''
u0 = Constant(0.0)
wavebc= Expression("A*omega/k*cosh(k*x[1])/sinh(k*H)",
		A=A,g=gravity,omega=omega,k=k,H=Hw,degree=2)

zero = Constant(0.0)
zero_2d = Constant((0.0, 0.0))

def left_boundary(x):
    return near(x[0],0.0) 
def right_ice_boundary(x):
    return near(x[0],Wx) and (x[1] > Hc)
def right_water_boundary(x):
    return near(x[0],Wx) and (x[1] < Hc) 

bcs = [ DirichletBC(W.sub(0), wavebc, left_boundary), 
	DirichletBC(W.sub(1), zero_2d, right_ice_boundary),
	DirichletBC(W.sub(0), zero, right_water_boundary) ]

''' 
Define variational problem
'''
sigma = 2.0*mu*sym(grad(u)) \
	+ lmbda*tr(sym(grad(u)))*Identity(u.geometric_dimension())
n = FacetNormal(mesh)

#Fluid domain
a_f = inner(grad(p), grad(q))*dXf - omega**2/gravity*p*q*dst
L_f = zero*q*dsb

#Solid domain
a_s = (inner(sigma, grad(v)) - rho*omega**2*inner(u,v))*dXs
#L_s = inner(zero_2d,v )*ds

#Interface fluid-solid
#a_i = (rho*omega**2 * inner(n('-'), u('-')) * q('-')\
#	 - omega*rhof*p('-')*inner(n('-'), v('-')))*dS

a_i = (rho*omega**2 * inner(n('+'), u('+')) * q('+')\
	 - omega*rhof*p('+')*inner(n('+'), v('+')))*dSi

#a_i = (rho*omega**2 * inner(avg(n), u('+'))*avg(q) \
#	 -omega*rhof* p('+')*inner(avg(n), v('+')))*dS(3)
# L_i = zero*q('-')*dS(3)

#Weak form
a = a_f + a_s + a_i
L = L_f #+ L_s

''' 
Compute solution
'''
s = Function(W)
A=assemble(a, keep_diagonal=True)
b=assemble(L)

for bc in bcs: bc.apply(A, b)
A.ident_zeros()
s = Function(W)
print(f'Solve starting at t={perf_counter()-t0} s')
solve(A, s.vector(), b)
pp,uu = split(s)
print(f'Solve finished at t={perf_counter()-t0} s')

writevtk = True
if writevtk:
	file = File("fgw.pvd")
	file << s
else:
	# Plot solution
	import matplotlib.pyplot as plt
	fig,ax=plt.subplots(2,1,figsize=(10,10))
	
	plt.subplot(2,1,1)
	dispamp= project( uu[0]**2 + uu[1]**2, FunctionSpace(mesh, 'P', 1))
	c=plot(uu[1]*1e6)
	plt.colorbar(c,orientation="horizontal")
	
	plt.subplot(2,1,2)
	scale = 10
	cc=plot(pp,cmap='seismic',vmin=-scale,vmax=scale)
	plt.colorbar(cc,orientation="horizontal")
	plt.tight_layout()
	plt.savefig('figures/poisson.png')
print(f'All done at t={perf_counter()-t0} s')
