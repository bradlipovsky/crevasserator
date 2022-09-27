'''
Harmonic analysis of surface gravity waves impacting an elastic, 
finite-thickness ice shelf with crevasses.
'''
from random import randint
import numpy as np
from time import perf_counter
import gmsh # ellpsoid with holes
import meshio
from lefm import sif

def modal_elasticity_solution(  x_crevasse=100,
                                verbose=0,
                                writevtk = True,
                                deletemesh = True):
    '''
    Carry out a modal elasticity solution. Returns SIFs.
    '''
    if verbose > 0:
        print(f'Starting simulation with crevasse at {x_crevasse}.')
    t0 = perf_counter()
    n = randint(0,1e6)
 
    # All the parameters are stored in the dictionary p
    m = {
        'lmbda': 8e9, 
        'mu' : 3e9,
        'rho' : 910,
        'rhof' : 1010,
        'Wx' : 100000,
        'Hw' : 500,
        'Hi' : 400,
        'h_crevasse' : 5,
        'w_crevasse' : 1,
        'x_crevasse' : x_crevasse,
        'gravity' : 9.8,
        'k' : 2*np.pi/200,
        'A' : 1,
        'max_element_size' : 40,
        'min_element_size' : 0.02,
        'Hc':0,
        'omega':0,
        'xf':0
    }
    m['xf'] = m['Wx']/4
    m['Hc'] = m['Hw'] - (m['rho']/m['rhof']) * m['Hi']
    m['omega'] = np.sqrt(m['gravity']*m['k']*np.tanh(m['k']*m['Hw']))

    if verbose > 1:
        print(f'\tThe swell phase velocity is {m["omega"]/m["k"]} m/s')
        print(f'\tThe swell wave length is {2*np.pi/m["k"]} m')
        print(f'\tThe swell wave period is {2*np.pi/m["omega"]} s')

    '''
    Make the mesh
    '''
    create_mesh(m,n)

    from dolfin import Mesh, MeshValueCollection, XDMFFile, FiniteElement, Measure, VectorElement, FunctionSpace, TrialFunction, TestFunction, Constant, Expression, split, DirichletBC, sym, grad, Identity, inner, FacetNormal, tr, assemble, solve, Function, near, File
    from dolfin.cpp.mesh import MeshFunctionSizet

    mesh = Mesh()
    with XDMFFile(f"mesh_{n}.xdmf") as infile:
        infile.read(mesh)

    mvc = MeshValueCollection("size_t", mesh, 2)
    with XDMFFile(f"mesh_{n}.xdmf") as infile:
        infile.read(mvc)
    mf = MeshFunctionSizet(mesh, mvc)

    mvc2 = MeshValueCollection("size_t", mesh, 1)
    with XDMFFile(f"facet_mesh_{n}.xdmf") as infile:
        infile.read(mvc2)
    mf2 = MeshFunctionSizet(mesh, mvc2)

    if deletemesh:
        from os import remove
        remove(f"mesh_{n}.xdmf")
        remove(f"facet_mesh_{n}.xdmf")
        remove(f"mesh_{n}.msh")
        remove(f"mesh_{n}.h5")
        remove(f"facet_mesh_{n}.h5")

    '''
    Mark domains/boundaries
    '''

    # Use dS when integrating over the interior boundaries
    # Use ds for the exterior boundaries
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
            A=m['A'],g=m['gravity'],omega=m['omega'],
            k=m['k'],H=m['Hw'],degree=2)

    zero = Constant(0.0)
    zero_2d = Constant((0.0, 0.0))

    def left_boundary(x):
        return near(x[0],0.0) 
    def right_ice_boundary(x):
        return near(x[0],m['Wx']) and (x[1] > m['Hc'])
    def right_water_boundary(x):
        return near(x[0],m['Wx']) and (x[1] < m['Hc']) 

    bcs = [ DirichletBC(W.sub(0), wavebc, left_boundary), 
        DirichletBC(W.sub(1), zero_2d, right_ice_boundary),
        DirichletBC(W.sub(0), zero, right_water_boundary) ]

    ''' 
    Define variational problem
    '''
    sigma = 2.0*m['mu']*sym(grad(u)) \
        + m['lmbda']*tr(sym(grad(u)))*Identity(u.geometric_dimension())
    n = FacetNormal(mesh)

    #Fluid domain
    a_f = inner(grad(p), grad(q))*dXf - m['omega']**2/m['gravity']*p*q*dst
    L_f = zero*q*dsb

    #Solid domain
    a_s = (inner(sigma, grad(v)) - m['rho']*m['omega']**2*inner(u,v))*dXs

    #Interface fluid-solid
    a_i = (m['rho']*m['omega']**2 * inner(n('+'), u('+')) * q('+')\
         - m['omega']*m['rhof']*p('+')*inner(n('+'), v('+')))*dSi

    #Weak form
    a = a_f + a_s + a_i
    L = L_f

    ''' 
    Compute solution
    '''
    s = Function(W)
    A=assemble(a, keep_diagonal=True)
    b=assemble(L)

    for bc in bcs: bc.apply(A, b)
    A.ident_zeros()
    s = Function(W)
    if verbose > 0:
        print(f'\tSolve (x={x_crevasse}) starting at '+\
              f't={(perf_counter()-t0):2.1f} s')
    solve(A, s.vector(), b)
    if verbose > 0:
        print(f'\tSolve (x={x_crevasse}) finished at t={perf_counter()-t0} s')

    if writevtk:
        file = File("fgw.pvd")
        file << s
    
    '''
    Calculate SIF
    '''
    Xc = m['xf']+m['x_crevasse']
    Yc = m['Hi']+m['Hc']-m['h_crevasse']
    Wc = m['w_crevasse']
    nu = m['lmbda']/2/(m['lmbda']+m['mu'])
    factor = m['mu']/(3-4*nu+1)
    pp,uu = split(s)
    [KI,KII] = sif(uu,Xc,Yc,Wc,factor,verbose=verbose)

    if verbose > 0:
        print(f'\tEvaluated SIFS (x={x_crevasse}), t={perf_counter()-t0} s')
        print(f'\t\tKI = {KI}, KII={KII}')
    return KI, KII

'''
Convert the mesh
'''
def convert_mesh(mesh, cell_type, prune_z=False):
    cells = mesh.get_cells_type(cell_type)
    cell_data = mesh.get_cell_data("gmsh:physical", cell_type)
    points = mesh.points[:, :2] if prune_z else mesh.points

    out_mesh = meshio.Mesh(points=points,
        cells={cell_type: cells},
        cell_data={"name_to_read": [cell_data]})

    return out_mesh



def create_mesh(g,n):
    '''
    create_mesh(g)
    g, geometry dictionary 
    n, a number to append to the mesh filename

    Writes mesh files mesh_n.msh, mesh_n.xdmf, and facet_mesh_n.xdmf for our
    particular (simple) ice shelf geometry

    '''

    gmsh.initialize()

    gmsh.model.occ.addRectangle(g['xf'], g['Hc'], 0.0, 
                               (g['Wx']-g['xf']), g['Hi'], tag=1)
    gmsh.model.occ.addRectangle(0.0,0.0,0.0,g['Wx'],g['Hw'], tag=2)
    gmsh.model.occ.addRectangle(g['xf']+g['x_crevasse']-g['w_crevasse']/2, 
                                g['Hi']+g['Hc']-g['h_crevasse'],0.0,
                                g['w_crevasse'],g['h_crevasse'],tag=3)
    gmsh.model.occ.cut([(2,1)],[(2,3)],tag=4)
    gmsh.model.occ.fragment([(2,4)],[(2,2)])
    gmsh.model.occ.synchronize()

    # Label the surfaces (ice and water)
    solid_tag = []
    for surface in gmsh.model.getEntities(dim=2):
        com = gmsh.model.occ.getCenterOfMass(surface[0], surface[1])
        if np.isclose(com[0], g['Wx']*.625, atol=g['max_element_size']/5):
            solid_tag.append(surface[1])
        else:
            fluid_tag = surface[1]

    gmsh.model.addPhysicalGroup(2,[fluid_tag],1)
    gmsh.model.addPhysicalGroup(2,solid_tag,2)

    # Label boundaries
    ice_interface = []
    for edge in gmsh.model.getEntities(dim=1):
        com = gmsh.model.occ.getCenterOfMass(edge[0], edge[1])
        if np.isclose(com[1], g['Hw']):
            water_surface = [edge[1]]
        elif np.isclose(com[0],g['xf']):
            ice_interface.append(edge[1])
        elif np.isclose(com[1],g['Hc']):
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
    gmsh.model.mesh.field.setNumber(threshold_field, 
        "IField", distance_field)
    gmsh.model.mesh.field.setNumber(threshold_field, 
        "LcMin", g['min_element_size'])
    gmsh.model.mesh.field.setNumber(threshold_field, 
        "LcMax", g['max_element_size'])
    gmsh.model.mesh.field.setNumber(threshold_field, "DistMin", 0)
    gmsh.model.mesh.field.setNumber(threshold_field, "DistMax", 100)
    min_field = gmsh.model.mesh.field.add("Min")
    gmsh.model.mesh.field.setNumbers(min_field, 
        "FieldsList", [threshold_field])
    gmsh.model.mesh.field.setAsBackgroundMesh(min_field)
    gmsh.option.setNumber('General.Verbosity', 2)

    gmsh.model.mesh.generate(2)
    gmsh.write(f"mesh_{n}.msh")
    gmsh.finalize()

    mesh_from_file = meshio.read(f"mesh_{n}.msh")

    triangle_mesh = convert_mesh(mesh_from_file, "triangle", prune_z=True)
    meshio.write(f"mesh_{n}.xdmf", triangle_mesh)

    line_mesh = convert_mesh(mesh_from_file, "line", prune_z=True)
    meshio.write(f"facet_mesh_{n}.xdmf", line_mesh)

