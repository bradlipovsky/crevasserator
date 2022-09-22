from dolfin import *
from os import path
from mshr import *
import numpy as np
import multiprocessing
from functools import partial
import scipy.integrate as integrate
import scipy.special as special

def elasticity_solutions(case='full-minus-prestress', 
                         geometry = {'W':60000,
                                     'H':300,
                                     'Lc':15, 
                                     'Wc':1, 
                                     'Hc': 5, 
                                     'fl':0, 
                                     'swell_wavelength':156,
                                     'ice_wavelength':3000}, 
                         materials = {'E':1e10,
                                      'nu':0.3, 
                                      'rho':910, 
                                      'rhow':1024, 
                                      'g':9.81},
                         crevasse_location="surface",
                         swell_amplitude=0.0,
                         swell_transmission_coeff=0.14,
                         swell_phase=0.0,
                         swell_forcing="everything",
                         refinement=1,
                         verbose=False):
    """
    elasticity_solutions solves the equations of plane strain elasticity 
    with boundary conditions that are representative of an ice shelf. 

    :param case: gives numerous simplifications of the most realistic 
    boundary conditions. For many of the cases, analytical solutions 
    are available (and are displayed if verbose=True).
    
    :param footloose: is the length of a submarine ice foot (the front of 
    the foot is at zero)
    
    :return: the solution U and the mesh
    """ 
        
    # Meshing parameters
    number_of_refinement_iterations = refinement
    length_of_refinement_region = 1e6 # set to a huge number --> refine everywhere
    waterline_dy = 2
    ice_front_dy = 4
    crevasse_num_pts = 250 # 250 pts @ Nominal 5m length => 2cm spacing
    crevasse_tip_num_pts = 50 # 50 pts @ Nominal 1m length => 2cm spacing
    
    if verbose:
        print(f'Crevasse points wall/tip: '\
                f'{crevasse_num_pts}/{crevasse_tip_num_pts}')
    
    # Type fewer characters later...
    W = geometry['W']
    H = geometry['H']
    Hc = geometry['Hc']
    Lc = geometry['Lc']
    Wc = geometry['Wc']
    E = materials['E']
    nu= materials['nu']
    rho=materials['rho']
    rhow=materials['rhow']
    g=materials['g']
    swell_wavelength=geometry['swell_wavelength']
    ice_wavelength=geometry['ice_wavelength']

    if 'fl' in geometry.keys():
        footloose=geometry['fl']
    else:
        footloose=0.0
        
    if verbose:
        print('Running %s model:'%case)
    
    D, flexural_gravity_wavelength, lam = fgl(materials,geometry)
    crevasse_refinement=length_of_refinement_region* \
                flexural_gravity_wavelength
    
    mu = E/2./(1+nu)
    K = E/(3*(1-2*nu))
    lmbda = E*nu/(1+nu)/(1-2*nu)
    Hw = H * rho / rhow
    
    
    
    def right(x, on_boundary):
        return near(x[0], W) and on_boundary

    def left(x, on_boundary):
        return near(x[0], 0) and on_boundary

    def bottom_fun(x, on_boundary):
        return near(x[1], 0) and on_boundary

    class Bottom(SubDomain):
        def inside(self, x, on_boundary):
            return near(x[1], 0) and on_boundary

    class Top(SubDomain):
#         def inside(self, x, on_boundary):
#             return near(x[1], H) and on_boundary
        def inside(self, x, on_boundary):
            return ((near(x[1], H) and x[0]>=footloose)\
                    or (near(x[1], Hw) and x[0]<=footloose))\
                    and on_boundary

    class Front(SubDomain):
#         def inside(self, x, on_boundary):
#             return near(x[0], 0) and on_boundary
        def inside(self, x, on_boundary):
            return (near(x[0], 0) \
                    or (near(x[0], footloose) and x[1]>=Hw))\
                    and on_boundary
        
    class BottomCrevasseWalls(SubDomain):
        def inside(self,x,on_boundary):
            return (near(x[0],Lc-Wc/2) or near(x[0],Lc-Wc/2)) \
                    and (x[1]<=Hc)\
                    and on_boundary
        
    class BottomCrevasseTip(SubDomain):
        def inside(self,x,on_boundary):
            return (x[0] >= Lc-Wc/2) and (x[0] <= Lc+Wc/2)\
                    and near(x[1],Hc)\
                    and on_boundary
        
    class SurfaceCrevasseWalls(SubDomain):
        def inside(self,x,on_boundary):
            return (near(x[0],Lc-Wc/2) or near(x[0],Lc-Wc/2)) \
                    and (x[1]>=H-Hc)\
                    and on_boundary
        
    class SurfaceCrevasseTip(SubDomain):
        def inside(self,x,on_boundary):
            return (x[0] >= Lc-Wc/2) and (x[0] <= Lc+Wc/2)\
                    and near(x[1],H-Hc)\
                    and on_boundary
        
    '''
    Define the geometry and the mesh
    '''
    if verbose:
        print('     Generating Mesh')
        
    
    ice = build_ice(H,Hw,swell_amplitude,ice_front_dy,
                    waterline_dy,W,footloose)
    crevasse = build_crevasse(geometry,crevasse_location,
                    crevasse_num_pts,crevasse_tip_num_pts)
    

    if (swell_amplitude > 0.0) and (case != 'full-minus-prestress'):
        print('not implemented')
        exit()
        
    
    mesh = generate_mesh ( ice - crevasse, 100)

    # Refine the mesh.  
    # From: https://fenicsproject.discourse.group\
    #   /t/prescribing-spatially-varying-cell-sizes/527
    d = mesh.topology().dim()

    # Refine in a region of +/- 2*FGL from the crevasse. 
    # Also refine near the ice front.
    refined_region = CompiledSubDomain("((x[0] < Lc + dx)"\
                        "& (x[0] > Lc - dx))"\
                        "| (x[0] < dx)",H=H,Lc=Lc,dx=crevasse_refinement)
    
    for i in range(number_of_refinement_iterations):
        r_markers = MeshFunction("bool", mesh, d, False)
        refined_region.mark(r_markers, True)
        mesh = refine(mesh,r_markers)
    
#     mesh = RectangleMesh(Point(0., 0.), Point(W, H), Nx, Ny)

    facets = MeshFunction("size_t", mesh, 1)
    facets.set_all(0)
    Front().mark(facets, 2)
    Bottom().mark(facets, 1)
    
    if crevasse_location=="bottom":
        BottomCrevasseWalls().mark(facets, 3)
        BottomCrevasseTip().mark(facets,4)
    else:
        SurfaceCrevasseWalls().mark(facets,3)
        SurfaceCrevasseTip().mark(facets,4)

    ds = Measure("ds", subdomain_data=facets)
    V = VectorFunctionSpace(mesh, 'CG', 2)
    u = TrialFunction(V)
    v = TestFunction(V)
    
    if verbose:
        print("     Creating forms")
        
    if case == 'full-minus-prestress':
        
        if (swell_forcing == 'everything') \
            or (swell_forcing == 'front only'):

            if verbose:
                print('     Applying swell bc on the ice front')
            P_fro  = Expression(("(x[1]<Hw) ? rhow*g*(Hw "
                            "+ A*sin(2*pi*x[0]/L + P) - x[1]) : 0","0"), 
                            degree=1,
                            Hw=Hw, rhow=rhow,g=g,
                            A=swell_amplitude,
                            L=swell_wavelength,P=swell_phase,pi=np.pi)
        else:
            P_fro  = Expression(("(x[1]<Hw) ? rhow*g*(Hw - x[1]) : 0","0"), 
                            degree=1,
                            Hw=Hw, rhow=rhow,g=g)
            
        if (swell_forcing == 'everything') \
                    or (swell_forcing == 'bottom only'):

            if verbose:
                print('     Applying swell bc on the ice bottom')
            P_bot  = Expression(("0","(x[1]<Hw) ? rhow*g*(Hw + "
                            "A*sin(2*pi*x[0]/L + P) - x[1]) : 0"), 
                            degree=1,
                            Hw=Hw, rhow=rhow,g=g,
                            A=swell_amplitude*swell_transmission_coeff,
                            L=ice_wavelength,
                            P=swell_phase,pi=np.pi)
        else:
            P_bot  = Expression(("0","(x[1]<Hw) ? rhow*g*(Hw - x[1]) : 0"), 
                            degree=1,
                            Hw=Hw, rhow=rhow,g=g)

        P0_fro = Expression(("(x[0]>=fl) ? rho*g*(H-x[1]) :"
                             "rho*g*(Hw-x[1])","0"), degree=1,
                            Hw=Hw, rhow=rhow,g=g, rho=rho, H=H,fl=footloose)
        
        P0_bot = Expression(("0","(x[0]>=fl) ? rho*g*(H-x[1]) :"
                            "rho*g*(Hw-x[1])"), degree=1,
                            Hw=Hw, rhow=rhow,g=g, rho=rho,
                            H=H,pi=np.pi,fl=footloose) 
            
#         P0_fro = Expression(("rho*g*(H-x[1]) ","0"), degree=1,
#                             Hw=Hw, rhow=rhow,g=g, rho=rho, H=H) 
#         P0_bot = Expression(("0","rho*g*(H-x[1]) "), degree=1,
#                             Hw=Hw, rhow=rhow,g=g, rho=rho, H=H,pi=np.pi) 
        
        bc = DirichletBC(V, Constant((0.,0.)), right)
        a = inner(sigma(v,lmbda,mu),eps(u))*dx + rhow*g*u[1]*v[1]*ds(1)
        L = dot(P_fro, v)*ds(2) - dot(P0_fro, v)*ds(2) \
           +dot(P_bot, v)*ds(1) - dot(P0_bot, v)*ds(1) \
           +dot(P_fro, v)*ds(3) - dot(P0_fro, v)*ds(3) \
           +dot(P_bot, v)*ds(4) - dot(P0_bot, v)*ds(4) 
        
    if case == 'wrong-bottom-minus-prestress':

        P0_fro = Expression(("rho*g*(H-x[1]) ","0"), degree=1,Hw=Hw, rhow=rhow,g=g, rho=rho, H=H) 
        P_fro = Expression(("(x[1]<Hw) ? rhow*g*(Hw-x[1]) : 0","0"), degree=1,Hw=Hw, rhow=rhow,g=g)
        
        bc = DirichletBC(V, Constant((0.,0.)), right)
        a = inner(sigma(v,lmbda,mu),eps(u))*dx 
        L = dot(P_fro, v)*ds(2) - dot(P0_fro, v)*ds(2) 
        
    if case == 'uniform-end-load-prestress':

        P0_fro = Expression(("rho*g*H*(1-rho/rhow) ","0"), degree=1,
                            Hw=Hw, rhow=rhow,g=g, rho=rho, H=H) 
        
        bc = DirichletBC(V, Constant((0.,0.)), right)
        a = inner(sigma(v,lmbda,mu),eps(u))*dx 
        L =  - dot(P0_fro, v)*ds(2) 

    if case == 'full':
        P_bot = Expression(("0","(x[1]<Hw) ? rhow*g*(Hw-x[1]) : 0"), degree=1,Hw=Hw, rhow=rhow,g=g)
        P_fro = Expression(("(x[1]<Hw) ? rhow*g*(Hw-x[1]) : 0","0"), degree=1,Hw=Hw, rhow=rhow,g=g)
        f = Constant((0.0, -rho*g))
        
        bc = DirichletBC(V, Constant((0.,0.)), right)
        a = inner(sigma(v,lmbda,mu),eps(u))*dx + rhow*g*u[1]*v[1]*ds(1)
        L = inner(f, v)*dx  + dot(P_bot, v)*ds(1) + dot(P_fro, v)*ds(2) 
        
    if case == 'wrong-bottom-bc':
        P_bot = Expression(("0","(x[1]<Hw) ? rhow*g*(Hw-x[1]) : 0"), degree=1,Hw=Hw, rhow=rhow,g=g)
        P_fro = Expression(("(x[1]<Hw) ? rhow*g*(Hw-x[1]) : 0","0"), degree=1,Hw=Hw, rhow=rhow,g=g)
        f = Constant((0.0, -rho*g))
        
        bc = DirichletBC(V, Constant((0.,0.)), right)
        a = inner(sigma(v,lmbda,mu),eps(u))*dx
        L = inner(f, v)*dx  + dot(P_bot, v)*ds(1) + dot(P_fro, v)*ds(2) 
        
    elif case == 'uniform-end-load':
        P_fro = Expression(("rhow*g*Hw/2","0"), degree=1,Hw=Hw, rhow=rhow,g=g)
        
        bc1 = DirichletBC(V.sub(0), Constant(0.), right)
        bc2 = DirichletBC(V.sub(1), Constant(0.), bottom_fun)
        bc = [bc1,bc2]
        
        a = inner(sigma(v),eps(u))*dx #+ rhow*g*u[1]*v[1]*ds(1)
        L =  dot(P_fro, v)*ds(2) 
        
    elif case == 'gravity-only':
        f = Constant((0.0, -rho*g))

        bc1 = DirichletBC(V.sub(0), Constant((0.)), right)
        bc2 = DirichletBC(V.sub(1), Constant(0.), bottom_fun)
        bc3 = DirichletBC(V.sub(0), Constant((0.)), left)
        bc = [bc1,bc2,bc3]
        
        a = inner(sigma(v,lmbda,mu),eps(u))*dx 
        L = inner(f, v)*dx 
    elif case == 'gravity-free-front':
        f = Constant((0.0, -rho*g))
        
        bc1 = DirichletBC(V.sub(0), Constant((0.)), right)
        bc2 = DirichletBC(V.sub(1), Constant(0.), bottom_fun)
        bc = [bc1,bc2]
        
        a = inner(sigma(v),eps(u))*dx 
        L = inner(f, v)*dx 
        
    elif case == 'gravity-loaded-front':
        f = Constant((0.0, -rho*g))
        P_fro = Expression(("rhow*g*Hw/2","0"), degree=1,Hw=Hw, rhow=rhow,g=g)
        
        bc1 = DirichletBC(V.sub(0), Constant((0.)), right)
        bc2 = DirichletBC(V.sub(1), Constant(0.), bottom_fun)
        bc = [bc1,bc2]
        
        a = inner(sigma(v,lmbda,mu),eps(u))*dx 
        L = inner(f, v)*dx + dot(P_fro, v)*ds(2) 

    if case == 'gravity-bottom-pressure':
        P_bot = Expression(("0","(x[1]<Hw) ? rhow*g*(Hw-x[1]) : 0"), degree=1,Hw=Hw, rhow=rhow,g=g)
        f = Constant((0.0, -rho*g))
        
        bc = DirichletBC(V, Constant((0.,0.)), right)
        a = inner(sigma(v,lmbda,mu),eps(u))*dx 
        L = inner(f, v)*dx  + dot(P_bot, v)*ds(1) 
    
    set_log_active(False)
    #    set_log_level(10)
    
    if verbose:
        print("     SOLVING")
    U = Function(V)
    solve(a==L,U,bc)
    
    
    ux,uz=-1,-1
    x0 = footloose
    
    
    if case == 'uniform-end-load':
        sig0 = rhow * g * Hw / 2
        f1 = nu/(1-nu)
        f2 = 1 - f1**2
        ux = W*sig0/(lmbda+2*mu)/f2
        uz = ux * f1 * H/W
        x0 = 0
        
    elif case == 'gravity-only':
        print('Vertical deflection:')
        uz = -rho * g * H / (lmbda+2*mu) * H / 2
        ux = 0
        x0 = W/2
    
    elif case == 'gravity-free-front':
        f1 = nu/(1-nu)
        f2 = 1 - f1**2
        uz = (-rho * g * H) / (lmbda+2*mu) / f2 * H / 2
        ux = f1 * W / H * uz / 2
        x0 = 0
        
    elif case == 'gravity-loaded-front':
    
        x0 = W/2
        sig0 = rhow * g * Hw / 2
        f1 = nu/(1-nu)
        f2 = 1 - f1**2
        ux_loaded = x0*sig0/(lmbda+2*mu)/f2
        uz_loaded = ux_loaded * f1 * H/x0
        
        uz_free = (-rho * g * H) / (lmbda+2*mu) / f2 * H / 2
        ux_free = f1 * x0 / H * uz_free 
        
        ux = ux_free + ux_loaded
        uz = uz_free + uz_loaded
        
    elif case == 'full':
    
        x0 = W/2
        sig0 = rhow * g * Hw / 2
        f1 = nu/(1-nu)
        f2 = 1 - f1**2
        ux_loaded = x0*sig0/(lmbda+2*mu)/f2
        uz_loaded = ux_loaded * f1 * H/x0
        
        uz_free = (-rho * g * H) / (lmbda+2*mu) / f2 * H / 2
        ux_free = f1 * x0 / H * uz_free 
        
        ux = ux_free + ux_loaded
        uz = uz_free + uz_loaded

    if verbose == True:
        print ('Results:')
        print('     Vertical deflection:')
        if uz != -1:
            print ('         Analytical: %f'%uz)
        print ('         Numerical:  %f'%U(x0,H)[1])
        print('x0=%f'%x0)
        print('     Horizontal deflection:')
        if ux!=-1:
            print ('         Analytical: %f'%ux)
        print ('         Numerical:  %f'%U(x0,H/2)[0])

        print(' ')
    return U,mesh

def build_crevasse(geom,crevasse_location,crevasse_num_pts,
                    crevasse_tip_num_pts):
    Lc = geom['Lc']
    Wc = geom['Wc']
    Hc = geom['Hc']
    H = geom['H']
    
    crevasse_points = []    
    if crevasse_location=="surface":
        for i in range(crevasse_num_pts+1):
            x = Lc-Wc/2
            y = float(H - i*Hc/crevasse_num_pts)
            crevasse_points.append( Point(x,y))
    #         print((x,y))
        for i in range(crevasse_tip_num_pts+1):
            y = H-Hc
            x = float(Lc - Wc/2 + i*Wc/crevasse_tip_num_pts)
            crevasse_points.append( Point(x,y))
    #         print((x,y))    
        for i in range(crevasse_num_pts+1):
            x = Lc+Wc/2
            y = float(H - (crevasse_num_pts-i)*Hc/crevasse_num_pts)
            crevasse_points.append( Point(x,y))
    #         print((x,y))
    
    if crevasse_location=="bottom":
        for i in range(crevasse_num_pts+1):
            x = Lc+Wc/2
            y = float(i*Hc/crevasse_num_pts)
            crevasse_points.append( Point(x,y))
    #         print((x,y))
        for i in range(crevasse_tip_num_pts+1):
            y = Hc
            x = float(Lc + Wc/2 - i*Wc/crevasse_tip_num_pts)
            crevasse_points.append( Point(x,y))
    #         print((x,y))
        for i in range(crevasse_num_pts+1):
            x = Lc-Wc/2
            y = float(Hc - i*Hc/crevasse_num_pts)
            crevasse_points.append( Point(x,y))
    #         print((x,y))
    crevasse = Polygon(crevasse_points)
    return crevasse

def build_ice(H,Hw,swell_amplitude,ice_front_dy,waterline_dy,W,footloose):
#     boundary_points = [Point(0., 0), Point(W, 0), Point(W, H)]
#     boundary_points = [Point(0., 0)]

    boundary_points = []
    typical_spacing = H/4
    nx = np.ceil(W / typical_spacing)
    ny = np.ceil(H / typical_spacing)
    
    x_points_bottom = np.linspace(0,W,nx)
    y_points_right = np.linspace(0,H,ny)
    x_points_top = np.linspace(W,0,nx)
    
    for this_x in x_points_bottom:
        boundary_points.append( Point(this_x,0) )
    for this_y in y_points_right:
        boundary_points.append( Point(W,this_y) )
    for this_x in x_points_top:
        boundary_points.append( Point(this_x,H) )

    # These are the points along the ice front
    y_points_top = np.arange(H,Hw+swell_amplitude,-ice_front_dy)
    y_points_mid = np.arange(Hw+swell_amplitude,Hw-swell_amplitude,-waterline_dy)
    y_points_bot = np.arange(Hw-swell_amplitude,0,-ice_front_dy)
    y_points = np.concatenate((y_points_top, y_points_mid, y_points_bot))
    first_waterline_point = 0
    
    for this_y in y_points:
        if this_y > Hw:
            boundary_points.append( Point(footloose,this_y) ) 
        else:
            if first_waterline_point == 0:
                boundary_points.append( Point(footloose,Hw) ) 
                boundary_points.append( Point(0.0,Hw) ) 
                first_waterline_point = 1
                if this_y == Hw:
                    continue
            boundary_points.append( Point(0.0,this_y) ) 
    ice = Polygon( boundary_points )
    return ice

def eps(v):
    return sym(grad(v))

def sigma(v,lmbda,mu):
    dim = v.geometric_dimension()
    return 2.0*mu*eps(v) + lmbda*tr(eps(v))*Identity(dim)





