from dolfin import *
from mshr import *
# import matplotlib.pyplot as plt
# from time import perf_counter
import numpy as np
# import pickle
# import scipy.integrate as integrate
# import scipy.special as special
# from os import path
import multiprocessing
from scipy.optimize import fminbound
from functools import partial
import scipy.integrate as integrate
import scipy.special as special



def elasticity_solutions(case='full-minus-prestress', 
                         geometry = {'W':20000,'H':200,'Lc':500, 'Wc':1, 'Hc': 5}, 
                         materials = {'E':1e10, 'nu':0.3, 'rho':910, 'rhow':1024, 'g':9.81},
                         crevasse_location="surface",
                         swell_amplitude=0.0,
                         swell_phase=0.0,
                         swell_wavelength=1000.0,
			 swell_wavelength_in_ice=2000.0,
                         swell_forcing="everything",
                         refinement=3,
                         verbose=False):
    """
    elasticity_solutions solves the equations of plane strain elasticity with boundary conditions
    that are representative of an iceshelf. 

    :param case: gives numerous simplifications of the most realistic boundary conditions. For many
    of the cases, analytical solutions are available (and are displayed if verbose=True).
    :param footloose: is the length of a submarine ice foot (the front of the foot is at zero)
    :return: the solution U and the mesh
    """ 
    
    if verbose:
        print('Running %s model:'%case)
        
    # Meshing parameters
    number_of_refinement_iterations = refinement
    waterline_dy = 2
    ice_front_dy = 4
    crevasse_num_pts = 250
    crevasse_tip_num_pts = 50
    
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
    if 'fl' in geometry.keys():
        footloose=geometry['fl']
    else:
        footloose=0.0
    
    D, flexural_gravity_wavelength, lam = fgl(materials,geometry)
    crevasse_refinement=2*flexural_gravity_wavelength
    
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
        def inside(self, x, on_boundary):
            return ((near(x[1], H) and x[0]>=footloose)\
                    or (near(x[1], Hw) and x[0]<=footloose))\
                    and on_boundary

    class Front(SubDomain):
#         def inside(self, x, on_boundary):
#             return near(x[0], 0) and on_boundary
        def inside(self, x, on_boundary):
            return ((near(x[0], 0) and x[1]<=Hw)\
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
        
    three_corners = [Point(0., 0), Point(W, 0), Point(W, H)]
    ice = build_ice(H,Hw,swell_amplitude,ice_front_dy,waterline_dy,three_corners,footloose)
    crevasse = build_crevasse(geometry,crevasse_location,crevasse_num_pts,crevasse_tip_num_pts)
    

    if (swell_amplitude > 0.0) and (case != 'full-minus-prestress'):
        print ('SWELL IS ONLY IMPLEMENTED IN THE full-minus-prestress CASE.')
        
    
    mesh = generate_mesh ( ice - crevasse, 100)

    # Refine the mesh.  
    # From: https://fenicsproject.discourse.group/t/prescribing-spatially-varying-cell-sizes/527
    d = mesh.topology().dim()

    # Refine in a region of +/- 2*FGL from the crevasse. 
    # Also refine near the ice front.
    refined_region = CompiledSubDomain("((x[0] < Lc + dx) & (x[0] > Lc - dx))\
                        | (x[0] < dx)",H=H,Lc=Lc,dx=crevasse_refinement)
    
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
        
        if (swell_forcing == 'everything')  or (swell_forcing == 'front only'):
            if verbose:
                print('     Applying swell boundary condition on the ice front')
            P_fro  = Expression(("(x[1]<Hw) ? rhow*g*(Hw + A*sin(2*pi*x[0]/L + P) - x[1]) : 0","0"), 
                            degree=1,
                            Hw=Hw, rhow=rhow,g=g,
                            A=swell_amplitude,L=swell_wavelength,P=swell_phase,pi=np.pi)
        else:
            P_fro  = Expression(("(x[1]<Hw) ? rhow*g*(Hw - x[1]) : 0","0"), 
                            degree=1,
                            Hw=Hw, rhow=rhow,g=g)
            
        if (swell_forcing == 'everything') or (swell_forcing == 'bottom only'):
            if verbose:
                print('     Applying swell boundary condition on the ice bottom')
            P_bot  = Expression(("0","(x[1]<Hw) ? rhow*g*(Hw + A*sin(2*pi*x[0]/Li + P) - x[1]) : 0"), 
                            degree=1,
                            Hw=Hw, rhow=rhow,g=g,
                            A=swell_amplitude,L=swell_wavelength_in_ice,P=swell_phase,pi=np.pi)
        else:
            P_bot  = Expression(("0","(x[1]<Hw) ? rhow*g*(Hw - x[1]) : 0"), 
                            degree=1,
                            Hw=Hw, rhow=rhow,g=g)

        P0_fro = Expression(("(x[0]>fl) ? rho*g*(H-x[1]) : rho*g*(Hw-x[1])","0"), degree=1,
                            Hw=Hw, rhow=rhow,g=g, rho=rho, H=H,fl=footloose) 
        P0_bot = Expression(("0","(x[0]>fl) ? rho*g*(H-x[1]) : rho*g*(Hw-x[1])"), degree=1,
                            Hw=Hw, rhow=rhow,g=g, rho=rho, H=H,pi=np.pi,fl=footloose) 
            
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

def build_crevasse(geom,crevasse_location,crevasse_num_pts,crevasse_tip_num_pts):
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

def build_ice(H,Hw,swell_amplitude,ice_front_dy,waterline_dy,boundary_points,footloose):
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

def sif(geom,mats,verbose=False,loc='surface',swell_amplitude=0.0,
        swell_phase=0.0,swell_forcing='everything',refinement=3):
    
    U,mesh = elasticity_solutions(geometry=geom,materials=mats, crevasse_location=loc,
                                 swell_amplitude=swell_amplitude,swell_phase=swell_phase,
                                 swell_forcing=swell_forcing,
                                 refinement=refinement,verbose=verbose)


    Rc = 0.375
    if loc=='surface':
        x1 = geom['Lc']-geom['Wc']/2
        y1 = geom['H']-geom['Hc']+Rc
        x2 = geom['Lc']+geom['Wc']/2
        y2 = geom['H']-geom['Hc']+Rc
        xtip = geom['Lc']
        ytip = geom['H'] - geom['Hc']
    if loc=='bottom':
        x1 = geom['Lc']-geom['Wc']/2
        y1 = geom['Hc']-Rc
        x2 = geom['Lc']+geom['Wc']/2
        y2 = geom['Hc']-Rc
        xtip = geom['Lc']
        ytip = geom['Hc']

    #
    # Plot the location of the CTOD measurements
    #

#     fig,ax=plt.subplots(figsize=(10,5))
#     plot(mesh)
#     plt.plot(x1,y1,'ok')
#     plt.plot(x2,y2,'ok')
#     plt.plot(xtip,ytip,'or')
#     plt.xlim([geom['Lc']-2*geom['Hc'],geom['Lc']+2*geom['Hc']])
#     plt.ylim([geom['H']-geom['Hc']*4,geom['H']])
#     plt.show()
    
    
    r = sqrt( (x1-xtip)**2 + (y1 - ytip)**2)
    u1 = U[0]((x1,y1))
    u2 = U[0]((x2,y2))
    v1 = U[1]((x1,y1))
    v2 = U[1]((x2,y2))

    du = u2 - u1
    dv = v2 - v1

    mu = mats['E']/2./(1+mats['nu'])
    lmbda = mats['E']*mats['nu']/(1+mats['nu'])/(1-2*mats['nu'])
    
    KI = du*sqrt(2*pi / r) * mu/(3-4*mats['nu']+1)
    KII = dv*sqrt(2*pi / r) * mu/(3-4*mats['nu']+1)
    
    return (KI,KII)


def sif_wrapper(swell_phase,this_run,crevasse_location,geom,mats,
                    swell_forcing,verbose=False):
    if verbose:
        print('     Phase = %f rad'%swell_phase)
    g = geom
    g['Lc'] = crevasse_location
    these_Ks = sif(g,mats,verbose=False,loc=this_run, swell_amplitude=1.0,
                swell_phase=swell_phase,swell_forcing=swell_forcing)
    if verbose:
        print('     KI = %f'%these_Ks[0]);
    return these_Ks

def find_extreme_phase(this_run,mode,geom,mats,verbose,swell_forcing,extrema,L):
    if verbose:
        if extrema=='max':
            print('Finding Phase with MAX K:');
        elif extrema=='min':
            print('Finding Phase with MIN K:');
    if mode=='I':
        obj_fun = lambda phase : sif_wrapper(phase,this_run,L,geom,mats,
                                             swell_forcing,verbose)[0]
    elif mode=='II':
        obj_fun = lambda phase : sif_wrapper(phase,this_run,L,geom,mats,
                                             swell_forcing,verbose)[1]

    if extrema=='max':
        # Max of f == the min of -f
        MINUS_ONE = -1
        wrapwrapwrap = lambda x : MINUS_ONE*obj_fun(x)
        extreme_phase,extreme_KI,trash,trash = fminbound(wrapwrapwrap,0,2*np.pi,
                                                         full_output=True,xtol=1e-4)
        extreme_KI=MINUS_ONE*extreme_KI
        
    elif extrema=='min':
        # "Bounded minimization for scalar functions."
        extreme_phase,extreme_KI,trash,trash = fminbound(obj_fun,0,2*np.pi,
                                                 full_output=True,xtol=1e-4)
    return extreme_phase, extreme_KI

def call_pmap(geom,mats,this_run,mode,crevasse_locations,nproc=96,verbose=False,
        swell_forcing='everything',extrema='max'):
    pool = multiprocessing.Pool(processes=nproc)
    find_extreme_phase_partial = partial (find_extreme_phase,this_run,mode,geom,mats,
                                        verbose,swell_forcing,extrema)
    result_list = pool.map(find_extreme_phase_partial, crevasse_locations)
    
    pool.close()
    pool.join()
    
    return result_list

def fgl(mats,geom):
    D = mats['E']/(1-mats['nu']**2) * geom['H']**3 / 12
    flexural_gravity_wavelength = 2*np.pi*(D/(mats['rhow']*mats['g']))**(1/4)
    lam = (4*D/(mats['rhow']*mats['g']))**(1/4)
    return D, flexural_gravity_wavelength, lam

def analytical_KI(geom,mats,swell_height=0):
    
    f_new = lambda x: f(x,geom,mats)
    dK = integrate.quad(f_new, 0, geom['Hc'])
    K_crack_face_loading_surface = 2*mats['rho']*mats['g']/np.sqrt(np.pi*geom['Hc']) * dK[0]

    sig0 = mats['rho']*mats['g']*(geom['H']+swell_height) / 2 *(1-mats['rho']/mats['rhow'])
    KI_analytical = 1.12 * sig0 * sqrt(np.pi * geom['Hc'])
    KI_analytical += K_crack_face_loading_surface
    
    f_bot_new = lambda x: f_bot(x,geom,mats)
    dK_bot = integrate.quad(f_bot_new, 0, geom['Hc'])
    K_crack_face_loading_bottom  =  2/np.sqrt(np.pi*geom['Hc']) * dK_bot[0]
    KI_analytical_bottom = KI_analytical - K_crack_face_loading_surface
    
    return KI_analytical, KI_analytical_bottom

def analytical_KI_bending(geom,mats,Lcs):
    
    D, flexural_gravity_wavelength,lam = fgl(mats,geom)
    r = mats['rho']/mats['rhow']
    m0 = mats['rho']*mats['g']*geom['H']**3 / 12 * (3*r - 2*r**2 - 1)
    II = geom['H']**3 / 12
    M_flex = m0 * np.exp(-Lcs/lam)*(np.cos(Lcs/lam) + np.sin(Lcs/lam))
    sig_flex = M_flex * geom['H']/2 / II 

    KI_analytical, KI_analytical_bottom = analytical_KI(geom,mats)

    KI_analytical_bending = KI_analytical/1e6 \
                + 1.12 * (sig_flex) * np.sqrt(np.pi * geom['Hc']) / 1e6

    KI_analytical_bending_bottom = KI_analytical_bottom/1e6 \
                              - 1.12 * (sig_flex) * np.sqrt(np.pi * geom['Hc']) / 1e6
    
    return KI_analytical_bending,KI_analytical_bending_bottom

def f(y,geom,mats):
    # Integral kernel for Surface Crevasses
    gamma = y/geom['Hc']
    lambd = geom['Hc']/geom['H']
    val =  3.52*(1-gamma)/(1-lambd)**(3/2)
    val += - (4.35-5.28*gamma)/(1-lambd)**(1/2)
    val += ( (1.30 - 0.3*gamma**(3/2)) / (1-gamma**2)**(1/2) + 0.83 - 1.76*gamma) \
            * (1 - (1-gamma)*lambd)
    return val


def f_bot(y,geom,mats):
    # Integral kernel for bottom crevasses
    Hw = mats['rho']/mats['rhow'] * geom['H']
    sig = lambda y: mats['rhow']*mats['g']*(Hw-y) - mats['rho']*mats['g']*(geom['H']-y)
    gamma = y/geom['Hc']
    lambd = geom['Hc']/geom['H']
    kernel =  3.52*(1-gamma)/(1-lambd)**(3/2)
    kernel += - (4.35-5.28*gamma)/(1-lambd)**(1/2)
    kernel += ( (1.30 - 0.3*gamma**(3/2)) / (1-gamma**2)**(1/2) + 0.83 - 1.76*gamma) \
                * (1 - (1-gamma)*lambd)
    val = sig(y) * kernel
    return val
