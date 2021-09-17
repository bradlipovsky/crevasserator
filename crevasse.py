from dolfin import *
from mshr import *
# import matplotlib.pyplot as plt
# from time import perf_counter
import numpy as np
# import pickle
# import scipy.integrate as integrate
# import scipy.special as special
# from os import path
from scipy.optimize import fminbound
def elasticity_solutions(case='full-minus-prestress', 
                         geometry = {'W':20000,'H':200,'Lc':500, 'Wc':1, 'Hc': 5}, 
                         materials = {'E':1e10, 'nu':0.3, 'rho':910, 'rhow':1024, 'g':9.81},
                         crevasse_location="surface",
                         swell_amplitude=0.0,
                         swell_phase=0.0,
                         swell_wavelength=1000.0,
                         verbose=False):

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
            return near(x[1], H) and on_boundary

    class Front(SubDomain):
        def inside(self, x, on_boundary):
            return near(x[0], 0) and on_boundary
        
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


    # Define the geometry and the mesh
    
    boundary_points = [Point(0., 0), Point(W, 0), Point(W, H)]
       
    ice_front_num_pts=H # One meter spacing at ice front
    for i in range(ice_front_num_pts):
        boundary_points.append( Point(0.0,float((ice_front_num_pts-i)*H/ice_front_num_pts)))
    
    ice = Polygon( boundary_points )
    
    
    crevasse_num_pts = 250
    crevasse_tip_num_pts = 50
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


    if (swell_amplitude > 0.0) and (case != 'full-minus-prestress'):
        print ('SWELL IS ONLY IMPLEMENTED IN THE full-minus-prestress CASE.')
    
        
    crevasse = Polygon(crevasse_points)
    
    mesh = generate_mesh ( ice - crevasse, 100)

    # Refine the mesh.  From: https://fenicsproject.discourse.group/t/prescribing-spatially-varying-cell-sizes/527
    d = mesh.topology().dim()
    refined_region = CompiledSubDomain("((x[0] < Lc + H/2) & (x[0] > Lc - H/2)) | (x[0] < Lc+2*H)",H=H,Lc=Lc)
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

#     V = VectorFunctionSpace(mesh, 'Lagrange', degree=1)
    V = VectorFunctionSpace(mesh, 'CG', 2)
    u = TrialFunction(V)
    v = TestFunction(V)
    
   
    if case == 'full-minus-prestress':
       
        P0_fro = Expression(("rho*g*(H-x[1]) ","0"), degree=1,
                            Hw=Hw, rhow=rhow,g=g, rho=rho, H=H) 
        
        P_fro  = Expression(("(x[1]<Hw) ? rhow*g*(Hw + A*sin(2*pi*x[0]/L + P) - x[1]) : 0","0"), 
                            degree=1,
                            Hw=Hw, rhow=rhow,g=g,
                            A=swell_amplitude,L=swell_wavelength,P=swell_phase,pi=np.pi)
        
        P_bot  = Expression(("0","(x[1]<Hw) ? rhow*g*(Hw + A*sin(2*pi*x[0]/L + P)-x[1]) : 0"), 
                            degree=1,
                            Hw=Hw, rhow=rhow,g=g,
                            A=swell_amplitude,L=swell_wavelength,P=swell_phase)
        
        P0_bot = Expression(("0","rho*g*(H-x[1]) "), degree=1,
                            Hw=Hw, rhow=rhow,g=g, rho=rho, H=H,pi=np.pi) 
        
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

        P0_fro = Expression(("rho*g*H*(1-rho/rhow) ","0"), degree=1,Hw=Hw, rhow=rhow,g=g, rho=rho, H=H) 
        
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
    
    U = Function(V)
    solve(a==L,U,bc)
    
    
    ux,uz=-1,-1
    x0 = 0
    
    
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
        print ('---- %s -----'%case)
        print('Vertical deflection:')
        if uz != -1:
            print ('    Analytical: %f'%uz)
        print ('    Numerical:  %f'%U(x0,H)[1])
        print('Horizontal deflection:')
        if ux!=-1:
            print ('    Analytical: %f'%ux)
        print ('    Numerical:  %f'%U(x0,H/2)[0])

        print(' ')
    return U,mesh

def eps(v):
    return sym(grad(v))

def sigma(v,lmbda,mu):
    dim = v.geometric_dimension()
    return 2.0*mu*eps(v) + lmbda*tr(eps(v))*Identity(dim)

def sif(geom,mats,verbose=False,loc='surface',swell_amplitude=0.0,swell_phase=0.0):
    
    U,mesh = elasticity_solutions(geometry=geom,crevasse_location=loc,
                                 swell_amplitude=swell_amplitude,swell_phase=swell_phase)


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


def sif_wrapper(swell_phase,this_run,crevasse_location,geom,mats,verbose=False):
    if verbose:
        print('     Phase = %f rad'%swell_phase)
    g = geom
    g['Lc'] = crevasse_location
    these_Ks = sif(g,mats,verbose=False,loc=this_run, swell_amplitude=1.0,swell_phase=swell_phase)
    if verbose:
        print('     KI = %f'%these_Ks[0]);
    return these_Ks

def find_max_phase(this_run,mode,L,geom,mats,verbose=False):
    if verbose:
        print('Finding Phase with max K:');
    if mode=='I':
        obj_fun = lambda phase : sif_wrapper(phase,this_run,L,geom,mats,verbose)[0]
    elif mode=='II':
        obj_fun = lambda phase : sif_wrapper(phase,this_run,L,geom,mats,verbose)[1]
        
    max_phase,max_KI,trash,trash = fminbound(obj_fun,0,2*np.pi,full_output=True,xtol=1e-3)
#     print('Just finished length %f'%L)
    return max_phase, max_KI

def call_pmap(this_run,mode):
    
    pool = multiprocessing.Pool(processes=96)
    find_max_phase_partial = partial (find_max_phase,this_run,mode)
    result_list = pool.map(find_max_phase_partial, Lcs_swell)
    
    pool.close()
    pool.join()
    
    return result_list