import numpy as np
from scipy.optimize import fminbound

def sif(geom,mats,verbose=False,loc='surface',swell_amplitude=0.0,
        swell_phase=0.0,swell_forcing='everything',refinement=3):
    '''
    This function calculates SIFs from an elasticity solution.
    '''    


    # First, calculate the elasticity solution.
    U,mesh = elasticity_solutions(geometry=geom,
                materials=mats, crevasse_location=loc,
                swell_amplitude=swell_amplitude,swell_phase=swell_phase,
                swell_forcing=swell_forcing,
                refinement=refinement,verbose=verbose)

    # Next, calculate the SIFs.
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
    '''
    This function is a helper function that puts the function sif() 
    into a form that can be called in the objective function called 
    by fminbound within the function find_extreme_phase.
    '''

    g = geom
    g['Lc'] = crevasse_location
    these_Ks = sif(g,mats,verbose=False,loc=this_run, swell_amplitude=1.0,
                swell_phase=swell_phase,swell_forcing=swell_forcing)
    if verbose:
        print(f"     KI(p={swell_phase:.2f},L={geom['Lc']:.2f})"\
                f" = {these_Ks[0]:.2f}");
    return these_Ks



def find_extreme_phase(this_run,mode,geom,
                        mats,verbose,
                        swell_forcing,extrema,L):
    '''
    Calculates the wave phase at which the extremal SIF is achieved.  This
    function is run using multiprocessing in the function call_pmap, below.
    '''

    if verbose:
        print(f'SEARCHING for {extrema} K{mode} at L={L:.2f}...');
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
        extreme_phase,extreme_KI,trash,trash = fminbound(wrapwrapwrap,
                                         0,2*np.pi,
                                         full_output=True,xtol=1e-3)
        extreme_KI=MINUS_ONE*extreme_KI
        
    elif extrema=='min':
        # "Bounded minimization for scalar functions."
        extreme_phase,extreme_KI,trash,trash = fminbound(obj_fun,0,2*np.pi,
                                                 full_output=True,xtol=1e-3)
    if verbose:
        print(f'FOUND {extrema} K{mode} at L={L:.2f}.');

    return extreme_phase, extreme_KI

def call_pmap(geom,mats,this_run,mode,
            crevasse_locations,nproc=96,verbose=False,
            swell_forcing='everything',extrema='max'):
    '''
    This is the function you want to call if you want to run a parameter
    space study in parallel. call_pmap is really just a wrapper that 
    provides a nicer interface to the function find_extreme_phase through
    partial and pool.map.
    '''

    pool = multiprocessing.Pool(processes=nproc)
    
    find_extreme_phase_partial = partial (find_extreme_phase,
                                        this_run,mode,geom,mats,
                                        verbose,swell_forcing,extrema)
    if verbose:
        print('Created pool. Calling map.')
        
    result_list = pool.map(find_extreme_phase_partial, crevasse_locations)
    
    pool.close()
    pool.join()
    
    return result_list

def fgl(mats,geom):
    '''
    Calculates the flexural gravity wavelength and the flexural rigidity.
    '''

    D = mats['E']/(1-mats['nu']**2) * geom['H']**3 / 12
    flexural_gravity_wavelength = 2*np.pi*\
        (D/(mats['rhow']*mats['g']))**(1/4)
    lam = (4*D/(mats['rhow']*mats['g']))**(1/4)
    return D, flexural_gravity_wavelength, lam



def analytical_KI(geom,mats,swell_height=0):
    '''
    Calculates the van der Veen analytical solutions for KI
    '''
        
    f_new = lambda x: f(x,geom,mats)
    dK = integrate.quad(f_new, 0, geom['Hc'])
    K_crack_face_loading_surface = 2*mats['rho']*\
            mats['g']/np.sqrt(np.pi*geom['Hc']) * dK[0]

    sig0 = mats['rho']*mats['g']*(geom['H']+swell_height) / \
            2 *(1-mats['rho']/mats['rhow'])
    KI_analytical = 1.12 * sig0 * sqrt(np.pi * geom['Hc'])
    KI_analytical += K_crack_face_loading_surface
    
    f_bot_new = lambda x: f_bot(x,geom,mats)
    dK_bot = integrate.quad(f_bot_new, 0, geom['Hc'])
    K_crack_face_loading_bottom  =  2/np.sqrt(np.pi*geom['Hc']) * dK_bot[0]
    KI_analytical_bottom = KI_analytical - K_crack_face_loading_surface
    
    return KI_analytical, KI_analytical_bottom



def analytical_KI_bending(geom,mats,Lcs):
    '''
    Calculates analytical solutions that combine the van der Veen 
    solution with bending stresses.
    ''' 
 
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

def analytical_KI_footloose(geom,mats,Lcs):
    '''
    Calculates analytical solutions that combine the van der Veen 
    solution with bending stresses from footloose mechanism.
    ''' 
    
    D, flexural_gravity_wavelength,lam = fgl(mats,geom)
    r = mats['rho']/mats['rhow']
    m0 = mats['rho']*mats['g']*geom['H']**3 / 12 * (3*r - 2*r**2 - 1)
    II = geom['H']**3 / 12
    
    load = geom['fl'] * mats['g'] * mats['rho'] *(1-r)*geom['H']
    M_load = load * lam * np.exp(-Lcs/lam)* np.sin(Lcs/lam)
    sig_load = -M_load * geom['H']/2 / II 

    KI_analytical, KI_analytical_bottom = analytical_KI(geom,mats)

    M_flex = m0 * np.exp(-Lcs/lam)*(np.cos(Lcs/lam) + np.sin(Lcs/lam))
    sig_flex = M_flex * geom['H']/2 / II 
    KI_analytical_bending =  + 1.12 * (sig_flex) \
            * np.sqrt(np.pi * geom['Hc'])
    KI_analytical_bending_bottom =  - 1.12 * (sig_flex) \
            * np.sqrt(np.pi * geom['Hc'])
    
    KI_analytical_bending = KI_analytical/1e6 + KI_analytical_bending/1e6\
                + 1.12 * (sig_load) * np.sqrt(np.pi * geom['Hc']) / 1e6 \
            

    KI_analytical_bending_bottom = KI_analytical_bottom/1e6 \
                    + KI_analytical_bending_bottom/1e6\
                    - 1.12 * (sig_load) * np.sqrt(np.pi * geom['Hc']) / 1e6
    
    return KI_analytical_bending,KI_analytical_bending_bottom


def f(y,geom,mats):
    '''
    van der Veen Integral kernel for Surface Crevasses
    '''
    gamma = y/geom['Hc']
    lambd = geom['Hc']/geom['H']
    val =  3.52*(1-gamma)/(1-lambd)**(3/2)
    val += - (4.35-5.28*gamma)/(1-lambd)**(1/2)
    val += ( (1.30 - 0.3*gamma**(3/2)) \
            / (1-gamma**2)**(1/2) + 0.83 - 1.76*gamma) \
            * (1 - (1-gamma)*lambd)
    return val


def f_bot(y,geom,mats):
    '''
    van der Veen integral kernel for bottom crevasses
    '''
    Hw = mats['rho']/mats['rhow'] * geom['H']
    sig = lambda y: mats['rhow']*mats['g']*(Hw-y)\
             - mats['rho']*mats['g']*(geom['H']-y)
    gamma = y/geom['Hc']
    lambd = geom['Hc']/geom['H']
    kernel =  3.52*(1-gamma)/(1-lambd)**(3/2)
    kernel += - (4.35-5.28*gamma)/(1-lambd)**(1/2)
    kernel += ( (1.30 - 0.3*gamma**(3/2)) / \
            (1-gamma**2)**(1/2) + 0.83 - 1.76*gamma) \
                * (1 - (1-gamma)*lambd)
    val = sig(y) * kernel
    return val

def test_filename(filename):
    if path.exists(filename):
        print('  ')
        print('The output filename has already been used. \n'\
              'To be safe, rename this file if you want to re-run.')
        val = input("Type YES to continue.... ")
        if val!='YES':
            exit()
        print('  ')
