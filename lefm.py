import numpy as np
from scipy.optimize import fminbound

def sif(U,Xc,Yc,Wc,prefactor,verbose=0,loc='surface',Rc = 0.375):
    '''
    sif(U,Wc,Xc,prefactor,verbose=False,loc='surface')

    Calculates stress intensity factors (SIFs) from an input elasticity 
    solution, U. The crevasse tip is located at (x,y)=(Xc,Yc) and has a 
    width Wc. prefactor is equal to mu/(3-4*nu+1), with mu and nu being the 
    shear modulus and Poisson ratio for ice, respectively.
    
    If 'loc'=='surface' then the crevasse is assumed to be above (Xc,Yc).
    If 'loc'=='bottom' then the crevasse is assumed to be below (Xc,Yc).

    Rc is the distance from the crack tip where the SIF is measured. This
    point should be at least several elements away from the tip to make 
    sure that the field is well-resovled.

    Outputs both Mode I and Mode II sifs.

    '''    

    x1 = Xc-Wc/2
    x2 = Xc+Wc/2
    if loc=='surface':
        y1 = Yc+Rc
        y2 = Yc+Rc
    if loc=='bottom':
        y1 = Yc-Rc
        y2 = Yc-Rc

    if verbose > 1:
        print(f'(x1,y1)=({x1},{y1})')
        print(f'(x2,y2)=({x2},{y2})')
    u1 = U[0]((x1,y1))
    u2 = U[0]((x2,y2))
    v1 = U[1]((x1,y1))
    v2 = U[1]((x2,y2))

    du = u2 - u1
    dv = v2 - v1
    
    r = np.sqrt( (x1-Xc)**2 + (y1 - Yc)**2)
    KI = du*np.sqrt(2*np.pi / r) * prefactor
    KII = dv*np.sqrt(2*np.pi / r) * prefactor
    
    return (KI,KII)


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
