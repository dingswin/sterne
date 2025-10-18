#!/usr/bin/env python
"""
written in python3 by Hao Ding.
"""
import numpy as np
import astropy.units as u
from astropy import constants as const
import os, sys
from sterne import others
from psrqpy import QueryATNF
from shutil import which


def generate_parfile(pulsar):
    """
    The parfile generation only works for pulsar listed in PSRCAT.
    """
    parfile = pulsar + '.par'
    if which('psrcat')==None:
        print('PSRCAT has not been installed. Please install it from https://www.atnf.csiro.au/people/pulsar/psrcat/download.html.')
    # alias psrcat="psrcat -db_file $PSRCAT_FILE"
    # where $PSRCAT_FILE points to the psrcat.db
    results = os.system("psrcat %s -e > %s" % (pulsar, parfile))
    readfile = open(parfile, 'r')
    contents = readfile.read()
    readfile.close()
    if ('WARNING' in contents) and ('not in catalogue' in contents):
        print('Pulsar not found in PSRCAT! Aborting...')
        sys.exit()
    else:
        print('parfile for %s is made.' % pulsar)

def read_parfile(parfile):
    """
    Note
    ----
    For pulsars listed in PSRCAT, parfile can be made with generate_parfile.
    For other sources, parfile needs to be prepared by oneself, in accordance
    with the format of PSRCAT format.
    It is always important to update parameters to latest values before using them.
    
    Return parameters
    -----------------
    dict_parameter : dict ({str:float})
        Dictionary of parameters including following keys
        pb - binary orbital period (d);
        ecc - eccentricity;
        a1 - projected semi-major axis of orbit (m);
        t0 - epoch of periastron (MJD);
        om - omega, longitude of periastron (deg);
        omdot - periastron advance (deg/yr);
        om_asc - position angle of ascending node (deg);
        pbdot - first time derivative of orbital period (s/s);
        a1dot - first time derivative of A1 (m/s);
        sini - sine of inclination angle;
    """
    readfile = open(parfile, 'r')
    lines = readfile.readlines()
    readfile.close()
    keywords_needed = ['DECJ', 'PB ', 'ECC', 'A1 ', 'T0', 'OM ', 'OMDOT', 'OM_ASC', 'PBDOT', 'A1DOT', 'SINI', 'raj']
    parameters = [kw.strip().lower() for kw in keywords_needed]
    dict_parameter = {}
    for line in lines:
        for i in range(len(keywords_needed)):
            if (keywords_needed[i] in line) and (not line.startswith('#')):
                alist = line.split(' '*8)
                alist = [element.strip() for element in alist]
                while True:
                    try:
                        alist.remove('')
                    except ValueError:
                        break
                if parameters[i] == 'decj':
                    dict_parameter[parameters[i]] = alist[1]
                else:
                    dict_parameter[parameters[i]] = float(alist[1])
    dict_parameter['pb'] *= u.d
    dict_parameter['a1'] *= const.c * u.s
    dict_parameter['t0'] *= u.d
    dict_parameter['om'] *= u.deg
    try:
        dict_parameter['omdot'] *= u.deg/u.yr
    except KeyError:
        pass
    try:
        dict_parameter['om_asc'] *= u.deg
    except KeyError:
        pass
    try:
        dict_parameter['a1dot'] *= const.c
    except KeyError:
        pass
    try:
        dict_parameter['decj'] = u.deg * others.dms2deg(dict_parameter['decj'])
        dict_parameter['raj'] = u.deg * 15 * others.dms2deg(dict_parameter['raj'])
    except KeyError:
        print("'decj' is essential but is missing from %s. try to get it with psrqpy..." % parfile)
        psrname = parfile.split('.')[0]
        print('Guess the pulsar name to be %s' % psrname)
        query1 = QueryATNF(psrs=[psrname], params=['DECJ', 'RAJ'])
        decj = str(query1['DECJ'][0])
        decj = u.deg * others.dms2deg(decj)
        raj = str(query1['RAJ'][0])
        raj = u.deg * 15 * others.dms2deg(raj)
        print('decj=')
        print(decj)
        dict_parameter['decj'] = decj
        dict_parameter['raj'] = raj
    return dict_parameter

def solve_u(e, c, precision=1e-5):
    """
    Solve the equation
    u - e * sin(u) = c
    in a numerical way.
    Here, c stands for a constant not relevant to speed of light;
    e stands for eccentricity.

    Return parameters
    -----------------
    u : float
    iterations : int
    """
    x = c/(1-e) #first order approximation: sin(u) = u
    x1 = float('inf')
    iterations = 0
    while abs(x - x1) > precision:
        x1 = x
        x = e * np.sin(x1) + c
        iterations += 1
    return x, iterations

def reflex_motion(epoch, dict_of_orbital_parameters, incl, Om_asc, px):
    """
    Following mathematical formalism detailed in Eqn 55 through 63 
        in the Tempo2 paper (ref1), except that 1) the sign in Eqn 61 appears to be a typo
        and has been corrected, 2) the power-law index in Eq. 58 should be 0.5 in stead of 2.

    Caveats
    -------
    1. Time derivative of eccentricity is not taken into account.
    2. Two differently formulated A_u in Eqn 57 and Eqn 58 is considered the same.
    3. Relativistic deformations of the eccentricity, given by Eqn 59 and 60, is
        not taken into account.
    4. note that Eq. 58 has a typo, in the power-law index 2 (supposed to 0.5 according to Eq. 17a of Damour and Deruelle, 1986).



    Input paramters
    ---------------
    epoch : float
        in MJD.
    dict_of_orbital_parameters : dict
        See the function read_parfile()
    incl : float
        Inclination angle (rad).
    Om_asc : float
        Position angle of ascending node (deg).
    px : float
        Parallax (mas).

    Return parameters
    -----------------
    dRA : float
        Reflex-motion-related right ascension offset (in mas), corresponding to the vector e1
        in Eqn 54.
    dDEC : float
        Reflex-motion-related declination offset (in mas), corresponding to the vector e2
        in Eqn 54.

    Reference
    ---------
    1. Edwards, Hobbs and Manchester 2006 (2006MNRAS.372.1549E). 
    """
    DoP = dict_of_orbital_parameters
    epoch *= u.d
    incl *= u.rad
    Om_asc *= u.deg
    e, T0, Pb0, omega0, a0, dec = DoP['ecc'], DoP['t0'],\
        DoP['pb'], DoP['om'], DoP['a1'], DoP['decj']
    try:
        omdot = DoP['omdot']
    except KeyError:
        omdot = 0 * u.deg/u.s
    try:
        Pbdot = DoP['pbdot']
    except KeyError:
        Pbdot = 0
    try:
        adot = DoP['a1dot']
    except KeyError:
        adot = None

    n = (2*np.pi/Pb0 - np.pi*Pbdot*(epoch-T0)/(Pb0**2)) * u.rad #angular velocity
    #n = 2*np.pi/Pb0 + np.pi*Pbdot*(epoch-T0)/(Pb0**2) #angular velocity
    u1 = solve_u(e, (n*(epoch-T0)).value)[0] #u1 stands for u, not to clash with u=astropy.units
    #u1 *= u.rad
    A_u = u.rad * 2* np.arctan(((1+e)/(1-e))**0.5 * np.tan(u1/2))
    k = omdot/n
    omega = omega0 + k * A_u
    theta = omega + A_u
    if adot != None:
        a1 = a0 + adot * (epoch - T0) #equivalent to Eqn 71
    else:
        a1 = a0
    b_abs = a1 * (1 - e * np.cos(u1))
    b_AU = b_abs.to(u.AU).value
    offset = b_AU * px
    matr1 = np.mat([[np.sin(Om_asc), -np.cos(Om_asc), 0],
                    [np.cos(Om_asc), np.sin(Om_asc), 0],
                    [0, 0, 1]])
    #matr1 = np.mat([[np.cos(Om_asc), -np.sin(Om_asc), 0],
    #                [np.sin(Om_asc), np.cos(Om_asc), 0],
    #                [0, 0, 1]])
    matr2 = np.mat([[1, 0, 0],
                    [0, -np.cos(incl), -np.sin(incl)],
                    [0, np.sin(incl), -np.cos(incl)]])
    #matr2 = np.mat([[1, 0, 0],
    #                [0, np.cos(incl), -np.sin(incl)],
    #                [0, np.sin(incl), np.cos(incl)]])
    matr3 = np.mat([[offset*np.cos(theta)],
                    [offset*np.sin(theta)],
                    [0]])
    b = matr1 * matr2 * matr3
    dRA = (b.item(0,0)/np.cos(dec)).value #that can be directly added to RA
    dDEC = b.item(1,0)
    return np.array([dRA, dDEC]) #in mas
    
def __solve_u(e, M, tol=1e-12, maxiter=50):
    """
    Solve Kepler's equation: u - e*sin(u) = M
    via Newton–Raphson (radians in/out).
    """
    # initial guess: M + e*sin(M)
    U = M + e * np.sin(M)
    for _ in range(maxiter):
        f   = U - e*np.sin(U) - M
        fp  = 1 - e*np.cos(U)
        du  = -f/fp
        U  += du
        if abs(du) < tol:
            break
    return U

def __reflex_motion(epoch, DoP, incl, Om_asc, px):
    """
    Compute VLBI reflex‐motion offsets (dRA, dDEC) in mas.

    epoch : float (MJD)
    DoP    : dict of orbital parameters with astropy units
    incl   : inclination angle (rad)
    Om_asc : longitude of ascending node (deg)
    px     : parallax (mas)
    """
    # unpack & units
    incl   = incl * u.rad
    Om_asc = Om_asc * u.deg
    e      = DoP['ecc']
    T0     = DoP['t0']
    Pb0    = DoP['pb']
    omega0 = DoP['om']
    a1_0   = DoP['a1']
    decj   = DoP['decj']
    omdot  = DoP.get('omdot', 0*u.rad/u.s).to(u.rad/u.s)
    pbdot  = DoP.get('pbdot', 0)       # in s/s
    xpbdot = DoP.get('xpbdot', 0)      # in s/s
    adot   = DoP.get('a1dot', None)    # in m/s

    # time since periastron, in seconds
    dt     = (epoch*u.d - T0).to(u.s).value
    Pb_s   = Pb0.to(u.s).value

    # orbits & phase (with pbdot+xpbdot correction)
    orbits  = dt/Pb_s - 0.5*(pbdot + xpbdot)*(dt/Pb_s)**2
    norbits = np.floor(orbits)
    phase   = 2*np.pi*(orbits - norbits)

    # eccentric anomaly u (rad)
    U = solve_u(e, phase)

    # true anomaly shift A_e(u)
    Ae = 2 * np.arctan(np.sqrt((1+e)/(1-e)) * np.tan(U/2))

    # periastron advance k = omdot / n
    n  = 2*np.pi / Pb_s
    k  = omdot.value / n

    # total argument of latitude
    omega = omega0.to(u.rad).value + k*Ae
    theta = omega + Ae

    # time‐varying semi‐major axis
    if adot is not None:
        a1 = a1_0 + adot * (epoch*u.d - T0)
    else:
        a1 = a1_0

    # orbital radius in AU
    r  = a1 * (1 - e*np.cos(U))
    r_AU = r.to(u.AU).value

    # sky‐plane offset (mas)
    off = r_AU * px

    # 3D→sky rotation
    incl_rad = incl.to(u.rad).value
    Om_rad   = Om_asc.to(u.rad).value
    R1 = np.array([[ np.sin(Om_rad), -np.cos(Om_rad), 0],
                   [ np.cos(Om_rad),  np.sin(Om_rad), 0],
                   [            0   ,             0   , 1]])
    R2 = np.array([[1,             0             ,           0           ],
                   [0, -np.cos(incl_rad), -np.sin(incl_rad)],
                   [0,  np.sin(incl_rad), -np.cos(incl_rad)]])
    vec = off * np.array([np.cos(theta), np.sin(theta), 0])
    b   = R1.dot(R2.dot(vec))

    # proper motion correction in RA
    dRA  = b[0] / np.cos(decj.to(u.rad).value)
    dDEC = b[1]

    return np.array([dRA, dDEC])  # in mas




def __reflex_motion(epoch, par, incl, Om_asc, px):
    """
    epoch   : array of MJDs (float)
    par     : dict with keys PB, T0, E, A1, OM, OMDOT (astropy Quantities or float)
    incl    : inclination (Quantity deg)
    Om_asc  : node (Quantity deg)
    px      : parallax (Quantity mas)
    returns : array shape (N,2) = [dRA(mas), dDec(mas)]
    """
    # unpack & cast to floats
    TWOPI = 2 * np.pi
    incl = incl * u.rad
    Om_asc = Om_asc * u.deg
    px = px * u.mas

    e      = float(par['ecc'])
    T0     = par['t0'].to(u.day).value
    PB     = par['pb'].to(u.day).value
    omega0 = par['om'].to(u.rad).value
    a1     = par['a1'].to(u.lightsecond).value
    omdot  = par.get('omdot', 0*u.deg/u.yr).to(u.rad/u.s).value

    ra0    = par['raj'].to(u.rad).value
    dec0   = par['decj'].to(u.rad).value
    incl   = incl.to(u.rad).value
    node   = Om_asc.to(u.rad).value
    px     = px.to(u.mas).value

    # times and anomalies
    dt     = (epoch - T0) * 86400.0        # seconds since T0
    M      = TWOPI * ((dt / (PB * 86400.0)) % 1.0)
    u_an   = solve_u(e, M)
    Ae     = 2 * np.arctan(np.sqrt((1+e)/(1-e)) * np.tan(u_an/2))
    omega  = omega0 + omdot * dt
    theta  = omega + Ae

    # orbital radius → mas
    r_ls   = a1 * (1 - e * np.cos(u_an)) * u.lightsecond  # light-sec
    r_AU   = r_ls.to(u.AU).value
    R      = r_AU * px                   # mas

    # break into equatorial components (Presto’s formula)
    cO, sO = np.cos(node), np.sin(node)
    ci, si = np.cos(incl),  np.sin(incl)
    X = R * (np.cos(theta)*cO - np.sin(theta)*sO*ci)
    Y = R * (np.cos(theta)*sO + np.sin(theta)*cO*ci)
    Z = R * (            np.sin(theta)*si)

    # sky‐unit vectors
    eRA  = np.array([-np.sin(ra0),  np.cos(ra0),   0.0]) / np.cos(dec0)
    eDec = np.array([
        -np.cos(ra0)*np.sin(dec0),
        -np.sin(ra0)*np.sin(dec0),
         np.cos(dec0)
    ])

    # dot to get offsets
    dRA  = X * eRA[0]  + Y * eRA[1]  + Z * eRA[2]
    dDec = X * eDec[0] + Y * eDec[1] + Z * eDec[2]

    return np.array([dRA, dDec])  

def __reflex_motion(epoch, DoP, incl, Om_asc, px):
    """
    epoch    : MJD (float or array)
    DoP      : dict from read_parfile(), containing:
                 'ecc'   : float (unitless)
                 't0'    : Quantity in u.d
                 'pb'    : Quantity in u.d
                 'om'    : Quantity in u.deg
                 'omdot' : Quantity in u.deg/u.yr  (optional)
                 'a1'    : Quantity in metres (a1*c*u.s)
                 'decj'  : Quantity in u.deg
    incl     : float or Quantity in u.deg  (CCW from North→East)
    Om_asc   : float or Quantity in u.deg  (CCW from North→East)
    px       : float or Quantity in u.mas

    returns  : np.ndarray([dRA, dDEC]) in mas, following PRESTO’s
               reflex_motion formalism (Ω measured CW from East→North)
    """
    # —————————————————————————————————————————————————————
    # 1) Constants (PRESTO style)
    TWOPI       = 2.0 * np.pi
    DEGTORAD    = np.pi / 180.0
    SECPERJULYR = 31557600.0
    LSKPC2MAS   = 2.003988804115705e-03

    # 2) Unpack and convert to plain floats
    e      = DoP['ecc']
    T0     = DoP['t0'].to(u.d).value
    Pb     = DoP['pb'].to(u.d).value
    OM0    = DoP['om'].to(u.deg).value
    OMDOT  = DoP.get('omdot', 0*u.deg/u.yr).to(u.deg/u.yr).value
    # get a1 in light-seconds
    A1_sec = (DoP['a1'] / const.c).to(u.s).value
    dec    = DoP['decj'].to(u.rad).value

    # allow incl, Om_asc, px as Quantity or float
    incl_deg = incl.to(u.deg).value   if hasattr(incl, 'unit') else incl
    Om_user  = Om_asc.to(u.deg).value if hasattr(Om_asc, 'unit') else Om_asc
    px_mas   = px.to(u.mas).value     if hasattr(px, 'unit') else px

    # 3) Time since periastron
    t      = np.atleast_1d(epoch)
    dt_days = t - T0

    # 4) Mean anomaly M (rad)
    M = TWOPI * ((dt_days % Pb) / Pb)

    # 5) Eccentric → true anomaly
    EA = solve_u(e, M)
    TA = 2.0 * np.arctan(np.sqrt((1+e)/(1-e)) * np.tan(EA/2.0))

    # 6) Argument of periastron ω(t) (rad)
    ω_deg = OM0 + (dt_days * 86400.0)/SECPERJULYR * OMDOT
    ω_rad = ω_deg * DEGTORAD

    # 7) Orbital phase φ = TA + ω
    φ = TA + ω_rad

    # 8) Deproject semi-major axis a = A1/sin(i)
    i_rad   = incl_deg * DEGTORAD
    sin_i   = np.sin(i_rad)
    a_light = A1_sec / sin_i

    # 9) Radius vector r (light-sec)
    r = a_light * (1.0 - e**2) / (1.0 + e * np.cos(TA))

    # 10) Damour & Taylor coords (light-sec)
    xs = -r * np.sin(φ) * sin_i
    ys = -r * np.cos(φ)
    zs = -r * np.sin(φ) * np.cos(i_rad)

    # 11) Angular offsets (mas)
    dist_kpc = 1.0 / px_mas
    ys_mas   = -ys / dist_kpc * LSKPC2MAS
    zs_mas   = -zs / dist_kpc * LSKPC2MAS

    # 12) Convert YOUR CCW-from-North Ω → PRESTO’s CW-from-East
    #     PRESTO wants Ω_presto measured CW from East→North,
    #     so Ω_presto = 90° − Ω_user
    Ωp_rad = (90.0 - Om_user) * DEGTORAD
    cO, sO = np.cos(Ωp_rad), np.sin(Ωp_rad)

    # 13) PRESTO’s sky‐plane projection
    dRA  = (cO * ys_mas - sO * zs_mas) / np.cos(dec)
    dDEC =  sO * ys_mas + cO * zs_mas

    # 14) Return consistent shape
    result = np.vstack([dRA, dDEC])
    if result.shape[1] == 1:
        return result[:,0]
    return result

# Keep your original solve_u function, it is correct.
def __solve_u(e, M, tol=1e-12, maxiter=50):
    """
    Solve Kepler's equation: u - e*sin(u) = M
    via Newton–Raphson (radians in/out).
    """
    U = M + e * np.sin(M)
    for _ in range(maxiter):
        f = U - e*np.sin(U) - M
        fp = 1 - e*np.cos(U)
        du = -f/fp
        U += du
        if abs(du) < tol:
            break
    return U

def __reflex_motion(epoch, dict_of_orbital_parameters, incl, Om_asc, px):
    """
    Following mathematical formalism detailed in Eqn 55 through 63
        in the Tempo2 paper (ref1), except that 1) the sign in Eqn 61 appears to be a typo
        and has been corrected, 2) the power-law index in Eq. 58 should be 0.5 in stead of 2.

    ... (Docstring truncated for brevity)

    Reference
    ---------
    1. Edwards, Hobbs and Manchester 2006 (2006MNRAS.372.1549E).
    """
    DoP = dict_of_orbital_parameters
    epoch *= u.d
    incl *= u.rad
    Om_asc *= u.deg
    e, T0, Pb0, omega0, a0, dec = DoP['ecc'], DoP['t0'],\
        DoP['pb'], DoP['om'], DoP['a1'], DoP['decj']
    try:
        omdot = DoP['omdot']
    except KeyError:
        omdot = 0 * u.deg/u.s
    try:
        Pbdot = DoP['pbdot']
    except KeyError:
        Pbdot = 0
    try:
        adot = DoP['a1dot']
    except KeyError:
        adot = None

    n = (2*np.pi/Pb0 - np.pi*Pbdot*(epoch-T0)/(Pb0**2)) * u.rad #angular velocity
    #n = 2*np.pi/Pb0 + np.pi*Pbdot*(epoch-T0)/(Pb0**2) #angular velocity
    u1 = solve_u(e, (n*(epoch-T0)).value) #u1 stands for u, not to clash with u=astropy.units
    #u1 *= u.rad
    A_u = u.rad * 2* np.arctan(((1+e)/(1-e))**0.5 * np.tan(u1/2))
    k = omdot/n
    omega = omega0 + k * A_u
    theta = omega + A_u
    if adot != None:
        a1 = a0 + adot * (epoch - T0) #equivalent to Eqn 71
    else:
        a1 = a0
    b_abs = a1 * (1 - e * np.cos(u1))
    b_AU = b_abs.to(u.AU).value
    offset = b_AU * px

    # M1: Rotation by Ascending Node Position Angle (Om_asc)
    matr1 = np.mat([[np.sin(Om_asc), -np.cos(Om_asc), 0],
                    [np.cos(Om_asc), np.sin(Om_asc), 0],
                    [0, 0, 1]])

    # M2: Rotation by Inclination (incl) -- CORRECTED SIGN
    matr2 = np.mat([[1, 0, 0],
                    [0, -np.cos(incl), np.sin(incl)], # Corrected: Changed -np.sin(incl) to +np.sin(incl)
                    [0, np.sin(incl), -np.cos(incl)]])

    # M3: Position vector in the orbital plane (scaled by offset)
    matr3 = np.mat([[offset*np.cos(theta)],
                    [offset*np.sin(theta)],
                    [0]])

    # Final transformed position vector (b_x = dRA*cos(dec), b_y = dDEC)
    b = matr1 * matr2 * matr3

    dRA = (b.item(0,0)/np.cos(dec)).value #that can be directly added to RA
    dDEC = b.item(1,0)

    return np.array([dRA, dDEC]) #in mas


class reflex_motion_detectability:
    """
    Purpose
    -------
    This class is used to 
        1) quantify VLBI measurability of reflex motions,
        2) roughly assess intrinsic a1dot (so that extrinsic a1dot can be more reliable).
    """
    def __init__(self):
        pass
    def calculate_eta_orb(self, a1, px, err_px, rcs):
        """
        Input parameter
        ---------------
        a1 : float
            projected semi-major axis (in lt-sec).
        px : float
            parallax in mas.
        err_px : float
            error of parallax in mas.
        rcs : float
            reduced chi-square.
        """
        a1 *= const.c * u.s
        a1_AU = a1.to(u.AU).value
        eta_orb = 2 * a1_AU * px / err_px / np.sqrt(rcs)
        return eta_orb
    def calculate_eta_orb_with_pmparin(self, pmparin, **kwargs):
        """
        Input parameter
        ---------------
        pmparin : str
            pmpar.in file that is used to derive px, err_px and rcs.
        kwargs :
            a1 : float
                projected semi-major axis (in lt-sec).
        """
        from sterne import priors
        if not os.path.exists(pmparin):
            print('%s does not exist; aborting' % pmparin)
            sys.exit(1)
        pmparout = pmparin.replace('pmpar.in','pmpar.out')
        os.system("pmpar %s > %s" % (pmparin, pmparout))
        [ra, error_ra, dec, error_dec, mu_a, error_mu_a, mu_d, error_mu_d, px, err_px, rcs, junk] = priors.readpmparout(pmparout)
        
        try:
            psrname = kwargs['psrname']
        except KeyError:
            psrname = ''

        try:
            a1 = kwargs['a1']
        except KeyError:
            print('a1 is not provided; fetching from PSRCAT')
            if psrname == '':
                psrname = pmparin.split('.')[0]
                print('Guess the pulsar name to be %s' % psrname)
            if psrname.startswith('J') or psrname.startswith('B'):
                query1 = QueryATNF(psrs=[psrname], params=['A1'])
                a1 = float(query1['A1'][0])
                print(a1)
            else:
                print('pulsar name is not clear; exiting')
                sys.exit()
        
        eta_orb = self.calculate_eta_orb(a1, px, err_px, rcs)
        return eta_orb


