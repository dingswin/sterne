#!/usr/bin/env python
"""
written in python3 by Hao Ding.
"""
import numpy as np
import astropy.units as u
from astropy import constants
import os, sys
import howfun

def generate_parfile(pulsar):
    """
    The parfile generation only works for pulsar listed in PSRCAT.
    """
    #os.system("psrcat %s -e > %s" % (pulsar, pulsar+'.par'))
    parfile = pulsar + '.par'
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
        a1 - projected semi-major axis of orbit (light-second);
        t0 - epoch of periastron (MJD);
        om - omega, longitude of periastron (deg);
        omdot - periastron advance (deg/yr);
        om_asc - position angle of ascending node (deg);
        pbdot - first time derivative of orbital period (s/s);
        a1dot - first time derivative of A1 (light-second/s);
        sini - sine of inclination angle;
    """
    readfile = open(parfile, 'r')
    lines = readfile.readlines()
    readfile.close()
    keywords_needed = ['DECJ', 'PB ', 'ECC', 'A1 ', 'T0', 'OM ', 'OMDOT', 'OM_ASC', 'PBDOT', 'A1DOT', 'SINI']
    parameters = [kw.strip().lower() for kw in keywords_needed]
    dict_parameter = {}
    for line in lines:
        for i in range(len(keywords_needed)):
            if keywords_needed[i] in line:
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
    dict_parameter['a1'] *= constants.c * u.s
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
        dict_parameter['a1dot'] *= constants.c
    except KeyError:
        pass
    try:
        dict_parameter['decj'] = u.deg * howfun.dms2deg(dict_parameter['decj'])
    except KeyError:
        pass
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
        in the Tempo2 paper (ref1).

    Caveats
    -------
    1. Time derivative of eccentricity is not taken into account.
    2. Two differently formulated A_u in Eqn 57 and Eqn 58 is considered the same.
    3. Relativistic deformations of the eccentricity, given by Eqn 59 and 60, is
        not taken into account.


    Input paramters
    ---------------
    epoch : float
        in MJD.
    dict_of_orbital_parameters : dict
        See the function read_parfile()
    incl : float
        Inclination angle (deg).
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
    incl *= u.deg
    Om_asc *= u.deg
    e, T0, Pb0, Pbdot, omega0, omdot, a0, dec = DoP['ecc'], DoP['t0'],\
        DoP['pb'], DoP['pbdot'], DoP['om'], DoP['omdot'], DoP['a1'], DoP['decj']
    try:
        adot = DoP['a1dot']
    except KeyError:
        adot = None

    n = (2*np.pi/Pb0 + np.pi*Pbdot*(epoch-T0)/(Pb0**2)) * u.rad #angular velocity
    #n = 2*np.pi/Pb0 + np.pi*Pbdot*(epoch-T0)/(Pb0**2) #angular velocity
    u1 = solve_u(e, (n*(epoch-T0)).value)[0] #u1 stands for u, not to clash with u=astropy.units
    #u1 *= u.rad
    A_u = u.rad * 2* np.arctan(((1+e)/(1-e))**2 * np.tan(u1/2))
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
    matr2 = np.mat([[1, 0, 0],
                    [0, -np.cos(incl), -np.sin(incl)],
                    [0, np.sin(incl), -np.cos(incl)]])
    matr3 = np.mat([[offset*np.cos(theta)],
                    [offset*np.sin(theta)],
                    [0]])
    b = matr1 * matr2 * matr3
    dRA = (b.item(0,0)/np.cos(dec)).value #that can be directly added to RA
    dDEC = b.item(1,0)
    return np.array([dRA, dDEC]) #in mas
