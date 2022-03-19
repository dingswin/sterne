#!/usr/bin/env python
"""
written in python3 by Hao Ding in March 2022.
"""
import numpy as np
import astropy.units as u
from astropy import constants
from sterne.model import positions as _positions
import sys

def calculate_kopeikin_om_asc(a1, a1dot, incl, mu_a, mu_d, om_asc_siding_east=True):
    """
    Convention
    ----------
    Tempo2.
    
    Formalism
    ---------
    a1dot/a1 = mu*cot(incl)*sin(theta_mu-om_asc)

    Input parameters
    ----------------
    a1 : float
        a1 = a * sin(inc), where a is orbital semi-major axis (in lt-sec).
    inc : float
        orbital inclination angle (in rad).
    mu_a : float
        proper motion in RA (mas/yr).
    mu_d : float
        proper mtoion in Dec (mas/yr).
    a1dot : float
        in 1e0 lt-sec/sec. 
        a1dot due to inclination variation caused by proper motion.
        Here, a1dot denotes time derivative of a1.
    om_asc_siding_east : boolean (default : True)
        The om_asc sides east or not on the sky. 
        It is a hassel that each pair of a1dot and incl corresponds to two om_asc.

    Output parameter
    ----------------
    om_asc : float
        orbital ascending node longitude (in deg).
    """
    om_asc_siding_east = 2 * int(om_asc_siding_east) - 1 ## convert True, False to 1, -1.
    a1 *= u.m
    a1dot *= u.m/u.s
    mu_a *= u.mas/u.yr
    mu_d *= u.mas/u.yr
    mu = (mu_a**2 + mu_d**2)**0.5
    theta_mu = np.arctan2(mu_a, mu_d)
    sin_om1 = (a1dot / a1 / mu * np.tan(incl)).to(1/u.rad)
    print(sin_om1)
    om1 = np.arcsin(sin_om1.value)
    om_asc = theta_mu.value - om1
    if np.cos(om_asc) * om_asc_siding_east < 0:
        om_asc = np.pi - om_asc
    om_asc *= 180./np.pi
    return om_asc

def __calculate_a1dot_pm(a1, inc, mu_a, mu_d, om_asc):
    """
    Convention
    ----------
    Tempo2.

    Input parameters
    ----------------
    a1 : float
        a1 = a * sin(inc), where a is orbital semi-major axis (in lt-sec).
    inc : float
        orbital inclination angle (in rad).
    mu_a : float
        proper motion in RA (mas/yr).
    mu_d : float
        proper mtoion in Dec (mas/yr).
    om_asc : float
        orbital ascending node longitude (in deg).
    
    Output parameter
    ----------------
    a1dot_pm : float
        in 1e0 lt-sec/sec. 
        a1dot due to inclination variation caused by proper motion.
        Here, a1dot denotes time derivative of a1.
    """
    inc *= u.rad
    om_asc *= u.deg
    mu_a *= u.mas/u.yr
    mu_d *= u.mas/u.yr
    mu = (mu_a**2 + mu_d**2)**0.5
    theta_mu = np.arctan2(mu_a, mu_d) ## position angle of mu (east of north)
    a1dot_pm = mu * np.sin(theta_mu - om_asc) / np.tan(inc)
    a1dot_pm *= a1
    return 1e0 * a1dot_pm.to(u.rad/u.s).value ## in 1e0 lt-sec/sec

def __calculate_res_a1dot(a1, a1dot, inc, mu_a, mu_d, om_asc):
    """
    res_a1dot = a1dot - a1dot_pm

    Input parameters
    ----------------
    a1dot : float
        in 1e0 lt-sec/sec.
    
    Output parameter
    ----------------
    res_a1dot : float
        in 1e0 lt-sec/sec.
    """
    a1dot_pm = calculate_a1dot_pm(a1, inc, mu_a, mu_d, om_asc)
    res_a1dot = a1dot - a1dot_pm 
    return res_a1dot
def __calculate_equivalent_total_res_a1dot(list_of_dict_timing, dict_parameters):
    """
    Output parameter
    ----------------
    equivalent_total_res_a1dot : float
        in 1e0 lt-sec/sec.
        It would be 0, if list_of_res_a1dot == np.array([]).
    """
    LoD_timing = list_of_dict_timing 
    list_of_res_a1dot = np.array([]) 
    for i in range(len(LoD_timing)):
        if LoD_timing[i] != {}:
            FP = _positions.filter_dictionary_of_parameter_with_index(dict_parameters, i)
            Ps = list(FP.keys())
            Ps.sort()
            a1 = LoD_timing[i]['a1'].value
            res_a1dot = calculate_res_a1dot(a1, FP[Ps[0]],\
                FP[Ps[2]], FP[Ps[3]], FP[Ps[4]], FP[Ps[5]])
            list_of_res_a1dot = np.append(list_of_res_a1dot, res_a1dot)
    if len(list_of_res_a1dot) == 0:
        print("The list_of_res_a1dot is empty. Make sure not all parfiles are ''.")
        sys.exit(1)
    sum_res_a1dot = np.sum(list_of_res_a1dot**2)
    equivalent_total_res_a1dot = sum_res_a1dot**0.5
    return equivalent_total_res_a1dot
