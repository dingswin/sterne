#!/usr/bin/env python
"""
written in python3 by Hao Ding in March 2022.
"""
import numpy as np
import astropy.units as u
from astropy import constants
from sterne.model import positions as _positions
import sys

def a1dot_pm_formalisim(a1, inc, mu_a, mu_d, om_asc):
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

def calculate_a1dot_pm(list_of_dict_timing, parameters_dict):
    """
    Output parameter
    ----------------
    list_of_modeled_a1dot : list of floats
        in lt-sec/sec.
    """
    LoD_timing = list_of_dict_timing 
    list_of_modeled_a1dot = np.array([])
    for i in range(len(LoD_timing)):
        if LoD_timing[i] != {}:
            FP = _positions.filter_dictionary_of_parameter_with_index(parameters_dict, i)
            Ps = list(FP.keys())
            Ps.sort()
            a1 = (LoD_timing[i]['a1'] / constants.c / u.s).value ## convert to lt-sec
            modeled_a1dot = a1dot_pm_formalisim(a1, FP[Ps[3]], FP[Ps[4]], FP[Ps[5]], FP[Ps[6]])
            list_of_modeled_a1dot = np.append(list_of_modeled_a1dot, modeled_a1dot)

    if len(list_of_modeled_a1dot) == 0:
        print("The list_of_modeled_a1dot is empty. Make sure not all parfiles are ''.")
        sys.exit(1)
    return list_of_modeled_a1dot

