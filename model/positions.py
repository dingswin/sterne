#!/usr/bin/env python
"""
written in python3 by Hao Ding.
"""
import numpy as np
import astropy.units as u
from astropy import constants
import os, sys
from model import reflex_motion
import novas.compat.solsys as solsys
from novas.compat.eph_manager import ephem_open

def positions(refepoch, epochs, list_of_dict_timing, parameter_filter_index, dict_parameters):
    FP = filter_dictionary_of_parameter_with_index(dict_parameters, parameter_filter_index)
    Ps = list(FP.keys())
    Ps.sort()
    ra_models = np.array([])
    dec_models = np.array([])
    for i in range(len(epochs)):
        ra_model, dec_model = position(refepoch, epochs[i], FP[Ps[0]],\
            FP[Ps[1]], FP[Ps[2]], FP[Ps[3]], FP[Ps[4]], FP[Ps[5]], FP[Ps[6]], list_of_dict_timing)
        ra_models = np.append(ra_models, ra_model)
        dec_models = np.append(dec_models, dec_model)
    return np.concatenate([ra_models, dec_models])

def filter_dictionary_of_parameter_with_index(dict_of_parameters, filter_index):
    """
    Input parameters
    ----------------
    dict_of_parameters : dict
        Dictionary of astrometric parameters.
    filter_index : int
        A number (for the pmparin file) used to choose the right paramters for each gaussian distribution.

    Return parameters
    -----------------
    filtered_dict_of_parameters : dict
        Dictionary after the filtering with the filter index.
    """
    filtered_dict_of_parameters = dict_of_parameters.copy()
    for parameter in dict_of_parameters.keys():
        if not str(filter_index) in parameter.split('_'):
            del filtered_dict_of_parameters[parameter]
    filtered_dict_of_parameters = auto_fill_disabled_parameters(filtered_dict_of_parameters)
    return filtered_dict_of_parameters
def auto_fill_disabled_parameters(filtered_dict_of_parameters):
    roots = parameter_roots = ['dec', 'incl', 'mu_a', 'mu_d', 'om_asc', 'px', 'ra']
    for root in roots:
        if not any(root in parameter for parameter in filtered_dict_of_parameters.keys()):
            filtered_dict_of_parameters[root] = -999
    return filtered_dict_of_parameters

def position(refepoch, epoch, dec_rad, incl, mu_a, mu_d, om_asc, px, ra_rad,\
        dict_of_timing_parameters={}):
    """
    Outputs position given reference position, proper motion and parallax 
    (at a reference epoch) and time of interest.

    Input parameters
    ----------------
    refepoch : float
        Reference epoch, in MJD.
    ra_rad : rad
        Right ascension for barycentric frame, in hh:mm:ss.ssss;
        Also see inputgeocentricposition.
    dec_rad : rad
        Declination for barycentric frame, in dd:mm:ss.sss;
        Also see inputgeocentricposition.
    mu_a : float
        Proper motion in right ascension, in mas/yr.
    mu_d : float
        Proper motion in declination, in mas/yr.
    px : float 
        Parallax, in mas.
    epoch : float
        Time of interest, in MJD.
    useDE421 : bool
        Use DE421 if True (default), use DE405 if False.
    inputgeocentricposition : bool
        Use input ra and dec for barycentric frame if False (default);
        Use input ra and dec for geocentric frame if True.

    Return parameters
    -----------------
    ra_rad : float
        Right ascension for geocentric frame, in rad.
    dec_rad : float
        Declination for geocentric frame, in rad.

    Notes
    -----
    """
    dT = (epoch - refepoch) * (u.d).to(u.yr) #in yr
    ### proper motion effect ###
    dRA = dT * mu_a / np.cos(dec_rad) # in mas
    dDEC = dT * mu_d #in mas
    ### parallax effect ###
    #dRA, dDEC = np.array([dRA, dDEC])
    offset = np.array([dRA, dDEC]) #in mas
    if px != -999:
        offset += parallax_related_position_offset_from_the_barycentric_frame(epoch, ra_rad, dec_rad, px) #in mas
        if (incl != -999) and (om_asc != -999) and (dict_of_timing_parameters != {}):
            offset += reflex_motion.reflex_motion(epoch, dict_of_timing_parameters, incl, om_asc, px)
    ra_rad += offset[0] * (u.mas).to(u.rad)
    dec_rad += offset[1] * (u.mas).to(u.rad)
    return  ra_rad, dec_rad #rad
def parallax_related_position_offset_from_the_barycentric_frame(epoch, ra, dec, px):
    """
    Originality note
    ----------------
    This function is adopted from astrometryfit written by Adam Deller and Scott Ransom.

    Return parameters
    -----------------
    np.array([dRA, dDEC])
        dRA : float
            Right asension offset, in mas.
        dDEC : float
            Declination offset, in mas.

    References
    ---------
    1. Explanatory Supplement to Astronomical Almanac.
    2. NOVAS
    """
    #if not useDE421:
    #    ephem_open()
    #else:
    ephem_open(os.path.join(os.getenv("TEMPO2"), "T2runtime/ephemeris/DE421.1950.2050"))
    # This is the Earth position in X, Y, Z (AU) in ICRS wrt SSB 
    X, Y, Z = solsys.solarsystem(epoch+2400000.5, 3, 0)[0]  
    #print(X,Y,Z)
    # Following is from Astronomical Almanac Explanatory Supplement p 125-126
    dRA = px * (X * np.sin(ra) - Y * np.cos(ra)) / np.cos(dec) #in mas
    dDEC = px * (X * np.cos(ra) * np.sin(dec) + Y * np.sin(ra) * np.sin(dec) - Z * np.cos(dec)) #in mas
    return np.array([dRA, dDEC]) #in mas

def plot_model_given_astrometric_parameters(refepoch, ra, dec, mu_a, mu_d, px, start_epoch, end_epoch,\
        useDE421=True, inputgeocentricposition=False):
    """
    Input parameters
    ----------------
    start_epoch : float
        Epoch when the plot starts, in MJD.
    end_epoch : float
        Epoch when the plot ends, in MJD.
    """
    import matplotlib.pyplot as plt
    epoch_step = (end_epoch - start_epoch) / 100.
    epochs = np.arange(start_epoch, end_epoch+epoch_step, epoch_step)
    #print(epochs)
    ra_rads, dec_rads = np.array([]), np.array([])
    dRA_pxs, dDEC_pxs = np.array([]), np.array([])
    for epoch in epochs:
        ra_rad, dec_rad = position(refepoch, epoch, dec, mu_a, mu_d, px, ra, useDE421, inputgeocentricposition)
        dRA_px, dDEC_px = parallax_related_position_offset_from_the_barycentric_frame(epoch, ra, dec, px)
        ra_rads = np.append(ra_rads, ra_rad)
        dec_rads = np.append(dec_rads, dec_rad)
        dRA_pxs = np.append(dRA_pxs, dRA_px)
        dDEC_pxs = np.append(dDEC_pxs, dDEC_px)
    #print(ra_rads, dec_rads)
    fig = plt.figure()
    ax1 = fig.add_subplot(221)
    ax1.set_xlabel('RA. (rad)')
    ax1.set_ylabel('Decl. (rad)')
    ax1.plot(ra_rads, dec_rads)
    
    ax2 = fig.add_subplot(222)
    ax2.set_ylabel('Decl. offset (mas)')
    ax2.set_xlabel('RA. offset (mas)')
    ax2.plot(dRA_pxs, dDEC_pxs)

    ax3 = fig.add_subplot(223)
    ax3.set_ylabel('RA. offset (mas)')
    ax3.set_xlabel('epoch (MJD)')
    ax3.plot(epochs, dRA_pxs)
    
    ax4 = fig.add_subplot(224)
    ax4.set_ylabel('Decl. offset (mas)')
    ax4.set_xlabel('epoch (MJD)')
    ax4.plot(epochs, dDEC_pxs)
    
    fig.tight_layout() #always put this before savefig
    directory_to_save = 'sterne_figures/'
    if not os.path.exists(directory_to_save):
        os.mkdir(directory_to_save)
    plt.savefig(directory_to_save + '/astrometric_model.pdf')

