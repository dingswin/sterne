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

def positions(refepoch, epochs, dict_timing, parameter_filter_index, dict_parameters):
    """
    Input parameters
    ----------------
    parameter_filter_index : int
        A number (for the pmparin file) used to choose the right paramters for each gaussian distribution.
    """
    FP = filter_dictionary_of_parameter_with_index(dict_parameters, parameter_filter_index)
    Ps = list(FP.keys())
    Ps.sort()
    ra_models = np.array([])
    dec_models = np.array([])
    for i in range(len(epochs)):
        ra_model, dec_model = position(refepoch, epochs[i], FP[Ps[0]],\
            FP[Ps[2]], FP[Ps[3]], FP[Ps[4]], FP[Ps[5]], FP[Ps[6]], FP[Ps[7]], dict_timing)
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
    """
    so that each parameter can be pinpointed with a number
    """
    roots = parameter_roots = ['dec', 'efac', 'incl', 'mu_a', 'mu_d', 'om_asc', 'px', 'ra']
    for root in roots:
        if not any(root in parameter for parameter in filtered_dict_of_parameters.keys()):
            filtered_dict_of_parameters[root] = -999
    return filtered_dict_of_parameters

def position(refepoch, epoch, dec_rad, incl, mu_a, mu_d, om_asc, px, ra_rad,\
        dict_of_timing_parameters={}, **kwargs):
    """
    Outputs position given reference position, proper motion and parallax 
    (at a reference epoch) and time of interest.

    Input parameters
    ----------------
    refepoch : float
        Reference epoch for ra_rad and dec_rad, in MJD.
    ra_rad : float
        Right ascension for barycentric frame, in rad;
        Also see inputgeocentricposition.
    dec_rad : float
        Declination for barycentric frame, in rad;
        Also see inputgeocentricposition.
    mu_a : float
        Proper motion in right ascension, in mas/yr.
    mu_d : float
        Proper motion in declination, in mas/yr.
    px : float 
        Parallax, in mas.
    epoch : float
        Time when a position is to be calculated, in MJD.
    dict_of_timing_parameters : dict (default : {})
        In case reflex motion needs to be calculated, dict_of_timing_parameters that
        records orbital parameters obtained with timing should be provided.

    kwargs : dict
        Both currently not in use!
        1. useDE421 : bool
            Use DE421 if True (default), use DE405 if False.
        2. inputgeocentricposition : bool
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

def model_parallax_and_reflex_motion_offset(epoch, dict_parameters, dict_of_timing_parameters, no_px):
    """
    Notice
    ------
        This function is meant for plot.sky_position_evolution.parallax_signature. 
        There is at most 1 'px', 1 'incl' and 1 'om_asc' parameter.
        In case there are more than one 'ra's and 'dec's, all 'ra's and 'dec's are assumed to be very close on the sky.
    """
    px = None
    om_asc = None
    incl = None
    if type(dict_parameters) == dict:
        parameters = dict_parameters.keys()
    else: ## Table
        parameters = dict_parameters.colnames
    for parameter in parameters: ## for Table format
        if 'px' in parameter:
            px = dict_parameters[parameter]
        if 'om_asc' in parameter:
            om_asc = dict_parameters[parameter]
        if 'incl' in parameter:
            incl = dict_parameters[parameter]
        if 'ra' in parameter:
            ra_rad = dict_parameters[parameter] ## use the last ra_rad
        if 'dec' in parameter:
            dec_rad = dict_parameters[parameter] ## use the last dec_rad
    ra_offset, dec_offset = 0, 0
    if not no_px:
        ra_offset += np.cos(dec_rad) * parallax_related_position_offset_from_the_barycentric_frame(epoch, ra_rad, dec_rad, px)[0] #in mas; since it is on the sky plane, cos(dec_rad) is multiplied back
        dec_offset = parallax_related_position_offset_from_the_barycentric_frame(epoch, ra_rad, dec_rad, px)[1] # in mas
    if (om_asc != None) and (incl != None):
        ra_offset += np.cos(dec_rad) * reflex_motion.reflex_motion(epoch, dict_of_timing_parameters, incl, om_asc, px)[0]
        dec_offset += reflex_motion.reflex_motion(epoch, dict_of_timing_parameters, incl, om_asc, px)[1]
    return ra_offset, dec_offset
def model_parallax_and_reflex_motion_offsets(epochs, dict_parameters, dict_of_timing_parameters, **kwargs):
    """
    Inputs
    ------
    kwargs :
        no_px : boolean (default : False)
            If set to True, parallax effect would not be plotted.
    """
    ## >>> no_px
    try:
        no_px = kwargs['no_px']
    except KeyError:
        no_px = False
    ## <<<
    
    ra_offsets, dec_offsets = [], []
    for epoch in epochs:
        offset = model_parallax_and_reflex_motion_offset(epoch, dict_parameters, dict_of_timing_parameters, no_px)
        ra_offsets.append(offset[0])
        dec_offsets.append(offset[1])
    return np.concatenate((ra_offsets, dec_offsets))

def observed_positions_subtracted_by_proper_motion(refepoch, dict_VLBI, filter_index, dict_parameters, **kwargs):
    """
    kwargs :
        no_px : boolean (default : False)
            If set to True, parallax effects will be deducted as well.
    """
    try:
        no_px = kwargs['no_px']
    except KeyError:
        no_px = False
    
    epochs = dict_VLBI['epochs']
    NoE = len(epochs)
    ra_offsets, dec_offsets = [], []
    for i in range(NoE):
        ra = dict_VLBI['radecs'][i]
        dec = dict_VLBI['radecs'][i+NoE]
        ra_offset, dec_offset = observed_position_subtracted_by_proper_motion(refepoch, epochs[i], ra, dec, filter_index, dict_parameters, no_px)
        ra_offsets.append(ra_offset)
        dec_offsets.append(dec_offset)

    errs = dict_VLBI['errs']
    errs *= (u.rad).to(u.mas)
    ra_errs = errs[:NoE] * np.cos(dec)
    dec_errs = errs[NoE:]
    return np.concatenate((ra_offsets, dec_offsets)), np.concatenate((ra_errs, dec_errs))

def observed_position_subtracted_by_proper_motion(refepoch, epoch, ra, dec, filter_index, dict_parameters, no_px):

    filtered_dict_of_parameters = filter_dictionary_of_parameter_with_index(dict_parameters, filter_index)
    for parameter in filtered_dict_of_parameters:
        if 'mu_a' in parameter:
            mu_a = filtered_dict_of_parameters[parameter]
        if 'mu_d' in parameter:
            mu_d = filtered_dict_of_parameters[parameter]
        if 'ra' in parameter:
            ra_ref = filtered_dict_of_parameters[parameter] ## rad
        if 'dec' in parameter:
            dec_ref = filtered_dict_of_parameters[parameter] ## rad
        if 'px' in parameter:
            if no_px:
                px = filtered_dict_of_parameters[parameter] ## mas
            else:
                px = 0
            
    ra_offset = (ra - ra_ref) * (u.rad).to(u.mas) * np.cos(dec_ref) ## in mas
    dec_offset = (dec - dec_ref) * (u.rad).to(u.mas) ## in mas
    ra_offset -= mu_a * (epoch - refepoch) * (u.d).to(u.yr)
    dec_offset -= mu_d * (epoch - refepoch) * (u.d).to(u.yr)
    #offset = np.array([ra_offset, dec_offset])
    ra_offset -= np.cos(dec_ref) * parallax_related_position_offset_from_the_barycentric_frame(epoch, ra, dec, px)[0] #in mas; back to sky plane
    dec_offset -= parallax_related_position_offset_from_the_barycentric_frame(epoch, ra, dec, px)[1]
    return ra_offset, dec_offset

def simulate_positions_subtracted_by_proper_motion(refepoch, epochs, sim_table_row, filter_index, dict_parameters, dict_timing, **kwargs):
    """
    Inputs
    ------
    kwargs : 
        no_px : boolean (default : False)
            If set to True, parallax effect will be deducted.
    """
    try:
        no_px = kwargs['no_px']
    except KeyError:
        no_px = False

    NoE = len(epochs)
    dict_1sim = {}
    for parameter in sim_table_row.colnames[:-2]: ## excluding 'log_likelihood' and 'log_prior'
        dict_1sim[parameter] = sim_table_row[parameter]

    sim_radecs = positions(refepoch, epochs, dict_timing, filter_index, dict_1sim)
    ra_offsets, dec_offsets = [], []
    for i in range(NoE):
        ra_offset, dec_offset = observed_position_subtracted_by_proper_motion(refepoch, epochs[i], sim_radecs[i], sim_radecs[NoE+i], filter_index, dict_parameters, no_px=no_px)
        ra_offsets.append(ra_offset)
        dec_offsets.append(dec_offset)
    return np.concatenate((ra_offsets, dec_offsets))

def parallax_related_position_offset_from_the_barycentric_frame(epoch, ra, dec, px):
    """
    Originality note
    ----------------
    This function is adopted from astrometryfit written by Adam Deller and Scott Ransom.
    
    Input parameters
    ----------------
    epoch : float
        in MJD.
    ra : float
        in rad.
    dec : float
        in rad.
    px : float
        in mas.

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
    tempo2_dir = os.getenv("TEMPO2")
    if tempo2_dir == None:
        print('\nTEMPO2 not installed or its environment variable unset; aborting...') ## it does not need to be installed, but the T2runtime folder is needed.
        sys.exit(1)
    ephem_open(os.path.join(os.getenv("TEMPO2"), "T2runtime/ephemeris/DE421.1950.2050"))
    # This is the Earth position in X, Y, Z (AU) in ICRS wrt SSB 
    X, Y, Z = solsys.solarsystem(epoch+2400000.5, 3, 0)[0]  
    #print(X,Y,Z)
    # Following is from Astronomical Almanac Explanatory Supplement p 125-126
    dRA = px * (X * np.sin(ra) - Y * np.cos(ra)) / np.cos(dec) #in mas; here, np.cos(dec) has been divided, to be added to ra_rad
    dDEC = px * (X * np.cos(ra) * np.sin(dec) + Y * np.sin(ra) * np.sin(dec) - Z * np.cos(dec)) #in mas
    return np.array([dRA, dDEC]) #in mas

def plot_model_given_astrometric_parameters(refepoch, ra, dec, mu_a, mu_d, px, start_epoch, end_epoch,\
        useDE421=True, inputgeocentricposition=False):
    """
    Deprecated temporarily!!

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

