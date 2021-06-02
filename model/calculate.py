#!/usr/bin/env python
import numpy as np
import astropy.units as u
import os, sys
import howfun
def position(refepoch, ra_rad, dec_rad, pmra, pmdec, px, epoch, useDE421=True, inputgeocentricposition=False):
    """
    Outputs position given reference position, proper motion and parallax 
    (at a reference epoch) and time of interest.

    Input parameters
    ----------------
    refepoch : float
        Reference epoch, in MJD.
    ra : str
        Right ascension for barycentric frame, in hh:mm:ss.ssss;
        Also see inputgeocentricposition.
    dec : str
        Declination for barycentric frame, in dd:mm:ss.sss;
        Also see inputgeocentricposition.
    pmra : float
        Proper motion in right ascension, in mas/yr.
    pmdec : float
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
    #ra_rad, dec_rad = dms2rad(ra, dec) 
    dT = (epoch - refepoch) * (u.d).to(u.yr) #in yr
    ### proper motion effect ###
    dRA = dT * pmra / np.cos(dec_rad) # in mas
    dDEC = dT * pmdec #in mas
    ### parallax effect ###
    dRA, dDEC = np.array([dRA, dDEC]) + parallax_related_position_offset_from_the_barycentric_frame(epoch, ra_rad, dec_rad, px, useDE421)  # in mas
    if inputgeocentricposition:
        dRA, dDEC = np.array([dRA, dDEC]) - parallax_related_position_offset_from_the_barycentric_frame(refepoch, ra_rad, dec_rad, px, useDE421)  # in mas
    ra_rad += dRA * (u.mas).to(u.rad)
    dec_rad += dDEC * (u.mas).to(u.rad)
    print(howfun.deg2dms(ra_rad*180/np.pi/15.), howfun.deg2dms(dec_rad*180/np.pi))
    return  ra_rad, dec_rad #rad

def plot_model_given_astrometric_parameters(refepoch, ra, dec, pmra, pmdec, px, start_epoch, end_epoch,\
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
        ra_rad, dec_rad = position(refepoch, ra, dec, pmra, pmdec, px, epoch, useDE421, inputgeocentricposition)
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

def dms2rad(ra, dec):
    """
    Input parameters
    ----------------
    ra : str
        Right ascension, in hh:mm:ss.sss.
    dec : str
        Declination, in dd:mm:ss.ssss.

    Return parameters
    -----------------
    ra : float
        Right ascension, in rad.
    dec : float
        Declination, in rad.
    """
    ra = howfun.dms2deg(ra)
    ra *= 15 * np.pi/180 #in rad
    dec = howfun.dms2deg(dec)
    dec *= np.pi/180 #in rad
    return ra, dec
    
def parallax_related_position_offset_from_the_barycentric_frame(epoch, ra, dec, px, useDE421=True):
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
    import novas.compat.solsys as solsys
    from novas.compat.eph_manager import ephem_open
    #ra, dec = dms2rad(ra, dec) 
    if not useDE421:
        ephem_open()
    else:
        ephem_open(os.path.join(os.getenv("TEMPO2"), "T2runtime/ephemeris/DE421.1950.2050"))
    # This is the Earth position in X, Y, Z (AU) in ICRS wrt SSB 
    X, Y, Z = solsys.solarsystem(epoch+2400000.5, 3, 0)[0]  
    #print(X,Y,Z)
    # Following is from Astronomical Almanac Explanatory Supplement p 125-126
    dRA = px * (X * np.sin(ra) - Y * np.cos(ra)) / np.cos(dec) #in mas
    dDEC = px * (X * np.cos(ra) * np.sin(dec) + Y * np.sin(ra) * np.sin(dec) - Z * np.cos(dec)) #in mas
    return np.array([dRA, dDEC]) #in mas
