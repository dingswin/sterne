#!/usr/bin/env python
"""
Used to make plots of the publication quality;
"""
import os, sys
import astropy.units as u
import numpy as np
from matplotlib import pyplot as plt
from astropy.table import Table
from sterne import simulate
from sterne.model import positions

def parallax_signature(pmparins, parfiles, refepoch, posterior_samples='outdir/posterior_samples.dat', **kwargs):
    import matplotlib.gridspec as gridspec
    """
    Purpose
    -------
        to plot parallax (and reflex motion) signature revealed/shared by one or more pmparin(s).
    Notice
    ------
        1. only one 'px' parameter (and 'incl', 'om_asc') allowed in the parameter dictionary.
        2. pmparins and parfiles should follow the order of parameters provided by the posterior_samples.
        3. pmparins should share the same epochs.
        4. Only one parfile is needed. But to be safe try to make len(parfiles)==len(pmparins).

    Input parameters
    ----------------
    refepoch : float
        reference epoch (in MJD) for the 'ra' and 'dec' reference positions provided by the posterior_samples. 

    kwargs :
        1. time_resolution : int
            The larger the higher the time resolution.
    """
    try:
        time_resolution = kwargs['time_resolution']
    except KeyError:
        time_resolution = 100
    
    NoP = len(pmparins)
    LoD_VLBI = list_of_dict_VLBI = simulate.create_list_of_dict_VLBI(pmparins)
    LoD_timing = list_of_dict_timing = simulate.create_list_of_dict_timing(parfiles)
    dict_timing = LoD_timing[0] ## see the Notice

    t = Table.read(posterior_samples, format='ascii')
    parameters = t.colnames[:-2]
    dict_median, outputfile = simulate.make_a_brief_summary_of_Bayesian_inference(posterior_samples)
    
    epochs = LoD_VLBI[0]['epochs']
    NoE = len(epochs)
    min_epoch, max_epoch = min(epochs), max(epochs)
    Ts = np.arange(min_epoch, max_epoch, (max_epoch-min_epoch)/time_resolution)
    #model_radecs = positions(refepoch, Ts, LoD_timing, 0, dict_median)
    model_radec_offsets = positions.model_parallax_and_reflex_motion_offsets(Ts, dict_median, dict_timing)
    
    #################################
    ### plot the model first
    #################################
    fig1 = plt.figure()
    gs1 = gridspec.GridSpec(1, 2)
    
    #ax1 = fig1.add_subplot(121)
    ax1 = fig1.add_subplot(gs1[0])
    #ax2 = fig1.add_subplot(122)
    ax1.plot(Ts, model_radec_offsets[:len(Ts)])
    ax1.set_xlabel('time (MJD)')
    ax1.set_ylabel('RA. offset (mas)')
    ax1.set_title('RA-time (proper motion removed)')

    ax2 = fig1.add_subplot(gs1[1])
    ax2.plot(Ts, model_radec_offsets[len(Ts):])
    ax2.set_xlabel('time (MJD)')
    ax2.set_ylabel('Dec offset (mas)')
    ax2.set_title('Dec-time (proper motion removed)')
    #######################################
    ### plot the observed positions now
    #######################################
    for i in range(NoP):
        radec_offsets, errs = positions.observed_positions_subtracted_by_proper_motion(refepoch, LoD_VLBI[i], i, dict_median)
        ax1.scatter(epochs, radec_offsets[:NoE], marker='.')
        ax1.errorbar(epochs, radec_offsets[:NoE], yerr=errs[:NoE], fmt='.', markersize=5, capsize=3)
        ax2.scatter(epochs, radec_offsets[NoE:], marker='.')
        ax2.errorbar(epochs, radec_offsets[NoE:], yerr=errs[NoE:], fmt='.', markersize=5, capsize=3)
    
    gs1.tight_layout(fig1)
    plt.savefig('ra_dec_time_nopm.pdf')
    plt.clf()
