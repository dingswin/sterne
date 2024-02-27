#!/usr/bin/env python
"""
Used to make plots of the publication quality;
"""
import os, sys
import astropy.units as u
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from astropy.table import Table
from sterne import simulate
from sterne.model import positions

def parallax_signature(pmparins, parfiles, refepoch, posterior_samples='outdir/posterior_samples.dat', **kwargs):
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

    Example
    -------
    1) sky_position_evolution.parallax_signature(['J1939+2134.to.IBC01647.pmpar.in.dual.phscal','J1939+2134.to.IBC01648.pmpar.in.dual.phscal'], ['',''], 57850, legend_labels=['J194104','J194106'], N_random_draw=1000)
    should reproduce the figure in Ding et al. 2023 (but with the plot branch).
    2) parallax_signature(['J0509+3801.to.IBC18600068.pmpar.in','J0509+3801.to.IBC18600047.sec.dual.phscal.pmpar.in'], ['',''], 57381, legend_labels=['J051132','virtual calibrator'], N_random_draw=1000, legend_loc='upper right')
    should reproduce the figure in Ding et al. 2024 (DNS paper)

    Input parameters
    ----------------
    refepoch : float
        reference epoch (in MJD) for the 'ra' and 'dec' reference positions provided by the posterior_samples. 

    kwargs :
        1. time_resolution : int (default : 100)
            The larger the higher the time resolution.
        2. N_random_draw : int (default : -999)
            Number of random draw from the posterior sample.
            It is activated when N_random_draw > 5
        3. legend_labels : list of str
        4. colors : list of str
        5. legend_loc : str (default : 'lower left')
            where on the left panel to place the legend. There are 6 options: 'lower/upper left/center/right'.
        6. dim_random_draw_by : float (default : 1)
            reduce the transparency of random draw by a factor of dim_random_draw_by.
    """
    #########################
    ## set up variables
    #########################
    try:
        time_resolution = kwargs['time_resolution']
    except KeyError:
        time_resolution = 100
    try:
        N_random_draw = kwargs['N_random_draw']
    except KeyError:
        N_random_draw = -999
    
    NoP = len(pmparins)
    
    try:
        legend_labels = kwargs['legend_labels']
    except KeyError:
        if NoP >= 1:
            legend_labels = [''] * NoP
    try:
        colors = kwargs['colors']
    except KeyError:
        colors = ['mediumblue','tomato','r', 'y', 'lime']
    try:
        legend_loc = kwargs['legend_loc']
    except KeyError:
        legend_loc = 'lower left'
    try:
        dim_random_draw_by = kwargs['dim_random_draw_by']
    except KeyError:
        dim_random_draw_by = 2.0
    try:
        pmparin_preliminaries = kwargs['pmparin_preliminaries']
        if len(pmparin_preliminaries) != NoP:
            print('The number of pmpar.in.preliminary files has to\
                match that of pmpar.in files. Exiting for now.')
            sys.exit(1)
    except KeyError:
        pmparin_preliminaries = None
    
    LoD_VLBI = list_of_dict_VLBI = simulate.create_list_of_dict_VLBI(pmparins, pmparin_preliminaries)
    LoD_timing = list_of_dict_timing = simulate.create_list_of_dict_timing(parfiles)
    dict_timing = LoD_timing[0] ## see the Notice
    
    print('Reading %s...' % posterior_samples)
    t = Table.read(posterior_samples, format='ascii')
    print('Done reading %s...' % posterior_samples)
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
    
    ax1 = fig1.add_subplot(gs1[0])
    ax2 = fig1.add_subplot(gs1[1])
    
    ## >>> if show Bayesian parameter errors
    if N_random_draw > 5:
        for j in range(NoP):
            posterior_indice = np.random.randint(0, len(t), N_random_draw) 
            for index in posterior_indice:
                sim_radec_offsets = positions.simulate_positions_subtracted_by_proper_motion(refepoch, Ts, t[index], j, dict_median, dict_timing)
                #ax1.plot(Ts, sim_radec_offsets[:len(Ts)], color=colors[j], alpha=15./N_random_draw)
                ax1.plot(Ts, sim_radec_offsets[:len(Ts)], color=colors[j], alpha=15./N_random_draw/dim_random_draw_by)
                #ax2.plot(Ts, sim_radec_offsets[len(Ts):], color=colors[j], alpha=15./N_random_draw)
                ax2.plot(Ts, sim_radec_offsets[len(Ts):], color=colors[j], alpha=15./N_random_draw/dim_random_draw_by)
    ## <<<
    
    #model_linewidth = 0.7
    model_linewidth = 0.8
    ax1.plot(Ts, model_radec_offsets[:len(Ts)], color='magenta', lw=model_linewidth)
    ax1.set_xlabel('time (MJD)')
    ax1.set_ylabel('RA. offset (mas)')
    #ax1.set_title('RA-time (proper motion removed)')

    ax2.plot(Ts, model_radec_offsets[len(Ts):], color='magenta', lw=model_linewidth)
    ax2.set_xlabel('time (MJD)')
    ax2.set_ylabel('Dec offset (mas)')
    #ax2.set_title('Dec-time (proper motion removed)')




    #######################################
    ### plot the observed positions now
    #######################################

    for i in range(NoP):
        trs = transparency = errorbar_transparency(i, -0.5)
        radec_offsets, _ = positions.observed_positions_subtracted_by_proper_motion(refepoch, LoD_VLBI[i], i, dict_median)
        errs_new = simulate.adjust_errs_with_efac(LoD_VLBI[i], dict_median, i) ## apply efac (and efad) if requested
        dec = LoD_VLBI[i]['radecs'][-1]
        errs_new = positions.convert_radec_errs_rad2mas(errs_new, dec)
        ax1.scatter(epochs, radec_offsets[:NoE], marker='.', alpha=trs, color=colors[i])
        ax1.errorbar(epochs, radec_offsets[:NoE], yerr=errs_new[:NoE], fmt='.', markersize=5, capsize=3, alpha=trs, label=legend_labels[i], color=colors[i])
        ax2.scatter(epochs, radec_offsets[NoE:], marker='.', alpha=trs, color=colors[i])
        ax2.errorbar(epochs, radec_offsets[NoE:], yerr=errs_new[NoE:], fmt='.', markersize=5, capsize=3, alpha=trs, color=colors[i])

    ax1.legend(loc=legend_loc)
    gs1.tight_layout(fig1)
    plt.savefig('ra_dec_time_nopm_Bayesian.pdf')
    plt.clf()

def errorbar_transparency(layer, power_index):
    """
    Input parameter
    ---------------
    layer : int
        layer NO.
    power_index : float
        a negative number.
    
    Output
    ------
    transp : float
        value for the 'alpha' kwarg.
    """
    transp = (1 + layer)**power_index
    return transp

def reflex_motion_signature(pmparins, parfiles, refepoch, posterior_samples='outdir/posterior_samples.dat', **kwargs):
    """
    Purpose
    -------
        to plot reflex motion signature revealed/shared by one or more pmparin(s).

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
        1. phase_resolution : int (default : 0.01)
            A fraction of the orbital phase. The smaller the higher the time resolution.
        2. N_random_draw : int (default : -999)
            Number of random draw from the posterior sample.
            It is activated when N_random_draw > 5
        3. legend_labels : list of str
        4. colors : list of str
        5. legend_loc : str (default : 'lower left')
            where on the left panel to place the legend. There are 6 options: 'lower/upper left/center/right'.
        6. pmparin_preliminaries : list of str (default : None)
            A list of pmpar.in.preliminary files that record random errors. It is required when efac is requested,
            unless only using the errors in pmpar.in files for errorbars.
    """
    #########################
    ## set up variables
    #########################
    try:
        phase_resolution = kwargs['phase_resolution']
    except KeyError:
        phase_resolution = 0.01
    try:
        N_random_draw = kwargs['N_random_draw']
    except KeyError:
        N_random_draw = -999
    
    NoP = len(pmparins)
    
    try:
        legend_labels = kwargs['legend_labels']
    except KeyError:
        if NoP >= 1:
            legend_labels = [''] * NoP
    try:
        colors = kwargs['colors']
    except KeyError:
        colors = ['mediumblue','tomato','r', 'y', 'lime']
    try:
        legend_loc = kwargs['legend_loc']
    except KeyError:
        legend_loc = 'lower left'
    
    try:
        pmparin_preliminaries = kwargs['pmparin_preliminaries']
        if len(pmparin_preliminaries) != NoP:
            print('The number of pmpar.in.preliminary files has to\
                match that of pmpar.in files. Exiting for now.')
            sys.exit(1)
    except KeyError:
        pmparin_preliminaries = None
    
    LoD_VLBI = list_of_dict_VLBI = simulate.create_list_of_dict_VLBI(pmparins, pmparin_preliminaries)
    LoD_timing = list_of_dict_timing = simulate.create_list_of_dict_timing(parfiles)
    dict_timing = LoD_timing[0] ## see the Notice
    Pb = dict_timing['pb'].value ## orbital period in day

    t = Table.read(posterior_samples, format='ascii')
    parameters = t.colnames[:-2]
    dict_median, outputfile = simulate.make_a_brief_summary_of_Bayesian_inference(posterior_samples)
    
    epochs = LoD_VLBI[0]['epochs']
    NoE = len(epochs)
    min_epoch, max_epoch = min(epochs), max(epochs)
    #Ts = np.arange(min_epoch, max_epoch, time_resolution)
    orbital_phases = np.arange(0, 1, phase_resolution)
    Ts = orbital_phases * Pb + refepoch
    #model_radecs = positions(refepoch, Ts, LoD_timing, 0, dict_median)
    model_radec_offsets = positions.model_parallax_and_reflex_motion_offsets(Ts, dict_median, dict_timing, no_px=True)
    
    #################################
    ### plot the model first
    #################################
    fig1 = plt.figure()
    gs1 = gridspec.GridSpec(1, 2)
    
    ax1 = fig1.add_subplot(gs1[0])
    ax2 = fig1.add_subplot(gs1[1])
    
    ## >>> if show Bayesian parameter errors
    if N_random_draw > 10:
        for j in range(NoP):
            posterior_indice = np.random.randint(0, len(t), N_random_draw) 
            for index in posterior_indice:
                sim_radec_offsets = positions.simulate_positions_subtracted_by_proper_motion(refepoch, Ts, t[index], j, dict_median, dict_timing, no_px=True)
                ax1.plot(orbital_phases, sim_radec_offsets[:len(Ts)], color=colors[j], alpha=15./N_random_draw)
                ax2.plot(orbital_phases, sim_radec_offsets[len(Ts):], color=colors[j], alpha=15./N_random_draw)
    ## <<<
    
    model_linewidth = 0.7
    ax1.plot(orbital_phases, model_radec_offsets[:len(Ts)], color='magenta', lw=model_linewidth)
    ax1.set_xlabel('orbital phase')
    ax1.set_ylabel('RA. offset (mas)')
    #ax1.set_title('RA-time (proper motion removed)')

    ax2.plot(orbital_phases, model_radec_offsets[len(Ts):], color='magenta', lw=model_linewidth)
    ax2.set_xlabel('orbital phase')
    ax2.set_ylabel('Dec offset (mas)')
    #ax2.set_title('Dec-time (proper motion removed)')




    #######################################
    ### plot the observed positions now
    #######################################
    orbital_phases_obs = ((epochs - refepoch) / Pb) % 1
    for i in range(NoP):
        trs = transparency = errorbar_transparency(i, -0.5)
        radec_offsets, _ = positions.observed_positions_subtracted_by_proper_motion(refepoch, LoD_VLBI[i], i, dict_median, no_px=True)
        errs_new = simulate.adjust_errs_with_efac(LoD_VLBI[i], dict_median, i)
        dec = LoD_VLBI[i]['radecs'][-1]
        errs_new = positions.convert_radec_errs_rad2mas(errs_new, dec)
        ax1.scatter(orbital_phases_obs, radec_offsets[:NoE], marker='.', alpha=trs, color=colors[i])
        ax1.errorbar(orbital_phases_obs, radec_offsets[:NoE], yerr=errs_new[:NoE], fmt='.', markersize=5, capsize=3, alpha=trs, label=legend_labels[i], color=colors[i])
        ax2.scatter(orbital_phases_obs, radec_offsets[NoE:], marker='.', alpha=trs, color=colors[i])
        ax2.errorbar(orbital_phases_obs, radec_offsets[NoE:], yerr=errs_new[NoE:], fmt='.', markersize=5, capsize=3, alpha=trs, color=colors[i])

    ax1.legend(loc=legend_loc)
    plot_title = r'$P_b=%3.2f\,\mathrm{day}$; MJD %d is at orbital phase 0' % (Pb, refepoch)
    plt.suptitle(plot_title)
    gs1.tight_layout(fig1)
    plt.savefig('ra_dec_orbital_phase__nopm_nopx_Bayesian.pdf')
    plt.clf()

def projected_orbit(pmparins, parfiles, refepoch, posterior_samples='outdir/posterior_samples.dat', **kwargs):
    """
    Purpose
    -------
        to plot projected orbit revealed/shared by one or more pmparin(s).

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
        1. phase_resolution : int (default : 0.01)
            A fraction of the orbital phase. The smaller the higher the time resolution.
        2. N_random_draw : int (default : -999)
            Number of random draw from the posterior sample.
            It is activated when N_random_draw > 5
        3. legend_labels : list of str
        4. colors : list of str
        5. legend_loc : str (default : 'lower left')
            where on the left panel to place the legend. There are 6 options: 'lower/upper left/center/right'.
    """
    #########################
    ## set up variables
    #########################
    try:
        phase_resolution = kwargs['phase_resolution']
    except KeyError:
        phase_resolution = 0.01
    try:
        N_random_draw = kwargs['N_random_draw']
    except KeyError:
        N_random_draw = -999
    
    NoP = len(pmparins)
    
    try:
        legend_labels = kwargs['legend_labels']
    except KeyError:
        if NoP >= 1:
            legend_labels = [''] * NoP
    try:
        colors = kwargs['colors']
    except KeyError:
        colors = ['mediumblue','tomato','r', 'y', 'lime']
    try:
        legend_loc = kwargs['legend_loc']
    except KeyError:
        legend_loc = 'lower left'
    
    LoD_VLBI = list_of_dict_VLBI = simulate.create_list_of_dict_VLBI(pmparins)
    LoD_timing = list_of_dict_timing = simulate.create_list_of_dict_timing(parfiles)
    dict_timing = LoD_timing[0] ## see the Notice
    Pb = dict_timing['pb'].value ## orbital period in day

    t = Table.read(posterior_samples, format='ascii')
    parameters = t.colnames[:-2]
    dict_median, outputfile = simulate.make_a_brief_summary_of_Bayesian_inference(posterior_samples)
    
    epochs = LoD_VLBI[0]['epochs']
    NoE = len(epochs)
    min_epoch, max_epoch = min(epochs), max(epochs)
    #Ts = np.arange(min_epoch, max_epoch, time_resolution)
    orbital_phases = np.arange(0, 1, phase_resolution)
    Ts = orbital_phases * Pb + refepoch
    #model_radecs = positions(refepoch, Ts, LoD_timing, 0, dict_median)
    model_radec_offsets = positions.model_parallax_and_reflex_motion_offsets(Ts, dict_median, dict_timing, no_px=True)

    #################################
    ### plot the model first
    #################################
    a = plt.scatter(1000*model_radec_offsets[:len(Ts)], 1000*model_radec_offsets[len(Ts):], c=Ts-refepoch, cmap='viridis')
    cbar = plt.colorbar(a)
    cbar.set_label('time in an orbit (day)', rotation=90)
    #plt.gca().set_aspect('equal')
    ax1 = plt.gca()
    ax1.invert_xaxis()
    plt.xlabel(r'RA offset ($\mu$as)')
    plt.ylabel(r'Dec offset ($\mu$as)')
    plt.title('projected orbit')
    plt.tight_layout()
    plt.savefig('projected_orbit.pdf')
    plt.clf()
