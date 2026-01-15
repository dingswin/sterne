#!/usr/bin/env python
"""
sterne.simulate.py is written in python3 by Hao Ding.
The main code to run is simulate().
"""
import bilby
from astropy.time import Time
import numpy as np
import astropy.units as u
from astropy import constants
import os, sys
from sterne import others
from astropy.table import Table
from sterne.model import kopeikin_effects, reflex_motion
from sterne.model.positions import positions, position, filter_dictionary_of_parameter_with_index, parallax_related_position_offset_from_the_barycentric_frame
from sterne import priors as _priors

class Model:
    """
    A class to hold the model parameters (median values) and calculate astrometric 
    and reflex motion.

    Attributes
    ----------
    refepoch : float
        Reference epoch in MJD.
    ra : float
        Reference Right Ascension in radians.
    dec : float
        Reference Declination in radians.
    mu_a : float
        Proper motion in RA in mas/yr.
    mu_d : float
        Proper motion in Dec in mas/yr.
    px : float
        Parallax in mas.
    incl : float
        Orbital inclination in radians (if fitted/provided).
    om_asc : float
        Longitude of ascending node in degrees (if fitted/provided).
    dict_timing : dict
        Dictionary of orbital parameters (from parfile).
    """

    def __init__(self, dict_median, dict_timing, refepoch, index=0):
        """
        Parameters
        ----------
        dict_median : dict
            Dictionary of median values from Bayesian inference (output of 
            make_a_summary_of_bayesian_inference). keys are like 'ra_0', 'px_0'.
        dict_timing : dict
            Dictionary of orbital parameters (usually from reading the parfile).
            Can be empty {} if no binary motion is modeled.
        refepoch : float
            Reference epoch in MJD.
        index : int, optional
            The index of the pmparin/source to filter parameters for (default is 0).
            Used to map specific keys (e.g., 'ra_0') to generic attributes (e.g., 'ra').
        """
        self.refepoch = refepoch
        self.dict_timing = dict_timing
        
        # Filter parameters for this specific index using existing sterne utility
        # -999 indicates the parameter was not fitted/found.
        self.parameters = filter_dictionary_of_parameter_with_index(dict_median, index)
        print(dict_median.keys(), self.parameters.keys())
        
        # Astrometric Parameters
        self.ra = self.parameters.get('ra_{0}'.format(index))       # rad
        self.dec = self.parameters.get('dec_{0}'.format(index))     # rad
        self.mu_a = self.parameters.get('mu_a_{0}'.format(index))   # mas/yr
        self.mu_d = self.parameters.get('mu_d_{0}'.format(index))   # mas/yr
        self.px = self.parameters.get('px_{0}'.format(index))       # mas
        print(self.ra, self.dec, self.mu_a, self.mu_d, self.px)
        
        # Orbital Geometry Parameters
        self.incl = self.parameters.get('incl_{0}'.format(index))     # rad
        self.om_asc = self.parameters.get('om_asc_{0}'.format(index)) # deg

    def calculate_reflex_motion(self, epoch):
        """
        Calculate reflex motion vector at a specific epoch.
        
        Parameters
        ----------
        epoch : float
            Epoch in MJD.
            
        Returns
        -------
        np.array
            [dRA, dDEC] in mas. Returns [0,0] if orbital parameters are missing.
        """
        # Ensure we have valid orbital geometry and timing parameters
        valid_orbit = (self.dict_timing and 
                       self.incl is not None and self.incl != -999 and 
                       self.om_asc is not None and self.om_asc != -999)
        
        if valid_orbit:
            # Handle px: if px was not fitted (-999), assume 0 for reflex calc or passed value
            px_val = self.px if self.px != -999 else 0.0
            
            return reflex_motion.reflex_motion(
                epoch, 
                self.dict_timing, 
                self.incl, 
                self.om_asc, 
                px_val
            )
        else:
            return np.array([0.0, 0.0])

    def get_position_at_epoch(self, epoch):
        """
        Calculate the full position (RA, Dec) at a given epoch including 
        Proper Motion, Parallax, and Reflex Motion.
        
        Parameters
        ----------
        epoch : float
            Epoch in MJD.
            
        Returns
        -------
        tuple
            (RA, Dec) in radians (geocentric).
        """
        # Utilizes the existing 'positions' function from sterne.model.positions
        # imported in simulate.py
        ra_model, dec_model = positions(
            self.refepoch, 
            epoch, 
            self.dec, 
            self.incl, 
            self.mu_a, 
            self.mu_d, 
            self.om_asc, 
            self.px, 
            self.ra, 
            self.dict_timing
        )
        return ra_model, dec_model


def write_observation_summary(model_obj, obs_dict, output_filename):
    """
    Writes a fixed-width table summarizing observations and model offsets.

    Parameters
    ----------
    model_obj : Model
        The Model instance containing best-fit median parameters.
    obs_dict : dict
        The dictionary containing observation data for this specific source 
        (one element from list_of_dict_VLBI).
    output_filename : str
        The path to save the text file.
    """
    
    # Extract observation data
    epochs = obs_dict['epochs']
    n_obs = len(epochs)
    
    # VLBI data is stored as [RA_1...RA_n, Dec_1...Dec_n] in radians
    #
    vlbi_ras_rad = obs_dict['radecs'][:n_obs]
    vlbi_decs_rad = obs_dict['radecs'][n_obs:]
    
    # Errors are stored similarly in radians
    #
    vlbi_err_ras_rad = obs_dict['errs'][:n_obs]
    vlbi_err_decs_rad = obs_dict['errs'][n_obs:]

    # Define column widths and header
    preamble = "Model reference epoch: {0}\n".format(model_obj.refepoch)
    preamble = preamble + "Model reference right ascension (radians): {0}\n".format(model_obj.ra)
    preamble = preamble + "Model reference declination (radians): {0}\n".format(model_obj.dec)
    preamble = preamble + "Model proper motion R.A. (mas/yr): {0}\n".format(model_obj.mu_a)
    preamble = preamble + "Model proper motion Dec. (mas/yr): {0}\n".format(model_obj.mu_d)
    preamble = preamble + "Model parallax (mas): {0}\n".format(model_obj.px)
    preamble = preamble + "Model binary inclination (degrees): {0}\n".format(model_obj.incl)
    preamble = preamble + "Model binary ascending node longitude (degrees): {0}\n".format(model_obj.om_asc)
    header = (
        f"{'MJD':<12} {'Date':<12} "
        f"{'RA_deg':<15} {'ErrRA_ms':<12} "
        f"{'Dec_deg':<15} {'ErrDec_mas':<12} "
        f"{'Px_RA_mas':<12} {'Px_Dec_mas':<12} "
        f"{'Orb_RA_mas':<12} {'Orb_Dec_mas':<12}"
    )

    with open(output_filename, 'w') as f:
        f.write(preamble + "\n")
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")

        for i in range(n_obs):
            mjd = epochs[i]
            
            # 1 & 2: Dates
            # Convert MJD to YYYY-MM-DD
            date_str = Time(mjd, format='mjd').iso.split(' ')[0]

            # 3: VLBI RA (convert rad to deg)
            ra_deg = np.degrees(vlbi_ras_rad[i])

            # 4: VLBI RA Uncertainty (milliseconds)
            # rad -> deg -> hours -> seconds -> ms
            # Factor: (180/pi) / 15 * 3600 * 1000
            ra_err_ms = vlbi_err_ras_rad[i] * (180.0 / np.pi) / 15.0 * 3.6e6

            # 5: VLBI Dec (convert rad to deg)
            dec_deg = np.degrees(vlbi_decs_rad[i])

            # 6: VLBI Dec Uncertainty (milliarcseconds)
            # rad -> deg -> arcsec -> mas
            # Factor: (180/pi) * 3600 * 1000
            dec_err_mas = vlbi_err_decs_rad[i] * (180.0 / np.pi) * 3.6e6

            # 7 & 8: Parallax Offsets (mas)
            # Uses sterne.model.positions
            # Note: positions.py functions require radians for input RA/Dec
            print(mjd, model_obj.ra, model_obj.dec, model_obj.px)
            px_offsets = parallax_related_position_offset_from_the_barycentric_frame(
                mjd, model_obj.ra, model_obj.dec, model_obj.px
            )
            px_ra_offset = px_offsets[0]
            px_dec_offset = px_offsets[1]

            # 9 & 10: Orbital Model Offsets (mas)
            # Uses the method defined in the new Model class
            orb_offsets = model_obj.calculate_reflex_motion(mjd)
            orb_ra_offset = orb_offsets[0]
            orb_dec_offset = orb_offsets[1]

            # Write row
            row = (
                f"{mjd:<12.5f} {date_str:<12} "
                f"{ra_deg:<15.9f} {ra_err_ms:<12.4f} "
                f"{dec_deg:<15.9f} {dec_err_mas:<12.4f} "
                f"{px_ra_offset:<12.4f} {px_dec_offset:<12.4f} "
                f"{orb_ra_offset:<12.4f} {orb_dec_offset:<12.4f}"
            )
            f.write(row + "\n")

    print(f"Summary written to {output_filename}")

def simulate(refepoch, initsfile, pmparin, parfile, *args, **kwargs):
    """
    Input parameters
    ----------------
    refepoch : float
        Reference epoch (MJD).
    initsfile : str
        A file ending with '.inits' that contains priors of parameters to fit. initsfile 
        should be pre-made. It can be made with generate_initsfile(). Priors in initsfile 
        need to be updated before running simulate().
    pmparin : str
        A file ending with '.pmpar.in' which contains observed position info.
    parfile : str
        A parfile ending with '.par' which contains orbital info for a pulsar binary system.
        parfiles should be pre-made. 
        1) Each parfile can be made with 'psrcat -e PULSARNAME > PARFILENAME',
            using the PSRCAT catalog. Om_asc and incl in parfiles are so far unused.
            The timing parameters offered in parfiles should be updated before use.
        2) Only when a parfile is provided for a pmparin will reflex_motion be provoked to 
            estimate related position offset. In case where reflex_motion is not required,
            please provide '' for parfile. By doing so, reflex_motion will be turned off,
            even when the correspoinding shares indice are >=0.

    args : str(s)
        1) to provide extra pmparin files and parfiles.
        2) the order of args should be either pmparin1, parfile1, pmparin2, parfile2,....
        3) an example for two pulsars in a globular cluster: 
        4) an arg both containing '.pmpar.in' and ending with '.par' should be avoided. 
    kwargs : key=value
        1) shares : 2-D array 
            (default : [list(range(N)),[0]*N,[0]*N,[0]*N,[0]*N,[0]*N,[0]*N,list(range(N))]) 
            Used to assign shared parameters to fit and which paramters to not fit.
            The size of shares is 9*N, 9 refers to the 9 parameters ('dec','efac', 'efad', 'incl',
            'mu_a','mu_d','Om_asc','px','ra' in alphabetic order); N refers to the number
            of pmparins. As an example, for four pmparins, shares can be
            [[0,0,1,1],[0,1,2,2],[0,0,0,0],[0,0,1,1],[0,0,1,1],[0,0,1,1],[0,0,1,1],[0,0,0,0],[0,1,2,3]].
            Same numbers in the same row shares the same parameter (e.g. 'px' is shared by all
            pmparins). Furthermore, if shares[i][j]<0, it means the inference for
            parameter[i] with pmparins[j] is turned off. This turn-off function is not so
            useful now, but may be helpful in future.
        2) iterations : float (default : 200)
            'iterations' that will be passed to bilby.run_sampler().
            Changing "iterations" to over 500 would avoid fuzzy corner plots, while
            "interations"=1000 would make smooth corner plots.
            NOTE:
            when sampler is ptemcee, iterations is used for nsamples of the sampler.
            
        3) nwalkers : float (default : 30)
            'nwalkers' that will be passed to bilby.run_sampler().
        4) outdir : float
            'outdir' that will be passed to run_sampler().
        5) use_saved_samples : bool
            If True, run_sampler will be bypassed.
        6) a1dot_constraints : a list of a list of 2 floats (default : False)
            e.g. [[mu, sigma], []], (both in lt-sec/sec),
            where mu and sigma refers to the Gaussian distribution for a1dot.
            The length of a1dot_constraint needs to match len(pmparins), unless None.
        7) pmparin_preliminaries : list of str (default : None)
            A list of pmpar.in.preliminary files which record random errors. Once this is provided,
            EFAC will be fit for. Otherwise, EFAC will not be inferred (accordingly one more 
            degree of freedom). When pmparin_preliminaries==None, the inference for efac (and efad) would 
            be turned off.
            The error is corrected following the relation:
            errs_new**2 = errs_random**2 + (efac * errs_sys)**2, where errs_random and errs_sys
            stand for random errors and systematic errors, respectively.
            When efad is requested, it is applied to errors in declinations in the following way:
            errs_new**2 = errs_random**2 + (efac * efad * errs_sys)**2.
        8) sampler: str (default : 'emcee')
            when the posterior distribution is unimodal, using 'emcee' is fine. Otherwise, use 'ptemcee' instead.
        9) log_efac : bool (default : False)
            if True, efac and efad will actually be log(efac) and log(efad).

    Caveats
    -------
    We assume a1dot is predominantly attributed to the variation of inclination due to 
        the proper motion (Kopeikin, 1996), which is normally valid for pulsars in wide orbits.
        Should there be a remarkable a1dot owing to gravitational-wave damping, the reflex
        motion is normally not prominent (as the orbit is usually compact).

    ** Examples ** : (not updated to efac and efad situations)
        1) For two pulsars in a globular cluster:
            simulate(57444,'a.inits','p1.pmpar.in','','p2.pmpar.in','p2.par',shares=[[0,1],
                [-1,0],[0,1],[0,1],[-1,0],[0,0],[0,1]])
        2) For a pulsar with two in-beam calibrators:
            simulate(57444,'a.inits','i1.pmpar.in','p.par','i2.pmpar.in','p.par',
                shares=[[0,1],[0,0],[0,0],[0,0],[0,0],[0,0],[0,1]])
        3) For two pulsars in a globular cluster sharing an in-beam calibrator:
            simulate(57444,'a.inits','i1p1.pmpar.in', '', 'i2p1.pmpar.in','', 'i1p2.pmpar.in',
                'p2.par','i2p2.pmpar.in','p2.par',shares=[[1,2,3,4],[-1,-1,0,0],
                [1,1,2,2],[1,1,2,2],[-1,-1,0,0],[1,1,1,1],[1,2,3,4]])
    """
    ##############################################################
    ############ parse args to get pmparins, parfiles ############
    ##############################################################
    if not os.path.exists(initsfile):
        print('%s does not exist; aborting' % initsfile)
        sys.exit()
    args = list(args)
    args.reverse()
    args.append(parfile)
    args.append(pmparin)
    args.reverse() #put the major pmparin at first
    pmparins, parfiles = [], []
    for arg in args:
        if ('.pmpar.in' in arg) and arg.endswith('.par'):
            print('arg does not follow naming convention,\
                please follow the docstring! Aborting.')
            sys.exit()
        elif '.pmpar.in' in arg:
            pmparins.append(arg)
        elif arg.endswith('.par') or arg=='':
            parfiles.append(arg)
    NoP = len(pmparins)
    if NoP != len(parfiles):
        print('Unequal number of parfiles provided for pmparins.\
            See the docstring for more info. Aborting now.')
        sys.exit()

    ##############################################################
    #####################  parse kwargs  #########################
    ##############################################################
    try:
        shares = kwargs['shares']
    except KeyError:
        shares = [list(range(NoP)), [0]*NoP, [0]*NoP, [0]*NoP, [0]*NoP, [0]*NoP,\
            [0]*NoP, [0]*NoP, list(range(NoP))]
    print(pmparins, parfiles, initsfile, shares)
    
    try:
        outdir = kwargs['outdir']
    except KeyError:
        outdir = 'outdir'
    try:
        use_saved_samples = kwargs['use_saved_samples']
    except KeyError:
        use_saved_samples = False

    try:
        iterations = kwargs['iterations']
    except KeyError:
        iterations = 200
    try:
        nwalkers = kwargs['nwalkers']
    except KeyError:
        nwalkers = 30
    try:
        sampler = kwargs['sampler']
    except KeyError:
        sampler = 'emcee' ## when the posterior distribution is multi-modal, use 'ptemcee' instead
    #try:
    #    nsamples = kwargs['nsamples']
    #except KeyError:
    #    nsamples = 100

    try:
        a1dot_constraints = kwargs['a1dot_constraints']
    except KeyError:
        a1dot_constraints = False

    try:
        pmparin_preliminaries = kwargs['pmparin_preliminaries']
        if len(pmparin_preliminaries) != NoP:
            print('The number of pmpar.in.preliminary files has to\
                match that of pmpar.in files. Exiting for now.')
            sys.exit(1)
    except KeyError:
        pmparin_preliminaries = None
        shares[1] = [-1] * NoP ## turn off efac inference
        shares[2] = [-1] * NoP ## turn off efad inference

    try:
        log_efac = kwargs['log_efac']
    except KeyError:
        log_efac = False
    ##############################################################
    #################  get two list_of_dict ######################
    ##############################################################
    list_of_dict_timing = create_list_of_dict_timing(parfiles)

    list_of_dict_VLBI = create_list_of_dict_VLBI(pmparins, pmparin_preliminaries)

    
    ##############################################################
    ###################### run simulations #######################
    ##############################################################
    saved_posteriors = outdir + '/posterior_samples.dat'
    if not use_saved_samples:
        limits, DoD_additional_constraints = _priors.read_inits(initsfile)
        print(limits)
        priors = _priors.create_priors_given_limits_dict(limits, DoD_additional_constraints)
        
        likelihood = Gaussianlikelihood(refepoch, list_of_dict_timing, list_of_dict_VLBI,\
            shares, positions, DoD_additional_constraints, a1dot_constraints)
        if sampler != 'emcee':
            result = bilby.run_sampler(likelihood=likelihood, priors=priors,\
                sampler=sampler, nwalkers=nwalkers, outdir=outdir, pos0='prior', nsamples=iterations)
        else:
            result = bilby.run_sampler(likelihood=likelihood, priors=priors,\
                sampler=sampler, nwalkers=nwalkers, iterations=iterations,  outdir=outdir)
            
    jsonfile = outdir + '/label_result.json' 
    result = bilby.result.read_in_result(filename=jsonfile) 
    result.save_posterior_samples(filename=saved_posteriors)
    dict_median, outputfile = make_a_summary_of_bayesian_inference(saved_posteriors, refepoch,\
        list_of_dict_VLBI, list_of_dict_timing, log_efac)

    # Make a corner plot of the results
    result.plot_corner() ## this may fail when run in the background, therefore put in the last

    # Create a Model object for each pmparin index
    models = []
    for i in range(len(list_of_dict_timing)):
        # Instantiate Model with the best-fit median dictionary
        model_obj = Model(dict_median, list_of_dict_timing[i], refepoch, index=i)
        models.append(model_obj)
        write_observation_summary(model_obj, list_of_dict_VLBI[i], "outdir/observation_model_summary_{0}.txt" .format(i))

def create_list_of_dict_timing(parfiles):
    list_of_dict_timing = []
    for parfile in parfiles:
        if parfile != '':
            dict_of_timing_parameters = reflex_motion.read_parfile(parfile)
        else:
            dict_of_timing_parameters = {}
        list_of_dict_timing.append(dict_of_timing_parameters)
    print(list_of_dict_timing)
    return list_of_dict_timing

def create_list_of_dict_VLBI(pmparins, pmparin_preliminaries=None):
    NoP = len(pmparins)
    
    list_of_dict_VLBI = []
    for pmparin in pmparins:
        t = readpmparin(pmparin)
        radecs = np.concatenate([t['RA'], t['DEC']])
        errs = np.concatenate([t['errRA'], t['errDEC']])
        epochs = np.array(t['epoch'])
        dictionary = {}
        dictionary['epochs'] = epochs
        dictionary['radecs'] = radecs
        dictionary['errs'] = errs
        list_of_dict_VLBI.append(dictionary)
    if pmparin_preliminaries != None:
        if len(pmparin_preliminaries) != NoP:
            print('The number of pmpar.in.preliminary files has to\
                match that of pmpar.in files. Exiting for now.')
            sys.exit(1)
        for i in range(NoP):
            t = readpmparin(pmparin_preliminaries[i])
            if not (np.array(t['epoch']) == list_of_dict_VLBI[i]['epochs']).all():
                print('Epochs of the pmpar.in.preliminary files should match those of the pmpar.in files; exiting for now.')
                sys.exit(1)
            errs_random = np.concatenate([t['errRA'], t['errDEC']])
            list_of_dict_VLBI[i]['errs_random'] = errs_random
            errs = list_of_dict_VLBI[i]['errs']
            errs_sys = (errs**2 - errs_random**2)**0.5
            list_of_dict_VLBI[i]['errs_sys'] = errs_sys
    print(list_of_dict_VLBI)
    return list_of_dict_VLBI
    

def make_a_summary_of_bayesian_inference(samplefile, refepoch, list_of_dict_VLBI, list_of_dict_timing, log_efac):
    dict_median, outputfile = make_a_brief_summary_of_Bayesian_inference(samplefile, True, log_efac)
    writefile = open(outputfile, 'a')
    chi_sq, rchsq = calculate_reduced_chi_square(refepoch, list_of_dict_VLBI, list_of_dict_timing, dict_median, log_efac)
    writefile.write('\nchi-square = %f\nreduced chi-square = %f\n' % (chi_sq, rchsq))
    writefile.write('The reference epoch is MJD %d \n' % refepoch)
    writefile.close()
    return dict_median, outputfile

def make_a_brief_summary_of_Bayesian_inference(samplefile, write=True, log_efac=False):
    t = Table.read(samplefile, format='ascii')
    parameters = t.colnames[:-2]
    dict_median = {}
    dict_bound = {} #16% and 84% percentiles
    dict_median_log, dict_bound_log = {}, {}
    outputfile = samplefile.replace('posterior_samples', 'bayesian_estimates')
    if write:
        writefile = open(outputfile, 'w')
        writefile.write('#Medians of the simulated samples:\n')
        if log_efac:
            writefile.write('#log(efac) and log(efad) are shown in the corner plot for efac and efad.\n')
        writefile.write('#(Units: px in mas; mu_a and mu_d in mas/yr; incl in rad.)\n')
        for p in parameters:
            if 'om_asc' in p: ## for om_asc
                dict_median[p], upper_side_error, lower_side_error = others.periodic_sample2estimate(t[p]) ## the narrowest confidence interval is the error bound, the median of this interval is used as the median.
                writefile.write('%s = %f + %f - %f (deg)\n' % (p, dict_median[p], upper_side_error, lower_side_error)) 
            else:
                dict_median[p] = others.sample2median(t[p])
                dict_bound[p] = others.sample2median_range(t[p], 1)
                if 'ra' in p:
                    writefile.write('%s = %s + %f - %f (ms)\n' % (p, others.deg2dms(dict_median[p]*180/np.pi/15), (dict_bound[p][1]-dict_median[p])*180/np.pi/15*3600*1000, (dict_median[p]-dict_bound[p][0])*180/np.pi/15*3600*1000))
                elif 'dec' in p:
                    writefile.write('%s = %s + %f - %f (mas)\n' % (p, others.deg2dms(dict_median[p]*180/np.pi), (dict_bound[p][1]-dict_median[p])*180/np.pi*3600*1000, (dict_median[p]-dict_bound[p][0])*180/np.pi*3600*1000))
                elif ('efac' in p) or ('efad' in p):
                    if not log_efac:
                        writefile.write('%s = %f + %f - %f\n' % (p, dict_median[p],\
                            dict_bound[p][1]-dict_median[p], dict_median[p]-dict_bound[p][0]))
                        dict_median_log[p] = others.sample2median(np.log(t[p]))
                        dict_bound_log[p] = others.sample2median_range(np.log(t[p]), 1)
                        writefile.write('log(%s) = %f + %f - %f\n' % (p, dict_median_log[p],\
                            dict_bound_log[p][1]-dict_median_log[p], dict_median_log[p]-dict_bound_log[p][0]))
                    else:
                        writefile.write('log(%s) = %f + %f - %f\n' % (p, dict_median[p],\
                            dict_bound[p][1]-dict_median[p], dict_median[p]-dict_bound[p][0]))
                else:
                    writefile.write('%s = %f + %f - %f\n' % (p, dict_median[p],\
                        dict_bound[p][1]-dict_median[p], dict_median[p]-dict_bound[p][0]))
                            

        
        ## >>> estimate correlation coefficients
        DoR = dict_of_correlation_coefficient = {}
        writefile.write('\n#Correlation coefficients:\n')
        for i in range(1, len(parameters)):
            for j in range(i):
                key = 'r__' + parameters[j] + '__' + parameters[i]
                DoR[key] = np.corrcoef(t[parameters[j]], t[parameters[i]])[0,1]
                writefile.write('%s = %f\n' % (key, DoR[key]))
        #print(DoR)
        ## <<<
        writefile.close()
    else:
        for p in parameters:
            if 'om_asc' in p: ## for om_asc
                dict_median[p], upper_side_error, lower_side_error = others.periodic_sample2estimate(t[p]) ## the narrowest confidence interval is the error bound, the median of this interval is used as the median.
            else:
                dict_median[p] = others.sample2median(t[p])
                dict_bound[p] = others.sample2median_range(t[p], 1)
        
        ## >>> estimate correlation coefficients
        DoR = dict_of_correlation_coefficient = {}
        for i in range(1, len(parameters)):
            for j in range(i):
                key = 'r__' + parameters[j] + '__' + parameters[i]
                DoR[key] = np.corrcoef(t[parameters[j]], t[parameters[i]])[0,1]
        ## <<<
    return dict_median, outputfile
    

def calculate_reduced_chi_square(refepoch, list_of_dict_VLBI, list_of_dict_timing, dict_median, log_efac):
    LoD_VLBI, LoD_timing = list_of_dict_VLBI, list_of_dict_timing
    chi_sq = 0
    NoO = number_of_observations = 0
    for i in range(len(LoD_VLBI)):
        res = LoD_VLBI[i]['radecs'] - positions(refepoch, LoD_VLBI[i]['epochs'], LoD_timing[i], i, dict_median)
        errs_new = adjust_errs_with_efac(LoD_VLBI[i], dict_median, i, log_efac)
        chi_sq += np.sum((res/errs_new)**2) #if both RA and errRA are weighted by cos(DEC), the weighting is canceled out
        NoO += 2 * len(LoD_VLBI[i]['epochs'])
    DoF = degree_of_freedom = NoO - len(dict_median)
    rchsq = chi_sq / DoF
    return chi_sq, rchsq

def adjust_errs_with_efac(VLBI_dict, parameters_dict, parameter_filter_index, log_efac=False):
    FP = filter_dictionary_of_parameter_with_index(parameters_dict, parameter_filter_index)
    Ps = list(FP.keys())
    Ps.sort()
    efac = FP[Ps[1]]
    efad = FP[Ps[2]]
    
    N = int(len(VLBI_dict['errs']) / 2)
    if efad != -999:
        if log_efac:
            efads = np.concatenate([np.ones(N), np.exp(efad) * np.ones(N)])
        else:
            efads = np.concatenate([np.ones(N), efad * np.ones(N)])
    else:
        efads = np.ones(2 * N)
    
    if efac != -999:
        if log_efac:
            errs_new_sq = (VLBI_dict['errs_random'])**2 + (np.exp(efac) * efads * VLBI_dict['errs_sys'])**2
        else:
            errs_new_sq = (VLBI_dict['errs_random'])**2 + (efac * efads * VLBI_dict['errs_sys'])**2
    else: ## if efac is not to be inferred
        errs_new_sq = VLBI_dict['errs']**2
    errs_new = errs_new_sq**0.5
    return errs_new 




class Gaussianlikelihood(bilby.Likelihood):
    def __init__(self, refepoch, list_of_dict_timing, list_of_dict_VLBI, shares, positions, DoD_additional_constraints, a1dot_constraints=False, log_efac=False):
        """
        Addition of multiple Gaussian likelihoods

        Parameters
        ----------
        data: array_like
            The data to analyse
        sin_incl_constraints --> sin_incl_Gaussian_constraints
        """
        self.refepoch = refepoch
        self.LoD_VLBI = list_of_dict_VLBI
        self.LoD_timing = list_of_dict_timing
        self.positions = positions
        self.shares = shares
        self.number_of_pmparins = len(self.LoD_VLBI)
        #self.pmparin_preliminaries = pmparin_preliminaries
        self.dict_sin_incl_Gaussian_constraints = DoD_additional_constraints['sin_incl_Gaussian_constraints']
        self.log_efac = log_efac
        
        if a1dot_constraints != False:
            self.a1dot_constraints, self.a1dot_mus, self.a1dot_sigmas = self.parse_a1dot_constraints(a1dot_constraints)
        else:
            self.a1dot_constraints = False
        '''
        if sin_incl_constraints != None:
            self.sin_incl_constraints, self.sin_incl_mus, self.sin_incl_sigmas = self.parse_sin_incl_constraints(sin_incl_constraints)
        else:
            self.sin_incl_constraints = False
        ''' 

        parameters = _priors.get_parameters_from_shares(self.shares)
        print(parameters)
        super().__init__(parameters)

        

    def log_likelihood(self):
        """
        the name has to be log_likelihood, and the PDF has to do the log calculation.
        """
        log_p = 0
        for i in range(self.number_of_pmparins):
            res = self.LoD_VLBI[i]['radecs'] - self.positions(self.refepoch, self.LoD_VLBI[i]['epochs'], self.LoD_timing[i], i, self.parameters)
            errs_new = adjust_errs_with_efac(self.LoD_VLBI[i], self.parameters, i, self.log_efac) 
            log_p += -0.5 * np.sum((res/errs_new)**2) ##if both RA and errRA are weighted by cos(DEC), the weighting is canceled out
            log_p += -1 * np.sum(np.log(errs_new))
        
        if self.a1dot_constraints:
            modeled_a1dots = kopeikin_effects.calculate_a1dot_pm(self.LoD_timing, self.parameters)
            #print('ETRA=%.20f' % ETRA)
            res_a1dots = modeled_a1dots - self.a1dot_mus
            log_p += -0.5 * np.sum((res_a1dots / self.a1dot_sigmas)**2)

        if len(self.dict_sin_incl_Gaussian_constraints) != 0:
            for parameter in self.dict_sin_incl_Gaussian_constraints:
                sin_incl_mu, sin_incl_sigma = self.dict_sin_incl_Gaussian_constraints[parameter]
                res = sin_incl_mu - np.sin(self.parameters[parameter])
                log_p += -0.5 * (res / sin_incl_sigma)**2
        return log_p
    

    def parse_a1dot_constraints(self, a1dot_constraints):
        a1dot_mus = np.array([])
        a1dot_sigmas = np.array([])
        for a1dot_constraint in a1dot_constraints:
            if len(a1dot_constraint) == 2:
                a1dot_mu, a1dot_sigma = a1dot_constraint
                a1dot_mus = np.append(a1dot_mus, a1dot_mu)
                a1dot_sigmas = np.append(a1dot_sigmas, a1dot_sigma)
        if len(a1dot_mus) == 0:
            a1dot_constraints = False
        else:
            a1dot_constraints = True
        return a1dot_constraints, a1dot_mus, a1dot_sigmas

    def __parse_sin_incl_constraints(self, sin_incl_constraints):
        """
        for sin_incl_Gaussian_constraints
        """
        sin_incl_mus = np.array([])
        sin_incl_sigmas = np.array([])
        sin_incl_constraints = False
        for sin_incl_constraint in sin_incl_constraints:
            if len(sin_incl_constraint) == 2:
                sin_incl_mu, sin_incl_sigma = sin_incl_constraint
                sin_incl_mus = np.append(sin_incl_mus, sin_incl_mu)
                sin_incl_sigmas = np.append(sin_incl_sigmas, sin_incl_sigma)
                sin_incl_constraints = True  ## if only there is one sin_incl_Gaussian_constraint
            else: ## when equal to []
                sin_incl_mus = np.append(sin_incl_mus, None)
                sin_incl_sigmas = np.append(sin_incl_sigmas, None)
        return sin_incl_constraints, sin_incl_mus, sin_incl_sigmas

    def __log_p_sin_incl_residuals(self, dict_parameters, sin_incl_mus, sin_incl_sigmas):
        log_p = 0
        for key in dict_parameters:
            if 'incl' in key:
                pmparin_indice, parameter_root = _priors.parameter_name_to_pmparin_indice(key)
                index = pmparin_indice[0]
                if sin_incl_mus[index] != None:
                    sin_incl_residual = np.sin(dict_parameters[key]) - sin_incl_mus[index]
                    log_p += -0.5 * (sin_incl_residual/sin_incl_sigmas[index])**2
        return log_p


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
    ra = others.dms2deg(ra)
    ra *= 15 * np.pi/180 #in rad
    dec = others.dms2deg(dec)
    dec *= np.pi/180 #in rad
    return ra, dec
    






def readpmparin(pmparin):
    """
    Robust reader for pmpar.in files.
    Skips comments/blank lines and tolerates arbitrary whitespace.
    """
    epochs = []
    RAs = []
    errRAs = []
    DECs = []
    errDECs = []

    lines = open(pmparin).readlines()
    for line in lines:
        line = line.strip()

        # Skip blank lines and comments
        if not line or line.startswith('#'):
            continue

        precommentline = line.split('#')[0]
        parts = precommentline.split()  # split on any whitespace

        # Expect exactly 5 columns: epoch RA errRA DEC errDEC
        if len(parts) != 5:
            print(f"[readpmparin] Skipping malformed line: {line}")
            continue

        epoch, RA, errRA, DEC, errDEC = parts

        # get the epoch
        # decyear2mjd gives value in MJD, regardless of whether input is MJD or fractional year
        epoch = decyear2mjd(float(epoch))

        # DEC: d:m:s -> rad
        DEC = others.dms2deg(DEC)   # deg
        DEC *= np.pi / 180.0        # rad

        # RA: h:m:s -> rad
        RA = others.dms2deg(RA)     # hours
        RA *= 15.0 * np.pi / 180.0  # rad

        # errors
        errRA = float(errRA)        # seconds of time
        errRA *= 15.0 * np.pi / 180.0 / 3600.0   # -> rad

        errDEC = float(errDEC)      # arcseconds
        errDEC *= np.pi / 180.0 / 3600.0         # -> rad

        #print("Parsed errRA(sec->rad), errDEC(arcsec->rad):", errRA, errDEC)

        epochs.append(epoch)
        RAs.append(RA)
        DECs.append(DEC)
        errRAs.append(errRA)
        errDECs.append(errDEC)

    epochs = np.array(epochs)
    RAs = np.array(RAs)
    errRAs = np.array(errRAs)
    DECs = np.array(DECs)
    errDECs = np.array(errDECs)

    t = Table([epochs, RAs, errRAs, DECs, errDECs],
              names=['epoch', 'RA', 'errRA', 'DEC', 'errDEC'])
    t.sort('epoch')
    return t

def decyear2mjd(epoch):
    """
    """
    threshold = 10000
    if epoch > threshold:
        return epoch
    else:
        decyear = Time(epoch, format='decimalyear')
        MJD = float(format(decyear.mjd))
        return MJD

