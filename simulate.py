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
import others
from astropy.table import Table
from sterne.model import kopeikin_effects
from model.positions import positions, filter_dictionary_of_parameter_with_index
from sterne import priors as _priors
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
            The size of shares is 8*N, 8 refers to the 8 parameters ('dec','efac', 'incl',
            'mu_a','mu_d','Om_asc','px','ra' in alphabetic order); N refers to the number
            of pmparins. As an example, for four pmparins, shares can be
            [[0,0,1,1],[0,1,2,2],[0,0,1,1],[0,0,1,1],[0,0,1,1],[0,0,1,1],[0,0,0,0],[0,1,2,3]].
            Same numbers in the same row shares the same parameter (e.g. 'px' is shared by all
            pmparins). Furthermore, if shares[i][j]<0, it means the inference for
            parameter[i] with pmparins[j] is turned off. This turn-off function is not so
            useful now, but may be helpful in future.
        2) iterations : float (default : 200)
            'iterations' that will be passed to bilby.run_sampler().
            Changing "iterations" to over 500 would avoid fuzzy corner plots, while
            "interations"=1000 would make smooth corner plots.
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
            degree of freedom). When pmparin_preliminaries==None, the inference for efac would be
            turned off.
            The error is corrected following the relation:
            errs_new**2 = errs_random**2 + (efac * errs_sys)**2, where errs_random and errs_sys
            stand for random errors and systematic errors, respectively.

    Caveats
    -------
    We assume a1dot is predominantly attributed to the variation of inclination due to 
        the proper motion (Kopeikin, 1996), which is normally valid for pulsars in wide orbits.
        Should there be a remarkable a1dot owing to gravitational-wave damping, the reflex
        motion is normally not prominent (as the orbit is usually compact).

    ** Examples ** :
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
        shares = [list(range(NoP)), [0]*NoP, [0]*NoP, [0]*NoP, [0]*NoP,\
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

        result = bilby.run_sampler(likelihood=likelihood, priors=priors,\
            sampler='emcee', nwalkers=nwalkers, iterations=iterations, outdir=outdir)
    jsonfile = outdir + '/label_result.json' 
    result = bilby.result.read_in_result(filename=jsonfile) 
    result.save_posterior_samples(filename=saved_posteriors)
    make_a_summary_of_bayesian_inference(saved_posteriors, refepoch,\
        list_of_dict_VLBI, list_of_dict_timing)
    result.plot_corner() ## this may fail when run in the background, therefore put in the last

def create_list_of_dict_timing(parfiles):
    from model import reflex_motion
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
    

def make_a_summary_of_bayesian_inference(samplefile, refepoch, list_of_dict_VLBI, list_of_dict_timing):
    dict_median, outputfile = make_a_brief_summary_of_Bayesian_inference(samplefile)
    writefile = open(outputfile, 'a')
    chi_sq, rchsq = calculate_reduced_chi_square(refepoch, list_of_dict_VLBI, list_of_dict_timing, dict_median)
    writefile.write('\nchi-square = %f\nreduced chi-square = %f\n' % (chi_sq, rchsq))
    writefile.write('The reference epoch is MJD %d \n' % refepoch)
    writefile.close()

def make_a_brief_summary_of_Bayesian_inference(samplefile):
    t = Table.read(samplefile, format='ascii')
    parameters = t.colnames[:-2]
    dict_median = {}
    dict_bound = {} #16% and 84% percentiles
    outputfile = samplefile.replace('posterior_samples', 'bayesian_estimates')
    writefile = open(outputfile, 'w')
    writefile.write('#Medians of the simulated samples:\n')
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
    return dict_median, outputfile
    

def calculate_reduced_chi_square(refepoch, list_of_dict_VLBI, list_of_dict_timing, dict_median):
    LoD_VLBI, LoD_timing = list_of_dict_VLBI, list_of_dict_timing
    chi_sq = 0
    NoO = number_of_observations = 0
    for i in range(len(LoD_VLBI)):
        res = LoD_VLBI[i]['radecs'] - positions(refepoch, LoD_VLBI[i]['epochs'], LoD_timing[i], i, dict_median)
        errs_new = adjust_errs_with_efac(LoD_VLBI[i], dict_median, i)
        chi_sq += np.sum((res/errs_new)**2) #if both RA and errRA are weighted by cos(DEC), the weighting is canceled out
        NoO += 2 * len(LoD_VLBI[i]['epochs'])
    DoF = degree_of_freedom = NoO - len(dict_median)
    rchsq = chi_sq / DoF
    return chi_sq, rchsq

def adjust_errs_with_efac(VLBI_dict, parameters_dict, parameter_filter_index):
    FP = filter_dictionary_of_parameter_with_index(parameters_dict, parameter_filter_index)
    Ps = list(FP.keys())
    Ps.sort()
    efac = FP[Ps[1]]
    if efac != -999: 
        errs_new_sq = (VLBI_dict['errs_random'])**2 + (efac * VLBI_dict['errs_sys'])**2
    else: ## if efac is not to be inferred
        errs_new_sq = VLBI_dict['errs']**2
    return errs_new_sq**0.5




class Gaussianlikelihood(bilby.Likelihood):
    def __init__(self, refepoch, list_of_dict_timing, list_of_dict_VLBI, shares, positions, DoD_additional_constraints, a1dot_constraints=False):
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
            errs_new = adjust_errs_with_efac(self.LoD_VLBI[i], self.parameters, i) 
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
    """
    epochs = RAs = errRAs = DECs = errDECs = np.array([])
    lines = open(pmparin).readlines()
    for line in lines:
        #if 'epoch' in line and not line.strip().startswith('#'):
        #    refepoch = line.split('=')[1].strip()
        if line.count(':')==4 and (not line.strip().startswith('#')): 
            epoch, RA, errRA, DEC, errDEC = line.strip().split(' ')
            epoch = decyear2mjd(float(epoch.strip())) #in MJD
            DEC = others.dms2deg(DEC.strip()) #in deg
            DEC *= np.pi/180. #in rad
            RA = others.dms2deg(RA.strip()) #in hr
            RA *= 15*np.pi/180. #in rad
            errRA = float(errRA.strip()) #in s
            errRA *= 15 * np.pi/180./3600. #in rad
            errDEC = float(errDEC.strip()) #in arcsecond
            errDEC *= np.pi/180./3600 #in rad

            epochs = np.append(epochs, epoch)
            RAs = np.append(RAs, RA)
            DECs = np.append(DECs, DEC)
            errRAs = np.append(errRAs, errRA)
            errDECs = np.append(errDECs, errDEC)
    t = Table([epochs, RAs, errRAs, DECs, errDECs], names=['epoch', 'RA', 'errRA', 'DEC', 'errDEC'])
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

