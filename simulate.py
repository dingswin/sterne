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
import howfun
from astropy.table import Table
from model import reflex_motion, kopeikin_effects
from model.positions import positions
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
            (default : [[0]*N,list(range(N)),[0]*N,[0]*N,[0]*N,[0]*N,[0]*N,list(range(N))]) 
            Used to assign shared parameters to fit and which paramters to not fit.
            The size of shares is 8*N, 8 refers to the 8 parameters ('a1dot', 'dec','incl',
            'mu_a','mu_d','Om_asc','px','ra' in alphabetic order); N refers to the number
            of pmparins. As an example, for four pmparins, shares can be
            [[0,0,1,1], [0,1,2,2],[0,0,1,1],[0,0,1,1],[0,0,1,1],[0,0,1,1],[0,0,0,0],[0,1,2,3]].
            Same numbers in the same row shares the same parameter (e.g. 'px' is shared by all
            pmparins). Furthermore, if shares[i][j]<0, it means the inference for
            parameter[i] with pmparins[j] is turned off. This turn-off function is not so
            useful now, but may be helpful in future.
        2) iterations : float (default : 100)
            'iterations' that will be passed to bilby.run_sampler().
            Changing "iterations" to over 500 would avoid fuzzy corner plots, while
            "interations"=1000 would make smooth corner plots.
        3) nwalkers : float
            'nwalkers' that will be passed to bilby.run_sampler().
        4) outdir : float
            'outdir' that will be passed to run_sampler().
        5) use_saved_samples : bool
            If True, run_sampler will be bypassed.

    Caveats
    -------
    We assume a1dot is predominantly attributed to the variation of inclination due to 
        the proper motion (Kopeikin, 1996), which is normally valid for pulsars in wide orbits.
        Should there be a remarkable a1dot owing to gravitational-wave damping, the reflex
        motion is normally not prominent (as the orbit is usually compact).

    ** Examples ** :
        1) For two pulsars in a globular cluster:
            simulate(57444,'a.inits','p1.pmpar.in','','p2.pmpar.in','p2.par',shares=[[-1,0],[0,1],
                [-1,0],[0,1], [0,1],[-1,0],[0,0],[0,1]])
        2) For a pulsar with two in-beam calibrators:
            simulate(57444,'a.inits','i1.pmpar.in','p.par','i2.pmpar.in','p.par',
                shares=[[0,0],[0,1],[0,0],[0,0],[0,0],[0,0],[0,0],[0,1]])
        3) For two pulsars in a globular cluster sharing an in-beam calibrator:
            simulate(57444,'a.inits','i1p1.pmpar.in', '', 'i2p1.pmpar.in','', 'i1p2.pmpar.in',
                'p2.par','i2p2.pmpar.in','p2.par',shares=[[-1,-1,0,0],[1,2,3,4],[-1,-1,0,0],
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
    try:
        shares = kwargs['shares']
    except KeyError:
        shares = [[0]*NoP, list(range(NoP)), [0]*NoP, [0]*NoP, [0]*NoP, [0]*NoP,\
            [0]*NoP, list(range(NoP))]
    
    print(pmparins, parfiles, initsfile, shares)
    ##############################################################
    #################  get two list_of_dict ######################
    ##############################################################
    list_of_dict_timing = []
    for parfile in parfiles:
        if parfile != '':
            dict_of_timing_parameters = reflex_motion.read_parfile(parfile)
        else:
            dict_of_timing_parameters = {}
        list_of_dict_timing.append(dict_of_timing_parameters)
    print(list_of_dict_timing)

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
    print(list_of_dict_VLBI)
    
    ##############################################################
    ###################### run simulations #######################
    ##############################################################
    try:
        outdir = kwargs['outdir']
    except KeyError:
        outdir = 'outdir'
    saved_posteriors = outdir + '/posterior_samples.dat'
    try:
        use_saved_samples = kwargs['use_saved_samples']
    except KeyError:
        use_saved_samples = False
    if not use_saved_samples:
        likelihood = Gaussianlikelihood(refepoch, list_of_dict_timing, list_of_dict_VLBI,\
            shares, positions)
        #initsfile = pmparins[0].replace('pmpar.in', 'inits')
        #if not os.path.exists(initsfile):
        #    generate_initsfile(pmparin, refepoch)
        #generate_initsfile(refepoch, pmparins, parfiles, shares, 20)
        limits = read_inits(initsfile)
        print(limits)
        priors = {}
        for parameter in limits.keys():
            if limits[parameter][2] == 'Uniform':
                priors[parameter] = bilby.core.prior.Uniform(minimum=limits[parameter][0],\
                    maximum=limits[parameter][1], name=parameter, latex_label=parameter)
            elif limits[parameter][2] == 'Gaussian':
                priors[parameter] = bilby.prior.Gaussian(mu=limits[parameter][0],\
                    sigma=limits[parameter][1], name=parameter, latex_label=parameter)
            elif limits[parameter][2] == 'Sine':
                priors[parameter] = bilby.prior.Sine(minimum=limits[parameter][0],\
                    maximum=limits[parameter][1], name=parameter, latex_label=parameter)
        try:
            iterations = kwargs['iterations']
        except KeyError:
            iterations = 100
        try:
            nwalkers = kwargs['nwalkers']
        except KeyError:
            nwalkers = 100
        result = bilby.run_sampler(likelihood=likelihood, priors=priors,\
            sampler='emcee', nwalkers=nwalkers, iterations=iterations, outdir=outdir)
        result.plot_corner()
        result.save_posterior_samples(filename=saved_posteriors)
    
    make_a_summary_of_bayesian_inference(saved_posteriors, refepoch,\
        list_of_dict_VLBI, list_of_dict_timing)


def make_a_summary_of_bayesian_inference(samplefile, refepoch, list_of_dict_VLBI, list_of_dict_timing):
    t = Table.read(samplefile, format='ascii')
    parameters = t.colnames[:-2]
    dict_median = {}
    dict_bound = {} #16% and 84% percentiles
    outputfile = samplefile.replace('posterior_samples', 'bayesian_estimates')
    writefile = open(outputfile, 'w')
    writefile.write('#Medians of the simulated samples:\n')
    writefile.write('#(Units: px in mas; ra and dec in rad; mu_a and mu_d in mas/yr; om_asc in deg and incl in rad.)\n')
    for p in parameters:
        dict_median[p] = howfun.sample2median(t[p])
        dict_bound[p] = howfun.sample2median_range(t[p], 1)
        writefile.write('%s = %.18f + %.18f - %.18f\n' % (p, dict_median[p],\
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

    chi_sq, rchsq = calculate_reduced_chi_square(refepoch, list_of_dict_VLBI, list_of_dict_timing, dict_median)
    writefile.write('\nchi-square = %f\nreduced chi-square = %f\n' % (chi_sq, rchsq))
    writefile.close()
def calculate_reduced_chi_square(refepoch, list_of_dict_VLBI, list_of_dict_timing, dict_median):
    LoD_VLBI, LoD_timing = list_of_dict_VLBI, list_of_dict_timing
    chi_sq = 0
    NoO = number_of_observations = 0
    for i in range(len(LoD_VLBI)):
        res = LoD_VLBI[i]['radecs'] - positions(refepoch, LoD_VLBI[i]['epochs'], LoD_timing[i], i, dict_median)
        chi_sq += np.sum((res/LoD_VLBI[i]['errs'])**2) #if both RA and errRA are weighted by cos(DEC), the weighting is canceled out
        NoO += 2 * len(LoD_VLBI[i]['epochs'])
    DoF = degree_of_freedom = NoO - len(dict_median)
    rchsq = chi_sq / DoF
    return chi_sq, rchsq


def read_inits(initsfile):
    #initsfile = pmparin.replace('pmpar.in', 'inits')
    readfile = open(initsfile, 'r')
    lines = readfile.readlines()
    readfile.close()
    dict_limits = {}
    for line in lines:
        if not line.startswith('#'):
            for keyword in ['a1dot', 'ra', 'dec', 'mu_a', 'mu_d', 'px', 'incl', 'om_asc']:
                if keyword in line:
                    parameter = line.split(':')[0].strip()
                    limits = line.split(':')[-1].strip().split(',')
                    limits = [limit.strip() for limit in limits]
                    limits[0] = float(limits[0])
                    limits[1] = float(limits[1])
                    dict_limits[parameter] = limits
    return dict_limits 




class Gaussianlikelihood(bilby.Likelihood):
    def __init__(self, refepoch, list_of_dict_timing, list_of_dict_VLBI, shares, positions):
        """
        Addition of multiple Gaussian likelihoods

        Parameters
        ----------
        data: array_like
            The data to analyse
        """
        self.refepoch = refepoch
        self.LoD_VLBI = list_of_dict_VLBI
        self.LoD_timing = list_of_dict_timing
        self.positions = positions
        self.shares = shares
        self.number_of_pmparins = len(self.LoD_VLBI)

        parameters = get_parameters_from_shares(self.shares)
        print(parameters)
        super().__init__(parameters)

    def log_likelihood(self):
        """
        the name has to be log_likelihood, and the PDF has to do the log calculation.
        """
        log_p = 0
        for i in range(self.number_of_pmparins):
            res = self.LoD_VLBI[i]['radecs'] - self.positions(self.refepoch, self.LoD_VLBI[i]['epochs'], self.LoD_timing[i], i, self.parameters)
            log_p += -0.5 * np.sum((res/self.LoD_VLBI[i]['errs'])**2) #if both RA and errRA are weighted by cos(DEC), the weighting is canceled out
            
        ETRA = equivalent_total_res_a1dot = kopeikin_effects.calculate_equivalent_total_res_a1dot(self.LoD_timing, self.parameters)
        log_p += -0.5 * ETRA**2
        return log_p
    

def get_parameters_from_shares(shares):
    parameters = {}
    roots = parameter_roots = ['a1dot', 'dec', 'incl', 'mu_a', 'mu_d', 'om_asc', 'px', 'ra']
    NoP = number_of_pmparins = len(shares[0])
    for i in range(len(roots)):
        list_of_strings = group_elements_by_same_values(shares[i])
        for string in list_of_strings:
            parameter = roots[i] + string
            parameters[parameter] = None
    return parameters
def group_elements_by_same_values(alist):
    """
    Input parameters
    ----------------
    alist : list of int

    Return parameters
    -----------------
    
    """
    LoS = list_of_string = []
    alist = np.array(alist)
    GEI = grouped_element_indice = []
    N = len(alist)
    for i in range(len(alist)):
        if (not i in GEI) and (alist[i]>=0):
            each_group = np.where(alist==alist[i])[0]
            each_group = each_group.tolist()
            GEI += each_group
            each_group.sort()
            each_group = [str(element) for element in each_group]
            str_of_group = '_' + '_'.join(each_group)
            LoS.append(str_of_group)
    return LoS





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
            DEC = howfun.dms2deg(DEC.strip()) #in deg
            DEC *= np.pi/180. #in rad
            RA = howfun.dms2deg(RA.strip()) #in hr
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

