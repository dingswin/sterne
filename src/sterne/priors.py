#!/usr/bin/env python
"""
sterne.priors.py is written in python3 by Hao Ding.
The main code to run is generate_initsfile().
"""
import numpy as np
import astropy.units as u
from astropy import constants
import os, sys
import others
import simulate
import bilby
from bilby.core.prior.base import Prior
from bilby.core.prior import PriorDict

def generate_initsfile(refepoch, pmparins, shares, HowManySigma=20, **kwargs):
    """
    Used to generate initsfile describing Uniform prior distribution of parameters.
    One has to mannually change the priors in the produced initsfile to mu,sigma,Gaussian, if 
    Gaussian distribution is requested.
    Common parameters might have more than 1 list of priors.
    In such cases, the larger outer bound will be adopted.

    Notice
    ------
    Additional inclination constraints should be added into the initsfile in the same incl line after the likelihood distribution declaration.

    Input parameters
    ----------------
    refepoch : float
        Reference epoch.
    pmparins : list of str
        List of pmparin files, e.g., ['J2222-0137.pmpar.in'].
    shares : 2-D array
        A 8*N 2-D array detailing which fitted paramters are commonly used by which pmparins.
        See the docstring for simulate.simulate() for more details.

    kwargs : 
        1) incl_prior : list of 2 floats
            in rad. e.g. [lower_limit, upper_limit] for all inclinations.
            ## In future, it will be extended to list of list of 2 floats
            ## e.g. [[lower_limit, upper_limit]] for orbital inclinations.
        2) om_asc_prior : list of 2 floats
            in deg. e.g. [lower_limit, upper_limit] for all orbit ascending node longitudes.
            ## in future, it will be extended to 2D array.
        3) efac_prior : list of 2 floats (default : [0,15])
            EFAC is used to find appropriate systematics following the relation:
            err_new**2 = err_random**2 + (EFAC * err_sys_old)**2.
            Here, the EFAC_prior would apply to all EFACs.
    """
    if type(pmparins) != list:
        print('pmparins has to be a list. Exiting for now.')
        sys.exit(1)
    HMS = HowManySigma
    roots = ['dec', 'mu_a', 'mu_d', 'px', 'ra']
    dict_limits = create_dictionary_of_boundaries_with_pmpar(refepoch, pmparins, HowManySigma)
    parameters = get_parameters_from_shares(shares)
    try:
        incl_prior = kwargs['incl_prior']
    except KeyError:
        incl_prior = [0, 3.141592653589793]
    try:
        om_asc_prior = kwargs['om_asc_prior']
    except KeyError:
        om_asc_prior = [0, 360]
    try:
        efac_prior = kwargs['efac_prior']
    except KeyError:
        efac_prior = [0, 15]
    inits = pmparins[0].replace('.pmpar.in','')
    inits = inits + '.inits'
    writefile = open(inits, 'w')
    writefile.write('#Prior info at MJD %f.\n' % refepoch)
    writefile.write('#%d reduced-chi-squre-corrected sigma limits are used.\n' % HMS)
    writefile.write('#If Uniform or Sine distribution is requested, then the two values stand for lower and upper limit.\n')
    writefile.write('#If Gaussian distribution is requested, then the two values stand for mu and sigma.\n')
    writefile.write('#The Uniform prior info is based on the pmpar results.\n')
    writefile.write('#Units: dec and ra in rad; px in mas; mu_a and mu_d in mas/yr; incl in rad; om_asc in deg.\n')
    writefile.write('#parameter name explained: dec_0_1, for example, means this dec parameter is inferred for both pmparin0 and pmparin1.\n')
    for parameter in parameters.keys():
        if not (('om_asc' in parameter) or ('incl' in parameter) or ('efac' in parameter)):
            related_pmparins_indice, root = parameter_name_to_pmparin_indice(parameter)
            lower_limit, upper_limit = render_parameter_boundaries(parameter, dict_limits)
            writefile.write('%s: %.11f,%.11f,Uniform\n' % (parameter, lower_limit, upper_limit))
        elif 'incl' in parameter:
            writefile.write('%s: %.11f,%.11f,Sine\n' % (parameter, incl_prior[0], incl_prior[1]))
        elif 'om_asc' in parameter: 
            writefile.write('%s: %f,%f,Uniform\n' % (parameter, om_asc_prior[0], om_asc_prior[1]))
        elif 'efac' in parameter:
            writefile.write('%s: %f,%f,Uniform\n' % (parameter, efac_prior[0], efac_prior[1]))
        else:
            pass
    writefile.close()

def create_dictionary_of_boundaries_with_pmpar(refepoch, pmparins, HowManySigma=20):
    """
    do not cover 'incl' and 'om_asc'.
    """
    HMS = HowManySigma
    roots = ['dec', 'mu_a', 'mu_d', 'px', 'ra']
    dec_lows, dec_ups, mu_a_lows, mu_a_ups, mu_d_lows, mu_d_ups,\
        px_lows, px_ups, ra_lows, ra_ups = [], [], [], [], [],\
        [],[],[],[],[]
    for i in range(len(pmparins)):
        pmparout = pmparins[i].replace('pmpar.in','pmpar.out')
        replace_pmparin_refepoch(pmparins[i], refepoch)
        os.system("pmpar %s > %s" % (pmparins[i], pmparout))
        [ra, error_ra, dec, error_dec, mu_a, error_mu_a, mu_d, error_mu_d, px, error_px, rchsq, junk] = readpmparout(pmparout)
        errors = np.array([error_ra, error_dec, error_mu_a, error_mu_d, error_px])
        print(errors, rchsq)
        errors *= rchsq**0.5
        print(errors)
        error_ra, error_dec, error_mu_a, error_mu_d, error_px = errors
        dec_lows.append(dec - HMS * error_dec)
        dec_ups.append(dec + HMS * error_dec)
        mu_a_lows.append(mu_a - HMS * error_mu_a)
        mu_a_ups.append(mu_a + HMS * error_mu_a)
        mu_d_lows.append(mu_d - HMS * error_mu_d)
        mu_d_ups.append(mu_d + HMS * error_mu_d)
        px_lows.append(px - HMS * error_px)
        px_ups.append(px + HMS * error_px)
        ra_lows.append(ra - HMS * error_ra)
        ra_ups.append(ra + HMS * error_ra)
    dec_limits, mu_a_limits, mu_d_limits, px_limits, ra_limits = {},{},{},{},{}
    dec_limits['low'], dec_limits['up'] = dec_lows, dec_ups
    mu_a_limits['low'], mu_a_limits['up'] = mu_a_lows, mu_a_ups
    mu_d_limits['low'], mu_d_limits['up'] = mu_d_lows, mu_d_ups
    px_limits['low'], px_limits['up'] = px_lows, px_ups
    ra_limits['low'], ra_limits['up'] = ra_lows, ra_ups
    dict_limits = {}
    for root in roots:
        exec("dict_limits[root] = %s_limits" % root)
    return dict_limits
    

def render_parameter_boundaries(parameter, dict_limits):
    """
    use the minimum and maximum value for parameters calculated from the relevant pmparins
    """
    related_pmparins_indice, root = parameter_name_to_pmparin_indice(parameter)
    lows, ups = [], []
    for i in related_pmparins_indice:
        lows.append(dict_limits[root]['low'][i])
        ups.append(dict_limits[root]['up'][i])
    lower_limit, upper_limit = min(lows), max(ups)
    return lower_limit, upper_limit
    
def parameter_name_to_pmparin_indice(string):
    """
    Example :
        input 'mu_a_0_2_3' --> out: ([0,2,3], 'mu_a')
    """
    alist = string.split('_')
    pmparin_indice = []
    parameter_root = []
    for element in alist:
        try:
            pmparin_indice.append(int(element))
        except ValueError:
            parameter_root.append(element)
    parameter_root = '_'.join(parameter_root)
    return pmparin_indice, parameter_root

def readpmparout(pmparout):
    """
    The function serves to offer priors for the simulation.
    """
    rchsq = 0
    lines = open(pmparout).readlines()
    for line in lines:
        if 'epoch' in line:
            epoch = line.split('=')[1].strip()
        if 'Reduced' in line:
            rchsq = float(line.split('=')[1].strip())
        for estimate in ['mu_a', 'mu_d', 'pi']:
            if estimate in line:
                #globals()['line'] = line
                #string = ("%s = " % estimate.strip())
                value = line.split('=')[-1].split('+')[0].strip()
                #print(value)
                #print(estimate.strip())
                exec("%s='%s'" % (estimate.strip(), value), globals())
        if 'RA' in line: #here, due to a bug in exec(), it is not combined with the other three parameters
            RA = line.split('=')[-1].split('+')[0].strip()
            RA = others.dms2deg(RA)
        if 'Dec  ' in line:
            Dec = line.split('=')[-1].split('+')[0].strip()
            Dec = others.dms2deg(Dec)

    for line in lines:
        if 'RA' in line:
            error_RA = float(line.split('+-')[1].strip().split(' ')[0])
        if 'Dec  ' in line:
            error_Dec = float(line.split('+-')[1].strip().split(' ')[0])
        for estimate in ['mu_a', 'mu_d', 'pi']:
            if estimate in line:
                error = line.split('+-')[1].strip().split(' ')[0]
                exec("error_%s = %s" % (estimate.strip(), error), globals())
                #exec("print(error_%s)" % estimate)
                exec("%s = float(%s)" % (estimate.strip(), estimate.strip()), globals())
    RA *= 15 * np.pi/180. #rad
    Dec *= np.pi/180. #rad
    error_RA *= 15 * np.pi/180./3600. #rad
    error_Dec *= np.pi/180./3600. #rad
    return RA, error_RA, Dec, error_Dec, mu_a, error_mu_a, mu_d, error_mu_d, pi, error_pi, rchsq, float(epoch)
    


def replace_pmparin_refepoch(pmparin, refepoch):
    """
    refepoch in MJD
    """
    readfile = open(pmparin, 'r')
    lines = readfile.readlines()
    for i in range(len(lines)):
        if 'epoch' in lines[i] and (not lines[i].strip().startswith('#')):
            lines[i] = 'epoch = ' + str(refepoch) + '\n'
    readfile.close()
    writefile = open(pmparin, 'w')
    writefile.writelines(lines)
    writefile.close()

class __Sine_deg(Prior):
    """
    Note
    ----
    P(i) ~ sin(i), i.e., probability density proportional to sin(i).
    Edited from bilby.core.prior.base.Prior.Sine() (which offers more docstrings).
    The only difference (see difference 1, 2, 3, 4) is the unit is deg instead of rad.
    
    Reference
    ---------
    Ashton et al. 2019
    """
    def __init__(self,  minimum=0, maximum=180, name=None,\
            latex_label=None, unit=None, boundary=None): ## difference 1
        super(Sine_deg, self).__init__(name=name, latex_label=latex_label, unit=unit,
                                   minimum=minimum, maximum=maximum, boundary=boundary)
        self.maximum *= np.pi/180. ## difference 2
        self.minimum *= np.pi/180. ## difference 3

    def rescale(self, val):
        self.test_valid_for_rescaling(val)
        norm = 1 / (np.cos(self.minimum) - np.cos(self.maximum))
        return 180./np.pi*np.arccos(np.cos(self.minimum) - val / norm) ## difference 4

    def prob(self, val):
        return np.sin(val) / 2 * self.is_in_prior_range(val)

    def cdf(self, val):
        _cdf = np.atleast_1d((np.cos(val) - np.cos(self.minimum)) /
                             (np.cos(self.maximum) - np.cos(self.minimum)))
        _cdf[val > self.maximum] = 1
        _cdf[val < self.minimum] = 0
        return _cdf

def get_parameters_from_shares(shares):
    parameters = {}
    roots = parameter_roots = ['dec', 'efac', 'incl', 'mu_a', 'mu_d', 'om_asc', 'px', 'ra']
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


def read_inits(initsfile):
    """
    the additional constraints should be given in the same parameter line after the likelihood distribution requests.

    Output
    ------
    DoD_additional_constraints : dict of 2 dict
    """
    #initsfile = pmparin.replace('pmpar.in', 'inits')
    readfile = open(initsfile, 'r')
    lines = readfile.readlines()
    readfile.close()
    dict_limits = {}
    DoD_additional_constraints = dict_of_dict_additional_constraints = {}
    DoD_additional_constraints['sin_incl_Gaussian_constraints'] = {}
    DoD_additional_constraints['sin_incl_limits_constraints'] = {}
    for line in lines:
        if not line.startswith('#'):
            for keyword in ['ra', 'dec', 'mu_a', 'mu_d', 'px', 'incl', 'om_asc', 'efac']:
                if keyword in line:
                    parameter = line.split(':')[0].strip()
                    limits = line.split(':')[-1].strip().split(',')
                    limits = [limit.strip() for limit in limits]
                    limits[0] = float(limits[0])
                    limits[1] = float(limits[1])
                    dict_limits[parameter] = limits[:3]
                    
                    try:
                        limits[3] = float(limits[3]) ## additional constraints
                        limits[4] = float(limits[4])
                        if (limits[5] == 'Sine_Gaussian') and ('incl' in parameter):
                            DoD_additional_constraints['sin_incl_Gaussian_constraints'][parameter] = limits[3:5]
                        elif (limits[5] == 'Sine_limits') and ('incl' in parameter):
                            DoD_additional_constraints['sin_incl_limits_constraints'][parameter] = limits[3:5]
                    except IndexError:
                        pass
    return dict_limits, DoD_additional_constraints


def create_priors_given_limits_dict(limits, DoD_additional_constraints):
    SILC = sin_incl_limits_constraints = DoD_additional_constraints['sin_incl_limits_constraints']
    if len(SILC) == 0:
        priors = PriorDict()
    else:
        APC = additional_prior_constraints(DoD_additional_constraints)
        priors = PriorDict(conversion_function=APC.sin_incl_limits)
        for parameter in SILC:
            parameter_virtual = parameter.replace('incl', 'sin_incl')
            sin_incl_min, sin_incl_max = SILC[parameter][:2]
            priors[parameter_virtual] = bilby.core.prior.Constraint(minimum=sin_incl_min, maximum=sin_incl_max)

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
    return priors

class additional_prior_constraints:
    def __init__(self, DoD_additional_constraints):
        self.SILC = sin_incl_limits_constraints = DoD_additional_constraints['sin_incl_limits_constraints']
    def sin_incl_limits(self, dict_parameters):
        dictPs = dict_parameters.copy()
        if len(self.SILC) != 0:
            for parameter in self.SILC:
                parameter_virtual = parameter.replace('incl', 'sin_incl')
                dictPs[parameter_virtual] = np.sin(dictPs[parameter])
        return dictPs
