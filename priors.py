#!/usr/bin/env python
"""
sterne.priors.py is written in python3 by Hao Ding.
The main code to run is generate_initsfile().
"""
import numpy as np
import astropy.units as u
from astropy import constants
import os, sys
import howfun
import simulate
from bilby.core.prior.base import Prior

def generate_initsfile(refepoch, pmparins, shares, HowManySigma=20, **kwargs):
    """
    Used to generate initsfile describing Uniform prior distribution of parameters.
    One has to mannually change the priors in the produced initsfile to mu,sigma,Gaussian, if 
    Gaussian distribution is requested.
    Common parameters might have more than 1 list of priors.
    In such cases, the larger outer bound will be adopted.

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
            [lower_limit, upper_limit] for all inclinations.
        2) a1dot_prior : list of 2 floats
            [lower_limit, upper_limit] for all derivatives of projected semi-major axis
        3) om_asc_prior : list of 2 floats
            [lower_limit, upper_limit] for all orbit ascending node longitudes.
    """
    HMS = HowManySigma
    roots = ['dec', 'mu_a', 'mu_d', 'px', 'ra']
    dict_limits = create_dictionary_of_boundaries_with_pmpar(refepoch, pmparins, HowManySigma)
    parameters = simulate.get_parameters_from_shares(shares)
    try:
        incl_prior = kwargs['incl_prior']
    except KeyError:
        incl_prior = [0, 3.141592653589793]
    try:
        om_asc_prior = kwargs['om_asc_prior']
    except KeyError:
        om_asc_prior = [0, 360]
    try:
        a1dot_prior = kwargs['a1dot_prior']
    except KeyError:
        a1dot_prior = [0, 1]
    inits = pmparins[0].replace('.pmpar.in','')
    inits = inits + '.inits'
    writefile = open(inits, 'w')
    writefile.write('#Prior info at MJD %f.\n' % refepoch)
    writefile.write('#%d reduced-chi-squre-corrected sigma limits are used.\n' % HMS)
    writefile.write('#If Uniform or Sine distribution is requested, then the two values stand for lower and upper limit.\n')
    writefile.write('#If Gaussian distribution is requested, then the two values stand for mu and sigma.\n')
    writefile.write('#The unit of a1dot is 1e20 ls-sec/sec.\n')
    writefile.write('#The Uniform prior info is based on the pmpar results.\n')
    writefile.write('#Units: dec and ra in rad; px in mas; mu_a and mu_d in mas/yr; incl in rad; om_asc in deg.\n')
    writefile.write('#parameter name explained: dec_0_1, for example, means this dec parameter is inferred for both pmparin0 and pmparin1.\n')
    for parameter in parameters.keys():
        if (not 'om_asc' in parameter) and (not 'incl' in parameter) and (not 'a1dot' in parameter):
            related_pmparins_indice, root = parameter_name_to_pmparin_indice(parameter)
            lower_limit, upper_limit = render_parameter_boundaries(parameter, dict_limits)
            writefile.write('%s: %.11f,%.11f,Uniform\n' % (parameter, lower_limit, upper_limit))
        elif 'incl' in parameter:
            writefile.write('%s: %.11f,%.11f,Sine\n' % (parameter, incl_prior[0], incl_prior[1]))
        elif 'a1dot' in parameter:
            writefile.write('%s: %f,%f,Gaussian\n' % (parameter, 1e20*a1dot_prior[0], 1e20*a1dot_prior[1]))
        else: ## om_asc
            writefile.write('%s: %f,%f,Uniform\n' % (parameter, om_asc_prior[0], om_asc_prior[1]))
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
            RA = howfun.dms2deg(RA)
        if 'Dec  ' in line:
            Dec = line.split('=')[-1].split('+')[0].strip()
            Dec = howfun.dms2deg(Dec)

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

class Sine_deg(Prior):
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
