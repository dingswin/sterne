#!/usr/bin/env python
"""
sterne.priors.py is written in python3 by Hao Ding.
The main code to run is generate_initsfile().
"""
import bilby, inspect
from astropy.time import Time
import numpy as np
import astropy.units as u
from astropy import constants
import os, sys
import howfun
from astropy.table import Table
import simulate

def generate_initsfile(refepoch, pmparins, shares, HowManySigma=20, **kwargs):
    """
    Used to generate initsfile.
    Common parameters might have more than 1 list of priors.
    In such cases, the larger outer bound will be adopted.

    kwargs : 
        1) incl_prior : list of 2 floats
            [lower_limit, upper_limit] for all inclinations.
    """
    HMS = HowManySigma
    roots = ['dec', 'mu_a', 'mu_d', 'px', 'ra']
    dict_limits = create_dictionary_of_boundaries_with_pmpar(refepoch, pmparins, HowManySigma)
    parameters = simulate.get_parameters_from_shares(shares)
    try:
        incl_prior = kwargs['incl_prior']
    except KeyError:
        incl_prior = [0, 360]
    inits = pmparins[0].replace('.pmpar.in','')
    inits = inits + '.inits'
    writefile = open(inits, 'w')
    writefile.write('#Prior info at MJD %f.\n' % refepoch)
    writefile.write('#%d reduced-chi-squre-corrected sigma limits are used.\n' % HMS)
    writefile.write('#The prior info is based on the pmpar results.\n')
    writefile.write('#Units: dec and ra in rad; px in mas; mu_a and mu_d in mas/yr; incl and om_asc in deg.\n')
    writefile.write('#parameter name explained: dec_0_1, for example, means this dec parameter is inferred for both pmparin0 and pmparin1.\n')
    for parameter in parameters.keys():
        if (not 'om_asc' in parameter) and (not 'incl' in parameter):
            related_pmparins_indice, root = parameter_name_to_pmparin_indice(parameter)
            lower_limit, upper_limit = render_parameter_boundaries(parameter, dict_limits)
            writefile.write('%s: %.11f,%.11f\n' % (parameter, lower_limit, upper_limit))
        elif 'incl' in parameter:
            writefile.write('%s: %f,%f\n' % (parameter, incl_prior[0], incl_prior[1]))
        else:
            writefile.write('%s: 0,360\n' % parameter)
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

    
