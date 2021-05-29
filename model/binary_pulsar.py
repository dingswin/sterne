#!/usr/bin/env python

#  This file is part of the astrometricfit distribution 
#  (https://github.com/adamdeller/astrometricfit/).
#  Copyright (c) 2018 Adam Deller and Scott Ransom
#  
#  This program is free software: you can redistribute it and/or modify  
#  it under the terms of the GNU General Public License as published by  
#  the Free Software Foundation, version 3.
# 
#  This program is distributed in the hope that it will be useful, but 
#  WITHOUT ANY WARRANTY; without even the implied warranty of 
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU 
#  General Public License for more details.
# 
#  You should have received a copy of the GNU General Public License 
#  along with this program. If not, see <http://www.gnu.org/licenses/>.

import numpy as np
import astropy.coordinates as coords
import astropy.units as u
import astropy.constants as const
import astropy.time as time
from astropy.coordinates.angles import Angle
import matplotlib.pyplot as plt
import novas.compat
import novas.compat.solsys as solsys
from novas.compat.eph_manager import ephem_open
from presto import binary_psr as bp
import os, sys
import yaml
import pickle
import copy

def get_coords(infile="pulsar.pmpar.in"):
    posns = []
    mjds = []
    ra_errs = []
    dec_errs = []
    vlbiepoch = 56800.0
    with open(infile) as f:
        for line in f:
            if len(line.strip()) > 0 and line[0] is not "#":
                sline = line.strip().split()
                if sline[0]=="epoch":
                    vlbiepoch = float(sline[2])
                else:
                    mjds.append(float(sline[0]))
                    posns.append(coords.SkyCoord(sline[1]+" "+sline[3],
                                                 unit=(u.hourangle, u.deg),
                                                 frame='icrs'))
                    ra_errs.append(float(sline[2]) * np.cos(posns[-1].dec))
                    dec_errs.append(float(sline[4]))
    ra_errs = np.asarray(ra_errs) * 15.0 * u.arcsec # in true arcsec
    dec_errs = np.asarray(dec_errs) * u.arcsec
    return vlbiepoch, posns, ra_errs, dec_errs, np.asarray(mjds)

def posn_with_pmpx(ra, dec, pmra, pmdec, px, baryMJD, posepoch,
                   eposn, dRAother=0.0 * u.rad, dDECother=0.0 * u.rad):
    dt = (baryMJD - posepoch) * u.d * u.mas / u.yr
    px = px * u.mas # Parallax in milli-arcsec

    sr, cr = np.sin(ra), np.cos(ra)
    sd, cd = np.sin(dec), np.cos(dec)

    # This could be used for orbital effects (should be Angles)
    dRA = dRAother
    dDEC = dDECother

    # delta position from PM
    dRA += (dt * pmra / cd).decompose()
    dDEC += (dt * pmdec).decompose()

    # This is the Earth position in X, Y, Z (AU) in ICRS wrt SSB
    X, Y, Z = eposn
    # Following is from Astronomical Almanac Explanatory Supplement p 125-126
    dRA += px * (X * sr - Y * cr) / cd
    dDEC += px * (X * cr * sd + Y * sr * sd - Z * cd)
    return ra + dRA, dec + dDEC

def lnlike(theta, baryMJDs, meas, errs, eposns, posepoch, usepsr=False):
    rarad, decrad, pmra, pmdec, px, Omega, inc = theta
    ra = Angle(rarad, unit=u.rad)
    dec = Angle(decrad, unit=u.rad)
    if usepsr:
        # The psr object is present, use it to calculate orbital reflex motion
        psr_dRA, psr_dDEC = usepsr.reflex_motion(baryMJDs, inc, Omega, 1.0/px)
    else:
        # No reflex motion to be accounted for, set to zero
        psr_dRA = np.zeros_like(baryMJDs) * u.rad
        psr_dDEC = np.zeros_like(baryMJDs) * u.rad
    model_vals = [posn_with_pmpx(ra, dec, pmra, pmdec, px, mjd, posepoch,
                                 ep, dRAother=dra, dDECother=ddec) \
                  for mjd, ep, dra, ddec in zip(baryMJDs, eposns,
                                                psr_dRA*u.mas, psr_dDEC*u.mas)]
    ras = np.asarray([x[0].rad for x in model_vals])
    decs = np.asarray([x[1].rad for x in model_vals])
    model = np.concatenate((ras, decs))
    diff = (meas - model)
    diff[:len(ras)] *= np.cos(decs)
    errs2 = np.power(errs, 2)
    ll = -0.5 * np.sum(np.log(2.0*np.pi*errs2) + np.power(diff,2) / errs2)
    return ll

def boundOmegaInc(parameterlist): # parameterlist[-2] = Omega, parameterlist[-1] = inc
    parameterlist[-1] -= 360*int((parameterlist[-1]+90)/360.0)
    if parameterlist[-1] < -90:
        parameterlist[-1] += 360
    if parameterlist[-1] < 0:
        parameterlist[-1] = -parameterlist[-1]
        parameterlist[-2] += 180
    if parameterlist[-1] > 180:
        parameterlist[-1] = 360 -  parameterlist[-1]
        parameterlist[-2] += 180
    parameterlist[-2] -= 360.0*int(parameterlist[-2]/360.0)
    if parameterlist[-2] < 0:
        parameterlist[-2] += 360

def lnprior(theta, initvals, usepsr):
    # Get the limits
    try:
        pmra_limits = initvals['pmra_limits']
        pmdec_limits = initvals['pmdec_limits']
        px_limits = initvals['px_limits']
        Omega_limits = initvals['Omega_limits']
        inc_limits = initvals['inc_limits']
        try:
            if initvals['noincambiguity']:
                otherinc_limits = inc_limits
            else:
                otherinc_limits = [180.0 - inc_limits[1], 180.0 - inc_limits[0]]
        except KeyError:
            otherinc_limits = [180.0 - inc_limits[1], 180.0 - inc_limits[0]]
        dra_mas = float(initvals['ra_bound'])
        ddec_mas = float(initvals['dec_bound'])
    except KeyError as e:
        print "Missing required init range key ", str(e), " - aborting"
        sys.exit()

    # First fix any out of range Omega or inc values (bound them to the range 0,360 and 0,180 respectively)
    orginc = theta[-1]
    orgomega = theta[-2]
    boundOmegaInc(theta)

    # use simple uniform priors for all parameters, 
    # except for inclination where prior can be uniform in cos i if desired
    ra, dec, pmra, pmdec, px, Omega, inc = theta
    lnp = 0.0

    refpos = coords.SkyCoord(initvals['ra'] + " " + initvals['dec'], unit=(u.hourangle, u.deg), frame='icrs')
    ra_limits = [refpos.ra.to(u.rad).value - dra_mas*u.mas/u.rad, \
                 refpos.ra.to(u.rad).value + dra_mas*u.mas/u.rad]
    dec_limits = [refpos.dec.to(u.rad).value - ddec_mas*u.mas/u.rad, \
                  refpos.dec.to(u.rad).value + ddec_mas*u.mas/u.rad]

    # If we're outside the limits, immediately return np.inf
    if not(ra_limits[0] < ra < ra_limits[1] and \
        dec_limits[0] < dec < dec_limits[1] and \
        pmra_limits[0] < pmra < pmra_limits[1] and \
        pmdec_limits[0] < pmdec < pmdec_limits[1] and \
        px_limits[0] < px < px_limits[1] and \
        Omega_limits[0] < Omega < Omega_limits[1] and \
        (inc_limits[0] < inc < inc_limits[1] or \
        otherinc_limits[0] < inc < otherinc_limits[1])):
        return np.inf

    # If the inits file says so, apply a uniform prior on cos i
    try:
        incprior = initvals['incprior']
    except KeyError:
        incprior = "uniformi"
    if incprior == "uniformcosi":
        lnp += np.log(np.fabs(np.cos(inc*np.pi/180.0)))
    elif incprior != "uniformi":
        print "I don't know how to implement inclination prior", incprior
        sys.exit()

    # If xdot is available, use this as a prior on the Omega and inc
    try:
        # We are assuming that the measured xdot has a gaussian uncertainty
        xdot = float(initvals['xdot'])
        xdot_sigma = float(initvals['xdot_sigma'])
        x = usepsr.par.A1
        pmxdot = 1.54e-16*x*(pmdec*np.cos(Omega*np.pi/180.0) - pmra*np.sin(Omega*np.pi/180.0))/np.tan(inc*np.pi/180.0)
        err_sigma = np.fabs(pmxdot - xdot)/xdot_sigma
        lnp += -0.5*(np.log(2*np.pi) + err_sigma*err_sigma)
    except KeyError:
        pass

    return lnp

def lnprob(theta, baryMJDs, meas, errs, eposns, posepoch, usepsr=False, initvals=None):
    lp = lnprior(theta, initvals, usepsr)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, baryMJDs, meas, errs, eposns, posepoch, usepsr)

def sampler_chain_to_dict(names, sampler_chain):
    chains = [sampler_chain[:,:,ii].T for ii in range(len(names))]
    return dict(zip(names,chains))

def plot_chains(chain_dict, file=False):
    np = len(chain_dict)
    fig, axes = plt.subplots(np, 1, sharex=True, figsize=(8, 9))
    for ii, name in enumerate(chain_dict.keys()):
        axes[ii].plot(chain_dict[name], color="k", alpha=0.3)
        axes[ii].set_ylabel(name)
    axes[np-1].set_xlabel("Step Number")
    fig.tight_layout()
    if file:
        fig.savefig(file)
        plt.close()
    else:
        plt.show()
        plt.close()

def make_posn_plots(theta, MJDs, ras, decs, ra_errs, dec_errs, posepoch, psr, filebase=None):
    ra, dec, pmra, pmdec, px, Omega, inc = theta
    refpos = coords.SkyCoord(ra=ra*u.rad, dec=dec*u.rad, frame='icrs')
    modelMJDs = np.linspace(MJDs.min()-20.0, MJDs.max()+20.0, 100)
    model_eposns = [solsys.solarsystem(mjd+2400000.5, 3, 0)[0] \
                        for mjd in modelMJDs]
    model_nopsr = [posn_with_pmpx(ra*u.rad, dec*u.rad, pmra, pmdec, px, mjd, posepoch, ep,
                                  dRAother=0.0*u.rad, 
                                  dDECother=0.0*u.rad)
                   for mjd, ep in zip(modelMJDs, model_eposns)]
    # Compute the reflex motion from the outer orbit
    model_pdRAs, model_pdDECs = psr.reflex_motion(modelMJDs, inc, Omega, 1.0/px)
    pdRAs, pdDECs = psr.reflex_motion(baryMJDs, inc, Omega, 1.0/px)
    model_psr = [posn_with_pmpx(ra*u.rad, dec*u.rad, pmra, pmdec, px, mjd, posepoch, ep,
                                dRAother=dra, dDECother=ddec)
                 for mjd, ep, dra, ddec in \
                     zip(modelMJDs, model_eposns, 
                         model_pdRAs*u.mas, model_pdDECs*u.mas)]
    cosd = np.cos(refpos.dec)
    plt.plot(modelMJDs, [((x[0] - refpos.ra)*cosd).to(u.mas).value 
                         for x in model_nopsr], 'k-', label="No reflex motion")
    plt.plot(modelMJDs, [((x[0] - refpos.ra)*cosd).to(u.mas).value 
                         for x in model_psr], 'r-', label="With reflex motion")
    plt.errorbar(baryMJDs, ((ras - refpos.ra)*cosd).to(u.mas).value,
                 yerr=ra_errs.to(u.mas).value, fmt='.', label="Measurements")
    plt.ylabel("RA - %s (mas)" % \
                   refpos.ra.to_string(unit=u.hour, sep='hms', 
                                       format='latex', pad=True))
    plt.xlabel("MJD")
    #plt.legend(loc=4)
    plt.legend()
    if filebase is not None:
        plt.savefig(filebase+"_RA.png")
        plt.close()
    else:
        plt.show()
    plt.plot(modelMJDs, [(x[1] - refpos.dec).to(u.mas).value 
                         for x in model_nopsr], 'k-', label="No reflex motion")
    plt.plot(modelMJDs, [(x[1] - refpos.dec).to(u.mas).value 
                         for x in model_psr], 'r-', label="With reflex motion")
    plt.errorbar(baryMJDs, (decs - refpos.dec).to(u.mas).value,
                 yerr=dec_errs.to(u.mas).value, fmt='.', label="Measurements")
    plt.ylabel("DEC - %s (mas)" % \
                   refpos.dec.to_string(unit=u.deg, sep='dms', 
                                           format='latex', pad=True))
    plt.xlabel("MJD")
    #plt.legend(loc=1)
    plt.legend()
    if filebase is not None:
        plt.savefig(filebase+"_DEC.png")
        plt.close()
    else:
        plt.show()


def main(parfile, pmparfile, initsfile):
    # Import DE421 for NOVAS
    ephem_open(os.path.join(os.getenv("TEMPO2"), "T2runtime/ephemeris/DE421.1950.2050"))
    
    # Read in the starting values for parameters and their allowed ranges
    initvals = yaml.load(open(initsfile))
    try:
        sourcename = initvals['name']
        mu_a  = float(initvals['pmra'])
        mu_d  = float(initvals['pmdec'])
        px    = float(initvals['px'])
        Omega = float(initvals['Omega'])
        inc   = float(initvals['inc'])
        ra    = initvals['ra']
        dec   = initvals['dec']
        posstring = ra + " " + dec
    except KeyError as e:
        print "Missing required key ", str(e), " in ", initsfile
        sys.exit()
    
    start_pos = coords.SkyCoord(posstring, unit=(u.hourangle, u.deg), frame='icrs')
    
    psr = bp.binary_psr(parfile)
    
    # Get the measurements and convert them to arrays of angles
    vlbiepoch, posns, ra_errs, dec_errs, baryMJDs = get_coords(pmparfile)
    
    # This is the Earth position in X, Y, Z (AU) in ICRS wrt SSB
    eposns = [solsys.solarsystem(mjd+2400000.5, 3, 0)[0] \
              for mjd in baryMJDs]
    
    errs = np.concatenate((ra_errs.to(u.rad).value, dec_errs.to(u.rad).value)) # in radians
    
    ras = np.asarray([pos.ra.rad for pos in posns]) * u.rad
    decs = np.asarray([pos.dec.rad for pos in posns]) * u.rad
    meas = np.concatenate((ras, decs)) # in radians
    
    try:
        pmrasigma  = float(initvals['pmra_sigma'])
        pmdecsigma = float(initvals['pmdec_sigma'])
        pxsigma    = float(initvals['px_sigma'])
        Omegasigma = float(initvals['Omega_sigma'])
        incsigma   = float(initvals['inc_sigma'])
        rasigma    = (float(initvals['ra_sigma'])*u.mas).to(u.rad).value
        decsigma   = (float(initvals['dec_sigma'])*u.mas).to(u.rad).value
        Omega_limits = initvals['Omega_limits']
        inc_limits   = initvals['inc_limits']
        pmra_limits = initvals['pmra_limits']
        pmdec_limits = initvals['pmdec_limits']
        px_limits = initvals['px_limits']
        dra_mas = float(initvals['ra_bound'])
        ddec_mas = float(initvals['dec_bound'])
        try:
            otherinc_limits = inc_limits
            trybothinc = False
            if not initvals['noincambiguity']:
                trybothinc = True
                otherinc_limits = [180.0 - inc_limits[1], 180.0 - inc_limits[0]]
        except KeyError:
            trybothinc = True  
            otherinc_limits = [180.0 - inc_limits[1], 180.0 - inc_limits[0]]
        if Omega_limits[1] < Omega_limits[0]:
            print "Bad value for Omega limits:", Omega_limits, "in inits file", initsfile
        if inc_limits[1] < inc_limits[0]:
            print "Bad value for inclination limits:", inc_limits, "in inits file", initsfile
        if pmra_limits[1] < pmra_limits[0]:
            print "Bad value for pmra limits:", pmra_limits, "in inits file", initsfile
        if pmdec_limits[1] < pmdec_limits[0]:
            print "Bad value for pmdec limits:", pmdec_limits, "in inits file", initsfile
        if px_limits[1] < px_limits[0]:
            print "Bad value for px limits:", px_limits, "in inits file", initsfile
        if px < px_limits[0] or px > px_limits[1]:
            print "Starting value for px outside range!", px, px_limits
        if mu_a < pmra_limits[0] or mu_a > pmra_limits[1]:
            print "Starting value for pmra outside range!", mu_a, pmra_limits
        if mu_d < pmdec_limits[0] or mu_d > pmdec_limits[1]:
            print "Starting value for pmdec outside range!", mu_d, pmdec_limits
        if Omega < Omega_limits[0] or Omega > Omega_limits[1]:
            print "Starting value for Omega outside range!", Omega, Omega_limits
        if (inc > inc_limits[0] and inc < inc_limits[1]) or (trybothinc and inc > otherinc_limits[0] and inc < otherinc_limits[1]):
            pass
        else:
            print "Starting value for inc outside range!", inc, inc_limits
    except KeyError as e:
        print "Missing required key ", str(e), " in ", initsfile
        sys.exit()
    
    
    ndim, nwalkers = 7, 100
    # How to scale starting random offsets
    scales = np.asarray([rasigma*1.5, decsigma*1.5, 
                         pmrasigma*1.5, pmdecsigma*1.5, 
                         pxsigma*1.5, Omegasigma*1.5, incsigma*1.5])
    
    import emcee
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob,
                                    args=(baryMJDs, meas, errs, eposns, vlbiepoch, psr, initvals),
                                    threads=10)
    start = np.asarray([start_pos.ra.rad, start_pos.dec.rad,
                        mu_a, mu_d, px, Omega, inc])
    pos = [start + scales * np.random.randn(ndim) for i in range(nwalkers)]
    
    # Fix any out of bounds Omega values
    for i in range(nwalkers):
        while pos[i][-2] < Omega_limits[0] or pos[i][-2] > Omega_limits[1]:
            if pos[i][-2] + 360 < Omega_limits[1]:
                pos[i][-2] += 360
                continue
            if pos[i][-2] - 360 > Omega_limits[0]:
                pos[i][-2] -= 360
                continue
            pos[i][-2] = Omega + np.random.randn()*Omegasigma
    
    # Fix any out of bounds inc values
    for i in range(nwalkers):
        while (pos[i][-1] < inc_limits[0] or pos[i][-1] > inc_limits[1]) and (pos[i][-1] < otherinc_limits[0] or pos[i][-1] > otherinc_limits[1]):
            pos[i][-1] = inc + np.random.randn()*incsigma
        if trybothinc and i % 2 == 0:
            pos[i][-1] = 180.0 - pos[i][-1] 
    
    saved_sampler_chain = '.sampler_chain.dat'
    if os.path.exists(saved_sampler_chain):
        print('sampler.chain read from .sampler_chain.dat')
        readfile = open(saved_sampler_chain, 'rb')
        sampler_chain = pickle.load(readfile)
        readfile.close()
    else:
        print "About to run MCMC"
        sampler.run_mcmc(pos, 1500)
        print "Finished, getting samples"
        sampler_chain = sampler.chain
        writefile = open(saved_sampler_chain, 'wb')
        pickle.dump(sampler_chain, writefile)
        writefile.close()
        print('sampler.chain saved to .sampler_chain.dat')
        
    samples = sampler_chain[:, 200:, :].reshape((-1, ndim))

    # Put the inclination and Omega back in the range 0,180 and 0,360 respectively
    print "Fixing Omega and inc if necessary"
    for i in range(len(samples)):
        orginc = samples[i][-1]
        orgomega = samples[i][-2]
        boundOmegaInc(samples[i])
        if samples[i][-1] > 180 or samples[i][-1] < 0:
            print "Inc was ", orginc, "is now", samples[i][-1]
        if samples[i][-2] > 360 or samples[i][-2] < 0:
            print "Omega was ", orgomega, "is now", samples[i][-2]
    print "Done fixing"

    
    
    
    import corner
    plt.rc('text', usetex=True)
    variables = ["RA", "DEC", "PMRA", "PMDEC", "PX", "Omega", "Inclination"]
    cornerplot_samples = copy.deepcopy(samples)
    median_RA_rad = np.median(cornerplot_samples[:,0])
    median_DEC_rad = np.median(cornerplot_samples[:,1])
    cornerplot_samples -= [median_RA_rad, median_DEC_rad, 0, 0, 0, 0, 0] #get offset (in rad) of RA/DEC from respective median value
    cornerplot_samples[:, 0:2] *= 180/np.pi*3600*1000  #convert from rad to mas
    cornerplot_samples[:, 0] *= np.cos(median_DEC_rad) 
    cornerplot_variables = [r"$\Delta\alpha$\,(mas)", r"$\Delta\delta$\,(mas)", r"$\mu_{\alpha}\,(\mathrm{mas~{yr}^{-1}})$",\
        r"$\mu_{\delta}\,(\mathrm{mas~{yr}^{-1}})$", r"$\varpi$\,(mas)", r"$\Omega$\,(deg)", r"$i$\,(deg)"]
    fig = corner.corner(cornerplot_samples, labels=cornerplot_variables, label_kwargs={"fontsize": 22})
    fig.tight_layout()
    #fig.subplots_adjust(right=1.5, top=1.5)
    #return fig.get_axes()
    #sys.exit()
    for ax in fig.get_axes():
        ax.tick_params(axis='both', labelsize=14, pad=0)
        #ax.xaxis.labelpad = 100
        
    #fig.savefig(sourcename + "_triangle.pdf", dpi=300, pad_inches=0.3, bbox_inches='tight')
    fig.savefig(sourcename + "_triangle.pdf")
    plt.close()
    transposed = samples.T
    for i, x in enumerate(transposed):
        n, b = np.histogram(x, bins=50, range=[x.min(), x.max()])
        output = open(variables[i] + ".histogram.txt", "w")
        binwidth = b[1] - b[0]
        for na, ba in zip(n, b):
            output.write("%.6f %.6g\n" % (ba+binwidth/2.0, na))
        output.close()
    
    chains = sampler_chain_to_dict(variables, sampler_chain)
    plot_chains(chains, file=sourcename + "_chains.png")
    
    # Print the best MCMC values and ranges
    ranges = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                 zip(*np.percentile(samples, [16, 50, 84], axis=0)))
    print "Post-MCMC values (50th percentile +/- (16th/84th percentile):"
    for name, vals in zip(variables, ranges):
        print "%8s:"%name, "%25.15g (+ %12.5g  / - %12.5g)"%vals
    
    sys.exit()
    # Make plots of the highest-likelihood sample
    besttheta = sampler.flatchain[sampler.flatlnprobability.argmax()]
    print "Best theta:", besttheta
    bestposn = coords.SkyCoord(ra=besttheta[0]*u.rad, dec=besttheta[1]*u.rad, frame='icrs')
    print "Position at epoch %.1f is:" % vlbiepoch, \
        bestposn.ra.to_string(pad=True, sep='hms', unit='hour', precision=6), \
        bestposn.dec.to_string(pad=True, sep='dms', unit='deg') 
    make_posn_plots(besttheta, baryMJDs, ras, decs, ra_errs, dec_errs, 
                    vlbiepoch, psr, sourcename+"_best")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print "Usage: %s <timing .par file> <VLBI pmpar.in file> <initial values + ranges file>" % sys.argv[0]
        sys.exit()

    parfile = sys.argv[1]
    pmparfile = sys.argv[2]
    initsfile = sys.argv[3]
    main(parfile, pmparfile, initsfile)
