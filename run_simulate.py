#!/usr/bin/env python
###########################################################################
###-- NOTICE: RUN UNDER pmparesults directories !!!
############################################################################
import os, sys
import argparse
from sterne import simulate

code_description = "A command-line wrapper for sterne.simulate.simulate(). NOTE to self (others can ignore): Run under pmparesults directories!"
example = "run_simulate.py 57849 J0030+0451.inits J0030+0451.pmpar.in '' -r J0030+0451.pmpar.in.preliminary -s [[0],[0],[-1],[0],[0],[-1],[0],[0]] -i 5000 -n 30;                                       run_simulate.py 57850 J1939+2134.inits J1939+2134.to.IBC01647.pmpar.in '' J1939+2134.to.IBC01648.pmpar.in '' -r J1939+2134.to.IBC01647.pmpar.in.preliminary J1939+2134.to.IBC01648.pmpar.in.preliminary -s [[0,1],[0,0],[-1,-1],[0,0],[0,0],[-1,-1],[0,0],[0,1]] -i 5000 -n 30"

parser = argparse.ArgumentParser(description=code_description, prog='run_simulate.py', usage=example)
parser.add_argument("epoch", metavar="epoch", type=int,
                    help="reference epoch in MJD")
parser.add_argument("priors", metavar="priors", type=str,
                    help="the .inits file")
parser.add_argument("pmparins", metavar="pmparins", type=str, 
                    nargs='+', help='.pmpar.in and parfiles files')
parser.add_argument("-r", "--prelimpmpars", dest="prelimpmpars", 
                    metavar="prelimpmpars", nargs='+', type=str, 
                    help=".pmpar.in.preliminary files")
parser.add_argument("-s", "--shares", dest="shares", type=str, 
                    metavar="shares", help="a 2-D array assigning which parameters to share (see sterne.simulate.simulate)")
parser.add_argument("-i", "--iterations", dest="iterations", type=int, 
                    default=1000, metavar="iterations", help="the depths that random walkers cover")
parser.add_argument("-n", "--nwalkers", dest="nwalkers", type=int, default=30,
                    metavar="nwalkers", help="the number of random walkers")
parser.add_argument("-a", "--a1dot", dest="a1dot_constraints", type=str, 
                    default=False, metavar="a1dot_constraints", 
                    help="a1dot constraints, a list of list of 2 floats. e.g. [[mu, sigma], []], (both in lt-sec/sec), where mu and sigma refers to the Gaussian distribution for a1dot. The length of a1dot_constraint needs to match len(pmparins), unless None.")
parser.add_argument("-o", "--outdir", dest="outdir", type=str, default='outdir',
                    metavar="outdir", help="the output directory")
parser.add_argument("-c", "--clearoutdir", dest="clearoutdir", default='False', 
                    action="store_true", help="clear outdir")

options         = parser.parse_args()
refepoch        = options.epoch
initsfiles      = options.priors
pmparins        = options.pmparins
clearoutdir     = options.clearoutdir

kwargs = {}
kwargs['iterations'] = options.iterations
kwargs['nwalkers'] = options.nwalkers
kwargs['pmparin_preliminaries'] = options.prelimpmpars
kwargs['outdir'] = options.outdir
exec("kwargs['shares'] = %s" % options.shares)
exec("kwargs['a1dot_constraints'] = %s" % options.a1dot_constraints)
print(kwargs)

if clearoutdir:
    print('\ndeleting old files inside %s now.\n' % kwargs['outdir'])
    os.system('rm -rf %s/*' % kwargs['outdir'])

simulate.simulate(refepoch, initsfiles, *pmparins, **kwargs)
