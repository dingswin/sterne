#!/usr/bin/env python
###########################################################################
###-- NOTICE: RUN UNDER pmparesults directories !!!
############################################################################
import os,sys
from optparse import OptionParser
from sterne import simulate

"""
usage = "usage: run under pmparesults directories\n%prog []\n-n --noprepare\n-b --binno\n-h or --help for more"
parser = OptionParser(usage)
parser.add_option("-n", "--noprepare", dest="prepare", default="True",
                  action="store_false",help="NOT running prepare_astrometric_epoch.py beforehand")
parser.add_option("-b", "--binno", dest="binno", default=-1,
                  help="choose only one bin to reduce, input bin number")
(options, junk) = parser.parse_args()
prepare         = options.prepare
binno           = int(options.binno)
"""
usage = "run under pmparesults directories\n%prog + argv[1]"
if len(sys.argv) != 2:
    print(usage)
    sys.exit(1)

junk, parameters = sys.argv
print(parameters)
os.system('rm -rf outdir/*')
exec('simulate.simulate(%s)' % parameters)
