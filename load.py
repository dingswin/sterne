#!/usr/bin/env python
from astropy.time import Time
import numpy as np
from astropy.table import Table
import howfun
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
            RA = howfun.dms2deg(RA.strip()) #in hr
            RA *= 15*np.pi/180. #in rad
            DEC = howfun.dms2deg(DEC.strip()) #in deg
            DEC *= np.pi/180. #in rad
            errRA = float(errRA.strip()) #in s
            errRA *= 1000*15*np.cos(DEC) #in mas
            errDEC = float(errDEC.strip()) #in arcsecond
            errDEC *= 1000 #in mas

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

