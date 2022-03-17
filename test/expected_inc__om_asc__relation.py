#!/usr/bin/env python
import numpy as np
import astropy.units as u
from astropy import constants
from matplotlib import pyplot as plt

a1 = 21.13 * u.rad
a1dot = -1.7e-14 * u.rad/u.s
mu_a = 0.50 * u.mas/u.yr
mu_d = -6.85 * u.mas/u.yr
mu = (mu_a**2+mu_d**2)**0.5
theta_mu = np.arctan2(mu_a,mu_d)
oms = np.arange(0,360,0.1)
oms *= u.deg
cotis = a1dot / a1 / mu / np.sin(theta_mu-oms)
cotincs = cotis.to(1/u.rad).value
print(cotincs)
incs = np.arctan(1./cotincs)
incs *= 180. / np.pi
plt.plot(oms, incs)
plt.xlabel('om_asc (deg)')
plt.ylabel('inclination (deg)')
plt.savefig('om_inc_relation.pdf')
plt.clf()
