import numpy as np
import time
import matplotlib.pyplot as plt

import BreakupUtilities as util

plt.close('all')


###############################################################################
# DEFINE SIMULATION PARAMETERS
#
# Use meters, kilograms, seconds, radians for all units
###############################################################################


# # Sun-Synchronous Orbit (Envisat)
# SMA_meters = 7143.4*1000.
# ECC = 0.0001427
# INC_rad = 98.1478*np.pi/180.
# RAAN_rad = 224.8380*np.pi/180.
# AOP_rad = 93.2798*np.pi/180.
# TA_rad = 0.*np.pi/180.

# # Spacecraft drag parameters
# mass = 8211.
# area_m2 = 85.76
# Cd = 2.2


# Starlink-1294
SMA_meters = 6924.7136*1000.
ECC = 0.000862984089
INC_rad = 52.925049*np.pi/180.
RAAN_rad = 268.072247*np.pi/180.
AOP_rad = 102.174332*np.pi/180.
TA_rad = 208.291901*np.pi/180.

# Spacecraft drag parameters
mass = 260.
area_m2 = 17.5
Cd = 2.2


# Propagation time in seconds
tfinal_sec = 86400.*365.25*10


###############################################################################
# Setup and Run SALT Propagator
###############################################################################

# Propagation time
tin = np.array([0., tfinal_sec])
    
# Propagator parameters
params = {}
params['Nquad'] = 20
params['area'] = area_m2
params['mass'] = mass
params['Cd'] = Cd
LEO_flag = True

# Run SALT propagator
start_time = time.time()
kep = np.array([SMA_meters, ECC, INC_rad, RAAN_rad, AOP_rad])
output_dict = util.long_term_propagator(tin, kep, params, LEO_flag)

print('Propagation Time [sec]:', time.time() - start_time)

# Retrieve output for plots
tsec = output_dict['tsec']
yout = np.zeros((len(tsec), 5))
yout[:,0] = output_dict['SMA']
yout[:,1] = output_dict['ECC']
yout[:,2] = output_dict['INC']
yout[:,3] = output_dict['RAAN']
yout[:,4] = output_dict['AOP']


# Plot output
util.plot_elements(tsec, yout)











