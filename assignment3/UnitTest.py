import numpy as np
import os
from datetime import datetime
import time
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
from pathlib import Path
from matplotlib import style

import EstimationUtilities as EstUtil
import TudatPropagator as prop
import ConjunctionUtilities as ConjUtil

###############################################################################
# Basic I/O
###############################################################################

main_directory = Path(__file__).parent.parent  # A3G5-space-debris-2025 directory (root) 
# assignment_data_directory = main_directory.joinpath('assignment3/data/group5')
style.use(main_directory.joinpath("default/default.mplstyle"))

# Set directory for assignment data
assignment_data_directory = main_directory.joinpath('assignment3/data/unit_test')

# Load RSO catalog file
rso_file = os.path.join(assignment_data_directory, 'unit_test_rso_catalog.pkl')
rso_dict = ConjUtil.read_catalog_file(rso_file)
obj_id = 91000

print('')
print('RSO File contains', len(rso_dict), 'objects')
print('rso_dict contains the following objects:')
print(list(rso_dict.keys()))

print('')
print('Data for each object are stored as a dictionary and can be retrieved using the object ID')
print('The following fields are available')
print(list(rso_dict[obj_id].keys()))


# Load truth data for estimation case
truth_file = os.path.join(assignment_data_directory, 'truth_unit_test_41240.pkl')
t_truth, X_truth, state_params = EstUtil.read_truth_file(truth_file)

print('')
print('The truth data file contains the following:')
print('t_truth shape', t_truth.shape)
print('X_truth shape', X_truth.shape)
print('X at t0', X_truth[0,:])
print('state_params')
print(state_params)


# Load measurement data for estimation case
meas_file = os.path.join(assignment_data_directory, 'meas_unit_test_radar_41240.pkl')
state_params, meas_dict, sensor_params = EstUtil.read_measurement_file(meas_file)

print('')
print('The measurement file contains an augmented state params dictionary that' 
      ' includes the initial state and covariance for the filter to use')
print(state_params)
print('meas_dict fields:')
print(list(meas_dict.keys()))
print('tk_list length', len(meas_dict['tk_list']))
print('Y at t0', meas_dict['Yk_list'][0])
print('sensor_params fields:')
print(list(sensor_params.keys()))



###############################################################################
# Compute TCA Example

# This code performs a unit test of the compute_TCA function. The object
# parameters are such that a collision is expected 30 minutes after the
# initial epoch (zero miss distance).

# The TCA function is run twice, using a fixed step RK4 and variable step
# RKF78 to compare the results.
#
###############################################################################

print('\nBegin TCA Test')

# Initial time and state vectors
t0 = (datetime(2024, 3, 23, 5, 30, 0) - datetime(2000, 1, 1, 12, 0, 0)).total_seconds()

X1 = np.array([[ 3.75944379e+05],
               [ 6.08137408e+06],
               [ 3.28340214e+06],
               [-5.32161464e+03],
               [-2.32172417e+03],
               [ 4.89152047e+03]])

X2 = np.array([[ 3.30312011e+06],
               [-2.69542170e+06],
               [-5.71365135e+06],
               [-4.06029364e+03],
               [-6.22037456e+03],
               [ 9.09217382e+02]])

# Basic setup parameters
bodies_to_create = ['Sun', 'Earth', 'Moon']
bodies = prop.tudat_initialize_bodies(bodies_to_create) 

rso1_params = {}
rso1_params['mass'] = 260.
rso1_params['area'] = 17.5
rso1_params['Cd'] = 2.2
rso1_params['Cr'] = 1.3
rso1_params['sph_deg'] = 8
rso1_params['sph_ord'] = 8    
rso1_params['central_bodies'] = ['Earth']
rso1_params['bodies_to_create'] = bodies_to_create

rso2_params = {}
rso2_params['mass'] = 100.
rso2_params['area'] = 1.
rso2_params['Cd'] = 2.2
rso2_params['Cr'] = 1.3
rso2_params['sph_deg'] = 8
rso2_params['sph_ord'] = 8    
rso2_params['central_bodies'] = ['Earth']
rso2_params['bodies_to_create'] = bodies_to_create

int_params = {}
int_params['tudat_integrator'] = 'rk4'
int_params['step'] = 1.

# Expected result
TCA_true = 764445600.0  
rho_true = 0.

# Interval times
tf = t0 + 3600.
trange = np.array([t0, tf])

# RK4 test
start = time.time()
T_list, rho_list = ConjUtil.compute_TCA(X1, X2, trange, rso1_params, rso2_params, 
                                        int_params, bodies)



print('')
print('RK4 TCA unit test runtime [seconds]:', time.time() - start)
print('RK4 TCA error [seconds]:', T_list[0]-TCA_true)
print('RK4 miss distance error [m]:', rho_list[0]-rho_true)


# RK78 test
int_params['tudat_integrator'] = 'rkf78'
int_params['step'] = 10.
int_params['max_step'] = 1000.
int_params['min_step'] = 1e-3
int_params['rtol'] = 1e-12
int_params['atol'] = 1e-12


start = time.time()
T_list, rho_list = ConjUtil.compute_TCA(X1, X2, trange, rso1_params, rso2_params, 
                                        int_params, bodies)


print('')
print('RK78 TCA unit test runtime [seconds]:', time.time() - start)
print('RK78 TCA error [seconds]:', T_list[0]-TCA_true)
print('RK78 miss distance error [m]:', rho_list[0]-rho_true)



###############################################################################
# Run Unscented Kalman Filter Example
#
# This code performs a unit test of the UKF function. 
#
#
###############################################################################

print('\nBegin UKF Test')

# Truth and measurement data
truth_file = os.path.join(assignment_data_directory, 'truth_unit_test_41240.pkl')
meas_file = os.path.join(assignment_data_directory, 'meas_unit_test_radar_41240.pkl')
state_params, meas_dict, sensor_params = EstUtil.read_measurement_file(meas_file)


# Setup filter parameters such as process noise
Qeci = 1e-12*np.diag([1., 1., 1.])
Qric = 1e-12*np.diag([1., 1., 1.])

filter_params = {}
filter_params['Qeci'] = Qeci
filter_params['Qric'] = Qric
filter_params['alpha'] = 1.
filter_params['gap_seconds'] = 600.

# Choose integration parameters
# int_params = {}
# int_params['tudat_integrator'] = 'rk4'
# int_params['step'] = 10.

int_params = {}
int_params['tudat_integrator'] = 'rkf78'
int_params['step'] = 10.
int_params['max_step'] = 1000.
int_params['min_step'] = 1e-3
int_params['rtol'] = 1e-12
int_params['atol'] = 1e-12

# Initialize tudat bodies
bodies_to_create = ['Earth', 'Sun', 'Moon']
bodies = prop.tudat_initialize_bodies(bodies_to_create)

# Run filter
filter_output = EstUtil.ukf(state_params, meas_dict, sensor_params, int_params, filter_params, bodies)

# Filter output contains the estimated state, covariance, and post-fit residuals. 
# The dictionary keys are the epoch times, each can used to retrieve the results
# as shown below.
print('')
print('filter output fields:')
print('number of output entries: ', len(filter_output.keys()))

final_covariance = filter_output[meas_dict["tk_list"][-1]]["covar"]

plt.figure()
ax = sns.heatmap(
    final_covariance,
    annot=True,
    cmap='viridis',
    fmt=".2f",
    cbar=True,
    linewidths=0.8,          # Thickness of the grid lines
    linecolor='white',       # Grid line color
    square=True              # Optional: makes cells square
)

# Add labels
ax.set_title("Covariance Matrix Heatmap")
ax.set_xlabel("Parameter indexes")
ax.set_ylabel("Parameter indexes")

# Beautify ticks
ax.tick_params(bottom=False, left=False)

ax.grid(False)  # Removes the default axes grid

# Optional: align ticks to center of each square
ax.set_xticks(ax.get_xticks(), labels=ax.get_xticklabels(), rotation=0)
ax.set_yticks(ax.get_yticks(), labels=ax.get_yticklabels(), rotation=0)

plt.tight_layout()
plt.show()


# For the unit test, truth data are provided which can be used to compute errors.
# For the assignment, with no true states, you can still consider the residuals

# Load truth data
pklFile = open(truth_file, 'rb' )
data = pickle.load( pklFile )
t_truth = data[0]
X_truth = data[1]
pklFile.close()

# Times
t0 = t_truth[0]
tk_list = sorted(list(filter_output.keys()))
thrs = [(tk - t0)/3600. for tk in tk_list]

# Number of states and measurements
Xo = filter_output[tk_list[0]]['state']
resids0 = filter_output[tk_list[0]]['resids']
n = len(Xo)
p = len(resids0)

print('sample output at each epoch:')
print(filter_output[tk_list[0]])

# Compute state errors
X_err = np.zeros((n, len(filter_output)))
X_err_ric = np.zeros((3, len(filter_output)))

resids = np.zeros((p, len(filter_output)))
sig_x = np.zeros(len(filter_output),)
sig_y = np.zeros(len(filter_output),)
sig_z = np.zeros(len(filter_output),)
sig_dx = np.zeros(len(filter_output),)
sig_dy = np.zeros(len(filter_output),)
sig_dz = np.zeros(len(filter_output),)

for kk in range(len(tk_list)):
    tk = tk_list[kk]
    X = filter_output[tk]['state']
    P = filter_output[tk]['covar']
    
    truth_ind = list(t_truth).index(tk)
            
    X_true = X_truth[truth_ind,:].reshape(6,1)
    X_err[:,kk] = (X - X_true).flatten()
    sig_x[kk] = np.sqrt(P[0,0])
    sig_y[kk] = np.sqrt(P[1,1])
    sig_z[kk] = np.sqrt(P[2,2])
    sig_dx[kk] = np.sqrt(P[3,3])
    sig_dy[kk] = np.sqrt(P[4,4])
    sig_dz[kk] = np.sqrt(P[5,5])
        
    resids[:,kk] = filter_output[tk]['resids'].flatten()
    

# Compute and print statistics
print('\n\nState Error and Residuals Analysis')
print('\n\t\t\t\t  Mean\t\tSTD')
print('----------------------------------------')
print('X ECI [m]\t\t', '{0:0.2E}'.format(np.mean(X_err[0,35:])), '\t{0:0.2E}'.format(np.std(X_err[0,35:])))
print('Y ECI [m]\t\t', '{0:0.2E}'.format(np.mean(X_err[1,35:])), '\t{0:0.2E}'.format(np.std(X_err[1,35:])))
print('Z ECI [m]\t\t', '{0:0.2E}'.format(np.mean(X_err[2,35:])), '\t{0:0.2E}'.format(np.std(X_err[2,35:])))
print('dX ECI [m/s]\t', '{0:0.2E}'.format(np.mean(X_err[3,35:])), '\t{0:0.2E}'.format(np.std(X_err[3,35:])))
print('dY ECI [m/s]\t', '{0:0.2E}'.format(np.mean(X_err[4,35:])), '\t{0:0.2E}'.format(np.std(X_err[4,35:])))
print('dZ ECI [m/s]\t', '{0:0.2E}'.format(np.mean(X_err[5,35:])), '\t{0:0.2E}'.format(np.std(X_err[5,35:])))

# Convert angles to degrees
resids[1,:] *= 180./np.pi
resids[2,:] *= 180./np.pi

print('Range [m]\t\t', '{0:0.2E}'.format(np.mean(resids[0,:])), '\t{0:0.2E}'.format(np.std(resids[0,:])))
print('Az [deg]\t\t', '{0:0.2E}'.format(np.mean(resids[1,:])), '\t{0:0.2E}'.format(np.std(resids[1,:])))
print('El [deg]\t\t', '{0:0.2E}'.format(np.mean(resids[2,:])), '\t{0:0.2E}'.format(np.std(resids[2,:])))
    

# State Error Plots   
plt.figure()
plt.subplot(3,1,1)
plt.plot(thrs, X_err[0,:], 'k.')
plt.plot(thrs, 3*sig_x, 'k--')
plt.plot(thrs, -3*sig_x, 'k--')
plt.ylabel('X Err [m]')

plt.subplot(3,1,2)
plt.plot(thrs, X_err[1,:], 'k.')
plt.plot(thrs, 3*sig_y, 'k--')
plt.plot(thrs, -3*sig_y, 'k--')
plt.ylabel('Y Err [m]')

plt.subplot(3,1,3)
plt.plot(thrs, X_err[2,:], 'k.')
plt.plot(thrs, 3*sig_z, 'k--')
plt.plot(thrs, -3*sig_z, 'k--')
plt.ylabel('Z Err [m]')

plt.xlabel('Time [hours]')

plt.figure()
plt.subplot(3,1,1)
plt.plot(thrs, X_err[3,:], 'k.')
plt.plot(thrs, 3*sig_dx, 'k--')
plt.plot(thrs, -3*sig_dx, 'k--')
plt.ylabel('dX Err [m/s]')

plt.subplot(3,1,2)
plt.plot(thrs, X_err[4,:], 'k.')
plt.plot(thrs, 3*sig_dy, 'k--')
plt.plot(thrs, -3*sig_dy, 'k--')
plt.ylabel('dY Err [m/s]')

plt.subplot(3,1,3)
plt.plot(thrs, X_err[5,:], 'k.')
plt.plot(thrs, 3*sig_dz, 'k--')
plt.plot(thrs, -3*sig_dz, 'k--')
plt.ylabel('dZ Err [m/s]')

plt.xlabel('Time [hours]')



# Residuals
plt.figure()

plt.subplot(3,1,1)
plt.plot(thrs, resids[0,:], 'k.')
plt.ylabel('Range [m]')

plt.subplot(3,1,2)
plt.plot(thrs, resids[1,:], 'k.')
plt.ylabel('RA [deg]')

plt.subplot(3,1,3)
plt.plot(thrs, resids[2,:], 'k.')
plt.ylabel('DEC [deg]')

plt.xlabel('Time [hours]')
    
    

plt.show()

















