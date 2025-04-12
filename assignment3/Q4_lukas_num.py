import numpy as np
import os
from datetime import datetime
from pathlib import Path
import time
import matplotlib.pyplot as plt
import pickle

import EstimationUtilities as EstUtil
import TudatPropagator as prop
import ConjunctionUtilities as ConjUtil
import numpy as np
from scipy.linalg import inv
from tudatpy.interface import spice
from tudatpy.astro import element_conversion, two_body_dynamics
from helper import *

bodies_to_create = ['Sun', 'Earth', 'Moon']

bodies = tudat_initialize_bodies(bodies_to_create)

meas_file = Path(__file__).parent / "data" / "group5" / "q4_meas_iod_99005.pkl"
state_params, meas_dict, sensor_params = EstUtil.read_measurement_file(meas_file)

print(f"State params {state_params}")

print(f"Sensor parames {sensor_params}")

print(f"Meas Dict {meas_dict}")

sensor_pos_ecef = sensor_params['sensor_ecef']

epochs_tdb_et = meas_dict['tk_list']
tof = epochs_tdb_et[1] - epochs_tdb_et[0]
mu_earth =  central_body_gravitational_parameter = bodies.get_body(
        "Earth"
    ).gravitational_parameter


Yk1 = meas_dict['Yk_list'][0]
Yk2 = meas_dict['Yk_list'][1]


current_directory = os.getcwd()
spice.load_kernel(current_directory + "/assignment3/naif0012.tls")
spice.load_kernel(current_directory + "/assignment3/pck00011.tpc")
spice.load_kernel(current_directory + "/assignment3/earth_latest_high_prec.bpc")

rotation_matrix_t1 = spice.compute_rotation_matrix_between_frames("IAU_EARTH", "J2000", epochs_tdb_et[0])
rotation_matrix_t2 = spice.compute_rotation_matrix_between_frames("IAU_EARTH", "J2000", epochs_tdb_et[1])

# Standard deviations (given)
sigma_dict = sensor_params['sigma_dict']
sigma_rg = sigma_dict['rg']
sigma_ra = sigma_dict['ra']
sigma_dec = sigma_dict['dec']



# Define the number of samples
n_samples = 100#1000

# Generate synthetic measurements for range, right ascension, and declination
np.random.seed(42)  # For reproducibility
rg1_samples = np.random.normal(Yk1[0, 0], sigma_rg, n_samples)
ra1_samples = np.random.normal(Yk1[1, 0], sigma_ra, n_samples)
dec1_samples = np.random.normal(Yk1[2, 0], sigma_dec, n_samples)

rg2_samples = np.random.normal(Yk2[0, 0], sigma_rg, n_samples)
ra2_samples = np.random.normal(Yk2[1, 0], sigma_ra, n_samples)
dec2_samples = np.random.normal(Yk2[2, 0], sigma_dec, n_samples)

# Initialize lists to store the computed results for all samples

X1_eci_vals = []
X2_eci_vals = []
result_1_vals = []
result_2_vals = []
v1_eci_vals = []
v2_eci_vals = []


mu = np.array([Yk1[0, 0], Yk1[1, 0], Yk1[2, 0], Yk2[0, 0], Yk2[1, 0], Yk2[2, 0]])
sigma = np.array([sigma_rg, sigma_ra, sigma_dec, sigma_rg, sigma_ra, sigma_dec])

# Covariance matrix (assuming uncorrelated measurements)
cov_matrix = (np.diag([sigma_rg**2, sigma_ra**2, sigma_dec**2, 
                             sigma_rg**2, sigma_ra**2, sigma_dec**2]))



# Iterate over all samples to store the velocity components
for sample_index in range(n_samples):
    print('Iteration: ', sample_index)

    x = np.array([rg1_samples[sample_index], ra1_samples[sample_index], dec1_samples[sample_index], 
                  rg2_samples[sample_index], ra2_samples[sample_index], dec2_samples[sample_index]])

    pdf_grv = (1 / ((2 * np.pi) ** 3 * np.sqrt(np.linalg.det(cov_matrix)))) * np.exp(-0.5 * (x - mu) @ np.linalg.inv(cov_matrix) @ (x - mu).T)
    print(pdf_grv)
    print("Determinant of the covariance matrix:", np.linalg.det(cov_matrix))

    # Get the state vectors sv_1 and sv_2 from the objective function
    sv_1, sv_2 = obj_fun(rg1_samples[sample_index], ra1_samples[sample_index], dec1_samples[sample_index], 
                         rg2_samples[sample_index], ra2_samples[sample_index], dec2_samples[sample_index]
                         , rotation_matrix_t1, rotation_matrix_t2, epochs_tdb_et, mu_earth, tof, bodies, sensor_pos_ecef)

    # Append the state vectors
    X1_eci_vals.append(sv_1)
    X2_eci_vals.append(sv_2)

    # Extract velocity components from sv_1 and sv_2
    v1_eci_vals.append(sv_1[3:])  # Assuming sv_1[3:] contains velocity components
    v2_eci_vals.append(sv_2[3:])  # Assuming sv_2[3:] contains velocity components

    # Compute the Jacobian matrices for sv_1 and sv_2
    jacobian_1, jacobian_2 = compute_jacobian(x, rotation_matrix_t1, rotation_matrix_t2, 
                                               epochs_tdb_et, mu_earth, tof, bodies, sensor_pos_ecef)
    

    # Compute the determinants of the Jacobians
    det_J1 = np.linalg.det(jacobian_1)
    det_J2 = np.linalg.det(jacobian_2)

    # Multiply the determinant Taylor expansion by the GRV PDF
    result_1 = 1 / np.abs(det_J1) * pdf_grv
    result_2 = 1 / np.abs(det_J2) * pdf_grv

    # Append results for later use
    result_1_vals.append(result_1)
    result_2_vals.append(result_2)

# Convert lists of velocity vectors to numpy arrays for easier manipulation
v1_eci_vals = np.array(v1_eci_vals)
v2_eci_vals = np.array(v2_eci_vals)

# Plotting the first velocity vector distribution in a separate figure
fig1 = plt.figure(figsize=(10, 7))
ax1 = fig1.add_subplot(111, projection='3d')

# Scatter plot for v1_eci_vals with color mapping by result_1_vals
scat1 = ax1.scatter(v1_eci_vals[:, 0], v1_eci_vals[:, 1], v1_eci_vals[:, 2], c=result_1_vals, cmap='viridis', s=1, alpha=0.5)
ax1.set_title('First Velocity Vector Distribution')
ax1.set_xlabel('Vx (m/s)')
ax1.set_ylabel('Vy (m/s)')
ax1.set_zlabel('Vz (m/s)')
fig1.colorbar(scat1, ax=ax1, label='Result 1')

plt.show()

# Plotting the second velocity vector distribution in a separate figure
fig2 = plt.figure(figsize=(10, 7))
ax2 = fig2.add_subplot(111, projection='3d')

# Scatter plot for v2_eci_vals with color mapping by result_2_vals
scat2 = ax2.scatter(v2_eci_vals[:, 0], v2_eci_vals[:, 1], v2_eci_vals[:, 2], c=result_2_vals, cmap='plasma', s=1, alpha=0.5)
ax2.set_title('Second Velocity Vector Distribution')
ax2.set_xlabel('Vx (m/s)')
ax2.set_ylabel('Vy (m/s)')
ax2.set_zlabel('Vz (m/s)')
fig2.colorbar(scat2, ax=ax2, label='Result 2')

plt.show()