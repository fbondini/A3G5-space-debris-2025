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
from sympy import symbols, Matrix, diff, exp, pi, sqrt, series, cos, sin, asin, Piecewise, simplify, pprint

from tudatpy.interface import spice

bodies = prop.tudat_initialize_bodies()

meas_file = Path(__file__).parent / "data" / "group5" / "q4_meas_iod_99005.pkl"
state_params, meas_dict, sensor_params = EstUtil.read_measurement_file(meas_file)

print(meas_dict)
print(sensor_params)

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

# Setup the Gaussian Random Vector (GRV)
# Measurement variables and uncertainties
rg1, ra1, dec1, rg2, ra2, dec2 = symbols('rg1 ra1 dec1 rg2 ra2 dec2')

# Standard deviations (given)
sigma_dict = sensor_params['sigma_dict']
sigma_rg = sigma_dict['rg']
sigma_ra = sigma_dict['ra']
sigma_dec = sigma_dict['dec']

# Mean vector (symbolic for flexibility)
 
X = Matrix([rg1, ra1, dec1, rg2, ra2, dec2])
mu = Matrix([Yk1[0, 0], Yk1[1, 0], Yk1[2, 0], Yk2[0, 0], Yk2[1, 0], Yk2[2, 0]])
sigma = Matrix([sigma_rg, sigma_ra, sigma_dec, sigma_rg, sigma_ra, sigma_dec])

# Covariance matrix (assuming uncorrelated measurements)
cov_matrix = Matrix(np.diag([sigma_rg**2, sigma_ra**2, sigma_dec**2, 
                             sigma_rg**2, sigma_ra**2, sigma_dec**2]))

# PDF of the Gaussian random vector (GRV)
x = Matrix([rg1, ra1, dec1, rg2, ra2, dec2])
pdf_grv = (1 / ((2 * pi) ** 3 * sqrt(cov_matrix.det()))) * exp(-0.5 * (x - mu).T * cov_matrix.inv() * (x - mu))

# Step 2: Position vectors in ECEF (using sympy functions for symbolic variables)
def position_vector(rg, ra, dec, r_gs):
    """
    Calculate the position vector in ECEF coordinates from range, elevation, and right ascension.
    """
    pos = rg * Matrix([
        cos(dec) * cos(ra),
        cos(dec) * sin(ra),
        sin(dec)
    ])

    pos[0] += r_gs[0]
    pos[1] += r_gs[1]
    pos[2] += r_gs[2]
    
    return pos


    

# Initial and final position vectors (ECEF)
r1_ecef = position_vector(rg1, ra1, dec1, sensor_pos_ecef)
r2_ecef = position_vector(rg2, ra2, dec2, sensor_pos_ecef)


# ECI position vectors
r1_eci = rotation_matrix_t1 * r1_ecef
r2_eci = rotation_matrix_t2 * r2_ecef


def lambert_solver_symbolic(r1, r2, tof, mu):
    # Step 1: Precomputations
    r1_norm = r1.norm()
    r2_norm = r2.norm()

    c = (r2 - r1).norm()
    s = (r1_norm + r2_norm + c) / 2

    # Unit vectors and cross product for normal vector
    ir1 = r1 / r1_norm
    ir2 = r2 / r2_norm
    ih = ir1.cross(ir2)
    ih = ih / ih.norm()

    # Lambert parameter lambda
    lambda_sq = 1 - c / s
    lam = sqrt(lambda_sq)
    lam = lam * Piecewise((1, (ir1.cross(ir2)).dot(ih) >= 0), (-1, True))

    # Scaling factors
    gamma = sqrt(mu * s / 2)
    rho = (r1_norm - r2_norm) / c
    sigma = sqrt(1 - rho**2)

    # Householder's iteration for symbolic x (single revolution case)
    def f(x): 
        return asin(Piecewise((lam * x, abs(lam * x) <= 1), (1, True))) - asin(Piecewise((x, abs(x) <= 1), (1, True))) - tof / sqrt(mu)

    # Guess and iterate
    x = 0.5  # Initial guess for x
    for _ in range(2):  # Householder iterations
        fx = f(x)
        dfx = lam / sqrt(1 - (lam * x)**2) - 1 / sqrt(1 - x**2)
        d2fx = -(lam**2 * x) / (1 - (lam * x)**2)**(3/2) + x / (1 - x**2)**(3/2)
        correction = fx / dfx * (1 + 0.5 * (fx * d2fx) / (dfx**2))
        x = x - correction


    # Calculate y from x
    y = sqrt(1 - lam**2 * x**2)

    # Radial velocity components
    Vr1 = gamma * ((lam * y - x) - rho * (lam * y + x)) / r1_norm
    Vr2 = -gamma * ((lam * y - x) + rho * (lam * y + x)) / r2_norm

    # Transverse velocity components
    Vt1 = gamma * sigma * (y + lam * x) / r1_norm
    Vt2 = gamma * sigma * (y + lam * x) / r2_norm

    # Transverse unit vectors
    it1 = ih.cross(ir1)
    it2 = ih.cross(ir2)

    # Final velocity vectors in RTN frame
    v1_rtn = Matrix([Vr1, Vt1, 0])
    v2_rtn = Matrix([Vr2, Vt2, 0])

    # Rotation matrix from RTN to ECI
    R_rtn_to_eci_t1 = Matrix.hstack(ir1, it1, ih)
    R_rtn_to_eci_t2 = Matrix.hstack(ir2, it2, ih)

    # Rotate velocity vectors from RTN to ECI
    v1_eci = R_rtn_to_eci_t1 * v1_rtn
    v2_eci = R_rtn_to_eci_t2 * v2_rtn

    return v1_eci, v2_eci

v1_eci, v2_eci = lambert_solver_symbolic(r1_eci, r2_eci, tof, mu_earth)
# Step 5: State vectors in ECI frame
X1_eci = Matrix.vstack(simplify(r1_eci[0]), simplify(r1_eci[1]), simplify(r1_eci[2]), simplify(v1_eci[0]), simplify(v1_eci[1]), simplify(v1_eci[2])) # Initial state vector
X2_eci = Matrix.vstack(simplify(r2_eci[0]), simplify(r2_eci[1]), simplify(r2_eci[2]), simplify(v2_eci[0]), simplify(v2_eci[1]), simplify(v2_eci[2])) # Initial state vector
print('simplified')

variables = [rg1, ra1, dec1, rg2, ra2, dec2]

# Step 6: Jacobian matrices with respect to measurement vector
J1 = X1_eci.jacobian(variables)
J2 = X2_eci.jacobian(variables)

print('OK - Jacobian')

# Step 7: Determinants of the Jacobians
det_J1 = J1.det()
det_J2 = J2.det()

print('OK - Determinant')

# Step 8: Taylor expansion of the determinant to the sixth order
taylor_exp_J1 = series(det_J1, X, 0, 7).removeO()
taylor_exp_J2 = series(det_J2, X, 0, 7).removeO()

# Step 9: Multiply the determinant Taylor expansion by the GRV PDF
result_1 = 1/taylor_exp_J1 * pdf_grv
result_2 = 1/taylor_exp_J2 * pdf_grv


# Define the number of samples
n_samples = 100

# Generate synthetic measurements for range, right ascension, and declination
np.random.seed(42)  # For reproducibility
rg1_samples = np.random.normal(Yk1[0, 0], sigma_rg, n_samples)
ra1_samples = np.random.normal(Yk1[1, 0], sigma_ra, n_samples)
dec1_samples = np.random.normal(Yk1[2, 0], sigma_dec, n_samples)

rg2_samples = np.random.normal(Yk2[0, 0], sigma_rg, n_samples)
ra2_samples = np.random.normal(Yk2[1, 0], sigma_ra, n_samples)
dec2_samples = np.random.normal(Yk2[2, 0], sigma_dec, n_samples)

# Initialize lists to store the computed results for all samples
v1_eci_vals = []
v2_eci_vals = []
result_1_vals = []
result_2_vals = []

# Iterate over all samples
for sample_index in range(n_samples):
    print('OK')
    # Substitute the sample measurements into the symbolic expressions
    result_1_numerical = result_1.subs({
        rg1: rg1_samples[sample_index], 
        ra1: ra1_samples[sample_index],
        dec1: dec1_samples[sample_index], 
        rg2: rg2_samples[sample_index], 
        ra2: ra2_samples[sample_index], 
        dec2: dec2_samples[sample_index]
    })
    
    result_2_numerical = result_2.subs({
        rg1: rg1_samples[sample_index], 
        ra1: ra1_samples[sample_index],
        dec1: dec1_samples[sample_index], 
        rg2: rg2_samples[sample_index], 
        ra2: ra2_samples[sample_index], 
        dec2: dec2_samples[sample_index]
    })
    
    # Evaluate the symbolic expressions numerically
    result_1_numerical = result_1_numerical.evalf()
    result_2_numerical = result_2_numerical.evalf()
    
    # Substitute into symbolic expressions for velocities
    v1_eci_numerical = v1_eci.subs({
        rg1: rg1_samples[sample_index], 
        ra1: ra1_samples[sample_index], 
        dec1: dec1_samples[sample_index],
        rg2: rg2_samples[sample_index], 
        ra2: ra2_samples[sample_index], 
        dec2: dec2_samples[sample_index]
    }).evalf()

    v2_eci_numerical = v2_eci.subs({
        rg1: rg1_samples[sample_index], 
        ra1: ra1_samples[sample_index], 
        dec1: dec1_samples[sample_index],
        rg2: rg2_samples[sample_index], 
        ra2: ra2_samples[sample_index], 
        dec2: dec2_samples[sample_index]
    }).evalf()

    # Append the results to the respective lists
    v1_eci_vals.append([v1_eci_numerical[0], v1_eci_numerical[1], v1_eci_numerical[2]])
    v2_eci_vals.append([v2_eci_numerical[0], v2_eci_numerical[1], v2_eci_numerical[2]])
    
    result_1_vals.append(result_1_numerical)
    result_2_vals.append(result_2_numerical)

# Convert lists to numpy arrays for easier manipulation
v1_eci_vals = np.array(v1_eci_vals)
v2_eci_vals = np.array(v2_eci_vals)
result_1_vals = np.array(result_1_vals)
result_2_vals = np.array(result_2_vals)

# Plotting the first velocity vector distribution in a separate figure
fig1 = plt.figure(figsize=(10, 7))
ax1 = fig1.add_subplot(111, projection='3d')
scat1 = ax1.scatter(v1_eci_vals[:, 0], v1_eci_vals[:, 1], v1_eci_vals[:, 2], c=result_1_vals, cmap='viridis', s=1, alpha=0.5)
ax1.set_title('First Velocity Vector Distribution')
ax1.set_xlabel('Vx (m/s)')
ax1.set_ylabel('Vy (m/s)')
ax1.set_zlabel('Vz (m/s)')
fig1.colorbar(scat1, ax=ax1, label='result_1')
plt.show()

# Plotting the second velocity vector distribution in a separate figure
fig2 = plt.figure(figsize=(10, 7))
ax2 = fig2.add_subplot(111, projection='3d')
scat2 = ax2.scatter(v2_eci_vals[:, 0], v2_eci_vals[:, 1], v2_eci_vals[:, 2], c=result_2_vals, cmap='plasma', s=1, alpha=0.5)
ax2.set_title('Second Velocity Vector Distribution')
ax2.set_xlabel('Vx (m/s)')
ax2.set_ylabel('Vy (m/s)')
ax2.set_zlabel('Vz (m/s)')
fig2.colorbar(scat2, ax=ax2, label='result_2')
plt.show()