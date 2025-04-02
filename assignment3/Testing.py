import pickle
import numpy as np
from tudatpy import util
from tudatpy import constants
import ConjunctionUtilities
import TudatPropagator
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import eigh
from scipy.stats import norm, chi2
from scipy.spatial.distance import mahalanobis
import seaborn as sns
from statsmodels.graphics.gofplots import qqplot
# Path to the pickle file
file_path = 'data/group5/estimated_rso_catalog.pkl'

# Load the pickle file
try:
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
except FileNotFoundError:
    print(f"File not found: {file_path}")
except pickle.UnpicklingError:
    print("Error unpickling the file.")

rso_catalog =  data[0]
IDs = rso_catalog.keys()
states = np.zeros((len(IDs), 6))
mass = np.zeros((len(IDs), 1))
Cov = np.zeros((len(IDs), 6,6))
area = np.zeros((len(IDs), 1))
Cd = np.zeros((len(IDs), 1))
Cr = np.zeros((len(IDs), 1))
epoch_tdb = np.zeros_like(Cd)

for i in range(len(IDs)):
    states[i,:] = np.array(rso_catalog[list(IDs)[i]]['state']).flatten()
    
    mass[i,:] = rso_catalog[list(IDs)[i]]['mass']
    
    Cd[i,:] = rso_catalog[list(IDs)[i]]['Cd']
    
    Cr[i,:] = rso_catalog[list(IDs)[i]]['Cr']
    
    Cov[i,:] = rso_catalog[list(IDs)[i]]['covar']

    area[i , :] = rso_catalog[list(IDs)[i]]['area']

    epoch_tdb[i , :] = rso_catalog[list(IDs)[i]]['epoch_tdb']

#### TRIAL OF PROPAGATION ####

ID = 40697  # ID of the to-defend-object

# Retreive the information regarding the to-defed-body, including mean state and covariance matrix (ECI, 2025-04-01 12:00 TDB)
Cd_1 = rso_catalog[ID]['Cd']

Cr_1 = rso_catalog[ID]['Cr']

area_1 = rso_catalog[ID]['area']

mass_1 = rso_catalog[ID]['mass']

state_1 = rso_catalog[ID]['state']

COV_1 = rso_catalog[ID]['covar']

epoch_tdb_1 = rso_catalog[ID]['epoch_tdb']


# Define additional parameters for the propagation

sph_deg = 8

sph_ord = 8

central_bodies = ['Earth']

bodies_to_create = ['Earth', 'Sun', 'Moon']

# Create dictionary of state parameters for the propagation

state_params = dict(
    central_bodies = central_bodies , 
    bodies_to_create = bodies_to_create , 
    mass = mass_1 , area = area_1 , 
    Cd = Cd_1 , Cr = Cr_1 , 
    sph_deg = sph_deg , 
    sph_ord = sph_ord
    )

# Define integrator parameters = dict(step , max_step, min_step,rtol, atol, tudat_integrator)
int_params = dict(
    tudat_integrator = 'rkf78',
    step = 10,
    max_step = 600,
    min_step = 1e-3,
    rtol = 1e-10,
    atol = 1e-10
)

# Define final time  
propagation_time  =2*constants.JULIAN_DAY 

final_time = epoch_tdb_1 + propagation_time

# Call the function to propagate the result

tf , Xf , Pf = TudatPropagator.propagate_state_and_covar(state_1 , COV_1 , [epoch_tdb_1 , final_time], state_params, int_params)

print("Initial State")
print(state_1)
print("Final State")
print(Xf)
print("Initial Covariance")
print(COV_1)
print("Final Covariance")
print(Pf)

import matplotlib.pyplot as plt
def plot_ellipsoid(ax, cov_matrix, center, label, color):
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = eigh(cov_matrix)
    
    # Sort eigenvalues and eigenvectors
    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    
    # Generate points for the ellipsoid (unit sphere)
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))
    
    # Scale the points using the square root of eigenvalues
    x = np.sqrt(eigenvalues[0]) * x
    y = np.sqrt(eigenvalues[1]) * y
    z = np.sqrt(eigenvalues[2]) * z
    
    # Rotate the points by eigenvectors
    for i in range(len(x)):
        for j in range(len(x)):
            [x[i, j], y[i, j], z[i, j]] = (
                eigenvectors @ np.array([x[i, j], y[i, j], z[i, j]]).T
            )
    
    # Translate to the center
    x += center[0]
    y += center[1]
    z += center[2]
    
    # Plot the ellipsoid using a wireframe for better visualization
    ax.plot_wireframe(x, y, z, color=color, alpha=0.5)

# Example usage with initial and final covariance matrices
initial_cov_POS = COV_1[0:3, 0:3]

final_cov = Pf[0:3 , 0:3]

initial_state = np.array([0, 0, 0])
final_state = np.array([0, 0, 0])

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot initial uncertainty ellipsoid
plot_ellipsoid(ax, initial_cov_POS, initial_state, label="Initial", color="blue")

# Plot final uncertainty ellipsoid
plot_ellipsoid(ax, final_cov, final_state, label="Final", color="red")

# Set plot labels and legend
ax.set_xlabel("X [m]")
ax.set_ylabel("Y [m]")
ax.set_zlabel("Z [m]")
ax.set_title("3D Uncertainty Ellipsoid")
ax.legend()


# Number of samples
N = 1000

# Generate N samples from the initial state distribution
initial_samples = np.random.multivariate_normal(mean=state_1.flatten(), cov=COV_1, size=N)

# Array to store final states
final_states = np.zeros((N, 6))

# Propagate each sample for 1 hour
propagation_time_1hr = propagation_time  # 1 hour in seconds
final_time_1hr = epoch_tdb_1 + propagation_time_1hr

for i in range(N):
    _ ,final_state = TudatPropagator.propagate_orbit(
        initial_samples[i], [epoch_tdb_1, final_time_1hr], state_params, int_params
    )
    final_states[i] = final_state[-1,:]
    print(i)

# Compare the distribution of final states with a normal distribution
mean_final = Xf.flatten()  # Mean from the propagation of the mean
cov_final = Pf  # Covariance from the propagation of the mean

# Plot histograms and Q-Q plots for each state component
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

# Plot histograms and normal distribution overlays
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for i in range(6):
    # Histogram of the final states
    sns.histplot(final_states[:, i], kde=True, ax=axes[i], color="skyblue", label="Samples")
    
    # Overlay normal distribution
    x = np.linspace(mean_final[i] - 3 * np.sqrt(cov_final[i, i]), mean_final[i] + 3 * np.sqrt(cov_final[i, i]), 100)
    axes[i].plot(x, norm.pdf(x, mean_final[i], np.sqrt(cov_final[i, i])), color="red", label="Normal")
    
    # Labels
    axes[i].set_title(f"State Component {i+1}")
    axes[i].legend()

plt.tight_layout()
plt.show()

# Plot Q-Q plots in a separate figure
fig, qq_axes = plt.subplots(2, 3, figsize=(15, 10))
qq_axes = qq_axes.flatten()

for i in range(6):
    qqplot(final_states[:, i], line="s", ax=qq_axes[i])
    qq_axes[i].set_title(f"Q-Q Plot Component {i+1}")

plt.tight_layout()
plt.show()

from scipy.stats import kstest
# Standardize the final states before applying the KS test
standardized_states = (final_states - mean_final) / np.sqrt(np.diag(cov_final))

# Perform the KS test on standardized data
for i in range(standardized_states.shape[1]):  # Loop over state components
    stat, p_value = kstest(standardized_states[:, i], 'norm', args=(0, 1))
    print(f"Component {i+1} - KS Test p-value: {p_value}")

from scipy.stats import anderson
for i in range(final_states.shape[1]):
    result = anderson(final_states[:, i])
    print(f"Component {i+1} - Anderson-Darling Test Statistic: {result.statistic}")
    for sl, cv in zip(result.significance_level, result.critical_values):
        print(f"Significance Level: {sl}% - Critical Value: {cv}")
    if result.statistic > result.critical_values[2]:  # Compare with 5% significance level
        print(f"Component {i+1} - Reject null hypothesis at 5% significance level.")
    else:
        print(f"Component {i+1} - Fail to reject null hypothesis at 5% significance level.")

