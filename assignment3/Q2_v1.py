
from pathlib import Path

import matplotlib.pyplot as plt
from debugpy.common.log import timestamp_format

from EstimationUtilities import *
from TudatPropagator import *
from ConjunctionUtilities import *
import numpy as np
import traceback
from tudatpy import astro
from tudatpy.kernel.astro import two_body_dynamics
from tudatpy.kernel.astro import frame_conversion
import sys
from tudatpy.kernel.astro import two_body_dynamics

plt.rcParams.update({'font.size': 18})
#############################################################
#Load the utilities, nicely sort them out #################
#############################################################
bodies = prop.tudat_initialize_bodies()

rso_meas_path = str(Path(__file__).parent / "data" / "group5" / "estimated_rso_catalog.pkl")

Q2_file = str(Path(__file__).parent / "data" / "group5" / "q2_meas_maneuver_91159.pkl")
#print(f"Q2_file: {Q2_file}")

Q2_state_params, Q2_meas_dict, Q2_sensor_params = read_measurement_file(Q2_file)

#print(f"Q2_state_params: {Q2_state_params}")
#print(f"Q2_meas_dict: {Q2_meas_dict}")
#print(f"Q2_sensor_params: {Q2_sensor_params}")


Q2_epoch = Q2_state_params['epoch_tdb']
Q2_state = Q2_state_params['state']
Q2_covar = Q2_state_params['covar']
Q2_mass = Q2_state_params['mass']
Q2_area = Q2_state_params['area']
Q2_cd = Q2_state_params['Cd']
Q2_cr = Q2_state_params['Cr']
Q2_sphd = Q2_state_params['sph_deg']
Q2_spho = Q2_state_params['sph_ord']
Q2_cenb = Q2_state_params['central_bodies']
Q2_btcr = Q2_state_params['bodies_to_create']

times_of_measurements =  Q2_meas_dict['tk_list'] #Seconds since J2000
measurements_of_sensors = Q2_meas_dict['Yk_list']

#State parameters for integration:
state_params = {}
state_params['Cd']=Q2_cd
state_params['Cr']=Q2_cr
state_params['area']=Q2_area
state_params['mass']=Q2_mass
state_params['sph_deg']=Q2_sphd
state_params['sph_ord']=Q2_spho
state_params['central_bodies']=Q2_cenb
state_params['bodies_to_create']=Q2_btcr

#Sensor parameters
station_ecef = np.array(Q2_sensor_params['sensor_ecef'])
station_sigma = Q2_sensor_params['sigma_dict']

# Setup filter parameters such as process noise from unit test
Qeci = 1e-12*np.diag([1., 1., 1.])
Qric = 1e-12*np.diag([1., 1., 1.])

filter_params = {}
filter_params['Qeci'] = Qeci
filter_params['Qric'] = Qric
filter_params['alpha'] = 1.
filter_params['gap_seconds'] = 600.

#Integrator parameters from Unit Test .py
int_params = {}
int_params['tudat_integrator'] = 'rkf78'
int_params['step'] = 10.
int_params['max_step'] = 1000.
int_params['min_step'] = 1e-3
int_params['rtol'] = 1e-12
int_params['atol'] = 1e-12



#############################################################
#Process the measurement data for plotting ##################
#############################################################
measurements_arr = np.array(measurements_of_sensors)
times_arr = np.array(times_of_measurements)

gap_threshold = 150  #Minimum noticalbe gap
# Find indices where a gap occurs and split the arrays into clusters.
gap_indices = np.where(np.diff(times_arr) > gap_threshold)[0] + 1  # add 1 to adjust indices for splitting

# Split the times and measurements arrays into clusters.
time_clusters = np.split(times_arr, gap_indices)
measurement_clusters = np.split(measurements_arr, gap_indices)
saved_cluster_times = time_clusters.copy() # Save the times of each cluster for later use.

cluster_dicts = [] # Create a list of dictionaries to store cluster data for future use.
# Determine the number of clusters (e.g., 8 clusters).
n_clusters = len(time_clusters)

#############################################################
#Plot the measurements ######################################
#############################################################
fig, axs = plt.subplots(nrows=3, ncols=n_clusters, sharex='col', sharey='row', figsize=(8, 5))

# Loop over clusters and process/plot each one.
for col in range(n_clusters):
    # For each cluster, get its original time and measurements.
    cluster_time = time_clusters[col]
    cluster_measurements = measurement_clusters[col]

    # Save the cluster data into a dictionary for later use.
    created_dict = {}
    created_dict['tk_list'] = cluster_time  # Times (seconds since J2000)
    created_dict['Yk_list'] = cluster_measurements  # Cluster measurement values
    cluster_dicts.append(created_dict)


    rel_time = cluster_time - cluster_time[0] # Compute relative time for plotting within the cluster (starting at 0).
    #rel_time = cluster_time - times_arr[0] # Compute relative time for plotting within all measurements (starting at 0).

    # Top row: Plot the Range with a logarithmic scale for the y-axis.
    ax0 = axs[0, col] if n_clusters > 1 else axs[0]
    ax0.scatter(rel_time, cluster_measurements[:, 0])
    ax0.set_yscale('log')
    if col == 0:
        ax0.set_ylabel("Range")
    ax0.set_yticks([1e6, 2*1e6, 3*1e6])
    ax0.grid()
    ax0.set_title(f"Cluster {col + 1}")

    # Middle row: Plot Right Ascension (converted to degrees).
    ax1 = axs[1, col] if n_clusters > 1 else axs[1]
    ax1.scatter(rel_time, np.rad2deg(cluster_measurements[:, 1]), color='orange')
    if col == 0:
        ax1.set_ylabel("Right Ascension [deg]")
    ax1.grid()

    # Bottom row: Plot Declination (converted to degrees).
    ax2 = axs[2, col] if n_clusters > 1 else axs[2]
    ax2.scatter(rel_time, np.rad2deg(cluster_measurements[:, 2]), color='green')
    if col == 0:
        ax2.set_ylabel("Declination [deg]")
    ax2.grid()

fig.supxlabel("Time since measurement start [s]") # Add a single, centered x-axis label for the whole figure
plt.suptitle("Measurments of the radar system on object: id(91159)")
plt.tight_layout()
# plt.show()

#############################################################
#Use the UKF on the masurements until it dies ###############
#############################################################
filtered_outputs_V2 =ukf_until_first_truth(Q2_state_params, Q2_meas_dict, Q2_sensor_params, int_params, filter_params, bodies)

#############################################################
#Process the residuals from the dictionary ##################
#############################################################
found_residuals_until_failure = [output['resids'] for output in filtered_outputs_V2.values()]
found_residuals_until_failure = np.array(found_residuals_until_failure)

#############################################################
#Find the first jump in the residuals #######################
#############################################################
tol_max = 10**2
jump_indices = np.where(np.abs(found_residuals_until_failure[:,0]) > tol_max)[0]
filtered_outputs_v2_times = np.vstack(list(filtered_outputs_V2.keys()))

if jump_indices.size > 0:
    # Get the first jump index. Here, the jump occurs between index i and i+1.
    first_jump_index = jump_indices[0]

    # The corresponding times:
    pre_jump_time = filtered_outputs_v2_times[first_jump_index - 1]
    pre_jump_time = float(pre_jump_time)
    post_jump_time = filtered_outputs_v2_times[first_jump_index]
    post_jump_time=float(post_jump_time)

    print("First jump occurs between indices {} and {}.".format(first_jump_index - 1, first_jump_index))
    print(f"Pre-jump time: {pre_jump_time}")
    print(f"Post-jump time: {post_jump_time}")

    #Create the times of measurement points within this measurement cluster:
    ii=1
    list_of_measurement_times=[]
    while filtered_outputs_v2_times[first_jump_index+ii]-filtered_outputs_v2_times[first_jump_index+ii-1] <= (post_jump_time-pre_jump_time):
        list_of_measurement_times.append(filtered_outputs_v2_times[first_jump_index+ii-1])
        ii+=1

else:
    print(f"No jump greater than {tol_max} was found.")

#############################################################
#Prep the residuals for plotting ############################
#############################################################
x_original_keys = list(filtered_outputs_V2.keys())
num_points = len(x_original_keys)

# Generate sequential indices [0, 1, 2, ..., num_points-1] for the x-axis
x_indices = np.arange(num_points)

#############################################################
#Plot the residuals out #####################################
#############################################################
fig, axs = plt.subplots(3, 1, figsize=(10, 6), sharex=True)

# Use x_indices for the x-values to remove gaps
axs[0].scatter(x_indices, np.abs(found_residuals_until_failure[:, 0]), color="C0") # s=10 for smaller points
axs[0].axhline(3*station_sigma['rg'], color='C4', linestyle='--', linewidth=1, label=f"3Sigma of the station")
axs[0].set_ylabel("Range [m]")

axs[1].scatter(x_indices, np.abs(found_residuals_until_failure[:, 1]), color="C1")
axs[1].axhline(3*station_sigma['ra'], color='C4', linestyle='--', linewidth=1, label=f"3Sigma of the station")
axs[1].set_ylabel("Right ascension [rad]")

axs[2].scatter(x_indices, np.abs(found_residuals_until_failure[:, 2]), color="C2")
axs[2].axhline(3*station_sigma['dec'], color='C4', linestyle='--', linewidth=1, label=f"3Sigma of the station")
axs[2].set_ylabel("Declination [rad]")
axs[2].set_xlabel("Measurement Index") # X-axis now represents the index

# --- Calculate where to add vertical lines based on original key gaps ---
x_values_sorted = np.sort(np.array(x_original_keys))# Convert original keys to a numpy array and sort them
line_indices_positions = []

if len(x_values_sorted) > 1: # Check if there are enough points to calculate differences
    diffs = np.diff(x_values_sorted)
    if len(diffs) > 0: # Check if differences were actually computed
        median_diff = np.median(diffs)
        # Define a threshold for a "large" gap (e.g., > 2x median difference)
        # Add a small epsilon (1e-9) in case median_diff is 0 or all diffs are equal
        threshold = 2.0 * median_diff + 1e-9

        # Find indices 'i' in the *sorted* key array where the difference
        # between key[i] and key[i+1] exceeds the threshold.
        gap_indices = np.where(diffs > threshold)[0]

        # The vertical line should be placed *between* index 'i' and 'i+1'
        # on the sequential index plot. Position = i + 0.5
        line_indices_positions = gap_indices + 0.5

# Draw the vertical lines on each subplot at the calculated index positions
# These lines indicate where large gaps existed in the original x-values (keys)
for ax in axs:
    for idx_pos in line_indices_positions:
        label = 'End of current cluster of data'
        ax.axvline(idx_pos, color='red', linestyle='--', linewidth=1, alpha=0.8, label=label)
    ax.axvline(first_jump_index, color='black', linestyle='--', linewidth=1, label="First residual above station error")
    ax.set(yscale='log')
    ax.grid(True, which='both', axis='y', linestyle=':', linewidth=0.5)
    ax.grid(True, which='major', axis='x', linestyle=':', linewidth=0.5)

# Add a legend if vertical lines were added
if len(line_indices_positions) > 0:
    # Get handles and labels from the last axis (they are shared)
    handles, labels = axs[2].get_legend_handles_labels()
    # Create a unique list of handles/labels for the legend
    by_label = dict(zip(labels, handles))
    axs[2].legend(by_label.values(), by_label.keys())

plt.suptitle("Abolute residuals of the UKF until failure")
plt.tight_layout()
# plt.show()

#############################################################
#Process the residuals for least squares ####################
#############################################################
state_pre_maneuver_ukf = filtered_outputs_V2[pre_jump_time]['state']
cov_pre_maneuver_ukf =filtered_outputs_V2[pre_jump_time]['covar']
# print(f"state_pre_maneuver_ukf: {state_pre_maneuver_ukf}")
# print(f"cov_pre_maneuver_ukf: {cov_pre_maneuver_ukf}")

obs_post_jump_list=[]
indexes_for_last_plot_list=[]
residuals_for_last_plot=[]
for ii in range(len(list_of_measurement_times)):
    obs_post_jump_list.append(measurements_of_sensors[first_jump_index+ii])
    indexes_for_last_plot_list.append(first_jump_index+ii)
    tk_ii = post_jump_time + ii * 10
    residuals_for_last_plot.append(filtered_outputs_V2[tk_ii]['resids'])
print(f"indexes_for_last_plot_list: {indexes_for_last_plot_list}")
print(f"residuals_for_last_plot: {residuals_for_last_plot}")
#############################################################
#Set up station in Lat,Lon, Hei, then convert it to ECI #####
#############################################################
station_eci_list = []
for ii in range(len(list_of_measurement_times)):
    station_eci = eceftoeci(station_ecef, list_of_measurement_times[ii], bodies)
    station_eci_list.append(station_eci)

#############################################################
#Set up a fixed step size integrator (RKF78 for precision baby)
#############################################################
amount_of_data_points = 200
time_step_size = (post_jump_time-pre_jump_time)/amount_of_data_points
print(f"Numerical simulations will use a timestep size of {time_step_size} seconds")
rk4_fixed_int_params = {}
rk4_fixed_int_params['tudat_integrator'] = 'rk4'
rk4_fixed_int_params['step'] = time_step_size

#############################################################
#Start least squares optimization ###########################
#############################################################
print(f"Least Squares started")
#Create weight matrix assuming uncorrelated measurements W = diag(1/sigma^2)!
N=len(list_of_measurement_times)
weight_diagonal = [
    1.0 / (station_sigma['rg']**2),
    1.0 / (station_sigma['ra']**2),
    1.0 / (station_sigma['dec']**2)
]
weight_matrix = np.kron(np.eye(N), np.diag(weight_diagonal))
print(f"Using Weight Matrix (diagonal): {weight_diagonal}")

initial_guess_params_norm = [100.0, 100.0, 100.0, 0.5] #Initial Guess: dVx, dVy, dVz, tM_norm

# Bounds: dV can be anything, tM must be within the normalized interval pre-jump, post-jump, and not on these values
bounds = ([-np.inf, -np.inf, -np.inf, 0.1], # Lower bounds [dVx, dVy, dVz, tM]
          [ np.inf,  np.inf,  np.inf, 1.0])   # Upper bounds [dVx, dVy, dVz, tM]

# --- Run Optimization ---
try:
    dVx_init, dVy_init, dVz_init, tM_norm_init = initial_guess_params_norm
    print(f"Initial Guess: dV=[{dVx_init}, {dVy_init}, {dVz_init}], tM_norm={tM_norm_init}\n")

    result = least_squares(
        calculate_residuals_for_least_squares_multi_interp,
        initial_guess_params_norm,
        args=(state_pre_maneuver_ukf, pre_jump_time, #Pre-maneuver state and time
              list_of_measurement_times, obs_post_jump_list, station_eci_list, # Observation data
              post_jump_time, # End time defining the interval for tM_norm calculation
              Q2_state_params, rk4_fixed_int_params, bodies,
              weight_matrix), # Pass 3Nx3N weight matrix or None
        bounds=bounds,
        method='trf',
        jac='3-point',
        x_scale='jac',
        ftol=None, #If None and ‘method’ is not ‘lm’, the termination by this condition is disabled.
        loss='linear', # 'linear' since we use weigths, so results should be gaussian
        max_nfev=500,
        verbose=2
    )

    # --- Process Results ---
    if result.success:
        print("\nOptimization Successful!")
        optimized_params_norm = result.x
        dVx, dVy, dVz, tM_norm = optimized_params_norm
        # Calculate absolute maneuver time from normalized result
        t_impulse = pre_jump_time + tM_norm * (post_jump_time - pre_jump_time)  # Using post_jump_time as interval end

        print(f"  Optimized dV [m/s]: [{dVx:.4f}, {dVy:.4f}, {dVz:.4f}]")
        print(f"  Optimized Maneuver Time (normalized): {tM_norm:.6f}")
        print(f"  Optimized Maneuver Time (absolute): {t_impulse:.4f}")
        print(f"  Final Cost: {result.cost:.6e}")
        print(f"  Number of Function Evaluations: {result.nfev}")
        print(f"  Termination Status: {result.status}")
        print(f"  Termination Message: {result.message}")

        # --- Calculate and Print Final Unweighted Residuals ---
        print("\n  Calculating final unweighted observation residuals...")
        try:
            # Call the function again with optimized params and crucially NO weighting
            final_raw_residuals_vector = calculate_residuals_for_least_squares_multi_interp(
                optimized_params_norm,  # Use the optimal parameters found
                state_pre_maneuver_ukf,
                pre_jump_time,
                list_of_measurement_times,
                obs_post_jump_list,
                station_eci_list,
                post_jump_time,
                state_params=Q2_state_params,
                int_params=rk4_fixed_int_params,
                bodies=bodies,
                full_weight_matrix=None  # SET WEIGHTS TO NONE!!!!
            )

            print("  Final Unweighted Residuals (Measured - Predicted):")
            num_measurements = len(list_of_measurement_times)
            if len(final_raw_residuals_vector) == 3 * num_measurements:
                for k in range(num_measurements):
                    # Extract residuals for measurement k
                    res_k = final_raw_residuals_vector[k * 3: k * 3 + 3]
                    t_meas_k = list_of_measurement_times[k]
                    # Convert angles to degrees for easier interpretation
                    delta_range_m = res_k[0]
                    delta_ra_deg = np.degrees(res_k[1])  # Convert rad to deg
                    delta_dec_deg = np.degrees(res_k[2])  # Convert rad to deg

                    print(f"    Time {t_meas_k}: "
                          f"dRange = {delta_range_m:} m, "
                          f"dRA = {delta_ra_deg} deg, "
                          f"dDec = {delta_dec_deg} deg")
            else:
                print(f"  Could not process final residuals vector shape: {final_raw_residuals_vector.shape}")
                print(f"  Raw vector: {final_raw_residuals_vector}")


        except Exception as post_err:
            print(f"\n  Error calculating final unweighted residuals: {post_err}")
            traceback.print_exc()
        # ---------------------------------------------------------

    else:
        print(f"\nOptimization Failed: {result.message}")
        print(f"  Termination Status: {result.status}")
        # print(f"  Result object:\n{result}") # Optional detailed failure info


except Exception as e:
    print(f"\nError during least_squares optimization: {e}")
    traceback.print_exc()

#############################################################
#Propagate from pre-jump to jump time #######################
#############################################################
rk4_fixed_int_params['step'] = (t_impulse - pre_jump_time)/amount_of_data_points

tf, Xf, Pf = propagate_state_and_covar(state_pre_maneuver_ukf, #x0
    cov_pre_maneuver_ukf, #p0
    np.array([pre_jump_time, t_impulse]),#t_vec
    state_params, #state_params, bodies=
    rk4_fixed_int_params, #int_params
    bodies) #bodies)
Asd = np.array([Xf[0], Xf[1], Xf[2], Xf[3] + dVx, Xf[4] + dVy, Xf[5] + dVz])
print(f"tf: {tf}")
print(f"Xf: {Asd}")
print(f"Pf: {Pf}")

residuals_for_last_plot_arr = np.array(residuals_for_last_plot)
indexes_for_last_plot = np.array(indexes_for_last_plot_list)
num_measurements = len(list_of_measurement_times)
if final_raw_residuals_vector is not None and len(final_raw_residuals_vector) == 3 * num_measurements:
    # Reshape the final *optimized* residuals
    optimized_residuals_arr = final_raw_residuals_vector.reshape((num_measurements, 3))

    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(10, 8)) # Adjust figure size if needed
    print(f"\nPlotting data shapes:")
    print(f"Original residuals shape: {residuals_for_last_plot_arr.shape}")
    print(f"Optimized residuals shape: {optimized_residuals_arr.shape}")
    print(f"Indexes shape: {indexes_for_last_plot.shape}")

    # Ensure indexes_for_last_plot has the same length as the number of measurements
    if len(indexes_for_last_plot) == num_measurements:

        # Plot Range residuals
        scatter_orig_0 = axs[0].scatter(indexes_for_last_plot, np.abs(residuals_for_last_plot_arr[:, 0]), color="C0")
        scatter_opt_0 = axs[0].scatter(indexes_for_last_plot, np.abs(optimized_residuals_arr[:, 0]), color="C1", marker='x')
        # --- CHANGE: Remove label here, store handle ---
        line3sigma_0 = axs[0].axhline(3*station_sigma['rg'], color='C4', linestyle='--', linewidth=1, label="_nolegend_") # Use internal label
        axs[0].set_ylabel("Range [m]")
        # --- REMOVE individual legend ---
        # axs[0].legend(loc='upper right')

        # Plot Right Ascension residuals
        scatter_orig_1 = axs[1].scatter(indexes_for_last_plot, np.abs(residuals_for_last_plot_arr[:, 1]), color="C0",)
        scatter_opt_1 = axs[1].scatter(indexes_for_last_plot, np.abs(optimized_residuals_arr[:, 1]), color="C1", marker='x')
        # --- CHANGE: Remove label here ---
        axs[1].axhline(3*station_sigma['ra'], color='C4', linestyle='--', linewidth=1, label="_nolegend_")
        axs[1].set_ylabel("Right ascension [rad]")
        # --- REMOVE individual legend ---
        # axs[1].legend(loc='upper right')

        # Plot Declination residuals
        scatter_orig_2 = axs[2].scatter(indexes_for_last_plot, np.abs(residuals_for_last_plot_arr[:, 2]), color="C0", alpha=0.7) # Note: alpha only here? Add to others?
        scatter_opt_2 = axs[2].scatter(indexes_for_last_plot, np.abs(optimized_residuals_arr[:, 2]), color="C1", marker='x')
        # --- CHANGE: Remove label here ---
        axs[2].axhline(3*station_sigma['dec'], color='C4', linestyle='--', linewidth=1, label="_nolegend_")
        axs[2].set_ylabel("Declination [rad]")
        axs[2].set_xlabel("Measurement Index")
        # --- REMOVE individual legend ---
        # axs[2].legend(loc='upper right')

        # Common settings for all subplots
        for ax in axs:
            ax.set_yscale("log")
            ax.grid(True, which='both', linestyle=':', linewidth=0.5)

        # --- Create the Shared Legend ---
        # Add the handle for the 3 sigma line (we stored line3sigma_0 earlier)
        # Update the labels list
        # Update ncol
        handles = [scatter_orig_0, scatter_opt_0, line3sigma_0]
        labels = ["Original residual", "Optimized residual", "3 Sigma"]
        fig.legend(handles, labels, loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0.01)) # ncol=3

        # Adjust layout to prevent title overlap and make space for the legend
        plt.suptitle("The effect of the optimization on the final residuals")
        plt.tight_layout(rect=[0, 0.1, 1, 0.95])

        #plt.show()
    else:
        print(f"Error: Mismatch between number of measurements ({num_measurements}) and indexes_for_last_plot length ({len(indexes_for_last_plot)})")

else:
    print("\nSkipping plot: Final optimized residuals were not calculated successfully or had unexpected shape.")
    if final_raw_residuals_vector is not None:
         print(f"  Shape was: {final_raw_residuals_vector.shape}")