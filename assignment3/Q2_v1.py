
from pathlib import Path

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
#Nice plots for clusters and measurements ###################
#############################################################
measurements_arr = np.array(measurements_of_sensors)
times_arr = np.array(times_of_measurements)

# Define gap threshold in seconds: if the difference between consecutive times exceeds this value, it's considered a new cluster.
gap_threshold = 150  # Adjust as needed

# Find indices where a gap occurs and split the arrays into clusters.
gap_indices = np.where(np.diff(times_arr) > gap_threshold)[0] + 1  # add 1 to adjust indices for splitting

# Split the times and measurements arrays into clusters.
time_clusters = np.split(times_arr, gap_indices)
measurement_clusters = np.split(measurements_arr, gap_indices)

# Save the times of each cluster for later use.
saved_cluster_times = time_clusters.copy()

# Create a list of dictionaries to store cluster data for future use.
cluster_dicts = []
# Each dictionary will hold:
#   'tk_list' : the time values (seconds since J2000) for the cluster
#   'Yk_list' : the corresponding measurement arrays for the cluster

# Determine the number of clusters (e.g., 8 clusters).
n_clusters = len(time_clusters)

# Create a subplot grid with 3 rows (for Range, Right Ascension, and Declination)
# and one column per cluster. We share the x-axis per column and y-axis per row.
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

    # Compute relative time for plotting within the cluster (starting at 0).
    rel_time = cluster_time - cluster_time[0]

    # Top row: Plot the Range with a logarithmic scale for the y-axis.
    ax0 = axs[0, col] if n_clusters > 1 else axs[0]
    ax0.scatter(rel_time, cluster_measurements[:, 0])
    ax0.set_yscale('log')
    if col == 0:
        ax0.set_ylabel("Range")
    ax0.set_title(f"Cluster {col + 1}")

    # Middle row: Plot Right Ascension (converted to degrees).
    ax1 = axs[1, col] if n_clusters > 1 else axs[1]
    ax1.scatter(rel_time, np.rad2deg(cluster_measurements[:, 1]), color='orange')
    if col == 0:
        ax1.set_ylabel("Right Ascension [deg]")

    # Bottom row: Plot Declination (converted to degrees).
    ax2 = axs[2, col] if n_clusters > 1 else axs[2]
    ax2.scatter(rel_time, np.rad2deg(cluster_measurements[:, 2]), color='green')
    if col == 0:
        ax2.set_ylabel("Declination [deg]")
    ax2.set_xlabel("Time (s)")

plt.tight_layout()
# plt.show()

#Use the UKF until it fails (it will fail because of the DeltaV manuver):
filtered_outputs_V2 =ukf_until_first_truth(Q2_state_params, Q2_meas_dict, Q2_sensor_params, int_params, filter_params, bodies)

#Get the residuals:
found_residuals_until_failure = [output['resids'] for output in filtered_outputs_V2.values()]
found_residuals_until_failure = np.array(found_residuals_until_failure)

#Plot the absolute of the residuals as a function of time:
fig,axs = plt.subplots(3,1, figsize = (8,5))

axs[0].scatter(filtered_outputs_V2.keys(), np.abs(found_residuals_until_failure[:,0]), label= "r_g", color = "C0")
axs[1].scatter(filtered_outputs_V2.keys(), np.abs(found_residuals_until_failure[:,1]), label= "r_a", color = "C1")
axs[2].scatter(filtered_outputs_V2.keys(), np.abs(found_residuals_until_failure[:,2]), label= "dec", color = "C2")

for ax in axs:
    ax.set(yscale='log')
    ax.grid(True)

plt.tight_layout()
# plt.show()

# Find indices where the absolute difference is greater than a specified value (
# consult diagram above if needed)
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

#Get the positions and covariances of these two times:
state_pre_maneuver_ukf = filtered_outputs_V2[pre_jump_time]['state']
# print(f"state_post_maneuver_ukf: {state_pre_maneuver_ukf}")
cov_pre_maneuver_ukf =filtered_outputs_V2[pre_jump_time]['covar']
resids_pre_maneuver_ukf = filtered_outputs_V2[pre_jump_time]['resids']
# print(f"resids_pre_maneuver_ukf: {resids_pre_maneuver_ukf}")
measured_pre_jump = measurements_of_sensors[first_jump_index-1]
#print(f"measured_pre_jump: {measured_pre_jump}")

state_post_maneuver_ukf = filtered_outputs_V2[post_jump_time]['state']
# print(f"state_post_maneuver_ukf: {state_post_maneuver_ukf}")
cov_post_maneuver_ukf =filtered_outputs_V2[post_jump_time]['covar']
resids_post_maneuver_ukf = filtered_outputs_V2[post_jump_time]['resids']
# print(f"resids_post_maneuver_ukf: {resids_post_maneuver_ukf}")

obs_post_jump_list=[]
for ii in range(len(list_of_measurement_times)):
    obs_post_jump_list.append(measurements_of_sensors[first_jump_index+ii])
# print(obs_post_jump_list)

h_pre_maneuver = np.cross(state_pre_maneuver_ukf[:3].flatten(), state_pre_maneuver_ukf[3:6].flatten())
h_post_maneuver = np.cross(state_post_maneuver_ukf[:3].flatten(), state_post_maneuver_ukf[3:6].flatten())
print(f"Sanity check for the conservation of angular momentum, this is the change in angular momentum of the two states:"
      f" {np.linalg.norm(h_pre_maneuver) - np.linalg.norm(h_post_maneuver)}")

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
rkf78_fixed_int_params = {}
rkf78_fixed_int_params['tudat_integrator'] = 'rk78_fixed'
rkf78_fixed_int_params['step'] = time_step_size

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

#Lambert for initial guess of the deltaV
sun_gm = 398600.4418 # [m^3/s^2]

# Given initial values
initial_epoch = pre_jump_time # [s]
final_epoch = post_jump_time # [s]
departure_pos_initial_epoch = state_pre_maneuver_ukf[0:3] # [m]
target_pos_final_epoch = state_post_maneuver_ukf[0:3] # [m]

# Solve Lambert's problem
lambert_targeter = two_body_dynamics.LambertTargeterIzzo(
	departure_position = departure_pos_initial_epoch,
	arrival_position = target_pos_final_epoch,
	time_of_flight = final_epoch - initial_epoch,
	gravitational_parameter = sun_gm)

# Extract results
v1, v2 = lambert_targeter.get_velocity_vectors()
# Print the results
np.set_printoptions(formatter={'float': '{: 9.3e}'.format})
print('|V1|{tab} = {v1:6.3f}{tab} [km/s] {cr}|V2|{tab} = {v2:6.3f}{tab} [km/s]'.format(v1=np.linalg.norm(v1)/1e3, v2=np.linalg.norm(v2)/1e3, cr='\n', tab='\t'))
print(' V1{tab} = {v1}{tab} [km/s] {cr} V2{tab} = {v2}{tab} [km/s]'.format(v1 = np.array2string(v1/1e3, formatter = {'float_kind': '{0:+6.3f}'.format}), v2 = np.array2string(v2/1e3, formatter = {'float_kind': '{0:+6.3f}'.format}), cr='\n', tab='\t'))

initial_guess_params_norm = [100.0, 100.0, 100.0, 0.5] #Initial Guess: dVx, dVy, dVz, tM_norm

# Bounds: dV can be anything, tM must be within the normalized interval pre-jump, post-jump
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
              Q2_state_params, rkf78_fixed_int_params, bodies,
              weight_matrix), # Pass 3Nx3N weight matrix or None
        bounds=bounds,
        method='trf',
        jac='3-point',
        x_scale='jac',
        #diff_step=1e-6,  # Explicit relative step size
        ftol=None, #If None and ‘method’ is not ‘lm’, the termination by this condition is disabled.
        loss='linear', # 'linear' if residuals are expected to be Gaussian
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
                int_params=rkf78_fixed_int_params,
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
# cost_list = []
# tm_tries = [0.0, 0.1, 0.25, 0.5, 0.75]
# for element in tm_tries:
#     res_tm_try = calculate_residuals_for_least_squares_multi_interp(
#                 [optimized_params_norm[0], optimized_params_norm[1], optimized_params_norm[2], element],
#                 state_pre_maneuver_ukf,
#                 pre_jump_time,
#                 list_of_measurement_times,
#                 obs_post_jump_list,
#                 station_eci_list,
#                 post_jump_time,
#                 state_params=Q2_state_params,
#                 int_params=rkf78_fixed_int_params,
#                 bodies=bodies,
#                 full_weight_matrix=None
#             )
#     res_tm_sq =res_tm_try**2
#     cost_list.append(res_tm_sq)
# fig, ax = plt.subplot()
# ax.scatter(tm_tries, cost_list)
# plt.show()

#############################################################
#Propagate from pre-jump to jump time #######################
#############################################################
rkf78_fixed_int_params['step'] = (t_impulse - pre_jump_time)/amount_of_data_points

tf, Xf, Pf = propagate_state_and_covar(state_pre_maneuver_ukf, #x0
    cov_pre_maneuver_ukf, #p0
    np.array([pre_jump_time, t_impulse]),#t_vec
    state_params, #state_params, bodies=
    rkf78_fixed_int_params, #int_params
    bodies) #bodies)
Asd = np.array([Xf[0], Xf[1], Xf[2], Xf[3] + dVx, Xf[4] + dVy, Xf[5] + dVz])
print(f"tf: {tf}")
print(f"Xf: {Asd}")
print(f"Pf: {Pf}")
#
# #Propagate the pre_maneuver state forward, until the time of the already changed state:
# t_pre_maneuver_list = []
# state_pre_maneuver_list = []
# cov_pre_maneuver_list = []
# timespace = np.linspace(pre_jump_time, post_jump_time, amount_of_data_points)
#
# for ii in range(len(timespace)):
#     time=timespace[ii]
#     #print(f"Iteration {ii}, end_time: {time}")
#     t_pre_maneuver, state_pre_maneuver, cov_pre_maneuver = propagate_state_and_covar(
#     state_pre_maneuver_ukf, #x0
#     cov_pre_maneuver_ukf, #p0
#     np.array([pre_jump_time, time]),#t_vec
#     state_params, #state_params, bodies=
#     rkf78_fixed_int_params, #int_params
#     bodies) #bodies
#     # print(f"t_pre_maneuver :{t_pre_maneuver}")
#     # print(f"state_pre_maneuver :{state_pre_maneuver}")
#     # print(f"cov_pre_maneuver :{cov_pre_maneuver}")
#     t_pre_maneuver_list.append(t_pre_maneuver)
#     state_pre_maneuver_list.append(state_pre_maneuver)
#     cov_pre_maneuver_list.append(cov_pre_maneuver)
#
# #Propagate the post-maneuver state backwards:
# t_post_maneuver_list = []
# state_post_maneuver_list = []
# cov_post_maneuver_list = []
#
# for ii in range(len(timespace)):
#     time = timespace[(len(timespace)-1)-ii]
#     #print(f"Iteration {ii}, start time: {time}")
#     t_post_maneuver, state_post_maneuver, cov_post_maneuver = propagate_state_and_covar_back(
#     state_post_maneuver_ukf, #x0
#     cov_post_maneuver_ukf, #p0
#     np.array([time, post_jump_time]),#t_vec
#     state_params, #state_params, bodies=
#     rkf78_fixed_int_params, #int_params
#     bodies) #bodies
#     # print(f"t_post_maneuver :{t_post_maneuver}")
#     # print(f"state_post_maneuver :{state_post_maneuver}")
#     # print(f"cov_post_maneuver :{cov_post_maneuver}")
#     t_post_maneuver_list.append(t_post_maneuver)
#     state_post_maneuver_list.append(state_post_maneuver)
#     cov_post_maneuver_list.append(cov_post_maneuver)
#
# #Format the found values to numpy arrays:
# t_pre_maneuver = np.array(t_pre_maneuver_list)
# # print(f"t_pre_maneuver :{t_pre_maneuver}")
# state_pre_maneuver = np.array(state_pre_maneuver_list)
# # print(f"state_pre_maneuver :{state_pre_maneuver}")
# cov_pre_maneuver = np.array(cov_pre_maneuver_list)
#
# t_post_maneuver = np.array(t_post_maneuver_list)
# # print(f"t_post_maneuver :{t_post_maneuver}")
# state_post_maneuver = np.array(state_post_maneuver_list)
# # print(f"state_post_maneuver :{state_post_maneuver}")
# cov_post_maneuver = np.array(cov_post_maneuver_list)
#
# #The backwards propagation needs to be changed:
# t_post_maneuver_forward = np.ones_like(t_post_maneuver)
# state_post_maneuver_forward = np.ones_like(state_post_maneuver)
# cov_post_maneuver_forward = np.ones_like(cov_post_maneuver)
# for ii in range(amount_of_data_points):
#     t_post_maneuver_forward[ii] = t_post_maneuver[(amount_of_data_points-1)-ii]
#     state_post_maneuver_forward[ii, :]=state_post_maneuver[(amount_of_data_points-1)-ii, :]
#     cov_post_maneuver_forward[ii, :] = cov_post_maneuver[(amount_of_data_points-1)-ii, :]
# #DEBUG:
# # print(f" If this is all zeros, we are good: {t_post_maneuver_forward-t_pre_maneuver}")
# # print(f"state_post_maneuver_forward: {state_post_maneuver_forward}")
#
# #Check if there is a point of position where these two orbits are together:
# positions_pre_maneuver = state_pre_maneuver[:, :3]
# velocities_pre_maneuver = state_pre_maneuver[:, 3:6]
# # print(f"positions_pre_maneuver: {positions_pre_maneuver}")
# positions_post_maneuver = state_post_maneuver_forward[:, :3]
# velocities_post_maneuver = state_post_maneuver_forward[:, 3:6]
# # print(f"positions_post_maneuver: {positions_post_maneuver}")
#
# #DEBUG
# # print(f"Sanity check for the propagated pre-maneuver state: {state_pre_maneuver[0, :] - state_pre_maneuver_ukf}")
# # print(f"Sanity check for the propagated post-maneuver state: {state_post_maneuver_forward[-1, :] - state_post_maneuver_ukf}")
#
#
# #Validate the single impulsive burn assumption:
# delta_r = positions_post_maneuver-positions_pre_maneuver
# delta_v = velocities_post_maneuver - velocities_pre_maneuver
# fig, axs = plt.subplots(2, 1, figsize=(8,5))
# axs[0].scatter(t_post_maneuver_forward, np.linalg.norm(delta_r, axis=1), color = "C0")
# axs[0].set_ylabel("Position difference [m]")
#
# axs[1].scatter(t_post_maneuver_forward, np.linalg.norm(delta_v, axis=1), color = "C0")
# axs[1].set_ylabel("Velocity difference [m/s]")
# for ax in axs:
#     ax.get_yaxis().get_major_formatter().set_scientific(False)
#     # Optionally, disable scientific notation on the x-axis as well:
#     # ax.get_xaxis().get_major_formatter().set_scientific(False)
# plt.show()
#
#
# # velocity_pre_maneuver = state_pre_maneuver[:, 3:6]
# # velocity_pre_maneuver_norm= np.linalg.norm(velocity_pre_maneuver, axis = 1)
# # # print(f"t_pre_maneuver_list: {t_pre_maneuver_list}")
# # # print(f"velocity_evolution: {velocity_evolution}")
# # velocity_post_maneuver = state_post_maneuver[:, 3:6]
# # velocity_post_maneuver_norm = np.linalg.norm(velocity_post_maneuver, axis =1)
#
# #The backwards propagation needs to be reversed so that the times match with the pre_maneuver ones:
# t_post_maneuver_forward = np.ones_like(t_post_maneuver)
# velocity_post_maneuver_forward = np.ones_like(state_post_maneuver[:, 3:6])
# delta_V = {}
# for ii in range(1000):
#     t_post_maneuver_forward[ii] = t_post_maneuver[(1000-1)-ii]
#     velocity_post_maneuver_current=velocity_post_maneuver[(1000-1)-ii, :]
#     velocity_post_maneuver_forward[ii, :] = velocity_post_maneuver_current
#     velocity_difference = velocity_post_maneuver_forward[ii, :] - velocity_pre_maneuver[ii, :]
#     print(f"{np.linalg.norm(velocity_difference)}: {velocity_difference} = {velocity_post_maneuver_forward[ii, :]} - {velocity_pre_maneuver[ii, :]}")
#     delta_V[f"{t_post_maneuver_forward[ii]}"] = np.linalg.norm(velocity_difference)
#
# #DEBUG:
# print(t_post_maneuver_forward-t_pre_maneuver)
# print(t_post_maneuver-t_pre_maneuver)
#
# # # Prepare a parent dictionary to hold the filtered outputs, keyed by cluster names.
# # filtered_outputs = {}
# #
# # # Loop over each cluster dictionary
# # for i, d in enumerate(cluster_dicts):
# #     # Call the filtering function using the current cluster's dictionary
# #     filter_output = ukf(Q2_state_params, d, Q2_sensor_params, int_params, filter_params, bodies)
# #
# #     # Store the filter output under a key that identifies the cluster.
# #     # For instance, use 'cluster_1', 'cluster_2', etc.
# #     key = f"cluster_{i + 1}"
# #     filtered_outputs[key] = filter_output
#
# # Now, filtered_outputs is a dictionary where each key corresponds to a cluster's filtered output.
# # Initialize lists to collect time stamps and state values
# all_times = []
# first3_states = []   # first 3 elements of the state vector
# last3_states = []    # last 3 elements of the state vector
# state_list = []
# kepler_list = []
# covar_list = []
#
# # Iterate through each cluster's filtered output dictionary
# for cluster_key, cluster_data in filtered_outputs.items():
#     # Ensure the times are processed in sorted order
#     for t in sorted(cluster_data.keys()):
#         all_times.append(t)
#         # Convert the state to a flat array (if it isn't already)
#         state = np.array(cluster_data[t]['state'])
#         state_list.append(state)
#         kepler_six = astro.element_conversion.cartesian_to_keplerian(state, 3.986004415e14)
#         kepler_list.append(kepler_six)
#         covar = np.array(cluster_data[t]['covar'])
#         covar_list.append(covar)
#
# # Convert lists to numpy arrays for easier handling.
# all_times_OG = np.array(all_times)
# all_times = all_times_OG-all_times_OG[0]
# all_states = np.array(state_list)
# all_kepler = np.array(kepler_list)
# all_covar = np.array(covar_list)
# first3_states = all_states[:, :3]   # shape: (num_points, 3)
# last3_states = all_states[:, 3:6]       # shape: (num_points, 3)
#
# # Create a figure with two subplots (vertical layout), sharing the x-axis.
# fig, axs = plt.subplots(2, 1, figsize=(8,5), sharex=True)
#
# # Plot the first 3 state elements on the top subplot.
# axs[0].scatter(all_times, first3_states[:, 0], label='Position ECI x-axis')
# axs[0].scatter(all_times, first3_states[:, 1], label='Position ECI y-axis')
# axs[0].scatter(all_times, first3_states[:, 2], label='Position ECI z-axis')
# axs[0].set_ylabel('Positions [m]')
# axs[0].set_title('State Elements vs. Time (Combined)')
#
# # Plot the last 3 state elements on the bottom subplot.
# axs[1].scatter(all_times, last3_states[:, 0], label='Velocity ECI x-axis')
# axs[1].scatter(all_times, last3_states[:, 1], label='Velocity ECI y-axis')
# axs[1].scatter(all_times, last3_states[:, 2], label='Velocity ECI z-axis')
# axs[1].set_xlabel('Seconds since initial epoch')
# axs[1].set_ylabel('Velocity [m/s]')
#
# for ax in axs:
#     ax.get_yaxis().get_major_formatter().set_useOffset(False)
#     ax.legend()
#     ax.grid(True)
#
# plt.tight_layout()
# plt.show()
#
# pos_norm = np.linalg.norm(first3_states, axis=1)  # Norm of position vectors
# vel_norm = np.linalg.norm(last3_states, axis=1)   # Norm of velocity vectors
#
# # Create an extra figure with two subplots for the norms.
# fig, axs = plt.subplots(2, 1, figsize=(8,5), sharex=True)
#
# # Plot the position norms (first 3 state elements) in the top subplot.
# axs[0].scatter(all_times, pos_norm, label='Position Norm', color='C0')
# axs[0].set_ylabel('Position Norm [m]')
# axs[0].set_title('State Norms vs. Time')
#
#
# # Plot the velocity norms (last 3 state elements) in the bottom subplot.
# axs[1].scatter(all_times, vel_norm, label='Velocity Norm', color='C1')
# axs[1].set_xlabel('Seconds since initial epoch')
# axs[1].set_ylabel('Velocity Norm [m/s]')
#
# for ax in axs:
#     ax.get_yaxis().get_major_formatter().set_useOffset(False)
#     ax.grid(True)
# plt.tight_layout()
# plt.show()
#
# # Create an extra figure with two subplots for the norms.
# fig, axs = plt.subplots(3, 1, figsize=(8, 5), sharex=True)
#
# # Plot the position norms (first 3 state elements) in the top subplot.
# axs[0].scatter(all_times_OG, all_kepler[:, 0], color='C0')
# axs[0].set_ylabel("Semi-major axis [m]")
#
#
# # Plot the velocity norms (last 3 state elements) in the bottom subplot.
# axs[1].scatter(all_times_OG, all_kepler[:, 1],  color='C1')
# axs[1].set_ylabel('Eccentricity [-]')
#
# # Plot the velocity norms (last 3 state elements) in the bottom subplot.
# axs[2].scatter(all_times_OG, np.rad2deg(all_kepler[:, 2]),  color='C2')
# axs[2].set_ylabel('Inclination')
# axs[2].set_xlabel('Seconds since initial epoch')
#
# for ax in axs:
#     ax.get_yaxis().get_major_formatter().set_useOffset(False)
#     ax.get_xaxis().get_major_formatter().set_useOffset(False)
#     ax.grid(True)
# plt.tight_layout()
# plt.show()
#
# # eccentricity_plot = all_kepler[:, 1] - all_kepler[0, 1]
# # plt.scatter(all_times_OG, eccentricity_plot)
# # plt.show()
#
# # Compute eccentricity difference between each timestep
# delta_e = all_kepler[:, 1] - all_kepler[0, 1]
#
# # Compute the difference between successive values in delta.
# diff_e = np.diff(delta_e)
#
# # Find indices where the absolute difference is greater than 0.02.
# jump_indices = np.where(np.abs(diff_e) > 0.02)[0]
#
# if jump_indices.size > 0:
#     # Get the first jump index. Here, the jump occurs between index i and i+1.
#     first_jump_index = jump_indices[0]
#
#     # The corresponding times:
#     pre_jump_time = all_times_OG[first_jump_index]
#     post_jump_time = all_times_OG[first_jump_index + 1]
#
#     print("First jump occurs between indices {} and {}.".format(first_jump_index, first_jump_index + 1))
#     print("Pre-jump time: {}".format(pre_jump_time))
#     print("Post-jump time: {}".format(post_jump_time))
# else:
#     print("No jump greater than 0.02 was found.")
# #Lambert targeter from Fundamentals of Astrodynamics
#
# # Given constants
# mu = bodies.get("Earth").gravitational_parameter # [m^3/s^2]
#
# # Given initial values
# initial_epoch = pre_jump_time # [s]
# final_epoch = post_jump_time # [s]
# departure_pos_initial_epoch = first3_states[first_jump_index, :] # [m]
# target_pos_final_epoch = first3_states[first_jump_index+1, :] # [m]
#
# # Solve Lambert's problem
# lambert_targeter = two_body_dynamics.LambertTargeterIzzo(
# 	departure_position = departure_pos_initial_epoch,
# 	arrival_position = target_pos_final_epoch,
# 	time_of_flight = final_epoch - initial_epoch,
# 	gravitational_parameter = mu)
#
# # Extract results
# v1, v2 = lambert_targeter.get_velocity_vectors()
#
# # Print the results
# np.set_printoptions(formatter={'float': '{: 9.3e}'.format})
# print('|V1|{tab} = {v1:6.3f}{tab} [km/s] {cr}|V2|{tab} = {v2:6.3f}{tab} [km/s]'.format(v1=np.linalg.norm(v1)/1e3, v2=np.linalg.norm(v2)/1e3, cr='\n', tab='\t'))
# print(' V1{tab} = {v1}{tab} [km/s] {cr} V2{tab} = {v2}{tab} [km/s]'.format(v1 = np.array2string(v1/1e3, formatter = {'float_kind': '{0:+6.3f}'.format}), v2 = np.array2string(v2/1e3, formatter = {'float_kind': '{0:+6.3f}'.format}), cr='\n', tab='\t'))
#
# DV1 = np.linalg.norm(v1) - np.linalg.norm(last3_states[first_jump_index, :])
# DV2 = np.linalg.norm(v2) - np.linalg.norm(last3_states[first_jump_index+1, :])
# print(f"DeltaV is: {DV1} + {DV2} = {DV1 + DV2} [m/s]")
#
# print(f"np.shape(all_covar[first_jump_index+1, :]): {np.shape(all_covar[first_jump_index+1, :])}")
#
# # ASD = Q2_state_params.copy()
# # ASD['state'] =
# # ASD['covar']
# rk4_int_params = {}
# rk4_int_params['tudat_integrator'] = 'rk4'
# rk4_int_params['step'] = 10.
#
# t_pre_maneuver_list = []
# state_pre_maneuver_list = []
# cov_pre_maneuver_list = []
# timespace = np.linspace(initial_epoch, final_epoch, 1000)
#
# for ii in range(len(timespace)):
#     time=timespace[ii]
#     #print(f"Iteration {ii}, end_time: {time}")
#     t_pre_maneuver, state_pre_maneuver, cov_pre_maneuver = propagate_state_and_covar(
#     all_states[first_jump_index, :], #x0
#     all_covar[first_jump_index, :], #p0
#     np.array([initial_epoch, time]),#t_vec
#     state_params, #state_params, bodies=
#     rk4_int_params, #int_params
#     bodies) #bodies
#     t_pre_maneuver_list.append(t_pre_maneuver)
#     state_pre_maneuver_list.append(state_pre_maneuver)
#     cov_pre_maneuver_list.append(cov_pre_maneuver)
#
# state_pre_maneuver = np.array(state_pre_maneuver_list)
# t_pre_maneuver = np.array(t_pre_maneuver_list)
# t_post_maneuver_list = []
# state_post_maneuver_list = []
# cov_post_maneuver_list = []
#
# for ii in range(len(timespace)):
#     time = timespace[(len(timespace)-1)-ii]
#     #print(f"Iteration {ii}, start time: {time}")
#     t_post_maneuver, state_post_maneuver, cov_post_maneuver = propagate_state_and_covar_back(
#     all_states[first_jump_index+1, :], #x0
#     all_covar[first_jump_index+1, :], #p0
#     np.array([time, final_epoch]),#t_vec
#     state_params, #state_params, bodies=
#     rk4_int_params, #int_params
#     bodies) #bodies
#     t_post_maneuver_list.append(t_post_maneuver)
#     state_post_maneuver_list.append(state_post_maneuver)
#     cov_post_maneuver_list.append(cov_post_maneuver)
#
# state_post_maneuver = np.array(state_post_maneuver_list)
# t_post_maneuver = np.array(t_post_maneuver_list)
# velocity_pre_maneuver = state_pre_maneuver[:, 3:6]
# velocity_pre_maneuver_norm= np.linalg.norm(velocity_pre_maneuver, axis = 1)
# # print(f"t_pre_maneuver_list: {t_pre_maneuver_list}")
# # print(f"velocity_evolution: {velocity_evolution}")
# velocity_post_maneuver = state_post_maneuver[:, 3:6]
# velocity_post_maneuver_norm = np.linalg.norm(velocity_post_maneuver, axis =1)
# # print(f"t_post_maneuver_list: {t_post_maneuver_list}")
# # print(f"velocity_post_maneuver: {velocity_post_maneuver}")
#
#
# plt.scatter(t_pre_maneuver_list, velocity_pre_maneuver_norm, color="C0")
# plt.scatter(t_post_maneuver_list, velocity_post_maneuver_norm, color="C1")
# plt.show()
#
# t_post_maneuver_forward = np.ones_like(t_post_maneuver)
# velocity_post_maneuver_forward = np.ones_like(state_post_maneuver[:, 3:6])
# delta_V = {}
# for ii in range(1000):
#     t_post_maneuver_forward[ii] = t_post_maneuver[(1000-1)-ii]
#     velocity_post_maneuver_current=velocity_post_maneuver[(1000-1)-ii, :]
#     velocity_post_maneuver_forward[ii, :] = velocity_post_maneuver_current
#     velocity_difference = velocity_post_maneuver_forward[ii, :] - velocity_pre_maneuver[ii, :]
#     print(f"{np.linalg.norm(velocity_difference)}: {velocity_difference} = {velocity_post_maneuver_forward[ii, :]} - {velocity_pre_maneuver[ii, :]}")
#     delta_V[f"{t_post_maneuver_forward[ii]}"] = np.linalg.norm(velocity_difference)
#
# #DEBUG:
# # print(t_post_maneuver_forward-t_pre_maneuver)
# # print(t_post_maneuver-t_pre_maneuver)
# plt.plot(np.linspace(0,1,1000), t_post_maneuver, linestyle = "--", color = "C0")
# plt.plot(np.linspace(0,1,1000), t_post_maneuver_forward, linestyle = "--", color = "C1")
# plt.plot(np.linspace(0,1,1000), t_pre_maneuver, linestyle = "--", color = "C1")
# plt.show()
#
# plt.plot(delta_V.keys(), delta_V.values())
# plt.show()

# rso_read = read_catalog_file(rso_meas_path)
# #print(f"rso_read: {rso_read}")
#
# def eci_to_ecef(r_eci, t, omega=7.2921150e-5):
#     """
#     Convert a position vector from ECI to ECEF coordinates given time t (in seconds since J2000).
#
#     Parameters:
#       - r_eci: A NumPy array representing the ECI vector [x, y, z] (meters).
#       - t: Time in seconds since J2000.
#       - omega: Earth's rotation rate (default ~7.2921150e-5 rad/s).
#
#     Returns:
#       A NumPy array representing the vector in ECEF coordinates.
#     """
#     theta = omega * t
#     # Rotation about Z-axis by -theta:
#     R = np.array([[np.cos(theta), np.sin(theta), 0],
#                   [-np.sin(theta), np.cos(theta), 0],
#                   [0, 0, 1]])
#     return R @ r_eci
#
#
# def plot_earth_with_trajectory_and_sensors_ecef(radius=6371e3,
#                                                 sensor_points=None,
#                                                 propagation_positions=None,
#                                                 propagation_time=None,
#                                                 sensor_params=None):
#     """
#     Plot a 3D Earth as a colourless (wireframe) sphere in ECEF coordinates, along with:
#       - A propagation trajectory provided in ECI coordinates (converted to ECEF using the supplied time array).
#       - Sensor positions in ECEF.
#       - Each sensor's field-of-view (FOV) frustum computed using the provided sensor parameters.
#
#     Parameters:
#       - radius: Earth radius (used for drawing the sphere) in meters.
#       - sensor_points: List of sensor position vectors [x, y, z] in ECEF (meters).
#       - propagation_positions: Array or list of propagation positions in ECI (meters).
#       - propagation_time: Array (or scalar) of times in seconds since J2000 corresponding to the propagation positions.
#       - sensor_params: Dictionary of sensor parameters, for example:
#             {
#                 'elevation_lim': [0.0872664626, 1.5707963268],
#                 'azimuth_lim':   [0.0, 6.2831853072],
#                 'range_lim':     [0.0, 5000000.0],
#                 'FOV_hlim':      [-0.0872664626, 0.0872664626],
#                 'FOV_vlim':      [-0.0872664626, 0.0872664626],
#                 'sun_elmask':    -3.1415926536
#             }
#     """
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#
#     # --- Plot Earth Wireframe in ECEF ---
#     u = np.linspace(0, 2 * np.pi, 100)
#     v = np.linspace(0, np.pi, 50)
#     x_earth = radius * np.outer(np.cos(u), np.sin(v))
#     y_earth = radius * np.outer(np.sin(u), np.sin(v))
#     z_earth = radius * np.outer(np.ones(np.size(u)), np.cos(v))
#     ax.plot_wireframe(x_earth, y_earth, z_earth, color='k', rstride=5, cstride=5)
#
#     # --- Plot Propagation Trajectory ---
#     if propagation_positions is not None:
#         propagation_positions = np.array(propagation_positions)
#         if propagation_positions.ndim != 2 or propagation_positions.shape[1] != 3:
#             raise ValueError("Propagation positions should be a 2D array with shape (N, 3)")
#         # Convert each propagation position in ECI to ECEF using its corresponding time.
#         if propagation_time is not None:
#             # Support either a scalar time (all points use the same t) or an array of times.
#             if np.isscalar(propagation_time):
#                 traj_ecef = np.array([eci_to_ecef(pt, propagation_time) for pt in propagation_positions])
#             else:
#                 propagation_time = np.array(propagation_time)
#                 if propagation_time.shape[0] != propagation_positions.shape[0]:
#                     raise ValueError("Length of propagation_time must match number of propagation positions")
#                 traj_ecef = np.array([eci_to_ecef(pt, t) for pt, t in zip(propagation_positions, propagation_time)])
#         else:
#             # If no time is provided, assume positions are already in ECEF.
#             traj_ecef = propagation_positions
#         ax.plot(traj_ecef[:, 0], traj_ecef[:, 1], traj_ecef[:, 2],
#                 color='g', linewidth=2, label='Trajectory')
#         ax.scatter(traj_ecef[:, 0], traj_ecef[:, 1], traj_ecef[:, 2],
#                    color='g', s=20)
#
#     # --- Plot Sensor Positions and Their FOVs ---
#     if sensor_points is not None:
#         for sensor_pos in sensor_points:
#             sensor_pos = np.array(sensor_pos)
#             ax.scatter(sensor_pos[0], sensor_pos[1], sensor_pos[2],
#                        color='r', s=50, label='Sensor')
#
#             if sensor_params is not None:
#                 # Build a local coordinate frame at the sensor's location in ECEF.
#                 up_vec = sensor_pos / np.linalg.norm(sensor_pos)
#                 # Local east vector: use [-y, x, 0]. (This works for sensors not at the pole.)
#                 east_vec = np.array([-sensor_pos[1], sensor_pos[0], 0])
#                 if np.linalg.norm(east_vec) < 1e-6:
#                     east_vec = np.array([0, 1, 0])
#                 else:
#                     east_vec = east_vec / np.linalg.norm(east_vec)
#                 north_vec = np.cross(up_vec, east_vec)
#                 north_vec = north_vec / np.linalg.norm(north_vec)
#
#                 # Determine the sensor’s boresight direction using the midpoints of the provided limits.
#                 elev_lim = sensor_params.get('elevation_lim', [0, np.pi / 2])
#                 azim_lim = sensor_params.get('azimuth_lim', [0, 2 * np.pi])
#                 elev_center = (elev_lim[0] + elev_lim[1]) / 2.0
#                 azim_center = (azim_lim[0] + azim_lim[1]) / 2.0
#
#                 d = (np.cos(elev_center) * np.cos(azim_center) * north_vec +
#                      np.cos(elev_center) * np.sin(azim_center) * east_vec +
#                      np.sin(elev_center) * up_vec)
#                 d = d / np.linalg.norm(d)
#
#                 # Compute "right" and "true up" directions for constructing the FOV frustum.
#                 sensor_right = np.cross(d, up_vec)
#                 if np.linalg.norm(sensor_right) < 1e-6:
#                     sensor_right = east_vec
#                 sensor_right = sensor_right / np.linalg.norm(sensor_right)
#                 sensor_true_up = np.cross(sensor_right, d)
#                 sensor_true_up = sensor_true_up / np.linalg.norm(sensor_true_up)
#
#                 # Use the sensor's far range (from sensor parameters) to define the FOV.
#                 range_lim = sensor_params.get('range_lim', [0, 1e6])
#                 far_range = range_lim[1]
#
#                 # Field-of-View limits (assumed symmetric about boresight).
#                 fov_h = sensor_params.get('FOV_hlim', [-0.1, 0.1])
#                 fov_v = sensor_params.get('FOV_vlim', [-0.1, 0.1])
#                 half_angle_h = abs(fov_h[1])
#                 half_angle_v = abs(fov_v[1])
#
#                 half_width = far_range * np.tan(half_angle_h)
#                 half_height = far_range * np.tan(half_angle_v)
#
#                 far_center = sensor_pos + far_range * d
#                 # Compute the four corners of the far-plane.
#                 top_right = far_center + sensor_right * half_width + sensor_true_up * half_height
#                 top_left = far_center - sensor_right * half_width + sensor_true_up * half_height
#                 bottom_right = far_center + sensor_right * half_width - sensor_true_up * half_height
#                 bottom_left = far_center - sensor_right * half_width - sensor_true_up * half_height
#
#                 # Draw lines from the sensor to each far-plane corner.
#                 for corner in [top_right, top_left, bottom_left, bottom_right]:
#                     line = np.vstack((sensor_pos, corner))
#                     ax.plot(line[:, 0], line[:, 1], line[:, 2], color='b', linestyle='--')
#                 # Connect the far-plane corners to outline the sensor's FOV.
#                 frustum_edges = np.array([top_left, top_right, bottom_right, bottom_left, top_left])
#                 ax.plot(frustum_edges[:, 0], frustum_edges[:, 1], frustum_edges[:, 2],
#                         color='b', linestyle='-')
#
#     ax.set_box_aspect([1, 1, 1])
#     ax.set_xlabel("X [m]")
#     ax.set_ylabel("Y [m]")
#     ax.set_zlabel("Z [m]")
#     plt.legend()
#     plt.show()
#
# # Example points: (latitude, longitude, height)
# # For instance, (0, 0, 0) is on the Equator at the prime meridian,
# # (45, 90, 1000) is 45°N, 90°E with an elevation of 1 km.
# sensor_points = Q2_sensor_params['sensor_ecef']
# print(type(sensor_points))
# print(f"sensor_points: {sensor_points}")
# # Sensor parameters as provided.
# sensor_params = {
#     'elevation_lim': [0.08726646259971647, 1.5707963267948966],
#     'azimuth_lim': [0.0, 6.283185307179586],
#     'range_lim': [0.0, 5000000.0],
#     'FOV_hlim': [-0.08726646259971647, 0.08726646259971647],
#     'FOV_vlim': [-0.08726646259971647, 0.08726646259971647],
#     'sun_elmask': -3.141592653589793
# }
# plot_earth_with_trajectory_and_sensors_ecef(radius=6371e3,
#                                                 sensor_points=sensor_points,
#                                                 propagation_positions=propagated_positions,
#                                                 propagation_time=t_prop_Q2,
#                                                 sensor_params=sensor_params)
