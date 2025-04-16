"""Small introduction to the object characterisation question."""
import numpy as np
from pathlib import Path
from EstimationUtilities import (
    read_measurement_file, ukf, ukf_full, get_pos_vectors,
    compute_magnitude_in_time, compute_magnitude, model_magnitude_meas
)
from scipy.optimize import curve_fit
from ConjunctionUtilities import read_catalog_file
from TudatPropagator import tudat_initialize_bodies, propagate_orbit_wdepvars
from dataclasses import dataclass
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.style as style
from tudatpy.numerical_simulation import propagation_setup

import contextlib
import io

# # Constants
MAG_SUN = -26.74

# # Directories
main_directory = Path(__file__).parent.parent  # A3G5-space-debris-2025 directory (root) 
assignment_data_directory = main_directory.joinpath('assignment3/data/group5')
style.use(main_directory.joinpath("default/default.mplstyle"))

# Data files directories
catalog_data_file = assignment_data_directory.joinpath('estimated_rso_catalog.pkl')
radar_measurement_data_file = assignment_data_directory.joinpath('q3_radar_meas_objchar_91662.pkl')
optical_measurement_data_file = assignment_data_directory.joinpath('q3_optical_meas_objchar_91662.pkl')

# # Setup variables for the main script
@dataclass
class MeasurementData:
    def __init__(self, state_params, meas_dict, sensor_params):
        self.state_params = state_params
        self.meas_dict = meas_dict
        self.sensor_params = sensor_params


# Integration parameters
# TODO: justify them
int_params = {}
int_params['tudat_integrator'] = 'rkf78'
int_params['step'] = 10.
int_params['max_step'] = 1000.
int_params['min_step'] = 1e-3
int_params['rtol'] = 1e-12
int_params['atol'] = 1e-12

# Integration parameters for magnitude (fixed step)
int_params_mag = {
    "tudat_integrator": "rk4",
    "step": 1
}

# Dependent variables
depvars = [
    propagation_setup.dependent_variable.relative_position("Sun", "Earth"),
    propagation_setup.dependent_variable.inertial_to_body_fixed_rotation_frame("Earth")
]

# UKF parameters
Qeci = 1e-12*np.diag([1., 1., 1.])
Qric = 1e-12*np.diag([1., 1., 1.])

filter_params = {}
filter_params['Qeci'] = Qeci
filter_params['Qric'] = Qric
filter_params['alpha'] = 1.
filter_params['gap_seconds'] = 600.

bodies_to_create = ['Earth', 'Sun', 'Moon']
bodies = tudat_initialize_bodies(bodies_to_create)

def covariance_to_correlation(cov):
    stddev = np.sqrt(np.diag(cov))
    corr = cov / np.outer(stddev, stddev)
    corr[cov == 0] = 0  # to handle division by zero if any
    return corr

def split_segments(x, y, gap_indices):
    segments = []
    start = 0
    for idx in gap_indices:
        segments.append((x[start:idx+1], y[start:idx+1]))
        start = idx + 1
    segments.append((x[start:], y[start:]))
    return segments

def is_float(value):
    try:
        float(value)
        return True
    except ValueError:
        return False
    
def get_closest_key(d, target):
    numeric_keys = [k for k in d.keys() if is_float(k)]
    return min(numeric_keys, key=lambda k: abs(float(k) - target))

labels = [
    "$r_x$", "$r_y$", "$r_z$",
    "$v_x$", "$v_y$", "$v_z$",
    "$C_d$", "$C_r$"
]

def print_start_iter(i):
    print("#######################")
    print()
    print(f"### STARTING ITER {i+1} ###")
    print()
    print("#######################")

def plot_correlation_matrix(cor, labels, types_to_plot):
    for type in types_to_plot:
        if type == 'corr_abs':
            corr_to_plot = np.abs(cor)
            title = "Correlation Matrix Heatmap - Absolute values"
        elif type == 'corr_log':
            corr_to_plot = np.log10(np.abs(cor))  # Add epsilon to avoid log(0)
            title = "Correlation Matrix Heatmap - Log scale"
        else:
            corr_to_plot = cor
            title = "Correlation Matrix Heatmap"
        
        plt.figure(figsize=(10, 8))
        ax = sns.heatmap(
            corr_to_plot,
            annot=True,
            cmap='viridis',
            fmt=".2f",
            cbar=True,
            linewidths=.9,
            linecolor='white',
            square=True
        )

        ax.set_title(title)
        ax.set_xlabel("Parameters")
        ax.set_ylabel("Parameters")

        # Set LaTeX-style parameter labels
        ax.set_xticks(np.arange(len(labels)) + 0.5)  # +0.5 centers labels in seaborn heatmap
        ax.set_yticks(np.arange(len(labels)) + 0.5)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_yticklabels(labels, rotation=0)

        ax.tick_params(bottom=False, left=False)
        ax.grid(False)

        plt.tight_layout()

if __name__ == "__main__":
    # Load the data from the pickle files
    rso_dict = read_catalog_file(catalog_data_file)
    optical_data = MeasurementData(*read_measurement_file(optical_measurement_data_file))
    radar_data = MeasurementData(*read_measurement_file(radar_measurement_data_file))

    sensor_params = radar_data.sensor_params
    meas_dict = radar_data.meas_dict
    state_params = radar_data.state_params

    # ### First analysis
    # Visualize the residuals from standard UKF and determine whether useful 
    # information can be extracted by better modelling SRP and drag

    # first_filter_output = ukf(state_params, meas_dict, sensor_params, int_params, filter_params, bodies)
    # residuals_array = [first_filter_output[i]["resids"].flatten() for i in list(first_filter_output.keys())]
    # times = np.array(list(first_filter_output.keys())) / 60.0
    # times -= times[0]
    # residuals_array = np.array(residuals_array)

    # # Convert the second and third elements of each residual to degrees
    # residuals_array[:, 1] = np.degrees(residuals_array[:, 1])  # Convert alpha_T to degrees
    # residuals_array[:, 2] = np.degrees(residuals_array[:, 2])  # Convert delta_T to degrees

    # fig, axs = plt.subplots(3, 1, figsize=(8, 12))
    # labels = ['$\\rho$', '$\\alpha_T$', '$\\delta_T$']
    # ylabels_units = ["m", "deg", "deg"]

    # for i in range(3):
    #     residuals = residuals_array[:, i]
    #     mean_residual = np.mean(residuals)
    #     std_residual = np.std(residuals)

    #     axs[i].plot(residuals, label=f'Residuals in {labels[i]}')
    #     axs[i].axhline(mean_residual, color='r', linestyle='--', label='Mean')
    #     axs[i].axhline(mean_residual + std_residual, color='g', linestyle='--', label='Mean ± Std Dev')
    #     axs[i].axhline(mean_residual - std_residual, color='g', linestyle='--')
    #     axs[i].set_xlabel('Time since start [min]')
    #     axs[i].set_ylabel(f'Residual [{ylabels_units[i]}]')
    #     axs[i].set_title(f'Residuals in {labels[i]}')
    #     axs[i].legend()
    #     axs[i].grid()

    # plt.tight_layout()


    # ### Observe the magnitude measurements and the residuals
    magnitudes = np.array(optical_data.meas_dict["Yk_list"]).flatten()
    times = np.array(optical_data.meas_dict["tk_list"]).flatten()
    # times -= times[0]

    tout, Xout, depvars_history = propagate_orbit_wdepvars(
        np.array(optical_data.state_params['state']),
        [optical_data.state_params["epoch_tdb"], optical_data.meas_dict["tk_list"][-1]],
        optical_data.state_params,
        int_params_mag,
        bodies,
        depvars
    )

    # Mask used to compare meas and model values only at the tk_list times
    mask = np.isin(tout, optical_data.meas_dict["tk_list"])
    tout_trim = tout[mask]

    sun_position_vector, obs_position_vector, ss_position_vector = get_pos_vectors(
        Xout[:,:3], depvars_history, optical_data.sensor_params
    )

    model_magnitudes = compute_magnitude_in_time(
        sun_position_vector, obs_position_vector, ss_position_vector,
        state_params['area'], state_params['Cr'], MAG_SUN
    )

    model_magnitudes_trim = model_magnitudes[mask]

    residuals = magnitudes - model_magnitudes_trim
    time_hours = (tout_trim - tout_trim[0]) / 3600

    # Detect gap — assume a time jump > 10x the median time step is a gap
    time_diff = np.diff(time_hours)
    gap_idx = np.where(time_diff > 10 * np.median(time_diff))[0]

    # Split all three plots
    time_segments = split_segments(time_hours, time_hours, gap_idx)
    measured_segments = split_segments(time_hours, magnitudes, gap_idx)
    model_segments = split_segments(time_hours, model_magnitudes_trim, gap_idx)
    residual_segments = split_segments(time_hours, residuals, gap_idx)

    # Pre-fit residuals
    plt.figure()

    temp = None
    for i, (t_seg, r_seg) in enumerate(residual_segments):
        label = "Residuals" if i == 0 else None
        temp, = plt.plot(t_seg, r_seg, label=label, color=temp.get_color() if temp is not None else None)

    plt.title("Pre-fit modelled and measured magnitudes residuals")
    plt.xlabel("Time [hours]")
    plt.ylabel("Magnitude")
    plt.grid(True)
    plt.tight_layout()
    plt.legend()

    # Plot 2: Measured vs Modelled
    plt.figure()

    temp = None
    for i, (t_seg, m_seg) in enumerate(measured_segments):
        label = "Measured" if i == 0 else None
        temp, = plt.plot(t_seg, m_seg, label=label, color=temp.get_color() if temp is not None else None)

    temp = None
    for i, (t_seg, mod_seg) in enumerate(model_segments):
        label = "Modelled" if i == 0 else None
        temp, = plt.plot(t_seg, mod_seg, label=label, color=temp.get_color() if temp is not None else None)

    plt.title("Modelled (pre-fit) and measured magnitudes")
    plt.xlabel("Time [hours]")
    plt.ylabel("Magnitude")
    plt.legend(loc="best")
    plt.grid(True)
    plt.tight_layout()

    # ### Final estimation
    MAX_ITERS = 10
    TOL_MAX_RESID = [20, 0.05, 0.05, 0.005]
    TOL_PARAM_DIFF = 1e-5
    state_params['area'] = 10
    prev_param = state_params['area']
    current_state_params = state_params
    final_residuals = {
        "radar_resid_tk": None,
        "radar_resid": None,
        "optical_resid_tk": None,
        "optical_resid": None
    }

    # Iterate until tolerance reached or max number of iters
    for iter_number in range(MAX_ITERS):
        print_start_iter(iter_number)
        comb_max_resid = np.empty(4)
        ss_position_vector = []

        # # UKF step
        print("UKF estimation...")
        with contextlib.redirect_stdout(io.StringIO()):
            filter_output = ukf(current_state_params, meas_dict, sensor_params, int_params, filter_params, bodies)
        residuals_array = np.array([filter_output[i]["resids"].flatten() for i in list(filter_output.keys())])

        comb_max_resid[:3] = np.max(residuals_array, axis=0)
        ukf_times = list(filter_output.keys())
        final_residuals["radar_resid_tk"] = ukf_times
        final_residuals["radar_resid"] = residuals_array

        updated_state = filter_output[ukf_times[0]]["state"]

        # # Propagate current state estimation
        print("Propagating...")
        tout, Xout, depvars_history = propagate_orbit_wdepvars(
            updated_state,
            [current_state_params["epoch_tdb"], optical_data.meas_dict["tk_list"][-1]],
            current_state_params,
            int_params_mag,
            bodies,
            depvars
        )

        magnitudes = np.array(optical_data.meas_dict["Yk_list"]).flatten()
        times = np.array(optical_data.meas_dict["tk_list"]).flatten()

        mask = np.isin(tout, optical_data.meas_dict["tk_list"])
        tout_trim = tout[mask]

        sun_position_vector, obs_position_vector, ss_position_vector = get_pos_vectors(
            Xout[:,:3], depvars_history, optical_data.sensor_params
        )

        sun_position_vector = sun_position_vector[mask]
        obs_position_vector = obs_position_vector[mask]
        ss_position_vector = ss_position_vector[mask]

        # # Magnitude non-linear fit step
        print("Fitting the magnitude obs...")
        xdata = np.column_stack((sun_position_vector, obs_position_vector, ss_position_vector))
        ydata = magnitudes
        # p0 = [current_state_params["area"], current_state_params["Cr"]]
        p0 = current_state_params["area"]
        print(f"- params0: {p0}")
        param, param_cov = curve_fit(model_magnitude_meas, xdata, ydata, p0, bounds=[0.01,100])
        print(f"- params: {param}")
        print(f"- cov: {param_cov}")
        print("Saving current estimation...")
        final_param_cov = param_cov
        current_state_params["area"] = param

        magnitude_residuals = model_magnitude_meas(xdata, param) - ydata
        comb_max_resid[3] = np.max(magnitude_residuals)

        final_residuals["optical_resid_tk"] = tout_trim
        final_residuals["optical_resid"] = magnitude_residuals

        relative_diff = (param - prev_param) / param
        prev_param = param

        if np.all(comb_max_resid < TOL_MAX_RESID) or relative_diff < TOL_PARAM_DIFF:
            state_params = current_state_params
            break

    # Plotting
    fig, axs = plt.subplots(3, 1, figsize=(8, 12))
    labels = ['$\\rho$', '$\\alpha_T$', '$\\delta_T$']
    ylabels_units = ["m", "deg", "deg"]

    for i in range(3):
        residuals = final_residuals["radar_resid"][:, i]
        mean_residual = np.mean(residuals)
        std_residual = np.std(residuals)

        axs[i].plot(residuals, label=f'Residuals in {labels[i]}')
        axs[i].axhline(mean_residual, color='r', linestyle='--', label='Mean')
        axs[i].axhline(mean_residual + std_residual, color='g', linestyle='--', label='Mean ± Std Dev')
        axs[i].axhline(mean_residual - std_residual, color='g', linestyle='--')
        # axs[i].set_xlabel('Time since start [min]') # TODO: MAKE THIS LOOK BETTER
        axs[i].set_ylabel(f'Residual [{ylabels_units[i]}]')
        axs[i].set_title(f'Residuals in {labels[i]}')
        axs[i].legend()
        axs[i].grid()

    plt.tight_layout()

    magnitudes = np.array(optical_data.meas_dict["Yk_list"]).flatten()
    times = np.array(optical_data.meas_dict["tk_list"]).flatten()
    # times -= times[0]

    tout, Xout, depvars_history = propagate_orbit_wdepvars(
        np.array(optical_data.state_params['state']),
        [optical_data.state_params["epoch_tdb"], optical_data.meas_dict["tk_list"][-1]],
        optical_data.state_params,
        int_params_mag,
        bodies,
        depvars
    )

    # Mask used to compare meas and model values only at the tk_list times
    mask = np.isin(tout, optical_data.meas_dict["tk_list"])
    tout_trim = tout[mask]

    sun_position_vector, obs_position_vector, ss_position_vector = get_pos_vectors(
        Xout[:,:3], depvars_history, optical_data.sensor_params
    )

    model_magnitudes = compute_magnitude_in_time(
        sun_position_vector, obs_position_vector, ss_position_vector,
        state_params['area'], state_params['Cr'], MAG_SUN
    )

    model_magnitudes_trim = model_magnitudes[mask]

    residuals = magnitudes - model_magnitudes_trim
    time_hours = (tout_trim - tout_trim[0]) / 3600

    # Detect gap — assume a time jump > 10x the median time step is a gap
    time_diff = np.diff(time_hours)
    gap_idx = np.where(time_diff > 10 * np.median(time_diff))[0]

    # Split all three plots
    time_segments = split_segments(time_hours, time_hours, gap_idx)
    measured_segments = split_segments(time_hours, magnitudes, gap_idx)
    model_segments = split_segments(time_hours, model_magnitudes_trim, gap_idx)
    residual_segments = split_segments(time_hours, residuals, gap_idx)

    # Pre-fit residuals
    plt.figure()

    temp = None
    for i, (t_seg, r_seg) in enumerate(residual_segments):
        label = "Residuals" if i == 0 else None
        temp, = plt.plot(t_seg, r_seg, label=label, color=temp.get_color() if temp is not None else None)

    plt.title("Post-fit modelled and measured magnitudes residuals")
    plt.xlabel("Time [hours]")
    plt.ylabel("Magnitude")
    plt.grid(True)
    plt.tight_layout()
    plt.legend()

    # Plot 2: Measured vs Modelled
    plt.figure()

    temp = None
    for i, (t_seg, m_seg) in enumerate(measured_segments):
        label = "Measured" if i == 0 else None
        temp, = plt.plot(t_seg, m_seg, label=label, color=temp.get_color() if temp is not None else None)

    temp = None
    for i, (t_seg, mod_seg) in enumerate(model_segments):
        label = "Modelled" if i == 0 else None
        temp, = plt.plot(t_seg, mod_seg, label=label, color=temp.get_color() if temp is not None else None)

    plt.title("Modelled (post-fit) and measured magnitudes")
    plt.xlabel("Time [hours]")
    plt.ylabel("Magnitude")
    plt.legend(loc="best")
    plt.grid(True)
    plt.tight_layout()

    plt.show()