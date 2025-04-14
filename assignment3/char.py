"""Small introduction to the object characterisation question."""
import numpy as np
from pathlib import Path
from EstimationUtilities import read_measurement_file, ukf, ukf_full, get_pos_vectors, compute_magnitude_in_time, compute_magnitude
from ConjunctionUtilities import read_catalog_file
from TudatPropagator import tudat_initialize_bodies, propagate_orbit_wdepvars
from dataclasses import dataclass
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.style as style
from tudatpy.numerical_simulation import propagation_setup

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

    first_filter_output = ukf(state_params, meas_dict, sensor_params, int_params, filter_params, bodies)
    residuals_array = [first_filter_output[i]["resids"].flatten() for i in list(first_filter_output.keys())]
    times = np.array(list(first_filter_output.keys())) / 60.0
    times -= times[0]
    residuals_array = np.array(residuals_array)

    # Convert the second and third elements of each residual to degrees
    residuals_array[:, 1] = np.degrees(residuals_array[:, 1])  # Convert alpha_T to degrees
    residuals_array[:, 2] = np.degrees(residuals_array[:, 2])  # Convert delta_T to degrees

    fig, axs = plt.subplots(3, 1, figsize=(8, 12))
    labels = ['$\\rho$', '$\\alpha_T$', '$\\delta_T$']
    ylabels_units = ["m", "deg", "deg"]

    for i in range(3):
        residuals = residuals_array[:, i]
        mean_residual = np.mean(residuals)
        std_residual = np.std(residuals)

        axs[i].plot(residuals, label=f'Residuals in {labels[i]}')
        axs[i].axhline(mean_residual, color='r', linestyle='--', label='Mean')
        axs[i].axhline(mean_residual + std_residual, color='g', linestyle='--', label='Mean ± Std Dev')
        axs[i].axhline(mean_residual - std_residual, color='g', linestyle='--')
        axs[i].set_xlabel('Time since start [min]')
        axs[i].set_ylabel(f'Residual [{ylabels_units[i]}]')
        axs[i].set_title(f'Residuals in {labels[i]}')
        axs[i].legend()
        axs[i].grid()

    plt.tight_layout()


    # ### Observe the magnitude measurements and the residuals
    magnitudes = np.array(optical_data.meas_dict["Yk_list"]).flatten()
    times = np.array(optical_data.meas_dict["tk_list"]).flatten()
    # times -= times[0]
    plt.figure()

    plt.plot(magnitudes)
    plt.title("Magnitude Measurements")
    plt.xlabel("Measurement Index")
    plt.ylabel("Magnitude")

    plt.grid()
    plt.tight_layout()

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

    plt.figure()

    plt.plot(model_magnitudes_trim)
    plt.title("Magnitude model")
    plt.xlabel("Measurement Index")
    plt.ylabel("Magnitude")

    plt.grid()
    plt.tight_layout()

    
    plt.show()

    # Run filter
    # filter_output = ukf_full(state_params, meas_dict, sensor_params, params_variance, int_params, filter_params, bodies)  # test with full parameters

    # final_corr = covariance_to_correlation(filter_output[meas_dict["tk_list"][-1]]["covar"])  # covariance matrix to correlation matrix
    
    # print("Keys in state_params:", list(state_params.keys()))

    # # Compare estimated parameters
    # estimated_params = filter_output[meas_dict["tk_list"][-1]]["state"]
    # state_params_vec = [
    #     state_params["state"],
    #     state_params["mass"],
    #     state_params["area"],
    #     state_params["Cd"],
    #     state_params["Cr"],
    # ]
    # print("Estimated parameters:")
    # print(estimated_params) 
    # print("State parameters vector:")
    # print(state_params_vec)

    # print("Standard deviations:")
    # print(np.sqrt(np.diag(filter_output[meas_dict["tk_list"][-1]]["covar"])))

    # # Plot covariance matrix: (it can be 'corr', 'corr_abs', 'corr_log')
    # plot_correlation_matrix(final_corr, labels, ['corr_abs', 'corr_log'])
    # plt.show()
    