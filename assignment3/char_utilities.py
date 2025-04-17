"""Utility functions for plots in characterisation."""
from matplotlib import pyplot as plt
from matplotlib import gridspec
import numpy as np
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

def plot_radar_residuals(time, residuals_matrix, title=False):
    time = (time - time[0]) / 60  # Convert time to minutes
    residuals_matrix[:, 1] = np.degrees(residuals_matrix[:, 1])
    residuals_matrix[:, 2] = np.degrees(residuals_matrix[:, 2])

    # Print RMS
    rms_range = np.sqrt(np.mean(residuals_matrix[:, 0]**2))
    rms_ra = np.sqrt(np.mean(residuals_matrix[:, 1]**2))
    rms_dec = np.sqrt(np.mean(residuals_matrix[:, 2]**2))

    print()
    print(f"RMS range: {rms_range:.4f} m")
    print(f"RMS RA: {rms_ra:.4f} deg")
    print(f"RMS DEC: {rms_dec:.4f} deg")

    # Split time and residuals into arcs based on gaps
    time_diff = np.diff(time)
    gap_idx = np.where(time_diff > 10 * np.median(time_diff))[0]
    arc_start_idxs = np.insert(gap_idx + 1, 0, 0)
    arc_end_idxs = np.append(gap_idx, len(time) - 1)

    # Labels
    labels = ['$\\rho$', '$\\alpha_T$', '$\\delta_T$']
    ylabels_units = ["m", "deg", "deg"]
    names = ["Range", "RA", "DEC"]

    # Plot for each residual component
    print()
    for i in range(3):
        # ➤ Compute mean and std once across entire dataset
        full_resid = residuals_matrix[:, i]
        mean_res = np.mean(full_resid)
        std_res = np.std(full_resid)

        print(f"Mean {names[i]}: {mean_res} {ylabels_units[i]}")
        print(f"Sigma {names[i]}: {std_res} {ylabels_units[i]}")

        n_arcs = len(arc_start_idxs)
        fig = plt.figure(figsize=(4 * n_arcs, 4))
        gs = gridspec.GridSpec(1, n_arcs, width_ratios=[1] * n_arcs, wspace=0.05)

        y_min, y_max = float('inf'), float('-inf')
        axes = []

        # First pass: plot residuals and calculate global y-limits
        for j in range(n_arcs):
            idxs = slice(arc_start_idxs[j], arc_end_idxs[j] + 1)
            arc_time = time[idxs]
            arc_resid = residuals_matrix[idxs, i]

            ax = fig.add_subplot(gs[j])
            axes.append(ax)

            ax.plot(arc_time, arc_resid, label=f'Residuals')
            ax.axhline(mean_res, color='r', linestyle='--', label='Mean' if j == 0 else "")
            ax.axhline(mean_res + std_res, color='g', linestyle='--', label='Mean ± Std Dev' if j == 0 else "")
            ax.axhline(mean_res - std_res, color='g', linestyle='--')

            y_min = min(y_min, np.min(arc_resid))
            y_max = max(y_max, np.max(arc_resid))

            # Clean up axis edges
            if j > 0:
                ax.spines['left'].set_visible(False)
                ax.tick_params(labelleft=False)
            if j < n_arcs - 1:
                ax.spines['right'].set_visible(False)
                ax.tick_params(labelright=False)

            ax.set_xlim(arc_time[0], arc_time[-1])
            ax.set_xlabel("Time [min]")
            ax.grid()

        # Set common y-label and y-limits
        axes[0].set_ylabel(f'Residual [{ylabels_units[i]}]')
        for ax in axes:
            ax.set_ylim(y_min, y_max)

        if title:
            axes[0].set_title(f'Residuals in {labels[i]}')
        handles, labels_ = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels_, loc='upper right')


def split_segments(x, y, gap_indices):
    segments = []
    start = 0
    for idx in gap_indices:
        segments.append((x[start:idx+1], y[start:idx+1]))
        start = idx + 1
    segments.append((x[start:], y[start:]))
    return segments

def plot_mag_residuals(time, residuals_vec, title=False):
    time = (time - time[0]) * 60  # Time in seconds

    # Print RMS
    rms_mag = np.sqrt(np.mean(residuals_vec**2))
    print()
    print(f"RMS mag: {rms_mag}")
    print(f"Mean mag residuals: {np.mean(residuals_vec)}")

    # ➤ Compute shared mean and std for the whole dataset
    mean_res = np.mean(residuals_vec)
    std_res = np.std(residuals_vec)

    # Split time and residuals into arcs based on gaps
    time_diff = np.diff(time)
    gap_idx = np.where(time_diff > 10 * np.median(time_diff))[0]
    arc_start_idxs = np.insert(gap_idx + 1, 0, 0)
    arc_end_idxs = np.append(gap_idx, len(time) - 1)

    # Labels
    labels = "magnitude"
    ylabels_units = "-"

    n_arcs = len(arc_start_idxs)
    fig = plt.figure(figsize=(4 * n_arcs, 4))
    gs = gridspec.GridSpec(1, n_arcs, width_ratios=[1] * n_arcs, wspace=0.05)

    y_min, y_max = float('inf'), float('-inf')
    axes = []

    # Plot each arc
    for j in range(n_arcs):
        idxs = slice(arc_start_idxs[j], arc_end_idxs[j] + 1)
        arc_time = time[idxs]
        arc_resid = residuals_vec[idxs]

        ax = fig.add_subplot(gs[j])
        axes.append(ax)

        ax.plot(arc_time, arc_resid, label=f'Residuals')
        ax.axhline(mean_res, color='r', linestyle='--', label='Mean' if j == 0 else "")
        ax.axhline(mean_res + std_res, color='g', linestyle='--', label='Mean ± Std Dev' if j == 0 else "")
        ax.axhline(mean_res - std_res, color='g', linestyle='--')

        y_min = min(y_min, np.min(arc_resid))
        y_max = max(y_max, np.max(arc_resid))

        # Clean up axis edges
        if j > 0:
            ax.spines['left'].set_visible(False)
            ax.tick_params(labelleft=False)
        if j < n_arcs - 1:
            ax.spines['right'].set_visible(False)
            ax.tick_params(labelright=False)

        ax.set_xlim(arc_time[0], arc_time[-1])
        ax.set_xlabel("Time [min]")
        ax.grid()

    # Set common y-label and y-limits
    axes[0].set_ylabel(f'Residual [{ylabels_units}]')
    for ax in axes:
        ax.set_ylim(y_min, y_max)

    if title:
        axes[0].set_title(f'Residuals in {labels}')
    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc='upper right')


def covariance_to_correlation(cov):
    stddev = np.sqrt(np.diag(cov))
    corr = cov / np.outer(stddev, stddev)
    corr[cov == 0] = 0  # to handle division by zero if any
    return corr

@dataclass
class MeasurementData:
    def __init__(self, state_params, meas_dict, sensor_params):
        self.state_params = state_params
        self.meas_dict = meas_dict
        self.sensor_params = sensor_params


def print_estimation_results(state_params):
    print()
    print(f"FINAL ESTIMATION RESULTS:")
    print(f"Mass: {state_params['mass']} kg")
    print(f"Area: {state_params['area'][0]} m^2")
    print(f"Cd: {state_params['Cd']}")
    print(f"Cr: {state_params['Cr']}")
