import sys
import numpy as np
import h5py
import matplotlib.animation as animation
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from scipy.stats import chisquare
import os

# Add the parent directory to the python path
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, '..'))
plots_dir = os.path.join(script_dir, 'generated_plots')
sys.path.append(parent_dir)

# Use the custom matplotlib style from pycosmohub if available
mplstyle_path = os.path.join(parent_dir, 'pycosmohub.mplstyle')
if os.path.exists(mplstyle_path):
    plt.style.use(mplstyle_path)


mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",  # Or "sans-serif", or specific LaTeX font
    "font.serif": ["Computer Modern Roman"],  # Default LaTeX font
})

# Check if directory for plots exists, if not create it
os.makedirs(plots_dir, exist_ok=True)

def return_integral(series, ax, num_bins, position='top', ngenic=True):
    """
    Calculate the integral of the histogram of a specified column in a dataset and annotate it on the plot axis.
    
    Parameters:
    - position: 'top' or 'below' — vertical placement of the annotation
    """
    counts, bins = np.histogram(series, bins=num_bins)
    bin_width = bins[1] - bins[0]
    integral = counts.sum() * bin_width

    # Determine vertical position
    if position == 'top':
        y_pos = 0.95
    elif position == 'below':
        y_pos = 0.875
    else:
        y_pos = 0.95  # default


    if ngenic:  
        label = fr"N-GenIC Integral: {integral:.2f}"
    else:
        label = f"PyCosmo Integral: {integral:.2f}"

    ax.text(
        0.95, y_pos,
        label,
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7)
    )
    return integral

def plot_comparison_histograms(ngenic_results, pycosmo_results, column_name, units, titles, filename, num_bins=100):
    """
    Plot histograms of two datasets on the same graph and return the integral of both.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot histograms on the same axis
    ax.hist(ngenic_results, bins=num_bins, color='blue', label=titles[0], alpha=0.7, histtype='stepfilled')
    ax.hist(pycosmo_results, bins=num_bins, color='orange', label=titles[1], alpha=0.7, histtype='stepfilled')

    # Label and title
    ax.set_xlabel(rf"${column_name} \ [{units[0]}]$", fontsize=16)
    ax.set_ylabel(r"Frequency", fontsize=16)
    ax.grid(True)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.legend(fontsize=14)

    # Optionally return integrals for both
    return_integral(ngenic_results, ax, num_bins, position='top', ngenic=True)
    return_integral(pycosmo_results, ax, num_bins, position='below', ngenic=False)

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, filename))
    plt.close(fig)


def plot_phase_space_diagrams(ngenic_results, pycosmo_results, num_bins=100):
    coord_labels = ['x', 'y', 'z']
    vel_labels = ['v_x', 'v_y', 'v_z']

    for i, column_name in enumerate(coord_labels):
        plot_comparison_histograms(
            ngenic_results[column_name],
            pycosmo_results[column_name],
            column_name,
            units=[r"\mathrm{Mpc}/h", r"\mathrm{Mpc}/h"],
            titles=["N-GenIC " + fr"${column_name}$" + " coordinates", "PyCosmo " + fr"${column_name}$" + " coordinates"],
            filename=f"{column_name}_coords_phase_space.pdf",
            num_bins=num_bins
        )

    for i, column_name in enumerate(vel_labels, start=3):
        plot_comparison_histograms(
            ngenic_results[column_name],
            pycosmo_results[column_name],
            column_name,
            units=['km/s', 'km/s'],
            titles=["N-GenIC " + fr"${column_name}$" + " values", "PyCosmo " + fr"${column_name}$" + " values"],
            filename=f"{column_name}_velocities_phase_space.pdf",
            num_bins=num_bins
        )

print("Creating histograms of final timestep...")

ngenic_path = os.path.join(script_dir, 'structure_formation_data/', 'final_ngenic_snapshot.csv')
full_ngenic_data = np.loadtxt(ngenic_path, delimiter=',', dtype=float, usecols = (0,1,2,3,4,5))
full_ngenic_data[:, :3] /= 1000
ngenic_headers = ['x', 'y', 'z', 'v_x', 'v_y', 'v_z']
ngenic_df = pd.DataFrame(full_ngenic_data, columns=ngenic_headers)

pycosmo_path = os.path.join(script_dir, 'structure_formation_data/', 'final_pycosmo_snapshot.csv')
full_pycosmo_data = np.loadtxt(pycosmo_path, delimiter=',', dtype=float, usecols = (0,1,2,3,4,5))
full_pycosmo_data[:, :3] /= 1000
pycosmo_headers = ['x', 'y', 'z', 'v_x', 'v_y', 'v_z']
pycosmo_df = pd.DataFrame(full_pycosmo_data, columns=ngenic_headers)

plot_phase_space_diagrams(ngenic_df, pycosmo_df)


def run_chi_squared_tests(ngenic_df, pycosmo_df, num_bins=100):
    print("\nChi-squared analysis of histogram differences:\n")
    columns = ['x', 'y', 'z', 'v_x', 'v_y', 'v_z']

    for col in columns:
        ngenic_vals = ngenic_df[col]
        pycosmo_vals = pycosmo_df[col]

        # Use common bins for both histograms
        all_data = np.concatenate((ngenic_vals, pycosmo_vals))
        bins = np.linspace(all_data.min(), all_data.max(), num_bins + 1)

        ngenic_hist, _ = np.histogram(ngenic_vals, bins=bins)
        pycosmo_hist, _ = np.histogram(pycosmo_vals, bins=bins)

        # Mask out zero expected bins to avoid divide-by-zero
        mask = pycosmo_hist > 0
        obs = ngenic_hist[mask]
        exp = pycosmo_hist[mask]

        if len(obs) == 0:
            print(f"{col}: Skipping (insufficient data in overlapping bins)")
            continue

        # Rescale expected values to match observed total (required for chisquare test)
        exp_rescaled = exp * (obs.sum() / exp.sum())
        chi2_stat, p_val = chisquare(f_obs=obs, f_exp=exp_rescaled)
        print(f"{col:>4}: chi-squared = {chi2_stat:.2f}, p-value = {p_val:.4f}")

# Run the chi-squared comparison
run_chi_squared_tests(ngenic_df, pycosmo_df)