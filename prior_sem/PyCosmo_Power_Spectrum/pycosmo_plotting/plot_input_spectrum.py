import numpy as np
import os
import matplotlib.pyplot as plt

"""
This script compares the input power spectrum from N-GenIC and PyCosmo.
It reads the power spectrum data from files, plots them on seperate graphs,
saving the plots in a specified directory.
"""

script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
parent_dir = os.path.dirname(script_dir)

# Use the custom style for the plots
plt.style.use(os.path.join(script_dir, "pycosmohub.mplstyle"))

# Get path for the plots directory
plots_dir = os.path.join(script_dir, 'generated_plots')

# Load pycosmo outputted power spectrum
pycosmo_data_path = os.path.join(parent_dir, 'outputted_power_spectrum', '32_input_spectrum.txt')
pycosmo_data = np.loadtxt(pycosmo_data_path, usecols=(0, 1))

# Load the first two columns from 'inputspec_lsf_512.txt'
ngenic_data_path = os.path.join(parent_dir, 'initial_conditions', 'inputspec_lsf_32.txt')
ngenic_data = np.loadtxt(ngenic_data_path, usecols=(0, 1))
ngenic_data = ngenic_data[1:] # Remove the first row, which contains the starting reshift and scale factor

# If the file has two columns: frequency and amplitude
if pycosmo_data.ndim == 2 and pycosmo_data.shape[1] == 2:
    log_pycosmo_wave_numbers = pycosmo_data[:, 0]
    log_pycosmo_delta_squared = pycosmo_data[:, 1]
else:
    # If the file has only one column (amplitude), use index as x
    log_pycosmo_wave_numbers= np.arange(len(pycosmo_data))
    log_pycosmo_delta_squared = pycosmo_data

# Separate columns
ngenic_wave_numbers = ngenic_data[:, 0]
ngenic_delta_squared = ngenic_data[:, 1]

# Plot both dimensionless power spectra on the same graph with different colors
plt.figure(figsize=(9, 6))
plt.plot(ngenic_wave_numbers, ngenic_delta_squared, label='N-GenIC', color='tab:blue', alpha=0.5)
plt.plot(10**log_pycosmo_wave_numbers, 10**log_pycosmo_delta_squared, label='PyCosmo', color='tab:orange', alpha=0.5)

"""# Load the first two columns from 'inputspec_lsf_512.txt'
py_ng_data_path = os.path.join(parent_dir, 'initial_conditions', 'inputspec_lsf_pycosmo_32.txt')
py_ng_data = np.loadtxt(ngenic_data_path, usecols=(0, 1))
py_ng_data = ngenic_data[1:] # Remove the first row, which contains the starting reshift and scale factor
py_ng_wave_numbers = py_ng_data[:, 0]
py_ng_delta_squared = py_ng_data[:, 1]
plt.plot(py_ng_wave_numbers, py_ng_delta_squared, label='N-GenIC ~ PyCosmo', color='tab:green', alpha=0.5)
"""
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$k$')
plt.ylabel(r'$\Delta^2$')
plt.grid(True, which='both', ls='--')
plt.legend(fontsize='small')
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'delta_squared_comparison.pdf'))
plt.show()

# Comparison of power spectra
ngenic_p_k = ngenic_delta_squared / (4 * np.pi * ngenic_wave_numbers**3)
log_pycosmo_p_k = log_pycosmo_delta_squared - np.log10(4 * np.pi * (10**log_pycosmo_wave_numbers)**3)

plt.figure(figsize=(9, 6))
plt.plot(ngenic_wave_numbers, ngenic_p_k, label='N-GenIC', color='tab:blue', alpha=0.5)
plt.plot(10**log_pycosmo_wave_numbers, 10**log_pycosmo_p_k, label='PyCosmo', color='tab:orange', alpha=0.5)
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$k$')
plt.ylabel(r'$P(k)$')
plt.grid(True, which='both', ls='--')
plt.legend(fontsize='small')
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'p_k_comparison.pdf'))
plt.show()
