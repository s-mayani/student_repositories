import os, sys
import PyCosmo
import numpy as np
import matplotlib.pyplot as plt

# Set up relative imports
current_file_path = os.path.abspath(__file__)
base_dir = os.path.dirname(os.path.dirname(current_file_path))
sys.path.insert(0, base_dir)  # Adds base_dir to Python path

# Use matplotlib style
style_path = os.path.join(base_dir, "pycosmo_plotting", "pycosmohub.mplstyle")
plt.style.use(style_path)

class PowerSpectrumClass:

    def __init__(self, parameters):
        """
        Initialize the PowerSpectrumClass with a parameter dictionary.
        """
        self.param_dictionary = parameters
        self.cosmo = PyCosmo.build()

        self.box_size = float(parameters["Box"])
        self.Nsample = int(parameters["Nsample"])
        self.z_start = float(parameters["Redshift"])

        self.k_count = int(self.Nsample)**3

        self.unitlength_in_cm = float(parameters["UnitLength_in_cm"])
        self.unitmass_in_g = float(parameters["UnitMass_in_g"])
        self.unitvelocity_in_cm_per_s = float(parameters["UnitVelocity_in_cm_per_s"])

        self.kmin = 2 * np.pi / (1000.0 * (3.085678e24 / self.unitlength_in_cm))
        self.nyquist = self.box_size * 2 * np.pi / (0.001 * (3.085678e24 / self.unitlength_in_cm))

        if self.kmin <= 0 or self.nyquist <= 0:
            raise ValueError("Invalid kmin or nyquist values. Ensure Box and Nsample are positive.")

        self.k_values = np.linspace(np.log10(self.kmin), np.log10(self.nyquist), self.k_count)

        print(f"Using k values from {self.kmin:.2e} to {self.nyquist:.2e} with {self.k_count} samples")

        if int(parameters["LinearFittingFunction"]) == 0:
            print("Using Eisenstein & Hu linear fitting function")
            self.cosmo.set(pk_type="EH")
        elif int(parameters["LinearFittingFunction"]) == 1:
            print("Using Bardeen, Bond, Kaiser & Szalay linear fitting function")
            self.cosmo.set(pk_type="BBKS")
        else:
            print("Using Boltzmann linear fitting function")
            self.cosmo.set(pk_type="boltz")

        self.cosmo.set(
            omega_m=float(parameters["Omega"]),
            omega_l=float(parameters["OmegaLambda"]),
            omega_b=float(parameters["OmegaBaryon"]),
            h=float(parameters["HubbleParam"]),
            pk_norm_type="sigma8",
            pk_norm=float(parameters["Sigma8"]),
        )

        self.pk_lin = None

    def compute_power_spectra(self):
        """
        Compute the linear power spectrum only.
        """
        assert np.all(np.isfinite(self.k_values)), "k_values must be finite"

        a = 1. / (1 + self.z_start)
        print(f"Computing linear power spectrum at redshift z={self.z_start} (a={a})")

        if self.cosmo.lin_pert is not None:
            print("Calling linear power spectrum...")
            try:
                result = self.cosmo.lin_pert.powerspec_a_k(a, 10**(self.k_values))
                print("Returned from linear power spectrum")
                self.pk_lin = result[:, 0]
            except Exception as e:
                print(f"Error in linear powerspec: {e}")
                raise
        else:
            print("WARNING: Linear power spectrum is not available at this redshift or model. Skipping.")
            self.pk_lin = None

    def plot_power_spectrum(self):
        """
        Plot the linear power spectrum.
        """
        if self.pk_lin is None:
            raise ValueError("No linear power spectrum data available. Run compute_power_spectra() first.")

        plt.figure(figsize=(15.5, 5.5))
        ax = plt.gca()

        ax.plot(self.k_values, np.log10(self.pk_lin), color='dodgerblue', linewidth=2, label='linear')

        ax.set_xlabel(r'$k \ [Mpc^{-1}]$', fontsize=28)
        ax.set_ylabel(r'$P(k) \ [Mpc^{3}]$', fontsize=28)
        ax.tick_params(axis='both', which='major', labelsize=24)

        for tick in ax.xaxis.get_major_ticks():
            tick.label1.set_fontweight('bold')
        for tick in ax.yaxis.get_major_ticks():
            tick.label1.set_fontweight('bold')

        textstr = f'Redshift: z = {self.z_start:.2f}'
        props = dict(boxstyle='round', facecolor='white', alpha=0.8)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=18,
                verticalalignment='top', bbox=props)

        plt.legend(loc='best')

        output_dir = "pycosmo_plotting/generated_plots"
        os.makedirs(output_dir, exist_ok=True)

        output_path = os.path.join(output_dir, f"{self.Nsample}_power_spectrum_plot.png")
        plt.savefig(output_path, bbox_inches='tight')

        print(f"Power spectrum plot saved in: {output_path}")

    def save_power_spectrum_for_ngenic(self, output_path=None):
        """
        Save the linear power spectrum in N-GenIC-compatible format.
        """
        if self.pk_lin is None:
            raise ValueError("Linear power spectrum not computed. Run compute_power_spectra() first.")

        if output_path is None:
            output_path = f"outputted_power_spectrum/{self.Nsample}_input_spectrum.txt"

        k_h_Mpc = 10**(self.k_values)
        delta_squared = 4 * np.pi * (k_h_Mpc)**3 * self.pk_lin
        log_k = np.log10(k_h_Mpc)
        log_delta_sqrd = np.log10(delta_squared)

        data = np.column_stack((log_k, log_delta_sqrd))
        np.savetxt(output_path, data, fmt="%.8e", delimiter=" ")

        print(f"Power spectrum saved in N-GenIC format to: {output_path}")
