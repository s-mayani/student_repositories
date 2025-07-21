import unittest
import numpy as np
import os
import sys

# Set up relative imports
current_file_path = os.path.abspath(__file__)
base_dir = os.path.dirname(os.path.dirname(current_file_path))
sys.path.insert(0, base_dir)

from power_spectrum_generation.power_spectrum_class import PowerSpectrumClass
from power_spectrum_generation.custom_power_spectrum import read_param_file

test_param_file = os.path.join(base_dir, "parameter_files", "pycosmo_input_512.param")
test_param_dictionary = read_param_file(test_param_file)

class TestPowerSpectrumComputation(unittest.TestCase):

    def setUp(self):
        # Setup a minimal valid instance of PowerSpectrumClass
        self.power_spectrum = PowerSpectrumClass(parameters=test_param_dictionary)
        
        # Optionally set a known redshift and k_values manually
        self.power_spectrum.z_start = 0  # z=0 should be safe
        self.power_spectrum.k_values = np.logspace(-3, 1, 50)  # Reasonable k-range in h/Mpc

    def test_compute_power_spectra(self):
        # Run the method
        self.power_spectrum.compute_power_spectra()
        
        # Check that the result has been stored correctly and has the correct shape
        self.assertIsNotNone(self.power_spectrum.pk_nonlin, "Non-linear power spectrum was not computed.")
        self.assertEqual(len(self.power_spectrum.pk_nonlin), len(self.power_spectrum.k_values), "Power spectrum length mismatch.")

        # Check for any NaNs or negative values in the output
        self.assertFalse(np.isnan(self.power_spectrum.pk_nonlin).any(), "NaNs found in non-linear power spectrum.")
        self.assertTrue((self.power_spectrum.pk_nonlin >= 0).all(), "Negative values found in non-linear power spectrum.")

if __name__ == "__main__":
    unittest.main()
