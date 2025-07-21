import unittest
import numpy as np
import os
import sys
import PyCosmo

# Set up relative imports
current_file_path = os.path.abspath(__file__)
base_dir = os.path.dirname(os.path.dirname(current_file_path))
sys.path.insert(0, base_dir)

from power_spectrum_generation.power_spectrum_class import PowerSpectrumClass

class TestkValues(unittest.TestCase):
    def setUp(self):
        # Example parameters for initializing the PowerSpectrumClass
        self.param_dictionary = {
            "Box": 100.0,
            "Nsample": 4,
            "Redshift": 0.0,
            "Omega": 0.3,
            "OmegaLambda": 0.7,
            "OmegaBaryon": 0.05,
            "HubbleParam": 0.7,
            "Sigma8": 0.8,
            "NonLinearFittingFunction": 0,
            "LinearFittingFunction": 0,
            "UnitLength_in_cm": 3.085677581e24,
            "UnitMass_in_g": 1.989e43,
            "UnitVelocity_in_cm_per_s": 1e5,
        }

        self.ps_class = PowerSpectrumClass(self.param_dictionary)

    def test_compute_power_spectrum_for_k_values(self):
        # Assuming PowerSpectrumClass has 'redshift' attribute
        a = 1. / (1 + self.ps_class.z_start)

        power_spectra = np.zeros(len(self.ps_class.k_values))

        i = 0

        for k in self.ps_class.k_values:  # Assuming PowerSpectrumClass has 'k_values' attribute
            try:
                pk = self.ps_class.cosmo.lin_pert.powerspec_a_k(a, 10**k)
                power_spectra[i] = pk
                i += 1
            except Exception as e:
                self.fail(f"Failed for k = {10**k}: {e}")
        
        print("Power spectra computed")

        # Add assertions to check the computed power spectra
        self.assertIsNotNone(power_spectra, "No power spectra were computed.")
        self.assertTrue(all(np.isscalar(pk) for pk in power_spectra), "Power spectrum values should be scalars.")
        # You might want to add more specific assertions based on expected values or properties

if __name__ == "__main__":
    unittest.main()