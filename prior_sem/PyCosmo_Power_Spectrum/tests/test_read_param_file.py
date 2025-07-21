import unittest
import tempfile
import os
import sys

# Set up relative imports
current_file_path = os.path.abspath(__file__)
base_dir = os.path.dirname(os.path.dirname(current_file_path))
sys.path.insert(0, base_dir)

from power_spectrum_generation.custom_power_spectrum import read_param_file

class TestReadParamFile(unittest.TestCase):

    def setUp(self):
        # Create a temporary file with mock parameter content
        self.temp_file = tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.param')
        self.temp_file.write("""Box 100.0
                                Nsample 64
                                Redshift 0.0
                                % This is a comment
                                Omega 0.3
                                OmegaLambda 0.7
                                """)
        self.temp_file.close()

    def tearDown(self):
        # Delete the temporary file after the test
        os.unlink(self.temp_file.name)

    def test_read_param_file(self):
        expected_params = {
            "Box": "100.0",
            "Nsample": "64",
            "Redshift": "0.0",
            "Omega": "0.3",
            "OmegaLambda": "0.7",
        }

        result = read_param_file(self.temp_file.name)
        self.assertEqual(result, expected_params)

if __name__ == '__main__':
    unittest.main()
