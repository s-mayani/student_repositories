import argparse
import os
import sys

# Set up relative imports
current_file_path = os.path.abspath(__file__)
base_dir = os.path.dirname(os.path.dirname(current_file_path))
sys.path.insert(0, base_dir)

from power_spectrum_generation.power_spectrum_class import PowerSpectrumClass

def read_param_file(file_path):
        """
        Reads a parameter file and extracts key-value pairs.

        :param file_path: Path to the parameter file
        :return: Dictionary containing parameter key-value pairs
        """
        params = {}
        try:
            with open(file_path, 'r') as file:
                for line in file:
                    # Remove comments and strip whitespace
                    line = line.split('%')[0].strip()

                    if line:
                        # Split key and value
                        key, value = line.split(None, 1)
                        params[key] = value
                    
        except FileNotFoundError:
            print(f"Error: File not found at {file_path}")

        except Exception as e:
            print(f"Error reading file: {e}")
        
        # Print key-value pairs in the params dictionary
        for key, value in params.items():
            print(f"{key}: {value}")

        return params


def run_spectrum_generation(parameters):
    """
    Function to run the actual simulation using the parameters.
    
    :param parameters: Dictionary containing simulation parameters
    """
    # Initialize the PowerSpectrumClass with the parameters
    power_spectrum = PowerSpectrumClass(parameters)

    # Compute the power spectrum
    power_spectrum.compute_power_spectra()

    # Output the plots of the results
    power_spectrum.plot_power_spectrum()

    # Save the power spectrum in N-GenIC-compatible format
    power_spectrum.save_power_spectrum_for_ngenic()

def main(param_file):
    """
    Main function to run the simulation with the specified parameter file.
    :param param_file: Path to the parameter file
    """
    # Get the relative path to the .param file
    param_file_path = os.path.join("parameter_files", param_file)
    
    # Check if the file exists
    if not os.path.isfile(param_file_path):
        print(f"Error: The file '{param_file_path}' does not exist!")
        return

    print(f"Using parameter file: {param_file}")

    # Logic to load and process the .param file
    parameters = read_param_file(param_file_path)

    # Run the simulation with the loaded parameters
    run_spectrum_generation(parameters)

# Import variables defined in input_parameters.param
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run the simulation with a specified parameter file.")
    
    parser.add_argument(
        'param_file',
        type=str,
        help="Path to the .param file to be used for the simulation"
    )
    
    args = parser.parse_args()

    main(args.param_file)


