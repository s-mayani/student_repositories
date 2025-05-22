import os
import csv

# Define the directory containing the files
directory = '.'  # Updated directory name

# Define the preconditioners
preconditioners = set()

# Initialize a dictionary to store the data
data = {}

# Initialize a set to store all unique N values
unique_N_values = set()

# Iterate through the files
for file in os.listdir(directory):
    if file.startswith('scalingPenning_') and file.endswith('.out'):
        print('Processing file:' + file)  # Print the file name
        N = file.split('_')[-1].split('.')[0]
        preconditioner = file.split('_')[-2].split('.')[0]
        unique_N_values.add(N)
        if preconditioner not in data:
            data[preconditioner] = {}
        preconditioners.add(preconditioner)
        with open(os.path.join(directory, file), 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                if line.startswith('Timings{0}> solve............... Wall max =') or line.startswith('Timings> solve............... Wall max ='):
                    value = lines[i+1].split('=')[1].strip()
                    data[preconditioner][N] = value

# Write the data to a CSV file
csv_file = 'Scaling_time.csv'
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Nodes'] + list(preconditioners))
    for N in sorted(unique_N_values, key=float):  # Sort keys numerically
        row = [N] + [data[preconditioner].get(N, 'NaN') for preconditioner in preconditioners]
        writer.writerow(row)

print('Data has been saved to '+ csv_file)