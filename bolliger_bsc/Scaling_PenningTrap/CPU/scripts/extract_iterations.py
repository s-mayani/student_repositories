import os
import csv


def proper_round(num, dec=0):
    num = str(num)[:str(num).index('.')+dec+2]
    if num[-1]>='5':
        return float(num[:-2-(not dec)]+str(int(num[-2-(not dec)])+1))
    return float(num[:-1])

# Define the directory containing the files
directory = '.'  # Updated directory name

# Define the preconditioners
preconditioners = ['Gauss-Seidel','CG','Richardson','SSOR']

# Initialize a dictionary to store the data
data = {}
data['Gauss-Seidel'] = 0
data['CG'] = 0
data['Richardson'] = 0
data['SSOR'] = 0
# Initialize a set to store all unique N values
unique_N_values = set()

# Iterate through the files
for file in os.listdir(directory):
    if file.startswith('scalingPenning_') and file.endswith('.out'):
        print('Processing file:' + file)  # Print the file name
        preconditioner = file.split('_')[-2].split('.')[0]
        with open(os.path.join(directory, file), 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                if line.startswith('scatter >') or line.startswith('scatter {0}>'):
                    value = lines[i+1].split(',')[1].strip()
                    data[preconditioner] += int(value)

# Write the data to a CSV file
csv_file = 'Iterations.csv'
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(list(preconditioners))
    row = [proper_round(data[preconditioner]/105,2) for preconditioner in preconditioners]
    writer.writerow(row)

print('Data has been saved to '+ csv_file)