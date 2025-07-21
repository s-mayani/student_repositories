#!/bin/bash
#SBATCH --cluster=merlin6                   # Cluster name                                        
#SBATCH --partition=hourly                   # Specify one or multiple partitions
#SBATCH --time=00:20:00                      # Define max time job will run

#SBATCH --job-name=genic-job                 # Job name
#SBATCH --output=genic-job-%j.out            # Output file
#SBATCH --error=genic-job-%j.err             # Error file
#SBATCH --hint=nomultithread                 # Mandatory for multithreaded jobs

#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=4                  # 4 tasks per node
#SBATCH --exclusive

srun --mpi=pmi2 -n 8 /psi/home/bobrov_e/SimGadget/ngenic/N-GenIC /psi/home/bobrov_e/SimGadget/ngenic/parameterfiles/lsf_512.param


