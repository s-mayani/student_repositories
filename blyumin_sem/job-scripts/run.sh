#!/bin/bash
#SBATCH --cluster=merlin6                   # Cluster name                                        
#SBATCH --partition=hourly                 # Specify one or multiple partitions
#SBATCH --time=00:05:00                     # Define max time job will run

#SBATCH --job-name=cosmoNew-run     # Job name    (default: sbatch)
#SBATCH --output=cosmo-4ranks-lb-%j.out # Output file (default: slurm-%j.out)
#SBATCH --error=cosmo-4ranks-lb-%j.err  # Error file  (default: slurm-%j.out)

#SBATCH --hint=nomultithread                # Without hyperthreading                    
#SBATCH --nodes=5
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1

export OMP_NUM_THREADS=1
export OMP_PROC_BIND=spread
export OMP_PLACES=cores

EXEC_DIR="/psi/home/bobrov_e/ippl/build-cosmo-openmp/cosmology-new"
DATA_DIR="/data/user/bobrov_e/data/zeldovich_ICs"

srun "$EXEC_DIR/StructureFormation" infile.dat tf.dat out $DATA_DIR FFT 0.01 LeapFrog --overallocate 1.0 --info 5
