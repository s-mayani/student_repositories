#!/bin/bash
#SBATCH --cluster=merlin6                   # Cluster name                                        
#SBATCH --partition=hourly                 # Specify one or multiple partitions
#SBATCH --time=00:30:00                     # Define max time job will run

#SBATCH --job-name=test-multirank     # Job name    (default: sbatch)
#SBATCH --output=test-multirank-30-%j.out # Output file (default: slurm-%j.out)
#SBATCH --error=test-multirank-30-%j.err  # Error file  (default: slurm-%j.out)

#SBATCH --hint=nomultithread                # Without hyperthreading                    
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1

export OMP_NUM_THREADS=1
export OMP_PROC_BIND=spread
export OMP_PLACES=cores

EXEC_DIR="/psi/home/bobrov_e/ippl/build-cosmo-openmp/test"
DATA_DIR="/data/user/bobrov_e/data/zeldovich_ICs"
export OMPI_MCA_pml=^ucx # for debugging 

srun "$EXEC_DIR/cosmology/TestHermiticity" infile.dat tf.dat out $DATA_DIR FFT 1.0 LeapFrog --overallocate 1.0 --info 5
