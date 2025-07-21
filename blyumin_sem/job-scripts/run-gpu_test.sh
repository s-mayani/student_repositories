#!/bin/bash
#SBATCH --time=00:05:00                 # Max job time
#SBATCH --nodes=1                       # One node (Gwendolen has one GPU node)
#SBATCH --ntasks=4                      # One MPI rank per GPU
#SBATCH --clusters=gmerlin6            # GPU cluster name
#SBATCH --partition=gwendolen          # Gwendolen GPU partition
#SBATCH --account=gwendolen            # Account for job accounting
#SBATCH --gpus=4                        # Request 4 GPUs

#SBATCH --output=test-gpu-%j.out      # Output log
#SBATCH --error=test-gpu-%j.err       # Error log

# Optional: set threading policies (can omit or adjust)
export OMP_NUM_THREADS=1

EXEC_DIR="/psi/home/bobrov_e/ippl/build-cosmo-gpu/test"
DATA_DIR="/data/user/bobrov_e/data/zeldovich_ICs"

srun "$EXEC_DIR/cosmology/TestHermiticity" infile.dat tf.dat out $DATA_DIR FFT 1.0 LeapFrog --overallocate 1.0 --info 5 --kokkos-map-device-id-by=mpi_rank

