#!/bin/bash
#SBATCH --time=00:05:00                 # Max job time
#SBATCH --nodes=1                       # One node (Gwendolen has one GPU node)
#SBATCH --ntasks=8                      # One MPI rank per GPU
#SBATCH --clusters=gmerlin6            # GPU cluster name
#SBATCH --partition=gwendolen          # Gwendolen GPU partition
#SBATCH --account=gwendolen            # Account for job accounting
#SBATCH --gpus=8                        # Request 8 GPUs

#SBATCH --output=cosmo-gpu-%j.out      # Output log
#SBATCH --error=cosmo-gpu-%j.err       # Error log

# Optional: set threading policies (can omit or adjust)
export OMP_NUM_THREADS=1

EXEC_DIR="/psi/home/bobrov_e/ippl/build-cosmo-gpu/cosmology-new"
DATA_DIR="/data/user/bobrov_e/data/zeldovich_ICs"

srun "$EXEC_DIR/StructureFormation" infile.dat tf.dat out $DATA_DIR FFT 0.01 LeapFrog --overallocate 1.0 --info 5 --kokkos-map-device-id-by=mpi_rank

