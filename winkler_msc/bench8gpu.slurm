#!/bin/bash
#SBATCH --error=bench1.err
#SBATCH --output=bench1.out
#SBATCH --time=00:40:00
#SBATCH --nodes=8
#SBATCH --ntasks=8
#SBATCH --clusters=gmerlin6
#SBATCH --partition=gwendolen
#SBATCH --account=gwendolen
#SBATCH --gpus=8

srun ./test/solver/TestFDTDSolver --kokkos-map-device-id-by=mpi_rank