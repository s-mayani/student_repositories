#!/bin/bash
##SBATCH --error=LandauDamping_pascal_correctess.err
##SBATCH --output=LandauDamping_pascal_correctness.out
#SBATCH --time=00:40:00
#SBATCH --nodes=1
##SBATCH --ntasks=4
#SBATCH --clusters=gmerlin6
#SBATCH --partition=gwendolen   # Running on the Gwendolen partition of the GPU cluster
#SBATCH --account=gwendolen
#SBATCH --exclusive
##SBATCH --gpus=4

N1=$1
N2=$2
N3=$3
preconditioner=$4
P1=$5
P2=$6
P3=$7

#run for preconditioned CG:

#srun ./LandauDamping  $N1 $N2 $N3 134217728 20 PCG 0.05 LeapFrog $preconditioner $P1 $P2 $P3 --overallocate 2.0 --info 10 --kokkos-map-device-id-by=mpi_rank


#run for CG without preconditioning:

srun ./LandauDamping  $N1 $N2 $N3 134217728 20 CG 0.05 LeapFrog --overallocate 2.0 --info 10 --kokkos-map-device-id-by=mpi_rank

