#!/bin/bash
#SBATCH --partition=hourly       # Using 'hourly' will grant higher priority
#SBATCH --ntasks-per-node=1      # No. of MPI ranks per node. Merlin CPU nodes have 44 cores
#SBATCH --cpus-per-task=44        # No. of OMP threads
#SBATCH --time=01:00:00          # Define max time job will run
#SBATCH --hint=nomultithread     # No hyperthreading

export OMP_NUM_THREADS=44
export OMP_PROC_BIND=spread
export OMP_PLACES=threads


N1=$1
N2=$2
N3=$3
preconditioner=$4
P1=$5
P2=$6
P3=$7

#run for preconditioned CG:

srun --cpus-per-task=44 ./LandauDamping $N1 $N2 $N3 134217728 20 PCG 0.01 LeapFrog $preconditioner $P1 $P2 $P3 --overallocate 2.0 --info 10

#run for CG without preconditioning:

#srun --cpus-per-task=44 ./LandauDamping $N1 $N2 $N3 134217728 20 CG 0.01 LeapFrog --overallocate 2.0 --info 10