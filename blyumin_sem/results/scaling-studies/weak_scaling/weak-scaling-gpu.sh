#!/bin/bash
#SBATCH --time=00:05:00                 # Max job time
#SBATCH --nodes=1                       # One node (Gwendolen has one GPU node)
#SBATCH --ntasks=8                      # One MPI rank per GPU
#SBATCH --clusters=gmerlin6            # GPU cluster name
#SBATCH --partition=gwendolen          # Gwendolen GPU partition
#SBATCH --account=gwendolen            # Account for job accounting
#SBATCH --gpus=8                        # Request 8 GPUs

#SBATCH --job-name=ic-weak-caling-gpu
#SBATCH --output=ic-weak-scaling-gpu-%j.out      # Output log
#SBATCH --error=ic-weak-scaling-gpu-%j.err       # Error log
#SBATCH --exclusive

# Optional: set threading policies (can omit or adjust)
export OMP_NUM_THREADS=1

EXEC_DIR="/psi/home/bobrov_e/ippl/build-cosmo-gpu/cosmology-new"
INPUT_DIR="/psi/home/bobrov_e/cosmo-sim/runCosmo/scaling_studies"
DATA_DIR="/psi/home/bobrov_e/cosmo-sim/runCosmo/scaling_studies/results/results_weak_gpu"

# For weak scaling use 8 gpus - each node will use 1 cpu 
# (therefore, 1 mpi rank per node, no multithreading).
# For each gpu (1,2,4,8) the total number of particles n^3
# must scale linearly, hence the choices of the numbers below.

for gpu in 1 2 4 8; do
  case $gpu in
      1) n=16 ;;
      2) n=20 ;;
      4) n=25 ;;
      8) n=32 ;;
  esac

  srun --gpus=$gpu --ntasks=$gpu "$EXEC_DIR/StructureFormation" "$INPUT_DIR/infile${n}.dat" "$INPUT_DIR/tf.dat" out "$DATA_DIR/${n}_particles_${gpu}_" FFT 0.01 LeapFrog --overallocate 1.0 --info 5
done

