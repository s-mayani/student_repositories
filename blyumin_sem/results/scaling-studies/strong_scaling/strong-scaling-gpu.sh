#!/bin/bash
#SBATCH --time=00:05:00                 # Max job time
#SBATCH --nodes=1                       # One node (Gwendolen has one GPU node)
#SBATCH --ntasks=8                      # One MPI rank per GPU
#SBATCH --clusters=gmerlin6            # GPU cluster name
#SBATCH --partition=gwendolen          # Gwendolen GPU partition
#SBATCH --account=gwendolen            # Account for job accounting
#SBATCH --gpus=8                        # Request 8 GPUs

#SBATCH --output=ic-strong-scaling-gpu-%j.out      # Output log
#SBATCH --error=ic-strong-scaling-gpu-%j.err       # Error log
#SBATCH --exclusive

# Optional: set threading policies (can omit or adjust)
export OMP_NUM_THREADS=1

EXEC_DIR="/psi/home/bobrov_e/ippl/build-cosmo-gpu/cosmology-new"
INPUT_DIR="/psi/home/bobrov_e/cosmo-sim/runCosmo/scaling_studies"
DATA_DIR="/psi/home/bobrov_e/cosmo-sim/runCosmo/scaling_studies/results/results_strong_gpu"

# For each problem size, run with different numbers of GPUs and MPI ranks
for n in 16 32 64 128 256; do
  for gpu in 1 2 4 8; do  # 4 GPUs/node → 4, 8, 16, 32 GPUs = 1–8 nodes    
    srun --gpus=$gpu --ntasks=$gpu "$EXEC_DIR/StructureFormation" "$INPUT_DIR/infile${n}.dat" "$INPUT_DIR/tf.dat" \
      out "$DATA_DIR/${n}_particles_${gpu}_" FFT 0.01 LeapFrog --overallocate 1.0 --info 5 --kokkos-map-device-id-by=mpi_rank
  done
done
