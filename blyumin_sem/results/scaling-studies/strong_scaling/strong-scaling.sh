#!/bin/bash
#SBATCH --cluster=merlin6                   # Cluster name                                        
#SBATCH --partition=hourly                 # Specify one or multiple partitions
#SBATCH --time=00:15:00                     # Define max time job will run

#SBATCH --job-name=ic-strong-scaling     # Job name    (default: sbatch)
#SBATCH --output=ic-strong-scaling-%j.out # Output file (default: slurm-%j.out)
#SBATCH --error=ic-strong-scaling-%j.err  # Error file  (default: slurm-%j.out)

#SBATCH --hint=nomultithread                # Without hyperthreading                    
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --exclusive

export OMP_NUM_THREADS=1
export OMP_PROC_BIND=spread
export OMP_PLACES=cores

EXEC_DIR="/psi/home/bobrov_e/ippl/build-cosmo-openmp/cosmology-new"
INPUT_DIR="/psi/home/bobrov_e/cosmo-sim/runCosmo/scaling_studies"
DATA_DIR="/psi/home/bobrov_e/cosmo-sim/runCosmo/scaling_studies/results"

# For strong scaling use 16 nodes - each node will use 1 cpu (therefore, 1 mpi rank per node).
# For each problem size (ranging from 16^3, to 512^3, 6 in total) run the code at 1,2,4,8,16 and 32 nodes  

for n in 16 32 64 128; do
  for nodes in 1 2 4 8 16; do
    srun --nodes=$nodes "$EXEC_DIR/StructureFormation" "$INPUT_DIR/infile${n}.dat" "$INPUT_DIR/tf.dat" out "$DATA_DIR/results_strong/${n}_particles_" FFT 0.01 LeapFrog --overallocate 1.0 --info 5
  done
done

