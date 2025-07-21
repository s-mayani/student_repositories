#!/bin/bash
#SBATCH --cluster=merlin6                   # Cluster name                                        
#SBATCH --partition=hourly                 # Specify one or multiple partitions
#SBATCH --time=00:05:00                     # Define max time job will run

#SBATCH --job-name=ic-weak-scaling     # Job name    (default: sbatch)
#SBATCH --output=ic-weak-scaling-%j.out # Output file (default: slurm-%j.out)
#SBATCH --error=ic-weak-scaling-%j.err  # Error file  (default: slurm-%j.out)

#SBATCH --hint=nomultithread                # Without hyperthreading                    
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1

export OMP_NUM_THREADS=1

EXEC_DIR="/psi/home/bobrov_e/ippl/build-cosmo-openmp/cosmology-new"
INPUT_DIR="/psi/home/bobrov_e/cosmo-sim/runCosmo/scaling_studies"
DATA_DIR="/psi/home/bobrov_e/cosmo-sim/runCosmo/scaling_studies/results"

# For weak scaling use 16 nodes - each node will use 1 cpu 
# (therefore, 1 mpi rank per node).
# For each node (1,2,4,8,and 16) the total number of particles
# must scale linearly, hence the choices of the numbers below.
# Note : make sure that the input files for each problem size exist under the input directory
# in the format expected below (infile${n}.dat)

for nodes in 1 2 4 8 16; do
  case $nodes in
      1) n=16 ;;
      2) n=20 ;;
      4) n=25 ;;
      8) n=32 ;;
     16) n=40 ;;
  esac

  srun --nodes=$nodes "$EXEC_DIR/StructureFormation" "$INPUT_DIR/infile${n}.dat" "$INPUT_DIR/tf.dat" out "$DATA_DIR/results_weak/${n}_particles_" FFT 0.01 LeapFrog --overallocate 1.0 --info 5
done


