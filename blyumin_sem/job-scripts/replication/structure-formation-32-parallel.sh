#!/bin/bash
#SBATCH --cluster=merlin6                   # Cluster name                                        
#SBATCH --partition=hourly                 # Specify one or multiple partitions
#SBATCH --time=01:00:00                     # Define max time job will run

#SBATCH --job-name=structure-formation-32-parallel      # Job name    (default: sbatch)
#SBATCH --output=structure-formation-32-parallel-%j.out # Output file (default: slurm-%j.out)
#SBATCH --error=structure-formation-32-parallel-%j.err  # Error file  (default: slurm-%j.out)

#SBATCH --hint=nomultithread                # Without hyperthreading                    
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8

export OMP_NUM_THREADS=8
export OMP_PROC_BIND=spread
export OMP_PLACES=cores

EXEC_DIR="/psi/home/bobrov_e/ippl/build-cosmo-openmp/cosmology"
DATA_DIR="/data/user/bobrov_e/data/lsf_32/"


# Uncomment the problem size you want

#srun "$EXEC_DIR/StructureFormation" $DATA_DIR 16 16 16 4096 100 FFT 1.0 LeapFrog --overallocate 1.0 --info 5 &
srun --cpus-per-task=8 "$EXEC_DIR/StructureFormation" $DATA_DIR 32 32 32 32768 100 FFT 1.0 LeapFrog --overallocate 1.0 --info 5 
#srun --cpus-per-task=44 "$EXEC_DIR/StructureFormation" data/lsf_64/ 64 64 64 262144 100 FFT 1.0 LeapFrog --overallocate 1.0 --info 5 &
#srun --cpus-per-task=44 "$EXEC_DIR/StructureFormation" data/lsf_128/ 128 128 128 2097152 100 FFT 1.0 LeapFrog --overallocate 1.0 --info 5 &
#srun --cpus-per-task=44 "$EXEC_DIR/StructureFormation" data/lsf_256/ 256 256 256 16777216 100 FFT 1.0 LeapFrog --overallocate 1.0 --info 5 &
#srun --cpus-per-task=44 "$EXEC_DIR/StructureFormation" $DATA_DIR 512 512 512 134217728 2000 FFT 1.0 LeapFrog --overallocate 1.0 --info 5
#srun --cpus-per-task=44 "$EXEC_DIR/StructureFormation" data/lsf_1048/ 1024 1024 1024 1073741824 00 FFT 1.0 LeapFrog --overallocate 1.0 --info 5 &

