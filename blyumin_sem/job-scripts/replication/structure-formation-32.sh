#!/bin/bash
#SBATCH --cluster=merlin6                   # Cluster name                                        
#SBATCH --partition=hourly                 # Specify one or multiple partitions
#SBATCH --time=00:20:00                     # Define max time job will run

#SBATCH --job-name=structure-formation      # Job name    (default: sbatch)
#SBATCH --output=structure-formation-%j.out # Output file (default: slurm-%j.out)
#SBATCH --error=structure-formation-%j.err  # Error file  (default: slurm-%j.out)
#SBATCH --hint=nomultithread                # Mandatory for multithreaded jobs                    

EXEC_DIR="/psi/home/bobrov_e/ippl/build-cosmo/cosmology"
DATA_DIR="/data/user/bobrov_e/data/lsf_32/"
# Uncomment the problem size you want

#srun "$EXEC_DIR/StructureFormation" $DATA_DIR 16 16 16 4096 100 FFT 1.0 LeapFrog --overallocate 1.0 --info 5 &
"$EXEC_DIR/StructureFormation" $DATA_DIR 32 32 32 32768 100 FFT 1.0 LeapFrog --overallocate 1.0 --info 5 &
#srun --cpus-per-task=44 "$EXEC_DIR/StructureFormation" data/lsf_64/ 64 64 64 262144 100 FFT 1.0 LeapFrog --overallocate 1.0 --info 5 &
#srun --cpus-per-task=44 "$EXEC_DIR/StructureFormation" data/lsf_128/ 128 128 128 2097152 100 FFT 1.0 LeapFrog --overallocate 1.0 --info 5 &
#srun --cpus-per-task=44 "$EXEC_DIR/StructureFormation" data/lsf_256/ 256 256 256 16777216 100 FFT 1.0 LeapFrog --overallocate 1.0 --info 5 &
#srun --cpus-per-task=44 "$EXEC_DIR/StructureFormation" data/lsf_512/ 512 512 512 134217728 100 FFT 1.0 LeapFrog --overallocate 1.0 --info 5 &
#srun --cpus-per-task=44 "$EXEC_DIR/StructureFormation" data/lsf_1048/ 1024 1024 1024 1073741824 00 FFT 1.0 LeapFrog --overallocate 1.0 --info 5 &
wait
