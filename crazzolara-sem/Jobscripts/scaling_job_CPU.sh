#!/bin/bash
#SBATCH --partition=hourly      # Using 'hourly' will grant higher priority
#SBATCH --time=00:20:00         # Define max time job will run
#SBATCH --exclusive             # The allocations will be exclusive if turned on
#SBATCH --hint=nomultithread    # Without hyperthreading

export OMP_NUM_THREADS=44
export OMP_PROC_BIND=spread
export OMP_PLACES=cores

EXEC_DIR="/data/user/crazzo_b/IPPL/ippl/build_openmp/alpine"

# Uncomment the problem size you want

srun --cpus-per-task=44 "$EXEC_DIR/StructureFormation" data/lsf_16/ 16 16 16 4096 100 FFT 1.0 LeapFrog --overallocate 1.0 --info 5 &
#srun --cpus-per-task=44 "$EXEC_DIR/StructureFormation" data/lsf_32/ 32 32 32 32768 100 FFT 1.0 LeapFrog --overallocate 1.0 --info 5 &
#srun --cpus-per-task=44 "$EXEC_DIR/StructureFormation" data/lsf_64/ 64 64 64 262144 100 FFT 1.0 LeapFrog --overallocate 1.0 --info 5 &
#srun --cpus-per-task=44 "$EXEC_DIR/StructureFormation" data/lsf_128/ 128 128 128 2097152 100 FFT 1.0 LeapFrog --overallocate 1.0 --info 5 &
#srun --cpus-per-task=44 "$EXEC_DIR/StructureFormation" data/lsf_256/ 256 256 256 16777216 100 FFT 1.0 LeapFrog --overallocate 1.0 --info 5 &
#srun --cpus-per-task=44 "$EXEC_DIR/StructureFormation" data/lsf_512/ 512 512 512 134217728 100 FFT 1.0 LeapFrog --overallocate 1.0 --info 5 &
#srun --cpus-per-task=44 "$EXEC_DIR/StructureFormation" data/lsf_1048/ 1024 1024 1024 1073741824 00 FFT 1.0 LeapFrog --overallocate 1.0 --info 5 &
wait