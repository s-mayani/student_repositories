#!/bin/bash
#SBATCH --time=00:15:00
#SBATCH --clusters=gmerlin6
#SBATCH --partition=gwendolen
#SBATCH --account=gwendolen
#SBATCH --exclusive

 
# Uncomment the problem size you want

srun ./StructureFormation data/lsf_16/ 16 16 16 4096 100 FFT 1.0 LeapFrog --overallocate 1.0 --info 5 --kokkos-map-device-id-by=mpi_rank &
#srun ./StructureFormation data/lsf_32/ 32 32 32 32768 100 FFT 1.0 LeapFrog --overallocate 1.0 --info 5 --kokkos-map-device-id-by=mpi_rank &
#srun ./StructureFormation data/lsf_64/ 64 64 64 262144 100 FFT 1.0 LeapFrog --overallocate 1.0 --info 5 --kokkos-map-device-id-by=mpi_rank & 
#srun ./StructureFormation data/lsf_128/ 128 128 128 2097152 100 FFT 1.0 LeapFrog --overallocate 1.0 --info 5 --kokkos-map-device-id-by=mpi_rank &
#srun ./StructureFormation data/lsf_256/ 256 256 256 16777216 100 FFT 1.0 LeapFrog --overallocate 1.0 --info 5 --kokkos-map-device-id-by=mpi_rank & 
#srun ./StructureFormation data/lsf_512/ 512 512 512 134217728 100 FFT 1.0 LeapFrog --overallocate 1.0 --info 5 --kokkos-map-device-id-by=mpi_rank & 
#srun ./StructureFormation data/lsf_1024/ 1024 1024 1024 1073741824 100 FFT 1.0 LeapFrog --overallocate 1.0 --info 5 --kokkos-map-device-id-by=mpi_rank &
wait

