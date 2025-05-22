#!/bin/bash
##SBATCH --error=test.err
##SBATCH --output=test.out
#SBATCH --time=00:20:00
#SBATCH --nodes=1
##SBATCH --ntasks=8
#SBATCH --clusters=gmerlin6
#SBATCH --partition=gwendolen
#SBATCH --account=gwendolen
##SBATCH --exclusive
##SBATCH --gpus=8

N=$1
scaling=$2
preconditioner=$3
P1=$4
P2=$5
P3=$6
srun ./TestCGSolver $N $scaling $preconditioner $P1 $P2 $P3 --info 5