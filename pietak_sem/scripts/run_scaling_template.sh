#!/bin/bash
#SBATCH --ntasks=NTASKS           
#SBATCH --tasks-per-node=TASKS_PER_NODE
#SBATCH --gpus=NTASKS
#SBATCH --exclusive

srun ./test/solver/fem/TestMaxwellDiffusionPolyZeroBCTimed $DIM $N >> $OUTFILE
