#!/bin/bash
#SBATCH --ntasks=4
#SBATCH --gpus=4
#SBATCH --exclusive

echo $COMMAND into $FILE
srun $COMMAND > $FILE
