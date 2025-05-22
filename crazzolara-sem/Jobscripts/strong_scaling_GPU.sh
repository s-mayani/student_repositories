#!/bin/bash

for power in {0..3}; do
    ntasks=`bc <<< "2 ^ $power"`
    sbatch --nodes=1 --ntasks=$ntasks --gpus=$ntasks --export=ALL, --output="Scaling$ntasks.out" --error="Scaling$ntasks.err" scaling_job_GPU.sh
done
