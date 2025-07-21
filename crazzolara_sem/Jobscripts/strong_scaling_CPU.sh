#!/bin/bash

for power in {0..3}; do
    nodes=`bc <<< "2 ^ $power"`
    sbatch --nodes=$nodes --ntasks=$nodes --ntasks-per-node=1 --export=ALL,nodes=$nodes --output="Scaling$nodes.out" --error="Scaling$nodes.err" scaling_job_CPU.sh
done
