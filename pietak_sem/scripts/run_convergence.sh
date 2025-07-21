#!/bin/bash

export COMMAND="./test/solver/fem/TestMaxwellDiffusionZeroBC 2"
export FILE="performance_final_trig_2d.csv"
sbatch run_convergence_template.sh

export COMMAND="./test/solver/fem/TestMaxwellDiffusionZeroBC 3"
export FILE="performance_final_trig_3d.csv"
sbatch run_convergence_template.sh


export COMMAND="./test/solver/fem/TestMaxwellDiffusionPolyZeroBC 2"
export FILE="performance_final_poly_2d.csv"
sbatch run_convergence_template.sh

export COMMAND="./test/solver/fem/TestMaxwellDiffusionPolyZeroBC 3"
export FILE="performance_final_poly_3d.csv"
sbatch run_convergence_template.sh
