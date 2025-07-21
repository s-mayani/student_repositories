#!/bin/bash

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# This function sets the slurm parameters that should be used, adjust it to fit
# the system you are running this on.
# Values that can be set are:
#    - NTASKS: sets the total number of tasks
#    - TASKS_PER_NODE: the number of tasks per node
# Note that additionally the same number of gpus are requested as there are
# tasks. For a closer look at this check the file "run_scaling_template.sh"
# ------------------------------------------------------------------------------
set_slurm_params () {
    if (( num_ranks == 8))
    then
        sed -i "s/NTASKS/$num_ranks/g" run_scaling_template_tmp.sh
        sed -i "s/TASKS_PER_NODE/4/g" run_scaling_template_tmp.sh
    else
        sed -i "s/NTASKS/$num_ranks/g" run_scaling_template_tmp.sh
        sed -i "s/TASKS_PER_NODE/$num_ranks/g" run_scaling_template_tmp.sh
    fi
}
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------


# The strong scaling plots
if true; then
outfile=data_strong_2d_v2.csv
echo "num_nodes num_ranks cell_spacing interp_error_coef recon_error solver_residue num_i time" > $outfile


for num_ranks in 1 2 4 8
do  
    cp run_scaling_template.sh run_scaling_template_tmp.sh
    set_slurm_params

    for n in 4096 8192 16384
    do
        echo $n
        export N="$n"
        export DIM="2"
        export OUTFILE="/dev/null"
        sbatch run_scaling_template_tmp.sh
        export OUTFILE="$outfile"
        for j in {1..10}
        do 
            sbatch run_scaling_template_tmp.sh
        done
    done
done
fi


if true; then
outfile=data_strong_3d_v2.csv
echo "num_nodes num_ranks cell_spacing interp_error_coef recon_error solver_residue num_i time" > $outfile


for num_ranks in 1 2 4 8
do  
    cp run_scaling_template.sh run_scaling_template_tmp.sh
    set_slurm_params

    for n in 128 256 512 
    do
        echo $n
        export N="$n"
        export DIM="3"
        export OUTFILE="/dev/null"
        sbatch run_scaling_template_tmp.sh
        export OUTFILE="$outfile"
        for j in {1..10}
        do 
            sbatch run_scaling_template_tmp.sh
        done
    done
done
fi


# The weak scaling plots
if true; then
outfile=data_weak_2d_v2.csv
echo "num_nodes num_ranks cell_spacing interp_error_coef recon_error solver_residue num_i time" > $outfile


for num_ranks in 1 2 4 8
do  
    cp run_scaling_template.sh run_scaling_template_tmp.sh
    set_slurm_params

    for i in 4096 8192 16384
    do
        n=$(printf %.0f $(bc <<< "scale=6;sqrt($num_ranks)*$i"))
        echo $n
        export N="$n"
        export DIM="2"
        export OUTFILE="/dev/null"
        sbatch run_scaling_template_tmp.sh
        export OUTFILE="$outfile"
        for j in {1..10}
        do 
            sbatch run_scaling_template_tmp.sh
        done
    done
done
fi

if true; then
outfile=data_weak_3d_v2.csv
echo "num_nodes num_ranks cell_spacing interp_error_coef recon_error solver_residue num_i time" > $outfile


for num_ranks in 1 2 4 8
do  
    cp run_scaling_template.sh run_scaling_template_tmp.sh
    set_slurm_params

    for i in 128 256 512
    do
        n=$(printf %.0f $(bc -l <<< "scale=6;e(l($num_ranks)/3)*$i"))
        echo $n
        export N="$n"
        export DIM="3"
        export OUTFILE="/dev/null"
        sbatch run_scaling_template_tmp.sh
        export OUTFILE="$outfile"
        for j in {1..10}
        do 
            sbatch run_scaling_template_tmp.sh
        done
    done
done
fi