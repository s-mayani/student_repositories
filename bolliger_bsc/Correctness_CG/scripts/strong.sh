for i in 6 7 8 9
do
    N=$((2**$i))

    sbatch --gpus=4 --ntasks=4 --output=convergence_CG_${N}.out --error=convergence_CG_${N}.err run.sh $i
    sbatch --gpus=4 --ntasks=4 --output=convergence_gauss-seidel_${N}.out --error=convergence_op_gauss-seidel_${N}.err run.sh $i gauss-seidel 2 2 0
    sbatch --gpus=4 --ntasks=4 --output=convergence_richardson_${N}.out --error=convergence_richardson_${N}.err run.sh $i richardson 4 0
    sbatch --gpus=4 --ntasks=4 --output=convergence_ssor_${N}.out --error=convergence_ssor_${N}.err run.sh $i ssor 4 2 1.57079632679
done