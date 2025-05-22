for i in 2
  do
  N=$((2**$i))
    #run for CG without preconditioning:

    sbatch --gpus=$N --ntasks=$N --output=correctness_CG_${N}.out --error=correctness_error_CG_${N}.err run.sh 32 32 32


    #run for preconditioned CG:

    #sbatch --gpus=$N --ntasks=$N --output=correctness_Gauss-Seidel_${N}.out --error=correctness_error_Gauss-seidel_${N}.err run.sh 32 32 32 gauss-seidel 2 2 0
    #sbatch --gpus=$N --ntasks=$N --output=correctness_Richardson_${N}.out --error=correctness_error_Richardson_${N}.err run.sh 32 32 32 richardson 4 0
    #sbatch --gpus=$N --ntasks=$N --output=correctness__SSOR_${N}.out --error=correctness_error_SSOR_${N}.err run.sh 32 32 32 ssor 4 2 1.57079632679 
done