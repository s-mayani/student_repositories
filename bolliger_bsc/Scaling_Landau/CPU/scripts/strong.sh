for i in 0 1 2 3 4
  do
  N=$((2**$i))
  #run for CG without preconditioning:
  #sbatch --nodes=$N --output=scaling_CG_${N}.out --error=scaling_error_CG_${N}.err run.sh 256 256 256 
  
  #run for preconditioned CG:

  sbatch --nodes=$N --output=scaling_Gauss-Seidel_${N}.out --error=scaling_error_Gauss-seidel_${N}.err run.sh 256 256 256 gauss-seidel 2 2 0 
  sbatch --nodes=$N --output=scaling_Richardson_${N}.out --error=scaling_error_Richardson_${N}.err run.sh 256 256 256 richardson 4 0 
  sbatch --nodes=$N --output=scaling_SSOR_${N}.out --error=scaling_error_SSOR_${N}.err run.sh 256 256 256 ssor 4 2 1.57079632679 
done