## Source code
**Implementations can be found here:**
- [Solver](https://github.com/manuel5975p/ippl/tree/migrate_from_working_experiments/src/MaxwellSolvers)
- [Renderer](https://github.com/manuel5975p/ippl/blob/rendering/src/Utility/Rendering.hpp)
- [Free Electron Laser Testcase](https://github.com/manuel5975p/ippl/blob/migrate_from_working_experiments/alpine/FreeElectronLaser.cpp)


## Reproducers
It is assumed that the current working directory is the root of this repository

### Figure 1
Requires [Eigen](https://gitlab.com/libeigen/eigen) and [Sciplot](https://github.com/sciplot/sciplot/)
```
cd experiments
g++ murabc_eigenvalues -O3 -march=native -DNDEBUG -omurabc_eigenvalues
./murabc_eigenvalues
```
### Figure 2
Requires [Eigen](https://gitlab.com/libeigen/eigen)
```
cd experiments
g++ boris.cpp -O3 -march=native -DNDEBUG -oboris
./boris
./boris_gammaplot.gp
```
### Figure 3
Replace `sm_80` (Merlin) with your GPU arch.
```
cd experiments/fft_vs_stencil
nvcc -std=c++20 -arch=sm_80 bench.cu -lcufft -lcublas -o fftbench
./fftbench
./plot.gp
```

### Figure 6 & 7
```
git clone https://github.com/manuel5975p/ippl.git
cd ippl
git checkout -b migrate_from_working_experiments 059807ac100669d5910effa6c78b12affeaab6ce
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DIPPL_PLATFORMS=openmp -DENABLE_SOLVERS=True -DENABLE_TESTS=True
make TestFDTDSolver -j4
./test/solver/TestFDTDSolver
../plotscripts/gaussplot.gp
../plotscripts/ampereplot.gp
```
### Figure 8
```py
git clone https://github.com/manuel5975p/ippl.git
cd ippl
mkdir build
cd build
cmake .. -DIPPL_PLATFORMS=cuda -DENABLE_SOLVERS=True -DENABLE_TESTS=True -DCMAKE_BUILD_TYPE=Release
make TestFDTDSolver -j4
./test/solver/TestFDTDSolver #Maybe with srun?
git clone https://github.com/aryafallahi/mithra.git
cd mithra/src
mpicxx *.cpp beam.cc -DNDEBUG -O3 -march=native -flto -flto-partition=one -omithra
mpirun <mpi args> ./mithra ../prj/FEL-IR/job-files/FEL-IR-a.job
cp power-sampling/power-NSFD-0.txt ../../
cd ../..
gnuplot
> set logscale y
> set yrange [1:]
> plot 'radiation.txt' w lines, 'power-NSDF-0.txt' w lines
```
### Figure 10
```
git clone https://github.com/manuel5975p/ippl.git --branch strongscaling
cd ippl
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DENABLE_SOLVERS=True -DENABLE_TESTS=True
make TestFDTDSolver -j12
cp ../../*.slurm .
sbatch bench1gpu.slurm
sbatch bench2gpu.slurm
sbatch bench4gpu.slurm
sbatch bench8gpu.slurm
cat *.out > strong_scaling.txt
../../plots/wscplot.gp
```
### Figure 9
```
git clone https://github.com/manuel5975p/ippl.git --branch weak_scaling
cd ippl
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DENABLE_SOLVERS=True -DENABLE_TESTS=True
make TestFDTDSolver -j12
cp ../../*.slurm .
sbatch bench1gpu.slurm
sbatch bench2gpu.slurm
sbatch bench4gpu.slurm
sbatch bench8gpu.slurm
cat *.out > weak_scaling.txt
../../plots/scplot.gp
```