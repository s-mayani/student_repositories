# Run scripts

This folder contains the scripts which are needed to generate the date for the report. Note that all the scripts need to be run in the root of the build folder of `IPPL` and that it expects that the targets `TestMaxwellDiffusionZeroBC`, `TestMaxwellDiffusionPolyZeroBC`, and `TestMaxwellDiffusionPolyZeroBCTimed` are compiled. For the compilation see the `IPPL` documentation.

The important two scripts are:
 - `run_convergence.sh` A bash script which generates the convergence study data. This program will launch 4 slurm jobs which will generate the corresponding output files, which then can be copied over to the corresponding folders in the `/report/` part. **Note** that if the slurm parameters need to be changed the `run_convergence_template.sh` file should be looked at.
 - `run_scaling.sh` A bash script which generates the scaling study data. This program will launch a bunch (~528) slurm jobs which then will generate the correct data for the scaling study. **Note:** One most likely needs to adjust the slurm parameters to run the program, for this see the corresponding section in the `run_scaling.sh` file.