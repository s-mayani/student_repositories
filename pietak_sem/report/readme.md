# Report

This folder contains the Latex files and the plotting scripts needed for the creation of the report.

The plotting is done in R and the respective files are in the `fig_gen/` folder and there the respective subfolders. The files are:
 - `plot_convergence.r` This will plot the convergence study, the files it requires are `performance_final_trig_2d.csv`, `performance_final_trig_3d.csv`, `performance_final_poly_2d.csv`, and `performance_final_poly_3d.csv` and they need to be located in the same folder as the script. They are created using the `run_convergence.sh` script of the `/scripts/` folder.
 - `plot_scaling_strong.r` This will plot the strong scaling, the files it requires are `data_strong_2d_v2.csv` and `data_strong_3d_v2.csv`, both of them are created using the `run_scaling.sh` scrip of the `/scripts/` folder.
 - `plot_scaling_weak.r` This will plot the weak scaling, the files it requires are `data_weak_2d_v2.csv` and `data_weak_3d_v2.csv`, both of them are created using the `run_scaling.sh` scrip of the `/scripts/` folder.
 - `plotField.r` This script will plot the field which was used as the thumbnail on the title page of the report. It requires the data file `field.csv`, generating this file is somewhat more convoluted as it requires adding additional code to one of the test cases, to print out the data. To this end the `TestMaxwellDiffusionZeroBC.cpp` file needs to be adjusted, first number of points `numPoints` needs to be increased to something like 1000 and secondly the following code needs to be added after the line containing `Kokkos::View funcVals = solver.reconstructToPoints(positions);`:
   ```cpp
   auto hFuncVals = Kokkos::create_mirror_view(funcVals);
   auto hPositions = Kokkos::create_mirror_view(positions);
   Kokkos::deep_copy(hFuncVals, funcVals);
   Kokkos::deep_copy(hPositions, positions);


   for (size_t r = 0; r < ippl::Comm->size(); ++r) <%
       if (ippl::Comm->rank() == r) <%
           std::ofstream file;
           if (r == 0) <%
               file.open("field.csv");
               file << "x,y,vx,vy\n";
           %> else <%
               file.open("field.csv", std::ios::app);
           %>
           for (size_t i = 0; i < hPositions.extent(0); ++i) <%
               file << hPositions(i)<:0:> << "," << hPositions(i)<:1:> << ","
                    << hFuncVals(i)<:0:> << "," << hFuncVals(i)<:1:> << "\n";
           %>
           file.close();
       %>
       ippl::Comm->barrier();
   %>
    ```
   Then run the program with `TestMaxwellDiffusionZeroBC 2 1024`. This will generate a file `field.csv` which will contain the correct data and on this the plotting scrip can be used.

**Note** that all the R files should be run from the folder `/report/` in order for the paths to be correct.