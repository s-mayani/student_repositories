## Data Visualization Files

Every directory contains its own `README.md` that lists its contents further.



**Overview of present directories:**

* `./coefficients_study/`: Notebooks for analyzing the collisional coefficients dumped by the Langevin solver
* `./hessian_convergence_study`: Notebook for visualizing the convergence rates of the custom Hessian operator
* `./langevin_study/`: Notebook for visualizing emittances of both Langevin and Ulmer's reference solver. 
  * Also allows explorations of other beam statistics (`vmean`, `rmean`, `rvrms`, ...)
  * Contains minimal Notebook for visualizing VTK files without ParaView (see `./langevin_study/plot_vtk_data/`)
* `./src/`: Contains often used utility functions for the notebooks contained in this directory. Also contains `./src/mscPlotting/mplstyle` defining project wide font style for matplotlib plots