# Building blocks for Finite Element computations in IPPL

Bachelor Thesis by Lukas Bühler

## Links & Resources

- [Thesis](https://gitlab.psi.ch/AMAS-students/buehler_bsc/-/raw/main/thesis/thesis.pdf)
- [Final Presentation](https://gitlab.psi.ch/AMAS-students/buehler_bsc/-/raw/main/presentations/3_final/pres_final.pdf)
- [Mid-term Presentation](https://gitlab.psi.ch/AMAS-students/buehler_bsc/-/raw/main/presentations/2_mid_term/pres_mid_term.pdf)
- [Three-week Presentation](https://gitlab.psi.ch/AMAS-students/buehler_bsc/-/raw/main/presentations/1_three_week/pres_three_week.pdf)
- [IPPL 'fem-framework' branch (on s-mayani's fork)](https://github.com/s-mayani/ippl/tree/fem-framework)

## Building IPPL

Building and installation instructions can also be found in the Appendix of the [thesis](https://gitlab.psi.ch/AMAS-students/buehler_bsc/-/raw/main/thesis/thesis.pdf).

1. Clone the [IPPL build scripts repository](https://github.com/IPPL-framework/ippl-build-scripts)
2. Edit the file `300-build-ippl` to fetch the IPPL fork and switch the branch

   The following text block starting on line 32:

   ```sh
   cd "${ITB_SRC_DIR}"
   if [ -d "ippl" ]
   then
       echo "Found existing IPPL source directory"
   else
       echo "Clone ippl repo ... "
       #if [ ! -z "$USE_SSH" ]; then
           git clone git@github.com:s-mayani/ippl.git
           git fetch
           git switch fem-framework
       #else
       #    git clone https://github.com/IPPL-framework/ippl.git
       #fi
   fi
   ```

3. Run the 'build everything' script with the following options:

   ```
   ./999-build-everything -t serial --ippl --heffte --kokkos --nounit --export
   ```

## Helpful resources

- [PSI Thesis Tools and Tips](http://amas.web.psi.ch/people/aadelmann/ETH-Accel-Lecture-1/projects/good-to-know.html)
- [DefElement "an encyclopedia of finite element definitions"](https://defelement.com)
