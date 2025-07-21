## Generalized Chainable Differential Operators

This repo has been created to test the feasibility of chainable (templated) differential stencils.
With these one can write more complicated Operators in a concise and extensible manner (i.e. Hessian).

In the file `field.h` there is a datastructure imitating the behavior of IPPL::Barefield for Scalar and Matrix fields.
The chainable operators are defined in `hessian.h`
