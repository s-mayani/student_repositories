---
marp: true
paginate: true
# auto-scaling true
---

## Chainable Stencil Operators
<!-- <br/> -->
Achieves Order 2 error convergence. `mixed` being `GeneralizedHessOp<DiffType::Centered, DiffType::Forward, Difftype::Backward>(field, hInv);`

![width:400px](figures/rel_error_avg.png)
<!-- ![width:400px](figures/x_emittance_serial.png) -->

---
## Resimulate DIH Problem with P3M

There can appear an unphysical peak at higher `r_cut`. This is not due to parallelization. See the serial version below:
![width:400px](figures/x_emittance_serial.png)

Need to look at other spatial components of Emittance. If there is only a peak for the x-dimension there might be something wrong with the particles themselves? (Check this assertion with Sonali)