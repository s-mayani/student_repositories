// In ChargedParticles.cpp:
// sp.add("output_type",Solver_t::GRAD) -> sp.add("output_type",Solver_t::SOL)
P->scatterCIC(NP, 0, hr);
P->solver_mp->solve();
P->E_m = - grad(P->rho_m);
P->gatherCIC();
