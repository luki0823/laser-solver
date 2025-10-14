/*
==============================================================================
 File:        flux.hpp
 Purpose:     Defines physical flux F(U) for the 1D Euler equations.
 Description:
   Implements the flux vector:
     F = [rho*u, rho*u^2 + p, (E + p)*u]

 Dependencies:
   - eos.hpp  (for pressure)
   - types.hpp

 Notes:
   - Used by numerical flux solvers (e.g., Rusanov, HLLC).
==============================================================================
*/

#pragma once
#include "ls/types.hpp"
#include "ls/eos.hpp"

namespace ls {
  inline State flux_phys(const State& U, double gamma)
  {
    double u = U.rhou / U.rho;
    double p = (gamma - 1.0) * (U.E - 0.5 * U.rho * u * u); 
    
    State F;
    F.rho = U.rho * u;  // mass flux
    F.rhou = U.rho * u * u + p; // momentum flux
    F.E = (U.E + p) * u; // energy flux

    return F;

  }
} // namespace ls