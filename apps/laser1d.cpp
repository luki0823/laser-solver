/*
==============================================================================
 File:        laser1d.cpp
 Purpose:     Main driver for 1D Euler + laser heating simulation.
 Description:
   Initializes grid, variables, and runs time integration loop using
   finite-volume discretization with Rusanov flux and explicit time-stepping.

 Workflow:
   1. Initialize grid and state (rho, u, p)
   2. Loop over time:
        - Compute flux divergence
        - Add source term
        - Update with Euler / SSP-RK3
   3. Write results to CSV.

 Output:
   data/out/stepXXXX.csv for post-processing.

 Future:
   - Extend to 2D axisymmetric
   - Replace Rusanov with HLLC
   - Add MUSCL reconstruction and RK3 integrator
==============================================================================
*/


#include "ls/eos.hpp"
#include "ls/flux.hpp"
#include "ls/riemann.hpp"
#include <iostream>

int main() {
    using namespace ls;

    const double gamma = 1.4;

    // Left and right primitive states
    double rhoL = 1.0, uL = 0.0, pL = 1e5;
    double rhoR = 0.125, uR = 0.0, pR = 1e4;

    // Convert to conservatives
    State UL = prim_to_cons(rhoL, uL, pL, gamma);
    State UR = prim_to_cons(rhoR, uR, pR, gamma);

    // Compute Rusanov flux at interface
    State Fstar = rusanov_flux(UL, UR, gamma);

    std::cout << "Rusanov interface flux:\n";
    std::cout << "  Mass flux     = " << Fstar.rho  << "\n";
    std::cout << "  Momentum flux = " << Fstar.rhou << "\n";
    std::cout << "  Energy flux   = " << Fstar.E    << "\n";
    return 0;
}