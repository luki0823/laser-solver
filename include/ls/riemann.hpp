/*
==============================================================================
 File:        riemann.hpp
 Purpose:     Defines numerical flux functions for the Euler equations.
 Description:
   Implements the Rusanov (Local Lax–Friedrichs) flux:
     F* = 0.5 * (F_L + F_R) - 0.5 * a_max * (U_R - U_L)
   where a_max = max(|u_L| + c_L, |u_R| + c_R).

 Math Background:
   - Derived from the conservative form of Euler equations.
   - Adds numerical dissipation proportional to local wave speeds,
     ensuring stability near discontinuities (e.g., shocks).

 Dependencies:
   - flux.hpp (for F(U))
   - eos.hpp  (for sound speed)
   - types.hpp

 Notes:
   - Rusanov is robust but diffusive.
   - Later, this can be replaced with HLLC for sharper contact waves.
==============================================================================
*/
#pragma once
#include "ls/types.hpp"
#include "ls/eos.hpp"
#include "ls/flux.hpp"
#include <algorithm>
#include <cmath>

namespace ls {

// -----------------------------------------------------------------------------
// Rusanov (Local Lax–Friedrichs) numerical flux function
// -----------------------------------------------------------------------------
inline State rusanov_flux(const State& UL, const State& UR, double gamma)
{
    // --- Step 1: compute physical fluxes on both sides ---
    State FL = flux_phys(UL, gamma);  // left flux F(U_L)
    State FR = flux_phys(UR, gamma);  // right flux F(U_R)

    // --- Step 2: compute local velocities and sound speeds ---
    double uL = UL.rhou / UL.rho;                     // left velocity
    double uR = UR.rhou / UR.rho;                     // right velocity
    double pL = (gamma - 1.0) * (UL.E - 0.5 * UL.rho * uL * uL); // left pressure
    double pR = (gamma - 1.0) * (UR.E - 0.5 * UR.rho * uR * uR); // right pressure
    double cL = std::sqrt(gamma * pL / UL.rho);       // left sound speed
    double cR = std::sqrt(gamma * pR / UR.rho);       // right sound speed

    // --- Step 3: find the maximum characteristic wave speed ---
    double a_max = std::max(std::fabs(uL) + cL, std::fabs(uR) + cR);

    // --- Step 4: compute the Rusanov flux at the interface ---
    // The central term (FL + FR)/2 averages left/right fluxes
    // The dissipative term adds stability proportional to a_max
    State F;
    F.rho  = 0.5 * (FL.rho  + FR.rho ) - 0.5 * a_max * (UR.rho  - UL.rho );
    F.rhou = 0.5 * (FL.rhou + FR.rhou) - 0.5 * a_max * (UR.rhou - UL.rhou);
    F.E    = 0.5 * (FL.E    + FR.E   ) - 0.5 * a_max * (UR.E    - UL.E   );

    return F;
}

} // namespace ls