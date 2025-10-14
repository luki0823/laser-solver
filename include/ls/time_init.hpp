/*
==============================================================================
 File:        time_int.hpp
 Purpose:     Time integration utilities for 1D finite-volume Euler solver.

 Math background:
   Semi-discrete finite volume (cell-average U_i):
     dU_i/dt = -(F_{i+1/2} - F_{i-1/2}) / dx + S_i(t)

   Here we implement the homogeneous part (no source yet):
     dU_i/dt = -(F_{i+1/2} - F_{i-1/2}) / dx

   Numerical interface flux uses Rusanov (Local Lax–Friedrichs):
     F* = 0.5*(F_L + F_R) - 0.5*a_max*(U_R - U_L), 
     a_max = max(|u_L| + c_L, |u_R| + c_R)

 Provided functions:
   - cfl_dt():       compute dt from CFL condition
   - flux_divergence(): assemble RHS (dU/dt) from interface fluxes
   - step_euler():   single Forward–Euler step (with optional source hook)

 Design notes:
   - Reconstruction is first-order (piecewise-constant) via reconstruct_first().
     Interface states (UL, UR) come from a ghosted array respecting BCs.
   - The API keeps a BC selector (BCKind) so you can swap BCs later without
     changing the time integrator.
   - step_euler() accepts an optional "post_source" lambda so you can add
     laser heating later without rewriting the integrator.

 Future:
   - Add SSP-RK3 with the same RHS + optional source pattern.
==============================================================================
*/
#pragma once
#include "ls/types.hpp"
#include "ls/eos.hpp"
#include "ls/riemann.hpp"
#include "ls/recon.hpp"

#include <vector>
#include <algorithm>
#include <functional>
#include <cmath>

namespace ls {

// -----------------------------------------------------------------------------
// Compute dt from the CFL condition:
//   dt = CFL * dx / max_i( |u_i| + c_i ),  where c = sqrt(gamma p / rho)
// Guards with a small epsilon to avoid div-by-zero if the state is trivial.
// -----------------------------------------------------------------------------
inline double cfl_dt(const std::vector<State>& U, double gamma, double dx, double CFL)
{
    double amax = 0.0;
    for (const auto& s : U) {
        const double rho = std::max(s.rho, 1e-14);  // avoid blowups
        const double u   = s.rhou / rho;
        const double p   = (gamma - 1.0) * (s.E - 0.5 * rho * u * u);
        const double c   = std::sqrt(std::max(0.0, gamma * p / rho));
        amax = std::max(amax, std::fabs(u) + c);
    }
    return CFL * dx / std::max(amax, 1e-14);
}

// -----------------------------------------------------------------------------
// Build RHS = dU/dt from interface fluxes:
//   dU_i/dt = -( F_{i+1/2} - F_{i-1/2} ) / dx
//
// Steps:
//   1) Reconstruct interface states (UL, UR) using first-order + BCs
//   2) Compute numerical flux at each interface with Rusanov
//   3) Take conservative difference to form RHS per cell
//
// Arrays:
//   U     : size nx         (cell averages)
//   UL/UR : size nx+1       (interface states for i=0..nx)
//   F     : size nx+1       (interface fluxes for i=0..nx)
//   dU    : size nx         (cell RHS)
// -----------------------------------------------------------------------------
inline void flux_divergence(const std::vector<State>& U,
                            double gamma,
                            double dx,
                            BCKind bc,
                            std::vector<State>& dU)
{
    const int nx = static_cast<int>(U.size());
    dU.assign(nx, {0.0, 0.0, 0.0});

    // 1) First-order reconstruction -> interface states
    std::vector<State> UL, UR;
    reconstruct_first(U, gamma, bc, UL, UR);

    // 2) Numerical flux at interfaces (nx+1 faces)
    std::vector<State> F(nx + 1);
    for (int i = 0; i <= nx; ++i) {
        F[i] = rusanov_flux(UL[i], UR[i], gamma);
    }

    // 3) Conservative difference: -(F[i+1/2] - F[i-1/2]) / dx
    for (int i = 0; i < nx; ++i) {
        dU[i].rho  = -(F[i + 1].rho  - F[i].rho ) / dx;
        dU[i].rhou = -(F[i + 1].rhou - F[i].rhou) / dx;
        dU[i].E    = -(F[i + 1].E    - F[i].E   ) / dx;
    }
}

// -----------------------------------------------------------------------------
// Single Forward–Euler step:
//   U^{n+1} = U^n + dt * RHS(U^n, t^n)
// with optional post_source(dt, t+dt, Utemp) hook applied AFTER the Euler update
// so you can add e.g., laser energy to the energy equation later.
//
// Returns the dt it used (so the caller can accumulate time).
// -----------------------------------------------------------------------------
inline double step_euler(std::vector<State>& U,
                         double gamma,
                         double dx,
                         double CFL,
                         BCKind bc,
                         double t,
                         const std::function<void(double /*dt*/, double /*tnext*/, std::vector<State>& /*U*/)> &post_source = nullptr)
{
    // Compute stable dt from CFL:
    const double dt = cfl_dt(U, gamma, dx, CFL);

    // Build RHS from flux divergence:
    std::vector<State> dU;
    flux_divergence(U, gamma, dx, bc, dU);

    // Forward–Euler update:
    for (size_t i = 0; i < U.size(); ++i) {
        U[i] = U[i] + dt * dU[i];
    }

    // Optional source term applied after Euler step (explicit, unsplit simple form)
    if (post_source) {
        post_source(dt, t + dt, U);
    }

    return dt;
}

} // namespace ls
