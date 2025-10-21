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
#include "ls/bc.hpp"

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
                            std::vector<State>& dU,
                            Recon recon)
{
    const int nx = static_cast<int>(U.size());
    dU.assign(nx, {0.0, 0.0, 0.0});

    // 1) First-order reconstruction -> interface states
    std::vector<State> UL, UR;
    if (recon == Recon::WENO3)
        reconstruct_weno3(U, gamma, bc, UL, UR);
    else
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



//inline double step_euler(std::vector<State>& U,//
//                         double gamma,
//                         double dx,
//                         double CFL,
//                         BCKind bc,
//                         double t,
 //                        const std::function<void(double /*dt*/, double /*tnext*/, std::vector<State>& /*U*/)> &post_source = nullptr)
/*{
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

*/



inline double rk3_step(std::vector<State>& U,
                       double t,
                       double gamma,
                       double dx,
                       double CFL,
                       BCKind bc,
                       Recon recon,
                       const std::function<void(double /*dt*/, double /*tnext*/, std::vector<State>& /*U*/)> &post_source = nullptr)
{
    const double dt = cfl_dt(U,gamma,dx,CFL);

    // -- Stage 1 U1 = U^n + dt * L(U^n, t)
    std::vector<State> Ln; //L(Un)
    flux_divergence(U, gamma, dx, bc, Ln, recon);
    std::vector<State> U1 = U;
    for (size_t i = 0; i < U.size(); ++i){
        U1[i] = U1[i] + dt * Ln[i];
    }
    if (post_source) post_source(dt, t+dt, U1);

    // Stage 2 3/4 U^n + 1/4 * (U1 + dt * L( U1, t+dt))
    std::vector<State> L1;
    flux_divergence(U1, gamma, dx, bc, L1, recon);
    std::vector<State> U2 = U1;

    for (size_t i = 0; i<U.size(); ++i){
        State tmp = U1[i] + dt*L1[i];
        //convex combination
        U2[i].rho  = 0.75 * U[i].rho  + 0.25 * tmp.rho;
        U2[i].rhou = 0.75 * U[i].rhou + 0.25 * tmp.rhou;
        U2[i].E    = 0.75 * U[i].E    + 0.25 * tmp.E;

    }
    if (post_source) post_source(dt, t + dt, U2);

    // --- Stage 3: U^{n+1} = 1/3 U^n + 2/3 ( U2 + dt * L(U2, t+dt) )
    std::vector<State> L2;
    flux_divergence(U2, gamma, dx, bc, L2, recon);
    for (size_t i = 0; i < U.size(); ++i) {
        State tmp = U2[i] + dt * L2[i];
        U[i].rho  = (1.0/3.0) * U[i].rho  + (2.0/3.0) * tmp.rho;
        U[i].rhou = (1.0/3.0) * U[i].rhou + (2.0/3.0) * tmp.rhou;
        U[i].E    = (1.0/3.0) * U[i].E    + (2.0/3.0) * tmp.E;
    }
    if (post_source) post_source(dt, t + dt, U);

    return dt; // caller advances time by this
}

inline void advance_ssprk3(std::vector<State>& U, double& t, double t_end,
                           double gamma, double dx, double CFL,
                           BCKind bc, Recon recon,
                           auto&& on_step, auto&& post_source)
{
    using namespace ls;
    const int nx_total = (int)U.size();
    const int nx_phys  = nx_total - 2*NG;   // real cells only

    std::vector<State> L(nx_total), U1(nx_total), U2(nx_total);

    int step = 0;
    while (t < t_end)
    {
        //--------------------------------------------------
        // 1️⃣ Fill ghosts before any flux/divergence call
        //--------------------------------------------------
        fill_ghosts(U, bc);

        // CFL based on physical cells
        double dt = cfl_dt(U, gamma, dx, CFL);
        if (t + dt > t_end) dt = t_end - t;

        //--------------------------------------------------
        // 2️⃣ Stage 1
        //--------------------------------------------------
        flux_divergence(U, gamma, dx, bc, L, recon);
        for (int i = NG; i < NG + nx_phys; ++i)
            U1[i] = U[i] + dt * L[i];

        fill_ghosts(U1, bc);
        if (post_source) post_source(t, dt, U1);

        //--------------------------------------------------
        // 3️⃣ Stage 2
        //--------------------------------------------------
        flux_divergence(U1, gamma, dx, bc, L, recon);
        for (int i = NG; i < NG + nx_phys; ++i)
            U2[i] = 0.75*U[i] + 0.25*(U1[i] + dt*L[i]);

        fill_ghosts(U2, bc);
        if (post_source) post_source(t + 0.5*dt, dt, U2);

        //--------------------------------------------------
        // 4️⃣ Stage 3
        //--------------------------------------------------
        flux_divergence(U2, gamma, dx, bc, L, recon);
        for (int i = NG; i < NG + nx_phys; ++i)
            U[i] = (1.0/3.0)*U[i] + (2.0/3.0)*(U2[i] + dt*L[i]);

        fill_ghosts(U, bc);
        if (post_source) post_source(t + dt, dt, U);

        //--------------------------------------------------
        // 5️⃣ Advance time and call callback
        //--------------------------------------------------
        t += dt;
        ++step;
        on_step(dt, t, U);
    }
}


} // namespace ls
