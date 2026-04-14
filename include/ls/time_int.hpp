/*
==============================================================================
 File:        time_int.hpp
 Purpose:     SSP RK3 time integration for 1D and 2D Euler
 Author:      Lucas Pierce
==============================================================================
*/
#pragma once

#include <vector>
#include <cmath>
#include <algorithm>

#ifdef LS_USE_OPENMP
  #include <omp.h>
#endif

#include "ls/types.hpp"
#include "ls/bc.hpp"
#include "ls/fv_update.hpp"
#include "ls/mesh.hpp"
#include "ls/eos.hpp"

namespace ls {

// ============================================================================
// Enforce Positivity + Vacuum Safety
//   - Floors rho and internal energy
//   - Prevents huge velocities in near-vacuum by killing momentum
// ============================================================================
inline void enforce_positivity_2d(std::vector<ls::State2D>& U,
                                 const ls::Mesh2D& mesh,
                                 const ls::EOSIdealGas& eos)
{
    (void)eos;

    const double rho_min = 1e-12;    // consistent with eos.hpp
    const double e_min   = 1e-10;


    // NEW: treat anything below this as "numerical vacuum"
    // Start with 1e-6 or 1e-5 for stability, then tighten later.
    //const double rho_vac = 1e-6;

    const double u_max = 2e20;

    const int i0 = mesh.interior_i_start();
    const int i1 = mesh.interior_i_end();
    const int j0 = mesh.interior_j_start();
    const int j1 = mesh.interior_j_end();

    #ifdef LS_USE_OPENMP
    #pragma omp parallel for collapse(2) schedule(static)
    #endif
        for (int J = j0; J <= j1; ++J) {
            for (int I = i0; I <= i1; ++I) {
            // ---- inside the (I,J) loop ----
            auto& Ui = U[mesh.index(I,J)];

            // sanitize NaNs/Infs
            if (!std::isfinite(Ui.rho))  Ui.rho  = rho_min;
            if (!std::isfinite(Ui.rhou)) Ui.rhou = 0.0;
            if (!std::isfinite(Ui.rhov)) Ui.rhov = 0.0;
            if (!std::isfinite(Ui.E))    Ui.E    = e_min;

            // density floor
            if (Ui.rho < rho_min) {
            const double rho_old = Ui.rho;
            const double u_old = (std::isfinite(rho_old) && std::fabs(rho_old) > 0.0) ? (Ui.rhou / rho_old) : 0.0;
            const double v_old = (std::isfinite(rho_old) && std::fabs(rho_old) > 0.0) ? (Ui.rhov / rho_old) : 0.0;

            Ui.rho  = rho_min;
            Ui.rhou = rho_min * u_old;
            Ui.rhov = rho_min * v_old;
            }

            // velocities
            double u = Ui.rhou / Ui.rho;
            double v = Ui.rhov / Ui.rho;

            // speed cap (avoid "vacuum -> infinite u")
            const double u_max = 2e20; // tune if needed
            double speed = std::sqrt(u*u + v*v);
            if (!std::isfinite(speed) || speed > u_max) {
                const double s = u_max / (speed + 1e-300);
                u *= s; v *= s;
                Ui.rhou = Ui.rho * u;
                Ui.rhov = Ui.rho * v;
            }

            // positivity of internal energy
            const double kinetic = 0.5 * Ui.rho * (u*u + v*v);
            double e_int = Ui.E - kinetic;
            if (!std::isfinite(e_int) || e_int < e_min) {
                Ui.E = kinetic + e_min;
            }
        }
    }
}


// ============================================================================
// 1D RK3 integrator
// ============================================================================
inline void advance_rk3_1d(
    std::vector<State1D>& U,
    std::vector<State1D>& rhs,
    const Mesh1D& mesh,
    const EOSIdealGas& eos,
    const Bc1D& bc,
    ReconType recon_type,
    double dt)
{
    std::vector<State1D> U0 = U;
    std::vector<State1D> U1 = U;
    std::vector<State1D> U2 = U;

    // Stage 1
    apply_bc_1d(U0, mesh, bc);
    compute_rhs_1d(U0, rhs, mesh, eos, recon_type);

    for (int I = 0; I < mesh.nx_tot; ++I)
        U1[I] = U0[I] + rhs[I] * dt;

    // Stage 2
    apply_bc_1d(U1, mesh, bc);
    compute_rhs_1d(U1, rhs, mesh, eos, recon_type);

    for (int I = 0; I < mesh.nx_tot; ++I)
        U2[I] = 0.75 * U0[I] + 0.25 * (U1[I] + rhs[I] * dt);

    // Stage 3
    apply_bc_1d(U2, mesh, bc);
    compute_rhs_1d(U2, rhs, mesh, eos, recon_type);

    for (int I = 0; I < mesh.nx_tot; ++I)
        U[I] = (1.0/3.0) * U0[I] + (2.0/3.0) * (U2[I] + rhs[I] * dt);

    apply_bc_1d(U, mesh, bc);
}

// ============================================================================
// 2D RK3 integrator
// ============================================================================
inline void advance_rk3_2d(
    std::vector<State2D>& U,
    std::vector<State2D>& rhs,
    const Mesh2D& mesh,
    const EOSIdealGas& eos,
    const Bc2D& bc,
    ReconType recon_type,
    double dt)
{
    std::vector<State2D> U0 = U;
    std::vector<State2D> U1 = U;
    std::vector<State2D> U2 = U;

    // ------------------------------------------------------------
    // Stage 1
    // ------------------------------------------------------------
    apply_bc_2d(U0, mesh, bc);
    compute_rhs_2d(U0, rhs, mesh, eos, recon_type);

    #ifdef LS_USE_OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int k = 0; k < (int)U.size(); ++k)
        U1[k] = U0[k] + rhs[k] * dt;

    enforce_positivity_2d(U1, mesh, eos);

    // ------------------------------------------------------------
    // Stage 2
    // ------------------------------------------------------------
    apply_bc_2d(U1, mesh, bc);
    compute_rhs_2d(U1, rhs, mesh, eos, recon_type);

    #ifdef LS_USE_OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int k = 0; k < (int)U.size(); ++k)
        U2[k] = 0.75 * U0[k] + 0.25 * (U1[k] + rhs[k] * dt);

    enforce_positivity_2d(U2, mesh, eos);

    // ------------------------------------------------------------
    // Stage 3
    // ------------------------------------------------------------
    apply_bc_2d(U2, mesh, bc);
    compute_rhs_2d(U2, rhs, mesh, eos, recon_type);

    #ifdef LS_USE_OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int k = 0; k < (int)U.size(); ++k)
        U[k] = (1.0/3.0) * U0[k] + (2.0/3.0) * (U2[k] + rhs[k] * dt);

    apply_bc_2d(U, mesh, bc);
    enforce_positivity_2d(U, mesh, eos);

}

// ============================================================================
// 2D SSP RK3 with source term S(x,y,t) on energy equation
// ============================================================================
template <typename Source>
inline void advance_rk3_2d_with_source(
    std::vector<State2D>& U,
    std::vector<State2D>& rhs,
    const Mesh2D& mesh,
    const EOSIdealGas& eos,
    const Bc2D& bc,
    ReconType recon_type,
    const Source& S,
    double t_n,
    double dt)
{
    const int N = mesh.nx_tot * mesh.ny_tot;

    std::vector<State2D> U0 = U;
    std::vector<State2D> U1(N);
    std::vector<State2D> U2(N);

    // Stage 1
    apply_bc_2d(U, mesh, bc);
    compute_rhs_2d_with_source(U, rhs, mesh, eos, recon_type, S, t_n, dt);

    #ifdef LS_USE_OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int i = 0; i < N; ++i) {
        U1[i] = U0[i] + rhs[i] * dt;
    }

    enforce_positivity_2d(U1, mesh, eos);

    // Stage 2
    apply_bc_2d(U1, mesh, bc);
    compute_rhs_2d_with_source(U1, rhs, mesh, eos, recon_type, S, t_n + dt,dt);

    #ifdef LS_USE_OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int i = 0; i < N; ++i) {
        U2[i] = State2D{
            .rho  = 0.75 * U0[i].rho  + 0.25 * (U1[i].rho  + rhs[i].rho  * dt),
            .rhou = 0.75 * U0[i].rhou + 0.25 * (U1[i].rhou + rhs[i].rhou * dt),
            .rhov = 0.75 * U0[i].rhov + 0.25 * (U1[i].rhov + rhs[i].rhov * dt),
            .E    = 0.75 * U0[i].E    + 0.25 * (U1[i].E    + rhs[i].E    * dt)
        };
    }

    enforce_positivity_2d(U2, mesh, eos);

    // Stage 3
    apply_bc_2d(U2, mesh, bc);
    compute_rhs_2d_with_source(U2, rhs, mesh, eos, recon_type, S, t_n + 0.5 * dt,dt);

    #ifdef LS_USE_OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int i = 0; i < N; ++i) {
        U[i].rho  = (1.0/3.0) * U0[i].rho  + (2.0/3.0) * (U2[i].rho  + rhs[i].rho  * dt);
        U[i].rhou = (1.0/3.0) * U0[i].rhou + (2.0/3.0) * (U2[i].rhou + rhs[i].rhou * dt);
        U[i].rhov = (1.0/3.0) * U0[i].rhov + (2.0/3.0) * (U2[i].rhov + rhs[i].rhov * dt);
        U[i].E    = (1.0/3.0) * U0[i].E    + (2.0/3.0) * (U2[i].E    + rhs[i].E    * dt);
    }

    enforce_positivity_2d(U, mesh, eos);
}

} // namespace ls
