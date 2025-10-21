/*
==============================================================================
 File:        recon.hpp
 Purpose:     Interface reconstruction for finite-volume Euler solvers.
              (This version implements FIRST-ORDER / PIECEWISE-CONSTANT.)

 Math (what this file implements):
   • For each interface i+1/2, we need left/right states (U_L, U_R).
   • First-order (piecewise-constant) sets:
         U_L(i+1/2) = U_i
         U_R(i+1/2) = U_{i+1}
   • To define interfaces at the domain ends, we use 1 ghost cell on each side.
     Here we support two simple BC choices:
       - CopyEnds        : ext[0] = U[0],       ext[nx+1] = U[nx-1]   (open-ish)
       - ReflectLeftCopy : ext[0] mirrors u,    ext[nx+1] = U[nx-1]   (wall at left)

   • With a ghosted array ext of length nx+2:
         UL[i] = ext[i],   UR[i] = ext[i+1],   for i=0..nx

 Why this design:
   • Keeps the reconstruction API stable when you add MUSCL/WENO later
     (same UL/UR outputs, you just change the reconstruction internals).
   • BC handling is explicit but lightweight; easy to swap/extend.

 Dependencies:
   - types.hpp (State).
   - <cmath> for sqrt (only used in the reflective ghost helper).

 Future:
   - Add MUSCL (MC limiter) and WENO3 with the same UL/UR interface.
==============================================================================
*/
#pragma once
#include "ls/types.hpp"
#include <vector>
#include <cmath>
#include <algorithm>

namespace ls {

// -------------------------------
// Simple BC switch
// -------------------------------
enum class BCKind {
  CopyEnds,           // ext[0] = U[0], ext[nx+1] = U[nx-1]
  ReflectLeftCopy,    // reflective wall at left, copy at right
  Periodic
};

enum class Recon {
  FirstOrder,
  WENO3
};


// -----------------------------------------------------------------------------
// Helper: build a LEFT reflective ghost from the first interior state.
//   Wall condition (1D planar):
//     u_g = -u_1,  rho_g = rho_1,  p_g = p_1
//   Then convert back to conservative components.
//   We need p_1 from U_1 and gamma:
//       p = (gamma-1) * (E - 1/2 rho u^2)
// -----------------------------------------------------------------------------
inline State reflective_left_ghost(const State& U1, double gamma)
{
    const double rho = U1.rho;
    const double u1  = U1.rhou / std::max(rho, 1e-14); // guard tiny rho
    const double p1  = (gamma - 1.0) * (U1.E - 0.5 * rho * u1 * u1);
    const double ug  = -u1;

    State Ug;
    Ug.rho  = rho;
    Ug.rhou = rho * ug;
    Ug.E    = p1/(gamma - 1.0) + 0.5 * rho * ug * ug;  // E = p/(γ-1) + 1/2 ρ u^2
    return Ug;
}

// -----------------------------------------------------------------------------
// Build a ghosted array: ext has size nx+2
//   ext[0]      : left ghost
//   ext[1..nx]  : interior copy of U[0..nx-1]
//   ext[nx+1]   : right ghost
// -----------------------------------------------------------------------------
inline void make_ghosted(const std::vector<State>& U,
                         double gamma,
                         BCKind bc,
                         std::vector<State>& ext)
{
    const int nx = static_cast<int>(U.size());
    ext.resize(nx + 2);

    // Right ghost: simple copy-outflow for now (easy to change later)
    ext[nx + 1] = U.back();

    // Left ghost depends on BC
    if (bc == BCKind::ReflectLeftCopy) {
        ext[0] = reflective_left_ghost(U.front(), gamma);
    } else {
        // CopyEnds
        ext[0] = U.front();
    }

    // Interior
    for (int i = 0; i < nx; ++i) ext[i + 1] = U[i];
}

// -----------------------------------------------------------------------------
// First-order (piecewise-constant) reconstruction.
// Output:
//   UL, UR vectors of size (nx+1), containing interface states for i=0..nx.
// Usage pattern in flux divergence:
//   1) make_ghosted(U, gamma, bc, ext)
//   2) for each interface i: UL[i]=ext[i], UR[i]=ext[i+1]
// -----------------------------------------------------------------------------
inline void reconstruct_first(const std::vector<State>& U,
                              double gamma,
                              BCKind bc,
                              std::vector<State>& UL,
                              std::vector<State>& UR)
{
    const int nx = static_cast<int>(U.size());

    // Build ghosted array (nx+2)
    std::vector<State> ext;
    make_ghosted(U, gamma, bc, ext);

    // Interfaces (nx+1)
    UL.resize(nx + 1);
    UR.resize(nx + 1);
    for (int i = 0; i <= nx; ++i) {
        UL[i] = ext[i];       // left state at interface i+1/2
        UR[i] = ext[i + 1];   // right state at interface i+1/2
    }
}


// -----------------------------------------------------------------------------
// WENO3 reconstruction (component-wise)
// Produces left/right interface values UL, UR from cell-centered U
// Reference: Shu (1997), Jiang & Shu (1996)
// -----------------------------------------------------------------------------
inline void reconstruct_weno3(const std::vector<State>& U,
                              double /*gamma*/,
                              BCKind /*bc*/,
                              std::vector<State>& UL,
                              std::vector<State>& UR)
{
    const int nx = (int)U.size();
    UL.assign(nx+1, State{});  // interfaces j = 0..nx
    UR.assign(nx+1, State{});

    const double eps = 1e-6;

    // Helper: WENO3 for a scalar q on cell centers -> interface arrays qL, qR
    auto weno3_scalar = [&](const std::vector<double>& q, std::vector<double>& qL, std::vector<double>& qR)
    {
        const int n = (int)q.size();
        qL.assign(n+1, 0.0);   // j = 0..n
        qR.assign(n+1, 0.0);

        // Left state at interface j = i+1/2, computed from cell i stencil (i-1, i, i+1)
        // This fills j = 2..(n-1)
        for (int i = 1; i <= n-2; ++i) {
            double q0 = 1.5*q[i] - 0.5*q[i-1];
            double q1 = 0.5*q[i] + 0.5*q[i+1];
            double b0 = (q[i]   - q[i-1])*(q[i]   - q[i-1]);
            double b1 = (q[i+1] - q[i])  *(q[i+1] - q[i]);
            double a0 = (1.0/3.0) / ((eps + b0)*(eps + b0));
            double a1 = (2.0/3.0) / ((eps + b1)*(eps + b1));
            double w0 = a0 / (a0 + a1);
            double w1 = a1 / (a0 + a1);
            qL[i+1] = w0*q0 + w1*q1;   // fills j = i+1
        }

        // Right state at interface j = i+1/2, computed from cell (i+1) stencil mirrored
        // Here we fill j = 1..(n-2)
        for (int i = 1; i <= n-2; ++i) {
            // mirrored two linear candidates around i
            double q0 = 0.5*q[i] + 0.5*q[i-1];
            double q1 = 1.5*q[i] - 0.5*q[i+1];
            double b0 = (q[i]   - q[i-1])*(q[i]   - q[i-1]);
            double b1 = (q[i+1] - q[i])  *(q[i+1] - q[i]);
            // mirror linear weights: (2/3, 1/3)
            double a0 = (2.0/3.0) / ((eps + b0)*(eps + b0));
            double a1 = (1.0/3.0) / ((eps + b1)*(eps + b1));
            double w0 = a0 / (a0 + a1);
            double w1 = a1 / (a0 + a1);
            qR[i] = w0*q0 + w1*q1;     // fills j = i
        }

        // Endpoints and missing near-boundary interfaces:
        qL[0] = q[0];                 // interface j=0 (left boundary)
        qR[0] = q[0];

        // Fallback 2nd-order one-sided for the first interior interface j=1
        // Left state at j=1 should come from cell 0 (can’t build full WENO3)
        qL[1] = 0.5*(q[0] + q[1]);

        // Fallback for the last interior interface j=n-1 (right state)
        qR[n-1] = 0.5*(q[n-1] + q[n-2]);

        qL[n] = q[n-1];               // interface j=n (right boundary)
        qR[n] = q[n-1];
    };

    // Extract component arrays (rho, u, E) with robust u
    std::vector<double> rho(nx), u(nx), E(nx);
    for (int i = 0; i < nx; ++i) {
        double rh = std::max(U[i].rho, 1e-12); // avoid division by tiny
        rho[i] = rh;
        u[i]   = U[i].rhou / rh;
        E[i]   = U[i].E;
    }

    // Reconstruct each scalar
    std::vector<double> rhoL, rhoR, uL, uR, EL, ER;
    weno3_scalar(rho, rhoL, rhoR);
    weno3_scalar(u,   uL,   uR);
    weno3_scalar(E,   EL,   ER);

    // Pack back into conservative states at interfaces j=0..nx
    for (int j = 0; j <= nx; ++j) {
        UL[j].rho  = rhoL[j];
        UL[j].rhou = rhoL[j] * uL[j];
        UL[j].E    = EL[j];

        UR[j].rho  = rhoR[j];
        UR[j].rhou = rhoR[j] * uR[j];
        UR[j].E    = ER[j];
    }
}


} // namespace ls
