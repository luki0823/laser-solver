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
  ReflectLeftCopy     // reflective wall at left, copy at right
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

} // namespace ls
