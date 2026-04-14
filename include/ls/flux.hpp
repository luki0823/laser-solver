/*
==============================================================================
 File:        flux.hpp
 Purpose:     Euler flux functions and Rusanov (LLF) numerical fluxes
 Author:      Lucas Pierce
 Description:
   - 1D: physical flux F(U) in x and Rusanov flux at interfaces
   - 2D: physical fluxes F(U) in x, G(U) in y, and Rusanov fluxes
==============================================================================
*/
#pragma once

#include <algorithm>
#include <cmath>

#include "ls/types.hpp"
#include "ls/eos.hpp"

namespace ls {

// -----------------------------------------------------------------------------
// 1D: physical flux in x-direction
//
// U = [rho, rho*u, E]
// F = [rho*u, rho*u^2 + p, u*(E + p)]
// -----------------------------------------------------------------------------
inline State1D flux_x(const State1D& U, const EOSIdealGas& eos) {
    Prim1D W = eos.cons_to_prim(U);  // (rho, u, p)
    const double rho = W.rho;
    const double u   = W.u;
    const double p   = W.p;

    State1D F;
    F.rho  = rho * u;
    F.rhou = rho * u * u + p;
    F.E    = u * (U.E + p);
    return F;
}

// -----------------------------------------------------------------------------
// 1D: Rusanov (local Lax–Friedrichs) numerical flux in x
//
// F_num = 0.5 [F(UL) + F(UR)] - 0.5 s_max (UR - UL)
// s_max = max(|u_L| + a_L, |u_R| + a_R)
// -----------------------------------------------------------------------------
inline State1D rusanov_flux_x(const State1D& UL,
                              const State1D& UR,
                              const EOSIdealGas& eos)
{
    const State1D FL = flux_x(UL, eos);
    const State1D FR = flux_x(UR, eos);

    const double aL = eos.sound_speed(UL);
    const double aR = eos.sound_speed(UR);

    double rhoL, uL, pL;
    eos.cons_to_prim(UL, rhoL, uL, pL);

    double rhoR, uR, pR;
    eos.cons_to_prim(UR, rhoR, uR, pR);

    const double smax = std::max(std::fabs(uL) + aL,
                                 std::fabs(uR) + aR);

    return 0.5 * (FL + FR) - 0.5 * smax * (UR - UL);
}

// -----------------------------------------------------------------------------
// 2D: physical fluxes in x and y
//
// State2D U = [rho, rho*u, rho*v, E]
//
// F(U) (x-direction) = [
//   rho*u,
//   rho*u^2 + p,
//   rho*u*v,
//   u*(E + p)
// ]
//
// G(U) (y-direction) = [
//   rho*v,
//   rho*u*v,
//   rho*v^2 + p,
//   v*(E + p)
// ]
// -----------------------------------------------------------------------------
inline State2D flux_x(const State2D& U, const EOSIdealGas& eos) {
    Prim2D W = eos.cons_to_prim(U);  // (rho, u, v, p)
    const double rho = W.rho;
    const double u   = W.u;
    const double v   = W.v;
    const double p   = W.p;

    State2D F;
    F.rho  = rho * u;
    F.rhou = rho * u * u + p;
    F.rhov = rho * u * v;
    F.E    = u * (U.E + p);
    return F;
}

inline State2D flux_y(const State2D& U, const EOSIdealGas& eos) {
    Prim2D W = eos.cons_to_prim(U);  // (rho, u, v, p)
    const double rho = W.rho;
    const double u   = W.u;
    const double v   = W.v;
    const double p   = W.p;

    State2D G;
    G.rho  = rho * v;
    G.rhou = rho * u * v;
    G.rhov = rho * v * v + p;
    G.E    = v * (U.E + p);
    return G;
}

// -----------------------------------------------------------------------------
// 2D: Rusanov flux in x-direction
//
// F_num_x = 0.5 [F(UL) + F(UR)] - 0.5 s_max (UR - UL)
// s_max   = max(|u_L| + a_L, |u_R| + a_R)  (normal speed in x)
// -----------------------------------------------------------------------------
inline State2D rusanov_flux_x(const State2D& UL,
                              const State2D& UR,
                              const EOSIdealGas& eos)
{
    const State2D FL = flux_x(UL, eos);
    const State2D FR = flux_x(UR, eos);

    const double aL = eos.sound_speed(UL);
    const double aR = eos.sound_speed(UR);

    double rhoL, uL, vL, pL;
    eos.cons_to_prim(UL, rhoL, uL, vL, pL);

    double rhoR, uR, vR, pR;
    eos.cons_to_prim(UR, rhoR, uR, vR, pR);

    const double smax = std::max(std::fabs(uL) + aL,
                                 std::fabs(uR) + aR);

    return 0.5 * (FL + FR) - 0.5 * smax * (UR - UL);
}

// -----------------------------------------------------------------------------
// 2D: Rusanov flux in y-direction
//
// F_num_y = 0.5 [G(UL) + G(UR)] - 0.5 s_max (UR - UL)
// s_max   = max(|v_L| + a_L, |v_R| + a_R)  (normal speed in y)
// -----------------------------------------------------------------------------
inline State2D rusanov_flux_y(const State2D& UL,
                              const State2D& UR,
                              const EOSIdealGas& eos)
{
    const State2D GL = flux_y(UL, eos);
    const State2D GR = flux_y(UR, eos);

    const double aL = eos.sound_speed(UL);
    const double aR = eos.sound_speed(UR);

    double rhoL, uL, vL, pL;
    eos.cons_to_prim(UL, rhoL, uL, vL, pL);

    double rhoR, uR, vR, pR;
    eos.cons_to_prim(UR, rhoR, uR, vR, pR);

    const double smax = std::max(std::fabs(vL) + aL,
                                 std::fabs(vR) + aR);

    return 0.5 * (GL + GR) - 0.5 * smax * (UR - UL);
}


}