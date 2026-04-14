/*
==============================================================================
 File:        eos.hpp
 Purpose:     Equation of state utilities for Euler solvers (1D + 2D)
 Author:      Lucas Pierce
 Description:
   - Ideal gas EOS with ratio of specific heats gamma
   - Conservative <-> primitive conversions for:
       * 1D: State1D  (rho, rhou, E)  <-> (rho, u, p)
       * 2D: State2D  (rho, rhou, rhov, E) <-> (rho, u, v, p)
   - Pressure and sound speed helpers that work on States directly
==============================================================================
*/


#pragma once
#include <cmath>
#include "ls/types.hpp"


namespace ls{

  
// -----------------------------------------------------------------------------
// Primitive variable structs (handy for debugging / clarity)
// -----------------------------------------------------------------------------
struct Prim1D {
    double rho;  // density
    double u;    // velocity in x
    double p;    // pressure
};

struct Prim2D {
    double rho;  // density
    double u;    // velocity in x
    double v;    // velocity in y
    double p;    // pressure
};  


// -----------------------------------------------------------------------------
// Ideal-gas EOS
// -----------------------------------------------------------------------------
struct EOSIdealGas {
    double gamma{1.4};

    // -----------------------------
    // 1D: conservative -> primitive
    // -----------------------------
    inline Prim1D cons_to_prim(const State1D& U) const {
        const double rho_min = 1e-8;
        const double e_min   = 1e-10;

        Prim1D W;
        W.rho = U.rho;
        if (W.rho < rho_min) W.rho = rho_min;

        W.u   = U.rhou / W.rho;

        const double kinetic = 0.5 * W.rho * W.u * W.u;
        double e_int   = U.E - kinetic;
        if (e_int < e_min) e_int = e_min;

        W.p = (gamma - 1.0) * e_int;
        return W;
    }

    inline void cons_to_prim(const State1D& U,
                             double& rho, double& u, double& p) const
    {
        Prim1D W = cons_to_prim(U);
        rho = W.rho;
        u   = W.u;
        p   = W.p;
    }

    // 1D: primitive -> conservative
    inline State1D prim_to_cons(const Prim1D& W) const {
        const double kinetic = 0.5 * W.rho * W.u * W.u;
        const double e_int   = W.p / (gamma - 1.0);
        State1D U;
        U.rho  = W.rho;
        U.rhou = W.rho * W.u;
        U.E    = e_int + kinetic;
        return U;
    }

    inline State1D prim_to_cons(double rho, double u, double p) const {
        Prim1D W{rho, u, p};
        return prim_to_cons(W);
    }

    // -----------------------------
    // 2D: conservative -> primitive
    // -----------------------------
    inline Prim2D cons_to_prim(const State2D& U) const {
        const double rho_min = 1e-8;
        const double e_min   = 1e-10;

        Prim2D W;
        W.rho = U.rho;
        if (W.rho < rho_min) W.rho = rho_min;

        W.u   = U.rhou / W.rho;
        W.v   = U.rhov / W.rho;

        const double kinetic = 0.5 * W.rho * (W.u*W.u + W.v*W.v);
        double e_int   = U.E - kinetic;
        if (e_int < e_min) e_int = e_min;

        W.p = (gamma - 1.0) * e_int;
        return W;
    }

    inline void cons_to_prim(const State2D& U,
                             double& rho, double& u, double& v, double& p) const
    {
        Prim2D W = cons_to_prim(U);
        rho = W.rho;
        u   = W.u;
        v   = W.v;
        p   = W.p;
    }

    // 2D: primitive -> conservative
    inline State2D prim_to_cons(const Prim2D& W) const {
        const double kinetic = 0.5 * W.rho * (W.u*W.u + W.v*W.v);
        const double e_int   = W.p / (gamma - 1.0);
        State2D U;
        U.rho  = W.rho;
        U.rhou = W.rho * W.u;
        U.rhov = W.rho * W.v;
        U.E    = e_int + kinetic;
        return U;
    }

    inline State2D prim_to_cons(double rho, double u, double v, double p) const {
        Prim2D W{rho, u, v, p};
        return prim_to_cons(W);
    }

    // -----------------------------
    // Pressure & sound speed
    // -----------------------------
    inline double pressure(const State1D& U) const {
        Prim1D W = cons_to_prim(U);
        return W.p;
    }

    inline double pressure(const State2D& U) const {
        Prim2D W = cons_to_prim(U);
        return W.p;
    }

    inline double sound_speed(const State1D& U) const {
        Prim1D W = cons_to_prim(U);
        return std::sqrt(gamma * W.p / W.rho);
    }

    inline double sound_speed(const State2D& U) const {
        Prim2D W = cons_to_prim(U);
        return std::sqrt(gamma * W.p / W.rho);
    }
};

}