/*
==============================================================================
 File:        recon.hpp
 Purpose:     Reconstruction utilities for 1D/2D Euler solvers (WENO3 + more)
 Author:      Lucas Pierce (extended with MUSCL/WENO5/TENO5)
 Description:
   - Reconstructs left/right states at interfaces
   - 1D: UL[iface], UR[iface]
   - 2D: dimension-by-dimension (x then y)
   - Includes:
       * WENO3
       * MUSCL-TVD (MC limiter)
       * WENO5-JS
       * WENO5-Z
       * TENO5
   - OpenMP-safe: loops write to unique face indices

 IMPORTANT FIX (2026-01):
   WENO5/WENO5Z/TENO5 MUST use *shifted* stencils for right state:
     Left state at i+1/2 uses (i-2,i-1,i,i+1,i+2)
     Right state at i+1/2 uses (i-1,i,i+1,i+2,i+3)
   Your previous code "mirrored" the stencil without i+3,
   causing oscillations / checkerboarding / non-smooth expansion.
==============================================================================
*/

#pragma once
#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <stdexcept>

#ifdef LS_USE_OPENMP
  #include <omp.h>
#endif

#include "ls/types.hpp"
#include "ls/eos.hpp"
#include "ls/mesh.hpp"

namespace ls {

// Small eps for WENO3; and very small eps for WENO5-type smoothness
static constexpr double WENO_EPS  = 1e-6;
static constexpr double WENO5_EPS = 1e-6;

struct WenoDebug1D {
    bool enabled   = false;
    int  iface_IF  = -1;
};

// ---------------------
// MUSCL slope limiters
// ---------------------
inline double minmod(double a, double b) {
    if (a * b <= 0.0) return 0.0;
    return (std::fabs(a) < std::fabs(b)) ? a : b;
}

inline double mc_limiter(double dm, double dp) {
    // Monotonized Central: minmod( 2dm, 0.5(dm+dp), 2dp )
    const double a = 2.0 * dm;
    const double b = 0.5 * (dm + dp);
    const double c = 2.0 * dp;
    return minmod(a, minmod(b, c));
}

// Proper MUSCL interface reconstruction needs (fm1,f0,f1,f2)
// interface is between f0 (left cell) and f1 (right cell)
inline void muscl_scalar(double fm1, double f0, double f1, double f2,
                         double& fL, double& fR)
{
    // slope in left cell (i): uses (i-1,i,i+1) -> (fm1,f0,f1)
    const double dm0 = f0 - fm1;
    const double dp0 = f1 - f0;
    const double s0  = mc_limiter(dm0, dp0);

    // slope in right cell (i+1): uses (i,i+1,i+2) -> (f0,f1,f2)
    const double dm1 = f1 - f0;
    const double dp1 = f2 - f1;
    const double s1  = mc_limiter(dm1, dp1);

    fL = f0 + 0.5 * s0;
    fR = f1 - 0.5 * s1;
}

// ---------------------
// WENO3 scalar recon
// ---------------------
inline void weno3_scalar(
    double fm1, double f0, double f1, double f2,
    double& fL, double& fR,
    const WenoDebug1D* dbg = nullptr,
    int IF = -1)
{
    // Left state at i+1/2
    const double p0_L = 0.5 * f0 + 0.5 * f1;
    const double p1_L = -0.5 * fm1 + 1.5 * f0;

    const double beta0_L = (f0 - f1) * (f0 - f1);
    const double beta1_L = (fm1 - f0) * (fm1 - f0);

    const double d0 = 0.75;
    const double d1 = 0.25;

    const double a0_L = d0 / ((WENO_EPS + beta0_L) * (WENO_EPS + beta0_L));
    const double a1_L = d1 / ((WENO_EPS + beta1_L) * (WENO_EPS + beta1_L));
    const double w_sum_L = a0_L + a1_L;
    const double w0_L = a0_L / w_sum_L;
    const double w1_L = a1_L / w_sum_L;

    fL = w0_L * p0_L + w1_L * p1_L;

    // Right state at i+1/2
    const double p0_R = -0.5 * f2 + 1.5 * f1;
    const double p1_R =  0.5 * f0 + 0.5 * f1;

    const double beta0_R = (f2 - f1) * (f2 - f1);
    const double beta1_R = (f1 - f0) * (f1 - f0);

    const double a0_R = d0 / ((WENO_EPS + beta0_R) * (WENO_EPS + beta0_R));
    const double a1_R = d1 / ((WENO_EPS + beta1_R) * (WENO_EPS + beta1_R));
    const double w_sum_R = a0_R + a1_R;
    const double w0_R = a0_R / w_sum_R;
    const double w1_R = a1_R / w_sum_R;

    fR = w0_R * p0_R + w1_R * p1_R;

    if (dbg && dbg->enabled && (dbg->iface_IF < 0 || dbg->iface_IF == IF)) {
        std::cout << "WENO3 scalar at IF=" << IF << "\n";
        std::cout << "  stencil (fm1,f0,f1,f2) = "
                  << fm1 << ", " << f0 << ", " << f1 << ", " << f2 << "\n";
        std::cout << "  LEFT: fL=" << fL << "  RIGHT: fR=" << fR << "\n\n";
    }
}

// ---------------------
// WENO5-JS scalar (helpers)
// ---------------------
inline double weno5_left_js(double fm2, double fm1, double f0, double f1, double f2)
{
    // Candidate polys at i+1/2 (left state)
    const double p0 = ( 2.0*fm2 - 7.0*fm1 + 11.0*f0) / 6.0;
    const double p1 = (-1.0*fm1 + 5.0*f0  +  2.0*f1) / 6.0;
    const double p2 = ( 2.0*f0  + 5.0*f1  -  1.0*f2) / 6.0;

    // Smoothness indicators (Jiang–Shu)
    const double b0 = (13.0/12.0)*std::pow(fm2 - 2.0*fm1 + f0, 2.0)
                    + ( 1.0/4.0 )*std::pow(fm2 - 4.0*fm1 + 3.0*f0, 2.0);
    const double b1 = (13.0/12.0)*std::pow(fm1 - 2.0*f0 + f1, 2.0)
                    + ( 1.0/4.0 )*std::pow(fm1 - f1, 2.0);
    const double b2 = (13.0/12.0)*std::pow(f0 - 2.0*f1 + f2, 2.0)
                    + ( 1.0/4.0 )*std::pow(3.0*f0 - 4.0*f1 + f2, 2.0);

    const double d0 = 0.1, d1 = 0.6, d2 = 0.3;

    const double a0 = d0 / std::pow(WENO5_EPS + b0, 2.0);
    const double a1 = d1 / std::pow(WENO5_EPS + b1, 2.0);
    const double a2 = d2 / std::pow(WENO5_EPS + b2, 2.0);

    const double as = a0 + a1 + a2;
    const double w0 = a0 / as;
    const double w1 = a1 / as;
    const double w2 = a2 / as;

    return w0*p0 + w1*p1 + w2*p2;
}

// Right state at i+1/2 uses shifted stencil (i-1,i,i+1,i+2,i+3)
inline double weno5_right_js(double fm1, double f0, double f1, double f2, double f3)
{
    // Equivalent to left reconstruction with reversed stencil
    return weno5_left_js(f3, f2, f1, f0, fm1);
}

// Correct WENO5-JS scalar reconstruction using 6 values total (fm2..f3)
inline void weno5js_scalar(double fm2, double fm1, double f0, double f1, double f2, double f3,
                           double& fL, double& fR)
{
    fL = weno5_left_js (fm2, fm1, f0, f1, f2);
    fR = weno5_right_js(fm1, f0,  f1, f2, f3);
}

// ---------------------
// WENO5-Z scalar
// ---------------------
inline double weno5_left_z(double fm2, double fm1, double f0, double f1, double f2, double p)
{
    const double b0 = (13.0/12.0)*std::pow(fm2 - 2.0*fm1 + f0, 2.0)
                    + ( 1.0/4.0 )*std::pow(fm2 - 4.0*fm1 + 3.0*f0, 2.0);
    const double b1 = (13.0/12.0)*std::pow(fm1 - 2.0*f0 + f1, 2.0)
                    + ( 1.0/4.0 )*std::pow(fm1 - f1, 2.0);
    const double b2 = (13.0/12.0)*std::pow(f0 - 2.0*f1 + f2, 2.0)
                    + ( 1.0/4.0 )*std::pow(3.0*f0 - 4.0*f1 + f2, 2.0);

    const double tau5 = std::fabs(b0 - b2);
    const double d0 = 0.1, d1 = 0.6, d2 = 0.3;

    const double a0 = d0 * (1.0 + std::pow(tau5/(WENO5_EPS + b0), p));
    const double a1 = d1 * (1.0 + std::pow(tau5/(WENO5_EPS + b1), p));
    const double a2 = d2 * (1.0 + std::pow(tau5/(WENO5_EPS + b2), p));
    const double as = a0 + a1 + a2;

    const double w0 = a0/as, w1 = a1/as, w2 = a2/as;

    const double p0 = ( 2.0*fm2 - 7.0*fm1 + 11.0*f0) / 6.0;
    const double p1 = (-1.0*fm1 + 5.0*f0  +  2.0*f1) / 6.0;
    const double p2 = ( 2.0*f0  + 5.0*f1  -  1.0*f2) / 6.0;

    return w0*p0 + w1*p1 + w2*p2;
}

inline double weno5_right_z(double fm1, double f0, double f1, double f2, double f3, double p)
{
    return weno5_left_z(f3, f2, f1, f0, fm1, p);
}

inline void weno5z_scalar(double fm2, double fm1, double f0, double f1, double f2, double f3,
                          double& fL, double& fR, double p = 2.0)
{
    fL = weno5_left_z (fm2, fm1, f0, f1, f2, p);
    fR = weno5_right_z(fm1, f0,  f1, f2, f3, p);
}


// ---------------------
// TENO5 scalar (Fu et al. targeted ENO style)
// Reference: Computational Transport Phenomena, Sec. 5.4 (TENO),
// esp. Eqs. (5.111)–(5.113) for scale separation + cutoff.
// ---------------------
inline double teno5_left(
    double fm2, double fm1, double f0, double f1, double f2,
    double CT = 1e-5,   // cutoff threshold (tunable)
    double C  = 1.0,    // book uses C = 1
    double q  = 6.0)    // book uses q = 6
{
    // Candidate polynomials at i+1/2 (same as WENO5 left)
    const double p0 = ( 2.0*fm2 - 7.0*fm1 + 11.0*f0) / 6.0;
    const double p1 = (-1.0*fm1 + 5.0*f0  +  2.0*f1) / 6.0;
    const double p2 = ( 2.0*f0  + 5.0*f1  -  1.0*f2) / 6.0;

    // Jiang–Shu smoothness indicators (same as WENO5)
    const double b0 = (13.0/12.0)*std::pow(fm2 - 2.0*fm1 + f0, 2.0)
                    + ( 1.0/4.0 )*std::pow(fm2 - 4.0*fm1 + 3.0*f0, 2.0);
    const double b1 = (13.0/12.0)*std::pow(fm1 - 2.0*f0 + f1, 2.0)
                    + ( 1.0/4.0 )*std::pow(fm1 - f1, 2.0);
    const double b2 = (13.0/12.0)*std::pow(f0 - 2.0*f1 + f2, 2.0)
                    + ( 1.0/4.0 )*std::pow(3.0*f0 - 4.0*f1 + f2, 2.0);

    // Full-stencil indicator for TENO5:
    // book states for 5th order: tauK = |beta2 - beta1|
    const double tau5 = std::fabs(b2 - b1);

    // Strong scale separation measure (Eq. 5.111 style):
    // gamma_k = ((C + tau/beta_k)^q)
    // Add eps in denominators to avoid blow-ups
    const double eps = WENO5_EPS;
    const double g0 = std::pow(C + tau5 / (b0 + eps), q);
    const double g1 = std::pow(C + tau5 / (b1 + eps), q);
    const double g2 = std::pow(C + tau5 / (b2 + eps), q);

    const double gsum = g0 + g1 + g2;
    // If something pathological happens, fall back to linear 5th order combo
    if (!std::isfinite(gsum) || gsum <= 0.0) {
        const double d0 = 0.1, d1 = 0.6, d2 = 0.3;
        return d0*p0 + d1*p1 + d2*p2;
    }

    // Normalized smoothness measures (Eq. 5.112)
    const double X0 = g0 / gsum;
    const double X1 = g1 / gsum;
    const double X2 = g2 / gsum;

    // Sharp cutoff (Eq. 5.113)
    const double dlt0 = (X0 < CT) ? 0.0 : 1.0;
    const double dlt1 = (X1 < CT) ? 0.0 : 1.0;
    const double dlt2 = (X2 < CT) ? 0.0 : 1.0;

    // Optimal (linear) weights for 5th order (same as WENO5)
    const double d0 = 0.1, d1 = 0.6, d2 = 0.3;

    // Renormalize on the surviving stencils
    const double wsum = dlt0*d0 + dlt1*d1 + dlt2*d2;

    // If all got cut (can happen with a too-large CT), revert to linear weights
    if (wsum <= 0.0) {
        return d0*p0 + d1*p1 + d2*p2;
    }

    const double w0 = (dlt0*d0) / wsum;
    const double w1 = (dlt1*d1) / wsum;
    const double w2 = (dlt2*d2) / wsum;

    return w0*p0 + w1*p1 + w2*p2;
}

// Right state at i+1/2 uses shifted stencil (i-1,i,i+1,i+2,i+3)
inline double teno5_right(
    double fm1, double f0, double f1, double f2, double f3,
    double CT = 1e-5, double C = 1.0, double q = 6.0)
{
    // mirror via reversal, consistent with your WENO5 right implementation
    return teno5_left(f3, f2, f1, f0, fm1, CT, C, q);
}

inline void teno5_scalar(
    double fm2, double fm1, double f0, double f1, double f2, double f3,
    double& fL, double& fR,
    double CT = 1e-5, double C = 1.0, double q = 6.0)
{
    fL = teno5_left (fm2, fm1, f0, f1, f2, CT, C, q);
    fR = teno5_right(fm1, f0,  f1, f2, f3, CT, C, q);
}



// ---------------------
// 1-D WENO3 reconstruction
// ---------------------
inline void weno3_reconstruct_1d(
    const std::vector<State1D>& U,
    std::vector<State1D>& UL,
    std::vector<State1D>& UR,
    const Mesh1D& mesh,
    const WenoDebug1D* dbg = nullptr)
{
    const int I0 = mesh.interior_start();
    const int I1 = mesh.interior_end();
    const int nx_tot = mesh.nx_tot;

    if (mesh.ng < 2) {
        throw std::runtime_error("WENO3 requires mesh.ng >= 2 (two ghost cells per side)");
    }

    #ifdef LS_USE_OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int IF = I0; IF <= I1; ++IF) {
        const int im1 = IF - 2;
        const int i0  = IF - 1;
        const int i1  = IF;
        const int i2  = IF + 1;

        if (im1 < 0 || i2 >= nx_tot) continue;

        double rhoL, rhoR, rhouL, rhouR, EL, ER;

        weno3_scalar(U[im1].rho,  U[i0].rho,  U[i1].rho,  U[i2].rho,  rhoL,  rhoR, dbg, IF);
        weno3_scalar(U[im1].rhou, U[i0].rhou, U[i1].rhou, U[i2].rhou, rhouL, rhouR, dbg, IF);
        weno3_scalar(U[im1].E,    U[i0].E,    U[i1].E,    U[i2].E,    EL,    ER,    dbg, IF);

        const int iface_idx = IF - 1;
        UL[iface_idx] = {rhoL, rhouL, EL};
        UR[iface_idx] = {rhoR, rhouR, ER};
    }
}

// -----------------------------------------------------------------------------
// Interface indexing helpers (2D)
// -----------------------------------------------------------------------------
inline int idx_xface(int I, int J, int nx_tot) { return J*(nx_tot + 1) + I; }
inline int idx_yface(int I, int J, int nx_tot) { return J*nx_tot + I; }

struct WenoDebug2D {
    bool enabled = false;
    int i_face = -1;
    int j_face = -1;
};

// ---------------------
// 2D WENO3 recon (existing)
// ---------------------
inline void weno3_reconstruct_2d_x(
    const std::vector<State2D>& U,
    std::vector<State2D>& ULx,
    std::vector<State2D>& URx,
    const Mesh2D& mesh,
    const WenoDebug2D* dbg = nullptr)
{
    const int nx_tot = mesh.nx_tot;

    const int i0 = mesh.interior_i_start();
    const int i1 = mesh.interior_i_end();
    const int j0 = mesh.interior_j_start();
    const int j1 = mesh.interior_j_end();

    #ifdef LS_USE_OPENMP
    #pragma omp parallel for collapse(2) schedule(static)
    #endif
    for (int J = j0; J <= j1; ++J) {
        for (int I = i0; I <= i1 + 1; ++I) {
            const int idxF = idx_xface(I, J, nx_tot);

            const int im1 = I - 2;
            const int i0c = I - 1;
            const int i1c = I;
            const int i2c = I + 1;

            const int id_im1 = mesh.index(im1, J);
            const int id_i0  = mesh.index(i0c, J);
            const int id_i1  = mesh.index(i1c, J);
            const int id_i2  = mesh.index(i2c, J);

            double rhoL, rhoR, rhouL, rhouR, rhovL, rhovR, EL, ER;

            weno3_scalar(U[id_im1].rho,  U[id_i0].rho,  U[id_i1].rho,  U[id_i2].rho,  rhoL,  rhoR);
            weno3_scalar(U[id_im1].rhou, U[id_i0].rhou, U[id_i1].rhou, U[id_i2].rhou, rhouL, rhouR);
            weno3_scalar(U[id_im1].rhov, U[id_i0].rhov, U[id_i1].rhov, U[id_i2].rhov, rhovL, rhovR);
            weno3_scalar(U[id_im1].E,    U[id_i0].E,    U[id_i1].E,    U[id_i2].E,    EL,    ER);

            ULx[idxF] = {rhoL, rhouL, rhovL, EL};
            URx[idxF] = {rhoR, rhouR, rhovR, ER};

            if (dbg && dbg->enabled &&
                (dbg->i_face < 0 || dbg->i_face == I) &&
                (dbg->j_face < 0 || dbg->j_face == J)) {
                #ifdef LS_USE_OPENMP
                if (omp_in_parallel()) continue;
                #endif
                std::cout << "WENO3 2D-x face (I,J)=(" << I << "," << J << ")\n";
            }
        }
    }
}

inline void weno3_reconstruct_2d_y(
    const std::vector<State2D>& U,
    std::vector<State2D>& ULy,
    std::vector<State2D>& URy,
    const Mesh2D& mesh)
{
    const int nx_tot = mesh.nx_tot;
    const int ny_tot = mesh.ny_tot;

    const int i0 = mesh.interior_i_start();
    const int i1 = mesh.interior_i_end();
    const int j0 = mesh.interior_j_start();
    const int j1 = mesh.interior_j_end();

    if (mesh.ng < 2) {
        throw std::runtime_error("WENO3 (2D y) requires mesh.ng >= 2 (two ghost cells per side)");
    }

    #ifdef LS_USE_OPENMP
    #pragma omp parallel for collapse(2) schedule(static)
    #endif
    for (int J = j0; J <= j1 + 1; ++J) {
        for (int I = i0; I <= i1; ++I) {
            const int idxF = idx_yface(I, J, nx_tot);

            const int jm1 = J - 2;
            const int j0c = J - 1;
            const int j1c = J;
            const int j2c = J + 1;

            if (jm1 < 0 || j2c >= ny_tot) continue;

            const int id_jm1 = mesh.index(I, jm1);
            const int id_j0  = mesh.index(I, j0c);
            const int id_j1  = mesh.index(I, j1c);
            const int id_j2  = mesh.index(I, j2c);

            double rhoL, rhoR, rhouL, rhouR, rhovL, rhovR, EL, ER;

            weno3_scalar(U[id_jm1].rho,  U[id_j0].rho,  U[id_j1].rho,  U[id_j2].rho, rhoL, rhoR);
            weno3_scalar(U[id_jm1].rhou, U[id_j0].rhou, U[id_j1].rhou, U[id_j2].rhou, rhouL, rhouR);
            weno3_scalar(U[id_jm1].rhov, U[id_j0].rhov, U[id_j1].rhov, U[id_j2].rhov, rhovL, rhovR);
            weno3_scalar(U[id_jm1].E,    U[id_j0].E,    U[id_j1].E,    U[id_j2].E,   EL, ER);

            ULy[idxF] = {rhoL, rhouL, rhovL, EL};
            URy[idxF] = {rhoR, rhouR, rhovR, ER};
        }
    }
}




// ---------------------
// 2D generic recon dispatcher (MUSCL/WENO5/WENO5Z/TENO5 + fallback to WENO3)
// ---------------------
// ---------------------
// ---------------------
// 2D generic recon dispatcher (MUSCL/WENO5/WENO5Z/TENO5)
// FULL REVISION: adds two-stage robustness for TENO5
//   (1) clamp reconstructed primitives to local stencil bounds (rho,p always; u,v optional)
//   (2) per-face fallback triggered if clamp occurred OR state invalid
//   (3) if fallback still bad -> revert to first-order (cell-centered primitives)
// ---------------------
inline void reconstruct_2d_x(
    const std::vector<State2D>& U,
    std::vector<State2D>& ULx,
    std::vector<State2D>& URx,
    const Mesh2D& mesh,
    ReconType recon,
    const EOSIdealGas& eos,
    ReconType fallback = ReconType::WENO5Z)
{
    const int nx_tot = mesh.nx_tot;
    const int ny_tot = mesh.ny_tot;

    const int i0 = mesh.interior_i_start();
    const int i1 = mesh.interior_i_end();
    const int j0 = mesh.interior_j_start();
    const int j1 = mesh.interior_j_end();

    // --- small local helpers (kept inside function to match your style) ---
    auto cons_to_prim_safe = [&](const State2D& Ui) -> std::array<double,4> {
        // returns {rho,u,v,p}
        constexpr double rho_min = 1e-8;
        constexpr double e_min   = 1e-10;
        constexpr double p_min   = 1e-8;

        double rho  = std::isfinite(Ui.rho)  ? Ui.rho  : rho_min;
        double rhou = std::isfinite(Ui.rhou) ? Ui.rhou : 0.0;
        double rhov = std::isfinite(Ui.rhov) ? Ui.rhov : 0.0;
        double E    = std::isfinite(Ui.E)    ? Ui.E    : e_min;

        rho = std::max(rho, rho_min);
        const double u = rhou / rho;
        const double v = rhov / rho;

        const double kinetic = 0.5 * rho * (u*u + v*v);
        double eint = E - kinetic;
        if (!std::isfinite(eint) || eint < e_min) eint = e_min;

        double p = (eos.gamma - 1.0) * eint;
        if (!std::isfinite(p) || p < p_min) p = p_min;

        return {rho, u, v, p};
    };

    auto prim_to_cons_safe = [&](double rho, double u, double v, double p) -> State2D {
        constexpr double rho_min = 1e-8;
        constexpr double p_min   = 1e-8;

        rho = (std::isfinite(rho) ? rho : rho_min);
        p   = (std::isfinite(p)   ? p   : p_min);
        rho = std::max(rho, rho_min);
        p   = std::max(p,   p_min);

        u = std::isfinite(u) ? u : 0.0;
        v = std::isfinite(v) ? v : 0.0;

        const double E = p/(eos.gamma - 1.0) + 0.5*rho*(u*u + v*v);
        return State2D{rho, rho*u, rho*v, E};
    };

    auto state_ok = [&](const State2D& S) -> bool {
        if (!std::isfinite(S.rho) || !std::isfinite(S.rhou) || !std::isfinite(S.rhov) || !std::isfinite(S.E))
            return false;
        if (S.rho <= 0.0) return false;

        const double u = S.rhou / S.rho;
        const double v = S.rhov / S.rho;
        const double kinetic = 0.5 * S.rho * (u*u + v*v);
        const double eint = S.E - kinetic;
        if (!std::isfinite(eint) || eint <= 0.0) return false;

        const double p = (eos.gamma - 1.0) * eint;
        if (!std::isfinite(p) || p <= 0.0) return false;

        return true;
    };

    auto recon_scalar = [&](ReconType rt,
                            double fm2,double fm1,double f0,double f1,double f2,double f3,
                            double& fL,double& fR)
    {
        switch (rt) {
            case ReconType::MUSCL:
                muscl_scalar(fm1, f0, f1, f2, fL, fR);
                break;
            case ReconType::WENO5:
                weno5js_scalar(fm2,fm1,f0,f1,f2,f3, fL,fR);
                break;
            case ReconType::WENO5Z:
                weno5z_scalar (fm2,fm1,f0,f1,f2,f3, fL,fR);
                break;
            case ReconType::TENO5:
                teno5_scalar  (fm2,fm1,f0,f1,f2,f3, fL,fR);
                break;
            default:
                weno3_scalar(fm1,f0,f1,f2, fL,fR);
                break;
        }
    };

    // ---- clamp helpers ----
    auto clamp = [&](double x, double lo, double hi) {
        return std::max(lo, std::min(x, hi));
    };

    auto stencil_minmax = [&](int k,
                              const std::array<double,4>& Wm2,
                              const std::array<double,4>& Wm1,
                              const std::array<double,4>& W0,
                              const std::array<double,4>& W1,
                              const std::array<double,4>& W2,
                              const std::array<double,4>& W3) -> std::pair<double,double>
    {
        double mn = std::min({Wm2[k], Wm1[k], W0[k], W1[k], W2[k], W3[k]});
        double mx = std::max({Wm2[k], Wm1[k], W0[k], W1[k], W2[k], W3[k]});
        return {mn, mx};
    };

    auto clamp_pair = [&](double& a, double& b, double lo, double hi, bool& did_clamp) {
        const double aa = clamp(a, lo, hi);
        const double bb = clamp(b, lo, hi);
        did_clamp = did_clamp || (aa != a) || (bb != b);
        a = aa; b = bb;
    };

    #ifdef LS_USE_OPENMP
    #pragma omp parallel for collapse(2) schedule(static)
    #endif
    for (int J = j0; J <= j1; ++J) {
        for (int I = i0; I <= i1 + 1; ++I) {
            const int idxF = idx_xface(I, J, nx_tot);

            // Face between cells (I-1) and (I)
            const int im2 = I - 3;
            const int im1 = I - 2;
            const int i0c = I - 1;
            const int i1c = I;
            const int i2c = I + 1;
            const int i3c = I + 2;

            if (im2 < 0 || i3c >= nx_tot || J < 0 || J >= ny_tot) continue;

            const int id_im2 = mesh.index(im2, J);
            const int id_im1 = mesh.index(im1, J);
            const int id_i0  = mesh.index(i0c, J);
            const int id_i1  = mesh.index(i1c, J);
            const int id_i2  = mesh.index(i2c, J);
            const int id_i3  = mesh.index(i3c, J);

            // Convert stencil conservative->primitive (rho,u,v,p)
            const auto Wm2 = cons_to_prim_safe(U[id_im2]);
            const auto Wm1 = cons_to_prim_safe(U[id_im1]);
            const auto W0  = cons_to_prim_safe(U[id_i0 ]);
            const auto W1  = cons_to_prim_safe(U[id_i1 ]);
            const auto W2  = cons_to_prim_safe(U[id_i2 ]);
            const auto W3  = cons_to_prim_safe(U[id_i3 ]);

            // Reconstruct primitive components
            double rhoL,rhoR, uL,uR, vL,vR, pL,pR;

            recon_scalar(recon, Wm2[0],Wm1[0],W0[0],W1[0],W2[0],W3[0], rhoL,rhoR);
            recon_scalar(recon, Wm2[1],Wm1[1],W0[1],W1[1],W2[1],W3[1], uL,uR);
            recon_scalar(recon, Wm2[2],Wm1[2],W0[2],W1[2],W2[2],W3[2], vL,vR);
            recon_scalar(recon, Wm2[3],Wm1[3],W0[3],W1[3],W2[3],W3[3], pL,pR);

            // ---- Clamp to stencil bounds (prevents positive-but-insane overshoots) ----
            auto [rho_mn, rho_mx] = stencil_minmax(0, Wm2,Wm1,W0,W1,W2,W3);
            auto [u_mn,   u_mx  ] = stencil_minmax(1, Wm2,Wm1,W0,W1,W2,W3);
            auto [v_mn,   v_mx  ] = stencil_minmax(2, Wm2,Wm1,W0,W1,W2,W3);
            auto [p_mn,   p_mx  ] = stencil_minmax(3, Wm2,Wm1,W0,W1,W2,W3);

            // Slightly expand bounds to avoid over-clamping smooth regions
            const double tiny = 1e-12;
            rho_mn = std::max(rho_mn - 1e-6*std::fabs(rho_mn) - tiny, 1e-8);
            rho_mx = rho_mx + 1e-6*std::fabs(rho_mx) + tiny;

            p_mn   = std::max(p_mn   - 1e-6*std::fabs(p_mn)   - tiny, 1e-8);
            p_mx   = p_mx   + 1e-6*std::fabs(p_mx)   + tiny;

            bool clamped = false;

            // Always clamp rho and p
            clamp_pair(rhoL, rhoR, rho_mn, rho_mx, clamped);
            clamp_pair(pL,   pR,   p_mn,   p_mx,   clamped);

            // Optional: clamp velocities too (turn on if still unstable)
            // clamp_pair(uL, uR, u_mn, u_mx, clamped);
            // clamp_pair(vL, vR, v_mn, v_mx, clamped);

            State2D UL = prim_to_cons_safe(rhoL,uL,vL,pL);
            State2D UR = prim_to_cons_safe(rhoR,uR,vR,pR);

            // Per-face fallback for TENO if clamp happened OR invalid state
            if (recon == ReconType::TENO5 && (clamped || !state_ok(UL) || !state_ok(UR))) {
                recon_scalar(fallback, Wm2[0],Wm1[0],W0[0],W1[0],W2[0],W3[0], rhoL,rhoR);
                recon_scalar(fallback, Wm2[1],Wm1[1],W0[1],W1[1],W2[1],W3[1], uL,uR);
                recon_scalar(fallback, Wm2[2],Wm1[2],W0[2],W1[2],W2[2],W3[2], vL,vR);
                recon_scalar(fallback, Wm2[3],Wm1[3],W0[3],W1[3],W2[3],W3[3], pL,pR);

                // Clamp fallback too
                clamped = false;
                clamp_pair(rhoL, rhoR, rho_mn, rho_mx, clamped);
                clamp_pair(pL,   pR,   p_mn,   p_mx,   clamped);
                // clamp_pair(uL, uR, u_mn, u_mx, clamped);
                // clamp_pair(vL, vR, v_mn, v_mx, clamped);

                UL = prim_to_cons_safe(rhoL,uL,vL,pL);
                UR = prim_to_cons_safe(rhoR,uR,vR,pR);

                // If still bad, revert to 1st order in x
                if (!state_ok(UL) || !state_ok(UR)) {
                    UL = prim_to_cons_safe(W0[0], W0[1], W0[2], W0[3]); // cell (I-1)
                    UR = prim_to_cons_safe(W1[0], W1[1], W1[2], W1[3]); // cell (I)
                }
            }

            ULx[idxF] = UL;
            URx[idxF] = UR;
        }
    }
}


inline void reconstruct_2d_y(
    const std::vector<State2D>& U,
    std::vector<State2D>& ULy,
    std::vector<State2D>& URy,
    const Mesh2D& mesh,
    ReconType recon,
    const EOSIdealGas& eos,
    ReconType fallback = ReconType::WENO5Z)
{
    const int nx_tot = mesh.nx_tot;
    const int ny_tot = mesh.ny_tot;

    const int i0 = mesh.interior_i_start();
    const int i1 = mesh.interior_i_end();
    const int j0 = mesh.interior_j_start();
    const int j1 = mesh.interior_j_end();

    auto cons_to_prim_safe = [&](const State2D& Ui) -> std::array<double,4> {
        // returns {rho,u,v,p}
        constexpr double rho_min = 1e-8;
        constexpr double e_min   = 1e-10;
        constexpr double p_min   = 1e-8;

        double rho  = std::isfinite(Ui.rho)  ? Ui.rho  : rho_min;
        double rhou = std::isfinite(Ui.rhou) ? Ui.rhou : 0.0;
        double rhov = std::isfinite(Ui.rhov) ? Ui.rhov : 0.0;
        double E    = std::isfinite(Ui.E)    ? Ui.E    : e_min;

        rho = std::max(rho, rho_min);
        const double u = rhou / rho;
        const double v = rhov / rho;

        const double kinetic = 0.5 * rho * (u*u + v*v);
        double eint = E - kinetic;
        if (!std::isfinite(eint) || eint < e_min) eint = e_min;

        double p = (eos.gamma - 1.0) * eint;
        if (!std::isfinite(p) || p < p_min) p = p_min;

        return {rho, u, v, p};
    };

    auto prim_to_cons_safe = [&](double rho, double u, double v, double p) -> State2D {
        constexpr double rho_min = 1e-8;
        constexpr double p_min   = 1e-8;

        rho = (std::isfinite(rho) ? rho : rho_min);
        p   = (std::isfinite(p)   ? p   : p_min);
        rho = std::max(rho, rho_min);
        p   = std::max(p,   p_min);

        u = std::isfinite(u) ? u : 0.0;
        v = std::isfinite(v) ? v : 0.0;

        const double E = p/(eos.gamma - 1.0) + 0.5*rho*(u*u + v*v);
        return State2D{rho, rho*u, rho*v, E};
    };

    auto state_ok = [&](const State2D& S) -> bool {
        if (!std::isfinite(S.rho) || !std::isfinite(S.rhou) || !std::isfinite(S.rhov) || !std::isfinite(S.E))
            return false;
        if (S.rho <= 0.0) return false;

        const double u = S.rhou / S.rho;
        const double v = S.rhov / S.rho;
        const double kinetic = 0.5 * S.rho * (u*u + v*v);
        const double eint = S.E - kinetic;
        if (!std::isfinite(eint) || eint <= 0.0) return false;

        const double p = (eos.gamma - 1.0) * eint;
        if (!std::isfinite(p) || p <= 0.0) return false;

        return true;
    };

    auto recon_scalar = [&](ReconType rt,
                            double fm2,double fm1,double f0,double f1,double f2,double f3,
                            double& fL,double& fR)
    {
        switch (rt) {
            case ReconType::MUSCL:
                muscl_scalar(fm1, f0, f1, f2, fL, fR);
                break;
            case ReconType::WENO5:
                weno5js_scalar(fm2,fm1,f0,f1,f2,f3, fL,fR);
                break;
            case ReconType::WENO5Z:
                weno5z_scalar (fm2,fm1,f0,f1,f2,f3, fL,fR);
                break;
            case ReconType::TENO5:
                teno5_scalar  (fm2,fm1,f0,f1,f2,f3, fL,fR);
                break;
            default:
                weno3_scalar(fm1,f0,f1,f2, fL,fR);
                break;
        }
    };

    // ---- clamp helpers ----
    auto clamp = [&](double x, double lo, double hi) {
        return std::max(lo, std::min(x, hi));
    };

    auto stencil_minmax = [&](int k,
                              const std::array<double,4>& Wm2,
                              const std::array<double,4>& Wm1,
                              const std::array<double,4>& W0,
                              const std::array<double,4>& W1,
                              const std::array<double,4>& W2,
                              const std::array<double,4>& W3) -> std::pair<double,double>
    {
        double mn = std::min({Wm2[k], Wm1[k], W0[k], W1[k], W2[k], W3[k]});
        double mx = std::max({Wm2[k], Wm1[k], W0[k], W1[k], W2[k], W3[k]});
        return {mn, mx};
    };

    auto clamp_pair = [&](double& a, double& b, double lo, double hi, bool& did_clamp) {
        const double aa = clamp(a, lo, hi);
        const double bb = clamp(b, lo, hi);
        did_clamp = did_clamp || (aa != a) || (bb != b);
        a = aa; b = bb;
    };

    #ifdef LS_USE_OPENMP
    #pragma omp parallel for collapse(2) schedule(static)
    #endif
    for (int J = j0; J <= j1 + 1; ++J) {
        for (int I = i0; I <= i1; ++I) {
            const int idxF = idx_yface(I, J, nx_tot);

            // Face between cells (I, J-1) and (I, J)
            const int jm2 = J - 3;
            const int jm1 = J - 2;
            const int j0c = J - 1;
            const int j1c = J;
            const int j2c = J + 1;
            const int j3c = J + 2;

            if (jm2 < 0 || j3c >= ny_tot || I < 0 || I >= nx_tot) continue;

            const int id_jm2 = mesh.index(I, jm2);
            const int id_jm1 = mesh.index(I, jm1);
            const int id_j0  = mesh.index(I, j0c);
            const int id_j1  = mesh.index(I, j1c);
            const int id_j2  = mesh.index(I, j2c);
            const int id_j3  = mesh.index(I, j3c);

            const auto Wm2 = cons_to_prim_safe(U[id_jm2]);
            const auto Wm1 = cons_to_prim_safe(U[id_jm1]);
            const auto W0  = cons_to_prim_safe(U[id_j0 ]);
            const auto W1  = cons_to_prim_safe(U[id_j1 ]);
            const auto W2  = cons_to_prim_safe(U[id_j2 ]);
            const auto W3  = cons_to_prim_safe(U[id_j3 ]);

            double rhoL,rhoR, uL,uR, vL,vR, pL,pR;

            recon_scalar(recon, Wm2[0],Wm1[0],W0[0],W1[0],W2[0],W3[0], rhoL,rhoR);
            recon_scalar(recon, Wm2[1],Wm1[1],W0[1],W1[1],W2[1],W3[1], uL,uR);
            recon_scalar(recon, Wm2[2],Wm1[2],W0[2],W1[2],W2[2],W3[2], vL,vR);
            recon_scalar(recon, Wm2[3],Wm1[3],W0[3],W1[3],W2[3],W3[3], pL,pR);

            // ---- Clamp to stencil bounds ----
            auto [rho_mn, rho_mx] = stencil_minmax(0, Wm2,Wm1,W0,W1,W2,W3);
            auto [u_mn,   u_mx  ] = stencil_minmax(1, Wm2,Wm1,W0,W1,W2,W3);
            auto [v_mn,   v_mx  ] = stencil_minmax(2, Wm2,Wm1,W0,W1,W2,W3);
            auto [p_mn,   p_mx  ] = stencil_minmax(3, Wm2,Wm1,W0,W1,W2,W3);

            const double tiny = 1e-12;
            rho_mn = std::max(rho_mn - 1e-6*std::fabs(rho_mn) - tiny, 1e-8);
            rho_mx = rho_mx + 1e-6*std::fabs(rho_mx) + tiny;

            p_mn   = std::max(p_mn   - 1e-6*std::fabs(p_mn)   - tiny, 1e-8);
            p_mx   = p_mx   + 1e-6*std::fabs(p_mx)   + tiny;

            bool clamped = false;
            clamp_pair(rhoL, rhoR, rho_mn, rho_mx, clamped);
            clamp_pair(pL,   pR,   p_mn,   p_mx,   clamped);
            // clamp_pair(uL, uR, u_mn, u_mx, clamped);
            // clamp_pair(vL, vR, v_mn, v_mx, clamped);

            State2D UL = prim_to_cons_safe(rhoL,uL,vL,pL);
            State2D UR = prim_to_cons_safe(rhoR,uR,vR,pR);

            if (recon == ReconType::TENO5 && (clamped || !state_ok(UL) || !state_ok(UR))) {
                recon_scalar(fallback, Wm2[0],Wm1[0],W0[0],W1[0],W2[0],W3[0], rhoL,rhoR);
                recon_scalar(fallback, Wm2[1],Wm1[1],W0[1],W1[1],W2[1],W3[1], uL,uR);
                recon_scalar(fallback, Wm2[2],Wm1[2],W0[2],W1[2],W2[2],W3[2], vL,vR);
                recon_scalar(fallback, Wm2[3],Wm1[3],W0[3],W1[3],W2[3],W3[3], pL,pR);

                clamped = false;
                clamp_pair(rhoL, rhoR, rho_mn, rho_mx, clamped);
                clamp_pair(pL,   pR,   p_mn,   p_mx,   clamped);
                // clamp_pair(uL, uR, u_mn, u_mx, clamped);
                // clamp_pair(vL, vR, v_mn, v_mx, clamped);

                UL = prim_to_cons_safe(rhoL,uL,vL,pL);
                UR = prim_to_cons_safe(rhoR,uR,vR,pR);

                if (!state_ok(UL) || !state_ok(UR)) {
                    UL = prim_to_cons_safe(W0[0], W0[1], W0[2], W0[3]); // cell (I-1)
                    UR = prim_to_cons_safe(W1[0], W1[1], W1[2], W1[3]); // cell (I)
                }
            }

            ULy[idxF] = UL;
            URy[idxF] = UR;
        }
    }
}


} // namespace ls
