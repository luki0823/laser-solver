#pragma once

#include <vector>
#include <string>
#include <algorithm>
#include <stdexcept>
#include <cmath>

#ifdef LS_USE_OPENMP
  #include <omp.h>
#endif

#include "ls/types.hpp"
#include "ls/mesh.hpp"
#include "ls/eos.hpp"
#include "ls/flux.hpp"
#include "ls/recon.hpp"
#include "ls/bc.hpp"

namespace ls {


inline Prim2D cons_to_prim_safe(const State2D& U, const EOSIdealGas& eos)
{
    constexpr double rho_min = 1e-8;
    constexpr double e_min   = 1e-10;

    Prim2D W{};
    double rho = std::isfinite(U.rho) ? U.rho : rho_min;
    rho = std::max(rho, rho_min);

    double u = (std::isfinite(U.rhou) ? U.rhou : 0.0) / rho;
    double v = (std::isfinite(U.rhov) ? U.rhov : 0.0) / rho;

    double E = std::isfinite(U.E) ? U.E : e_min;
    double kinetic = 0.5 * rho * (u*u + v*v);
    double eint = E - kinetic;
    if (!std::isfinite(eint) || eint < e_min) eint = e_min;

    double p = (eos.gamma - 1.0) * eint;
    if (!std::isfinite(p) || p < 1e-8) p = 1e-8; // face p floor should NOT be 6.6e-11

    W.rho = rho;
    W.u = u;
    W.v = v;
    W.p = p;
    return W;
}

inline State2D prim_to_cons_safe(const Prim2D& W, const EOSIdealGas& eos)
{
    constexpr double rho_min = 1e-8;
    constexpr double p_min   = 1e-8;

    double rho = std::max(W.rho, rho_min);
    double p   = std::max(W.p,   p_min);

    double u = std::isfinite(W.u) ? W.u : 0.0;
    double v = std::isfinite(W.v) ? W.v : 0.0;

    double E = p/(eos.gamma - 1.0) + 0.5*rho*(u*u + v*v);

    return State2D{rho, rho*u, rho*v, E};
}

inline bool state_ok(const State2D& U, const EOSIdealGas& eos)
{
    if (!std::isfinite(U.rho) || !std::isfinite(U.rhou) || !std::isfinite(U.rhov) || !std::isfinite(U.E))
        return false;
    if (U.rho <= 0.0) return false;

    const double u = U.rhou / U.rho;
    const double v = U.rhov / U.rho;
    const double kinetic = 0.5 * U.rho * (u*u + v*v);
    const double eint = U.E - kinetic;
    if (!std::isfinite(eint) || eint <= 0.0) return false;

    const double p = (eos.gamma - 1.0) * eint;
    if (!std::isfinite(p) || p <= 0.0) return false;

    return true;
}

// ============================================================================
// 1D RHS — supports First-Order and WENO3
// ============================================================================
inline void compute_rhs_1d(
    const std::vector<State1D>& U,
    std::vector<State1D>& rhs,
    const Mesh1D& mesh,
    const EOSIdealGas& eos,
    ReconType recon_type)
{
    const int I0 = mesh.interior_start();
    const int I1 = mesh.interior_end();
    const int nx_tot = mesh.nx_tot;

    rhs.assign(nx_tot, zero_state1d());

    // -------------------------------
    // FIRST ORDER
    // -------------------------------
    if (recon_type == ReconType::FirstOrder)
    {
        std::vector<State1D> F(nx_tot - 1, zero_state1d());

        #ifdef LS_USE_OPENMP
        #pragma omp parallel for schedule(static)
        #endif
        for (int IF = I0; IF <= I1 + 1; ++IF) {
            const State1D& UL = U[IF - 1];
            const State1D& UR = U[IF];
            F[IF - 1] = rusanov_flux_x(UL, UR, eos);
        }

        const double inv_dx = 1.0 / mesh.dx;

        #ifdef LS_USE_OPENMP
        #pragma omp parallel for schedule(static)
        #endif
        for (int I = I0; I <= I1; ++I) {
            rhs[I] = -(F[I] - F[I - 1]) * inv_dx;
        }

        return;
    }

    // -------------------------------
    // WENO3 RECONSTRUCTION
    // -------------------------------
    if (recon_type == ReconType::WENO3)
    {
        if (mesh.ng < 2) {
            throw std::runtime_error("WENO3 requires mesh.ng >= 2 (two ghost cells per side)");
        }

        std::vector<State1D> UL(nx_tot - 1, zero_state1d());
        std::vector<State1D> UR(nx_tot - 1, zero_state1d());

        weno3_reconstruct_1d(U, UL, UR, mesh);

        // Fill the rightmost boundary face (IF = I1+1) with 1st-order copy
        {
            const int IF = I1 + 1;
            const int iface_idx = IF - 1;

            const int left_cell  = IF - 1;
            const int right_cell = IF;

            UL[iface_idx] = U[left_cell];
            UR[iface_idx] = U[right_cell];
        }

        std::vector<State1D> F(nx_tot - 1, zero_state1d());

        #ifdef LS_USE_OPENMP
        #pragma omp parallel for schedule(static)
        #endif
        for (int IF = I0; IF <= I1 + 1; ++IF) {
            const State1D& UL_i = UL[IF - 1];
            const State1D& UR_i = UR[IF - 1];
            F[IF - 1] = rusanov_flux_x(UL_i, UR_i, eos);
        }

        const double inv_dx = 1.0 / mesh.dx;

        #ifdef LS_USE_OPENMP
        #pragma omp parallel for schedule(static)
        #endif
        for (int I = I0; I <= I1; ++I) {
            rhs[I] = -(F[I] - F[I - 1]) * inv_dx;
        }

        return;
    }
}


inline void sanitize_face_state(State2D& U)
{
    static long long fallback_count = 0;
    static long long badface_count = 0;

    constexpr double rho_min = 1e-8;
    constexpr double e_min   = 1e-10;
    constexpr double u_max   = 2e20;

    if (!std::isfinite(U.rho))  U.rho  = rho_min;
    if (!std::isfinite(U.rhou)) U.rhou = 0.0;
    if (!std::isfinite(U.rhov)) U.rhov = 0.0;
    if (!std::isfinite(U.E))    U.E    = e_min;

    // If density is below floor, preserve velocity (scale momentum)
    if (U.rho < rho_min) {
        const double rho_old = U.rho;
        U.rho = rho_min;
        if (std::isfinite(rho_old) && std::fabs(rho_old) > 0.0) {
            const double s = U.rho / rho_old;
            U.rhou *= s;
            U.rhov *= s;
        } else {
            U.rhou = 0.0;
            U.rhov = 0.0;
        }
    }

    double u = U.rhou / U.rho;
    double v = U.rhov / U.rho;
    const double sp = std::sqrt(u*u + v*v);

    if (!std::isfinite(sp) || sp > u_max) {
        const double s = u_max / (sp + 1e-300);
        u *= s;
        v *= s;
        U.rhou = U.rho * u;
        U.rhov = U.rho * v;
    }

    const double kinetic = 0.5 * U.rho * (u*u + v*v);
    if (!std::isfinite(U.E) || !std::isfinite(kinetic) || U.E < kinetic + e_min) {
        U.E = kinetic + e_min;
    }
}


// ============================================================================
// 2D RHS — supports First-Order and WENO3 (dimension-by-dimension)
// ============================================================================
inline void compute_rhs_2d(
    const std::vector<State2D>& U,
    std::vector<State2D>& rhs,
    const Mesh2D& mesh,
    const EOSIdealGas& eos,
    ReconType recon_type)
{
    const int nx_tot = mesh.nx_tot;
    const int ny_tot = mesh.ny_tot;

    const int i0 = mesh.interior_i_start();
    const int i1 = mesh.interior_i_end();
    const int j0 = mesh.interior_j_start();
    const int j1 = mesh.interior_j_end();

    rhs.assign(nx_tot * ny_tot, zero_state2d());

    auto idx_xface = [&](int i, int j) { return j*(nx_tot+1) + i; };
    auto idx_yface = [&](int i, int j) { return j*(nx_tot) + i; };

    // =========================================================================
    // FIRST-ORDER GODUNOV
    // =========================================================================
    if (recon_type == ReconType::FirstOrder)
    {
        std::vector<State2D> Fx( (nx_tot+1) * ny_tot, zero_state2d() );
        std::vector<State2D> Fy( nx_tot * (ny_tot+1), zero_state2d() );

        #ifdef LS_USE_OPENMP
        #pragma omp parallel for collapse(2) schedule(static)
        #endif
        for (int J = j0; J <= j1; ++J) {
            for (int I = i0; I <= i1+1; ++I) {
                const int idxF = idx_xface(I, J);
                const State2D UL = U[ mesh.index(I-1, J) ];
                const State2D UR = U[ mesh.index(I,   J) ];
                Fx[idxF] = rusanov_flux_x(UL, UR, eos);
            }
        }

        #ifdef LS_USE_OPENMP
        #pragma omp parallel for collapse(2) schedule(static)
        #endif
        for (int J = j0; J <= j1+1; ++J) {
            for (int I = i0; I <= i1; ++I) {
                const int idxF = idx_yface(I, J);
                const State2D UL = U[ mesh.index(I, J-1) ];
                const State2D UR = U[ mesh.index(I, J  ) ];
                Fy[idxF] = rusanov_flux_y(UL, UR, eos);
            }
        }

        const double inv_dx = 1.0 / mesh.dx;
        const double inv_dy = 1.0 / mesh.dy;

        #ifdef LS_USE_OPENMP
        #pragma omp parallel for collapse(2) schedule(static)
        #endif
        for (int J = j0; J <= j1; ++J) {
            for (int I = i0; I <= i1; ++I) {
                const int idx = mesh.index(I, J);

                const State2D FxR = Fx[idx_xface(I+1, J)];
                const State2D FxL = Fx[idx_xface(I,   J)];

                const State2D FyT = Fy[idx_yface(I, J+1)];
                const State2D FyB = Fy[idx_yface(I, J  )];

                rhs[idx] =
                    -(FxR - FxL) * inv_dx
                    -(FyT - FyB) * inv_dy;
            }
        }

        return;
    }

    // =========================================================================
    // WENO3
    // =========================================================================
    if (recon_type == ReconType::WENO3)
    {
        if (mesh.ng < 2) {
            throw std::runtime_error("WENO3 requires mesh.ng >= 2 (two ghost cells per side)");
        }

        std::vector<State2D> ULx( (nx_tot+1) * ny_tot, zero_state2d() );
        std::vector<State2D> URx( (nx_tot+1) * ny_tot, zero_state2d() );
        std::vector<State2D> ULy( nx_tot * (ny_tot+1), zero_state2d() );
        std::vector<State2D> URy( nx_tot * (ny_tot+1), zero_state2d() );

        weno3_reconstruct_2d_x(U, ULx, URx, mesh /*, dbg */);
        weno3_reconstruct_2d_y(U, ULy, URy, mesh);

        std::vector<State2D> Fx( (nx_tot+1) * ny_tot, zero_state2d() );
        std::vector<State2D> Fy( nx_tot * (ny_tot+1), zero_state2d() );

        #ifdef LS_USE_OPENMP
        #pragma omp parallel for collapse(2) schedule(static)
        #endif
        for (int J = j0; J <= j1; ++J) {
            for (int I = i0; I <= i1+1; ++I) {
                const int idxF = idx_xface(I, J);
                State2D UL = ULx[idxF];
                State2D UR = URx[idxF];
                sanitize_face_state(UL);
                sanitize_face_state(UR);
                Fx[idxF] = rusanov_flux_x(UL, UR, eos);
            }
        }

        #ifdef LS_USE_OPENMP
        #pragma omp parallel for collapse(2) schedule(static)
        #endif
        for (int J = j0; J <= j1+1; ++J) {
            for (int I = i0; I <= i1; ++I) {
                const int idxF = idx_yface(I, J);
                State2D UL = ULy[idxF];
                State2D UR = URy[idxF];
                sanitize_face_state(UL);
                sanitize_face_state(UR);
                Fy[idxF] = rusanov_flux_y(UL, UR, eos);
            }
        }

        const double inv_dx = 1.0 / mesh.dx;
        const double inv_dy = 1.0 / mesh.dy;

        #ifdef LS_USE_OPENMP
        #pragma omp parallel for collapse(2) schedule(static)
        #endif
        for (int J = j0; J <= j1; ++J) {
            for (int I = i0; I <= i1; ++I) {
                const int idx = mesh.index(I, J);

                const State2D FxR = Fx[idx_xface(I+1, J)];
                const State2D FxL = Fx[idx_xface(I,   J)];

                const State2D FyT = Fy[idx_yface(I, J+1)];
                const State2D FyB = Fy[idx_yface(I, J  )];

                rhs[idx] =
                    -(FxR - FxL) * inv_dx
                    -(FyT - FyB) * inv_dy;
            }
        }

        return;
    }
    // fv_update.hpp inside compute_rhs_2d(...)

    if (recon_type == ReconType::MUSCL ||
        recon_type == ReconType::WENO5 ||
        recon_type == ReconType::WENO5Z)
    {
        const int need_ng =
            (recon_type == ReconType::MUSCL) ? 2 :
            (recon_type == ReconType::WENO5 || recon_type == ReconType::WENO5Z || recon_type == ReconType::TENO5) ? 3 : 2;

        if (mesh.ng < need_ng) {
            throw std::runtime_error("Reconstruction requires more ghost cells than mesh.ng provides.");
        }

        std::vector<State2D> ULx( (nx_tot+1) * ny_tot, zero_state2d() );
        std::vector<State2D> URx( (nx_tot+1) * ny_tot, zero_state2d() );
        std::vector<State2D> ULy( nx_tot * (ny_tot+1), zero_state2d() );
        std::vector<State2D> URy( nx_tot * (ny_tot+1), zero_state2d() );

        reconstruct_2d_x(U, ULx, URx, mesh, recon_type, eos);
        reconstruct_2d_y(U, ULy, URy, mesh, recon_type, eos);

        std::vector<State2D> Fx( (nx_tot+1) * ny_tot, zero_state2d() );
        std::vector<State2D> Fy( nx_tot * (ny_tot+1), zero_state2d() );

        #ifdef LS_USE_OPENMP
        #pragma omp parallel for collapse(2) schedule(static)
        #endif
        for (int J = j0; J <= j1; ++J) {
            for (int I = i0; I <= i1+1; ++I) {
                const int idxF = idx_xface(I, J);
                State2D UL = ULx[idxF];
                State2D UR = URx[idxF];
                sanitize_face_state(UL);
                sanitize_face_state(UR);
                Fx[idxF] = rusanov_flux_x(UL,UR, eos);
            }
        }

        #ifdef LS_USE_OPENMP
        #pragma omp parallel for collapse(2) schedule(static)
        #endif
        for (int J = j0; J <= j1+1; ++J) {
            for (int I = i0; I <= i1; ++I) {
                const int idxF = idx_yface(I, J);
                State2D UL = ULy[idxF];
                State2D UR = URy[idxF];
                sanitize_face_state(UL);
                sanitize_face_state(UR);
                Fy[idxF] = rusanov_flux_y(UL, UR, eos);
            }
        }

        const double inv_dx = 1.0 / mesh.dx;
        const double inv_dy = 1.0 / mesh.dy;

        #ifdef LS_USE_OPENMP
        #pragma omp parallel for collapse(2) schedule(static)
        #endif
        for (int J = j0; J <= j1; ++J) {
            for (int I = i0; I <= i1; ++I) {
                const int idx = mesh.index(I, J);

                const State2D FxR = Fx[idx_xface(I+1, J)];
                const State2D FxL = Fx[idx_xface(I,   J)];

                const State2D FyT = Fy[idx_yface(I, J+1)];
                const State2D FyB = Fy[idx_yface(I, J  )];

                rhs[idx] = -(FxR - FxL) * inv_dx
                        -(FyT - FyB) * inv_dy;
            }
        }

        return;
    }

    if (recon_type == ReconType::TENO5)
    {
        const int need_ng =
            (recon_type == ReconType::MUSCL) ? 2 :
            (recon_type == ReconType::WENO5 || recon_type == ReconType::WENO5Z || recon_type == ReconType::TENO5) ? 3 : 2;

        if (mesh.ng < need_ng) {
            throw std::runtime_error("Reconstruction requires more ghost cells than mesh.ng provides.");
        }

        std::vector<State2D> ULx( (nx_tot+1) * ny_tot, zero_state2d() );
        std::vector<State2D> URx( (nx_tot+1) * ny_tot, zero_state2d() );
        std::vector<State2D> ULy( nx_tot * (ny_tot+1), zero_state2d() );
        std::vector<State2D> URy( nx_tot * (ny_tot+1), zero_state2d() );

        reconstruct_2d_x(U, ULx, URx, mesh, recon_type, eos);
        reconstruct_2d_y(U, ULy, URy, mesh, recon_type, eos);

        std::vector<State2D> Fx( (nx_tot+1) * ny_tot, zero_state2d() );
        std::vector<State2D> Fy( nx_tot * (ny_tot+1), zero_state2d() );

        #ifdef LS_USE_OPENMP
        #pragma omp parallel for collapse(2) schedule(static)
        #endif
        for (int J = j0; J <= j1; ++J) {
            for (int I = i0; I <= i1+1; ++I) {
                const int idxF = idx_xface(I, J);
                State2D UL = ULx[idxF];
                State2D UR = URx[idxF];
                sanitize_face_state(UL);
                sanitize_face_state(UR);
                Fx[idxF] = rusanov_flux_x(UL,UR, eos);
            }
        }

        #ifdef LS_USE_OPENMP
        #pragma omp parallel for collapse(2) schedule(static)
        #endif
        for (int J = j0; J <= j1+1; ++J) {
            for (int I = i0; I <= i1; ++I) {
                const int idxF = idx_yface(I, J);
                State2D UL = ULy[idxF];
                State2D UR = URy[idxF];
                sanitize_face_state(UL);
                sanitize_face_state(UR);
                Fy[idxF] = rusanov_flux_y(UL, UR, eos);
            }
        }

        const double inv_dx = 1.0 / mesh.dx;
        const double inv_dy = 1.0 / mesh.dy;

        #ifdef LS_USE_OPENMP
        #pragma omp parallel for collapse(2) schedule(static)
        #endif
        for (int J = j0; J <= j1; ++J) {
            for (int I = i0; I <= i1; ++I) {
                const int idx = mesh.index(I, J);

                const State2D FxR = Fx[idx_xface(I+1, J)];
                const State2D FxL = Fx[idx_xface(I,   J)];

                const State2D FyT = Fy[idx_yface(I, J+1)];
                const State2D FyB = Fy[idx_yface(I, J  )];

                rhs[idx] = -(FxR - FxL) * inv_dx
                        -(FyT - FyB) * inv_dy;
            }
        }

        return;
    }

    throw std::runtime_error("compute_rhs_2d: unsupported reconstruction type.");

}

// ============================================================================
// 2D RHS with source term S(x,y,t) applied to energy equation
// ============================================================================
// ============================================================================
// 2D RHS with source term S(x,y,t) applied to energy equation (LIMITED)
//   - rhs = -div(F) + S_E
//   - S_E is limited per RK stage to avoid single-cell "detonation" that
//     collapses dt via CFL.
// ============================================================================
template <typename Source>
inline void compute_rhs_2d_with_source(
    const std::vector<State2D>& U,
    std::vector<State2D>& rhs,
    const Mesh2D& mesh,
    const EOSIdealGas& eos,
    ReconType recon_type,
    const Source& S,
    double t,
    double dt_stage)
{
    // 1) Flux divergence part
    compute_rhs_2d(U, rhs, mesh, eos, recon_type);

    // 2) Source part (energy only)
    const int i0 = mesh.interior_i_start();
    const int i1 = mesh.interior_i_end();
    const int j0 = mesh.interior_j_start();
    const int j1 = mesh.interior_j_end();

    // Match your existing floors (keep consistent across codebase)
    constexpr double rho_min  = 1e-8;
    constexpr double e_min    = 1e-10;

    // Limit: max fraction of *current internal energy density* you can add per RK stage
    // dt_stage * S_E <= MAX_FRAC * e_int
    constexpr double MAX_FRAC = 0.10;

    // Guard against nonsense dt (should never happen, but keeps limiter safe)
    if (!std::isfinite(dt_stage) || dt_stage <= 0.0) {
        // If dt is invalid, do NOT add a source term (prevents NaNs from spreading)
        return;
    }

    #ifdef LS_USE_OPENMP
    #pragma omp parallel for collapse(2) schedule(static)
    #endif
    for (int J = j0; J <= j1; ++J) {
        for (int I = i0; I <= i1; ++I) {
            const int idx = mesh.index(I, J);

            const double xc = mesh.xc(I);
            const double yc = mesh.yc(J);

            const double SE_raw = S(xc, yc, t);  // [W/m^3] == [J/m^3/s]

            // If source is NaN/Inf, ignore it (or you can throw)
            if (!std::isfinite(SE_raw)) {
                continue;
            }

            // Compute a safe estimate of internal energy density in THIS cell (from U, not from rhs)
            const auto& Ui = U[idx];

            // density floor
            const double rho = std::max(Ui.rho, rho_min);

            // velocities (safe because rho floored)
            const double u = Ui.rhou / rho;
            const double v = Ui.rhov / rho;

            const double kinetic = 0.5 * rho * (u*u + v*v);

            double e_int = Ui.E - kinetic;
            if (!std::isfinite(e_int)) e_int = e_min;
            if (e_int < e_min) e_int = e_min;

            // Max allowed magnitude of S_E so that the energy change this stage is bounded
            const double max_SE = (MAX_FRAC * e_int) / dt_stage;

            // Clamp the source
            const double SE = std::clamp(SE_raw, -max_SE, +max_SE);

            rhs[idx].E += SE;
        }
    }
}


} // namespace ls
