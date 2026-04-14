/*
==============================================================================
 File:        laser2d.cpp
 Purpose:     2D Euler testbed for laser-ablation-style plume expansion
 Author:      Lucas Pierce (rewritten/cleaned)
 Notes:
   - Two modes:
       1) laser_mode = "initial_plume"
       2) laser_mode = "volumetric_source"
   - Supports recon: FirstOrder, MUSCL, WENO3, WENO5, WENO5Z, TENO5
==============================================================================
*/

#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>

#ifdef LS_USE_OPENMP
  #include <omp.h>
#endif

#include "ls/external/toml.hpp"
#include "ls/types.hpp"
#include "ls/mesh.hpp"
#include "ls/eos.hpp"
#include "ls/bc.hpp"
#include "ls/recon.hpp"
#include "ls/fv_update.hpp"
#include "ls/time_int.hpp"
#include "ls/io.hpp"
#include "ls/source.hpp"

using namespace ls;

// ============================================================================
// Diagnostics (dt collapse)
// ============================================================================
struct DtKiller2D {
    double amax = 0.0;
    int I = -1, J = -1;
    double rho = 0.0;
    double u = 0.0, v = 0.0;
    double cs = 0.0;
    double p = 0.0;
};

inline State2D make_cfl_state(const State2D& Ui, double rho_cfl_floor)
{
    State2D Ucfl = Ui;

    if (!std::isfinite(Ucfl.rho) || Ucfl.rho < rho_cfl_floor) {
        // Preserve velocities (as best as possible) when flooring rho
        const double rho_old = std::max(Ui.rho, 1e-12);
        const double u_old = Ui.rhou / rho_old;
        const double v_old = Ui.rhov / rho_old;

        Ucfl.rho  = rho_cfl_floor;
        Ucfl.rhou = rho_cfl_floor * u_old;
        Ucfl.rhov = rho_cfl_floor * v_old;

        if (!std::isfinite(Ucfl.E) || Ucfl.E < 1e-10) Ucfl.E = 1e-10;
    }

    return Ucfl;
}

inline DtKiller2D find_dt_killer_2d(const std::vector<State2D>& U,
                                   const Mesh2D& mesh,
                                   const EOSIdealGas& eos,
                                   double rho_cfl_floor)
{
    DtKiller2D d;

    for (int J = mesh.interior_j_start(); J <= mesh.interior_j_end(); ++J) {
        for (int I = mesh.interior_i_start(); I <= mesh.interior_i_end(); ++I) {
            const auto& Ui = U[mesh.index(I,J)];

            // CFL-safe state (rho flooring)
            const State2D Ucfl = make_cfl_state(Ui, rho_cfl_floor);

            const double rho = std::max(Ucfl.rho, 1e-12);
            const double u   = Ucfl.rhou / rho;
            const double v   = Ucfl.rhov / rho;
            if (!std::isfinite(u) || !std::isfinite(v)) continue;

            const double cs = eos.sound_speed(Ucfl);
            if (!std::isfinite(cs)) continue;

            const double a = std::sqrt(u*u + v*v) + cs;

            if (a > d.amax) {
                d.amax = a;
                d.I = I; d.J = J;
                d.rho = Ui.rho;         // report RAW rho (helpful)
                d.u = u; d.v = v;
                d.cs = cs;
                d.p = eos.pressure(Ui); // report RAW pressure
            }
        }
    }

    return d;
}

// ============================================================================
// Config
// ============================================================================
struct LaserConfig2D {
    int    nx{200};
    int    ny{200};
    double Lx{1.0};
    double Ly{1.0};

    double CFL{0.1};
    double t_end{2.0e-4};
    double gamma{1.4};
    std::string recon{"WENO3"};

    // initial uniform background gas
    double rho0{1.0};
    double u0{0.0};
    double v0{0.0};
    double p0{1.0e5};

    // BCs
    std::string left{"Outflow"};
    std::string right{"Outflow"};
    std::string bottom{"Outflow"};
    std::string top{"Outflow"};

    // output
    std::string out_folder{"data/laser2d"};
    int out_interval{10};

    // Mode
    std::string laser_mode{"volumetric_source"};

    // Plume IC
    struct PlumeIC {
        std::string shape{"circle"};  // "circle" | "box" | "oval_crater"
        double x0{0.5};
        double y0{0.5};
        double rx{0.02};
        double ry{0.02};
        double p_plume{2.0e8};
        double rho_plume{-1.0};
        double T_plume{-1.0};
    } plume;

    // Laser Gaussian pulse
    double A{5e12};
    double xL{0.5};
    double yL{0.5};
    double wx{0.02};
    double wy{0.02};
    double t0{1.0e-4};
    double wt{2.0e-5};

    // CFL safety knobs
    double rho_cfl_floor{1e-4};
    double dt_warn{1e-12};

    bool enable_dt_diagnostics{false};
};

static inline bool str_ieq(const std::string& a, const std::string& b)
{
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        if (std::tolower(a[i]) != std::tolower(b[i])) return false;
    }
    return true;
}

LaserConfig2D load_laser_config_2d(const std::string& path)
{
    LaserConfig2D c;
    auto tbl = toml::parse(path);

    c.nx    = toml::find_or(tbl, "nx",    c.nx);
    c.ny    = toml::find_or(tbl, "ny",    c.ny);
    c.Lx    = toml::find_or(tbl, "Lx",    c.Lx);
    c.Ly    = toml::find_or(tbl, "Ly",    c.Ly);
    c.CFL   = toml::find_or(tbl, "CFL",   c.CFL);
    c.t_end = toml::find_or(tbl, "t_end", c.t_end);
    c.gamma = toml::find_or(tbl, "gamma", c.gamma);
    c.recon = toml::find_or(tbl, "reconstruction", c.recon);

    c.out_folder   = toml::find_or(tbl, "out_folder",   c.out_folder);
    c.out_interval = toml::find_or(tbl, "out_interval", c.out_interval);

    c.laser_mode = toml::find_or(tbl, "laser_mode", c.laser_mode);

    c.rho_cfl_floor = toml::find_or(tbl, "rho_cfl_floor", c.rho_cfl_floor);
    c.dt_warn       = toml::find_or(tbl, "dt_warn", c.dt_warn);
    c.enable_dt_diagnostics = toml::find_or(tbl, "enable_dt_diagnostics", c.enable_dt_diagnostics);

    if (tbl.contains("bc")) {
        auto bc_tbl = toml::find(tbl, "bc");
        c.left   = toml::find_or(bc_tbl, "left",   c.left);
        c.right  = toml::find_or(bc_tbl, "right",  c.right);
        c.bottom = toml::find_or(bc_tbl, "bottom", c.bottom);
        c.top    = toml::find_or(bc_tbl, "top",    c.top);
    }

    if (tbl.contains("initial_condition")) {
        auto ic = toml::find(tbl, "initial_condition");
        c.rho0 = toml::find_or(ic, "rho0", c.rho0);
        c.u0   = toml::find_or(ic, "u0",   c.u0);
        c.v0   = toml::find_or(ic, "v0",   c.v0);
        c.p0   = toml::find_or(ic, "p0",   c.p0);
    }

    if (tbl.contains("plume_ic")) {
        auto p = toml::find(tbl, "plume_ic");
        c.plume.shape   = toml::find_or(p, "shape", c.plume.shape);
        c.plume.x0      = toml::find_or(p, "x0", c.plume.x0);
        c.plume.y0      = toml::find_or(p, "y0", c.plume.y0);
        c.plume.rx      = toml::find_or(p, "rx", c.plume.rx);
        c.plume.ry      = toml::find_or(p, "ry", c.plume.ry);
        c.plume.p_plume = toml::find_or(p, "p_plume", c.plume.p_plume);
        c.plume.T_plume = toml::find_or(p, "T_plume", c.plume.T_plume);
        c.plume.rho_plume = toml::find_or(p, "rho_plume", c.plume.rho_plume);
    }

    if (tbl.contains("laser")) {
        auto laser = toml::find(tbl, "laser");
        c.A  = toml::find_or(laser, "A",  c.A);
        c.xL = toml::find_or(laser, "x0", c.xL);
        c.yL = toml::find_or(laser, "y0", c.yL);
        c.wx = toml::find_or(laser, "wx", c.wx);
        c.wy = toml::find_or(laser, "wy", c.wy);
        c.t0 = toml::find_or(laser, "t0", c.t0);
        c.wt = toml::find_or(laser, "wt", c.wt);
    }

    return c;
}

// ============================================================================
// Plume region helper
// ============================================================================
static inline bool inside_plume_region(const LaserConfig2D::PlumeIC& p, double x, double y)
{
    if (str_ieq(p.shape, "circle")) {
        const double dx = x - p.x0;
        const double dy = y - p.y0;
        const double r  = p.rx;
        return (dx*dx + dy*dy) <= (r*r);
    }

    if (str_ieq(p.shape, "box")) {
        return (std::fabs(x - p.x0) <= p.rx) && (std::fabs(y - p.y0) <= p.ry);
    }

    if (str_ieq(p.shape, "oval_crater")) {
        const double dx = (x - p.x0) / p.rx;
        const double dy = (y - p.y0) / p.ry;
        return (dx*dx + dy*dy) <= 1.0;
    }

    // fallback: circle
    const double dx = x - p.x0;
    const double dy = y - p.y0;
    const double r  = p.rx;
    return (dx*dx + dy*dy) <= (r*r);
}

// ============================================================================
// dt computation (FIXED)
// ============================================================================
static inline double compute_dt_cfl(const std::vector<State2D>& U,
                                   const Mesh2D& mesh,
                                   const EOSIdealGas& eos,
                                   double CFL,
                                   double rho_cfl_floor,
                                   double t,
                                   double t_end)
{
    double smax = 1e-12;

#ifdef LS_USE_OPENMP
#pragma omp parallel for collapse(2) reduction(max:smax) schedule(static)
#endif
    for (int J = mesh.interior_j_start(); J <= mesh.interior_j_end(); ++J) {
        for (int I = mesh.interior_i_start(); I <= mesh.interior_i_end(); ++I) {
            const auto& Ui = U[mesh.index(I,J)];

            // Use CFL-safe state for BOTH velocities and sound speed
            const State2D Ucfl = make_cfl_state(Ui, rho_cfl_floor);

            const double rho = std::max(Ucfl.rho, 1e-12);
            const double u   = Ucfl.rhou / rho;
            const double v   = Ucfl.rhov / rho;
            if (!std::isfinite(u) || !std::isfinite(v)) continue;

            const double cs = eos.sound_speed(Ucfl);
            if (!std::isfinite(cs)) continue;

            const double a = std::sqrt(u*u + v*v) + cs;
            if (std::isfinite(a) && a > smax) smax = a;
        }
    }

    const double dt_cfl = CFL * std::min(mesh.dx, mesh.dy) / smax;
    const double dt_end = t_end - t;
    return std::min(dt_cfl, dt_end);
}

// ============================================================================
// Main
// ============================================================================
int main(int argc, char** argv)
{
    const std::string cfg = (argc > 1 ? argv[1] : "configs/laser2d.toml");
    const LaserConfig2D c = load_laser_config_2d(cfg);

    std::cout << "🚀 laser2d (method-comparison driver)\n";
    std::cout << "  config    = " << cfg << "\n";
    std::cout << "  nx=" << c.nx << ", ny=" << c.ny
              << ", Lx=" << c.Lx << ", Ly=" << c.Ly
              << ", CFL=" << c.CFL << ", t_end=" << c.t_end
              << ", recon=" << c.recon
              << ", laser_mode=" << c.laser_mode << "\n";

#ifdef LS_USE_OPENMP
    std::cout << "  OpenMP    = ON (" << omp_get_max_threads() << " max threads)\n";
#else
    std::cout << "  OpenMP    = OFF\n";
#endif

    // EOS
    EOSIdealGas eos;
    eos.gamma = c.gamma;

    // Reconstruction + ghost cells
    ReconType recon_type = parse_recon_type(c.recon);
    const int NG = num_ghost_cells(c.recon);

    // Mesh
    Mesh2D mesh(c.nx, c.ny, c.Lx, c.Ly, NG);
    const int nx_tot = mesh.nx_tot;
    const int ny_tot = mesh.ny_tot;

    // State arrays
    std::vector<State2D> U(nx_tot * ny_tot, zero_state2d());
    std::vector<State2D> rhs(nx_tot * ny_tot, zero_state2d());

    // BCs
    Bc2D bc2d{
        parse_bc_type(c.left),
        parse_bc_type(c.right),
        parse_bc_type(c.bottom),
        parse_bc_type(c.top)
    };

    // Background IC
#ifdef LS_USE_OPENMP
#pragma omp parallel for collapse(2) schedule(static)
#endif
    for (int J = 0; J < ny_tot; ++J) {
        for (int I = 0; I < nx_tot; ++I) {
            const double rho = c.rho0;
            const double u   = c.u0;
            const double v   = c.v0;
            const double p   = c.p0;
            const double E   = p / (eos.gamma - 1.0) + 0.5 * rho * (u*u + v*v);
            U[mesh.index(I,J)] = State2D{rho, rho*u, rho*v, E};
        }
    }

    // Initial plume override
    if (str_ieq(c.laser_mode, "initial_plume")) {
        const double rho_plume_safe =
            (c.plume.rho_plume > 0.0) ? c.plume.rho_plume : std::max(c.rho0, 1e-3);

        const double rho_min = 1e-8;
        const double p_min   = 1e-6;

        int count_plume = 0;

#ifdef LS_USE_OPENMP
#pragma omp parallel for collapse(2) reduction(+:count_plume) schedule(static)
#endif
        for (int J = 0; J < ny_tot; ++J) {
            for (int I = 0; I < nx_tot; ++I) {
                const double x = mesh.xc(I);
                const double y = mesh.yc(J);

                if (!inside_plume_region(c.plume, x, y)) continue;

                const double rho = std::max(rho_plume_safe, rho_min);
                const double u = 0.0;
                const double v = 0.0;
                const double p = std::max(c.plume.p_plume, p_min);
                const double E = p / (eos.gamma - 1.0) + 0.5 * rho * (u*u + v*v);

                U[mesh.index(I,J)] = State2D{rho, rho*u, rho*v, E};
                count_plume += 1;
            }
        }

        std::cout << "  plume_ic  = ON"
                  << " shape=" << c.plume.shape
                  << " cells=" << count_plume
                  << " rho_plume=" << rho_plume_safe
                  << " p_plume=" << c.plume.p_plume << "\n";

        if (c.plume.rho_plume <= 0.0 && c.plume.T_plume > 0.0) {
            std::cout << "  ⚠️ NOTE: plume_ic provides T_plume, but current driver does not "
                         "convert (p,T)->rho; provide plume_ic.rho_plume explicitly.\n";
        }
    }

    // Laser source (volumetric)
    GaussianPulse2D laser{
        c.A, c.xL, c.yL,
        c.wx, c.wy,
        c.t0, c.wt
    };

    // Output
    ensure_dir(c.out_folder);
    double t = 0.0;
    int step = 0;

    write_csv_2d(c.out_folder, step, t, mesh, U, eos);

    // Time loop
    while (t < c.t_end) {
        const double dt = compute_dt_cfl(U, mesh, eos, c.CFL, c.rho_cfl_floor, t, c.t_end);

        if (c.enable_dt_diagnostics && dt < c.dt_warn) {
            auto d = find_dt_killer_2d(U, mesh, eos, c.rho_cfl_floor);
            std::cout << "⚠️ dt collapse at step=" << step << " t=" << t
                      << " dt=" << dt
                      << " amax=" << d.amax
                      << " at (I,J)=(" << d.I << "," << d.J << ")"
                      << " x=" << mesh.xc(d.I) << " y=" << mesh.yc(d.J)
                      << " rho_raw=" << d.rho
                      << " u=" << d.u << " v=" << d.v
                      << " p_raw=" << d.p
                      << " cs=" << d.cs
                      << " rho_cfl_floor=" << c.rho_cfl_floor
                      << "\n";
        }

        if (str_ieq(c.laser_mode, "volumetric_source")) {
            advance_rk3_2d_with_source(U, rhs, mesh, eos, bc2d, recon_type, laser, t, dt);
        } else {
            advance_rk3_2d(U, rhs, mesh, eos, bc2d, recon_type, dt);
        }

        t += dt;
        ++step;

        if (step % c.out_interval == 0 || t >= c.t_end) {
            write_csv_2d(c.out_folder, step, t, mesh, U, eos);
            std::cout << "📝 Wrote step " << step << " at t=" << t << ", dt=" << dt << "\n";
        }
    }

    std::cout << "✅ Done. Final time: " << t << "\n";
    return 0;
}
