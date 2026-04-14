/*
==============================================================================
 File:        laser2d.cpp
 Purpose:     2D Euler testbed for laser-ablation-style plume expansion
 Author:      Lucas Pierce (rewritten/cleaned)
 Notes:
   - This driver is designed for your THESIS goal: compare spatial/temporal schemes.
   - Two modes:
       1) laser_mode = "initial_plume"      (recommended for thesis comparisons)
       2) laser_mode = "volumetric_source"  (Gaussian energy deposition)
   - Supports recon: FirstOrder, MUSCL, WENO3, WENO5, WENO5Z, TENO5
   - Adds a config-controlled dt diagnostic switch to silence spam.

 Requirements (your codebase already provides these):
   ls/types.hpp, ls/mesh.hpp, ls/eos.hpp, ls/bc.hpp, ls/recon.hpp, ls/fv_update.hpp,
   ls/time_int.hpp, ls/io.hpp, ls/source.hpp, ls/external/toml.hpp
==============================================================================
*/

#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <limits>

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
    double amax_raw = 0.0;     // |u|+cs from raw eos state
    double amax_cfl = 0.0;     // |u|+cs from CFL-safe state (rho floored)
    int I = -1, J = -1;
    double rho_raw = 0.0;
    double rho_used_raw = 0.0;
    double rho_used_cfl = 0.0;
    double u = 0.0, v = 0.0;
    double cs_raw = 0.0;
    double cs_cfl = 0.0;
    double p_raw = 0.0;
};

inline State2D make_cfl_state(const State2D& Ui, double rho_cfl_floor)
{
    State2D Ucfl = Ui;

    if (!std::isfinite(Ucfl.rho) || Ucfl.rho < rho_cfl_floor) {
        // Preserve velocities as best as possible
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

            const double rho_floor_uv = 1e-8;
            const double rho_uv = std::max(Ui.rho, rho_floor_uv);

            const double u = Ui.rhou / rho_uv;
            const double v = Ui.rhov / rho_uv;
            if (!std::isfinite(u) || !std::isfinite(v)) continue;

            const double cs_raw = eos.sound_speed(Ui);
            if (!std::isfinite(cs_raw)) continue;

            const double a_raw = std::sqrt(u*u + v*v) + cs_raw;

            const State2D Ucfl = make_cfl_state(Ui, rho_cfl_floor);
            const double cs_cfl = eos.sound_speed(Ucfl);
            if (!std::isfinite(cs_cfl)) continue;

            const double a_cfl = std::sqrt(u*u + v*v) + cs_cfl;

            if (a_cfl > d.amax_cfl) {
                d.amax_raw = a_raw;
                d.amax_cfl = a_cfl;
                d.I = I; d.J = J;

                d.rho_raw = Ui.rho;
                d.rho_used_raw = rho_uv;
                d.rho_used_cfl = Ucfl.rho;

                d.u = u; d.v = v;
                d.cs_raw = cs_raw;
                d.cs_cfl = cs_cfl;

                d.p_raw = eos.pressure(Ui);
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
    //   "initial_plume" or "volumetric_source"
    std::string laser_mode{"volumetric_source"};

    // Plume IC
    struct PlumeIC {
        std::string shape{"circle"};  // "circle" | "box" | "oval_crater"
        double x0{0.5};
        double y0{0.5};
        double rx{0.02};   // circle radius or x half-size
        double ry{0.02};   // y half-size (for oval/box)
        double p_plume{2.0e8};   // Pa
        double rho_plume{-1.0};  // if > 0, use directly
        double T_plume{-1.0};    // parsed but not used unless rho_plume missing (see note)
    } plume;

    // Laser Gaussian pulse (used only if laser_mode="volumetric_source")
    double A{5e12};
    double xL{0.5};
    double yL{0.5};
    double wx{0.02};
    double wy{0.02};
    double t0{1.0e-4};
    double wt{2.0e-5};

    // CFL safety knobs
    double rho_cfl_floor{1e-4};   // used ONLY for dt evaluation
    double dt_warn{1e-12};

    // Diagnostic spam control
    bool enable_dt_diagnostics{false};

    // ------------------------------------------------------------------
    // Thesis diagnostics (summary outputs)
    // ------------------------------------------------------------------
    bool enable_thesis_diagnostics{true};
    int  diag_interval{-1};            // if < 0, uses out_interval
    double shock_probe_y{-1.0};        // if < 0, uses Ly/2
    double shock_smooth_frac{0.05};    // fraction of max |dp/dx| for "smooth" mask
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

    // knobs
    c.rho_cfl_floor = toml::find_or(tbl, "rho_cfl_floor", c.rho_cfl_floor);
    c.dt_warn       = toml::find_or(tbl, "dt_warn", c.dt_warn);

    // Backward-compat: some configs used dt_diagnostics
    c.enable_dt_diagnostics = toml::find_or(tbl, "enable_dt_diagnostics",
                              toml::find_or(tbl, "dt_diagnostics", c.enable_dt_diagnostics));

    // Thesis diagnostics
    c.enable_thesis_diagnostics = toml::find_or(tbl, "enable_thesis_diagnostics", c.enable_thesis_diagnostics);
    c.diag_interval             = toml::find_or(tbl, "diag_interval", c.diag_interval);
    c.shock_probe_y             = toml::find_or(tbl, "shock_probe_y", c.shock_probe_y);
    c.shock_smooth_frac         = toml::find_or(tbl, "shock_smooth_frac", c.shock_smooth_frac);

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
        // optional field (you said “or specify rho_plume directly”)
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
// Thesis diagnostics helpers
//   These are intentionally "solution-only" metrics (no peeking into recon
//   internals) so they work for every reconstruction scheme.
// ============================================================================
struct ShockMetrics2D {
    bool ok{false};
    int J_probe{-1};
    double y_probe{0.0};
    double x_shock{0.0};
    double dpdx_max{0.0};
    double p_min{0.0};
    double p_max{0.0};
    int i10{-1}, i90{-1};
    double thickness_cells{0.0};
    double thickness_length{0.0};
};

static inline int nearest_row_for_y(const Mesh2D& mesh, double y_target)
{
    int bestJ = mesh.interior_j_start();
    double best = std::numeric_limits<double>::infinity();
    for (int J = mesh.interior_j_start(); J <= mesh.interior_j_end(); ++J) {
        const double dy = std::fabs(mesh.yc(J) - y_target);
        if (dy < best) { best = dy; bestJ = J; }
    }
    return bestJ;
}

static inline ShockMetrics2D compute_shock_metrics_row(const std::vector<State2D>& U,
                                                       const Mesh2D& mesh,
                                                       const EOSIdealGas& eos,
                                                       int J)
{
    ShockMetrics2D m;
    m.J_probe = J;
    m.y_probe = mesh.yc(J);

    const int i0 = mesh.interior_i_start();
    const int i1 = mesh.interior_i_end();
    const int n  = (i1 - i0 + 1);
    if (n < 5) return m;

    std::vector<double> p(n, 0.0);
    for (int I = i0; I <= i1; ++I) {
        p[I - i0] = eos.pressure(U[mesh.index(I,J)]);
    }

    auto mm = std::minmax_element(p.begin(), p.end());
    m.p_min = *mm.first;
    m.p_max = *mm.second;
    const double dp = m.p_max - m.p_min;
    if (!(std::isfinite(dp) && dp > 0.0)) return m;

    // Locate shock: max |dp/dx| along the probe row
    int imax = -1;
    double gmax = 0.0;
    for (int k = 1; k < n-1; ++k) {
        const double dpdx = (p[k+1] - p[k-1]) / (2.0 * mesh.dx);
        const double a = std::fabs(dpdx);
        if (std::isfinite(a) && a > gmax) { gmax = a; imax = k; }
    }
    if (imax < 0) return m;

    m.dpdx_max = gmax;
    m.x_shock  = mesh.xc(i0 + imax);

    // 10%-90% thickness based on min/max pressure along this line
    const double p10 = m.p_min + 0.10 * dp;
    const double p90 = m.p_min + 0.90 * dp;
    int i10 = -1, i90 = -1;
    for (int k = 0; k < n; ++k) {
        if (i10 < 0 && p[k] >= p10) i10 = k;
        if (i10 >= 0 && p[k] >= p90) { i90 = k; break; }
    }

    // If the wave is a rarefaction (decreasing), flip search direction
    if (i10 < 0 || i90 < 0) {
        const double p10d = m.p_max - 0.10 * dp;
        const double p90d = m.p_max - 0.90 * dp;
        i10 = -1; i90 = -1;
        for (int k = 0; k < n; ++k) {
            if (i10 < 0 && p[k] <= p10d) i10 = k;
            if (i10 >= 0 && p[k] <= p90d) { i90 = k; break; }
        }
    }

    if (i10 >= 0 && i90 >= 0 && i90 >= i10) {
        m.i10 = i0 + i10;
        m.i90 = i0 + i90;
        m.thickness_cells  = static_cast<double>(i90 - i10);
        m.thickness_length = m.thickness_cells * mesh.dx;
        m.ok = true;
    }

    return m;
}

static inline double total_variation_rho(const std::vector<State2D>& U, const Mesh2D& mesh)
{
    double tv = 0.0;
    for (int J = mesh.interior_j_start(); J <= mesh.interior_j_end(); ++J) {
        for (int I = mesh.interior_i_start(); I <= mesh.interior_i_end(); ++I) {
            const double r  = U[mesh.index(I,J)].rho;
            const double rx = U[mesh.index(I+1,J)].rho;
            const double ry = U[mesh.index(I,J+1)].rho;
            tv += std::fabs(rx - r) + std::fabs(ry - r);
        }
    }
    return tv;
}

static inline double smooth_curvature_proxy_p(const std::vector<State2D>& U,
                                              const Mesh2D& mesh,
                                              const EOSIdealGas& eos,
                                              double dpdx_smooth_cut)
{
    // A simple, scheme-agnostic proxy: average |\nabla^2 p| over regions where
    // |dp/dx| is small ("smooth" mask). Larger values generally correlate with
    // excess dissipation/dispersion in otherwise smooth areas.
    double sum = 0.0;
    long long cnt = 0;

    for (int J = mesh.interior_j_start()+1; J <= mesh.interior_j_end()-1; ++J) {
        for (int I = mesh.interior_i_start()+1; I <= mesh.interior_i_end()-1; ++I) {
            const double pC = eos.pressure(U[mesh.index(I,J)]);
            const double pL = eos.pressure(U[mesh.index(I-1,J)]);
            const double pR = eos.pressure(U[mesh.index(I+1,J)]);
            const double pB = eos.pressure(U[mesh.index(I,J-1)]);
            const double pT = eos.pressure(U[mesh.index(I,J+1)]);

            const double dpdx = (pR - pL) / (2.0 * mesh.dx);
            if (!std::isfinite(dpdx) || std::fabs(dpdx) > dpdx_smooth_cut) continue;

            const double d2pdx2 = (pR - 2.0*pC + pL) / (mesh.dx * mesh.dx);
            const double d2pdy2 = (pT - 2.0*pC + pB) / (mesh.dy * mesh.dy);
            const double lap = std::fabs(d2pdx2 + d2pdy2);
            if (std::isfinite(lap)) { sum += lap; cnt += 1; }
        }
    }

    if (cnt <= 0) return 0.0;
    return sum / static_cast<double>(cnt);
}

class ThesisDiagCSV {
public:
    void open(const std::string& folder)
    {
        const std::string path = folder + "/diagnostics.csv";
        out_.open(path, std::ios::out);
        out_ << "step,t,dt,wall_step_s,wall_total_s,";
        out_ << "shock_ok,shock_y,shock_x,dpdx_max,p_min,p_max,thickness_cells,thickness_m,";
        out_ << "TV_rho,smooth_lap_p\n";
        out_.flush();
    }

    void write(int step, double t, double dt,
               double wall_step_s, double wall_total_s,
               const ShockMetrics2D& sm,
               double tv_rho, double smooth_lap_p)
    {
        if (!out_.is_open()) return;
        out_ << step << "," << std::setprecision(16) << t << "," << dt << ","
             << wall_step_s << "," << wall_total_s << ","
             << (sm.ok ? 1 : 0) << "," << sm.y_probe << "," << sm.x_shock << ","
             << sm.dpdx_max << "," << sm.p_min << "," << sm.p_max << ","
             << sm.thickness_cells << "," << sm.thickness_length << ","
             << tv_rho << "," << smooth_lap_p << "\n";
        out_.flush();
    }

private:
    std::ofstream out_;
};

// ============================================================================
// Plume region helper
// ============================================================================
static inline bool inside_plume_region(const LaserConfig2D::PlumeIC& p, double x, double y)
{
    // "circle": (x-x0)^2 + (y-y0)^2 <= r^2 with r=rx
    if (str_ieq(p.shape, "circle")) {
        const double dx = x - p.x0;
        const double dy = y - p.y0;
        const double r  = p.rx;
        return (dx*dx + dy*dy) <= (r*r);
    }

    // "box": |x-x0| <= rx and |y-y0| <= ry
    if (str_ieq(p.shape, "box")) {
        return (std::fabs(x - p.x0) <= p.rx) && (std::fabs(y - p.y0) <= p.ry);
    }

    // "oval_crater": elliptical region: (x-x0)^2/rx^2 + (y-y0)^2/ry^2 <= 1
    //   (This is still 2D; "depth" is emulated by smaller ry and placing y0 near wall)
    if (str_ieq(p.shape, "oval_crater")) {
        const double dx = (x - p.x0) / p.rx;
        const double dy = (y - p.y0) / p.ry;
        return (dx*dx + dy*dy) <= 1.0;
    }

    // default fallback: circle
    {
        const double dx = x - p.x0;
        const double dy = y - p.y0;
        const double r  = p.rx;
        return (dx*dx + dy*dy) <= (r*r);
    }
}

// ============================================================================
// dt computation
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

            // u,v computed with a tiny rho floor to avoid division blowups
            const double rho_floor_uv = 1e-8;
            const double rho_uv = std::max(Ui.rho, rho_floor_uv);


            const State2D Ucfl = make_cfl_state(Ui, rho_cfl_floor);

            const double u = Ucfl.rhou / Ucfl.rho;
            const double v = Ucfl.rhov / Ucfl.rho;
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

    // Background IC: uniform (rho0,u0,v0,p0)
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

    // Optional: overwrite a region with a plume IC
    if (str_ieq(c.laser_mode, "initial_plume")) {
        // IMPORTANT STABILITY NOTE:
        // For your current Euler+ideal-gas closure, the safest plume spec is (rho_plume, p_plume).
        // If rho_plume is not provided, we choose a conservative rho from background
        // to avoid near-vacuum -> dt collapse.
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

            // overwrite state in plume region
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
                         "convert (p,T)->rho because the solver is not parameterized by R.\n"
                         "           Provide plume_ic.rho_plume explicitly for paper-consistent runs.\n";
        }
    }

    // Laser source object (only used if volumetric_source)
    GaussianPulse2D laser{
        c.A, c.xL, c.yL,
        c.wx, c.wy,
        c.t0, c.wt
    };

    // Output
    ensure_dir(c.out_folder);
    double t = 0.0;
    int step = 0;

    // Thesis diagnostics writer
    ThesisDiagCSV diag;
    const int diag_every = (c.diag_interval > 0) ? c.diag_interval : c.out_interval;
    const double y_probe = (c.shock_probe_y >= 0.0) ? c.shock_probe_y : 0.5 * c.Ly;
    const int J_probe = nearest_row_for_y(mesh, y_probe);
    if (c.enable_thesis_diagnostics) {
        diag.open(c.out_folder);
        std::cout << "  diagnostics = ON"
                  << " (every " << diag_every << " steps)"
                  << " shock_probe_y=" << y_probe
                  << " (J=" << J_probe << ", y=" << mesh.yc(J_probe) << ")"
                  << "\n";
    } else {
        std::cout << "  diagnostics = OFF\n";
    }

    using clock = std::chrono::steady_clock;
    const auto t_start_total = clock::now();
    double wall_total_s = 0.0;

    write_csv_2d(c.out_folder, step, t, mesh, U, eos);

    if (str_ieq(c.laser_mode, "volumetric_source") && c.t0 > c.t_end) {
        std::cout << "⚠️ NOTE: laser.t0 (" << c.t0 << ") > t_end (" << c.t_end
                  << "). Pulse center outside simulation window; source ~off.\n";
    }

    // Time loop
    while (t < c.t_end) {
        const auto t_start_step = clock::now();
        const double dt = compute_dt_cfl(U, mesh, eos, c.CFL, c.rho_cfl_floor, t, c.t_end);

        // Optional dt diagnostics
        if (c.enable_dt_diagnostics && dt < c.dt_warn) {
            auto d = find_dt_killer_2d(U, mesh, eos, c.rho_cfl_floor);
            std::cout << "⚠️ dt collapse at step=" << step << " t=" << t
                      << " dt=" << dt
                      << " amax_raw=" << d.amax_raw
                      << " amax_cfl=" << d.amax_cfl
                      << " at (I,J)=(" << d.I << "," << d.J << ")"
                      << " x=" << mesh.xc(d.I) << " y=" << mesh.yc(d.J)
                      << " rho_raw=" << d.rho_raw
                      << " rho_uv=" << d.rho_used_raw
                      << " rho_cfl=" << d.rho_used_cfl
                      << " u=" << d.u << " v=" << d.v
                      << " p_raw=" << d.p_raw
                      << " cs_raw=" << d.cs_raw
                      << " cs_cfl=" << d.cs_cfl
                      << " rho_cfl_floor=" << c.rho_cfl_floor
                      << "\n";
        }

        // Advance one step
        if (str_ieq(c.laser_mode, "volumetric_source")) {
            advance_rk3_2d_with_source(U, rhs, mesh, eos, bc2d, recon_type, laser, t, dt);
        } else {
            // default / "initial_plume"
            advance_rk3_2d(U, rhs, mesh, eos, bc2d, recon_type, dt);
        }

        const auto t_end_step = clock::now();
        const double wall_step_s = std::chrono::duration<double>(t_end_step - t_start_step).count();
        wall_total_s = std::chrono::duration<double>(t_end_step - t_start_total).count();

        t += dt;
        ++step;

        // Summary diagnostics (scheme comparison)
        if (c.enable_thesis_diagnostics && (step % diag_every == 0 || t >= c.t_end)) {
            const auto sm = compute_shock_metrics_row(U, mesh, eos, J_probe);
            const double dpdx_cut = std::max(1e-30, c.shock_smooth_frac * sm.dpdx_max);
            const double tv_rho = total_variation_rho(U, mesh);
            const double smooth_lap_p = smooth_curvature_proxy_p(U, mesh, eos, dpdx_cut);
            diag.write(step, t, dt, wall_step_s, wall_total_s, sm, tv_rho, smooth_lap_p);

            std::cout << "📌 diag step=" << step
                      << " t=" << t
                      << " wall_step_s=" << wall_step_s
                      << " shock_x=" << sm.x_shock
                      << " thickness_cells=" << sm.thickness_cells
                      << " TV_rho=" << tv_rho
                      << " smooth_lap_p=" << smooth_lap_p
                      << "\n";
        }

        if (step % c.out_interval == 0 || t >= c.t_end) {
            write_csv_2d(c.out_folder, step, t, mesh, U, eos);
            std::cout << "📝 Wrote step " << step << " at t=" << t << ", dt=" << dt << "\n";
        }
    }

    std::cout << "✅ Done. Final time: " << t << "\n";
    return 0;
}
