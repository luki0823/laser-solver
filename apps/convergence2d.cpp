// apps/convergence2d.cpp
//
// 2D convergence driver with:
//  - Exact entropy-wave advection solution (Euler-exact)
//  - Progress printing every N steps
//  - Diagnostics (rho_min/p_min/u_max, etc.) every N steps
//  - Stall detection: abort if t + dt == t (dt underflow)
//  - Optional dt_abort hard floor
//  - Optional fixed dt based on exact wave-speed (recommended for verification)
//
// Expected TOML keys (all optional; defaults provided):
//   nx, ny, Lx, Ly, CFL, t_end, gamma, reconstruction
//   rho0, p0, u0, v0, amp
//   results_csv, append
//   progress_interval, diag_interval
//   use_fixed_dt, dt_abort
//
// NOTE: Requires periodic BC support in your bc.hpp (BcType::Periodic).
//

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <limits>

#ifdef LS_USE_OPENMP
  #include <omp.h>
#endif

#include "ls/external/toml.hpp"
#include "ls/types.hpp"
#include "ls/mesh.hpp"
#include "ls/eos.hpp"
#include "ls/bc.hpp"
#include "ls/fv_update.hpp"
#include "ls/time_int.hpp"

using namespace ls;

static inline double wrap_periodic(double x, double L)
{
    double y = std::fmod(x, L);
    if (y < 0.0) y += L;
    return y;
}

struct ConvConfig2D {
    int nx{64}, ny{64};
    double Lx{1.0}, Ly{1.0};

    double CFL{0.2};
    double t_end{0.1};
    double gamma{1.4};
    std::string reconstruction{"WENO3"};

    // Exact-solution parameters
    double rho0{1.0};
    double p0{1.0};
    double u0{1.0};
    double v0{0.5};
    double amp{0.2};      // keep <= ~0.2 for positivity margin

    // Output
    std::string results_csv{"convergence_results.csv"};
    bool append{true};

    // Diagnostics / logging
    int progress_interval{200};  // print progress every N steps (0 disables)
    int diag_interval{200};      // print min/max diagnostics every N steps (0 disables)

    // Time-step control
    bool use_fixed_dt{true};     // recommended for convergence verification
    double dt_abort{1e-14};      // if dt < dt_abort, abort (<=0 disables)
};

static ConvConfig2D load_cfg(const std::string& path)
{
    ConvConfig2D c;
    auto tbl = toml::parse(path);

    c.nx  = toml::find_or(tbl, "nx", c.nx);
    c.ny  = toml::find_or(tbl, "ny", c.ny);
    c.Lx  = toml::find_or(tbl, "Lx", c.Lx);
    c.Ly  = toml::find_or(tbl, "Ly", c.Ly);

    c.CFL   = toml::find_or(tbl, "CFL", c.CFL);
    c.t_end = toml::find_or(tbl, "t_end", c.t_end);
    c.gamma = toml::find_or(tbl, "gamma", c.gamma);
    c.reconstruction = toml::find_or(tbl, "reconstruction", c.reconstruction);

    c.rho0 = toml::find_or(tbl, "rho0", c.rho0);
    c.p0   = toml::find_or(tbl, "p0",   c.p0);
    c.u0   = toml::find_or(tbl, "u0",   c.u0);
    c.v0   = toml::find_or(tbl, "v0",   c.v0);
    c.amp  = toml::find_or(tbl, "amp",  c.amp);

    c.results_csv = toml::find_or(tbl, "results_csv", c.results_csv);
    c.append      = toml::find_or(tbl, "append", c.append);

    c.progress_interval = toml::find_or(tbl, "progress_interval", c.progress_interval);
    c.diag_interval     = toml::find_or(tbl, "diag_interval", c.diag_interval);

    c.use_fixed_dt = toml::find_or(tbl, "use_fixed_dt", c.use_fixed_dt);
    c.dt_abort     = toml::find_or(tbl, "dt_abort", c.dt_abort);

    return c;
}

static inline double rho_exact(const ConvConfig2D& c, double x, double y, double t)
{
    const double xs = wrap_periodic(x - c.u0 * t, c.Lx);
    const double ys = wrap_periodic(y - c.v0 * t, c.Ly);

    const double sx = std::sin(2.0 * M_PI * xs / c.Lx);
    const double sy = std::sin(2.0 * M_PI * ys / c.Ly);

    return c.rho0 * (1.0 + c.amp * sx * sy);
}

static inline State2D exact_state(const ConvConfig2D& c, const EOSIdealGas& eos,
                                 double x, double y, double t)
{
    (void)eos;
    const double rho = rho_exact(c, x, y, t);
    const double u   = c.u0;
    const double v   = c.v0;
    const double p   = c.p0;

    const double E = p / (c.gamma - 1.0) + 0.5 * rho * (u*u + v*v);

    State2D U;
    U.rho  = rho;
    U.rhou = rho * u;
    U.rhov = rho * v;
    U.E    = E;
    return U;
}

static inline double compute_dt_cfl(const std::vector<State2D>& U,
                                   const Mesh2D& mesh,
                                   const EOSIdealGas& eos,
                                   double CFL,
                                   double t,
                                   double t_end)
{
    double smax = 1e-14;

#ifdef LS_USE_OPENMP
#pragma omp parallel for collapse(2) reduction(max:smax) schedule(static)
#endif
    for (int J = mesh.interior_j_start(); J <= mesh.interior_j_end(); ++J) {
        for (int I = mesh.interior_i_start(); I <= mesh.interior_i_end(); ++I) {
            const auto& Ui = U[mesh.index(I,J)];
            const double rho = std::max(Ui.rho, 1e-14);

            const double u = Ui.rhou / rho;
            const double v = Ui.rhov / rho;

            const double cs = eos.sound_speed(Ui); // must be finite
            if (!std::isfinite(u) || !std::isfinite(v) || !std::isfinite(cs)) continue;

            const double a = std::sqrt(u*u + v*v) + cs;
            if (std::isfinite(a)) smax = std::max(smax, a);
        }
    }

    const double dt_cfl = CFL * std::min(mesh.dx, mesh.dy) / smax;
    return std::min(dt_cfl, t_end - t);
}

struct Norms {
    double L1{0}, L2{0}, Linf{0};
};

static Norms compute_error_norms_rho(const std::vector<State2D>& U,
                                    const Mesh2D& mesh,
                                    const ConvConfig2D& c,
                                    double t)
{
    const int i0 = mesh.interior_i_start();
    const int i1 = mesh.interior_i_end();
    const int j0 = mesh.interior_j_start();
    const int j1 = mesh.interior_j_end();

    const double dxdy = mesh.dx * mesh.dy;
    const double area = c.Lx * c.Ly;

    double sum_abs = 0.0;
    double sum_sq  = 0.0;
    double max_abs = 0.0;

    for (int J = j0; J <= j1; ++J) {
        for (int I = i0; I <= i1; ++I) {
            const double x = mesh.xc(I);
            const double y = mesh.yc(J);

            const double re = rho_exact(c, x, y, t);
            const double rn = U[mesh.index(I,J)].rho;
            const double e  = rn - re;

            const double ae = std::abs(e);
            sum_abs += ae * dxdy;
            sum_sq  += e*e * dxdy;
            max_abs = std::max(max_abs, ae);
        }
    }

    Norms n;
    n.L1   = sum_abs / area;
    n.L2   = std::sqrt(sum_sq / area);
    n.Linf = max_abs;
    return n;
}

static void append_results_csv(const std::string& path,
                               bool append,
                               const std::string& header,
                               const std::string& line)
{
    bool need_header = false;

    if (!append) {
        need_header = true;
    } else {
        std::ifstream fin(path);
        need_header = !fin.good() || fin.peek() == std::ifstream::traits_type::eof();
    }

    std::ofstream fout(path, append ? std::ios::app : std::ios::out);
    if (!fout) throw std::runtime_error("Could not open results_csv: " + path);

    if (need_header) fout << header << "\n";
    fout << line << "\n";
}

// Diagnostics: scan for extreme values (helps pinpoint dt collapse / nonphysical states)
struct ScanDiag {
    double rho_min{ std::numeric_limits<double>::infinity() };
    double rho_max{ -std::numeric_limits<double>::infinity() };
    double p_min{ std::numeric_limits<double>::infinity() };
    double p_max{ -std::numeric_limits<double>::infinity() };
    double u_max{0.0};
    double v_max{0.0};
    bool   any_nan{false};
};

static ScanDiag scan_solution(const std::vector<State2D>& U,
                              const Mesh2D& mesh,
                              const EOSIdealGas& eos)
{
    ScanDiag d;

    const int i0 = mesh.interior_i_start();
    const int i1 = mesh.interior_i_end();
    const int j0 = mesh.interior_j_start();
    const int j1 = mesh.interior_j_end();

    for (int J = j0; J <= j1; ++J) {
        for (int I = i0; I <= i1; ++I) {
            const auto& Ui = U[mesh.index(I,J)];

            if (!std::isfinite(Ui.rho) || !std::isfinite(Ui.rhou) ||
                !std::isfinite(Ui.rhov) || !std::isfinite(Ui.E)) {
                d.any_nan = true;
                continue;
            }

            d.rho_min = std::min(d.rho_min, Ui.rho);
            d.rho_max = std::max(d.rho_max, Ui.rho);

            const double rho = std::max(Ui.rho, 1e-14);
            const double u = Ui.rhou / rho;
            const double v = Ui.rhov / rho;
            d.u_max = std::max(d.u_max, std::abs(u));
            d.v_max = std::max(d.v_max, std::abs(v));

            const double p = eos.pressure(Ui);
            if (!std::isfinite(p)) {
                d.any_nan = true;
            } else {
                d.p_min = std::min(d.p_min, p);
                d.p_max = std::max(d.p_max, p);
            }
        }
    }

    return d;
}

int main(int argc, char** argv)
{
    const std::string cfg_path = (argc > 1 ? argv[1] : "configs/convergence2d.toml");
    const auto c = load_cfg(cfg_path);

    EOSIdealGas eos;
    eos.gamma = c.gamma;

    ReconType recon = parse_recon_type(c.reconstruction);
    const int NG    = num_ghost_cells(c.reconstruction);

    // Periodic BCs are required for clean convergence of the advected exact solution.
    Bc2D bc;
    bc.left = bc.right = bc.bottom = bc.top = BcType::Periodic;

    Mesh2D mesh(c.nx, c.ny, c.Lx, c.Ly, NG);

    std::vector<State2D> U(mesh.nx_tot * mesh.ny_tot, zero_state2d());
    std::vector<State2D> rhs;

    // Initialize to exact state at t=0
    for (int J = 0; J < mesh.ny_tot; ++J) {
        for (int I = 0; I < mesh.nx_tot; ++I) {
            const double x = mesh.xc(I);
            const double y = mesh.yc(J);
            U[mesh.index(I,J)] = exact_state(c, eos, x, y, 0.0);
        }
    }
    apply_bc_2d(U, mesh, bc);

#ifdef LS_USE_OPENMP
#pragma omp parallel
{
#pragma omp single
    std::cout << "[info] OpenMP threads = " << omp_get_num_threads() << "\n";
}
#endif

    // Fixed dt based on exact characteristic speed (recommended for convergence verification)
    double dt_fixed = 0.0;
    if (c.use_fixed_dt) {
        const double rho_min = c.rho0 * (1.0 - c.amp);                // min density in exact solution
        const double cs0     = std::sqrt(c.gamma * c.p0 / rho_min);   // sound speed upper bound
        const double vmag    = std::sqrt(c.u0*c.u0 + c.v0*c.v0);
        const double smax    = vmag + cs0;

        dt_fixed = c.CFL * std::min(mesh.dx, mesh.dy) / smax;

        std::cout << "[info] use_fixed_dt=true  dt_fixed=" << std::scientific << dt_fixed
                  << "  (smax~" << smax << ")\n";
    } else {
        std::cout << "[info] use_fixed_dt=false (CFL dt computed from solution)\n";
    }

    // Integrate
    double t = 0.0;
    int steps = 0;
    double dt_sum = 0.0;

    const auto wall0 = std::chrono::steady_clock::now();

    while (t < c.t_end - 1e-15) {

        double dt = 0.0;
        if (c.use_fixed_dt) {
            dt = std::min(dt_fixed, c.t_end - t);
        } else {
            dt = compute_dt_cfl(U, mesh, eos, c.CFL, t, c.t_end);
        }

        if (!std::isfinite(dt) || dt <= 0.0) {
            std::cerr << "[fatal] Bad dt=" << dt << " at t=" << t << "\n";
            return 2;
        }

        // Abort if dt is so small that t cannot advance (double precision underflow)
        if (t + dt == t) {
            std::cerr << "[fatal] dt underflow: time cannot advance. t=" << t
                      << " dt=" << dt << "\n";
            // Print one scan to help debug what blew up
            const auto d = scan_solution(U, mesh, eos);
            std::cerr << "[fatal] scan: rho_min=" << d.rho_min << " rho_max=" << d.rho_max
                      << " p_min=" << d.p_min << " p_max=" << d.p_max
                      << " |u|_max=" << d.u_max << " |v|_max=" << d.v_max
                      << " any_nan=" << d.any_nan << "\n";
            return 3;
        }

        // Optional hard floor abort (good for catching collapse early)
        if (c.dt_abort > 0.0 && dt < c.dt_abort) {
            std::cerr << "[fatal] dt collapsed below dt_abort=" << c.dt_abort
                      << " at t=" << t << " dt=" << dt << "\n";
            const auto d = scan_solution(U, mesh, eos);
            std::cerr << "[fatal] scan: rho_min=" << d.rho_min << " rho_max=" << d.rho_max
                      << " p_min=" << d.p_min << " p_max=" << d.p_max
                      << " |u|_max=" << d.u_max << " |v|_max=" << d.v_max
                      << " any_nan=" << d.any_nan << "\n";
            return 4;
        }

        // Take one RK3 step
        advance_rk3_2d(U, rhs, mesh, eos, bc, recon, dt);

        t += dt;
        steps++;
        dt_sum += dt;

        // Progress print
        if (c.progress_interval > 0 &&
            (steps % c.progress_interval == 0 || t >= c.t_end - 1e-15)) {

            const double pct = 100.0 * t / c.t_end;

            std::cout << std::scientific
                      << "[progress] step=" << steps
                      << "  t=" << t << " / " << c.t_end
                      << "  (" << pct << "%)"
                      << "  dt=" << dt
                      << std::endl;
        }

        // Diagnostics print
        if (c.diag_interval > 0 &&
            (steps % c.diag_interval == 0 || t >= c.t_end - 1e-15)) {

            const auto d = scan_solution(U, mesh, eos);

            std::cout << std::scientific
                      << "[diag] rho_min=" << d.rho_min
                      << " rho_max=" << d.rho_max
                      << " p_min=" << d.p_min
                      << " p_max=" << d.p_max
                      << " |u|_max=" << d.u_max
                      << " |v|_max=" << d.v_max
                      << " any_nan=" << d.any_nan
                      << std::endl;

            // If something is clearly nonphysical, fail fast with a useful message.
            if (d.any_nan || !(d.rho_min > 0.0) || !(d.p_min > 0.0)) {
                std::cerr << "[fatal] Nonphysical state detected (nan or rho/p <= 0). Aborting.\n";
                return 5;
            }
        }
    }

    const auto wall1 = std::chrono::steady_clock::now();
    const double wall_s = std::chrono::duration<double>(wall1 - wall0).count();
    const double dt_mean = dt_sum / std::max(steps, 1);

    // Error norms on rho
    const Norms n = compute_error_norms_rho(U, mesh, c, t);

    const double h = std::max(mesh.dx, mesh.dy);

    const std::string header =
        "scheme,nx,ny,h,CFL,t_end,steps,dt_mean,wall_s,L1_rho,L2_rho,Linf_rho";

    std::ostringstream oss;
    oss.setf(std::ios::scientific);
    oss.precision(10);

    oss << c.reconstruction << ","
        << c.nx << "," << c.ny << ","
        << h << ","
        << c.CFL << ","
        << c.t_end << ","
        << steps << ","
        << dt_mean << ","
        << wall_s << ","
        << n.L1 << ","
        << n.L2 << ","
        << n.Linf;

    const std::string line = oss.str();

    std::cout << "[result] " << line << "\n";
    append_results_csv(c.results_csv, c.append, header, line);

    return 0;
}
