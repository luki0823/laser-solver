// =============================================================================
// File:        euler2d.cpp
// Purpose:     2D Euler solver driver (smooth advection test)
// Author:      Lucas Pierce
// Dependencies:
//   - ls/types.hpp
//   - ls/mesh.hpp
//   - ls/eos.hpp
//   - ls/flux.hpp
//   - ls/bc.hpp
//   - ls/recon.hpp
//   - ls/fv_update.hpp
//   - ls/time_int.hpp
//   - ls/io.hpp
//   - ls/external/toml.hpp
// =============================================================================

#include <iostream>
#include <vector>
#include <string>
#include <cmath>

#include "ls/external/toml.hpp"
#include "ls/types.hpp"
#include "ls/mesh.hpp"
#include "ls/eos.hpp"
#include "ls/flux.hpp"
#include "ls/bc.hpp"
#include "ls/recon.hpp"
#include "ls/fv_update.hpp"
#include "ls/time_int.hpp"
#include "ls/io.hpp"

using namespace ls;

// ------------------------------------------------------------
// Config for generic 2D Euler advection run
// ------------------------------------------------------------
struct Config2D {
    int    nx{128};
    int    ny{128};
    double Lx{1.0};
    double Ly{1.0};
    double CFL{0.4};
    double t_end{0.5};
    double gamma{1.4};
    std::string recon{"WENO3"};   // "WENO3" or "FirstOrder"

    // Advection IC parameters (smooth sinusoid)
    double rho0{1.0};  // base density
    double amp{0.1};   // amplitude of perturbation
    double p0{1.0};    // constant pressure
    double u0{0.5};    // constant x-velocity
    double v0{0.25};   // constant y-velocity
    double kx{1.0};    // mode in x
    double ky{1.0};    // mode in y

    // BCs
    // For a pure advection test you’ll usually want Periodic on all sides,
    // but we allow any type supported by BcType.
    std::string left{"Periodic"};
    std::string right{"Periodic"};
    std::string bottom{"Periodic"};
    std::string top{"Periodic"};

    // Output
    int         out_interval{20};
    std::string out_folder{"data/out2d"};
};

// ------------------------------------------------------------
// Load config from TOML
// ------------------------------------------------------------
Config2D load_config_2d(const std::string& path)
{
    Config2D c;
    auto tbl = toml::parse(path);

    // Top-level numerics
    c.nx    = toml::find_or(tbl, "nx",    c.nx);
    c.ny    = toml::find_or(tbl, "ny",    c.ny);
    c.Lx    = toml::find_or(tbl, "Lx",    c.Lx);
    c.Ly    = toml::find_or(tbl, "Ly",    c.Ly);
    c.CFL   = toml::find_or(tbl, "CFL",   c.CFL);
    c.t_end = toml::find_or(tbl, "t_end", c.t_end);
    c.gamma = toml::find_or(tbl, "gamma", c.gamma);
    c.recon = toml::find_or(tbl, "reconstruction", c.recon);

    // Advection IC block
    if (tbl.contains("advection_ic")) {
        auto ic = toml::find(tbl, "advection_ic");
        c.rho0 = toml::find_or(ic, "rho0", c.rho0);
        c.amp  = toml::find_or(ic, "amp",  c.amp);
        c.p0   = toml::find_or(ic, "p0",   c.p0);
        c.u0   = toml::find_or(ic, "u0",   c.u0);
        c.v0   = toml::find_or(ic, "v0",   c.v0);
        c.kx   = toml::find_or(ic, "kx",   c.kx);
        c.ky   = toml::find_or(ic, "ky",   c.ky);
    }

    // BC block
    if (tbl.contains("bc")) {
        auto bc_tbl = toml::find(tbl, "bc");
        c.left   = toml::find_or(bc_tbl, "left",   c.left);
        c.right  = toml::find_or(bc_tbl, "right",  c.right);
        c.bottom = toml::find_or(bc_tbl, "bottom", c.bottom);
        c.top    = toml::find_or(bc_tbl, "top",    c.top);
    }

    // Output block
    if (tbl.contains("output")) {
        auto out = toml::find(tbl, "output");
        c.out_interval = toml::find_or(out, "interval", c.out_interval);
        c.out_folder   = toml::find_or(out, "folder",   c.out_folder);
    }

    return c;
}

// ------------------------------------------------------------
// Initialize advecting smooth density bump
//
//  rho(x,y,0) = rho0 + amp * sin(2π kx x/Lx) * sin(2π ky y/Ly)
//  u, v, p    = constants
// ------------------------------------------------------------
void init_advection_2d(
    std::vector<State2D>& U,
    const Mesh2D& mesh,
    const Config2D& c,
    const EOSIdealGas& eos)
{
    const int nx_tot = mesh.nx_tot;
    const int ny_tot = mesh.ny_tot;

    for (int J = 0; J < ny_tot; ++J) {
        for (int I = 0; I < nx_tot; ++I) {

            double x = mesh.xc(I);
            double y = mesh.yc(J);

            double phase_x = 2.0 * M_PI * c.kx * x / c.Lx;
            double phase_y = 2.0 * M_PI * c.ky * y / c.Ly;

            double rho = c.rho0 + c.amp * std::sin(phase_x) * std::sin(phase_y);
            double p   = c.p0;
            double u   = c.u0;
            double v   = c.v0;

            double E = p / (eos.gamma - 1.0) + 0.5 * rho * (u*u + v*v);
            U[mesh.index(I,J)] = State2D{rho, rho*u, rho*v, E};
        }
    }
}

// ------------------------------------------------------------
// MAIN
// ------------------------------------------------------------
int main(int argc, char** argv)
{
    // Default config path (you can change this as needed)
    std::string cfg = (argc > 1 ? argv[1] : "configs/euler2d.toml");
    Config2D c = load_config_2d(cfg);

    // EOS
    EOSIdealGas eos;
    eos.gamma = c.gamma;

    // Reconstruction & ghost cells
    ReconType recon_type = parse_recon_type(c.recon);
    int NG = num_ghost_cells(c.recon);

    // Mesh
    Mesh2D mesh(c.nx, c.ny, c.Lx, c.Ly, NG);
    int nx_tot = mesh.nx_tot;
    int ny_tot = mesh.ny_tot;

    // State arrays
    std::vector<State2D> U(nx_tot*ny_tot, zero_state2d());
    std::vector<State2D> rhs(nx_tot*ny_tot, zero_state2d());

    // Boundary conditions
    Bc2D bc2d{
        parse_bc_type(c.left),
        parse_bc_type(c.right),
        parse_bc_type(c.bottom),
        parse_bc_type(c.top)
    };

    // Initial condition
    init_advection_2d(U, mesh, c, eos);
    apply_bc_2d(U, mesh, bc2d);

    // Output
    ensure_dir(c.out_folder);
    double t = 0.0;
    int step = 0;

    std::cout << "🚀 Starting 2D Euler run: "
              << "nx=" << c.nx << ", ny=" << c.ny
              << ", t_end=" << c.t_end
              << ", recon=" << c.recon << "\n";

    write_csv_2d(c.out_folder, step, t, mesh, U, eos);

    // Time loop
    while (t < c.t_end) {

        // Max wave speed for CFL
        double smax = 1e-12;
        for (int J = mesh.interior_j_start(); J <= mesh.interior_j_end(); ++J) {
            for (int I = mesh.interior_i_start(); I <= mesh.interior_i_end(); ++I) {
                const auto& Ui = U[mesh.index(I,J)];
                double rho = Ui.rho;
                double u   = Ui.rhou / rho;
                double v   = Ui.rhov / rho;
                double c_s = eos.sound_speed(Ui);
                double a   = std::sqrt(u*u + v*v) + c_s;
                smax = std::max(smax, a);
            }
        }

        double dt = std::min(
            c.CFL * std::min(mesh.dx, mesh.dy) / smax,
            c.t_end - t
        );

        advance_rk3_2d(U, rhs, mesh, eos, bc2d, recon_type, dt);

        t += dt;
        ++step;

        if (step % c.out_interval == 0 || t >= c.t_end) {
            write_csv_2d(c.out_folder, step, t, mesh, U, eos);
            std::cout << "📝 Wrote step " << step << " at t = " << t
                      << " (dt = " << dt << ")\n";
        }
    }

    std::cout << "✅ Done. Final time t = " << t << "\n";
    return 0;
}
