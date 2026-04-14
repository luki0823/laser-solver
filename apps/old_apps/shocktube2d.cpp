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
// 2-D Config structure
// ------------------------------------------------------------
struct Config2D {
    int    nx{200};
    int    ny{200};
    double Lx{1.0};
    double Ly{1.0};
    double CFL{0.1};
    double t_end{0.4};
    double gamma{1.4};
    std::string recon{"WENO3"};

    // BCs
    std::string left{"Wall"};
    std::string right{"Wall"};
    std::string bottom{"Wall"};
    std::string top{"Wall"};

    // IC block
    double rho0{1.0}, p0{1.0};
    double rho1{1.0}, p1{10.0};
    double cx{0.5}, cy{0.5}, radius{0.1};

    // Output
    int out_interval{10};
    std::string out_folder{"data/out2d"};
};

// ------------------------------------------------------------
// Load the config from TOML
// ------------------------------------------------------------
Config2D load_config_2d(const std::string& path)
{
    Config2D c;
    auto tbl = toml::parse(path);

    c.nx    = toml::find_or(tbl, "nx",    c.nx);
    c.ny    = toml::find_or(tbl, "ny",    c.ny);
    c.Lx    = toml::find_or(tbl, "Lx",    c.Lx);
    c.Ly    = toml::find_or(tbl, "Ly",    c.Ly);
    c.CFL   = toml::find_or(tbl, "CFL",   c.CFL);
    c.t_end = toml::find_or(tbl, "t_end", c.t_end);
    c.gamma = toml::find_or(tbl, "gamma", c.gamma);
    c.recon = toml::find_or(tbl, "reconstruction", c.recon);

    // BC block
    if (tbl.contains("bc")) {
        auto bc_tbl = toml::find(tbl, "bc");
        c.left   = toml::find_or(bc_tbl, "left",   c.left);
        c.right  = toml::find_or(bc_tbl, "right",  c.right);
        c.bottom = toml::find_or(bc_tbl, "bottom", c.bottom);
        c.top    = toml::find_or(bc_tbl, "top",    c.top);
    }

    // IC block
    if (tbl.contains("initial_condition")) {
        auto ic = toml::find(tbl, "initial_condition");
        c.rho0   = toml::find_or(ic, "rho0",   c.rho0);
        c.p0     = toml::find_or(ic, "p0",     c.p0);
        c.rho1   = toml::find_or(ic, "rho1",   c.rho1);
        c.p1     = toml::find_or(ic, "p1",     c.p1);
        c.cx     = toml::find_or(ic, "cx",     c.cx);
        c.cy     = toml::find_or(ic, "cy",     c.cy);
        c.radius = toml::find_or(ic, "radius", c.radius);
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
// Initialize a circular blast region
// ------------------------------------------------------------
void init_blast_2d(
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

            double dx = x - c.cx * c.Lx;
            double dy = y - c.cy * c.Ly;
            double r  = std::sqrt(dx*dx + dy*dy);

            double rho, p, u, v;
            if (r < c.radius) {
                rho = c.rho1;
                p   = c.p1;
            } else {
                rho = c.rho0;
                p   = c.p0;
            }

            u = 0.0;
            v = 0.0;

            double E = p/(eos.gamma - 1.0) + 0.5*rho*(u*u + v*v);
            U[mesh.index(I,J)] = State2D{rho, rho*u, rho*v, E};
        }
    }
}

// ------------------------------------------------------------
// MAIN
// ------------------------------------------------------------
int main(int argc, char** argv)
{
    std::string cfg = (argc > 1 ? argv[1] : "configs/shocktube2d.toml");
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

    // IC
    init_blast_2d(U, mesh, c, eos);
    apply_bc_2d(U, mesh, bc2d);

    // Output
    ensure_dir(c.out_folder);
    double t = 0.0;
    int step = 0;
    write_csv_2d(c.out_folder, step, t, mesh, U, eos);

    // Time loop
    while (t < c.t_end) {

        // Max wave speed
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

        double dt = std::min(c.CFL * std::min(mesh.dx, mesh.dy) / smax,
                             c.t_end - t);

        advance_rk3_2d(U, rhs, mesh, eos, bc2d, recon_type, dt);

        t += dt;
        ++step;

        if (step % c.out_interval == 0 || t >= c.t_end) {
            write_csv_2d(c.out_folder, step, t, mesh, U, eos);
            std::cout << "Wrote step " << step << " at t=" << t << "\n";
        }
    }

    std::cout << "Done. Final time: " << t << "\n";
    return 0;
}
