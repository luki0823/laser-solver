#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>

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
// Config for 2D error test
// ------------------------------------------------------------
struct ErrorConfig2D {
    int    nx{64};
    int    ny{64};
    double Lx{1.0};
    double Ly{1.0};
    double CFL{0.4};
    double t_end{0.1};
    double gamma{1.4};
    std::string recon{"WENO3"};   // "WENO3" or "FirstOrder"

    // Advection IC parameters
    double rho0{1.0};  // base density
    double amp{0.1};   // amplitude of perturbation
    double p0{1.0};    // constant pressure
    double u0{0.5};    // constant x-velocity
    double v0{0.25};   // constant y-velocity
    double kx{1.0};    // mode in x
    double ky{1.0};    // mode in y

    // BCs (use Outflow for this test; Periodic if you add it)
    std::string left{"Outflow"};
    std::string right{"Outflow"};
    std::string bottom{"Outflow"};
    std::string top{"Outflow"};

    // Output (optional CSV)
    std::string out_folder{"data/error2d"};
    bool write_final_csv{true};
};

ErrorConfig2D load_error_config_2d(const std::string& path)
{
    ErrorConfig2D c;
    auto tbl = toml::parse(path);

    c.nx    = toml::find_or(tbl, "nx",    c.nx);
    c.ny    = toml::find_or(tbl, "ny",    c.ny);
    c.Lx    = toml::find_or(tbl, "Lx",    c.Lx);
    c.Ly    = toml::find_or(tbl, "Ly",    c.Ly);
    c.CFL   = toml::find_or(tbl, "CFL",   c.CFL);
    c.t_end = toml::find_or(tbl, "t_end", c.t_end);
    c.gamma = toml::find_or(tbl, "gamma", c.gamma);
    c.recon = toml::find_or(tbl, "reconstruction", c.recon);

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

    if (tbl.contains("bc")) {
        auto bc_tbl = toml::find(tbl, "bc");
        c.left   = toml::find_or(bc_tbl, "left",   c.left);
        c.right  = toml::find_or(bc_tbl, "right",  c.right);
        c.bottom = toml::find_or(bc_tbl, "bottom", c.bottom);
        c.top    = toml::find_or(bc_tbl, "top",    c.top);
    }

    if (tbl.contains("output")) {
        auto out = toml::find(tbl, "output");
        c.out_folder      = toml::find_or(out, "folder",          c.out_folder);
        c.write_final_csv = toml::find_or(out, "write_final_csv", c.write_final_csv);
    }

    return c;
}

// ------------------------------------------------------------
// Initialize advecting smooth density bump
// ------------------------------------------------------------
void init_advection_2d(
    std::vector<State2D>& U,
    const Mesh2D& mesh,
    const ErrorConfig2D& c,
    const EOSIdealGas& eos)
{
    const int nx_tot = mesh.nx_tot;
    const int ny_tot = mesh.ny_tot;

    for (int J = 0; J < ny_tot; ++J) {
        for (int I = 0; I < nx_tot; ++I) {

            double x = mesh.xc(I);
            double y = mesh.yc(J);

            // Initial phase (t=0)
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
// Exact solution at time t for this advection problem
// ------------------------------------------------------------
inline double wrap_periodic(double x, double L)
{
    // Wrap x into [0, L) using modulo
    double w = std::fmod(x, L);
    if (w < 0.0) w += L;
    return w;
}

double exact_rho(
    double x, double y, double t,
    const ErrorConfig2D& c)
{
    // Trace back along characteristic: (x0, y0) = (x - u0 t, y - v0 t)
    double x0 = wrap_periodic(x - c.u0 * t, c.Lx);
    double y0 = wrap_periodic(y - c.v0 * t, c.Ly);

    double phase_x = 2.0 * M_PI * c.kx * x0 / c.Lx;
    double phase_y = 2.0 * M_PI * c.ky * y0 / c.Ly;

    return c.rho0 + c.amp * std::sin(phase_x) * std::sin(phase_y);
}

// ------------------------------------------------------------
// MAIN: 2D error test
// ------------------------------------------------------------
int main(int argc, char** argv)
{
    std::string cfg = (argc > 1) ? argv[1] : "configs/error2d_64.toml";
    ErrorConfig2D c = load_error_config_2d(cfg);

    std::cout << "2D error test config: " << cfg << "\n";
    std::cout << "  nx=" << c.nx << ", ny=" << c.ny
              << ", Lx=" << c.Lx << ", Ly=" << c.Ly
              << ", CFL=" << c.CFL
              << ", t_end=" << c.t_end
              << ", recon=" << c.recon << "\n";

    // EOS
    EOSIdealGas eos;
    eos.gamma = c.gamma;

    // Reconstruction + ghost cells
    ReconType recon_type = parse_recon_type(c.recon);
    int NG = num_ghost_cells(c.recon);

    // Mesh
    Mesh2D mesh(c.nx, c.ny, c.Lx, c.Ly, NG);
    int nx_tot = mesh.nx_tot;
    int ny_tot = mesh.ny_tot;

    std::vector<State2D> U(nx_tot*ny_tot, zero_state2d());
    std::vector<State2D> rhs(nx_tot*ny_tot, zero_state2d());

    // BCs
    Bc2D bc2d{
        parse_bc_type(c.left),
        parse_bc_type(c.right),
        parse_bc_type(c.bottom),
        parse_bc_type(c.top)
    };

    // Initial condition
    init_advection_2d(U, mesh, c, eos);
    apply_bc_2d(U, mesh, bc2d);

    // Time integration
    double t = 0.0;
    int step = 0;

    if (c.write_final_csv) {
        ensure_dir(c.out_folder);
    }

    while (t < c.t_end) {
        // Max wave speed (for CFL)
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
        
        if (step % 10 == 0) {
            std::cout << "Step = " << step << ", t  = " << t << ", dt = " << dt << "\n";
        }
        t += dt;


        ++step;
    }

    std::cout << "Finished time integration at t=" << t
              << " in " << step << " steps.\n";

    // Optional CSV
    if (c.write_final_csv) {
        write_csv_2d(c.out_folder, step, t, mesh, U, eos);
        std::cout << "Wrote final solution to " << c.out_folder << "\n";
    }

    // --------------------------------------------------------
    // Compute error vs exact solution (density only)
    // --------------------------------------------------------
    double L1 = 0.0;
    double L2 = 0.0;
    double Linf = 0.0;
    std::size_t N = 0;

    for (int J = mesh.interior_j_start(); J <= mesh.interior_j_end(); ++J) {
        for (int I = mesh.interior_i_start(); I <= mesh.interior_i_end(); ++I) {
            double x = mesh.xc(I);
            double y = mesh.yc(J);

            double rho_num = U[mesh.index(I,J)].rho;
            double rho_ex  = exact_rho(x, y, t, c);

            double diff = rho_num - rho_ex;
            double adiff = std::fabs(diff);

            L1 += adiff;
            L2 += diff * diff;
            Linf = std::max(Linf, adiff);
            ++N;
        }
    }

    L1 /= static_cast<double>(N);
    L2 = std::sqrt(L2 / static_cast<double>(N));

    std::cout << "Error norms (density):\n";
    std::cout << "  L1   = " << L1  << "\n";
    std::cout << "  L2   = " << L2  << "\n";
    std::cout << "  Linf = " << Linf << "\n";

    return 0;
}
