#include <iostream>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <vector>
#include <cmath>
#include <filesystem>

#include "ls/types2d.hpp"
#include "ls/mesh2d.hpp"
#include "ls/eos2d.hpp"
#include "ls/time_int2d.hpp"
#include "ls/bc2d.hpp"

using namespace ls;

// ------------------------------------------------------------
// Write one 2D snapshot to CSV (for Python visualization)
// ------------------------------------------------------------
static void write_csv2d(const std::string& fname,
                        const Grid2D& grid,
                        const std::vector<State2D>& U,
                        double gamma)
{
    int nx = grid.nx;
    int ny = grid.ny;
    int nx_tot = nx + 2*NG2D;

    std::ofstream f(fname);
    if (!f) {
        std::cerr << "❌ Could not open " << fname << " for writing.\n";
        return;
    }

    f << "x,y,rho,u,v,p,E\n";

    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            int ii = i + NG2D;
            int jj = j + NG2D;
            int c  = idx2D(ii, jj, nx_tot);

            double rho,u,v,p;
            cons_to_prim(U[c], gamma, rho, u, v, p);

            f << grid.x[i] << "," << grid.y[j] << ","
              << rho << "," << u << "," << v << "," << p << "," << U[c].E << "\n";
        }
    }
}

// ------------------------------------------------------------
// Main: 2D Euler WENO3 test problem
// ------------------------------------------------------------
int main()
{
    // -----------------------------
    // Problem / grid parameters
    // -----------------------------
    int    nx    = 128;     // cells in x
    int    ny    = 128;     // cells in y
    double Lx    = 1.0;     // domain length in x
    double Ly    = 1.0;     // domain length in y
    double gamma = 1.4;
    double CFL   = 0.1;
    double t_end = 0.5;

    // Output settings
    std::string out_dir = "data/out";
    int output_interval = 5;   // write every N RK steps

    // -----------------------------
    // Create grid
    // -----------------------------
    Grid2D grid(nx, ny, Lx, Ly);

    int nx_tot = nx + 2*NG2D;
    int ny_tot = ny + 2*NG2D;

    // Allocate solution (with ghosts)
    std::vector<State2D> U(nx_tot * ny_tot);

    // -----------------------------
    // Initial condition:
    //  uniform background flow + Gaussian density bump
    // -----------------------------
    double rho0 = 1.0;
    double p0   = 1.0;
    double u0   = 1.0;   // base velocity x
    double v0   = 0.5;   // base velocity y

    double cx    = 0.5 * Lx;
    double cy    = 0.5 * Ly;
    double sigma2 = 0.01;   // controls width of Gaussian bump

    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            double x = grid.x[i];
            double y = grid.y[j];

            double r2   = (x - cx)*(x - cx) + (y - cy)*(y - cy);
            double bump = 0.2 * std::exp(-r2 / sigma2);

            double rho = rho0 + bump;
            double u   = u0;
            double v   = v0;
            double p   = p0;

            double E = p / (gamma - 1.0) + 0.5 * rho * (u*u + v*v);

            int ii = i + NG2D;
            int jj = j + NG2D;
            int c  = idx2D(ii, jj, nx_tot);

            U[c] = {rho, rho*u, rho*v, E};
        }
    }

    // -----------------------------
    // Ensure output directory exists
    // -----------------------------
    std::filesystem::create_directories(out_dir);

    // -----------------------------
    // Time integration (SSP-RK3 + WENO3)
    // -----------------------------
    double t    = 0.0;
    int    step = 0;

    auto on_step = [&](double dt, double tn, const std::vector<State2D>& Ucur)
    {
        if (step % output_interval == 0) {
            std::ostringstream oss;
            oss << out_dir << "/step_"
                << std::setw(4) << std::setfill('0') << step << ".csv";

            std::cout << "📝 Writing " << oss.str()
                      << " at t = " << tn << " (dt = " << dt << ")\n";

            write_csv2d(oss.str(), grid, Ucur, gamma);
        }
        ++step;
    };

    std::cout << "🚀 Starting 2D WENO run: nx=" << nx
              << ", ny=" << ny << ", t_end=" << t_end << std::endl;

    advance_ssprk3_2d(
        U, t, t_end, gamma, grid, CFL,
        BC2D::Periodic,
        on_step
    );

    std::cout << "✅ Done. Final time t = " << t << std::endl;
    return 0;
}
