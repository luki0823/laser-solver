/*
================================================================================
 File:        apps/shocktube.cpp
 Purpose:     1-D Euler shock-tube driver using SSP-RK3 time integration.
 Author:      You :)
 Description:
   - Reads runtime parameters from a TOML file (see example below).
   - Initializes a Sod-like discontinuity.
   - Advances with SSP-RK3 (third-order TVD) in time.
   - Writes CSV snapshots to data/out/ for Python visualization.

 Math recap:
   Semi-discrete finite-volume form (cell average U_i):
     dU_i/dt = -(F_{i+1/2}^* - F_{i-1/2}^*) / dx

   SSP-RK3 (Shu–Osher):
     U^(1)   = U^n + dt * L(U^n)
     U^(2)   = 3/4 U^n + 1/4( U^(1) + dt * L(U^(1)) )
     U^(n+1) = 1/3 U^n + 2/3( U^(2) + dt * L(U^(2)) )

   where L(U) = - (F^*_{i+1/2} - F^*_{i-1/2}) / dx  (+ source later)

 Notes:
   • Reconstruction is first-order (piecewise-constant) right now.
     Later you can add WENO3 in recon.hpp and select via config.
   • Riemann solver: Rusanov (LLF); upgrade to HLLC later if desired.
================================================================================
*/
#include "ls/external/toml.hpp"  // header-only parser (downloaded by CMake)
#include "ls/types.hpp"
#include "ls/mesh1d.hpp"
#include "ls/eos.hpp"
#include "ls/recon.hpp"          // first-order reconstruction now; WENO later
#include "ls/riemann.hpp"        // Rusanov flux
#include "ls/time_int.hpp"       // cfl_dt, flux_divergence, advance_ssprk3
#include "ls/io.hpp"             // write_csv()
#include <iostream>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

using namespace ls;

int main(int argc, char** argv)
{
    // ---------------------------
    // 0) Parse CLI and TOML file
    // ---------------------------
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " config.toml\n";
        return 1;
    }

    toml::value cfg;
    try {
        cfg = toml::parse(argv[1]);
    } catch (const std::exception& e) {
        std::cerr << "Error parsing TOML config: " << e.what() << "\n";
        return 1;
    }

    // ---------------------------
    // 1) Simulation parameters
    // ---------------------------
    const int    nx    = toml::find_or(cfg, "nx",    200);
    const double L     = toml::find_or(cfg, "L",     1.0);
    const double CFL   = toml::find_or(cfg, "CFL",   0.45);
    const double t_end = toml::find_or(cfg, "t_end", 3.0e-3);
    const double gamma = toml::find_or(cfg, "gamma", 1.4);
    std::string recon_str = toml::find_or(cfg, "reconstruction", std::string("FirstOrder"));
    Recon recon = Recon::FirstOrder;
    if (recon_str == "WENO3") recon = Recon::WENO3;


    std::string bc_str = toml::find_or(cfg, "bc", std::string("ReflectLeftCopy"));
    BCKind bc = BCKind::CopyEnds;
    if (bc_str == "ReflectLeftCopy") bc = BCKind::ReflectLeftCopy;
    // (When you add more BCs, extend the mapping above.)

    // Optional future switch for reconstruction (kept for later WENO3):
    // std::string recon_str = toml::find_or(cfg, "reconstruction", std::string("FirstOrder"));

    // Output settings
    auto out = toml::find(cfg, "output");
    const int    Nwrite = toml::find_or(out, "interval", 50);
    const std::string outdir = toml::find_or(out, "folder", std::string("data/out"));

    // ---------------------------
    // 2) Build grid and IC
    // ---------------------------
    Grid1D grid(nx, L);
    std::vector<State> U(nx);

    auto ic = toml::find(cfg, "initial_condition");
    const std::string ic_type = toml::find_or(ic, "type", std::string("sod"));

    if (ic_type == "sod") {
        const double rhoL   = toml::find_or(ic, "rho_L",   1.0);
        const double pL     = toml::find_or(ic, "p_L",     100000.0);
        const double rhoR   = toml::find_or(ic, "rho_R",   0.125);
        const double pR     = toml::find_or(ic, "p_R",     10000.0);
        const double xSplit = toml::find_or(ic, "x_split", 0.5); // fraction of L

        for (int i = 0; i < nx; ++i) {
            const bool left = (grid.x[i] < xSplit * L);
            const double rho = left ? rhoL : rhoR;
            const double u   = 0.0;
            const double p   = left ? pL   : pR;
            U[i] = prim_to_cons(rho, u, p, gamma);
        }
    } else {
        std::cerr << "Unknown initial_condition.type='" << ic_type << "'.\n";
        return 1;
    }

    // ---------------------------
    // 3) Time integration (SSP-RK3)
    // ---------------------------
    double t = 0.0;
    int step_counter = 0;

    // Optional source term hook (laser later). For now: none.
    std::function<void(double,double,std::vector<State>&)> post_source = nullptr;

    // Per-step callback: write CSV every Nwrite steps
    auto on_step = [&](double dt, double tnow, const std::vector<State>& Ucur)
    {
        ++step_counter;
        if (step_counter % Nwrite == 0) {
            std::ostringstream fname;
            fname << outdir << "/step_" << std::setw(4) << std::setfill('0') << step_counter << ".csv";
            write_csv(fname.str(), grid, Ucur, gamma);
            std::cout << "Wrote " << fname.str()
                      << " at t=" << std::scientific << std::setprecision(6) << tnow
                      << " (dt=" << dt << ")\n";
        }
    };

    // Advance solution to t_end using SSP-RK3
    advance_ssprk3(U, t, t_end, gamma, grid.dx, CFL, bc, recon, on_step, post_source);

    // Final write (ensure you always have the last frame)
    {
        std::ostringstream fname;
        fname << outdir << "/step_" << std::setw(4) << std::setfill('0') << (step_counter + 1) << ".csv";
        write_csv(fname.str(), grid, U, gamma);
        std::cout << "Final write: " << fname.str()
                  << " at t=" << std::scientific << std::setprecision(6) << t << "\n";
    }

    std::cout << "Shock tube complete. Final time t=" << std::scientific << t << "\n";
    return 0;
}
