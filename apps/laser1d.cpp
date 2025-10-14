/*
==============================================================================
 File:        laser1d.cpp
 Purpose:     Evolve a simple 1D Euler shock tube with Forward–Euler.
 Description:
   - Grid: uniform [0,1]
   - IC:   left (rho=1,  u=0, p=1e5), right (rho=0.125, u=0, p=1e4)
   - BC:   left reflective wall, right open (copy)
   - Flux: Rusanov
   - Time: Forward–Euler with CFL dt

 Notes:
   - This is intentionally short and low-accuracy; SSP-RK3 comes next.
   - Use a small t_end so Forward–Euler remains stable enough to visualize.
==============================================================================
*/
#include "ls/types.hpp"
#include "ls/mesh1d.hpp"
#include "ls/eos.hpp"
#include "ls/recon.hpp"
#include "ls/time_init.hpp"

#include <vector>
#include <iostream>
#include <iomanip>

int main(){
    using namespace ls;

    // --- Parameters ---
    const double gamma = 1.4;
    const int    nx    = 200;
    const double L     = 1.0;
    const double CFL   = 0.3;
    const double t_end = 2.0e-3;   // keep short for Forward–Euler demo
    const BCKind bc    = BCKind::ReflectLeftCopy; // wall at x=0, open at x=L

    // --- Grid ---
    Grid1D grid(nx, L);

    // --- Initial condition (Sod-like) ---
    std::vector<State> U(nx);
    for (int i = 0; i < nx; ++i) {
        const bool left = (grid.x[i] < 0.5 * L);
        const double rho = left ? 1.0   : 0.125;
        const double u   = 0.0;
        const double p   = left ? 1e5   : 1e4;
        U[i] = prim_to_cons(rho, u, p, gamma);
    }

    // --- Time loop (Forward–Euler) ---
    double t = 0.0;
    int step = 0;
    while (t < t_end) {
        const double dt = step_euler(U, gamma, grid.dx, CFL, bc, t, nullptr);
        t += dt;
        ++step;
        if (step % 50 == 0) {
            // Quick diagnostics: min/max density and pressure
            double rmin = 1e99, rmax = -1e99, pmin = 1e99, pmax = -1e99;
            for (const auto& s : U) {
                const double rho = s.rho;
                const double u   = s.rhou / std::max(rho, 1e-14);
                const double p   = (gamma - 1.0) * (s.E - 0.5 * rho * u * u);
                rmin = std::min(rmin, rho); rmax = std::max(rmax, rho);
                pmin = std::min(pmin, p  ); pmax = std::max(pmax, p  );
            }
            std::cout << std::fixed << std::setprecision(6)
                      << "t=" << t << "  dt=" << dt
                      << "  rho[min,max]=[" << rmin << "," << rmax << "]"
                      << "  p[min,max]=["   << pmin << "," << pmax << "]\n";
        }
    }

    std::cout << "Done. Final time t=" << t << " after " << step << " steps.\n";
    return 0;
}
