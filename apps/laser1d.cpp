#include "ls/types.hpp"
#include "ls/mesh1d.hpp"
#include "ls/eos.hpp"
#include "ls/recon.hpp"
#include "ls/time_init.hpp"
#include "ls/io.hpp"

#include <vector>
#include <iostream>
#include <iomanip>
#include <sstream>

int main(){
    using namespace ls;

    // --- Parameters ---
    const double gamma = 1.4;
    const int    nx    = 200;
    const double L     = 1.0;
    const double CFL   = 0.3;
    const double t_end = 3.0e-3;
    const BCKind bc    = BCKind::ReflectLeftCopy;
    const int    Nwrite = 50;  // write interval (in steps)

    Grid1D grid(nx, L);

    // --- Initial condition ---
    std::vector<State> U(nx);
    for (int i = 0; i < nx; ++i) {
        const bool left = (grid.x[i] < 0.5 * L);
        const double rho = left ? 1.0   : 0.125;
        const double u   = 0.0;
        const double p   = left ? 1e5   : 1e4;
        U[i] = prim_to_cons(rho, u, p, gamma);
    }

    // --- Time loop ---
    double t = 0.0;
    int step = 0;

    while (t < t_end) {
        double dt = step_euler(U, gamma, grid.dx, CFL, bc, t, nullptr);
        t += dt;
        ++step;

        if (step % Nwrite == 0) {
            std::ostringstream fname;
            fname << "data/out/step_" << std::setw(4) << std::setfill('0') << step << ".csv";
            write_csv(fname.str(), grid, U, gamma);
            std::cout << "Wrote " << fname.str() << " at t=" << t << "\n";
        }
    }

    std::cout << "Done. Final time t=" << t << ", steps=" << step << "\n";
    return 0;
}
