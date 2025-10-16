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

int main() {
    using namespace ls;
    const double gamma = 1.6, L = 1.0, CFL = 0.45;   // CFL can be a bit larger with RK3
    const int nx = 200, Nwrite = 50;
    const double t_end = 3.0e-3;
    const BCKind bc = BCKind::ReflectLeftCopy;

    Grid1D grid(nx, L);

    // IC (Sod-like)
    std::vector<State> U(nx);
    for (int i=0;i<nx;++i) {
        bool left = grid.x[i] < 0.5*L;
        double rho = left ? 1.0 : 0.125;
        double u   = 0.0;
        double p   = left ? 1e5 : 1e4;
        U[i] = prim_to_cons(rho, u, p, gamma);
    }

    double t = 0.0;
    int step_counter = 0;
    auto on_step = [&](double dt, double tnow, const std::vector<State>& Ucur){
        ++step_counter;
        if (step_counter % Nwrite == 0) {
            std::ostringstream fname;
            fname << "data/out/step_" << std::setw(4) << std::setfill('0') << step_counter << ".csv";
            write_csv(fname.str(), grid, Ucur, gamma);
            std::cout << "RK3 wrote " << fname.str() << " at t=" << tnow << " (dt=" << dt << ")\n";
        }
    };

    // no source yet → pass nullptr
    advance_ssprk3(U, t, t_end, gamma, grid.dx, CFL, bc, on_step, nullptr);

    std::cout << "Done. Final t=" << t << "\n";
    return 0;
}
