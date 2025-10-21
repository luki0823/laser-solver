/*
==============================================================================
 File:        io.hpp
 Purpose:     Simple CSV output for 1D Euler solver results.

 Output format:
   x, rho, u, p, E

 Writes one file per output step to: data/out/step_####.csv
==============================================================================
*/
#pragma once
#include "ls/types.hpp"
#include "ls/eos.hpp"
#include "ls/mesh1d.hpp"
#include <fstream>
#include <iomanip>
#include <filesystem>
#include <iostream>


namespace ls {

inline void write_csv(const std::string& fname,
                      const Grid1D& grid,
                      const std::vector<State>& U,
                      double gamma)
{
    using namespace ls;
    const int nx_phys = grid.nx;
    const int NG = ls::NG;

    std::ofstream f(fname);
    if (!f.is_open()) {
        std::cerr << "Error: could not open " << fname << " for writing.\n";
        return;
    }

    f << "x,rho,u,p,E\n";
    f << std::scientific << std::setprecision(8);

    // loop only over physical cells, skipping ghosts
    for (int i = 0; i < nx_phys; ++i)
    {
        int j = i + NG;  // map physical -> full array index

        double rho = std::max(U[j].rho, 1e-12);
        double u   = U[j].rhou / rho;
        double E   = U[j].E;
        double p   = std::max((gamma - 1.0)*(E - 0.5*rho*u*u), 1e-12);

        f << grid.x[i] << "," << rho << "," << u << "," << p << "," << E << "\n";
    }

    f.close();
}

} // namespace ls
