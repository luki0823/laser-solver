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

namespace ls {

inline void write_csv(const std::string& filename,
                      const Grid1D& grid,
                      const std::vector<State>& U,
                      double gamma)
{
    std::filesystem::create_directories("../data/out");

    std::ofstream file(filename);
    file << "x,rho,u,p,E\n";
    file << std::scientific << std::setprecision(6);

    for (size_t i = 0; i < U.size(); ++i) {
        const double rho = U[i].rho;
        const double u   = U[i].rhou / std::max(rho, 1e-14);
        const double p   = (gamma - 1.0) * (U[i].E - 0.5 * rho * u * u);
        file << grid.x[i] << "," << rho << "," << u << "," << p << "," << U[i].E << "\n";
    }
    file.close();
}

} // namespace ls