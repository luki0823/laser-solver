/*
==============================================================================
 File:        io.hpp
 Purpose:     Output utilities for 1D and 2D Euler solvers
 Author:      Lucas Pierce
 Description:
   - Writes full primitive + conservative fields for visualization
   - Supports 1D and 2D
   - Automatically creates output directories
==============================================================================
*/
#pragma once

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <string>

#include "ls/types.hpp"
#include "ls/mesh.hpp"
#include "ls/eos.hpp"

namespace ls {

// -----------------------------------------------------------------------------
// Directory creation helper
// -----------------------------------------------------------------------------
inline void ensure_dir(const std::string& folder) {
    std::filesystem::create_directories(folder);
}

// -----------------------------------------------------------------------------
// 1D CSV output
//   Writes: x, rho, u, p, E, rhou, H
// -----------------------------------------------------------------------------
inline void write_csv_1d(
    const std::string& folder,
    int step,
    double t,
    const Mesh1D& mesh,
    const std::vector<State1D>& U,
    const EOSIdealGas& eos)
{
    ensure_dir(folder);

    char name[256];
    std::snprintf(name, sizeof(name), "%s/step_%04d.csv",
                  folder.c_str(), step);

    std::ofstream f(name);
    if (!f) {
        std::cerr << "❌ Failed to write file: " << name << "\n";
        return;
    }

    f << std::setprecision(15);

    f << "# time = " << t << "\n";
    f << "x,rho,u,p,E,rhou,H\n";

    int I0 = mesh.interior_start();
    int I1 = mesh.interior_end();

    for (int I = I0; I <= I1; ++I) {
        const State1D& Ui = U[I];
        double rho,u,p; eos.cons_to_prim(Ui, rho,u,p);
        double H = (Ui.E + p) / rho;

        f << mesh.xc(I) << ","
          << rho << ","
          << u << ","
          << p << ","
          << Ui.E << ","
          << Ui.rhou << ","
          << H << "\n";
    }
}

// -----------------------------------------------------------------------------
// 2D CSV output
//   Writes: x, y, rho, u, v, p, E, rhou, rhov, H
//
// NOTE: Only interior cells written, consistent with WENO ghost structure.
// -----------------------------------------------------------------------------
inline void write_csv_2d(
    const std::string& folder,
    int step,
    double t,
    const Mesh2D& mesh,
    const std::vector<State2D>& U,
    const EOSIdealGas& eos)
{
    ensure_dir(folder);

    char name[256];
    std::snprintf(name, sizeof(name), "%s/step_%04d.csv",
                  folder.c_str(), step);

    std::ofstream f(name);
    if (!f) {
        std::cerr << "❌ Failed to write file: " << name << "\n";
        return;
    }

    f << std::setprecision(15);

    f << "# time = " << t << "\n";
    f << "x,y,rho,u,v,p,E,rhou,rhov,H\n";

    int i0 = mesh.interior_i_start();
    int i1 = mesh.interior_i_end();
    int j0 = mesh.interior_j_start();
    int j1 = mesh.interior_j_end();

    for (int J = j0; J <= j1; ++J) {
        for (int I = i0; I <= i1; ++I) {
            int idx = mesh.index(I,J);
            const State2D& Ui = U[idx];

            double rho,u,v,p;
            eos.cons_to_prim(Ui, rho,u,v,p);
            double H = (Ui.E + p) / rho;

            f << mesh.xc(I) << ","
              << mesh.yc(J) << ","
              << rho << ","
              << u   << ","
              << v   << ","
              << p   << ","
              << Ui.E << ","
              << Ui.rhou << ","
              << Ui.rhov << ","
              << H << "\n";
        }
    }
}

} // namespace ls
