#include <iostream>
#include <fstream>
#include <string>
#include <filesystem>

#include "ls/types.hpp"
#include "ls/mesh.hpp"

namespace fs = std::filesystem;
using namespace ls;

int main() {
    // ------------------------------------------------------------
    // Parameters for testing (you can tweak these)
    // ------------------------------------------------------------
    std::string recon = "WENO3";  // or "WENO5", "FirstOrder", etc.
    int NG = num_ghost_cells(recon);

    int nx_1d = 100;
    double Lx_1d = 1.0;

    int nx_2d = 100;
    int ny_2d = 100;
    double Lx_2d = 1.0;
    double Ly_2d = 1.0;

    std::cout << "🔧 Reconstruction = " << recon << ", NG = " << NG << "\n";

    Mesh1D mesh1d(nx_1d, Lx_1d, NG);
    Mesh2D mesh2d(nx_2d, ny_2d, Lx_2d, Ly_2d, NG);

    // ------------------------------------------------------------
    // Ensure output directory exists
    // ------------------------------------------------------------
    fs::create_directories("data");

    // ------------------------------------------------------------
    // Write 1D mesh info
    // ------------------------------------------------------------
    {
        std::ofstream fout("data/mesh1d.csv");
        if (!fout) {
            std::cerr << "❌ Failed to open data/mesh1d.csv for writing\n";
            return 1;
        }

        fout << "I,logical_i,x,is_ghost\n";

        for (int I = 0; I < mesh1d.size(); ++I) {
            int logical_i = mesh1d.interior_i(I); // may be negative for ghosts
            bool is_ghost = (I < mesh1d.interior_start() ||
                             I > mesh1d.interior_end());

            // For ghosts, you might want logical_i = -1
            if (is_ghost) logical_i = -1;

            fout << I << ","
                 << logical_i << ","
                 << mesh1d.xc(I) << ","
                 << (is_ghost ? 1 : 0) << "\n";
        }

        std::cout << "✅ Wrote 1D mesh data to data/mesh1d.csv\n";
    }

    // ------------------------------------------------------------
    // Write 2D mesh info
    // ------------------------------------------------------------
    {
        std::ofstream fout("data/mesh2d.csv");
        if (!fout) {
            std::cerr << "❌ Failed to open data/mesh2d.csv for writing\n";
            return 1;
        }

        fout << "I,J,logical_i,logical_j,x,y,is_ghost\n";

        for (int J = 0; J < mesh2d.size_y(); ++J) {
            for (int I = 0; I < mesh2d.size_x(); ++I) {
                bool is_ghost =
                    (I < mesh2d.interior_i_start() || I > mesh2d.interior_i_end() ||
                     J < mesh2d.interior_j_start() || J > mesh2d.interior_j_end());

                int logical_i = is_ghost ? -1 : (I - mesh2d.ng);
                int logical_j = is_ghost ? -1 : (J - mesh2d.ng);

                fout << I << ","
                     << J << ","
                     << logical_i << ","
                     << logical_j << ","
                     << mesh2d.xc(I) << ","
                     << mesh2d.yc(J) << ","
                     << (is_ghost ? 1 : 0) << "\n";
            }
        }

        std::cout << "✅ Wrote 2D mesh data to data/mesh2d.csv\n";
    }

    return 0;
}
