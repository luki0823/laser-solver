/*
==============================================================================
 File:        mesh.hpp
 Purpose:     1D and 2D structured meshes with ghost cells
 Author:      Lucas Pierce
 Description:
   - Mesh1D: uniform 1D grid with ghost cells
   - Mesh2D: uniform 2D grid with ghost cells
   - Stores cell-center coordinates and indexing helpers
   - NG (ghost cells per side) is passed in from the outside
==============================================================================
*/

#pragma once
#include <vector>
#include <cstddef> // std::size_t

namespace ls {

// -----------------------------------------------------------------------------
// 1D mesh: cell-centered, uniform, with NG ghost cells on each side
//
// Physical domain: x in [0, L]
// Interior cells:  i = 0 ... nx-1
// Stored indices:  I = 0 ... nx_tot-1, where nx_tot = nx + 2*ng
// Mapping:         I = i + ng    (for interior cells)
//
// We store x-coordinates for all cells including ghosts:
//   x[I] = (I - ng + 0.5) * dx, so that interior centers are:
//
//   I = ng        → x = (0 + 0.5) * dx       (first interior cell)
//   I = ng+nx-1   → x = ((nx-1) + 0.5) * dx  (last interior cell)
// -----------------------------------------------------------------------------

struct Mesh1D {
    int    nx;      // number of interior cells
    int    ng;      // number of ghost cells on each side
    double L;       // physical length of domain
    double dx;      // cell width
    int    nx_tot;  // total cells including ghosts
    std::vector<double> x;  // cell-center coordinates (size = nx_tot)

    Mesh1D(int nx_in, double L_in, int ng_in)
        : nx(nx_in), ng(ng_in), L(L_in)
    {
        nx_tot = nx + 2 * ng;
        dx     = L / static_cast<double>(nx);

        x.resize(nx_tot);
        for (int I = 0; I < nx_tot; ++I) {
            // I runs over ghosts + interior; shift by -ng to get logical index
            x[I] = ( (I - ng) + 0.5 ) * dx;
        }
    }

    // Total number of cells (including ghosts)
    int size() const { return nx_tot; }

    // Index range of interior cells in storage
    int interior_start() const { return ng; }
    int interior_end()   const { return ng + nx - 1; }

    // Convenience: cell-center coordinate
    double xc(int I) const { return x[I]; }

    // Convert logical interior index i (0..nx-1) to storage index I
    int index(int i) const { return i + ng; }

    // (Optional) convert storage index to logical interior index
    int interior_i(int I) const { return I - ng; }
};

// -----------------------------------------------------------------------------
// 2D mesh: cell-centered, uniform, with NG ghost cells on each side
//
// Physical domain: x in [0, Lx], y in [0, Ly]
// Interior indices:
//   i = 0 ... nx-1  (x-direction)
//   j = 0 ... ny-1  (y-direction)
//
// Stored indices:
//   I = 0 ... nx_tot-1, where nx_tot = nx + 2*ng
//   J = 0 ... ny_tot-1, where ny_tot = ny + 2*ng
//
// Flattening into 1D:
//   idx(I, J) = I + nx_tot * J
//
// Cell centers:
//   x_centers[I] = ( (I - ng) + 0.5 ) * dx
//   y_centers[J] = ( (J - ng) + 0.5 ) * dy
// -----------------------------------------------------------------------------

struct Mesh2D {
  
  int   nx, ny; // interior cells in x and y
  int   ng;     // ghost cells on each side (both directions)
  double Lx, Ly; // physical domain size (meters)
  double dx, dy; // cell width
  int nx_tot; //totla cells in x including ghosts
  int ny_tot; //total cells in y including ghosts

  std::vector<double> x; // x-centers, size nx_tot
  std::vector<double> y; // y-centers, size ny_tot

    Mesh2D(int nx_in, int ny_in, double Lx_in, double Ly_in, int ng_in)
        : nx(nx_in), ny(ny_in), ng(ng_in), Lx(Lx_in), Ly(Ly_in)
    {
        nx_tot = nx + 2 * ng;
        ny_tot = ny + 2 * ng;

        dx = Lx / static_cast<double>(nx);
        dy = Ly / static_cast<double>(ny);

        x.resize(nx_tot);
        y.resize(ny_tot);

        for (int I = 0; I < nx_tot; ++I) {
            x[I] = ( (I - ng) + 0.5 ) * dx;
        }
        for (int J = 0; J < ny_tot; ++J) {
            y[J] = ( (J - ng) + 0.5 ) * dy;
        }
    }

    // Total cells including ghosts (per direction)
    int size_x() const { return nx_tot; }
    int size_y() const { return ny_tot; }

    // Interior ranges
    int interior_i_start() const { return ng; }
    int interior_i_end()   const { return ng + nx - 1; }
    int interior_j_start() const { return ng; }
    int interior_j_end()   const { return ng + ny - 1; }

    // Cell-center coordinates
    double xc(int I) const { return x[I]; }
    double yc(int J) const { return y[J]; }

    // Flattened 1D index for arrays like std::vector<State2D>
    // (0 <= I < nx_tot, 0 <= J < ny_tot)
    int index(int I, int J) const {
        return I + nx_tot * J;
    }

    // Convenience: interior logical indices (0..nx-1, 0..ny-1) → storage
    int idx_interior(int i, int j) const {
        int I = i + ng;
        int J = j + ng;
        return index(I, J);
    }

    // (Optional) inverse mapping from flat index to (I, J)
    void IJ_from_index(int k, int& I, int& J) const {
        I = k % nx_tot;
        J = k / nx_tot;
    }

};

} // namespace ls