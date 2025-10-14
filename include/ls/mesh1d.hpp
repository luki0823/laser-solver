/*
==============================================================================
 File:        mesh1d.hpp
 Purpose:     Defines 1D computational grid structure and indexing utilities.
 Description:
   Holds uniform grid info:
     nx  - number of cells
     dx  - cell width
     x[] - cell center coordinates

 Dependencies:
    - <vector>

 Example:
   ls::Grid1D grid(200, 1.0);
   double dx = grid.dx;
==============================================================================
*/

#pragma once
#include <vector>

namespace ls {

struct Grid1D {
  int nx;            // number of cells
  double L;          // domain length
  double dx;         // cell width
  std::vector<double> x; // cell-center positions

  Grid1D(int nx_, double L_)
  : nx(nx_), L(L_), dx(L_/nx_), x(nx_) {
    for(int i=0;i<nx;i++) x[i] = (i + 0.5) * dx;
  }
};

} // namespace ls