/*
==============================================================================
 File:        bc.hpp
 Purpose:     Boundary conditions for 1D and 2D Euler solvers
 Author:      Lucas Pierce
 Description:
   - Supports Wall, Symmetry, and Outflow (zero-gradient / farfield-like)
   - 1D: left/right boundaries
   - 2D: left/right/bottom/top boundaries
   - Designed to work with Mesh1D / Mesh2D and State1D / State2D
==============================================================================
*/

#pragma once

#include <vector>
#include <string>
#include <algorithm>
#include <stdexcept>

#ifdef LS_USE_OPENMP
  #include <omp.h>
#endif

#include "ls/types.hpp"
#include "ls/mesh.hpp"
#include "ls/eos.hpp"

namespace ls {

enum class BcType {
    Wall,
    Symmetry,
    Outflow,
    Periodic
};

inline BcType parse_bc_type(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), ::tolower);
    if (s == "wall")     return BcType::Wall;
    if (s == "symmetry") return BcType::Symmetry;
    if (s == "outflow" || s == "farfield" || s == "far-field") return BcType::Outflow;
    if (s == "periodic") return BcType::Periodic;
    throw std::runtime_error("Unknown boundary type: " + s);
}

struct Bc1D { BcType left; BcType right; };
struct Bc2D { BcType left; BcType right; BcType bottom; BcType top; };

inline Bc1D make_bc1d_from_strings(const std::string& left, const std::string& right) {
    return {parse_bc_type(left), parse_bc_type(right)};
}

inline Bc2D make_bc2d_from_strings(const std::string& left,
                                   const std::string& right,
                                   const std::string& bottom,
                                   const std::string& top) {
    return {parse_bc_type(left),
            parse_bc_type(right),
            parse_bc_type(bottom),
            parse_bc_type(top)};
}

// -----------------------------------------------------------------------------
// 1D boundary conditions
// -----------------------------------------------------------------------------
inline void apply_wall_symmetry_1d_left(std::vector<State1D>& U, const Mesh1D& mesh) {
    const int ng = mesh.ng;
    const int I0 = mesh.interior_start();
    for (int g = 1; g <= ng; ++g) {
        State1D val = U[I0];
        val.rhou = -val.rhou;
        U[I0 - g] = val;
    }
}

inline void apply_wall_symmetry_1d_right(std::vector<State1D>& U, const Mesh1D& mesh) {
    const int ng = mesh.ng;
    const int I1 = mesh.interior_end();
    for (int g = 1; g <= ng; ++g) {
        State1D val = U[I1];
        val.rhou = -val.rhou;
        U[I1 + g] = val;
    }
}

inline void apply_outflow_1d_left(std::vector<State1D>& U, const Mesh1D& mesh) {
    const int ng = mesh.ng;
    const int I0 = mesh.interior_start();
    for (int g = 1; g <= ng; ++g) U[I0 - g] = U[I0];
}

inline void apply_outflow_1d_right(std::vector<State1D>& U, const Mesh1D& mesh) {
    const int ng = mesh.ng;
    const int I1 = mesh.interior_end();
    for (int g = 1; g <= ng; ++g) U[I1 + g] = U[I1];
}

inline void apply_bc_periodic_1d(std::vector<State1D>& U, const Mesh1D& mesh)
{
    const int ng = mesh.ng;
    const int i0 = mesh.interior_start();
    const int i1 = mesh.interior_end();

    for (int g = 0; g < ng; ++g) U[g]          = U[i1 - (ng - 1) + g];
    for (int g = 0; g < ng; ++g) U[i1 + 1 + g] = U[i0 + g];
}

inline void apply_bc_1d(std::vector<State1D>& U, const Mesh1D& mesh, const Bc1D& bc)
{
    switch (bc.left) {
        case BcType::Wall:
        case BcType::Symmetry: apply_wall_symmetry_1d_left(U, mesh); break;
        case BcType::Outflow:  apply_outflow_1d_left(U, mesh); break;
        case BcType::Periodic: apply_bc_periodic_1d(U, mesh); break;
    }

    switch (bc.right) {
        case BcType::Wall:
        case BcType::Symmetry: apply_wall_symmetry_1d_right(U, mesh); break;
        case BcType::Outflow:  apply_outflow_1d_right(U, mesh); break;
        case BcType::Periodic: apply_bc_periodic_1d(U, mesh); break;
    }
}

inline void apply_bc_1d(std::vector<State1D>& U,
                        const Mesh1D& mesh,
                        const std::string& left,
                        const std::string& right)
{
    Bc1D bc = make_bc1d_from_strings(left, right);
    apply_bc_1d(U, mesh, bc);
}

// -----------------------------------------------------------------------------
// 2D boundary conditions
// -----------------------------------------------------------------------------
inline void apply_wall_symmetry_2d_left(std::vector<State2D>& U, const Mesh2D& mesh) {
    const int ng = mesh.ng;
    const int I0 = mesh.interior_i_start();
    const int j0 = mesh.interior_j_start();
    const int j1 = mesh.interior_j_end();

    #ifdef LS_USE_OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int J = j0; J <= j1; ++J) {
        const int idx_interior = mesh.index(I0, J);
        for (int g = 1; g <= ng; ++g) {
            const int idx_g = mesh.index(I0 - g, J);
            State2D val = U[idx_interior];
            val.rhou = -val.rhou;
            U[idx_g] = val;
        }
    }
}

inline void apply_wall_symmetry_2d_right(std::vector<State2D>& U, const Mesh2D& mesh) {
    const int ng = mesh.ng;
    const int I1 = mesh.interior_i_end();
    const int j0 = mesh.interior_j_start();
    const int j1 = mesh.interior_j_end();

    #ifdef LS_USE_OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int J = j0; J <= j1; ++J) {
        const int idx_interior = mesh.index(I1, J);
        for (int g = 1; g <= ng; ++g) {
            const int idx_g = mesh.index(I1 + g, J);
            State2D val = U[idx_interior];
            val.rhou = -val.rhou;
            U[idx_g] = val;
        }
    }
}

inline void apply_outflow_2d_left(std::vector<State2D>& U, const Mesh2D& mesh) {
    const int ng = mesh.ng;
    const int I0 = mesh.interior_i_start();
    const int j0 = mesh.interior_j_start();
    const int j1 = mesh.interior_j_end();

    #ifdef LS_USE_OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int J = j0; J <= j1; ++J) {
        const int idx_interior = mesh.index(I0, J);
        for (int g = 1; g <= ng; ++g) {
            const int idx_g = mesh.index(I0 - g, J);
            U[idx_g] = U[idx_interior];
        }
    }
}

inline void apply_outflow_2d_right(std::vector<State2D>& U, const Mesh2D& mesh) {
    const int ng = mesh.ng;
    const int I1 = mesh.interior_i_end();
    const int j0 = mesh.interior_j_start();
    const int j1 = mesh.interior_j_end();

    #ifdef LS_USE_OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int J = j0; J <= j1; ++J) {
        const int idx_interior = mesh.index(I1, J);
        for (int g = 1; g <= ng; ++g) {
            const int idx_g = mesh.index(I1 + g, J);
            U[idx_g] = U[idx_interior];
        }
    }
}

inline void apply_wall_symmetry_2d_bottom(std::vector<State2D>& U, const Mesh2D& mesh) {
    const int ng = mesh.ng;
    const int J0 = mesh.interior_j_start();
    const int i0 = mesh.interior_i_start();
    const int i1 = mesh.interior_i_end();

    #ifdef LS_USE_OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int I = i0; I <= i1; ++I) {
        const int idx_interior = mesh.index(I, J0);
        for (int g = 1; g <= ng; ++g) {
            const int idx_g = mesh.index(I, J0 - g);
            State2D val = U[idx_interior];
            val.rhov = -val.rhov;
            U[idx_g] = val;
        }
    }
}

inline void apply_wall_symmetry_2d_top(std::vector<State2D>& U, const Mesh2D& mesh) {
    const int ng = mesh.ng;
    const int J1 = mesh.interior_j_end();
    const int i0 = mesh.interior_i_start();
    const int i1 = mesh.interior_i_end();

    #ifdef LS_USE_OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int I = i0; I <= i1; ++I) {
        const int idx_interior = mesh.index(I, J1);
        for (int g = 1; g <= ng; ++g) {
            const int idx_g = mesh.index(I, J1 + g);
            State2D val = U[idx_interior];
            val.rhov = -val.rhov;
            U[idx_g] = val;
        }
    }
}

inline void apply_outflow_2d_bottom(std::vector<State2D>& U, const Mesh2D& mesh) {
    const int ng = mesh.ng;
    const int J0 = mesh.interior_j_start();
    const int i0 = mesh.interior_i_start();
    const int i1 = mesh.interior_i_end();

    #ifdef LS_USE_OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int I = i0; I <= i1; ++I) {
        const int idx_interior = mesh.index(I, J0);
        for (int g = 1; g <= ng; ++g) {
            const int idx_g = mesh.index(I, J0 - g);
            U[idx_g] = U[idx_interior];
        }
    }
}

inline void apply_outflow_2d_top(std::vector<State2D>& U, const Mesh2D& mesh) {
    const int ng = mesh.ng;
    const int J1 = mesh.interior_j_end();
    const int i0 = mesh.interior_i_start();
    const int i1 = mesh.interior_i_end();

    #ifdef LS_USE_OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int I = i0; I <= i1; ++I) {
        const int idx_interior = mesh.index(I, J1);
        for (int g = 1; g <= ng; ++g) {
            const int idx_g = mesh.index(I, J1 + g);
            U[idx_g] = U[idx_interior];
        }
    }
}

// ------------------------------------------------------------
// 2-D Periodic Boundary Conditions
// ------------------------------------------------------------
inline void apply_bc_periodic_2d(std::vector<State2D>& U, const Mesh2D& mesh)
{
    const int ng = mesh.ng;

    const int ix0 = mesh.interior_i_start();
    const int ix1 = mesh.interior_i_end();
    const int iy0 = mesh.interior_j_start();
    const int iy1 = mesh.interior_j_end();

    // X periodic
    #ifdef LS_USE_OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int J = iy0; J <= iy1; ++J) {
        for (int g = 0; g < ng; ++g) {
            U[mesh.index(ix0 - 1 - g, J)] = U[mesh.index(ix1 - g, J)];
            U[mesh.index(ix1 + 1 + g, J)] = U[mesh.index(ix0 + g, J)];
        }
    }

    // Y periodic
    #ifdef LS_USE_OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int I = ix0; I <= ix1; ++I) {
        for (int g = 0; g < ng; ++g) {
            U[mesh.index(I, iy0 - 1 - g)] = U[mesh.index(I, iy1 - g)];
            U[mesh.index(I, iy1 + 1 + g)] = U[mesh.index(I, iy0 + g)];
        }
    }

    // Corners (tiny) — keep serial
    for (int gI = 0; gI < ng; ++gI) {
        for (int gJ = 0; gJ < ng; ++gJ) {
            U[mesh.index(ix0-1-gI, iy0-1-gJ)] = U[mesh.index(ix1-gI, iy1-gJ)];
            U[mesh.index(ix1+1+gI, iy0-1-gJ)] = U[mesh.index(ix0+gI, iy1-gJ)];
            U[mesh.index(ix0-1-gI, iy1+1+gJ)] = U[mesh.index(ix1-gI, iy0+gJ)];
            U[mesh.index(ix1+1+gI, iy1+1+gJ)] = U[mesh.index(ix0+gI, iy0+gJ)];
        }
    }
}

inline void fill_corners_2d(std::vector<State2D>& U, const Mesh2D& mesh)
{
    const int ng = mesh.ng;
    const int i0 = mesh.interior_i_start();
    const int i1 = mesh.interior_i_end();
    const int j0 = mesh.interior_j_start();
    const int j1 = mesh.interior_j_end();

    // bottom-left
    for (int gi = 1; gi <= ng; ++gi)
        for (int gj = 1; gj <= ng; ++gj)
            U[mesh.index(i0-gi, j0-gj)] = U[mesh.index(i0-gi, j0-1)];

    // bottom-right
    for (int gi = 1; gi <= ng; ++gi)
        for (int gj = 1; gj <= ng; ++gj)
            U[mesh.index(i1+gi, j0-gj)] = U[mesh.index(i1+gi, j0-1)];

    // top-left
    for (int gi = 1; gi <= ng; ++gi)
        for (int gj = 1; gj <= ng; ++gj)
            U[mesh.index(i0-gi, j1+gj)] = U[mesh.index(i0-gi, j1+1)];

    // top-right
    for (int gi = 1; gi <= ng; ++gi)
        for (int gj = 1; gj <= ng; ++gj)
            U[mesh.index(i1+gi, j1+gj)] = U[mesh.index(i1+gi, j1+1)];
}


inline void apply_bc_2d(std::vector<State2D>& U, const Mesh2D& mesh, const Bc2D& bc)
{
    if (bc.left == BcType::Periodic &&
        bc.right == BcType::Periodic &&
        bc.bottom == BcType::Periodic &&
        bc.top == BcType::Periodic)
    {
        apply_bc_periodic_2d(U, mesh);
        return;
    }

    switch (bc.left) {
        case BcType::Wall:
        case BcType::Symmetry: apply_wall_symmetry_2d_left(U, mesh); break;
        case BcType::Outflow:  apply_outflow_2d_left(U, mesh); break;
        case BcType::Periodic: break;
    }

    switch (bc.right) {
        case BcType::Wall:
        case BcType::Symmetry: apply_wall_symmetry_2d_right(U, mesh); break;
        case BcType::Outflow:  apply_outflow_2d_right(U, mesh); break;
        case BcType::Periodic: break;
    }

    switch (bc.bottom) {
        case BcType::Wall:
        case BcType::Symmetry: apply_wall_symmetry_2d_bottom(U, mesh); break;
        case BcType::Outflow:  apply_outflow_2d_bottom(U, mesh); break;
        case BcType::Periodic: break;
    }

    switch (bc.top) {
        case BcType::Wall:
        case BcType::Symmetry: apply_wall_symmetry_2d_top(U, mesh); break;
        case BcType::Outflow:  apply_outflow_2d_top(U, mesh); break;
        case BcType::Periodic: break;
    }

    fill_corners_2d(U, mesh);
}

inline void apply_bc_2d(std::vector<State2D>& U,
                        const Mesh2D& mesh,
                        const std::string& left,
                        const std::string& right,
                        const std::string& bottom,
                        const std::string& top)
{
    Bc2D bc = make_bc2d_from_strings(left, right, bottom, top);
    apply_bc_2d(U, mesh, bc);
}

} // namespace ls
