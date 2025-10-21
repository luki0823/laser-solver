#pragma once
#include "ls/types.hpp"
#include "ls/recon.hpp"
#include <vector>
#include <algorithm>

namespace ls {

// Fill ghost cells for each BC type
inline void fill_ghosts(std::vector<State>& U, BCKind bc)
{
    const int NG = 2;
    const int nx_total = (int)U.size();
    const int nx = nx_total - 2*NG;

    // convenience references
    auto L = NG;           // first physical cell index
    auto R = NG + nx - 1;  // last physical cell index

    switch (bc)
    {
        case BCKind::CopyEnds:
            for (int g = 0; g < NG; ++g) {
                U[L-1-g] = U[L];       // left ghosts = first interior
                U[R+1+g] = U[R];       // right ghosts = last interior
            }
            break;

        case BCKind::ReflectLeftCopy:
            // Left wall reflection: mirror rho,E, flip u
            for (int g = 0; g < NG; ++g) {
                U[L-1-g] = U[L+g];
                U[L-1-g].rhou *= -1.0;
            }
            // Right outflow copy
            for (int g = 0; g < NG; ++g)
                U[R+1+g] = U[R];
            break;

        case BCKind::Periodic:
            for (int g = 0; g < NG; ++g) {
                U[L-1-g] = U[R-g];     // wrap left ghosts
                U[R+1+g] = U[L+g];     // wrap right ghosts
            }
            break;
    }
}

} // namespace ls
