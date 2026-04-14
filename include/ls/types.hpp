/*
==============================================================================
 File:        types.hpp
 Purpose:     Core consercative state and tiny algebra helpers
 Description:
   
 Dependencies:


 Example:

==============================================================================
*/
#pragma once
#include <string>
#include <algorithm>
#include <iostream>

namespace ls {


enum class ReconType {
    FirstOrder,
    WENO3,
    MUSCL,
    WENO5,
    WENO5Z,
    TENO5
};

inline ReconType parse_recon_type(const std::string& r) {
    std::string s = r;
    std::transform(s.begin(), s.end(), s.begin(), ::tolower);

    if (s == "firstorder" || s == "godunov") return ReconType::FirstOrder;
    if (s == "weno3" || s == "weno-3")       return ReconType::WENO3;

    if (s == "muscl" || s == "tvd2")         return ReconType::MUSCL;
    if (s == "weno5" || s == "weno-5" || s == "weno5-js") return ReconType::WENO5;
    if (s == "weno5z" || s == "weno-z" || s == "weno5-z") return ReconType::WENO5Z;
    if (s == "teno"  || s == "teno5" || s == "teno-5")    return ReconType::TENO5;

    throw std::runtime_error("Unknown reconstruction type: " + r);
}

inline int num_ghost_cells(const std::string& recon) {
    std::string r = recon;
    std::transform(r.begin(), r.end(), r.begin(), ::tolower);

    if (r == "firstorder" || r == "godunov") return 1;
    if (r == "muscl" || r == "tvd2")         return 2; // 2 is safe with your BC fill
    if (r == "weno3" || r == "weno-3")       return 2;
    if (r == "weno5" || r == "weno-5")       return 3;
    if (r == "weno5z" || r == "weno-z")      return 3;
    if (r == "teno" || r == "teno5")         return 3;
    if (r == "weno7" || r == "weno-7")       return 4;

    return 2;
}



// Conservative state: U = [rho, rho*u, E]
struct State1D { 
               double rho; //mass density
               double rhou; // momentum (rhou)
               double E; // energy density
};


// --- Algebraic operators for 1D ---
inline State1D operator+(State1D a, const State1D& b) {
    return {a.rho + b.rho, a.rhou + b.rhou, a.E + b.E};
}

inline State1D operator-(State1D a, const State1D& b) {
    return {a.rho - b.rho, a.rhou - b.rhou, a.E - b.E};
}

inline State1D operator*(double s, State1D a) {
    return {s * a.rho, s * a.rhou, s * a.E};
}

inline State1D operator*(State1D a, double s) {
    return s * a;
}

inline State1D& operator+=(State1D& a, const State1D& b) {
    a.rho  += b.rho;
    a.rhou += b.rhou;
    a.E    += b.E;
    return a;
}

inline State1D& operator-=(State1D& a, const State1D& b) {
    a.rho  -= b.rho;
    a.rhou -= b.rhou;
    a.E    -= b.E;
    return a;
}

inline State1D& operator*=(State1D& a, double s) {
    a.rho  *= s;
    a.rhou *= s;
    a.E    *= s;
    return a;
}

inline State1D operator-(const State1D& a) {
    return State1D{-a.rho, -a.rhou, -a.E};
}

// --- Utility functions ---
inline State1D zero_state1d() {
    return {0.0, 0.0, 0.0};
}

inline std::ostream& operator<<(std::ostream& os, const State1D& s) {
    os << "[rho=" << s.rho << ", rhou=" << s.rhou << ", E=" << s.E << "]";
    return os;
}

// --------------------------------------------------
// 2D Conservative State: U = [rho, rho*u, rho*v, E]
// --------------------------------------------------

struct State2D {
        
        double rho; // mass
        double rhou; // momentum in x
        double rhov;  // momentum in y
        double E;  // total energy density
};

// --- Algebraic operators for 2D ---
inline State2D operator+(State2D a, const State2D& b) {
    return {a.rho + b.rho, a.rhou + b.rhou, a.rhov + b.rhov, a.E + b.E};
}

inline State2D operator-(State2D a, const State2D& b) {
    return {a.rho - b.rho, a.rhou - b.rhou, a.rhov - b.rhov, a.E - b.E};
}

inline State2D operator*(double s, State2D a) {
    return {s * a.rho, s * a.rhou, s * a.rhov, s * a.E};
}

inline State2D operator*(State2D a, double s) {
    return s * a;
}

inline State2D& operator+=(State2D& a, const State2D& b) {
    a.rho  += b.rho;
    a.rhou += b.rhou;
    a.rhov += b.rhov;
    a.E    += b.E;
    return a;
}

inline State2D& operator-=(State2D& a, const State2D& b) {
    a.rho  -= b.rho;
    a.rhou -= b.rhou;
    a.rhov -= b.rhov;
    a.E    -= b.E;
    return a;
}

inline State2D& operator*=(State2D& a, double s) {
    a.rho  *= s;
    a.rhou *= s;
    a.rhov *= s;
    a.E    *= s;
    return a;
}

inline State2D operator-(const State2D& a) {
    return State2D{-a.rho, -a.rhou, -a.rhov, -a.E};
}


// --- Utility functions ---
inline State2D zero_state2d() {
    return {0.0, 0.0, 0.0, 0.0};
}

inline std::ostream& operator<<(std::ostream& os, const State2D& s) {
    os << "[rho=" << s.rho << ", rhou=" << s.rhou
       << ", rhov=" << s.rhov << ", E=" << s.E << "]";
    return os;
}

// -----------------------------------------------------------------------------
// Aliases for readability elsewhere in the code
// -----------------------------------------------------------------------------
using Cons1D = State1D;
using Cons2D = State2D;

} // namespace ls