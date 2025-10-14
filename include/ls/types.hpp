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
namespace ls {

// Conservative state: U = [rho, rho*u, E]
struct State { double rho, rhou, E; };

// Lightweight algebra so update code reads naturally.
inline State operator+(State a, const State& b){ return {a.rho+b.rho, a.rhou+b.rhou, a.E+b.E}; }
inline State operator-(State a, const State& b){ return {a.rho-b.rho, a.rhou-b.rhou, a.E-b.E}; }
inline State operator*(double s, State a){ return {s*a.rho, s*a.rhou, s*a.E}; }
inline State operator*(State a, double s){ return s*a; }

} // namespace ls