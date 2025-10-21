/*
==============================================================================
 File:        eos.hpp
 Purpose:     Defines thermodynamic relations for an ideal gas EOS.
 Description:
   Provides functions to compute pressure, sound speed, and conversions
   between primitive (rho, u, p) and conservative (rho, rho*u, E) variables.

 Dependencies:
   - types.hpp  (for struct State)
   - <cmath>

 Example:
   double p = ls::pressure(U, gamma);
   double c = ls::sound_speed(U, gamma);
==============================================================================

*/

#pragma once
#include "types.hpp" //include types for U
#include <cmath> //math funcs



namespace ls{

    //----------------------------------------------------------
    // IdealGas struct
    //   Encapsulates an equation of state (EOS) for an ideal gas.
    //   Provides functions to compute pressure and sound speed
    //   from the conservative state.
    //----------------------------------------------------------
    struct IdealGas {
        double gamma;   // ratio of specific heats, usually 1.4 for air

        // Constructor: initialize gamma (default = 1.4 if not provided).
        // "explicit" avoids unintentional implicit conversions.
        explicit IdealGas(double g = 1.4) : gamma(g) {}

        //------------------------------------------------------
        // Compute pressure from conservative state U
        //
        // Formula: p = (gamma - 1) * (E - 0.5 * rho * u^2)
        //
        // where:
        //   u = velocity = (rho*u)/rho
        //   E = total energy density
        //------------------------------------------------------
        inline double pressure(const State& U) const {
            double u = U.rhou / U.rho; // velocity = momentum / density
            return (gamma - 1.0) * (U.E - 0.5 * U.rho * u * u);
        }

        //------------------------------------------------------
        // Compute speed of sound from conservative state U
        //
        // Formula: c = sqrt(gamma * p / rho)
        //
        // where p is obtained from the pressure function above.
        //------------------------------------------------------
        inline double sound_speed(const State& U) const {
            double p = pressure(U);           // compute thermodynamic pressure
            return std::sqrt(gamma * p / U.rho);
        }
    };

    //----------------------------------------------------------
    // Convert primitive variables (rho, u, p) into conservative
    // variables (rho, rhou, E).
    //
    // Input:
    //   rho   = density
    //   u     = velocity
    //   p     = pressure
    //   gamma = ratio of specific heats
    //
    // Output: returns a State struct {rho, rhou, E}
    //
    // Formula for total energy density:
    //   E = p / (gamma - 1) + 0.5 * rho * u^2
    //----------------------------------------------------------
    inline State prim_to_cons(double rho, double u, double p, double gamma) {
        return {
            rho,                          // density
            rho * u,                      // momentum = rho * u
            p / (gamma - 1.0) + 0.5 * rho * u * u  // total energy density
        };
    }

    //----------------------------------------------------------
    // Convert conservative variables back to primitives.
    //
    // Input:
    //   U     = State {rho, rhou, E}
    //   gamma = ratio of specific heats
    //
    // Output (passed by reference):
    //   rho = density
    //   u   = velocity
    //   p   = pressure
    //
    // This function modifies the variables rho, u, p directly.
    // That's why return type is void.
    //----------------------------------------------------------
    inline void cons_to_prim(const State& U, double gamma,
                            double& rho, double& u, double& p) {
        rho = std::max(U.rho,1e-12);                // extract density
        u   = U.rhou / rho;         // velocity = momentum / density
        p   = (gamma - 1.0) *       // ideal gas relation
            (U.E - 0.5 * rho * u * u);
    }   

}