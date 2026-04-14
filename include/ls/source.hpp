/*
==============================================================================
 File:        source.hpp
 Purpose:     Defines laser heat source term (Gaussian temporal-spatial).
 Description:
   Models absorbed laser energy as a Gaussian in space/time:
     S_E = A * exp(-((x-x0)/wx)^2) * exp(-((t-t0)/wt)^2)

 Example:
   ls::GaussianPulse laser(1e8, 0.0, 5e-6, 30e-9, 8e-9);
   double Q = laser(x, t);
==============================================================================
*/

#pragma once
#include <cmath>

namespace ls {

// Simple 1D Gaussian pulse in space & time for energy source
struct GaussianPulse1D {
    double A;   // amplitude [W/m^3]
    double x0;  // center position
    double wx;  // spatial width
    double t0;  // center time
    double wt;  // temporal width

    double operator()(double x, double t) const {
        double xs = (x - x0) / wx;
        double ts = (t - t0) / wt;
        return A * std::exp(-(xs*xs)) * std::exp(-(ts*ts));
    }
};

struct GaussianPulse2D {
    double A;   ///< amplitude [W/m^3]
    double x0;  ///< center x-position
    double y0;  ///< center y-position
    double wx;  ///< spatial width in x
    double wy;  ///< spatial width in y
    double t0;  ///< center time
    double wt;  ///< temporal width

    double operator()(double x, double y, double t) const {
        const double xs = (x - x0) / wx;
        const double ys = (y - y0) / wy;
        const double ts = (t - t0) / wt;
        // exp(-(xs^2 + ys^2 + ts^2))
        return A * std::exp(-(xs*xs + ys*ys + ts*ts));
    }
};

} // namespace ls
