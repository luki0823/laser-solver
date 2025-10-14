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
