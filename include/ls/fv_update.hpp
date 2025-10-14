/*
==============================================================================
 File:        fv_update.hpp
 Purpose:     Handles flux divergence and time-stepping for 1D finite-volume.
 Description:
   Provides:
     - flux_divergence(): computes spatial derivative of fluxes
     - step_euler(): single-step time integrator (Forward Euler)

 Dependencies:
   - riemann.hpp
   - grid.hpp
   - eos.hpp

 Notes:
   - Ensure ghost cells are applied before flux_divergence().
==============================================================================
*/
