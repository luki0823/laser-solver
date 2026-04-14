# laser2d: High-Order Finite-Difference Solver for Compressible Flow

## Overview

`laser2d` is a two-dimensional compressible flow solver developed to study shock-dominated phenomena relevant to nanosecond laser ablation. The code focuses on evaluating high-order spatial reconstruction schemes, including ENO/WENO/TENO families, with an emphasis on shock resolution, numerical dissipation, and computational efficiency.

The solver is designed for controlled numerical experiments, including convergence studies and nonlinear shock-dominated test cases.

---

## Features

- Finite-difference formulation for compressible Euler equations
- High-order reconstruction schemes:
  - First-order upwind
  - WENO3
  - WENO5
  - TENO5
- Explicit time integration (Forward Euler / RK-based methods depending on configuration)
- Shock-capturing capability for discontinuities and steep gradients
- Parallel execution via OpenMP
- Built-in diagnostics for:
  - Shock location and propagation
  - Shock thickness (resolution)
  - Total variation and smoothness metrics
  - Runtime and computational cost

---

## Governing Equations

The solver advances the 2D compressible Euler equations in conservative form:

- Mass conservation
- Momentum conservation (x and y)
- Energy conservation

An ideal gas equation of state is used to close the system.

---

## Code Structure

laser2d/
├── build/                  
├── src/                    
├── include/                
├── scripts/                
├── runs/                   
├── diagnostics/            
├── run_laser2d_sweep.sh    
└── README.md

---

## Installation

### Requirements

- C++ compiler with OpenMP support (e.g., g++)
- CMake

### Build

mkdir build
cd build
cmake ..
make -j

Executable:

build/bin/laser2d

---

## Running the Solver

### Single Run

./build/bin/laser2d

### Batch Runs

chmod +x run_laser2d_sweep.sh
./run_laser2d_sweep.sh

Override parameters:

NX=512 NY=512 OMP_THREADS=12 ./run_laser2d_sweep.sh

---

## Output Data

runs/<timestamp>_<nx>x<ny>_omp<threads>/<scheme>_<nx>x<ny>_omp<threads>/data/

Each timestep CSV includes:

- x, y
- rho
- u, v
- p
- E
- rhou, rhov
- H

---

## Diagnostics

diagnostics_<scheme>_<nx>x<ny>_omp<threads>_<timestamp>.csv

Includes:

- Shock location
- Shock thickness
- Max pressure gradient
- Total variation
- Runtime metrics

Enable:

ENABLE_THESIS_DIAG=1 DIAG_INTERVAL=100 ./run_laser2d_sweep.sh

---

## Post-Processing

python scripts/plot_slice.py

---

## Numerical Experiments

### Convergence Study
- Verifies formal order

### Nonlinear Shock Tests
- Shock accuracy
- Shock resolution
- Dissipation behavior

---

## Key Findings

- High-order schemes converge as expected
- Lower-order methods are more diffusive
- Shock locations are consistent
- High-order captures secondary shocks
- TENO5 provides best resolution

---

## Future Work

- Real-gas models
- Higher-order time integration
- Adaptive mesh refinement
- GPU acceleration

---

## Author

Lucas P.  
M.S. Mechanical Engineering  
South Dakota School of Mines and Technology  

---

## License

Add a license if distributing publicly.