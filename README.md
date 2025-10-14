# 🚀 Next Stage Roadmap — High-Order Solver & Laser Ablation Development

This document outlines the roadmap for expanding the current **1D Euler shock-tube solver** into a **3rd-order accurate**, **2D**, and **laser-ablation-capable** framework.

It’s divided into three stages:
1. Higher-order accuracy in 1D  
2. 2D finite-volume extension  
3. Physical modeling for laser–matter interaction

---

## 🧩 Stage A — 3rd-Order Accuracy in 1D

### **A1. Time Integration — SSP-RK3 (Third-Order TVD)**

**Math (Shu–Osher form):**

$$
\begin{aligned}
U^{(1)} &= U^n + \Delta t\,\mathcal{L}(U^n) \\\\
U^{(2)} &= \tfrac{3}{4} U^n + \tfrac{1}{4}\left(U^{(1)} + \Delta t\,\mathcal{L}(U^{(1)})\right) \\\\
U^{n+1} &= \tfrac{1}{3} U^n + \tfrac{2}{3}\left(U^{(2)} + \Delta t\,\mathcal{L}(U^{(2)})\right)
\end{aligned}
$$

where

$$
\mathcal{L}(U) = -\frac{1}{\Delta x}\left(F^*_{i+\tfrac{1}{2}} - F^*_{i-\tfrac{1}{2}}\right) + S(U,t)
$$

**Implementation Notes**
- Add `advance_ssprk3(U, t, t_end, CFL, dx, eos, bc, source_hook)` in `time_int.hpp`.
- Reuse your existing `flux_divergence()` and optional `source_hook(dt, tnext, U)` at each stage.
- Recompute `dt` using CFL each outer loop.

✅ **Done When**
- Sod tube shows less diffusion than Forward Euler.  
- Stable for CFL ≈ 0.3–0.5 with Rusanov.

---

### **A2. Spatial Reconstruction — WENO3 (Third-Order Upwind)**

**Goal:** Replace piecewise-constant reconstruction with a third-order weighted nonlinear scheme.

**Scalar WENO3 equations:**

$$
q_{i+\tfrac{1}{2}}^- = \omega_0 q^{(0)} + \omega_1 q^{(1)}
$$

Weights:

$$
\alpha_k = \frac{d_k}{(\epsilon + \beta_k)^2}, \quad \omega_k = \frac{\alpha_k}{\alpha_0 + \alpha_1}
$$

**Implementation Steps**
- Add `reconstruct_weno3(U, eos, bc, UL, UR)` in `recon.hpp`.
- Keep the same UL/UR interface so the time integrator is unchanged.
- Start with **component-wise** reconstruction; later use **characteristic projection**.

**Pitfalls**
- Choose small $\epsilon \approx 10^{-6}$.
- Clamp minimum $(\rho, p)$ near vacuum.
- Verify on Sod and Lax tests.

✅ **Done When**
- Contact discontinuity and shock are sharper than 1st-order.  
- No spurious oscillations in smooth regions.

---

### **A3. Riemann Solver — HLLC (Optional Upgrade)**

**Why:** Rusanov is simple but overly diffusive; HLLC resolves contact waves.

**HLLC Concept**
- Estimate left/right wave speeds $S_L, S_R$.
- Compute contact speed $S_M$.
- Construct star states $U_L^*, U_R^*$.

**Implementation**
- Add `hllc_flux(UL, UR, gamma)` in `riemann.hpp`.
- Add enum `Riemann::Rusanov | Riemann::HLLC`.

✅ **Done When**
- Contact discontinuity captured sharply with fewer grid points.

---

### **A4. Boundary Conditions — Refactor**

Create a dedicated `bc.hpp` containing:
- `Periodic`
- `CopyOutflow`
- `Reflective`
- (Later) `FixedPressure`, `Inflow`

✅ **Done When**
- Changing BCs is a one-line switch in `apps/laser1d.cpp`.

---

## 🌐 Stage B — 2D Finite-Volume Framework

### **B1. Grid & Data Layout**

Structured grid:

$$
i = 0..n_x-1, \quad j = 0..n_y-1
$$

Flatten storage for cache efficiency:

$$
\text{idx}(i,j) = i + n_x j
$$

Define `Grid2D {nx, ny, Lx, Ly, dx, dy, x(i), y(j)}` in `mesh2d.hpp`.

---

### **B2. 2D Euler Equations**

$$
\frac{dU_{i,j}}{dt} =
 -\frac{F_{i+\tfrac{1}{2},j}-F_{i-\tfrac{1}{2},j}}{\Delta x}
 -\frac{G_{i,j+\tfrac{1}{2}}-G_{i,j-\tfrac{1}{2}}}{\Delta y}
 + S_{i,j}
$$

where:
- $F(U)$ = flux in x-direction  
- $G(U)$ = flux in y-direction (swap $u \leftrightarrow v$)

and

$$
U = [\rho, \rho u, \rho v, E]
$$

---

### **B3. Reconstruction & Splitting**

Use **dimensional splitting (Strang)**:
1. Update along x-direction.  
2. Update along y-direction.  
3. Combine results.

Reconstruction: start with **WENO3 (component-wise)** in each direction.

---

### **B4. CFL in 2D**

$$
\Delta t = \text{CFL}\cdot
\min\left(
\frac{\Delta x}{\max(|u|+c)},
\frac{\Delta y}{\max(|v|+c)}
\right)
$$

or conservative estimate:

$$
\Delta t = \frac{\text{CFL}}{\lambda_x + \lambda_y},
\quad
\lambda_x = \frac{|u|+c}{\Delta x},\;
\lambda_y = \frac{|v|+c}{\Delta y}
$$

✅ **Done When**
- 2D Sod or vortex advection runs stable and physically correct.

---

## ☄️ Stage C — Laser Ablation Physics

### **C1. Energy Source (Beer–Lambert Absorption)**

$$
S_E(x,t) = I_0(t)\,\alpha\,e^{-\alpha x}
$$

where:
- $I_0(t)$ = incident laser intensity  
- $\alpha = 1/\delta$ = absorption coefficient (penetration depth)

**Implementation**
- Add `source.hpp` with:
  ```cpp
  double BeerLambert(double x, double t, double I0, double alpha);
