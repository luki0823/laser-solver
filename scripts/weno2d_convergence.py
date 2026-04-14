#!/usr/bin/env python3
import glob
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# User configuration
# ============================================================
# These runs should correspond to your configs/error2d_*.toml
# and the folders used in [output].folder.
#
# Example: we ran:
#   error2d_64.toml  -> data/error2d_64
#   error2d_128.toml -> data/error2d_128
#   error2d_256.toml -> data/error2d_256
#   error2d_512.toml -> data/error2d_512
#
runs = [
    {"nx":  64, "ny":  64, "Lx": 1.0, "Ly": 1.0, "folder": "data/error2d_64"},
    {"nx": 128, "ny": 128, "Lx": 1.0, "Ly": 1.0, "folder": "data/error2d_128"},
    {"nx": 256, "ny": 256, "Lx": 1.0, "Ly": 1.0, "folder": "data/error2d_256"},
    {"nx": 512, "ny": 512, "Lx": 1.0, "Ly": 1.0, "folder": "data/error2d_512"},
]

# These must match the parameters used in 2d_error_test.cpp / error2d_*.toml
gamma = 1.4  # not actually used for rho error, but here for clarity

# Advection IC parameters:
rho0 = 1.0
amp  = 0.1
p0   = 1.0      # constant pressure
u0   = 0.5      # constant x-velocity
v0   = 0.25     # constant y-velocity
kx   = 1.0      # mode in x
ky   = 1.0      # mode in y

field = "rho"   # which field to measure error on (density)


# ============================================================
# Helpers
# ============================================================
def find_last_csv(folder):
    """Return the last step_*.csv in folder (sorted by name)."""
    files = sorted(glob.glob(os.path.join(folder, "step_*.csv")))
    if not files:
        raise RuntimeError(f"No step_*.csv found in {folder}")
    return files[-1]


def read_time_from_csv(path):
    """Read '# time = ...' from first line of CSV."""
    with open(path, "r") as f:
        first = f.readline().strip()
    if first.startswith("# time"):
        try:
            t_str = first.split("=", 1)[1].strip()
            return float(t_str)
        except Exception:
            return None
    return None


def wrap_periodic(x, L):
    """Wrap x into [0,L) for periodicity."""
    w = np.fmod(x, L)
    w[w < 0.0] += L
    return w


def exact_rho(x, y, t, Lx, Ly):
    """
    Exact solution for rho(x,y,t) with:
      rho(t=0) = rho0 + amp sin(2π kx x/Lx) sin(2π ky y/Ly)
      u = u0, v = v0, p = p0 constant
    """
    x0 = wrap_periodic(x - u0 * t, Lx)
    y0 = wrap_periodic(y - v0 * t, Ly)

    phase_x = 2.0 * np.pi * kx * x0 / Lx
    phase_y = 2.0 * np.pi * ky * y0 / Ly

    return rho0 + amp * np.sin(phase_x) * np.sin(phase_y)


# ============================================================
# Main: load data, compute errors, print convergence, plot
# ============================================================
errors_L1 = []
errors_L2 = []
errors_Linf = []
hs = []   # grid spacing (we'll use h = Lx/nx)

for run in runs:
    nx = run["nx"]
    ny = run["ny"]
    Lx = run["Lx"]
    Ly = run["Ly"]
    folder = run["folder"]

    csv_path = find_last_csv(folder)
    print(f"Loading {csv_path} for nx={nx}, ny={ny}")
    df = pd.read_csv(csv_path, comment="#")

    # Get time from header
    t = read_time_from_csv(csv_path)
    if t is None:
        raise RuntimeError(f"Could not read time from {csv_path}")
    print(f"  final time t = {t:.6e}")

    # Extract grid
    x_vals = np.sort(df["x"].unique())
    y_vals = np.sort(df["y"].unique())

    if len(x_vals) != nx or len(y_vals) != ny:
        raise RuntimeError(
            f"Grid mismatch: expected ({nx},{ny}), "
            f"found ({len(x_vals)},{len(y_vals)}) in {csv_path}"
        )

    # Reshape rho to (ny, nx)
    rho_num = df[field].values.reshape(ny, nx)

    # Build 2D coordinate arrays
    X, Y = np.meshgrid(x_vals, y_vals)

    # Exact solution
    rho_ex = exact_rho(X, Y, t, Lx, Ly)

    # Error fields
    diff = rho_num - rho_ex
    adiff = np.abs(diff)

    N = diff.size
    L1 = np.sum(adiff) / N
    L2 = np.sqrt(np.sum(diff**2) / N)
    Linf = np.max(adiff)

    h = Lx / nx

    hs.append(h)
    errors_L1.append(L1)
    errors_L2.append(L2)
    errors_Linf.append(Linf)

    print(f"  L1   = {L1:.6e}")
    print(f"  L2   = {L2:.6e}")
    print(f"  Linf = {Linf:.6e}")
    print("")

# Convert to numpy arrays and sort by h (coarse->fine)
hs = np.array(hs)
errors_L1 = np.array(errors_L1)
errors_L2 = np.array(errors_L2)
errors_Linf = np.array(errors_Linf)

order = np.argsort(hs)[::-1]  # largest h first
hs = hs[order]
errors_L1 = errors_L1[order]
errors_L2 = errors_L2[order]
errors_Linf = errors_Linf[order]

# ------------------------------------------------------------
# Print convergence rates between successive resolutions
# ------------------------------------------------------------
print("\nConvergence rates (density):")
print("h           L1            L2            Linf         p_L1    p_L2    p_Linf")
print("----------------------------------------------------------------------------")

pL1_list = []
pL2_list = []
pLi_list = []

for i in range(len(hs)):
    h = hs[i]
    L1 = errors_L1[i]
    L2 = errors_L2[i]
    Li = errors_Linf[i]

    if i == 0:
        print(f"{h:8.3e}  {L1:12.6e}  {L2:12.6e}  {Li:12.6e}    ---     ---     ---")
    else:
        pL1 = np.log(errors_L1[i-1] / L1) / np.log(hs[i-1] / h)
        pL2 = np.log(errors_L2[i-1] / L2) / np.log(hs[i-1] / h)
        pLi = np.log(errors_Linf[i-1] / Li) / np.log(hs[i-1] / h)
        pL1_list.append(pL1)
        pL2_list.append(pL2)
        pLi_list.append(pLi)
        print(f"{h:8.3e}  {L1:12.6e}  {L2:12.6e}  {Li:12.6e}  {pL1:7.3f}  {pL2:7.3f}  {pLi:7.3f}")

print("----------------------------------------------------------------------------")

if pL1_list:
    print(f"Average orders:  p_L1 = {np.mean(pL1_list):.3f}, "
          f"p_L2 = {np.mean(pL2_list):.3f}, p_Linf = {np.mean(pLi_list):.3f}")


#set params
plt.rcParams.update({'font.size': 20})


# ------------------------------------------------------------
# Plot errors vs h (log-log)
# ------------------------------------------------------------
plt.figure(figsize=(18,16))
plt.loglog(hs, errors_L1, "o-", label="L1")
plt.loglog(hs, errors_L2, "s-", label="L2")
plt.loglog(hs, errors_Linf, "d-", label="Linf")

# Optional: annotate slopes between last two points
"""if len(hs) >= 2:
    # use the finest two for a "local" slope
    i0, i1 = -2, -1
    h0, h1 = hs[i0], hs[i1]
    e0, e1 = errors_L2[i0], errors_L2[i1]  # L2 slope as representative
    p_est = np.log(e0/e1) / np.log(h0/h1)
    txt = f"~ h^{p_est:.2f}"
    # place text near the last point
    plt.text(h1*1.1, e1*1.2, txt)
"""
plt.gca().invert_xaxis()  # finer (smaller h) on the right if you like

plt.xlabel(r"$h = \frac{L_x}{n_x} = \frac{L_y}{n_y}$")
plt.ylabel("Error in " + r"$\rho$")
plt.title("2D Advection Error vs Grid Spacing")
plt.grid(True, which="both", ls=":")
plt.legend()
plt.tight_layout()
plt.show()
