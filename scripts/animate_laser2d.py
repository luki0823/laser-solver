#!/usr/bin/env python3
"""
animate_steps.py

Animate your Laser2D / Euler2D CSV outputs of the form:

    # time = 0
    x,y,rho,u,v,p,E,rhou,rhov,H
    6e-06,6e-06,1.78,0,0,210000000,318181818.18,0,0,296731358.53
    ...

Robust features:
- Skips comment lines starting with '#'
- Reads proper comma-separated columns
- Reshapes scattered x,y points into a regular (ny, nx) grid
- Supports derived fields: mach, T
"""

from __future__ import annotations

import argparse
import glob
import os
import re
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# ----------------------------
# Utilities
# ----------------------------

_TIME_RE = re.compile(r"^\s*#\s*time\s*=\s*([+-]?\d+(\.\d+)?([eE][+-]?\d+)?)\s*$")


def parse_time_from_file(filepath: str) -> Optional[float]:
    """Reads only the first ~5 lines and extracts '# time = ...' if present."""
    try:
        with open(filepath, "r") as f:
            for _ in range(5):
                line = f.readline()
                if not line:
                    break
                m = _TIME_RE.match(line)
                if m:
                    return float(m.group(1))
    except OSError:
        return None
    return None


def read_step_csv(filepath: str) -> pd.DataFrame:
    """
    Reads one step_XXXX.csv file robustly.

    - comment="#" drops the '# time = ...' line
    - expects header: x,y,rho,u,v,p,E,rhou,rhov,H
    """
    df = pd.read_csv(filepath, comment="#")
    # Basic sanity check
    required = {"x", "y"}
    if not required.issubset(df.columns):
        raise ValueError(f"{filepath}: missing required columns {required}. Found: {list(df.columns)}")
    return df


def reshape_to_grid(df: pd.DataFrame, field: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Reshape (x,y,field) into regular mesh arrays (X, Y, Z) with shape (ny, nx).

    Works as long as x and y form a tensor-product grid (which your output does).
    """
    if field not in df.columns:
        raise KeyError(f"Field '{field}' not in columns: {list(df.columns)}")

    # Unique sorted coordinates
    xs = np.sort(df["x"].unique())
    ys = np.sort(df["y"].unique())
    nx = xs.size
    ny = ys.size

    # Map each (x,y) row into indices
    x_to_i = {x: i for i, x in enumerate(xs)}
    y_to_j = {y: j for j, y in enumerate(ys)}

    Z = np.full((ny, nx), np.nan, dtype=float)

    # Fill Z (vectorized-ish approach)
    xi = df["x"].map(x_to_i).to_numpy()
    yj = df["y"].map(y_to_j).to_numpy()
    vals = df[field].to_numpy(dtype=float)

    Z[yj, xi] = vals

    # Build mesh for plotting
    X, Y = np.meshgrid(xs, ys)
    return X, Y, Z


def compute_derived_fields(df: pd.DataFrame, gamma: float) -> Dict[str, np.ndarray]:
    """
    Compute derived fields from columns if available.
    Returns dict mapping derived_name -> numpy array aligned with df rows.
    """
    derived: Dict[str, np.ndarray] = {}

    # velocity magnitude
    if {"u", "v"}.issubset(df.columns):
        u = df["u"].to_numpy(float)
        v = df["v"].to_numpy(float)
        derived["speed"] = np.sqrt(u * u + v * v)

    # Mach number: |u| / a, with a = sqrt(gamma*p/rho)
    if {"p", "rho", "u", "v"}.issubset(df.columns):
        rho = df["rho"].to_numpy(float)
        p = df["p"].to_numpy(float)
        a = np.sqrt(np.maximum(gamma * p / np.maximum(rho, 1e-300), 0.0))
        speed = derived.get("speed", np.sqrt(df["u"].to_numpy(float) ** 2 + df["v"].to_numpy(float) ** 2))
        derived["mach"] = speed / np.maximum(a, 1e-300)

    # Temperature (if ideal gas and you want it): T = p / rho (assuming R=1 nondimensional)
    if {"p", "rho"}.issubset(df.columns):
        rho = df["rho"].to_numpy(float)
        p = df["p"].to_numpy(float)
        derived["T"] = p / np.maximum(rho, 1e-300)

    return derived


# ----------------------------
# Animation
# ----------------------------

@dataclass
class Options:
    data_dir: str
    field: str
    gamma: float
    stride: int
    fixed_limits: bool
    vmin: Optional[float]
    vmax: Optional[float]
    save_mp4: bool
    mp4_path: str
    fps: int


def find_step_files(data_dir: str) -> List[str]:
    files = sorted(glob.glob(os.path.join(data_dir, "step_*.csv")))
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description="Animate step_*.csv outputs.")
    parser.add_argument("--data_dir", default=None, help="Folder containing step_*.csv (default: prompt)")
    parser.add_argument("--field", default=None, help="Field to plot: rho,p,u,v,E,rhou,rhov,H,mach,T (default: prompt)")
    parser.add_argument("--gamma", type=float, default=None, help="Gamma for derived fields (mach). Default: prompt (1.4)")
    parser.add_argument("--stride", type=int, default=None, help="Frame stride (default: prompt (1))")
    parser.add_argument("--fixed_limits", action="store_true", help="Use fixed color limits (vmin/vmax).")
    parser.add_argument("--vmin", type=float, default=None, help="Fixed vmin (only if --fixed_limits).")
    parser.add_argument("--vmax", type=float, default=None, help="Fixed vmax (only if --fixed_limits).")
    parser.add_argument("--save", action="store_true", help="Save MP4 instead of (or in addition to) showing.")
    parser.add_argument("--mp4", default="animation.mp4", help="MP4 output path (default: animation.mp4)")
    parser.add_argument("--fps", type=int, default=15, help="FPS for MP4 (default: 30)")
    args = parser.parse_args()

    print("\n🎞️  CSV Animation Tool\n")

    data_dir = args.data_dir
    if not data_dir:
        data_dir = input("Data directory [data/laser2d]: ").strip() or "data/laser2d"

    files = find_step_files(data_dir)
    if not files:
        raise SystemExit(f"❌ No files found matching {os.path.join(data_dir, 'step_*.csv')}")

    print(f"📂 Found {len(files)} frames")

    field = args.field
    if not field:
        field = input("Field (rho, p, u, v, E, rhou, rhov, H, mach, T) [p]: ").strip() or "p"

    gamma = args.gamma
    if gamma is None:
        g_in = input("Gamma [1.4]: ").strip()
        gamma = float(g_in) if g_in else 1.4

    stride = args.stride
    if stride is None:
        s_in = input("Frame stride (1 = every frame) [1]: ").strip()
        stride = int(s_in) if s_in else 1
    stride = max(1, stride)

    fixed_limits = args.fixed_limits
    if not args.fixed_limits:
        fl = input("Use fixed color limits? [y/N]: ").strip().lower()
        fixed_limits = (fl == "y")

    vmin = args.vmin
    vmax = args.vmax
    if fixed_limits and (vmin is None or vmax is None):
        lim = input("Enter vmin,vmax (comma-separated) [auto from first frame]: ").strip()
        if lim:
            parts = [p.strip() for p in lim.split(",")]
            if len(parts) != 2:
                raise SystemExit("❌ Expected two numbers: vmin,vmax")
            vmin, vmax = float(parts[0]), float(parts[1])

    save_mp4 = args.save
    if not args.save:
        sm = input("Save animation to MP4? [Y/n]: ").strip().lower()
        save_mp4 = (sm != "n")

    mp4_path = args.mp4
    fps = args.fps

    # Pre-sample first file to set up axes
    df0 = read_step_csv(files[0])
    derived0 = compute_derived_fields(df0, gamma)

    if field in derived0:
        df0 = df0.copy()
        df0[field] = derived0[field]

    X, Y, Z0 = reshape_to_grid(df0, field)

    # Determine vmin/vmax if fixed limits requested but not specified
    if fixed_limits and (vmin is None or vmax is None):
        finite = Z0[np.isfinite(Z0)]
        if finite.size == 0:
            vmin, vmax = 0.0, 1.0
        else:
            vmin, vmax = float(np.min(finite)), float(np.max(finite))

    # Plot setup
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    t0 = parse_time_from_file(files[0])
    title_time = f"t = {t0:.2e}" if t0 is not None else os.path.basename(files[0])
    ax.set_title(f"Pressure Field | {title_time}) #{field}  | ({title_time})")

    # Use imshow for speed; extent uses physical coordinates
    extent = [X.min(), X.max(), Y.min(), Y.max()]
    if fixed_limits:
        im = ax.imshow(Z0, origin="lower", extent=extent, aspect="auto", vmin=vmin, vmax=vmax)
    else:
        im = ax.imshow(Z0, origin="lower", extent=extent, aspect="auto")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(field)

    # Animation function
    frame_indices = list(range(0, len(files), stride))

    def update(k: int):
        idx = frame_indices[k]
        fp = files[idx]
        df = read_step_csv(fp)
        derived = compute_derived_fields(df, gamma)
        if field in derived:
            df = df.copy()
            df[field] = derived[field]

        _, _, Z = reshape_to_grid(df, field)

        # Update image
        im.set_data(Z)

        # Autoscale if not fixed
        if not fixed_limits:
            finite = Z[np.isfinite(Z)]
            if finite.size > 0:
                im.set_clim(float(np.min(finite)), float(np.max(finite)))

        t = parse_time_from_file(fp)
        title_time2 = f"t = {t:.2e}" if t is not None else os.path.basename(fp)
        ax.set_title(f"Pressure Field | {title_time2}")#{field}  | ({title_time2})")
        return (im,)

    anim = FuncAnimation(fig, update, frames=len(frame_indices), interval=33, blit=False)

    if save_mp4:
        # Requires ffmpeg installed
        print(f"💾 Saving MP4 to: {mp4_path}")
        anim.save(mp4_path, fps=fps, dpi=150)

    plt.show()


if __name__ == "__main__":
    main()
