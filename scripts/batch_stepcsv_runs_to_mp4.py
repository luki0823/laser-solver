#!/usr/bin/env python3
"""
batch_stepcsv_runs_to_mp4.py

Batch-convert runs/*/*/data/step_*.csv into MP4s.

Your layout:
runs/
  <timestamp_dir>/
    <scheme>_<NX>x<NY>_omp<CORES>/
      data/
        step_0000.csv
        step_0100.csv
        ...

Each step CSV:
  # time = ...
  x,y,rho,u,v,p,E,rhou,rhov,H
  <nx*ny rows>

Output:
MP4_Runs/<timestamp_dir>/<run_dir>/<field>/<field>__<scheme>__nx<NX>ny<NY>.mp4

Features:
- Supports run folder names:
    * WENO5_nx200ny200_omp12
    * FirstOrder_250x250_omp12
- Looks for step files in run_dir/data (fallback to recursive)
- Builds ONE colorbar per MP4 (no stacking)
- Optional fixed color limits across frames: --fixed_limits
- Verbose progress logging: --verbose
- Frame subsampling: --stride N
- Debug-style progress lines:
    [run] <timestamp>/<run> : found N frames
    [mp4] field=... -> output.mp4

Requirements:
  pip install numpy pandas matplotlib
  and ffmpeg available on PATH.
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter


# Run folder formats supported:
#   WENO5_nx200ny200_omp12
#   FirstOrder_250x250_omp12
RUN_RE_A = re.compile(r"^(?P<scheme>.+?)_nx(?P<nx>\d+)ny(?P<ny>\d+)_omp(?P<omp>\d+)$")
RUN_RE_B = re.compile(r"^(?P<scheme>.+?)_(?P<nx>\d+)x(?P<ny>\d+)_omp(?P<omp>\d+)$")

# Frame files:
#   step_0000.csv, step_0100.csv, ...
STEP_RE = re.compile(r"^step_(?P<step>\d+)\.csv$")


def vlog(msg: str, verbose: bool) -> None:
    if verbose:
        print(msg)


def parse_run_folder_name(name: str) -> Optional[Dict[str, str]]:
    m = RUN_RE_A.match(name)
    if m:
        return m.groupdict()
    m = RUN_RE_B.match(name)
    if m:
        return m.groupdict()
    return None


def find_timestamp_dirs(runs_dir: Path) -> List[Path]:
    return sorted([p for p in runs_dir.iterdir() if p.is_dir()])


def find_run_dirs(ts_dir: Path) -> List[Path]:
    out: List[Path] = []
    for p in ts_dir.iterdir():
        if p.is_dir() and parse_run_folder_name(p.name):
            out.append(p)
    return sorted(out)


def gather_step_files(run_dir: Path) -> List[Tuple[int, Path]]:
    """
    Prefer run_dir/data/*.csv. If missing, fall back to run_dir/**/step_*.csv.
    """
    data_dir = run_dir / "data"
    if data_dir.exists() and data_dir.is_dir():
        candidates = [p for p in data_dir.iterdir() if p.is_file()]
    else:
        candidates = [p for p in run_dir.rglob("step_*.csv") if p.is_file()]

    steps: List[Tuple[int, Path]] = []
    for p in candidates:
        m = STEP_RE.match(p.name)
        if not m:
            continue
        steps.append((int(m.group("step")), p))

    steps.sort(key=lambda t: t[0])
    return steps


def read_step_csv(fp: Path) -> pd.DataFrame:
    """
    Reads a step CSV ignoring the '# time = ...' comment.
    """
    return pd.read_csv(fp, comment="#")


def infer_nx_ny_from_xy(df: pd.DataFrame) -> Tuple[int, int]:
    xs = np.sort(df["x"].unique())
    ys = np.sort(df["y"].unique())
    return int(xs.size), int(ys.size)


def step_df_to_grid(df: pd.DataFrame, nx: int, ny: int, field: str) -> np.ndarray:
    """
    Converts point-list (x,y,...) to a (ny,nx) grid by sorting (y,x) then reshaping.

    If nx*ny does not match row count, tries to infer nx/ny from unique x/y.
    """
    if field not in df.columns:
        raise KeyError(f"Field '{field}' not found. Columns={list(df.columns)}")
    if "x" not in df.columns or "y" not in df.columns:
        raise KeyError("CSV must contain 'x' and 'y' columns.")

    nrows = df.shape[0]
    if nrows != nx * ny:
        nx2, ny2 = infer_nx_ny_from_xy(df)
        if nx2 * ny2 != nrows:
            raise RuntimeError(
                f"Row count {nrows} != nx*ny ({nx}*{ny}={nx*ny}) "
                f"and cannot infer a structured grid from unique x/y."
            )
        nx, ny = nx2, ny2

    # Sort by y then x so we can reshape to scanlines.
    order = np.lexsort((df["x"].to_numpy(), df["y"].to_numpy()))
    vals = df[field].to_numpy()[order]
    return vals.reshape((ny, nx))


def compute_global_limits(
    step_files: List[Tuple[int, Path]],
    nx: int,
    ny: int,
    field: str,
    stride: int,
) -> Tuple[float, float]:
    vmin = np.inf
    vmax = -np.inf

    for k, (_, fp) in enumerate(step_files):
        if stride > 1 and (k % stride != 0):
            continue
        df = read_step_csv(fp)
        grid = step_df_to_grid(df, nx, ny, field)
        vmin = min(vmin, float(np.nanmin(grid)))
        vmax = max(vmax, float(np.nanmax(grid)))

    if not np.isfinite(vmin) or not np.isfinite(vmax):
        vmin, vmax = 0.0, 1.0
    if vmin == vmax:
        vmax = vmin + 1e-12
    return vmin, vmax


def render_mp4_from_steps(
    step_files: List[Tuple[int, Path]],
    out_mp4: Path,
    scheme: str,
    nx: int,
    ny: int,
    field: str,
    fps: int,
    stride: int,
    fixed_limits: bool,
    verbose: bool,
) -> None:
    out_mp4.parent.mkdir(parents=True, exist_ok=True)

    # Initialize from first frame we will actually render (respect stride)
    first_idx = 0
    if stride > 1:
        first_idx = 0  # we always render k=0
    step0, fp0 = step_files[first_idx]

    df0 = read_step_csv(fp0)
    grid0 = step_df_to_grid(df0, nx, ny, field)

    if fixed_limits:
        vmin, vmax = compute_global_limits(step_files, nx, ny, field, stride=stride)
    else:
        vmin = float(np.nanmin(grid0))
        vmax = float(np.nanmax(grid0))
        if vmin == vmax:
            vmax = vmin + 1e-12

    # --- Figure setup (ONE imshow + ONE colorbar) ---
    fig = plt.figure(figsize=(7, 6))
    ax = plt.gca()
    ax.set_xticks([])
    ax.set_yticks([])

    im = ax.imshow(grid0, origin="lower", vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    writer = FFMpegWriter(fps=fps, metadata={"artist": "laser_solver batch tool"})

    vlog(f"    ▶ building MP4: {out_mp4.name}", verbose)
    t0 = time.time()

    with writer.saving(fig, str(out_mp4), dpi=160):
        for k, (step, fp) in enumerate(step_files):
            if stride > 1 and (k % stride != 0):
                continue

            df = read_step_csv(fp)
            grid = step_df_to_grid(df, nx, ny, field)

            # Update image data
            im.set_data(grid)

            # Update limits if not fixed
            if not fixed_limits:
                vmin_f = float(np.nanmin(grid))
                vmax_f = float(np.nanmax(grid))
                if vmin_f == vmax_f:
                    vmax_f = vmin_f + 1e-12
                im.set_clim(vmin_f, vmax_f)
                cbar.update_normal(im)

            ax.set_title(f"{field} | {scheme} | nx={nx}, ny={ny} | step={step}")
            writer.grab_frame()

    dt = time.time() - t0
    vlog(f"    ✔ finished MP4: {out_mp4.name} ({dt:.2f}s)", verbose)

    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", type=Path, default=Path("runs"))
    ap.add_argument("--out_dir", type=Path, default=Path("MP4_Runs"))
    ap.add_argument(
        "--fields",
        type=str,
        default="p,rho,u,v,E",
        help="Comma-separated list of fields (must match CSV header).",
    )
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--stride", type=int, default=1, help="Use every Nth frame.")
    ap.add_argument(
        "--fixed_limits",
        action="store_true",
        help="Use one color scale across all frames per (run, field).",
    )
    ap.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress (run discovery, output paths, timings).",
    )
    args = ap.parse_args()

    runs_dir: Path = args.runs_dir
    out_dir: Path = args.out_dir
    fields = [f.strip() for f in args.fields.split(",") if f.strip()]
    verbose = args.verbose

    if not runs_dir.exists():
        raise SystemExit(f"runs_dir does not exist: {runs_dir}")

    ts_dirs = find_timestamp_dirs(runs_dir)
    if not ts_dirs:
        raise SystemExit(f"No timestamp folders found under: {runs_dir}")

    for ts_dir in ts_dirs:
        run_dirs = find_run_dirs(ts_dir)
        if not run_dirs:
            continue

        for run_dir in run_dirs:
            meta = parse_run_folder_name(run_dir.name)
            if not meta:
                continue

            scheme = meta["scheme"]
            nx = int(meta["nx"])
            ny = int(meta["ny"])

            step_files = gather_step_files(run_dir)
            if len(step_files) < 2:
                vlog(f"[skip] {ts_dir.name}/{run_dir.name} (not enough frames)", verbose)
                continue

            print(f"[run] {ts_dir.name}/{run_dir.name} : found {len(step_files)} frames")
            vlog(f"  ↳ parsed scheme={scheme} nx={nx} ny={ny}", verbose)
            vlog(f"  ↳ data dir={(run_dir / 'data')}", verbose)

            for field in fields:
                out_mp4 = (
                    out_dir
                    / ts_dir.name
                    / run_dir.name
                    / field
                    / f"{field}__{scheme}__nx{nx}ny{ny}.mp4"
                )

                print(f"[mp4] {ts_dir.name}/{run_dir.name} : saving {field} -> {out_mp4}")
                try:
                    render_mp4_from_steps(
                        step_files=step_files,
                        out_mp4=out_mp4,
                        scheme=scheme,
                        nx=nx,
                        ny=ny,
                        field=field,
                        fps=args.fps,
                        stride=args.stride,
                        fixed_limits=args.fixed_limits,
                        verbose=verbose,
                    )
                except Exception as e:
                    print(f"  !! failed field={field}: {e}")

    print("Done.")


if __name__ == "__main__":
    main()
