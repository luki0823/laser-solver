#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import glob
import os

# ----------------------------------------
# Config: where the 2D data lives
# ----------------------------------------
data_dir = "data/error2d_512"#"data/out2d"  # must match [output].folder in shocktube2d.toml

# ----------------------------------------
# Discover CSV frames
# ----------------------------------------
files = sorted(glob.glob(os.path.join(data_dir, "step_*.csv")))
if not files:
    raise RuntimeError(f"No output CSVs found in {data_dir}")

print(f"Found {len(files)} frames in {data_dir}")

# ----------------------------------------
# Helper: read time from first line of CSV
# ----------------------------------------
def read_time_from_csv(path):
    """
    Expects first line like: '# time = 0.0012345'
    Returns float(t) or None if not found/parsable.
    """
    with open(path, "r") as f:
        first = f.readline().strip()
    if first.startswith("# time"):
        try:
            t_str = first.split("=", 1)[1].strip()
            return float(t_str)
        except Exception:
            return None
    return None

# ----------------------------------------
# Load first frame to determine grid + available fields
# ----------------------------------------
first_path = files[0]
t0 = read_time_from_csv(first_path)
first = pd.read_csv(first_path, comment="#")

# spatial coords
x_vals = np.sort(first["x"].unique())
y_vals = np.sort(first["y"].unique())
nx = len(x_vals)
ny = len(y_vals)
print(f"Grid: nx = {nx}, ny = {ny}")

# fields that can be animated (anything that's not x or y)
all_cols = list(first.columns)
fields = [c for c in all_cols if c not in ("x", "y")]
print("Available fields:", fields)

# choose field
field = input(f"Field to animate {fields} [default: 'rho' if present, else first]: ").strip()
if not field:
    field = "rho" if "rho" in fields else fields[0]
if field not in fields:
    print(f"Field '{field}' not found, using '{fields[0]}' instead.")
    field = fields[0]

print(f"Animating field: {field}")

# ----------------------------------------
# Preload all frames for the chosen field
# ----------------------------------------
frames = []
times = []
for path in files:
    t = read_time_from_csv(path)
    times.append(t)

    df = pd.read_csv(path, comment="#")
    vals = df[field].values.reshape(ny, nx)  # assumes same ordering every time
    frames.append(vals)

frames = np.array(frames)  # shape: (nframes, ny, nx)

# global min/max for color scale
vmin = np.min(frames)
vmax = np.max(frames)
print(f"{field} range over all frames: [{vmin}, {vmax}]")

# ----------------------------------------
# Optional: base title
# ----------------------------------------
base_title = input("Enter a base title for the animation [default: '2D Shock/Blast']: ").strip()
if not base_title:
    base_title = "2D Shock/Blast"

# ----------------------------------------
# Set up figure + initial image
# ----------------------------------------
fig, ax = plt.subplots(figsize=(6, 5))

# extent for imshow: [xmin, xmax, ymin, ymax]
extent = [x_vals[0], x_vals[-1], y_vals[0], y_vals[-1]]

im = ax.imshow(frames[0],
               origin="lower",
               extent=extent,
               aspect="equal",
               vmin=vmin,
               vmax=vmax)

cbar = fig.colorbar(im, ax=ax)
cbar.set_label(field)

ax.set_xlabel("x")
ax.set_ylabel("y")

if times[0] is not None:
    title = ax.set_title(f"{base_title} – {field}, t = {times[0]:.2e} s")
else:
    title = ax.set_title(f"{base_title} – {field}, frame 0")

plt.tight_layout()

# ----------------------------------------
# Animation callbacks
# ----------------------------------------
def init():
    im.set_data(frames[0])
    if times[0] is not None:
        title.set_text(f"{base_title} – {field}, t = {times[0]:.2e} s")
    else:
        title.set_text(f"{base_title} – {field}, frame 0")
    return im, title

def update(i):
    im.set_data(frames[i])
    t = times[i]
    if t is not None:
        title.set_text(f"{base_title} – {field}, t = {t:.6e} s")
    else:
        title.set_text(f"{base_title} – {field}, frame {i}")
    return im, title

ani = FuncAnimation(
    fig,
    update,
    frames=len(frames),
    init_func=init,
    blit=False,
    interval=100,   # ms between frames
)

# Show preview (non-blocking so we can still ask about saving)
plt.show(block=False)

# ----------------------------------------
# Optional: save to MP4
# ----------------------------------------
decision = input("Do you want to save the animation? [y/n]: ").strip().lower()

if decision == "y":
    animation_name = input("Enter a name for the animation (without extension): ").strip()
    if not animation_name:
        animation_name = f"{field}_2d"

    fps_str = input("Enter FPS [default: 30]: ").strip()
    fps = int(fps_str) if fps_str else 30

    out_file = animation_name + ".mp4"
    print(f"Saving animation to {out_file} (this may take a moment)...")

    writer = FFMpegWriter(fps=fps, codec="libx264")
    ani.save(out_file, writer=writer)

    print(f"Saved as {out_file}")

# keep window open
plt.show()
