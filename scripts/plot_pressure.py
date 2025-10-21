"""
==============================================================================
 File:        animate_shocktube.py
 Purpose:     Animate 1D shock tube results from the C++ solver.
 Author:      Lucas Pierce
 Description:
   - Automatically detects whether CSVs are in data/out or build/data/out
   - Animates density, velocity, or pressure vs. x
   - Optionally saves the animation as an MP4 video

 Requirements:
   pip install pandas matplotlib
==============================================================================
"""
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import glob, os

# ------------------------------------------------------------
# 1. Automatically detect CSV output directory
# ------------------------------------------------------------
candidates = ["data/out", "build/data/out"]
data_dir = None

for d in candidates:
    if os.path.isdir(d) and glob.glob(os.path.join(d, "step_*.csv")):
        data_dir = d
        break

if data_dir is None:
    raise FileNotFoundError("No CSV files found in data/out or build/data/out. Run the solver first.")

print(f"✅ Loading shock tube data from: {data_dir}")

# ------------------------------------------------------------
# 2. Load CSV frames
# ------------------------------------------------------------
files = sorted(glob.glob(os.path.join(data_dir, "step_*.csv")))
frames = [pd.read_csv(f) for f in files]
x = frames[0]["x"].values
timesteps = len(frames)

# ------------------------------------------------------------
# 3. Select which quantity to visualize
# ------------------------------------------------------------
quantity = "p"  # options: 'rho', 'u', 'p'
y_labels = {"rho": "Density [kg/m³]", "u": "Velocity [m/s]", "p": "Pressure [Pa]"}
y_label = y_labels[quantity]

# ------------------------------------------------------------
# 4. Setup matplotlib figure
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 10))
line, = ax.plot([], [], lw=2, color="steelblue")
ax.set_xlim(x.min(), x.max())
ax.set_ylim(0, max(fr[quantity].max() for fr in frames) * 1.1)
ax.set_xlabel("x [m]")
ax.set_ylabel(y_label)
ax.set_title("Shock Tube Evolution")
ax.grid(True, alpha=0.3)

# ------------------------------------------------------------
# 5. Animation update function
# ------------------------------------------------------------
def update(i):
    data = frames[i]
    line.set_data(x, data[quantity])
    ax.set_title(f"Shock Tube Evolution — step {i+1}/{timesteps}")
    return line,

# ------------------------------------------------------------
# 6. Run animation
# ------------------------------------------------------------
ani = FuncAnimation(fig, update, frames=timesteps, interval=1000, blit=True)

# Uncomment to save animation as MP4 (requires ffmpeg)
# ani.save("shocktube.mp4", writer="ffmpeg", fps=15)
# print("🎥 Saved animation to shocktube.mp4")

plt.tight_layout()
plt.show()
