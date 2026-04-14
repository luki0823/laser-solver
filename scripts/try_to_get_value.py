import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path

# Set specific sizes for axis labels and tick labels
plt.rcParams['axes.labelsize'] = 14   # Fontsize of the x and y axis labels
plt.rcParams['xtick.labelsize'] = 14  # Fontsize of the x tick labels
plt.rcParams['ytick.labelsize'] = 14  # Fontsize of the y tick labels

# --------------------------------------------------
# SETTINGS
# --------------------------------------------------
run_folder = Path("runs/2026-03-26_21-37-44_512x512_omp12")
schemes = ["FirstOrder", "WENO3", "WENO5", "TENO5"]
field_name = "p"      # "p", "rho", "u", "v", etc.
step_index = -1       # last timestep

# Slice control:
# direction = "x"  -> plot field vs x at fixed y
# direction = "y"  -> plot field vs y at fixed x
direction = "x"

# Put either "center" or a number
slice_value = 0.0000048828125
# examples:
# slice_value = "center"
# slice_value = 0.001
# slice_value = 0.0035

# --------------------------------------------------
# HELPERS
# --------------------------------------------------
def load_step_csv(file_path):
    df = pd.read_csv(file_path, comment="#")

    x_vals = np.sort(df["x"].unique())
    y_vals = np.sort(df["y"].unique())

    nx = len(x_vals)
    ny = len(y_vals)

    fields = {}
    for col in ["rho", "u", "v", "p", "E", "rhou", "rhov", "H"]:
        fields[col] = df[col].to_numpy().reshape(ny, nx)

    with open(file_path, "r") as f:
        first_line = f.readline().strip()
    time = float(first_line.split("=")[1])

    return x_vals, y_vals, fields, time


def get_step_file(scheme, step_index=-1):
    data_dir = run_folder / f"{scheme}_512x512_omp12" / "data"

    files = sorted(
        glob.glob(str(data_dir / "step_*.csv")),
        key=lambda f: int(Path(f).stem.split("_")[1])
    )

    if not files:
        raise FileNotFoundError(f"No step files found for {scheme} in {data_dir}")

    return files[step_index]


def nearest_index(array, value):
    return int(np.argmin(np.abs(array - value)))


def extract_slice(x, y, field, direction="x", slice_value="center"):
    """
    direction = "x": field vs x at fixed y
    direction = "y": field vs y at fixed x
    """

    if direction == "x":
        if slice_value == "center":
            j = len(y) // 2
        else:
            j = nearest_index(y, float(slice_value))

        coord = x
        values = field[j, :]
        actual_slice = y[j]
        axis_label = "x [mm]"
        slice_label = f"y = {actual_slice:.6e}"

    elif direction == "y":
        if slice_value == "center":
            i = len(x) // 2
        else:
            i = nearest_index(x, float(slice_value))

        coord = y
        values = field[:, i]
        actual_slice = x[i]
        axis_label = "y [mm]"
        slice_label = f"x = {actual_slice:.6e}"

    else:
        raise ValueError("direction must be 'x' or 'y'")

    return coord, values, actual_slice, axis_label, slice_label


# --------------------------------------------------
# PLOT
# --------------------------------------------------
plt.figure(figsize=(15, 12))

times = {}

line_styles = ["-", "--", "-.", ":"]
#markers = ["o", "s", "d", "^"]
linewidths = [2.5, 2.5, 2.5, 2.5]

for i, scheme in enumerate(schemes):
    file_to_plot = get_step_file(scheme, step_index=step_index)

    x, y, fields, time = load_step_csv(file_to_plot)

    coord, values, actual_slice, axis_label, slice_label = extract_slice(
        x, y, fields[field_name],
        direction=direction,
        slice_value=slice_value
    )

    plt.plot(
        coord*1000,
        values/1e3,
        linestyle=line_styles[i % len(line_styles)],
        markevery=40,  # <-- spacing between markers (key for readability)
        linewidth=linewidths[i % len(linewidths)],
        label=scheme
    )

ylabel_map = {
    "p": "Pressure [kPa]",
    "rho": "Density [kg/m$^3$]",
    "u": "u velocity [m/s]",
    "v": "v velocity [m/s]",
    "E": "Energy",
    "rhou": "rho*u",
    "rhov": "rho*v",
    "H": "Enthalpy"
}

plt.xlabel(axis_label)
plt.ylabel(ylabel_map.get(field_name, field_name))
#plt.title(f"{field_name} comparison across schemes at {slice_label}")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.xlim(0,1.5)
plt.xticks(np.linspace(0,1.5,120))
#plt.savefig(f"Figures/allscheme_shock_slice_{direction}_2ms.png", dpi = 300)
plt.show()

print("\nRequested slice value:", slice_value)
print("Actual slice used:", slice_label)

print("\nTimes used:")
for scheme, time in times.items():
    print(f"{scheme}: {time:.6e} s")

