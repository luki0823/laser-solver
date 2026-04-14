import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

# =========================
# User settings
# =========================
CSV_PATH = "convergence_results.csv"   # change if needed
SAVE_DIR = "Figures"                   # set to None to disable saving
DPI = 200

# Only keep these schemes for convergence study
SCHEMES_TO_USE = ["FirstOrder", "WENO3", "WENO5", "TENO5"]

# Which norm to use in the main convergence plots
NORM_TO_PLOT = "L2_rho"   # options: "L1_rho", "L2_rho", "Linf_rho"

if SAVE_DIR is not None:
    os.makedirs(SAVE_DIR, exist_ok=True)

plt.rcParams.update({
    "figure.figsize": (7.5, 5.0),
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
})

# =========================
# Load data
# =========================
df = pd.read_csv(CSV_PATH)

required = {"scheme", "nx", "ny", "h", "L1_rho", "L2_rho", "Linf_rho", "wall_s"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"CSV missing required columns: {missing}")

# Keep only selected schemes
df = df[df["scheme"].isin(SCHEMES_TO_USE)].copy()

# Optional safety: keep only square grids
df = df[df["nx"] == df["ny"]].copy()

# Enforce scheme order in plots/tables
df["scheme"] = pd.Categorical(df["scheme"], categories=SCHEMES_TO_USE, ordered=True)

# Sort for consistent processing
df.sort_values(["scheme", "h"], ascending=[True, False], inplace=True)

if df.empty:
    raise ValueError("No data left after filtering. Check SCHEMES_TO_USE and CSV contents.")

print("Schemes included:", list(df["scheme"].cat.categories))
print("Schemes found in filtered data:", sorted(df["scheme"].dropna().unique()))
print("Resolutions:", sorted(df["nx"].unique()))
display(df.head(10))

# =========================
# Helper functions
# =========================
def observed_orders(sub: pd.DataFrame, err_col: str) -> pd.DataFrame:
    """
    Compute observed order p between successive refinements:
        p = log(e_coarse/e_fine) / log(h_coarse/h_fine)
    """
    s = sub.sort_values("h", ascending=False).reset_index(drop=True)

    if len(s) < 2:
        return pd.DataFrame(columns=["nx_coarse", "nx_fine", "h_coarse", "h_fine", f"p_{err_col}"])

    h = s["h"].to_numpy()
    e = s[err_col].to_numpy()

    # Guard against invalid values
    valid = (h[:-1] > 0) & (h[1:] > 0) & (e[:-1] > 0) & (e[1:] > 0)
    p = np.full(len(h) - 1, np.nan)
    p[valid] = np.log(e[:-1][valid] / e[1:][valid]) / np.log(h[:-1][valid] / h[1:][valid])

    out = pd.DataFrame({
        "nx_coarse": s["nx"].iloc[:-1].to_numpy(),
        "nx_fine":   s["nx"].iloc[1:].to_numpy(),
        "h_coarse":  h[:-1],
        "h_fine":    h[1:],
        f"p_{err_col}": p
    })
    return out

def loglog_fit_slope(h: np.ndarray, e: np.ndarray) -> float:
    """
    Fit slope in log10 space:
        log10(e) = a + p*log10(h)
    """
    valid = (h > 0) & (e > 0)
    h = h[valid]
    e = e[valid]

    if len(h) < 2:
        return np.nan

    x = np.log10(h)
    y = np.log10(e)
    p, _ = np.polyfit(x, y, 1)
    return p

# =========================
# Compute order tables
# =========================
err_cols = ["L1_rho", "L2_rho", "Linf_rho"]

order_rows = []
pairwise_tables = {}

for scheme, sub in df.groupby("scheme", observed=True):
    pairwise_tables[scheme] = {}

    for col in err_cols:
        pw = observed_orders(sub, col)
        pairwise_tables[scheme][col] = pw

        pcol = f"p_{col}"
        if len(pw) > 0 and pw[pcol].notna().any():
            p_mean = pw[pcol].mean()
            p_min = pw[pcol].min()
            p_max = pw[pcol].max()
        else:
            p_mean = np.nan
            p_min = np.nan
            p_max = np.nan

        order_rows.append({
            "scheme": scheme,
            "norm": col,
            "p_mean": p_mean,
            "p_min": p_min,
            "p_max": p_max
        })

order_summary = pd.DataFrame(order_rows).sort_values(["norm", "scheme"])
display(order_summary)

# =========================
# Plot 1: error vs h
# =========================
fig = plt.figure()

for scheme, sub in df.groupby("scheme", observed=True):
    sub = sub.sort_values("h", ascending=False)
    h = sub["h"].to_numpy()
    e = sub[NORM_TO_PLOT].to_numpy()

    p_fit = loglog_fit_slope(h, e)
    if np.isnan(p_fit):
        label = f"{scheme}"
    else:
        label = f"{scheme} (fit p={p_fit:.2f})"

    plt.loglog(h, e, marker="o", linewidth=1.8, label=label)

plt.gca().invert_xaxis()
plt.xlabel("Grid spacing h")
plt.ylabel(r"$L_2$ Error")
plt.title(fr"Convergence: $L_2$ Error vs h")
plt.legend()
plt.tight_layout()

if SAVE_DIR is not None:
    out = os.path.join(SAVE_DIR, f"convergence_{NORM_TO_PLOT}.png")
    plt.savefig(out, dpi=DPI, bbox_inches="tight")

plt.show()

# =========================
# Plot 2: observed order per refinement pair
# =========================
fig = plt.figure()

xtick_labels = None

for scheme in SCHEMES_TO_USE:
    if scheme not in pairwise_tables:
        continue

    pw = pairwise_tables[scheme][NORM_TO_PLOT]
    if pw.empty:
        continue

    xs = np.arange(len(pw))
    plt.plot(xs, pw[f"p_{NORM_TO_PLOT}"], marker="o", linewidth=1.8, label=scheme)

    # use the first valid set of refinement-pair labels
    if xtick_labels is None:
        xtick_labels = [f"{int(a)}→{int(b)}" for a, b in zip(pw["nx_coarse"], pw["nx_fine"])]

if xtick_labels is not None:
    plt.xticks(ticks=np.arange(len(xtick_labels)), labels=xtick_labels)

plt.xlabel("Refinement pair (nx coarse → nx fine)")
plt.ylabel(f"Observed order p ({NORM_TO_PLOT})")
plt.title(f"Observed order by refinement pair ({NORM_TO_PLOT})")
plt.legend()
plt.tight_layout()

if SAVE_DIR is not None:
    out = os.path.join(SAVE_DIR, f"orders_pairs_{NORM_TO_PLOT}.png")
    plt.savefig(out, dpi=DPI, bbox_inches="tight")

plt.show()

# =========================
# Plot 3: wall time scaling
# =========================
fig = plt.figure()

for scheme, sub in df.groupby("scheme", observed=True):
    sub = sub.sort_values("nx")
    cells = (sub["nx"] * sub["ny"]).to_numpy()
    wall = sub["wall_s"].to_numpy()

    valid = (cells > 0) & (wall > 0)
    cells = cells[valid]
    wall = wall[valid]

    if len(cells) == 0:
        continue

    plt.loglog(cells, wall, marker="o", linewidth=1.8, label=scheme)

plt.xlabel("Total cells (nx * ny)")
plt.ylabel("Wall time (s)")
plt.title("Runtime scaling")
plt.legend()
plt.tight_layout()

if SAVE_DIR is not None:
    out = os.path.join(SAVE_DIR, "runtime_scaling.png")
    plt.savefig(out, dpi=DPI, bbox_inches="tight")

plt.show()

# =========================
# Table for report
# =========================
pivot = (
    order_summary
    .pivot(index="scheme", columns="norm", values="p_mean")
    .reindex(SCHEMES_TO_USE)
)

display(pivot.round(3))