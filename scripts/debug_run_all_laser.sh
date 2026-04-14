#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# run_laser2d_sweep.sh
#
# - Runs laser2d for multiple schemes
# - Streams solver output live to terminal + logs
# - Generates MP4s for multiple fields, headless (no GUI stall)
# - Saves MP4s into: MP4_Runs/laser2d_<scheme>/<field>/
# - NEW: writes thesis diagnostics to diagnostics.csv and collects it
#
# Override via env:
#   NX=1000 NY=1000 OMP_THREADS=12 STRIDE=1 FPS=60 ./run_laser2d_sweep.sh
#
# Thesis diagnostics overrides:
#   ENABLE_THESIS_DIAG=1 DIAG_INTERVAL=100 SHOCK_PROBE_Y=-1 SHOCK_SMOOTH_FRAC=0.05 ./run_laser2d_sweep.sh
# ============================================================

# -------------------------
# Executables / scripts
# -------------------------
EXE="${EXE:-./build/bin/laser2d}"
ANIM_PY="${ANIM_PY:-./scripts/animate_laser2d.py}"

# -------------------------
# Run controls (override via env)
# -------------------------
NX="${NX:-256}"
NY="${NY:-256}"
GAMMA="${GAMMA:-1.66}"

# MP4 controls
STRIDE="${STRIDE:-2}"
FPS="${FPS:-30}"
DPI="${DPI:-160}"

# OpenMP controls
OMP_THREADS="${OMP_THREADS:-12}"
OMP_PROC_BIND="${OMP_PROC_BIND:-true}"   # true/false/close/spread
OMP_PLACES="${OMP_PLACES:-cores}"        # cores/threads/sockets

# Thesis diagnostics controls (NEW)
ENABLE_THESIS_DIAG="${ENABLE_THESIS_DIAG:-1}"  # 1/0
DIAG_INTERVAL="${DIAG_INTERVAL:-1}"          # -1 => follow out_interval (cpp default behavior)
SHOCK_PROBE_Y="${SHOCK_PROBE_Y:- -1.0}"        # <0 => Ly/2
SHOCK_SMOOTH_FRAC="${SHOCK_SMOOTH_FRAC:-0.05}" # smooth mask fraction of max|dp/dx|

# Outputs
MP4_BASE="${MP4_BASE:-MP4_Runs}"
CFG_DIR="${CFG_DIR:-configs/laser_runs}"

# Schemes
SCHEMES=("TENO5") #"FirstOrder" "WENO3" "WENO5" "TENO5"

# Fields to animate.
# These must exist in the CSV header, OR be derived by the animator (c, mach, speed).
FIELDS=("rho" "p" "T" "u" "v" "E" "H" "rhou" "rhov" "c" "mach" "speed")

# Color scaling controls
AUTO_LIMITS="${AUTO_LIMITS:-1}"          # 1 = auto update limits, 0 = fixed at first frame
LIMIT_EVERY="${LIMIT_EVERY:-5}"          # update limits every N frames
USE_PERCENTILES="${USE_PERCENTILES:-1}"  # 1 = use percentiles for limits
PLO="${PLO:-1.0}"
PHI="${PHI:-99.0}"
SMOOTH_ALPHA="${SMOOTH_ALPHA:-0.2}"      # 0..1 smoothing for limits

# Headless safety
export MPLBACKEND=Agg

# -------------------------
# Checks
# -------------------------
if [[ ! -x "$EXE" ]]; then
  echo "❌ ERROR: laser2d executable not found or not executable:"
  echo "    $EXE"
  echo
  echo "Build it with:"
  echo "  cmake -S . -B build"
  echo "  cmake --build build -j"
  exit 1
fi

if [[ ! -f "$ANIM_PY" ]]; then
  echo "❌ ERROR: animator script not found:"
  echo "    $ANIM_PY"
  exit 1
fi

mkdir -p "$CFG_DIR" "$MP4_BASE"

HAS_STDBUF=0
if command -v stdbuf >/dev/null 2>&1; then
  HAS_STDBUF=1
fi

# -------------------------
# Session naming
# -------------------------
STAMP="$(date +"%Y-%m-%d_%H-%M-%S")"
GRID_TAG="${NX}x${NY}"
THREAD_TAG="omp${OMP_THREADS}"
SESSION_DIR="runs/${STAMP}_${GRID_TAG}_${THREAD_TAG}"
mkdir -p "$SESSION_DIR"

# NEW: central diagnostics collection area
DIAG_COLLECT_DIR="${SESSION_DIR}/diagnostics"
mkdir -p "$DIAG_COLLECT_DIR"

echo "[info] session:  $SESSION_DIR"
echo "[info] grid:     $GRID_TAG"
echo "[info] omp:      OMP_NUM_THREADS=$OMP_THREADS (PROC_BIND=$OMP_PROC_BIND, PLACES=$OMP_PLACES)"
echo "[info] mp4 base: $MP4_BASE"
echo "[info] thesis diag: ENABLE=${ENABLE_THESIS_DIAG} DIAG_INTERVAL=${DIAG_INTERVAL} SHOCK_PROBE_Y=${SHOCK_PROBE_Y} SHOCK_SMOOTH_FRAC=${SHOCK_SMOOTH_FRAC}"
echo

# -------------------------
# Helpers
# -------------------------
run_solver() {
  local cfg="$1"
  local log="$2"

  if [[ "$HAS_STDBUF" == "1" ]]; then
    stdbuf -oL -eL "$EXE" "$cfg" 2>&1 | tee "$log"
  else
    "$EXE" "$cfg" 2>&1 | tee "$log"
  fi
}

as_toml_bool() {
  # accepts 1/0/true/false (case-insensitive) and prints true/false
  local v="${1}"
  shopt -s nocasematch
  if [[ "$v" == "1" || "$v" == "true" || "$v" == "yes" || "$v" == "on" ]]; then
    echo "true"
  else
    echo "false"
  fi
  shopt -u nocasematch
}

# -------------------------
# Main sweep
# -------------------------
for scheme in "${SCHEMES[@]}"; do
  CASE_DIR="${SESSION_DIR}/${scheme}_${GRID_TAG}_${THREAD_TAG}"
  DATA_DIR="${CASE_DIR}/data"
  mkdir -p "$CASE_DIR" "$DATA_DIR"

  CFG="${CFG_DIR}/laser2d_${scheme}_${GRID_TAG}.toml"

  OUT_INTERVAL=1
  # If DIAG_INTERVAL is -1, let the code follow out_interval; otherwise we set it explicitly.
  # (We still write diag_interval so the config documents itself.)
  DIAG_INTERVAL_TOML="${DIAG_INTERVAL}"

  cat > "$CFG" <<EOF
# ===========================
# 2D Laser/Plume Numerics Test
# ===========================

nx = ${NX}
ny = ${NY}

Lx = 5e-3
Ly = 5e-3

CFL   = 0.1
t_end = 5e-6
gamma = ${GAMMA}

# dt collapse diagnostics (unchanged)
enable_dt_diagnostics = false

reconstruction = "${scheme}"

out_folder   = "${DATA_DIR}"
out_interval = ${OUT_INTERVAL}

# Thesis diagnostics (NEW)
enable_thesis_diagnostics = $(as_toml_bool "${ENABLE_THESIS_DIAG}")
diag_interval = ${DIAG_INTERVAL_TOML}
shock_probe_y = ${SHOCK_PROBE_Y}
shock_smooth_frac = ${SHOCK_SMOOTH_FRAC}

laser_mode = "initial_plume"

rho_cfl_floor = 1e-4
dt_warn       = 1e-12

[bc]
left   = "Symmetry"
right  = "Outflow"
bottom = "Wall"
top    = "Outflow"

[initial_condition]
rho0 = 1.6015
u0   = 0.0
v0   = 0.0
p0   = 0.1e6

[plume_ic]
shape     = "oval_crater"
x0        = 6.0e-5
y0        = 6.0e-5
rx        = 2.5e-4
ry        = 1.0e-4
p_plume   = 0.98e6
rho_plume = 0.1883

[laser]
A  = 5e12
x0 = 1.5e-3
y0 = 1.5e-3
wx = 3.0e-4
wy = 3.0e-4
t0 = 5e-8
wt = 1e-9
EOF

  echo
  echo "============================================================"
  echo "[RUN] scheme=${scheme}  grid=${GRID_TAG}  ${THREAD_TAG}"
  echo "      cfg:  ${CFG}"
  echo "      out:  ${DATA_DIR}"
  echo "      omp:  OMP_NUM_THREADS=${OMP_THREADS} PROC_BIND=${OMP_PROC_BIND} PLACES=${OMP_PLACES}"
  echo "============================================================"
  echo

  export OMP_NUM_THREADS="$OMP_THREADS"
  export OMP_PROC_BIND="$OMP_PROC_BIND"
  export OMP_PLACES="$OMP_PLACES"

  LOG="${CASE_DIR}/run.log"
  run_solver "$CFG" "$LOG"
  cp "$CFG" "${CASE_DIR}/case.toml"

  # NEW: collect diagnostics.csv if present
  # (The updated laser2d.cpp writes it as <out_folder>/diagnostics.csv)
  if [[ -f "${DATA_DIR}/diagnostics.csv" ]]; then
    cp "${DATA_DIR}/diagnostics.csv" "${CASE_DIR}/diagnostics.csv"
    mkdir -p "${DIAG_COLLECT_DIR}/${scheme}"
    cp "${DATA_DIR}/diagnostics.csv" "${DIAG_COLLECT_DIR}/${scheme}/diagnostics_${scheme}_${GRID_TAG}_${THREAD_TAG}_${STAMP}.csv"
    echo "[diag] -> ${CASE_DIR}/diagnostics.csv"
  else
    echo "[diag] (note) diagnostics.csv not found for scheme=${scheme} (maybe disabled?)"
  fi

  # MP4 root for this scheme: MP4_Runs/laser2d_<scheme>/<field>/
  SCHEME_MP4_ROOT="${MP4_BASE}/laser2d_${scheme}"
  mkdir -p "$SCHEME_MP4_ROOT"

  for field in "${FIELDS[@]}"; do
    FIELD_DIR="${SCHEME_MP4_ROOT}/${field}"
    mkdir -p "$FIELD_DIR"

    MP4_PATH="${FIELD_DIR}/laser2d_${scheme}_${field}_${GRID_TAG}_${THREAD_TAG}_${STAMP}.mp4"

    echo "[MP4] scheme=${scheme} field=${field}"
    set +e
    python3 "$ANIM_PY" \
      --data_dir "$DATA_DIR" \
      --field "$field" \
      --gamma "$GAMMA" \
      --stride "$STRIDE" \
      --fps "$FPS" \
      --dpi "$DPI" \
      --out_mp4 "$MP4_PATH" \
      $( [[ "$AUTO_LIMITS" == "1" ]] && echo "--auto_limits" ) \
      --limit_every "$LIMIT_EVERY" \
      $( [[ "$USE_PERCENTILES" == "1" ]] && echo "--use_percentiles" ) \
      --plo "$PLO" \
      --phi "$PHI" \
      --smooth_alpha "$SMOOTH_ALPHA"
    status=$?
    set -e

    if [[ $status -ne 0 ]]; then
      echo "⚠️  animator failed for field='${field}'. Skipping." >&2
      rm -f "$MP4_PATH" || true
    else
      echo "      -> ${MP4_PATH}"
    fi
  done
done

echo
echo "✅ Done."
echo "📁 Raw run outputs: ${SESSION_DIR}"
echo "🧾 Diag CSVs:       ${DIAG_COLLECT_DIR}/<scheme>/"
echo "🎞️  MP4s:           ${MP4_BASE}/laser2d_<scheme>/<field>/"