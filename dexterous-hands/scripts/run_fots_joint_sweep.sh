#!/usr/bin/env bash
# JOINT (dilate + shear together) parameter search for MODEL in {elastomer,fots,hydroshear}.
# Each parameter set is scored on BOTH the FOTS dilate frame and the shear frame; the script prints
# the BEST set for dilate and the BEST set for shear (separately), and writes those two best cells to
# the grid npz. Contrast with run_fots_compare.sh, which calibrates dilate THEN shear in stages.
#
# MODEL=elastomer|fots|hydroshear  (default elastomer).  METHOD=optuna (default; TPE over MIN,MAX
# bounds) | grid (every listed value).  e.g.  MODEL=fots JOBS=20 N_TRIALS=50 ./scripts/run_fots_joint_sweep.sh
# Per-model axes are chosen automatically; the npz is APPENDED to, so run each model to fill the grid.
set -euo pipefail
cd "$(dirname "$0")/.."                        # repo root (dexterous_hands/)
PY=${PY:-.venv/bin/python}
OUT=${OUT:-outputs/fots_compare}
mkdir -p "$OUT"
MARKERS="$OUT/markers_joint.npz"
[ "${CLEAN:-0}" = 1 ] && rm -f "$MARKERS"
MODEL=${MODEL:-elastomer}
METHOD=${METHOD:-optuna}
N_TRIALS=${N_TRIALS:-80}
JOBS=${JOBS:-4}                                 # max parallel eval subprocesses

# ----- geometry + motion (shared) -------------------------------------------------------------
COMMON="--ball-radius 3.0 --depth 1.5 --depth-query sdf --control-mode position"
DILATE_CENTER="--center-px 136 124"
SHEAR_PTS="--shear-start 136 118 0 --shear-stop 133 135 -5"

# ----- search ranges (comma-separated; optuna uses MIN,MAX as bounds, grid uses every value) ---
THICKNESS=${THICKNESS:-"3,9"}                    # elastomer_thickness (mm)   build
LAMBDA_D=${LAMBDA_D:-"3e5,1e6"}                  # dilation falloff (1/m^2)   build (log)
COMPRESSIBILITY=${COMPRESSIBILITY:-"0.5,0.95"}   # local/global mix           build
LAMBDA_S=${LAMBDA_S:-"3e4,1e6"}                  # shear locality (1/m^2)     read  (log)
SHEAR_SCALE=${SHEAR_SCALE:-"1,4"}                # shear_scale                read

$PY scripts/fots_joint_sweep.py --marker-model "$MODEL" $COMMON $DILATE_CENTER $SHEAR_PTS \
    --method "$METHOD" --n-trials "$N_TRIALS" --jobs "$JOBS" \
    --thickness "$THICKNESS" --lambda-d "$LAMBDA_D" --compressibility "$COMPRESSIBILITY" \
    --lambda-s "$LAMBDA_S" --shear-scale "$SHEAR_SCALE" \
    --save-markers "$MARKERS" --out-dilate "$OUT/joint_${MODEL}_dilate.png" --out-shear "$OUT/joint_${MODEL}_shear.png"

# Grid from all cells saved so far (run each MODEL to populate FOTS/HydroShear/Ours columns).
$PY scripts/fots_plot_grid.py --npz "$MARKERS" --out "$OUT/grid_joint.png" --figsize 14 5 || true
echo "Done ($MODEL) -> $OUT/  (joint_${MODEL}_{dilate,shear}.png, grid_joint.png; best params above)"
