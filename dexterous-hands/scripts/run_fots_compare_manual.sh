#!/usr/bin/env bash
# Manual (no-sweep) FOTS sim-vs-real marker comparison: render dilate + shear for each marker model
# (ours / FOTS / HydroShear) at HARDCODED params, on the same ContactDepth backdrop.
#
#   ours       = ElastomerTaxel with our compressibility + bonded-layer thickness
#   FOTS       = FOTS's published marker model driven by our ContactDepth (fixed calib lambdas)
#   HydroShear = the HydroFOTS reference model, ported (scripts/hydroshear_ref.py), on our geometry
#
# Only the SHAPE params matter for the marker fit (compressibility, elastomer-thickness, lambda_d,
# lambda_s, hydroshear-lambda-*); dilate_scale/shear_scale are best-fit automatically. Geometry
# (ball_radius, depth, centers) is set by hand below -- markers can't pin it. Edit the knobs and re-run.
set -euo pipefail
cd "$(dirname "$0")/.."                       # repo root (dexterous_hands/)
PY=${PY:-.venv/bin/python}
OUT=${OUT:-outputs/fots_compare}
mkdir -p "$OUT"
MARKERS="$OUT/markers.npz"                     # all 6 runs append here -> one 2x4 grid
# Kept across runs: each (motion,model) caches its params, so unchanged configs are skipped, not
# rerun. Pass CLEAN=1 to wipe the cache and recompute everything.
[ "${CLEAN:-0}" = 1 ] && rm -f "$MARKERS"

# ----- geometry + motion (shared) -------------------------------------------------------------
COMMON="--ball-radius 3.0 --depth 1.5 --depth-query sdf --control-mode position"
DILATE="--frame 0 --motion dilate --center-px 136 124"
SHEAR="--frame 1 --motion shear --shear-start 136 118 0 --shear-stop 133 135 -5"

# ----- per-model shape params -----------------------------------------------------------------
OURS="--compressibility 0.9 --elastomer-thickness 1 --lambda-d 82_00             --lambda-s 50_000 --shear-scale 0.4"
HYDROSHEAR="--hydroshear-mu 0.6          --hydroshear-lambda-d 82_000 --hydroshear-lambda-s 50_000 "
FOTS="--fots-lambda-d 0.0008 --fots-lambda-s 0.001" 

# each run appends its markers to $MARKERS (--save-markers) and writes its own panel (--out)
run () { echo "===== $1 ====="; shift; $PY scripts/fots_marker_compare.py "$@" --save-markers "$MARKERS"; }

# ----- ours -----------------------------------------------------------------------------------
run "ours dilate"        $COMMON $DILATE --marker-model elastomer  $OURS --out "$OUT/cmp_dilate.png"
run "ours shear"         $COMMON $SHEAR  --marker-model elastomer  $OURS --out "$OUT/cmp_shear.png"

# ----- FOTS -----------------------------------------------------------------------------------
run "fots dilate"        $COMMON $DILATE --marker-model fots $FOTS --out "$OUT/fots_dilate.png"
run "fots shear"         $COMMON $SHEAR  --marker-model fots $FOTS --out "$OUT/fots_shear.png"

# ----- HydroShear -----------------------------------------------------------------------------
run "hydroshear dilate"  $COMMON $DILATE --marker-model hydroshear $HYDROSHEAR --out "$OUT/hydroshear_dilate.png"
run "hydroshear shear"   $COMMON $SHEAR  --marker-model hydroshear $HYDROSHEAR --out "$OUT/hydroshear_shear.png"

# ----- 2x4 grid: [Real FOTS HydroShear Ours] x [dilate shear], one shared depth colorbar -------
$PY scripts/fots_plot_grid.py --npz "$MARKERS" --out "$OUT/grid.png" --figsize 14 5

echo "Done. Outputs in $OUT/  (grid.png, cmp_{dilate,shear}.png, fots_{dilate,shear}.png, hydroshear_{dilate,shear}.png)"
