#!/bin/bash
# Re-run the grasp sampler until every object variant meets its quota.
#
# Genesis can diverge into NaN constraint forces on a deeply interpenetrating sample.
# The sampler saves what it has and exits when that happens, so a single run on a set
# with curved shapes often stops early. Each attempt here appends to the same file.
#
#   bash scripts/sample_grasps_until_full.sh <task> <robot> <config> [max_attempts]
set -uo pipefail

TASK=$1; ROBOT=$2; CONFIG=$3; MAX_ATTEMPTS=${4:-12}
PY=${PY:-.venv/bin/python}
OUTPUT=$($PY -c "import yaml,sys;print(yaml.safe_load(open('$CONFIG'))['grasp_generation']['output_file'])")
QUOTA=$($PY -c "import yaml,sys;print(yaml.safe_load(open('$CONFIG'))['grasp_generation']['num_grasps'])")
echo "output: $OUTPUT  quota per variant: $QUOTA"

for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
    echo "=== attempt $attempt/$MAX_ATTEMPTS ==="
    $PY -u scripts/grasps_generator.py --task="$TASK" --robot="$ROBOT" --config="$CONFIG" --append
    status=$?
    # Exit status does not mean the cache is full: the sampler also exits 0 when it stalls
    # with every variant merely above the floor. Ask the file how many each variant has.
    short=$($PY - "$OUTPUT" "$QUOTA" <<'PYEOF'
import collections, sys, torch
path, quota = sys.argv[1], int(sys.argv[2])
try:
    d = torch.load(path, map_location="cpu")
except Exception:
    print(1); raise SystemExit
counts = collections.Counter(d["geom_idx"].tolist()) if "geom_idx" in d else {0: len(d["joint_pos"])}
print(sum(1 for n in counts.values() if n < quota))
PYEOF
)
    echo "=== attempt $attempt done (status $status); variants still short: $short ==="
    if [ "$short" = "0" ]; then
        echo "=== every variant met its quota ==="
        exit 0
    fi
done
echo "=== gave up after $MAX_ATTEMPTS attempts ==="
exit 1
