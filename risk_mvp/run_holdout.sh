#!/usr/bin/env bash
# Leave-one-object-out closed loop: per held-out object train the risk model (with Astra, VLM dropout 0.3)
# and the without-Astra model on the other objects, then eval_seq on the held-out object (all methods,
# including the handwritten Bayesian baseline). Runs one after another (parallel runs fight for the CPU).
#   bash risk_mvp/run_holdout.sh [tag]        -> outputs/risk_mvp/holdout_<tag>/<object>/...
set -euo pipefail
TAG=${1:-v3}
OUT=outputs/risk_mvp/holdout_$TAG
OBJECTS=(065-a_cups 056_tennis_ball 015_peach 065-d_cups)
mkdir -p "$OUT"
for O in "${OBJECTS[@]}"; do
  D=$OUT/$O
  mkdir -p "$D"
  echo "=== $O: train (with Astra, VLM dropout 0.3)"
  python risk_mvp/train_risk.py --data collect --out "$D/model" --val-objects "$O" --vlm-dropout 0.3 > "$D/train_model.log" 2>&1
  tail -6 "$D/train_model.log"
  echo "=== $O: train (without Astra)"
  python risk_mvp/train_risk.py --data collect --out "$D/novlm" --val-objects "$O" --no-vlm > "$D/train_novlm.log" 2>&1
  tail -6 "$D/train_novlm.log"
  echo "=== $O: closed loop"
  python risk_mvp/eval_seq.py --objects "$O" --instances 24 --model "$D/model/risk_collect.pt" \
      --model-novlm "$D/novlm/risk_collect.pt" --out "$D/eval" --risk-safety 4 --shake-s 1 > "$D/eval.log" 2>&1
  sed -n '/^method/,$p' "$D/eval.log"
done
python - "$OUT" <<'EOF'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
rows = []
for d in sorted(out.iterdir()):
    f = d/'eval'/'summary.json'
    if f.exists():
        S = json.loads(f.read_text())
        for m, v in S['methods'].items():
            rows.append((d.name, m, v['overall']['success'], v['by_grasp']['success'][0], v['by_grasp']['success'][-1],
                         v['overall']['broke'], v['overall']['dropped']))
methods = ['equal', 'tactile', 'prior', 'bayes_novlm', 'bayes', 'model_novlm', 'model', 'oracle']
print('\nheld-out objects: success all grasps / first grasp / last grasp')
print(f"{'object':16s}" + ''.join(f'{m:>17s}' for m in methods))
for o in sorted({r[0] for r in rows}):
    cell = {r[1]: f'{r[2]:.2f}/{r[3]:.2f}/{r[4]:.2f}' for r in rows if r[0] == o}
    print(f'{o:16s}' + ''.join(f"{cell.get(m, '-'):>17s}" for m in methods))
print(f"{'mean':16s}" + ''.join(
    f"{'/'.join(format(sum(r[i] for r in rows if r[1] == m)/max(1, sum(1 for r in rows if r[1] == m)), '.2f') for i in (2, 3, 4)):>17s}"
    for m in methods))
EOF
