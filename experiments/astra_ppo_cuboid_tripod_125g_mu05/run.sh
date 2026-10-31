#!/usr/bin/env bash
set -euo pipefail
mode=ppo
if [[ $# -gt 0 ]]; then
  if [[ $# -ne 2 || "$1" != '--mode' ]]; then
    echo 'Usage: bash run.sh [--mode ppo|no-ppo]' >&2; exit 2
  fi
  mode="$2"
fi
case "$mode" in ppo|no-ppo) ;; *) echo 'mode must be ppo or no-ppo' >&2; exit 2;; esac
cd "$(dirname "$0")/../.."
root="$PWD"
package="$root/experiments/astra_ppo_cuboid_tripod_125g_mu05"
label="astra_${mode//-/_}_cuboid_tripod_$(date +%Y%m%d_%H%M%S)"
run="$root/outputs/online_astra_ppo/$label"
mkdir -p "$run"
export PYOPENGL_PLATFORM=egl
export PYTHONPATH="$root/ur7e_xhand_sim:$root"
.venv/bin/python "$package/check.py" --mode "$mode" | tee "$run/preflight.log"
port=8775
ppo_args=()
if [[ "$mode" == ppo ]]; then ppo_args=(--online-ppo-config "$package/config.json"); fi
.venv/bin/python -u ur7e_xhand_sim/server.py --port "$port" --output "$run/bridge" --object-shape cuboid --cube-mass-g 125 --cube-friction 0.5 --robot-friction 0.5 --table-contact-force-limit-n "${TABLE_CONTACT_FORCE_LIMIT_N:-5}" --matched-online-physics "${ppo_args[@]}" > "$run/server.log" 2>&1 &
server_pid=$!
trap 'kill "$server_pid" 2>/dev/null || true; wait "$server_pid" 2>/dev/null || true' EXIT
.venv/bin/python - "$server_pid" "$run/server.log" "$mode" "$run" <<'PY'
import os,sys,time,urllib.request,json
from pathlib import Path
pid=int(sys.argv[1]);log=sys.argv[2];mode=sys.argv[3]
for _ in range(180):
 try:os.kill(pid,0)
 except ProcessLookupError:raise RuntimeError('Server exited; inspect '+log)
 try:
  req=urllib.request.Request('http://127.0.0.1:8775/describe',b'{}',{'Content-Type':'application/json'})
  with urllib.request.urlopen(req,timeout=10) as f:d=json.load(f)['result']
  assert bool(d.get('online_ppo',{}).get('enabled'))==(mode=='ppo'), 'Wrong controller mode'
  assert abs(d['physics_dt_s']-1/300)<1e-10 and abs(d['control_dt_s']-1/60)<1e-10
  assert d['solver_profile']['noslip_iterations']==5
  Path(sys.argv[4],'mode_readback.json').write_text(json.dumps({'mode':mode,'description':d},indent=2))
  print('BRIDGE READY: http://127.0.0.1:8775/ mode='+mode,flush=True);break
 except (OSError,KeyError):time.sleep(2)
else:raise RuntimeError('Bridge startup timeout; inspect '+log)
PY
printf 'Starting Astra; mode=%s. Logs: %s\n' "$mode" "$run"
.venv/bin/python -u ur7e_xhand_sim/run_s8v4.py --label "$label" --url "http://127.0.0.1:$port" --object-shape cuboid --reference "$package/demo/reference.json" --connection direct --max-turns 12 2>&1 | tee "$run/astra.log"
printf 'Agent results: %s/outputs/s8v4_agent/%s\nSimulation results: %s/outputs/s8v4_genesis/%s\n' "$root" "$label" "$root" "$label"
