#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
root="$PWD"
package="$root/experiments/astra_ppo_cylinder_125g_mu05"
label="astra_ppo_cylinder_$(date +%Y%m%d_%H%M%S)"
run="$root/outputs/online_astra_ppo/$label"
mkdir -p "$run"
export PYOPENGL_PLATFORM=egl
export PYTHONPATH="$root/ur7e_xhand_sim:$root"
.venv/bin/python "$package/check.py" | tee "$run/preflight.log"
port=8775
.venv/bin/python -u ur7e_xhand_sim/server.py --port "$port" --output "$run/bridge" --object-shape cylinder --cube-mass-g 125 --cube-friction 0.5 --robot-friction 0.5 --table-contact-force-limit-n "${TABLE_CONTACT_FORCE_LIMIT_N:-5}" --online-ppo-config "$package/config.json" > "$run/server.log" 2>&1 &
server_pid=$!
trap 'kill "$server_pid" 2>/dev/null || true; wait "$server_pid" 2>/dev/null || true' EXIT
.venv/bin/python - "$server_pid" "$run/server.log" <<'PY'
import os,sys,time,urllib.request,json
pid=int(sys.argv[1]);log=sys.argv[2]
for _ in range(180):
 try:os.kill(pid,0)
 except ProcessLookupError:raise RuntimeError('Server exited; inspect '+log)
 try:
  req=urllib.request.Request('http://127.0.0.1:8775/describe',b'{}',{'Content-Type':'application/json'})
  with urllib.request.urlopen(req,timeout=2) as f:d=json.load(f)['result']
  assert d['online_ppo']['enabled'];print('BRIDGE READY: http://127.0.0.1:8775/',flush=True);break
 except (OSError,KeyError):time.sleep(2)
else:raise RuntimeError('Bridge startup timeout; inspect '+log)
PY
printf 'Starting Astra + frozen seed1 tactile PPO. Logs: %s\n' "$run"
.venv/bin/python -u ur7e_xhand_sim/run_s8v4.py --label "$label" --url "http://127.0.0.1:$port" --object-shape cylinder --reference "$package/demo/reference.json" --connection direct --max-turns 12 2>&1 | tee "$run/astra.log"
printf 'Agent results: %s/outputs/s8v4_agent/%s\nSimulation results: %s/outputs/s8v4_genesis/%s\n' "$root" "$label" "$root" "$label"
