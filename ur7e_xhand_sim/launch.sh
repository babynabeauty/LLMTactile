#!/usr/bin/env bash
set -euo pipefail
sim_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$sim_root"
export PYOPENGL_PLATFORM=egl
export OMP_NUM_THREADS=8
exec "$sim_root/.venv/bin/python" -u ur7e_xhand_sim/server.py "$@"
