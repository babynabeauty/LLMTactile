#!/usr/bin/env bash
set -euo pipefail
bridge_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$bridge_root/dexterous-hands"
export PYOPENGL_PLATFORM=egl
export OMP_NUM_THREADS=8
exec "$bridge_root/.venv/bin/python" -u "$bridge_root/astra_bridge/server.py" "$@"
