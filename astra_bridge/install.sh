#!/usr/bin/env bash
set -euo pipefail
bridge_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$bridge_root"
expected_commit=4777fd5da08244acbc677dca6f6f9acfb2d69d2b
if [[ "$(git rev-parse HEAD)" != "$expected_commit" ]]; then
  echo "Expected upstream commit $expected_commit; checkout it before installing." >&2
  exit 1
fi
if [[ ! -x .venv/bin/python ]]; then
  uv venv .venv --python 3.12
fi
uv pip install --python .venv/bin/python 'torch==2.10.0' 'torchvision==0.25.0' --index-url https://download.pytorch.org/whl/cu128
if [[ -f astra_bridge/requirements.lock ]]; then
  uv pip install --python .venv/bin/python -r astra_bridge/requirements.lock -e ./Genesis -e ./Eden -e ./rsl_rl -e ./dexterous-hands --index-url https://pypi.tuna.tsinghua.edu.cn/simple
else
  uv pip install --python .venv/bin/python -e ./Genesis -e ./Eden -e ./rsl_rl -e ./dexterous-hands --index-url https://pypi.tuna.tsinghua.edu.cn/simple
fi
.venv/bin/python -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name())'
