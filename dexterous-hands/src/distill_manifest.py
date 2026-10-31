"""Run-status tracking and completion manifests for distillation sweeps.

Each training run writes a ``run_status.yaml`` into its (fixed) log directory;
``scripts/distill_status.py`` reads these records without inferring process liveness.
"""

from __future__ import annotations

import datetime
import os
import sys
import tempfile
from pathlib import Path

import yaml

RUN_STATUS_FILENAME = "run_status.yaml"
WANDB_ID_FILENAME = "wandb_run_id.txt"


def now_iso() -> str:
    """Local timestamp, second precision, e.g. ``2026-05-22T11:30:00``."""
    return datetime.datetime.now().isoformat(timespec="seconds")


def run_status_path(log_dir: str | Path) -> Path:
    return Path(log_dir) / RUN_STATUS_FILENAME


def read_run_status(log_dir: str | Path) -> dict:
    """Return the parsed run_status.yaml for ``log_dir`` ({} if absent or unreadable).

    Shared ``logs/`` directories contain runs owned by other users; a
    PermissionError on one of those files must not crash a sweep-wide status
    scan (it would otherwise leak a Python traceback into bash's stdout
    capture and that text ends up sbatched as garbage 'sensor' arguments).
    """
    path = run_status_path(log_dir)
    if not path.is_file():
        return {}
    try:
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except (OSError, yaml.YAMLError) as exc:
        print(f"WARN: could not read {path}: {exc}", file=sys.stderr)
        return {}


def write_run_status(log_dir: str | Path, **fields) -> dict:
    """Merge ``fields`` into the log dir's run_status.yaml (atomic write)."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    status = read_run_status(log_dir)
    status.update(fields)
    path = run_status_path(log_dir)
    fd, tmp = tempfile.mkstemp(dir=log_dir, prefix=".run_status_", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            yaml.safe_dump(status, f, sort_keys=False)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return status


def setup_wandb_resume(log_dir: str | Path, resuming: bool) -> str | None:
    """Persist a stable W&B run id in the log dir and set env vars so a resumed
    run continues the same W&B run.

    Returns the run id, or ``None`` if wandb is not installed.
    """
    try:
        import wandb
    except ModuleNotFoundError:
        return None
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    id_file = log_dir / WANDB_ID_FILENAME
    if id_file.is_file():
        run_id = id_file.read_text().strip()
    else:
        run_id = wandb.util.generate_id()
        id_file.write_text(run_id)
    # wandb.init() honors these env vars; rsl_rl's WandbSummaryWriter does not
    # pass id/resume itself, so this is how we pin the run.
    os.environ["WANDB_RUN_ID"] = run_id
    if resuming:
        os.environ["WANDB_RESUME"] = "allow"
        # rsl_rl calls wandb.config.update({"env_cfg": ...}) without
        # allow_val_change=True; on resume the existing wandb config has the
        # original env_cfg. The run contract already checks model/sensor compatibility;
        # logging metadata and iteration budgets may still differ. Patch the
        # method so resumed runs don't crash at the first .update().
        _patch_wandb_config_allow_val_change()
    return run_id


def _patch_wandb_config_allow_val_change() -> None:
    """Default ``wandb.Config.update(..., allow_val_change=True)`` for this process.

    Idempotent and a no-op if wandb's internals shift; we don't want this to
    become a new failure mode.
    """
    try:
        from wandb.sdk.wandb_config import Config
    except Exception:
        return
    if getattr(Config.update, "_dh_patched", False):
        return
    _orig_update = Config.update

    def update(self, d=None, allow_val_change=None):
        if allow_val_change is None:
            allow_val_change = True
        return _orig_update(self, d, allow_val_change=allow_val_change)

    update._dh_patched = True  # type: ignore[attr-defined]
    Config.update = update  # type: ignore[assignment]
