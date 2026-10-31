"""Stable experiment identities and non-destructive resume contracts (no simulator imports)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml


def file_identity(path):
    if not path:
        return None
    path = Path(path).expanduser()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path.resolve()), "sha256": digest.hexdigest()}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def serialize(value):
    """Serialize declared fields, avoiding computed properties that require gs.init()."""
    from enum import Enum
    from pydantic import BaseModel

    if isinstance(value, BaseModel):
        fields = set(type(value).model_fields) | set(value.model_extra or {})
        result = {key: serialize(getattr(value, key)) for key in sorted(fields)}
        result["_option_module_"] = getattr(value, "_option_module_", type(value).__module__)
        result["_option_class_"] = getattr(value, "_option_class_", type(value).__name__)
        return result
    if isinstance(value, dict):
        return {str(key): serialize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [serialize(item) for item in value]
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if callable(value):
        return f"{value.__module__}.{value.__qualname__}"
    raise TypeError(f"Cannot serialize configuration value {type(value).__name__}")


# These affect duration, reporting, and placement, not the learned policy contract.
REPORTING_RUNNER_KEYS = ("max_iterations", "run_name", "logger", "wandb_project", "device", "save_interval")


def without_reporting_keys(resolved):
    """Strip reporting-only runner fields so they neither break resume nor rename a run."""
    runner = resolved.get("runner_options")
    if not runner:
        return resolved
    return {**resolved, "runner_options": {k: v for k, v in runner.items() if k not in REPORTING_RUNNER_KEYS}}


def run_contract(config, args):
    resolved = serialize(config)
    runner = resolved.get("runner_options") or {}
    for key in REPORTING_RUNNER_KEYS:
        runner.pop(key, None)
    resolved.pop("cameras_options", None)
    assets = {}
    robot = config.scene_options.robot
    for label, path in {"robot": robot.file}.items():
        assets[label] = file_identity(path)
    sensors = getattr(args, "sensors", "none")
    if sensors not in ("", "none"):
        placement = sensors.split("/")[0]
        if placement not in ("links", "fingertip_links"):
            layout = placement.split("-")[0]
            assets["layout"] = file_identity(robot.metadata.tactile_probe_cfgs[layout])
    return {
        "version": 1,
        "task": args.task,
        "robot": getattr(args, "robot", None),
        "sensors": sensors,
        "config": resolved,
        "assets": assets,
        "teacher": file_identity(args.checkpoint),
        "actor_checkpoint": file_identity(args.actor_checkpoint),
        "critic_checkpoint": file_identity(args.critic_checkpoint),
    }


def cli_run_identity(args):
    """Separate direct CLI runs by model, stage, seed, profile and input file contents."""
    fields = (
        "stage",
        "algo",
        "model",
        "actor",
        "critic",
        "encoder",
        "tactile_encoder",
        "temporal_reduction",
        "seed",
        "tactile_imperfection",
    )
    inputs = {key: getattr(args, key, None) for key in fields}
    for key in ("checkpoint", "actor_checkpoint", "critic_checkpoint", "config", "full_config"):
        value = getattr(args, key, None)
        inputs[key] = file_identity(value) if value and value != "latest" else value
    return digest(inputs)[:12]


def validate_run_contract(config, args, log_dir):
    """Check before overwriting configs or contacting W&B; never remove an old run."""
    contract = run_contract(config, args)
    folder = Path(log_dir)
    path = folder / "run_contract.yaml"
    if path.is_file():
        previous = yaml.safe_load(path.read_text())
        if previous != contract:
            raise ValueError(f"Run configuration changed: {path}. Use a new run name; existing data is preserved.")
        if not args.resume:
            raise ValueError(f"Run already exists: {folder}. Use --resume or a new run name.")
    elif any(folder.glob("**/model_*.pt")):
        raise ValueError(f"Cannot verify legacy checkpoints in {folder}: no run_contract.yaml. Use a new run name.")
    else:
        folder.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(yaml.safe_dump(contract, sort_keys=False))
        temporary.replace(path)
    return contract
