"""Cluster-independent tactile experiment expansion and prerequisite checks."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

from checkpoint_validation import read_teacher_checkpoint
from model_config import TACTILE_ENCODER_CFGS

import yaml

from experiment_runs import digest, file_identity, serialize, without_reporting_keys

ROOT = Path(__file__).resolve().parents[1]
HANDS = ("wuji", "allegro", "sharpa")
TASKS = ("in_hand_repose", "in_fingers_rotate", "grasp_lift")
LAYOUTS = ("actual", "low", "high")
IMPERFECTIONS = ("clean", "mild", "strong")


def resolve_tactile_encoder(sensor: str, choice: str = "auto") -> str:
    """``auto`` reads the grid off the modality; a named encoder is used as given."""
    if choice != "auto":
        return choice
    if sensor == "none" or sensor.split("/")[1] in ("agg_bool", "agg_force", "link_bool", "link_force"):
        return "rnn"
    return "tactile_convrnn"


_GENESIS_READY = False


def init_genesis_for_overrides():
    """Eden's override merge serializes computed options that read ``gs.EPS``.

    A CPU init sets those globals without building a scene or touching a GPU.
    """
    global _GENESIS_READY
    if _GENESIS_READY:
        return
    import genesis as gs

    if getattr(gs, "EPS", None) is None:
        gs.init(backend=gs.cpu, logging_level="warning")
    _GENESIS_READY = True


def expand(path, names=None):
    doc = yaml.safe_load(Path(path).read_text())
    entries = doc["experiments"]
    names = list(entries) if not names else names
    jobs = []
    for name in names:
        if name not in entries:
            raise ValueError(f"Unknown experiment {name!r}; available: {', '.join(entries)}")
        cfg = {**doc.get("defaults", {}), **entries[name]}
        if cfg["hand"] not in HANDS or cfg["task"] not in TASKS:
            raise ValueError(f"Unsupported target: {cfg['hand']}/{cfg['task']}")
        for key, allowed in (("layouts", LAYOUTS), ("imperfections", IMPERFECTIONS)):
            if not cfg[key] or not set(cfg[key]) <= set(allowed):
                raise ValueError(f"{name}: {key} must select from {allowed}")
        if not cfg["seeds"] or any(type(s) is not int or s < 0 for s in cfg["seeds"]):
            raise ValueError(f"{name}: seeds must be nonnegative integers")
        if cfg.get("max_iters") is not None and (type(cfg["max_iters"]) is not int or cfg["max_iters"] <= 0):
            raise ValueError(f"{name}: max_iters must be a positive integer or null")
        if not cfg["sensor_types"]:
            raise ValueError(f"{name}: select at least one sensor type")
        if type(cfg["baseline"]) is not bool:
            raise ValueError(f"{name}: baseline must be true or false")
        if cfg["tactile_encoder"] != "auto" and cfg["tactile_encoder"] not in TACTILE_ENCODER_CFGS:
            raise ValueError(f"{name}: tactile_encoder must be 'auto' or one of {tuple(TACTILE_ENCODER_CFGS)}")
        for seed in dict.fromkeys(cfg["seeds"]):
            # A companion entry sharing a hand/task/seed must not re-emit the baseline:
            # it would resolve to the same run name and two jobs would share a log dir.
            variants = [("none", "clean")] if cfg["baseline"] else []
            for layout, modality, level in itertools.product(cfg["layouts"], cfg["sensor_types"], cfg["imperfections"]):
                placement = "links" if modality.startswith("link_") else f"{layout}-hand"
                variants.append((f"{placement}/{modality}", level))
            for sensors, level in dict.fromkeys(variants):
                job = dict(
                    experiment=name,
                    task=cfg["task"],
                    hand=cfg["hand"],
                    seed=seed,
                    sensors=sensors,
                    imperfection=level,
                    teacher_checkpoint=cfg.get("teacher_checkpoint"),
                    teacher_config=cfg.get("teacher_config"),
                    task_config=cfg.get("task_config"),
                    wandb_project=cfg.get("wandb_project") or f"tgs-distill-{name}",
                    model=cfg["model"],
                    encoder=cfg["encoder"],
                    temporal_reduction=cfg["temporal_reduction"],
                    tactile_encoder=resolve_tactile_encoder(sensors, cfg["tactile_encoder"]),
                    max_iters=cfg.get("max_iters"),
                )
                jobs.append(job)
    return jobs


def comparable(section):
    """Drop what differs between a saved post-build config and a freshly built one.

    ``surface`` is render-only and Genesis resolves its defaults during build, so a
    teacher's saved config reports ``smooth=False`` where a fresh one still holds
    ``None``. Asset paths are checkout-specific -- the teacher's name its training
    snapshot -- and content is fingerprinted separately, so only the name is compared.
    """
    if isinstance(section, dict):
        return {
            key: Path(value).name if key == "file" and isinstance(value, str) else comparable(value)
            for key, value in section.items()
            if key != "surface"
        }
    if isinstance(section, list):
        return [comparable(item) for item in section]
    return section


def modifiers(job, stage=2):
    return dict(
        robot=job["hand"],
        obj="primitives",
        stage=stage,
        sensors=job["sensors"],
        tactile_imperfection=job["imperfection"],
        model=job["model"],
        encoder=job["encoder"],
        tactile_encoder=job["tactile_encoder"],
        temporal_reduction=job["temporal_reduction"],
        max_iters=job["max_iters"] or -1,
        wandb_project=job["wandb_project"],
    )


def preflight(job, cache=None):
    """Build configs without building a scene, creating run folders, or logging in."""
    from registry import get_task_config
    from tactile_sensors import imperfection_params

    cache = {} if cache is None else cache
    errors = []
    fingerprints = {}
    for key in ("teacher_checkpoint", "teacher_config", "task_config"):
        path = job[key]
        if not path:
            if key != "task_config":
                errors.append(f"{key} is not set")
        elif not Path(path).is_file():
            errors.append(f"{key} does not exist: {path}")
        else:
            cache_key = ("file", str(Path(path).resolve()))
            if cache_key not in cache:
                cache[cache_key] = file_identity(path)
            fingerprints[key] = cache[cache_key]
    if "teacher_checkpoint" in fingerprints:
        try:
            key = ("checkpoint", job["teacher_checkpoint"])
            if key not in cache:
                read_teacher_checkpoint(job["teacher_checkpoint"])
                cache[key] = True
        except (ValueError, OSError) as exc:
            errors.append(str(exc))
    cfg = None
    try:
        if job["task_config"]:
            init_genesis_for_overrides()
        cfg = get_task_config(
            task_name=job["task"], modifiers=modifiers(job), config_override_path=job["task_config"] or ""
        )
        cfg.runner_options.seed = job["seed"]
        if job["max_iters"] is not None:
            cfg.runner_options.max_iterations = job["max_iters"]
        if "distill" not in cfg.runner_options.class_name.lower():
            raise ValueError("task_config overrides stage 2 with a non-distillation runner")
        groups = cfg.runner_options.obs_groups
        if any(name.startswith("priv_") for name in groups["student"]):
            raise ValueError("Student observations contain privileged groups")
        for group in groups["student"] + groups["teacher"]:
            if not hasattr(cfg.observation_options, group):
                raise ValueError(f"Observation set names {group!r}, which this configuration does not define")
        robot = cfg.scene_options.robot
        fingerprints["robot"] = file_identity(robot.file)
        if job["sensors"] != "none":
            placement, kind = job["sensors"].split("/")
            fingerprints["imperfection_params"] = imperfection_params(kind, job["imperfection"])
            if placement != "links":
                path = robot.metadata.tactile_probe_cfgs[placement.split("-")[0]]
                fingerprints["layout"] = file_identity(path)
                records = json.loads(Path(path).read_text())
                if job["tactile_encoder"].startswith("tactile_") and any(r["kind"] != "grid" for r in records):
                    raise ValueError(f"{job['tactile_encoder']} requires grids on every sensor patch")
        # A run is identified by its learned-policy contract, not by where it reports.
        fingerprints["resolved_config"] = without_reporting_keys(serialize(cfg))
        # Teacher and student must use the same physical task and teacher inputs.
        if "teacher_config" in fingerprints:
            from eden.utils.configs import EdenRLConfig

            key = ("teacher", job["teacher_config"])
            if key not in cache:
                cache[key] = EdenRLConfig.load_from_file(job["teacher_config"])
            teacher = cache[key]
            for section in (
                "scene_options",
                "action_options",
                "command_options",
                "event_options",
                "reward_options",
                "termination_options",
            ):
                a, b = getattr(teacher, section, None), getattr(cfg, section, None)
                if comparable(serialize(a) if a else None) != comparable(serialize(b) if b else None):
                    raise ValueError(f"Teacher/student {section} differ; align task_config with the teacher")
            for key in ("sim_dt", "decimation"):
                if getattr(teacher.env_options, key) != getattr(cfg.env_options, key):
                    raise ValueError(f"Teacher/student {key} differs")
            teacher_groups = teacher.runner_options.obs_groups
            if teacher_groups.get("actor") != groups["teacher"]:
                raise ValueError("Teacher actor observation groups do not match distillation teacher inputs")
            for group in groups["teacher"]:
                if serialize(getattr(teacher.observation_options, group)) != serialize(
                    getattr(cfg.observation_options, group)
                ):
                    raise ValueError(f"Teacher observation definition differs: {group}")
            actor = serialize(teacher.runner_options.actor)
            target = serialize(cfg.runner_options.teacher)
            # PPO has an exploration distribution; the deterministic distillation teacher does not.
            for model in (actor, target):
                model.pop("distribution_cfg", None)
                model.pop("_option_class_", None)
            if actor != target:
                raise ValueError("Teacher network configuration differs from the distillation teacher")
    except (ValueError, KeyError, AttributeError, OSError, TypeError) as exc:
        errors.append(str(exc))
    # Where a run reports is not part of what it trains, so it must not rename the run.
    contract = {key: value for key, value in job.items() if key != "wandb_project"}
    identity = digest({"job": contract, "inputs": fingerprints})[:16]
    return {**job, "run_name": f"distill-{job['experiment']}-{identity}", "inputs": fingerprints, "errors": errors}


def command(job):
    args = [
        "--task=" + job["task"],
        "--robot=" + job["hand"],
        "--stage=2",
        "--sensors=" + job["sensors"],
        "--tactile_imperfection=" + job["imperfection"],
        "--seed=" + str(job["seed"]),
        "--model=" + job["model"],
        "--encoder=" + job["encoder"],
        "--tactile_encoder=" + job["tactile_encoder"],
        "--temporal_reduction=" + job["temporal_reduction"],
        "--wandb_project=" + job["wandb_project"],
        "--run_name=" + job["run_name"],
        "--resume",
        "--no_loaded_video",
        "--no_checkpoint_video",
    ]
    if job["teacher_checkpoint"]:
        args.append("--checkpoint=" + job["teacher_checkpoint"])
    if job["task_config"]:
        args.append("--config=" + job["task_config"])
    if job["max_iters"] is not None:
        args.append("--max_iters=" + str(job["max_iters"]))
    return args
