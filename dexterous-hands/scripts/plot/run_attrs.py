"""Parse task / robot / sensor attributes out of a W&B run.

Resolution is *config-first, name-fallback*: we prefer fields logged into
``run.config`` by Eden, and fall back to regex-parsing the auto-generated run
display name when a field is missing.

The auto-generated run name (see ``src/registry.py``'s ``build_run_name``) is::

    {--run_name}-{task}-{robot}-{res}-{placement}-{type}-{config}

with ``/`` replaced by ``-``. A trailing ``-noisy`` token (after the sensor
type) marks a noisy run; absence of any noise marker means ``clean``.

For distillation students submitted by ``scripts/expand_distill_config.py``, the
``--run_name`` prefix is ``{STUDENT}-{tactile_encoder}`` (e.g.
``STUDENT-tactile_convrnn``) — the encoder is the only thing the caller adds
because ``build_run_name`` already appends task/robot/sensors/config. The
(res, placement, type) triple therefore appears once, inside the auto-suffix,
and the encoder appears once, in the prefix.
"""

from __future__ import annotations

from dataclasses import dataclass
import re

RESOLUTIONS = ("actual", "low", "med", "high")
PLACEMENTS = ("tips", "palm", "hand", "midfinger", "fingers")
ROBOTS = ("wuji", "sharpa", "xhand1", "allegro", "shadow")
TACTILE_ENCODERS = ("tactile_cnn", "tactile_convrnn", "rnn", "mlp")
# Sensor-config values that mean "no tactile input" (used both for teachers
# trained without sensors and for the `none` distillation student baseline).
NO_TACTILE = ("", "none", "no_tactile")
# rsl_rl runner classes — teachers are PPO (`OnPolicyRunner`), students are
# `DistillationRunner`. Both can have `no_tactile` in their run name, so this
# is the reliable disambiguator.
TEACHER_RUNNER = "OnPolicyRunner"
STUDENT_RUNNER = "DistillationRunner"


@dataclass(frozen=True)
class RunAttrs:
    run_id: str
    name: str
    task: str | None
    robot: str | None
    sensor_type: str | None
    placement: str | None
    resolution: str | None
    noise: str  # "clean" | "noisy"
    is_teacher: bool
    # Tactile encoder (e.g. tactile_convrnn / tactile_cnn / rnn / mlp). ``None``
    # for teachers or runs predating the encoder-in-prefix convention. Included
    # in `group_key` so the `models` sweep — same sensor, different encoder —
    # plots as distinct lines.
    tactile_encoder: str | None

    def group_key(self, by: str) -> tuple:
        """Identity tuple ignoring seed - runs sharing it are averaged together."""
        base = (self.task, self.robot)
        if self.is_teacher:
            return base + ("teacher",)
        if by == "sensor_type":
            return base + (self.sensor_type, self.tactile_encoder)
        return base + (
            self.placement,
            self.resolution,
            self.noise,
            self.sensor_type,
            self.tactile_encoder,
        )

    def line_label(self, by: str) -> str:
        if by == "sensor_type":
            return self.sensor_type or "?"
        return f"{self.placement}/{self.resolution}/{self.noise}"


_MODES = ("train", "play", "inference", "deploy", "optimize")


def _full_name(record: dict) -> str:
    """Pre-cap-removal runs have display names truncated to 64 chars (the cap was
    removed from registry.py, but historical runs are unchanged). ``config['log_dir']``
    keeps the full run name (``.../logs/{run_name}/{mode}/{timestamp}``), so prefer
    it when present to recover the trailing sensor token on those legacy runs.
    """
    log_dir = (record.get("config", {}) or {}).get("log_dir")
    if log_dir and "/logs/" in log_dir:
        tail = log_dir.split("/logs/", 1)[1]
        for mode in _MODES:
            if f"/{mode}/" in tail:
                return tail.split(f"/{mode}/", 1)[0]
        return tail
    return record.get("name", "")


def _find_in_config(config: dict, *keys: str):
    """Recursively search a (possibly nested) config dict for any of `keys`."""
    for k, v in config.items():
        if k in keys and not isinstance(v, (dict, list)):
            return v
    for v in config.values():
        if isinstance(v, dict):
            found = _find_in_config(v, *keys)
            if found is not None:
                return found
    return None


def _parse_sensor_spec(spec: str) -> tuple[str | None, str | None, str | None, bool]:
    """`low-tips/force[/noisy]` -> (res, placement, type, noisy)."""
    spec = spec.replace("/", "-")
    toks = spec.split("-")
    for i in range(len(toks) - 2):
        if toks[i] in RESOLUTIONS and toks[i + 1] in PLACEMENTS:
            noisy = i + 3 < len(toks) and toks[i + 3] == "noisy"
            return toks[i], toks[i + 1], toks[i + 2], noisy
    return None, None, None, False


def _parse_name(name: str) -> tuple[str | None, str | None, str | None, bool]:
    """Fallback: pull the (res, placement, type) triple + noise out of a run name.

    Scans left-to-right and keeps the *last* match, since the structured suffix
    is appended after any freeform ``--run_name`` prefix.
    """
    toks = name.split("-")
    res = placement = stype = None
    noisy = False
    for i in range(len(toks) - 2):
        if toks[i] in RESOLUTIONS and toks[i + 1] in PLACEMENTS:
            res, placement, stype = toks[i], toks[i + 1], toks[i + 2]
            noisy = i + 3 < len(toks) and toks[i + 3] == "noisy"
    return res, placement, stype, noisy


def parse_run(record: dict) -> RunAttrs:
    """`record` is a plain dict with keys: id, name, config."""
    config = record.get("config", {}) or {}
    # Full, untruncated run name (display name is capped at 64 chars).
    # Normalise the sensors `/` separator to `-` so token scans are uniform.
    name = _full_name(record).replace("/", "-")

    task = _find_in_config(config, "task", "task_name")
    robot = _find_in_config(config, "robot", "robot_name")
    sensors = _find_in_config(config, "sensors", "sensor_path", "sensor")
    runner_class = _find_in_config(config, "class_name", "algorithm_class")
    tactile_encoder = _find_in_config(config, "tactile_encoder")

    # Name fallback for task / robot.
    if not task:
        for tok in name.split("-"):
            if tok in ("in_fingers_rotate", "screwdriver", "in_hand_rotate"):
                task = tok
                break
    if not robot:
        for tok in name.split("-"):
            if tok in ROBOTS:
                robot = tok
                break

    sensor_str = str(sensors).strip().lower() if sensors is not None else ""
    is_no_tactile = sensor_str in NO_TACTILE if sensors is not None else ("no_tactile" in name.split("-"))

    # Teacher = privileged-RL run with no tactile input. A `DistillationRunner`
    # with no sensor is the "none" no-sensor student baseline, not a teacher.
    if runner_class == STUDENT_RUNNER:
        is_teacher = False
    elif runner_class == TEACHER_RUNNER:
        is_teacher = is_no_tactile
    else:
        is_teacher = is_no_tactile

    res = placement = stype = None
    is_noisy = False
    if sensors and not is_no_tactile:
        res, placement, stype, is_noisy = _parse_sensor_spec(str(sensors))
    if res is None and not is_teacher:
        res, placement, stype, is_noisy = _parse_name(name)
    # No-sensor distillation student: surface it as sensor_type="none" so it
    # plots as the no-sensor baseline line.
    if not is_teacher and stype is None and is_no_tactile:
        stype = "none"

    profile = re.search(r"-(clean|mild|strong)-[a-f0-9]{12}$", name)
    noise = _find_in_config(config, "tactile_imperfection") or (
        profile.group(1) if profile else ("noisy" if is_noisy else "clean")
    )

    # Encoder fallback: scan tokens for a known encoder name. Robust to having
    # the encoder appear before *or* after the task token (legacy prefixes had
    # sensor-name tokens in the prefix; new prefixes use the encoder).
    if tactile_encoder is None and not is_teacher:
        for tok in name.split("-"):
            if tok in TACTILE_ENCODERS:
                tactile_encoder = tok
                break

    return RunAttrs(
        run_id=record.get("id", ""),
        name=name,
        task=task,
        robot=robot,
        sensor_type=stype,
        placement=placement,
        resolution=res,
        noise=noise,
        is_teacher=is_teacher,
        tactile_encoder=tactile_encoder,
    )
