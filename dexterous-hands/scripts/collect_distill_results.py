#!/usr/bin/env python
"""Collect distillation evaluation curves into one JSON record per run.

Reads a tree of run directories -- each holding ``train/run_status.yaml`` and
``train/evaluation/<iteration>-<condition>/metrics.json`` -- and emits the series the
ablation figures plot. Runs are keyed by their teacher checkpoint so a re-pinned sweep
never mixes with the one it replaced.
"""

import argparse
import json
from pathlib import Path

import yaml

# What each task is judged on. These read the per-episode record rather than a summary field:
# the aggregates hide what separates the conditions -- every lift student raises all 17 training
# objects at least momentarily, so `lifted_fraction` is ~1 everywhere.
TASK_METRIC = {
    "in_hand_repose": ("successes", "Goals reached per episode", "mean"),
    "in_fingers_rotate": ("turns", "Total rotation (turns)", "mean"),
    "grasp_lift": ("successes", "Objects carried", "objects"),
}


def score(metrics: dict, field: str, how: str) -> float | None:
    """Reduce one evaluation's per-episode record to the number the figures plot."""
    per_episode = metrics.get("per_episode") or {}
    values = per_episode.get(field)
    if not values:
        return None
    if how == "mean":
        return sum(values) / len(values)
    # "objects": how many distinct object variants the policy ever carried to a settled goal.
    variants = per_episode.get("variant") or []
    if len(variants) != len(values):
        return None
    carried = {v for v, value in zip(variants, values) if value > 0}
    return float(len(carried))


def modality(sensors: str) -> str:
    """``actual-hand/force_torque`` -> ``force_torque``; the baseline stays ``none``."""
    return sensors.split("/")[-1] if sensors and sensors != "none" else "none"


def read_run(train: Path) -> dict | None:
    status = train / "run_status.yaml"
    if not status.is_file():
        return None
    meta = yaml.safe_load(status.read_text()) or {}
    task = meta.get("task")
    if task not in TASK_METRIC:
        return None
    field, _, how = TASK_METRIC[task]
    series: dict[str, list[list[float]]] = {}
    for path in sorted((train / "evaluation").glob("*/metrics.json")):
        label = path.parent.name
        iteration, _, condition = label.partition("-")
        try:
            value = score(json.loads(path.read_text()), field, how)
        except (OSError, ValueError):
            continue
        if isinstance(value, (int, float)):
            series.setdefault(condition, []).append([int(iteration), float(value)])
    if not series:
        return None
    return {
        "run": train.parent.name,
        "task": task,
        "robot": meta.get("robot"),
        "modality": modality(meta.get("sensors", "none")),
        "imperfection": meta.get("tactile_imperfection", "clean"),
        "teacher": meta.get("teacher_checkpoint"),
        "status": meta.get("status"),
        "last_iteration": meta.get("last_iteration"),
        "metric": field,
        "series": {k: sorted(v) for k, v in series.items()},
    }


def generalization_series(root: Path, final_iteration: int) -> dict:
    """Objects-carried curves keyed by ``(modality, imperfection)``.

    The lift students all carry every training object, so that task is judged on the unseen
    size variants instead, evaluated separately per checkpoint. A bare directory is the final
    checkpoint; ``@N`` is iteration N.
    """
    curves: dict[tuple[str, str], list[list[float]]] = {}
    for path in sorted(root.glob("*/metrics.json")):
        name = path.parent.name
        stem, _, iteration = name.partition("@")
        modality, _, level = stem.rpartition("-")
        value = score(json.loads(path.read_text()), "successes", "objects")
        if value is None:
            continue
        key = (modality, level)
        curves.setdefault(key, []).append([int(iteration or final_iteration), value])
    return {key: sorted(points) for key, points in curves.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", type=Path, help="Directory holding the run directories")
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--teacher", action="append", default=[],
                        help="Keep only runs distilled from this teacher checkpoint (repeatable)")
    parser.add_argument("--prefix", default="distill-", help="Run directory prefix to consider")
    parser.add_argument("--lift-generalization", type=Path,
                        help="Directory of grasp_lift evaluations on unseen objects; when given, "
                             "the lift curves are read from there instead of the training objects")
    parser.add_argument("--final-iteration", type=int, default=2500)
    args = parser.parse_args()

    runs = []
    for train in sorted(args.logs.glob(f"{args.prefix}*/train")):
        record = read_run(train)
        if record and (not args.teacher or record["teacher"] in args.teacher):
            runs.append(record)
    if args.lift_generalization:
        curves = generalization_series(args.lift_generalization, args.final_iteration)
        kept = []
        for record in runs:
            if record["task"] != "grasp_lift":
                kept.append(record)
                continue
            series = curves.get((record["modality"], record["imperfection"]))
            if series:
                kept.append({**record, "series": {"clean": series}, "metric": "objects_carried"})
        missing = len([r for r in runs if r["task"] == "grasp_lift"]) - \
            len([r for r in kept if r["task"] == "grasp_lift"])
        if missing:
            print(f"  dropped {missing} lift runs with no unseen-object evaluation")
        runs = kept

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"runs": runs}, indent=2) + "\n")
    pairs = {(r["task"], r["robot"]) for r in runs}
    print(f"{len(runs)} runs across {len(pairs)} hand/task pairs -> {args.output}")
    for task, robot in sorted(pairs):
        subset = [r for r in runs if (r["task"], r["robot"]) == (task, robot)]
        done = sum(r["last_iteration"] == 2999 for r in subset)
        print(f"  {task}-{robot}: {len(subset)} runs, {done} complete")


if __name__ == "__main__":
    main()
