#!/usr/bin/env python
"""Evaluate every grasp-and-lift student on objects it was not trained on.

The lift students train on one pinned set of 17 box and cylinder primitives at roughly 10 cm,
and every condition lifts all 17 at least momentarily, so that set cannot separate them. This
re-evaluates each student on nearby unseen sizes and at other densities, and counts how many
objects it can actually carry.

``variants_50`` is the discriminating set: the same shape family rescaled by 0.80--0.96 and
1.04--1.20, so every object is close to what the student trained on but none is one of them.
Real scanned meshes were tried first and turned out to be a floor rather than a harder test --
a student that carries all 17 training primitives carries none of the 21 YCB objects.

The lift task config pins ``scene_options.obj``, so ``--obj`` on the command line is overridden
by it; each condition therefore gets a generated config with the object list written in.

Run from the cluster checkout. Without ``--submit`` this prints what it would do.
"""

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

BASE = "conf/experiments/grasp_lift_allegro_fresh_v141_simple_contact_fast_v7_distill.yaml"
TEACHER = "teachers/f141/grasp_lift-allegro/model_4999.pt"
# The per-link representations are excluded from the study's figures, so they are not evaluated.
SKIP_MODALITIES = {"link_bool", "link_force"}


# How far the unseen variants stray from the training size, and the dead band that keeps any
# of them from simply being a training object again.
VARIANT_RANGE = (0.80, 1.20)
VARIANT_GAP = 0.04


def rescaled(prototype, factor: float):
    """A copy of one primitive at ``factor`` of its size."""
    import copy as _copy

    scaled = _copy.deepcopy(prototype)
    if hasattr(scaled, "size"):
        scaled.size = tuple(factor * v for v in scaled.size)
    else:
        scaled.radius = factor * scaled.radius
        scaled.height = factor * scaled.height
    return scaled


def size_variants(count: int):
    """``count`` unseen objects drawn from the training shapes at nearby scales."""
    from entities.objects import OBJECTS_10CM_16

    low, high = VARIANT_RANGE
    half = count // 2
    steps = ([low + (1 - VARIANT_GAP - low) * i / max(half - 1, 1) for i in range(half)]
             + [1 + VARIANT_GAP + (high - 1 - VARIANT_GAP) * i / max(count - half - 1, 1)
                for i in range(count - half)])
    def shape_key(options):
        if hasattr(options, "size"):
            return ("box", *(round(v, 6) for v in options.size))
        return ("cylinder", round(options.radius, 6), round(options.height, 6))

    # The training sizes repeat, so scaling one shape can land exactly on another and
    # reintroduce a seen object; nudge the scale until it does not.
    trained = {shape_key(o) for o in OBJECTS_10CM_16}
    variants = []
    for index, factor in enumerate(steps):
        prototype = OBJECTS_10CM_16[index % len(OBJECTS_10CM_16)]
        candidate = rescaled(prototype, factor)
        while shape_key(candidate) in trained:
            factor += 0.005
            candidate = rescaled(prototype, factor)
        variants.append(candidate)
    return variants


def named_objects(name: str):
    """The entity option objects for a named set, as the task modifier would build them."""
    from entities.objects import (OBJECT_SET_MANIFESTS, OBJECTS_4CM_16, OBJECTS_8CM_16,
                                  OBJECTS_10CM_16, load_manifest_object_set)

    if name.startswith("variants_"):
        return size_variants(int(name.split("_", 1)[1]))
    builtin = {"primitives": OBJECTS_4CM_16, "primitives_large": OBJECTS_8CM_16,
               "primitives_xl": OBJECTS_10CM_16}
    if name in builtin:
        return builtin[name]
    if name in OBJECT_SET_MANIFESTS:
        return load_manifest_object_set(name)
    raise SystemExit(f"Unknown object set {name!r}")


def serialized(options, rho: float | None, ghost: bool = False):
    """One entity as the config stores it, optionally as the pinned non-colliding goal marker."""
    import copy as _copy

    from experiment_runs import serialize
    from task_mods import _make_options_ghost

    if ghost:
        options = _copy.deepcopy(options)
        # `fixed` exists on the primitive options but not on the mesh ones, so the repo's own
        # helper decides which fields this class actually has.
        _make_options_ghost(options)
    entry = serialize(options)
    if rho is not None:
        material = dict(entry.get("material") or {})
        material.update({"rho": rho, "_option_module_": "eden.options.materials",
                         "_option_class_": "RigidMaterialOptions"})
        material.setdefault("friction", 1.0)
        entry["material"] = material
    return entry


def write_config(objects: str, rho: float | None, out: Path) -> Path:
    base = yaml.safe_load((ROOT / BASE).read_text())
    source = named_objects(objects)
    base.setdefault("scene_options", {})
    base["scene_options"]["obj"] = {"grouped_entities": [serialized(o, rho) for o in source]}
    base["scene_options"]["vis_obj"] = {
        "grouped_entities": [serialized(o, rho, ghost=True) for o in source]
    }
    header = (f"# Generated by scripts/submit_lift_generalization.py: {objects}"
              f"{'' if rho is None else f' at rho {rho:g}'}. Do not edit by hand.\n")
    out.write_text(header + yaml.safe_dump(base, sort_keys=False))
    return out


def students(logs: Path, final_iteration: int):
    found = []
    for status in sorted(logs.glob("distill-grasp_lift-allegro-*/train/run_status.yaml")):
        meta = yaml.safe_load(status.read_text()) or {}
        if meta.get("teacher_checkpoint") != TEACHER:
            continue
        checkpoint = status.parent / f"model_{final_iteration}.pt"
        if not checkpoint.is_file():
            continue
        modality = (meta.get("sensors") or "none").split("/")[-1]
        if modality in SKIP_MODALITIES:
            continue
        found.append((meta, checkpoint))
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs", type=Path, default=ROOT / "logs")
    parser.add_argument("--final-iteration", type=int, default=2500)
    parser.add_argument("--iteration", action="append", type=int, default=[],
                        help="Also evaluate these earlier checkpoints, so the panel is a curve")
    parser.add_argument("--objects", action="append", default=[],
                        help="Object set to evaluate on (repeatable)")
    parser.add_argument("--rho", action="append", type=float, default=[],
                        help="Extra density to evaluate at (repeatable); default density is 500")
    parser.add_argument("--rho-objects", default="variants_50",
                        help="Object set the density sweep runs on")
    parser.add_argument("--episodes-per-object", type=int, default=16,
                        help="Evaluation episodes per object; the env count is a multiple of the set size")
    parser.add_argument("--partition", default="hpc-mid")
    parser.add_argument("--time", default="02:00:00")
    parser.add_argument("--limit", type=int, default=0, help="Submit at most this many jobs")
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()

    objects = args.objects or ["primitives_xl", "variants_50"]
    conditions = [(name, None) for name in objects]
    conditions += [(args.rho_objects, rho) for rho in args.rho]

    config_dir = ROOT / "conf/experiments/generated"
    config_dir.mkdir(parents=True, exist_ok=True)
    out_root = ROOT / "logs/lift-generalization"

    jobs = []
    for name, rho in conditions:
        tag = name if rho is None else f"{name}-rho{rho:g}"
        config = write_config(name, rho, config_dir / f"grasp_lift_eval_{tag}.yaml")
        # The evaluator insists on balanced coverage, and the sets differ in size: 17
        # primitives against 21 held-out meshes.
        eval_envs = len(named_objects(name)) * args.episodes_per_object
        for iteration in [args.final_iteration, *args.iteration]:
          for meta, checkpoint in students(args.logs, iteration):
            modality = (meta.get("sensors") or "none").split("/")[-1]
            level = meta.get("tactile_imperfection", "clean")
            # Each checkpoint writes its own directory, so the collector can read a curve.
            stem = f"{modality}-{level}" if iteration == args.final_iteration \
                else f"{modality}-{level}@{iteration}"
            output = out_root / tag / stem
            jobs.append((tag, modality, level, [
                "--task=grasp_lift", "--robot=allegro", "--stage=2",
                "--sensors=" + (meta.get("sensors") or "none"),
                "--tactile_imperfection=" + level,
                "--seed=42", "--model=tac_mlp", "--encoder=rnn",
                "--tactile_encoder=" + (meta.get("tactile_encoder") or "rnn"),
                "--temporal_reduction=none",
                "--config=" + str(config.relative_to(ROOT)),
                "--checkpoint=" + str(checkpoint.relative_to(ROOT)),
                "--eval_envs=" + str(eval_envs),
                "--eval_seed=42", "--no_traces",
                "--output=" + str(output.relative_to(ROOT)),
            ]))
    if args.limit:
        jobs = jobs[:args.limit]

    for tag, modality, level, argv in jobs:
        print(f"[{tag}] {modality}/{level}")
        print("  " + shlex.join([sys.executable, "scripts/eval_smooth_teacher.py", *argv]))
    print(f"{len(jobs)} evaluations across {len(conditions)} conditions")
    if not args.submit:
        return 0
    (ROOT / "logs/slurm").mkdir(parents=True, exist_ok=True)
    for tag, modality, level, argv in jobs:
        subprocess.run([
            "sbatch", f"--job-name=liftgen-{tag}-{modality}-{level}",
            "--partition=" + args.partition, "--time=" + args.time,
            "scripts/slurm/run.sh", "scripts/eval_smooth_teacher.py", *argv,
        ], cwd=ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
