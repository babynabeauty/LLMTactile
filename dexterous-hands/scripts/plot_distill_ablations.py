#!/usr/bin/env python
"""Draw the distillation ablations from collected results.

Each hand/task pair gets a column and each ablation gets a row, following the learning-curve
layout of the original tactile ablations figure. The ``types`` view compares tactile types
over distillation iterations. The ``levels`` view holds the type at ``force_torque`` and
compares clean, mild, and strong sensor-imperfection profiles over the same iterations.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.transforms import blended_transform_factory  # noqa: E402

AXIS_LABEL = {
    "in_hand_repose": "Goals reached per episode",
    "in_fingers_rotate": "Total rotation (turns)",
    "grasp_lift": "Objects carried",
}
# Task names as the prose writes them, so a reader can match a panel to the text.
PAIR_TITLE = {
    "in_hand_repose-wuji": "in-hand repose / Wuji",
    "in_fingers_rotate-sharpa": "in-fingers rotate / Sharpa",
    "grasp_lift-allegro": "grasp lift / Allegro",
}
# Ordered severity, so one hue light->dark rather than three identities.
# Validated on the light surface with the skill's validate_palette.js --ordinal.
IMPERFECTION_COLOR = {"clean": "#86b6ef", "mild": "#2a78d6", "strong": "#104281"}
BASELINE_COLOR = "#8a8a85"
TEACHER_COLOR = "#0b0b0b"
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e3e6ea"
SURFACE = "#fcfcfb"
# Grouped by sensor family, matching the legend order across all panels.
TYPE_ORDER = ["agg_bool", "bool", "agg_force", "force",
              "force_torque", "depth", "proximity", "elastomer"]
# Trained and collected, but out of the study: a per-link reading is one vector for a whole
# finger segment, which is a different kind of device from the taxel arrays being compared.
EXCLUDED_TYPES = {"link_bool", "link_force"}
# Only where the run key is not already readable as a legend entry.
TYPE_LABEL = {
    "agg_bool": "agg. bool",
    "agg_force": "agg. force",
    "force_torque": "force+torque",
}
TYPE_COLOR = {
    "agg_bool": "#286653",
    "bool": "#262626",
    "agg_force": "#3047c9",
    "force": "#f0a34a",
    "force_torque": "#ce5874",
    "depth": "#6ca4e8",
    "proximity": "#3cbc57",
    "elastomer": "#9a60c4",
}


def checkpoints_through(final_iteration):
    """Every checkpoint a run should have been scored at, ending at ``final_iteration``.

    Checkpoints land every 500 updates, and a budget of N updates writes its last one at N-1,
    so 2999 closes the 3000-update ladder while 2500 is a checkpoint in its own right.
    """
    ladder = list(range(500, final_iteration + 1, 500))
    if final_iteration not in ladder:
        ladder.append(final_iteration)
    return ladder


def studied(run):
    """Whether this run is one of the cells the figures compare."""
    return run.get("modality") not in EXCLUDED_TYPES


def scored_at(run, iteration):
    """Whether this run has a clean evaluation at ``iteration``.

    Membership is read from the evaluations themselves rather than from ``last_iteration``,
    which a run only records once it finishes: a still-training cell has a usable checkpoint
    at 2500 long before it can claim one.
    """
    return any(point[0] == iteration for point in run.get("series", {}).get("clean", []))


def require_complete_matrix(runs, final_iteration):
    """Reject publication input unless every intended ablation cell is final."""
    final_runs = [run for run in runs if studied(run) and scored_at(run, final_iteration)]
    expected = {
        (pair, modality, "clean")
        for pair in PAIR_TITLE
        for modality in ["none", *TYPE_ORDER]
    }
    expected.update(
        (pair, "force_torque", level)
        for pair in PAIR_TITLE
        for level in ("mild", "strong")
    )
    observed = {
        (f'{run["task"]}-{run["robot"]}', run["modality"], run["imperfection"])
        for run in final_runs
    }
    missing = sorted(expected - observed)
    unexpected = sorted(observed - expected)
    non_unique = sorted(
        key for key in expected
        if sum(
            (f'{run["task"]}-{run["robot"]}', run["modality"], run["imperfection"]) == key
            for run in final_runs
        ) > 1
    )
    bad_series = []
    for run in final_runs:
        key = (f'{run["task"]}-{run["robot"]}', run["modality"], run["imperfection"])
        if key not in expected:
            continue
        iterations = [point[0] for point in run.get("series", {}).get("clean", [])
                      if point[0] <= final_iteration]
        if iterations != checkpoints_through(final_iteration):
            bad_series.append((key, iterations))
    if missing or unexpected or non_unique or bad_series:
        details = []
        if missing:
            details.append("missing=" + ", ".join("/".join(cell) for cell in missing))
        if unexpected:
            details.append("unexpected=" + ", ".join("/".join(cell) for cell in unexpected))
        if non_unique:
            details.append("non-unique=" + ", ".join("/".join(cell) for cell in non_unique))
        if bad_series:
            details.append("bad-series=" + ", ".join(
                "/".join(key) + f":{iterations}" for key, iterations in bad_series
            ))
        raise ValueError(
            f"publication matrix is incomplete at iteration {final_iteration}: "
            + "; ".join(details)
        )


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8, length=3)
    ax.set_axisbelow(True)


def mark_teacher(ax, teacher, name=False):
    """The privileged teacher's score, named on the rule itself rather than in the legend.

    Only the leftmost column is named; the rule means the same thing in every panel, so
    repeating the word across a row is clutter.
    """
    if teacher is None:
        return
    ax.axhline(teacher, color=TEACHER_COLOR, linewidth=1.4, linestyle=(0, (2, 2)), zorder=2)
    if name:
        ax.text(0.005, teacher, "teacher",
                transform=blended_transform_factory(ax.transAxes, ax.transData),
                ha="left", va="bottom", fontsize=8, color=TEACHER_COLOR)


def place_legend(ax, teacher, **kwargs):
    """Legend at the top left, tucked just under the teacher rule rather than over it.

    The teacher line sits near the top of every panel, so a plain upper-left legend lands on it
    and on whichever curves reach highest.
    """
    if teacher is None:
        ax.legend(loc="upper left", **kwargs)
        return
    low, high = ax.get_ylim()
    height = (high - low) or 1.0
    ax.legend(loc="upper left", bbox_to_anchor=(0.01, (teacher - low) / height - 0.03), **kwargs)


def draw_types(ax, subset, teacher, task, first):
    for name in TYPE_ORDER:
        run = next((r for r in subset if r["modality"] == name and r["imperfection"] == "clean"), None)
        points = (run or {}).get("series", {}).get("clean") or []
        if points:
            ax.plot([p[0] for p in points], [p[1] for p in points], linewidth=1.8,
                    marker="o", markersize=2.8, color=TYPE_COLOR[name],
                    label=TYPE_LABEL.get(name, name), zorder=3)
    base_run = next((r for r in subset if r["modality"] == "none"), None)
    points = (base_run or {}).get("series", {}).get("clean") or []
    if points:
        ax.plot([p[0] for p in points], [p[1] for p in points], linewidth=1.8,
                marker="o", markersize=2.8, color=BASELINE_COLOR, label="no tactile", zorder=2)
    mark_teacher(ax, teacher, name=first)
    ax.set_xlabel("Distillation iteration", fontsize=8, color=MUTED)
    ax.set_ylabel(AXIS_LABEL[task], fontsize=8, color=MUTED)
    if first:
        place_legend(ax, teacher, frameon=False, fontsize=9, labelcolor=INK, ncol=2)


def draw_levels(ax, subset, teacher, task, first):
    for level in ("clean", "mild", "strong"):
        run = next((r for r in subset if r["modality"] == "force_torque" and r["imperfection"] == level), None)
        points = (run or {}).get("series", {}).get("clean") or []
        if points:
            ax.plot([p[0] for p in points], [p[1] for p in points], linewidth=2,
                    marker="o", markersize=2.8, color=IMPERFECTION_COLOR[level], label=level, zorder=3)
    run = next((r for r in subset if r["modality"] == "none"), None)
    points = (run or {}).get("series", {}).get("clean") or []
    if points:
        ax.plot([p[0] for p in points], [p[1] for p in points], linewidth=2,
                marker="o", markersize=2.8, color=BASELINE_COLOR, label="no tactile", zorder=2)
    mark_teacher(ax, teacher, name=first)
    ax.set_xlabel("Distillation iteration", fontsize=8, color=MUTED)
    ax.set_ylabel(AXIS_LABEL[task], fontsize=8, color=MUTED)
    if first:
        place_legend(ax, teacher, frameon=False, fontsize=9, labelcolor=INK)


def draw(data, teachers, out_path, views, final_iteration=None, require_complete=False, dpi=200):
    runs = data["runs"]
    if require_complete:
        if final_iteration is None:
            raise ValueError("require_complete needs a final_iteration")
        require_complete_matrix(runs, final_iteration)
    excluded = 0
    if final_iteration is not None:
        keep = [r for r in runs if studied(r) and scored_at(r, final_iteration)]
        excluded = len(runs) - len(keep)
        # Curves stop at the checkpoint every cell is judged on, so a cell that ran further
        # does not appear to trail off past the others.
        runs = [{**r, "series": {k: [pt for pt in v if pt[0] <= final_iteration]
                                 for k, v in r.get("series", {}).items()}} for r in keep]
        if not runs:
            raise SystemExit(f"No runs reached iteration {final_iteration}; nothing to plot")
    pairs = [p for p in PAIR_TITLE if any(f"{r['task']}-{r['robot']}" == p for r in runs)]
    if not pairs:
        raise SystemExit("No runs for any known hand/task pair")
    height = 3.4 if len(views) == 1 else 6.6
    fig, axes = plt.subplots(len(views), len(pairs), figsize=(4.1 * len(pairs), height), squeeze=False)

    for col, pair in enumerate(pairs):
        task, robot = pair.rsplit("-", 1)
        subset = [r for r in runs if r["task"] == task and r["robot"] == robot]
        teacher = teachers.get(pair)
        for row, view in enumerate(views):
            ax = axes[row][col]
            style(ax)
            if view == "types":
                draw_types(ax, subset, teacher, task, col == 0)
            else:
                draw_levels(ax, subset, teacher, task, col == 0)
            if row == 0:
                ax.set_title(PAIR_TITLE[pair], fontsize=10, color=INK, pad=8)

    if excluded and not require_complete:
        fig.text(0.5, 0.005, f"Preliminary: {len(runs)} final runs at iteration {final_iteration}; "
                             f"{excluded} non-final attempts omitted.",
                 ha="center", fontsize=8, color=MUTED)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", facecolor=SURFACE, dpi=dpi)
    print(f"wrote {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--teachers", type=Path, help="JSON mapping '<task>-<hand>' to its teacher's score")
    parser.add_argument("--view", choices=["types", "levels", "both"], default="both")
    parser.add_argument("--final-iteration", type=int, default=None,
                        help="Plot only runs that reached this iteration; others are omitted and noted.")
    parser.add_argument("--require-complete", action="store_true",
                        help="Refuse to write unless all 39 publication cells are present once.")
    parser.add_argument("--dpi", type=int, default=200,
                        help="Raster output resolution (default: 200; PDF remains vector).")
    args = parser.parse_args()
    views = ("types", "levels") if args.view == "both" else (args.view,)
    teachers = json.loads(args.teachers.read_text()) if args.teachers else {}
    if args.require_complete and args.final_iteration is None:
        parser.error("--require-complete also requires --final-iteration")
    draw(json.loads(args.results.read_text()), teachers, args.output, views,
         args.final_iteration, args.require_complete, args.dpi)


if __name__ == "__main__":
    main()
