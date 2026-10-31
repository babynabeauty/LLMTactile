"""Ours-vs-TacSL force-field throughput comparison (FPS vs #envs, log-log).

TacSL (https://arxiv.org/pdf/2408.06506, Table II) reports tactile force-field
generation speed as a sweep over the number of parallel environments, at two
taxel resolutions: 10x10 (100 taxels) and 100x100 (10000 taxels). We transcribed
those numbers into ``data/sensor_benchmarks/tacsl.csv`` (columns:
``resolution,num_envs,fps`` where ``resolution`` is the taxel count, 100 or 10000).

This script overlays our own kinematic-taxel benchmark on the same axes. Our
numbers come from the ``probes_sensor_speed-*.txt`` logs, which sweep ``n_probes``
(the per-sensor probe/taxel count) and carry a ``batch_size`` (= num_envs) field.
We line up ``n_probes`` against TacSL's ``resolution`` (100 -> 10x10, 10000 ->
100x100) and average ``runtime_fps`` across repeat runs at each (n_probes,
batch_size).

Both sources are plotted as markered lines on log-log axes: color encodes the
source (ours vs TacSL, from the shared COLORS dict) and linestyle encodes
resolution (solid = low-res, dashed = high-res). When our logs only cover a
single env count the "line" is a single marker; rerun the benchmark with a
batch_size sweep to fill in the curve.
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot_benchmark_sensor as pbs  # noqa: E402 — shared parser + COLORS/labels.
import style  # noqa: F401 — registers Hanken Grotesk.

# TacSL resolution (taxel count) -> human label. Keys match tacsl.csv + our n_probes.
RESOLUTION_LABELS = {100: "10x10", 10000: "100x100"}
# Linestyle per resolution so a source reads as one color across both grids.
RESOLUTION_STYLES = {100: "-", 10000: "--"}
RESOLUTION_MARKERS = {100: "o", 10000: "s"}


def load_tacsl_csv(path):
    """tacsl.csv -> {resolution: [(num_envs, fps), ...]} sorted by num_envs."""
    curves = defaultdict(list)
    with Path(path).open() as fd:
        for row in csv.DictReader(fd):
            res = int(row["resolution"])
            curves[res].append((int(row["num_envs"]), float(row["fps"])))
    return {res: sorted(pts) for res, pts in curves.items()}


def load_ours_curves(paths, sensor, resolutions):
    """Parse our benchmark logs into {resolution: [(num_envs, mean_fps), ...]}.

    Keeps only rows for ``sensor`` whose ``n_probes`` matches one of the
    requested resolutions, then averages ``runtime_fps`` over repeat runs sharing
    the same (n_probes, batch_size).
    """
    by_key = defaultdict(list)  # (n_probes, batch_size) -> [fps, ...]
    for record in pbs.load_records(paths):
        if record.get("sensor") != sensor:
            continue
        n_probes = record.get("n_probes")
        batch = record.get("batch_size")
        fps = record.get("runtime_fps")
        if n_probes not in resolutions or batch is None or fps is None:
            continue
        by_key[(n_probes, batch)].append(fps)

    curves = defaultdict(list)
    for (n_probes, batch), vals in by_key.items():
        curves[n_probes].append((batch, sum(vals) / len(vals)))
    return {res: sorted(pts) for res, pts in curves.items()}


def _plot_source(ax, curves, *, color, label_prefix, alpha=1.0):
    """Draw one markered line per resolution for a single source."""
    for res in sorted(curves):
        pts = curves[res]
        xs = [x for x, _ in pts]
        ys = [y for _, y in pts]
        res_label = RESOLUTION_LABELS.get(res, str(res))
        ax.plot(
            xs,
            ys,
            color=color,
            alpha=alpha,
            linestyle=RESOLUTION_STYLES.get(res, "-"),
            marker=RESOLUTION_MARKERS.get(res, "o"),
            markersize=6,
            linewidth=2,
            label=f"{label_prefix} · {res_label}",
        )


def plot(tacsl_curves, ours_curves, *, args):
    fig, ax = plt.subplots(figsize=tuple(args.figsize))
    fig.subplots_adjust(left=0.13, right=0.985, top=0.90, bottom=0.13)

    _plot_source(
        ax,
        tacsl_curves,
        color=pbs.COLORS["tacsl_baseline"],
        label_prefix=pbs.SENSOR_LABELS["tacsl_baseline"],
    )
    _plot_source(
        ax,
        ours_curves,
        color=pbs.COLORS["ours"],
        label_prefix=pbs.SENSOR_LABELS["ours"],
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(pbs.X_AXIS_LABELS["batch_size"])
    ax.set_ylabel(pbs.Y_AXIS_LABELS["runtime_fps"])
    ax.set_title(args.title, fontsize="large", pad=4)
    ax.grid(True, which="major", axis="both", alpha=0.25, linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize="small", loc="upper left")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tacsl-csv", type=Path, required=True)
    parser.add_argument(
        "--ours-logs",
        type=Path,
        nargs="+",
        required=True,
        help="Benchmark log file(s) carrying our sensor's n_probes/batch_size/runtime_fps rows.",
    )
    parser.add_argument(
        "--sensor",
        default="kinematic_taxel",
        help="Which of our sensors to compare (must appear in the logs).",
    )
    parser.add_argument(
        "--resolutions",
        type=int,
        nargs="+",
        default=[100, 10000],
        help="Taxel counts to compare; must exist as both a tacsl.csv resolution and an n_probes value.",
    )
    parser.add_argument("--title", default="vs. TacSL")
    parser.add_argument("--figsize", type=float, nargs=2, default=(6.0, 4.5))
    parser.add_argument("--output", "-o", type=Path, required=True)
    args = parser.parse_args()

    resolutions = set(args.resolutions)
    tacsl_curves = {r: pts for r, pts in load_tacsl_csv(args.tacsl_csv).items() if r in resolutions}
    if not tacsl_curves:
        raise SystemExit(f"No rows in {args.tacsl_csv} for resolutions {sorted(resolutions)}.")

    ours_curves = load_ours_curves(args.ours_logs, args.sensor, resolutions)
    if not ours_curves:
        print(
            f"WARNING: no {args.sensor} rows at n_probes in {sorted(resolutions)} found in the logs; "
            "plotting TacSL only.",
            file=sys.stderr,
        )

    for res in sorted(resolutions):
        if res not in ours_curves:
            continue
        n_pts = len(ours_curves[res])
        if n_pts < 2:
            print(
                f"NOTE: our {args.sensor} curve at resolution {res} has only {n_pts} point "
                f"(env count{'s' if n_pts != 1 else ''}: {[x for x, _ in ours_curves[res]]}); "
                "rerun the benchmark with a batch_size sweep for a full curve.",
                file=sys.stderr,
            )

    out = plot(tacsl_curves, ours_curves, args=args)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
