"""Ours-vs-HydroShear FPS comparison (single panel).

Reads ``hydroshear.csv`` (columns: ``source,num_envs,mean_ms,std_ms``) and plots
env-steps/sec — converted via ``num_envs * 1000 / mean_ms`` — for each source as
a bar per env count. The CSV does not carry GPU memory, so this plot is FPS
only; for a memory comparison see ``plot_tacmap_comparison.py``.

Bar styling is inherited from plot_benchmark_sensor.py so this figure reads as
part of the same family as the combined benchmark plot.
"""

import argparse
import csv
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot_benchmark_sensor as pbs  # noqa: E402 — shared bar drawing + label render.
import style  # noqa: F401 — registers Hanken Grotesk.
from plot_tacmap_comparison import annotate_missing  # noqa: E402 — shared OOM/N/A helper.

matplotlib.rcParams["hatch.linewidth"] = 0.6


def load_hydroshear_csv(path, n_sensors):
    """One record per (source, num_envs) row. mean_ms is the total step time
    across num_envs envs, so env-steps/sec = num_envs * 1000 / mean_ms."""
    records = []
    sensor_for = {"ours": "ours", "hydroshear": "hydroshear_baseline"}
    with Path(path).open() as fd:
        for row in csv.DictReader(fd):
            src = row["source"].strip()
            if src not in sensor_for:
                continue
            envs = int(row["num_envs"])
            ms = float(row["mean_ms"])
            records.append(
                {
                    "sensor": sensor_for[src],
                    "n_sensors": n_sensors,
                    "batch_size": envs,
                    "runtime_fps": envs * 1000.0 / ms,
                }
            )
    return records


def plot_single(records, *, args):
    fig, ax = plt.subplots(figsize=tuple(args.figsize))
    fig.subplots_adjust(left=0.16, right=0.985, top=0.90, bottom=0.18)

    panel_args = SimpleNamespace(num_sensors=[args.n_sensors], noisy_num_sensors=[])
    label_specs = []
    info = pbs.draw_bars(
        ax,
        records,
        x_field="batch_size",
        y_field="runtime_fps",
        args=panel_args,
        label_specs=label_specs,
        unit=" FPS",
    )
    if args.ymax_fps is not None:
        ax.set_ylim(0, args.ymax_fps)
    ax.set_ylabel(pbs.Y_AXIS_LABELS["runtime_fps"])
    ax.set_xlabel(pbs.X_AXIS_LABELS["batch_size"])
    ax.set_title(args.title, fontsize="large", pad=4)
    pbs.add_legend(ax, info, loc="upper left", ncols=1, fontsize="small")

    # N/A markers for env counts where ours has data but HydroShear doesn't
    # (their paper lists 2048 envs as "not reported").
    ours_envs = {r["batch_size"] for r in records if r["sensor"] == "ours"}
    hs_envs = {r["batch_size"] for r in records if r["sensor"] == "hydroshear_baseline"}
    missing = sorted(ours_envs - hs_envs)
    if missing:
        annotate_missing(
            ax, info, missing_sensor="hydroshear_baseline", missing_envs=missing, label_specs=label_specs, text="N/A"
        )

    pbs.render_bar_labels(fig, label_specs)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hydroshear-csv", type=Path, required=True)
    parser.add_argument("--n-sensors", type=int, default=5)
    parser.add_argument("--title", default="vs. HydroShear")
    parser.add_argument("--ymax-fps", type=float, default=None)
    parser.add_argument("--figsize", type=float, nargs=2, default=(6.0, 4.0))
    parser.add_argument("--output", "-o", type=Path, required=True)
    args = parser.parse_args()

    records = load_hydroshear_csv(args.hydroshear_csv, args.n_sensors)
    if not records:
        raise SystemExit(f"No usable rows in {args.hydroshear_csv}.")

    out = plot_single(records, args=args)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
