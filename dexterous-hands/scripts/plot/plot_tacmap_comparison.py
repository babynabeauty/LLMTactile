"""Ours-vs-TacMap comparison: FPS panel on top, peak GPU memory on the bottom.

``Ours`` = our Genesis tactile_sensor_speed_full*.txt at sensor=elastomer_taxel,
n_sensors=5. ``TacMap`` = the paper's ``sim_fps_w_5_tacmap`` /
``gpu_mb_w_5_tacmap`` columns (rows where either is blank — OOM — are dropped).

Both panels share x (number of environments) but have independent y-scales, so
ours' headroom doesn't crush TacMap's bars in the FPS panel and TacMap's larger
memory footprint doesn't shrink ours' bars in the memory panel.

Bar styling (font, colors, value labels, darkened outlines) is inherited from
plot_benchmark_sensor.py so this figure reads as part of the same family as
the combined benchmark plot.
"""

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot_benchmark_sensor as pbs  # noqa: E402 — shared bar drawing + label render.
import style  # noqa: F401 — registers Hanken Grotesk.

matplotlib.rcParams["hatch.linewidth"] = 0.6

ANSI_RE = re.compile(r"\x1B\[[0-9;?]*[ -/]*[@-~]")


def parse_value(value):
    value = value.strip()
    if value in {"True", "False"}:
        return value == "True"
    try:
        number = float(value)
    except ValueError:
        return value
    return int(number) if number.is_integer() else number


def parse_kv_line(line):
    line = ANSI_RE.sub("", line).strip()
    if not line or "=" not in line:
        return None
    record = {}
    for item in line.split("|"):
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        record[key.strip()] = parse_value(value)
    return record or None


def load_ours_tactile(paths, sensor, n_sensors):
    by_bs_fps = defaultdict(list)
    by_bs_mem = defaultdict(list)
    for path in paths:
        with Path(path).open() as fd:
            for line in fd:
                rec = parse_kv_line(line)
                if rec is None:
                    continue
                if rec.get("sensor") != sensor or rec.get("n_sensors") != n_sensors:
                    continue
                bs = rec.get("batch_size")
                if bs is None:
                    continue
                if "runtime_fps" in rec:
                    by_bs_fps[bs].append(rec["runtime_fps"])
                if "peak_mem_mb" in rec:
                    by_bs_mem[bs].append(rec["peak_mem_mb"])
    return [
        {
            "sensor": "ours",
            "n_sensors": n_sensors,
            "batch_size": bs,
            "runtime_fps": float(np.mean(by_bs_fps[bs])),
            "peak_mem_mb": float(np.mean(by_bs_mem[bs])) if by_bs_mem.get(bs) else None,
        }
        for bs in sorted(by_bs_fps)
    ]


def load_tacmap_csv(path, n_sensors, fps_std=50.0, mem_std=100.0):
    """One row in the CSV → two synthetic records at value ± std. draw_bars
    derives error-bar height from ``np.std`` across same-x records, so two
    points symmetric around the reading give exactly the requested std while
    preserving the mean. We add error bars because the TacMap numbers were
    read off the paper's bar plot, not reported exactly."""
    records = []
    with Path(path).open() as fd:
        for row in csv.DictReader(fd):
            envs = int(row["num_envs"])
            fps_raw = row.get("sim_fps_w_5_tacmap", "").strip()
            mem_raw = row.get("gpu_mb_w_5_tacmap", "").strip()
            if not fps_raw or not mem_raw:
                continue
            fps = float(fps_raw)
            mem = float(mem_raw)
            for sign in (-1, +1):
                records.append(
                    {
                        "sensor": "tacmap_baseline",
                        "n_sensors": n_sensors,
                        "batch_size": envs,
                        "runtime_fps": fps + sign * fps_std,
                        "peak_mem_mb": mem + sign * mem_std,
                    }
                )
    return records


def annotate_missing(ax, info, *, missing_sensor, missing_envs, label_specs, text="OOM"):
    """Append bar-style ``text`` labels at each (missing_sensor, env) slot
    where no bar was drawn. Creates invisible zero-height Rectangles at the
    would-be bar positions and pushes them through label_specs, so
    render_bar_labels handles positioning + styling (rotation, font weight,
    path-effect stroke) identically to the real value labels.

    Bar width / offset math mirrors draw_bars' default (constant-physical-inch)
    rule so the placeholder lands at the same offset a real bar would have."""
    import matplotlib.patches as mpatches

    sensors_drawn = info["sensors"]
    count_specs = info["count_specs"]
    series_specs = []
    for s in sensors_drawn:
        if s == "no_sensors":
            series_specs.append((s, None, False))
        else:
            series_specs.extend((s, c, n) for (c, n) in count_specs)
    if not series_specs:
        return
    try:
        i = next(idx for idx, (s, _c, _n) in enumerate(series_specs) if s == missing_sensor)
    except StopIteration:
        return
    n_series = len(series_specs)
    # Match draw_bars' default width rule (constant BAR_WIDTH_INCHES, converted
    # to data units via the axis's pixel extent).
    panel_width_in = ax.get_position().width * ax.figure.get_size_inches()[0]
    width = pbs.BAR_WIDTH_INCHES * len(info["all_x_values"]) / panel_width_in
    offset = width * (i - (n_series - 1) / 2)
    all_x = info["all_x_values"]
    for env in missing_envs:
        if env not in all_x:
            continue
        x = info["x"][all_x.index(env)] + offset
        # Zero-height invisible bar at the slot: render_bar_labels reads x,
        # width, height/y from this to anchor the label, and a height of 0
        # triggers the "outside" placement (text just past the tip — i.e.
        # just above y=0 here).
        bar = mpatches.Rectangle((x - width / 2, 0), width, 0, alpha=0, fill=False)
        ax.add_patch(bar)
        label_specs.append((ax, bar, text, True))


def plot_dual(records, *, args):
    fig, (ax_top, ax_bot) = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=tuple(args.figsize),
        gridspec_kw={"height_ratios": [1, 1], "hspace": 0.30},
    )
    # Margins tuned for the 4x3 figure the .sh currently uses. Tight bottom
    # since the sublabel was removed; left margin leaves room for the
    # "GPU memory (GB)" / "Env steps / sec" ylabels.
    fig.subplots_adjust(left=0.20, right=0.97, top=0.88, bottom=0.13, hspace=0.30)

    panel_args = SimpleNamespace(num_sensors=[args.n_sensors], noisy_num_sensors=[])
    label_specs = []

    info_top = pbs.draw_bars(
        ax_top,
        records,
        x_field="batch_size",
        y_field="runtime_fps",
        args=panel_args,
        label_specs=label_specs,
        unit=" FPS",
    )
    if args.ymax_fps is not None:
        ax_top.set_ylim(0, args.ymax_fps)
    # Short ylabel: at small figsizes the full "Environment steps per second"
    # is taller than one panel and clips at the figure top.
    ax_top.set_ylabel("Environment steps per second", fontsize=9)
    ax_top.set_title(args.title, fontsize="medium", pad=2)
    # Headroom above the tallest bar so the legend doesn't sit on top of it.
    cur_top = ax_top.get_ylim()[1]
    ax_top.set_ylim(0, cur_top * 1.25)
    pbs.add_legend(ax_top, info_top, loc="upper left", ncols=1, fontsize=7)

    # Memory panel: same capacity-style bars as the combined plot — saturated
    # color shows free GPU memory (capacity - peak), a lighter segment stacks
    # on top to the capacity ceiling, and the value label reports MB used and
    # sits outside the bar tip (at the free/used seam).
    cap_mb = args.ymax_mem_mb if args.ymax_mem_mb is not None else pbs.GPU_TOTAL_MB
    info_bot = pbs.draw_bars(
        ax_bot,
        records,
        x_field="batch_size",
        y_field="peak_mem_mb",
        args=panel_args,
        label_specs=label_specs,
        unit=" MB used",
        capacity=cap_mb,
        label_outside=True,
    )
    # Extra headroom above the capacity ceiling for the always-outside labels.
    ax_bot.set_ylim(0, cap_mb * 1.4)
    cap_gb = int(cap_mb // 1024)
    gb_step = 12 if cap_gb >= 36 else 6 if cap_gb >= 18 else max(1, cap_gb // 4)
    gb_ticks = list(range(0, cap_gb + 1, gb_step))
    ax_bot.set_yticks([g * 1024 for g in gb_ticks])
    ax_bot.set_yticklabels([str(g) for g in gb_ticks])
    ax_bot.set_ylabel("GPU memory (GB)", fontsize=9)
    ax_bot.set_xlabel(pbs.X_AXIS_LABELS["batch_size"], fontsize=9, labelpad=2)

    # OOM markers for env counts where ours has data but tacmap doesn't (the
    # tacmap CSV leaves `*_w_5_tacmap` blank past their last working entry).
    ours_envs = {r["batch_size"] for r in records if r["sensor"] == "ours"}
    tac_envs = {r["batch_size"] for r in records if r["sensor"] == "tacmap_baseline"}
    missing = sorted(ours_envs - tac_envs)
    if missing:
        annotate_missing(
            ax_top,
            info_top,
            missing_sensor="tacmap_baseline",
            missing_envs=missing,
            label_specs=label_specs,
            text="OOM",
        )
        annotate_missing(
            ax_bot,
            info_bot,
            missing_sensor="tacmap_baseline",
            missing_envs=missing,
            label_specs=label_specs,
            text="OOM",
        )

    pbs.render_bar_labels(fig, label_specs)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    # bbox_inches='tight' expands the saved bbox to fit any labels that
    # overflow the manual margins (rotated ylabels in particular tend to clip
    # on small figures). pad_inches keeps a small breathing edge.
    fig.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ours-files", nargs="+", type=Path, required=True)
    parser.add_argument("--tacmap-csv", type=Path, required=True)
    parser.add_argument("--sensor", default="elastomer_taxel")
    parser.add_argument("--n-sensors", type=int, default=5)
    parser.add_argument("--title", default="vs. TacMap")
    parser.add_argument("--ymax-fps", type=float, default=None)
    parser.add_argument(
        "--ymax-mem-mb",
        type=float,
        default=None,
        help="Capacity (MB) for the memory panel — bars stack a lighter "
        "segment from peak up to this ceiling. Defaults to A6000 48 GB.",
    )
    parser.add_argument(
        "--max-envs", type=int, default=2048, help="Cap on the env counts shown. Ours past this are dropped."
    )
    parser.add_argument("--figsize", type=float, nargs=2, default=(6.0, 5.0))
    parser.add_argument("--output", "-o", type=Path, required=True)
    args = parser.parse_args()

    ours = [
        r for r in load_ours_tactile(args.ours_files, args.sensor, args.n_sensors) if r["batch_size"] <= args.max_envs
    ]
    if not ours:
        raise SystemExit(f"No ours records for sensor={args.sensor!r}, n_sensors={args.n_sensors}.")
    ours_envs = {r["batch_size"] for r in ours}
    tacmap = [r for r in load_tacmap_csv(args.tacmap_csv, args.n_sensors) if r["batch_size"] in ours_envs]

    out = plot_dual(ours + tacmap, args=args)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
