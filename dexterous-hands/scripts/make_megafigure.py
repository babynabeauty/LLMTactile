#!/usr/bin/env python
"""Assemble the paper's summary figure: rollout frames, tactile maps, and both ablations.

One column per hand/task pair and four bands down the page. The top band shows the winning
sensor's rollout twice at the same instant, once in a single environment and once across
sixteen; the second band is that sensor's taxel map at that instant; the last two are the
representation and imperfection ablations already drawn by ``plot_distill_ablations``.

The instant is chosen from the recording rather than fixed, at the frame where the sensor reads
the most over the most taxels, so the still shows the sensor doing something.

    python scripts/make_megafigure.py results.json -o figs/megafigure.pdf \\
        --teachers docs/experiments/teacher-scores.json
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from plot_distill_ablations import (  # noqa: E402
    PAIR_TITLE, INK, MUTED, SURFACE, draw_levels, draw_types, require_complete_matrix, scored_at,
    studied, style,
)
from plot_taxel_handmap import (  # noqa: E402
    UNITS, assemble, label_regions, layout, marker_size, projected_cells,
)

# The sensor that leads each pair, and the two recordings of it: one environment close up,
# and sixteen in a 4x4 grid (the object set has 17 variants, so one is left out).
BEST = {
    "in_hand_repose-wuji": ("wuji", "agg_force", "in_hand_repose-agg_force"),
    "in_fingers_rotate-sharpa": ("sharpa", "force_torque", "in_fingers_rotate-force_torque"),
    "grasp_lift-allegro": ("allegro", "depth", "grasp_lift-depth"),
}


def peak_instant(magnitude):
    """The frame where the sensor is doing the most, so the still is worth showing.

    Total magnitude alone can peak on one saturated taxel; weighting it by how many taxels are
    reading favours a frame with a real contact patch over a single spike.
    """
    load = magnitude.sum(axis=1)
    spread = (magnitude > 0.05 * magnitude.max()).sum(axis=1)
    return int(np.argmax(load * np.sqrt(spread)))


def video_frame(path: Path, index: int, fps: float, into: Path) -> Path:
    """One frame of a rollout, trimmed of its surrounding whitespace."""
    out = into / f"{path.stem}-{index}.png"
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-ss", f"{index / fps:.3f}", "-i", str(path),
         "-frames:v", "1", str(out)],
        check=True,
    )
    return out


def trimmed(path: Path) -> np.ndarray:
    """Crop a near-white border so the hand fills its panel."""
    image = mpimg.imread(path)
    rgb = image[..., :3]
    ink = (rgb.min(axis=2) < 0.93).any(axis=1), (rgb.min(axis=2) < 0.93).any(axis=0)
    rows, cols = np.flatnonzero(ink[0]), np.flatnonzero(ink[1])
    if not len(rows) or not len(cols):
        return image
    pad = 8
    r0, r1 = max(rows[0] - pad, 0), min(rows[-1] + pad, image.shape[0] - 1)
    c0, c1 = max(cols[0] - pad, 0), min(cols[-1] + pad, image.shape[1] - 1)
    return image[r0:r1 + 1, c0:c1 + 1]


def draw_map(ax, xs, ys, regions, values, vmax, size):
    ax.set_aspect("equal")
    ax.axis("off")
    pad = 0.07 * max(np.ptp(xs), np.ptp(ys))
    ax.set_xlim(xs.min() - pad, xs.max() + pad)
    ax.set_ylim(ys.min() - pad, ys.max() + pad)
    dots = ax.scatter(xs, ys, c=values, cmap="plasma", vmin=0.0, vmax=vmax, s=size)
    label_regions(ax, xs, ys, regions)
    return dots


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--teachers", type=Path)
    parser.add_argument("--final-iteration", type=int, default=2500)
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    import json

    data = json.loads(args.results.read_text())
    teachers = json.loads(args.teachers.read_text()) if args.teachers else {}
    if args.require_complete:
        require_complete_matrix(data["runs"], args.final_iteration)
    runs = [{**r, "series": {k: [p for p in v if p[0] <= args.final_iteration]
                             for k, v in r.get("series", {}).items()}}
            for r in data["runs"] if studied(r) and scored_at(r, args.final_iteration)]

    pairs = [p for p in PAIR_TITLE if any(f"{r['task']}-{r['robot']}" == p for r in runs)]
    figure = plt.figure(figsize=(13.6, 12.2))
    # Rollouts and maps get less height than the plots; the maps carry a colourbar column.
    grid = figure.add_gridspec(4, len(pairs), height_ratios=[1.35, 1.75, 1.9, 1.9], hspace=0.14,
                               wspace=0.16, left=0.045, right=0.985, top=0.945, bottom=0.045)

    with tempfile.TemporaryDirectory() as scratch:
        scratch = Path(scratch)
        for column, pair in enumerate(pairs):
            hand, sensor, stem = BEST[pair]
            npz = np.load(ROOT / f"rollouts/fig-solo-{stem}.npz", allow_pickle=False)
            links = layout(hand)
            field, positions = assemble(npz, links, sensor, "mean")
            xs, ys, _, regions = projected_cells(links, positions)
            label, factor = UNITS.get(sensor, ("Reading", 1.0))
            magnitude = (np.linalg.norm(field, axis=-1) if field.ndim == 3 else np.abs(field)) * factor
            if magnitude.max() < 0.01 and "(" in label:
                base, _, unit = label.rpartition("(")
                magnitude, label = magnitude * 1000.0, f"{base}(m{unit.rstrip(')')})"
            peak = float(magnitude.max())
            active = magnitude[magnitude > 0.02 * peak]
            vmax = (float(np.percentile(active, 99)) if active.size else peak) or 1.0
            times = npz["t"]
            fps = (len(times) - 1) / float(times[-1]) if float(times[-1]) else 40.0

            instant = peak_instant(magnitude)

            # Band 1: the same moment seen once close up and again across sixteen environments.
            band = grid[0, column].subgridspec(1, 2, width_ratios=[1, 1.25], wspace=0.04)
            solo = figure.add_subplot(band[0, 0])
            solo.imshow(trimmed(video_frame(ROOT / f"rollouts/fig-solo-{stem}.mp4", instant, fps, scratch)))
            solo.axis("off")
            solo.set_xlabel("one environment", fontsize=7, color=MUTED)
            grid16 = figure.add_subplot(band[0, 1])
            grid16.imshow(trimmed(video_frame(ROOT / f"rollouts/fig-grid17-{stem}.mp4", instant, fps, scratch)))
            grid16.axis("off")
            figure.text(grid[0, column].get_position(figure).x0, 0.975, PAIR_TITLE[pair],
                        fontsize=12, color=INK, ha="left", va="top")

            # Band 2: the taxel map at that same instant, beside its colourbar.
            band = grid[1, column].subgridspec(1, 2, width_ratios=[1, 0.05], wspace=0.04)
            ax = figure.add_subplot(band[0, 0])
            dots = draw_map(ax, xs, ys, regions, magnitude[instant], vmax,
                            marker_size(xs, ys, panels=1))
            cax = figure.add_subplot(band[0, 1])
            bar = figure.colorbar(dots, cax=cax,
                                  extend="max" if (magnitude > vmax).any() else "neither")
            bar.set_label(label, fontsize=7, color=MUTED)
            bar.ax.tick_params(labelsize=6, colors=MUTED)

            task, robot = pair.rsplit("-", 1)
            subset = [r for r in runs if r["task"] == task and r["robot"] == robot]
            for row, drawer in ((2, draw_types), (3, draw_levels)):
                ax = figure.add_subplot(grid[row, column])
                style(ax)
                drawer(ax, subset, teachers.get(pair), task, column == 0)
                if row == 2:  # Same x axis as the row below it, labelled once.
                    ax.set_xlabel("")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=args.dpi, facecolor=SURFACE, bbox_inches="tight")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
