#!/usr/bin/env python
"""Render per-taxel readings as a hand-shaped map, frame by frame.

Reads the ``.npz`` written by ``main.py --mode=play --save_tactile`` together with the hand's
probe layout, and draws every taxel at its place on a flattened hand: one column per finger with
the distal pad at the top, and the palm patches below. Taxels are colored by reading magnitude
and, for the vector sensors, carry an arrow for the in-pad direction.

The recorder captures one environment, so the map shows env 0 while a rollout video may tile
several. Writes an MP4, or a contact sheet of sampled frames with ``--contact-sheet``.

    python scripts/plot_taxel_handmap.py readings.npz --sensor tactile_force -o handmap.mp4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FFMpegWriter  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
# Thumb first, then the fingers as they sit on the hand; anything unrecognized lands after.
REGION_ORDER = ["thumb", "index", "middle", "ring", "pinky", "palm"]
# What each sensor reads, and the factor that puts it in a unit worth printing on a colourbar:
# depth, elastomer displacement and proximity are metres, and a millimetre scale keeps the
# ticks from reading 0.0000.
UNITS = {
    "depth": ("Contact depth (mm)", 1000.0),
    "elastomer": ("Marker displacement (mm)", 1000.0),
    "proximity": ("Proximity (mm)", 1000.0),
    "force": ("Force magnitude (N)", 1.0),
    "force_torque": ("Force magnitude (N)", 1.0),
    "agg_force": ("Aggregate force (N)", 1.0),
    "link_force": ("Link force (N)", 1.0),
    "bool": ("Contact", 1.0),
    "agg_bool": ("Contact", 1.0),
    "link_bool": ("Contact", 1.0),
}


def layout(hand: str) -> list[dict]:
    meta = json.loads((ROOT / f"src/assets/sensors/{hand}/actual/layout_metadata.json").read_text())
    return meta["links"]


def reference_positions(hand: str) -> np.ndarray:
    """Probe positions from the layout file, in its own order and reference pose."""
    directory = ROOT / f"src/assets/sensors/{hand}/actual"
    probe_file = next(p for p in sorted(directory.glob("probes_*.json")))
    points = []
    for link in json.loads(probe_file.read_text()):
        grid = link["probes"]
        for row in (grid if isinstance(grid[0], list) else [grid]):
            points.extend(probe["pos"] for probe in row)
    return np.asarray(points, dtype=float)


def spec_type(sensor_name: str) -> str | None:
    """``tactile_force_torque_r_wrist`` -> ``force_torque`` (link names contain underscores too)."""
    from tactile_sensors import spec_for_sensor_name

    spec = spec_for_sensor_name(sensor_name)
    return spec.name if spec else None


def distal_first(link: dict) -> tuple[int, str]:
    """Order a finger's pads from fingertip down to the knuckle."""
    name = link["link_name"]
    for rank, token in enumerate(("tip", "_3", "middle", "_2", "proximal", "_1B", "_1A")):
        if token in name:
            return rank, name
    return len(name), name


def pad_grids(hand: str) -> dict[str, np.ndarray]:
    """Reference probe positions per link, shaped (rows, cols, 3)."""
    directory = ROOT / f"src/assets/sensors/{hand}/actual"
    probe_file = next(iter(sorted(directory.glob("probes_*.json"))))
    grids = {}
    for link in json.loads(probe_file.read_text()):
        grid = link["probes"]
        rows = grid if isinstance(grid[0], list) else [grid]
        grids[link["link_name"]] = np.array([[p["pos"] for p in row] for row in rows], dtype=float)
    return grids


def along_rows(links: list[dict], grids: dict[str, np.ndarray]) -> dict[str, bool]:
    """Whether a pad's grid rows run along its finger, rather than around it.

    The two hands disagree -- Wuji indexes rows down the finger, Allegro indexes columns -- so
    the drawing orientation is measured instead of assumed: compare how the grid's row and
    column directions line up with the finger's own axis, taken as palm centroid to pad. Hands
    that sensorize only fingertips have no palm to measure from, so those fall back to the pad's
    longer physical side.
    """
    palm = [grids[link["link_name"]].reshape(-1, 3) for link in links if link.get("region") == "palm"]
    centre = np.concatenate(palm).mean(axis=0) if palm else None
    out = {}
    for link in links:
        name = link["link_name"]
        grid = grids[name]
        if link.get("region") == "palm":
            out[name] = True
            continue
        row_dir = grid[-1].mean(axis=0) - grid[0].mean(axis=0)
        col_dir = grid[:, -1].mean(axis=0) - grid[:, 0].mean(axis=0)
        if centre is None:
            out[name] = np.linalg.norm(row_dir) >= np.linalg.norm(col_dir)
            continue
        axis = grid.reshape(-1, 3).mean(axis=0) - centre
        axis = axis / (np.linalg.norm(axis) + 1e-12)
        project = lambda v: abs(float(np.dot(v / (np.linalg.norm(v) + 1e-12), axis)))  # noqa: E731
        out[name] = project(row_dir) >= project(col_dir)
    return out


def drawn_shape(link: dict, vertical: bool) -> tuple[int, int]:
    """Pad extent as drawn: (height, width) in cells, after any transpose."""
    rows, cols = link["shape"]
    return (rows, cols) if vertical else (cols, rows)


def link_spans(links):
    """Where each link's taxels sit on the flat probe axis, with their region."""
    spans, regions, start = [], [], 0
    for link in links:
        count = link["shape"][0] * link["shape"][1]
        spans.append((link["link_name"], start, start + count))
        regions.extend([link.get("region", "other")] * count)
        start += count
    return spans, regions


def palm_plane(positions, regions):
    """Two in-plane axes seen from the palm side, or None when the hand has no palm.

    Taking the plane from the palm alone rather than from the whole cloud matters: the fingers
    are curled around an object during a rollout, so a fit over every taxel tilts the view and
    foreshortens them. The palm is one rigid slab, and looking down its normal is the view a
    hand emoji shows.
    """
    picked = [i for i, region in enumerate(regions) if region == "palm"]
    if not picked:
        return None
    centred = positions[picked] - positions[picked].mean(axis=0)
    _, _, axes = np.linalg.svd(centred, full_matrices=False)
    return axes[:2]


def fingertip_row(links, positions, spans, regions):
    """Lay fingertip-only hands out as a hand rather than where the grasp put them.

    A hand that sensorizes just its fingertips gives no palm to orient against, and during a
    grasp the pads point inward at unrelated angles, so a projection scatters them. Each pad is
    instead flattened and placed in finger order, with the thumb opposed.

    Orientation comes from the pad's own grid rather than from where it sits: every layout is
    generated by the same procedure, so row 0 is the same end of every pad, and building the
    frame from the grid's row and column directions puts all the pads the same way up. Deriving
    it from the pad's position instead leaves the sign at the mercy of how the fingers happen to
    be curled, which flips whichever pads point the other way.
    """
    order = {name: index for index, name in enumerate(REGION_ORDER)}
    shapes = {link["link_name"]: link["shape"] for link in links}
    fingers = [s for s in spans if regions[s[1]] != "thumb"]
    fingers.sort(key=lambda s: order.get(regions[s[1]], len(order)))
    thumbs = [s for s in spans if regions[s[1]] == "thumb"]

    xs = np.zeros(len(regions))
    ys = np.zeros(len(regions))
    placed, gap = [], None
    for name, start, stop in fingers + thumbs:
        rows, cols = shapes[name]
        block = positions[start:stop]
        grid = block.reshape(rows, cols, 3)
        # Along the grid's rows and along its columns, in the pad's own frame.
        down = grid[-1].mean(axis=0) - grid[0].mean(axis=0)
        across = grid[:, -1].mean(axis=0) - grid[:, 0].mean(axis=0)
        # Longer side stands vertically, so a pad reads as a fingertip rather than a stripe.
        if np.linalg.norm(across) > np.linalg.norm(down):
            down, across = across, down
        # Row 0 of these layouts is the base of the pad, so the row direction runs toward the
        # fingertip and is the way up. Side flips with it, which turns the pad rather than
        # mirroring it.
        up = down / (np.linalg.norm(down) + 1e-12)
        side = np.dot(across, up) * up - across
        side = side / (np.linalg.norm(side) + 1e-12)
        centred = block - block.mean(axis=0)
        local = np.stack([centred @ side, centred @ up], axis=1)
        placed.append((start, stop, local * 1000.0))
        gap = max(gap or 0.0, np.ptp(local[:, 0]) * 1000.0)

    cursor = 0.0
    for index, (start, stop, local) in enumerate(placed):
        is_thumb = index >= len(fingers)
        if is_thumb:
            # Opposed: below the row and off its leading edge, the way a thumb faces the fingers.
            offset = np.array([-1.35 * gap, -1.5 * gap])
        else:
            offset = np.array([cursor, 0.0])
            cursor += gap * 1.35
        xs[start:stop] = local[:, 0] + offset[0]
        ys[start:stop] = local[:, 1] + offset[1]
    return xs, ys


def projected_cells(links, positions):
    """Place every taxel where it sits on the hand, seen from the palm side."""
    spans, regions = link_spans(links)
    axes = palm_plane(positions, regions)
    if axes is None:
        xs, ys = fingertip_row(links, positions, spans, regions)
        return xs, ys, spans, regions

    flat = (positions - positions.mean(axis=0)) @ axes.T

    def centroid(predicate):
        picked = [i for i, region in enumerate(regions) if predicate(region)]
        return flat[picked].mean(axis=0) if picked else None

    # The plane carries no orientation of its own: rotate so the palm-to-fingers direction
    # points up, then mirror if that leaves the thumb on the right.
    palm = centroid(lambda r: r == "palm")
    fingers = centroid(lambda r: r not in ("palm", "thumb"))
    if palm is not None and fingers is not None:
        along = fingers - palm
        angle = np.arctan2(along[1], along[0]) - np.pi / 2
        cos, sin = np.cos(-angle), np.sin(-angle)
        flat = flat @ np.array([[cos, sin], [-sin, cos]])

    thumb = centroid(lambda r: r == "thumb")
    others = centroid(lambda r: r not in ("thumb", "palm"))
    if thumb is not None and others is not None and thumb[0] > others[0]:
        flat = flat * np.array([-1.0, 1.0])

    # Millimetres read better than metres on the axes, and keep marker sizing sane.
    return flat[:, 0] * 1000.0, flat[:, 1] * 1000.0, spans, regions


def assemble(npz, links: list[dict], sensor_type: str, substep: str):
    """Stitch the per-link sensors into one (T, taxels, ...) field in layout link order.

    ``main.py --save_tactile`` writes one sensor per link, named ``tactile_<type>_<link>``, and
    each carries every simulation substep: with ``--temporal_reduction=none`` a control step
    holds ``decimation`` readings per taxel, so the probe axis is ``count * H`` long. Matching
    on link name rather than a flat probe order removes any ordering guesswork.
    """
    available = {str(n) for n in npz["sensor_names"]}
    blocks, positions, missing = [], [], []
    for link in links:
        name = f"tactile_{sensor_type}_{link['link_name']}"
        if name not in available:
            missing.append(link["link_name"])
            continue
        data = npz[f"{name}__data"]
        count = link["shape"][0] * link["shape"][1]
        if data.shape[1] % count:
            raise SystemExit(f"{name}: {data.shape[1]} rows do not divide {count} taxels")
        horizon = data.shape[1] // count
        # The sensors emit (N, H, *probe_layout): the substep axis comes before the probes,
        # so the flattened probe axis is substep-major. Reshaping it the other way round
        # interleaves substeps into neighbouring taxels and paints stripes across a pad.
        framed = data.reshape(data.shape[0], horizon, count, *data.shape[2:])
        if substep == "mean":
            framed = framed.mean(axis=1)
        elif substep == "last":
            framed = framed[:, -1]
        else:
            framed = framed[:, int(substep)]
        blocks.append(framed)
        positions.append(npz[f"{name}__pos"][0] if f"{name}__pos" in npz else None)
    if missing:
        raise SystemExit(f"No recorded sensor for: {', '.join(missing)}")
    stacked = np.concatenate(blocks, axis=1)
    known = [p for p in positions if p is not None]
    return stacked, (np.concatenate(known, axis=0) if len(known) == len(positions) else None)


def check_order(spans, recorded: np.ndarray, reference: np.ndarray, tol: float = 2e-4) -> None:
    """Fail loudly if the npz's probe order does not match the layout file's.

    The npz stores one flat probe axis, and a mismatch would put readings on the wrong taxels
    silently. Recorded positions are world-frame in whatever pose the hand happened to be in, so
    they cannot be compared to the layout's reference pose directly -- but a link is a rigid
    body, so each probe's distance to its own link's centroid is invariant to that pose and to
    the order of the links themselves.
    """
    for name, start, stop in spans:
        got, want = recorded[start:stop], reference[start:stop]
        radii = np.linalg.norm(got - got.mean(axis=0), axis=1)
        expected = np.linalg.norm(want - want.mean(axis=0), axis=1)
        worst = float(np.abs(radii - expected).max())
        if worst > tol:
            raise SystemExit(
                f"Probe order does not match the layout: {name} taxel radii differ by up to "
                f"{worst * 1e3:.2f} mm. Re-record, or pass --skip-order-check."
            )


def pad_axes(positions: np.ndarray, spans) -> dict[str, np.ndarray]:
    """A 2D basis per pad, so a world-frame vector can be drawn on the flattened map."""
    axes = {}
    for name, start, stop in spans:
        block = positions[start:stop]
        centered = block - block.mean(axis=0)
        # The two directions the pad actually spans; the third is its normal.
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        axes[name] = vt[:2]
    return axes


def marker_size(xs, ys, panels: int = 1) -> float:
    """Dot area that fills the taxel pitch without the dense pads merging into a blob."""
    span = max(np.ptp(xs), np.ptp(ys)) or 1.0
    pitch = span / max(np.sqrt(len(xs)), 1.0)
    return float(np.clip((pitch / span) * 900.0 / panels ** 0.5, 2.0, 40.0))


def label_regions(ax, xs, ys, regions) -> None:
    """One label per region: above it, or to its left when above is occupied.

    Above keeps the finger labels in a row, which is how the map reads. The exception is a
    thumb tucked under the fingers, as on Wuji, where the space above belongs to another pad;
    only a label that would land on taxels is moved, so the row stays intact.
    """
    span = max(np.ptp(xs), np.ptp(ys))
    reach = 0.045 * span
    points = np.column_stack([xs, ys])
    for region in dict.fromkeys(regions):
        mask = np.array([r == region for r in regions])
        x, y = xs[mask], ys[mask]
        if region == "palm":
            at, align = (x.mean(), y.min() - reach), ("center", "top")
        else:
            at, align = (x.mean(), y.max() + reach), ("center", "bottom")
            # Test the whole word, not its anchor: a centred label above a narrow pad still
            # runs sideways over its neighbours, which is what happens to a tucked-in thumb.
            half = 0.011 * span * len(region)
            probe = np.column_stack([np.linspace(at[0] - half, at[0] + half, 2 * len(region)),
                                     np.full(2 * len(region), at[1])])
            gaps = np.linalg.norm(points[None, :, :] - probe[:, None, :], axis=2).min(axis=1)
            if float(gaps.min()) < 0.5 * reach:
                at, align = (x.min() - reach, y.mean()), ("right", "center")
        ax.text(at[0], at[1], region, ha=align[0], va=align[1], fontsize=7, color="#52514e")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("readings", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--sensor", help="Sensor type, e.g. force_torque; default: the only one recorded")
    parser.add_argument("--hand", help="Override the hand recorded in the npz")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--vmax", type=float, default=None,
                        help="Color ceiling; default: a percentile of the active readings")
    parser.add_argument("--vmax-percentile", type=float, default=99.0)
    parser.add_argument("--active-fraction", type=float, default=0.02,
                        help="Fraction of the peak above which a taxel counts as reading")
    parser.add_argument("--substep", default="mean",
                        help="Which simulation substep to show: mean, last, or an index")
    parser.add_argument("--contact-sheet", type=int, default=0, metavar="N",
                        help="Write N sampled frames as a PNG strip instead of a video")
    parser.add_argument("--skip-order-check", action="store_true")
    args = parser.parse_args()

    npz = np.load(args.readings, allow_pickle=False)
    names = [str(n) for n in npz["sensor_names"]]
    types = sorted({spec_type(n) for n in names} - {None})
    sensor = args.sensor or (types[0] if types else None)
    if sensor not in types:
        raise SystemExit(f"{sensor!r} not recorded; available types: {', '.join(types)}")

    hand = args.hand or str(npz["robot"])
    if not hand:
        raise SystemExit("The recording does not name its hand; pass --hand")
    links = layout(hand)
    data, positions = assemble(npz, links, sensor, args.substep)
    if positions is None:
        raise SystemExit("The recording has no probe positions; the hand map needs them")
    # World-frame positions from the rollout itself: the taxels are drawn where they actually
    # are on the hand, not where a layout file happens to store them link-locally.
    xs, ys, spans, regions = projected_cells(links, positions)
    if data.shape[1] != len(xs):
        raise SystemExit(f"{sensor} has {data.shape[1]} taxels, layout has {len(xs)}")
    label, factor = UNITS.get(sensor, ("Reading", 1.0))
    magnitude = (np.linalg.norm(data, axis=-1) if data.ndim == 3 else np.abs(data)) * factor
    # Aggregate channels read in fractions of a millinewton, which leaves a colourbar of
    # zeroes; drop to milli-units when the whole episode sits near the bottom of the unit.
    if magnitude.max() < 0.01 and "(" in label:
        base, _, unit = label.rpartition("(")
        magnitude = magnitude * 1000.0
        label = f"{base}(m{unit.rstrip(')')})"
    # Scaling to the episode maximum washes the map out: contact touches ~4% of the taxels and
    # the peak is a brief spike, so everything typical renders as the bottom of the ramp. The
    # ceiling is a high percentile of the taxels actually reading something. "Actually reading"
    # has to be relative to the peak, not an absolute force: depth reads millimetres and the
    # aggregate channels are smaller still, so a fixed threshold in newtons excludes every
    # taxel on those sensors and silently leaves the map blank.
    peak = float(magnitude.max())
    active = magnitude[magnitude > args.active_fraction * peak]
    vmax = args.vmax or (float(np.percentile(active, args.vmax_percentile)) if active.size else peak) or 1.0
    clipped = float((magnitude > vmax).mean())

    if positions is not None and not args.skip_order_check:
        check_order(spans, positions, reference_positions(hand))

    frames = range(len(data))
    if args.contact_sheet:
        frames = np.linspace(0, len(data) - 1, args.contact_sheet).astype(int)

    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    ax.set_aspect("equal")
    ax.axis("off")
    pad = 0.06 * max(np.ptp(xs), np.ptp(ys))
    ax.set_xlim(xs.min() - pad, xs.max() + pad)
    ax.set_ylim(ys.min() - pad, ys.max() + pad)
    size = marker_size(xs, ys)
    scatter = ax.scatter(xs, ys, c=np.zeros_like(xs), cmap="plasma", vmin=0.0, vmax=vmax, s=size)
    label_regions(ax, xs, ys, regions)
    bar = fig.colorbar(scatter, ax=ax, fraction=0.03, pad=0.02,
                       extend="max" if clipped else "neither")
    bar.set_label(label, fontsize=9)
    spelled = sensor.replace("_", " ")
    title = ax.set_title(f"{hand} · {spelled}", fontsize=10)

    def draw(index):
        scatter.set_array(magnitude[index])
        title.set_text(f"{hand} · {spelled} · t={float(npz['t'][index]):.2f}s")
        return scatter

    if args.contact_sheet:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        sheet, sheet_axes = plt.subplots(1, len(frames), figsize=(3.2 * len(frames), 4.2))
        dots = None
        for cell, index in zip(np.atleast_1d(sheet_axes), frames):
            cell.set_aspect("equal")
            cell.axis("off")
            pad = 0.06 * max(np.ptp(xs), np.ptp(ys))
            cell.set_xlim(xs.min() - pad, xs.max() + pad)
            cell.set_ylim(ys.min() - pad, ys.max() + pad)
            dots = cell.scatter(xs, ys, c=magnitude[index], cmap="plasma", vmin=0.0, vmax=vmax,
                                s=marker_size(xs, ys, panels=len(frames)))
            label_regions(cell, xs, ys, regions)
            cell.set_title(f"t={float(npz['t'][index]):.2f}s", fontsize=9)
        bar = sheet.colorbar(dots, ax=np.atleast_1d(sheet_axes).tolist(), fraction=0.02, pad=0.01,
                             extend="max" if clipped else "neither")
        bar.set_label(label, fontsize=9)
        sheet.suptitle(f"{hand} · {sensor.replace(chr(95), chr(32))}", fontsize=11)
        sheet.savefig(args.output, dpi=160, facecolor="white", bbox_inches="tight")
        print(f"wrote {args.output} ({len(frames)} frames, vmax={vmax:.4f})")
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = FFMpegWriter(fps=args.fps, bitrate=2400)
    with writer.saving(fig, str(args.output), dpi=140):
        for index in frames:
            draw(index)
            writer.grab_frame()
    print(f"wrote {args.output} ({len(data)} frames, {len(xs)} taxels, "
          f"vmax={vmax:.4f}, {100 * clipped:.3f}% of readings clipped)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
