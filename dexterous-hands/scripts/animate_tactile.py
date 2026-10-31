"""Animate tactile sensor readings saved by ``main.py --mode=play --save_tactile``.

Usage:
    python scripts/animate_tactile.py logs/<run>/play/.../tactile_readings.npz
    python scripts/animate_tactile.py tactile_readings.npz --output tactile.mp4 --fps 20

Reads the .npz written by ``src/tactile_record.py``: per ``tactile_*`` sensor, a
per-step world-frame probe-position array and a per-step reading. Vector sensors
(elastomer / force_torque / force / proximity) read a 3D vector per probe; the
contact sensors (bool / depth / agg_force / agg_bool / link_*) read a scalar per probe.

Lays out one subplot per sensor and animates the episode:
  - vector sensor  -> 3D quiver of per-probe vectors anchored at the probes,
  - scalar sensor  -> 3D scatter of probes colored by the reading,
  - scalar w/o probe positions (link_* sensors) -> per-probe value-vs-time lines
    with a moving time cursor.

This mirrors the plotting split in ``scripts/sensors_viewer.py`` (vector-field vs
line plot), rendered offline from a recording instead of live via gs.recorders.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # noqa: E402 - headless render; no display needed.
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter  # noqa: E402


def _load(npz_path: Path) -> tuple[np.ndarray, list[dict], str]:
    """Load the recording into ``(t, sensors, title)``; one dict per sensor."""
    data = np.load(npz_path, allow_pickle=True)
    files = set(data.files)
    names = [str(x) for x in data["sensor_names"]]
    types = [str(x) for x in data["sensor_types"]]
    kinds = [str(x) for x in data["sensor_kinds"]]
    t = np.asarray(data["t"], dtype=np.float64)

    sensors: list[dict] = []
    for name, stype, kind in zip(names, types, kinds):
        if f"{name}__data" not in files:
            continue
        sensors.append(
            {
                "name": name,
                "type": stype,
                "kind": kind,
                "data": np.asarray(data[f"{name}__data"], dtype=np.float64),
                "pos": np.asarray(data[f"{name}__pos"], dtype=np.float64) if f"{name}__pos" in files else None,
            }
        )

    task = str(data["task"]) if "task" in files else ""
    robot = str(data["robot"]) if "robot" in files else ""
    title = " / ".join(p for p in (task, robot) if p) or Path(npz_path).stem
    return t, sensors, title


def _set_equal_3d(ax, pts: np.ndarray, margin: float = 0.15) -> None:
    """Give a 3D axis equal aspect with limits covering ``pts`` plus a margin."""
    lo = pts.reshape(-1, 3).min(axis=0)
    hi = pts.reshape(-1, 3).max(axis=0)
    center = (lo + hi) / 2.0
    span = float((hi - lo).max()) or 1.0
    half = span * (0.5 + margin)
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)
    ax.set_box_aspect((1.0, 1.0, 1.0))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")


def _make_vector_subplot(fig, n_rows: int, n_cols: int, idx: int, sensor: dict):
    """3D quiver of per-probe vectors anchored at moving probe positions."""
    ax = fig.add_subplot(n_rows, n_cols, idx, projection="3d")
    pos = sensor["pos"]  # (T, P, 3)
    vec = sensor["data"]  # (T, P, 3)

    max_norm = float(np.linalg.norm(vec, axis=-1).max())
    span = float((pos.reshape(-1, 3).max(axis=0) - pos.reshape(-1, 3).min(axis=0)).max()) or 1.0
    # Scale arrows so the largest reading spans ~25% of the probe cloud.
    scale = (0.25 * span / max_norm) if max_norm > 0 else 0.0

    _set_equal_3d(ax, pos)
    ax.set_title(f"{sensor['name']}\n({sensor['type']}, vector)", fontsize=8)
    ax.scatter(pos[0, :, 0], pos[0, :, 1], pos[0, :, 2], s=6, c="0.6", depthshade=False)
    state: dict = {"quiver": None}

    def update(frame: int) -> None:
        if state["quiver"] is not None:
            state["quiver"].remove()
        p = pos[frame]
        v = vec[frame] * scale
        state["quiver"] = ax.quiver(
            p[:, 0], p[:, 1], p[:, 2], v[:, 0], v[:, 1], v[:, 2], color="tab:red", linewidth=1.0
        )

    return update


def _make_scalar_subplot(fig, n_rows: int, n_cols: int, idx: int, sensor: dict):
    """3D scatter of probes colored by the scalar reading."""
    ax = fig.add_subplot(n_rows, n_cols, idx, projection="3d")
    pos = sensor["pos"]  # (T, P, 3)
    vals = sensor["data"]  # (T, P)

    vmin, vmax = (float(np.percentile(vals, 2)), float(np.percentile(vals, 98)))
    if vmin == vmax:
        vmax = vmin + 1e-6

    _set_equal_3d(ax, pos)
    ax.set_title(f"{sensor['name']}\n({sensor['type']}, scalar)", fontsize=8)
    scat = ax.scatter(pos[0, :, 0], pos[0, :, 1], pos[0, :, 2], c=vals[0], cmap="inferno", vmin=vmin, vmax=vmax, s=28)
    fig.colorbar(scat, ax=ax, shrink=0.6, pad=0.1)

    def update(frame: int) -> None:
        p = pos[frame]
        scat._offsets3d = (p[:, 0], p[:, 1], p[:, 2])
        scat.set_array(vals[frame])

    return update


def _make_line_subplot(fig, n_rows: int, n_cols: int, idx: int, sensor: dict, t: np.ndarray):
    """Per-probe value-vs-time lines with a moving cursor (no probe positions)."""
    ax = fig.add_subplot(n_rows, n_cols, idx)
    vals = sensor["data"]  # (T, P)

    for probe in range(vals.shape[1]):
        ax.plot(t, vals[:, probe], linewidth=0.8)
    cursor = ax.axvline(t[0], color="k", linewidth=1.0)
    ax.set_title(f"{sensor['name']}\n({sensor['type']}, scalar)", fontsize=8)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("reading")

    def update(frame: int) -> None:
        cursor.set_xdata([t[frame], t[frame]])

    return update


def build_animation(npz_path: Path, fps: int) -> tuple[FuncAnimation, plt.Figure]:
    t, sensors, title = _load(npz_path)
    if not sensors:
        raise SystemExit(f"No tactile sensor data found in {npz_path}")
    n_frames = len(t)

    n = len(sensors)
    n_cols = min(n, 3)
    n_rows = math.ceil(n / n_cols)
    fig = plt.figure(figsize=(5.0 * n_cols, 4.5 * n_rows))
    fig.suptitle(f"{title}  --  {npz_path.name}", fontsize=10)

    updaters = []
    for i, sensor in enumerate(sensors):
        idx = i + 1
        if sensor["kind"] == "vector" and sensor["pos"] is not None:
            updaters.append(_make_vector_subplot(fig, n_rows, n_cols, idx, sensor))
        elif sensor["pos"] is not None:
            updaters.append(_make_scalar_subplot(fig, n_rows, n_cols, idx, sensor))
        else:
            updaters.append(_make_line_subplot(fig, n_rows, n_cols, idx, sensor, t))

    time_text = fig.text(0.5, 0.94, "", ha="center", fontsize=9)

    def update(frame: int):
        for upd in updaters:
            upd(frame)
        time_text.set_text(f"t = {t[frame]:.3f} s   (frame {frame + 1}/{n_frames})")
        return []

    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000.0 / fps, blit=False)
    return anim, fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("npz", type=Path, help="Path to the tactile_readings.npz from --save_tactile.")
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output animation path (.mp4 or .gif). Default: alongside the npz as <stem>.mp4.",
    )
    parser.add_argument("--fps", type=int, default=20, help="Animation frame rate.")
    parser.add_argument("--dpi", type=int, default=120, help="Output resolution (dots per inch).")
    args = parser.parse_args()

    if not args.npz.is_file():
        raise SystemExit(f"npz file not found: {args.npz}")

    output = args.output or args.npz.with_suffix(".mp4")
    anim, fig = build_animation(args.npz, fps=args.fps)

    if output.suffix.lower() == ".gif":
        writer = PillowWriter(fps=args.fps)
    else:
        writer = FFMpegWriter(fps=args.fps, bitrate=4000)
    print(f"Rendering {output} ...")
    anim.save(str(output), writer=writer, dpi=args.dpi)
    plt.close(fig)
    print(f"Saved animation to {output}")


if __name__ == "__main__":
    main()
