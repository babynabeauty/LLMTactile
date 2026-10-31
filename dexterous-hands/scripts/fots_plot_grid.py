"""Render the 2x4 marker-comparison grid from the npz written by
``fots_marker_compare.py --save-markers``:

    rows = [dilate, shear];  cols = [Real, FOTS, HydroShear, Ours]

Real cells show the GelSight image + tracked flow; the model cells show the
ContactDepth backdrop (one shared indentation-depth colorbar) + that model's
best-fit marker flow. No Genesis needed -- pure read + plot.

    python scripts/fots_plot_grid.py --npz outputs/fots_compare/markers.npz \
        --out outputs/fots_compare/grid.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
FOTS_DATA = REPO_ROOT / "reference" / "FOTS" / "marker_calib" / "data"

MOTIONS = ["dilate", "shear"]  # rows
COLS = [("real", "Real"), ("fots", "FOTS"), ("hydroshear", "HydroShear"), ("elastomer", "Ours")]  # columns
QUIVER = dict(color="aqua", angles="xy", scale_units="xy", scale=0.3)


def _quiver_extras(ax, center, stop, center_rot=0.0, stop_rot=0.0):
    # "+" rotated by the indenter z-rotation; (4, 2, ang) = 4-spoke asterisk at `ang` deg.
    # negate so a +z (pad, y-up) rotation reads correctly on the y-down image axis.
    if center is not None and np.all(np.isfinite(center)):
        ax.plot(
            center[0],
            center[1],
            marker=(4, 2, -center_rot),
            color="deeppink",
            markersize=12,
            markeredgewidth=2,
            linestyle="None",
        )
        if stop is not None and np.all(np.isfinite(stop)):
            ax.annotate(
                "",
                xy=(stop[0], stop[1]),
                xytext=(center[0], center[1]),
                arrowprops=dict(arrowstyle="->", color="deeppink", lw=2),
            )
            ax.plot(
                stop[0],
                stop[1],
                marker=(4, 2, -stop_rot),
                color="deeppink",
                markersize=12,
                markeredgewidth=2,
                linestyle="None",
            )


def main() -> None:
    import sys

    sys.path.insert(0, str(REPO_ROOT / "scripts" / "plot"))
    import matplotlib.pyplot as plt
    import style  # noqa: F401 -- registers the bundled Hanken Grotesk font on import
    from PIL import Image

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", required=True, help="marker npz from --save-markers")
    ap.add_argument("--out", default="outputs/fots_compare/grid.png")
    ap.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        default=None,
        metavar=("W", "H"),
        help="figure size in inches (default: 4 per column x 4 per row)",
    )
    args = ap.parse_args()
    if not Path(args.npz).exists():
        raise SystemExit(f"no markers npz at {args.npz} -- run the compare/save step first "
                         f"(every --save-markers run failed or was skipped?).")
    d = np.load(args.npz)

    # shared depth color scale across every model cell present
    depths = [d[k] for k in d.files if k.endswith("__depth")]
    dmax_mm = max((float(x.max()) for x in depths), default=1.0) * 1000.0

    motions = [m for m in MOTIONS if f"{m}__pos" in d.files]
    # image aspect (H/W) so default figure matches the grid -> aspect-equal panels fill, no whitespace
    img0 = np.asarray(Image.open(FOTS_DATA / f"{int(d[f'{motions[0]}__frame'][0])}.png"))
    aspect = img0.shape[0] / img0.shape[1]
    panel_w = 4.0
    figsize = args.figsize if args.figsize else (panel_w * len(COLS), panel_w * aspect * len(motions))
    fig, axes = plt.subplots(len(motions), len(COLS), figsize=figsize, squeeze=False, constrained_layout=True)
    im_ref = None
    for r, motion in enumerate(motions):
        pos = d[f"{motion}__pos"].reshape(-1, 2)
        real = d[f"{motion}__real"].reshape(-1, 2)
        center = d.get(f"{motion}__center")
        stop = d.get(f"{motion}__stop")
        center_rot = float(d[f"{motion}__center_rot"][0]) if f"{motion}__center_rot" in d.files else 0.0
        stop_rot = float(d[f"{motion}__stop_rot"][0]) if f"{motion}__stop_rot" in d.files else 0.0
        frame = int(d[f"{motion}__frame"][0])
        img = np.asarray(Image.open(FOTS_DATA / f"{frame}.png"))
        H, W = img.shape[0], img.shape[1]
        extent = [0, W, H, 0]
        for c, (model, label) in enumerate(COLS):
            ax = axes[r][c]
            if model == "real":
                ax.imshow(img, extent=extent)
                flow = real
            else:
                dk, sk = f"{motion}__{model}__depth", f"{motion}__{model}__sim"
                if sk not in d.files:
                    ax.set_axis_off()
                    if r == 0:
                        ax.set_title(label)
                    continue
                if dk in d.files:
                    im_ref = ax.imshow(
                        d[dk] * 1000.0, extent=extent, cmap="magma", origin="upper", vmin=0.0, vmax=dmax_mm
                    )
                flow = d[sk].reshape(-1, 2)
            ax.quiver(pos[:, 0], pos[:, 1], flow[:, 0], flow[:, 1], **QUIVER)
            _quiver_extras(ax, center, stop, center_rot, stop_rot)
            ax.set_xlim(0, W)
            ax.set_ylim(H, 0)
            ax.set_aspect("equal")
            ax.set_axis_off()
            if r == 0:
                ax.set_title(label, fontsize=13)
        axes[r][0].text(
            -0.04,
            0.5,
            motion,
            transform=axes[r][0].transAxes,
            rotation=90,
            va="center",
            ha="right",
            fontsize=13,
            fontweight="bold",
        )

    if im_ref is not None:
        fig.colorbar(im_ref, ax=axes, location="right", fraction=0.02, pad=0.02, label="indentation depth (mm)")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"saved grid -> {out}")


if __name__ == "__main__":
    main()
