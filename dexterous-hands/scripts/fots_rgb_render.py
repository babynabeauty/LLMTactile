"""Render a GelSight-style RGB image from a precomputed ContactDepth map, using FOTS's
optical model (normals -> MLP normal->color -> + calibrated background).

Decoupled from Genesis: first precompute depth maps with
    python scripts/fots_marker_compare.py --motion dilate --center-px 133 123 \
        --ball-radius 6 --depth 0.2 --save-depth outputs/fots_compare/depths.npz
then iterate on RGB here (fast, no sim):
    python scripts/fots_rgb_render.py --npz outputs/fots_compare/depths.npz

The FOTS optical assets (background, bg offset, MLP weights) live under reference/FOTS/.
We replicate FOTSRender.generate locally to avoid FOTS's package __init__ (which imports a
hardware driver). Orientation: our depth is landscape (H=240,W=320); FOTS's renderer is
portrait (320x240), so we transpose in and back out.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

MM_TO_PIXEL = 19.58
SENSOR_W, SENSOR_H = 240, 320                 # FOTS portrait render frame (cols, rows)
PYRAMID_KERNELS = [51, 21, 11, 5]
KERNEL = 5

REPO_ROOT = Path(__file__).resolve().parents[1]
FOTS = REPO_ROOT / "reference" / "FOTS"
BG_PATH = FOTS / "assets" / "gel" / "gelsight_bg.npy"
BG_RENDER_PATH = FOTS / "utils" / "utils_data" / "ini_bg_fots_gelsight.npy"
MLP_PATH = FOTS / "mlp_calib" / "models" / "mlp_n2c_gelsight.pth"


class MLP(nn.Module):  # copy of reference/FOTS/mlp_calib/src/train/mlp_model.py
    def __init__(self, input_size=5, output_size=3, hidden_size=32):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, output_size)
        self.drop = nn.Dropout(p=0.05)

    def forward(self, x):
        x = self.drop(F.relu(self.fc1(x)))
        x = self.drop(F.relu(self.fc2(x)))
        x = self.drop(self.fc3(x))
        return self.fc4(x)


def _smooth_height(height, contact_mask):
    back = height.copy()
    for k in PYRAMID_KERNELS:
        height = cv2.GaussianBlur(height.astype(np.float32), (k, k), 0)
        height[contact_mask] = back[contact_mask]
    return cv2.GaussianBlur(height.astype(np.float32), (KERNEL, KERNEL), 0)


def _normals(height):
    h, w = height.shape
    c = height[1:h - 1, 1:w - 1]; _ = c
    top, bot = height[0:h - 2, 1:w - 1], height[2:h, 1:w - 1]
    left, right = height[1:h - 1, 0:w - 2], height[1:h - 1, 2:w]
    dzdx, dzdy = (bot - top) / 2.0, (right - left) / 2.0
    d = np.ones((h - 2, w - 2, 3))
    d[:, :, 0], d[:, :, 1] = dzdy, -dzdx
    n = d / np.linalg.norm(d, axis=2, keepdims=True)
    n = np.pad(n, ((1, 1), (1, 1), (0, 0)), "symmetric")
    return (n + 1.0) * 0.5


def _preproc(normal):
    xy = np.flip(np.column_stack(np.where(np.all(normal >= -100, axis=2))), axis=1)
    nxyz = normal.reshape(-1, 3)
    df = pd.DataFrame(np.hstack([xy, nxyz]), columns=["X", "Y", "Nx", "Ny", "Nz"])
    df["X"], df["Y"] = df["X"] / SENSOR_W, df["Y"] / SENSOR_H
    return torch.tensor(df[["X", "Y", "Nx", "Ny", "Nz"]].values, dtype=torch.float32)


class FotsOptical:
    """Replicates reference/FOTS/utils/fots_render.py FOTSRender.generate (no shadow)."""

    def __init__(self):
        self.background = np.load(BG_PATH).astype(np.float32)            # (320,240,3) BGR
        self.bg_render = np.load(BG_RENDER_PATH).astype(np.float32)      # (320,240,3)
        self.model = MLP()
        self.model.load_state_dict(torch.load(MLP_PATH, map_location="cpu"))
        self.model.eval()

    def render(self, height_px, contact_mask):
        """height_px, contact_mask: (320,240). Returns uint8 BGR (320,240,3)."""
        height = _smooth_height(height_px, contact_mask)
        with torch.no_grad():
            rgb = self.model(_preproc(_normals(height))).cpu().numpy().reshape(SENSOR_H, SENSOR_W, 3)
        sim = (rgb - self.bg_render) * 255.0 + self.background
        return np.clip(sim, 0, 255).astype(np.uint8)


def depth_to_rgb(optical: FotsOptical, depth_m: np.ndarray) -> np.ndarray:
    """Our landscape depth (H=240,W=320, meters) -> RGB (240,320,3) aligned to the markers."""
    height_px = depth_m * 1000.0 * MM_TO_PIXEL          # meters -> px indentation
    height_port = height_px.T                            # landscape (240,320) -> portrait (320,240)
    bgr = optical.render(height_port, height_port > 1e-6)
    rgb = bgr[..., ::-1]                                  # BGR -> RGB
    return np.ascontiguousarray(rgb.transpose(1, 0, 2))  # portrait -> landscape (240,320,3)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", required=True, help="depth npz from --save-depth")
    ap.add_argument("--key", default=None, help="which depth key to render (default: all)")
    ap.add_argument("--outdir", default="outputs/fots_compare", help="where to save RGB pngs")
    args = ap.parse_args()

    data = np.load(args.npz)
    keys = [args.key] if args.key else [k for k in data.files if not k.endswith("__meta")]
    optical = FotsOptical()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    import matplotlib.pyplot as plt
    for key in keys:
        depth_m = data[key]
        rgb = depth_to_rgb(optical, depth_m)
        fig, ax = plt.subplots(1, 2, figsize=(11, 5))
        im = ax[0].imshow(depth_m * 1000.0, cmap="magma"); ax[0].set_title(f"ContactDepth (mm) — {key}")
        fig.colorbar(im, ax=ax[0], fraction=0.046)
        ax[1].imshow(rgb); ax[1].set_title("FOTS-rendered RGB")
        for a in ax:
            a.set_xlabel("px")
        fig.tight_layout()
        outpath = outdir / f"rgb_{key}.png"
        fig.savefig(outpath, dpi=120)
        plt.close(fig)
        print(f"rendered {key}: RGB range [{rgb.min()},{rgb.max()}] -> {outpath}")


if __name__ == "__main__":
    main()
