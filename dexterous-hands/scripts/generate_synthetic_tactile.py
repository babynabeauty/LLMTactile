"""Generate low/high whole-hand contact layouts without modifying hardware-derived actual assets.

Budgets are approximate: each contact patch remains a dense grid with at least
2x2 cells. Both resolutions use identical contact patches. Radius covers the
worst sampled distance to a probe on each patch, including its boundary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import trimesh
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
BUDGETS = {"low": 128, "high": 512}
HANDS = ("wuji", "allegro", "sharpa")


def source_geometry(hand):
    if hand == "wuji":
        from generate_wuji_tactile import generate, load_geometry

        records, manifest = generate()
        _, _, meshes = load_geometry()
    elif hand == "sharpa":
        from generate_sharpa_tactile import generate, load_geometry

        records, manifest = generate("med")
        meshes, _, _ = load_geometry()
    else:
        from generate_allegro_tactile import HAND, generate

        records, manifest = generate()
        meshes = {m["link_name"]: trimesh.load_mesh(HAND / "mesh" / m["mesh"]) for m in manifest["links"]}
    return records, manifest, meshes


def surface_sampler(record, mesh):
    grid = np.array([[p["pos"] for p in row] for row in record["probes"]])
    axes = [np.linspace(0, 1, n) for n in grid.shape[:2]]
    interpolate = RegularGridInterpolator(axes, grid)
    mesh_mm = mesh.copy().apply_scale(1000)

    def sample(uv):
        raw = interpolate(uv)
        pos, _, faces = mesh_mm.nearest.on_surface(raw * 1000)
        # Nudge into the face before serialization, avoiding edge-normal ambiguity.
        pos = 0.9999 * pos + 0.0001 * mesh_mm.triangles_center[faces]
        return pos / 1000, mesh.face_normals[faces]

    # Sum segment lengths, retaining curvature on wrapped fingertips.
    height = np.linalg.norm(np.diff(grid, axis=0), axis=-1).sum(axis=0).mean()
    width = np.linalg.norm(np.diff(grid, axis=1), axis=-1).sum(axis=1).mean()
    return sample, float(height), float(width)


def uv_grid(rows, cols, *, boundary=False):
    if boundary:
        y, x = np.linspace(0, 1, rows), np.linspace(0, 1, cols)
    else:
        y, x = (np.arange(rows) + 0.5) / rows, (np.arange(cols) + 0.5) / cols
    return np.stack(np.meshgrid(y, x, indexing="ij"), axis=-1).reshape(-1, 2)


def generate(hand, profile):
    source, original, meshes = source_geometry(hand)
    patches = [surface_sampler(r, meshes[r["link_name"]]) for r in source]
    total_area = sum(h * w for _, h, w in patches)
    records, links = [], []
    for record, mapping, (sample, height, width) in zip(source, original["links"], patches, strict=True):
        count = BUDGETS[profile] * height * width / total_area
        rows = max(2, int(np.floor(np.sqrt(count * height / width) + 0.5)))
        cols = max(2, int(np.floor(np.sqrt(count * width / height) + 0.5)))
        positions, normals = sample(uv_grid(rows, cols))
        # At least 8 validation points across each cell, also sampling boundaries.
        coverage, _ = sample(uv_grid(max(33, rows * 8 + 1), max(33, cols * 8 + 1), boundary=True))
        cover_distance = float(cKDTree(positions).query(coverage)[0].max())
        radius = float(np.ceil(cover_distance * 1.05 * 1e8) / 1e8)
        probes = [
            dict(pos=p.round(8).tolist(), normal=n.round(8).tolist(), radius=radius)
            for p, n in zip(positions, normals, strict=True)
        ]
        records.append(
            dict(
                link_name=record["link_name"],
                kind="grid",
                probes=[probes[i * cols : (i + 1) * cols] for i in range(rows)],
            )
        )
        links.append(
            dict(
                link_name=record["link_name"],
                region=mapping["region"],
                shape=[rows, cols],
                count=rows * cols,
                radius_m=radius,
                max_coverage_distance_m=cover_distance,
                contact_patch_area_m2=height * width,
            )
        )
    manifest = dict(
        hand=hand,
        profile=profile,
        target_count=BUDGETS[profile],
        active_count=sum(m["count"] for m in links),
        links=links,
        coverage="Palm and finger contact patches; excludes dorsal surfaces and joint gaps",
        radius_note="1.05 times maximum sampled patch-to-nearest-probe distance, including boundaries",
        source_layout_sha256=hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest(),
        source_geometry=original.get("mesh_sha256", {}),
    )
    validate(records, manifest, meshes)
    return records, manifest


def validate(records, manifest, meshes):
    assert len(records) == len({r["link_name"] for r in records})
    for r, m in zip(records, manifest["links"], strict=True):
        assert r["link_name"] == m["link_name"] and min(m["shape"]) >= 2
        p = [p for row in r["probes"] for p in row]
        pos, normals = np.array([p["pos"] for p in p]), np.array([p["normal"] for p in p])
        assert len(p) == m["count"] and np.isfinite(pos).all() and np.isfinite(normals).all()
        np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1, atol=1e-6)
        mesh = meshes[r["link_name"]].copy().apply_scale(1000)
        _, distance, faces = mesh.nearest.on_surface(pos * 1000)
        assert distance.max() < 1e-4, (r["link_name"], distance.max())
        assert (np.einsum("ij,ij->i", normals, mesh.face_normals[faces]) > 0.95).all()
        assert cKDTree(pos).query(pos, k=2)[0][:, 1].min() > 1e-6
        assert all(p["radius"] == m["radius_m"] > m["max_coverage_distance_m"] for p in p)
    assert any(m["region"] == "palm" for m in manifest["links"])
    assert manifest["active_count"] == sum(m["count"] for m in manifest["links"])


def preview(records, manifest, destination):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
    import mujoco

    hand = manifest["hand"]
    files = {
        "wuji": "wuji/mjcf/right.xml",
        "sharpa": "sharpa/right_sharpa_wave.xml",
        "allegro": "allegro/xela_v4/right_hand.xml",
    }
    model = mujoco.MjModel.from_xml_path(str(ROOT / "src/assets/hands" / files[hand]))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    _, _, meshes = source_geometry(hand)
    world_meshes, clouds, circles = [], [], []
    for record in records:
        link = record["link_name"]
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, link)
        rot, offset = data.xmat[body].reshape(3, 3), data.xpos[body]
        mesh = meshes[link]
        world_meshes.append((mesh.vertices @ rot.T + offset)[mesh.faces] * 1000)
        pos = np.array([p["pos"] for row in record["probes"] for p in row])
        clouds.append((pos @ rot.T + offset) * 1000)
        for row in record["probes"]:
            for probe in row:
                normal = np.array(probe["normal"])
                basis = np.eye(3)[np.argmin(np.abs(normal))]
                u = np.cross(normal, basis)
                u /= np.linalg.norm(u)
                v = np.cross(normal, u)
                theta = np.linspace(0, 2 * np.pi, 17)
                ring = np.array(probe["pos"]) + probe["radius"] * (
                    np.cos(theta)[:, None] * u + np.sin(theta)[:, None] * v
                )
                circles.append((ring @ rot.T + offset) * 1000)
    points = np.concatenate(clouds)
    center = (points.max(0) + points.min(0)) / 2
    extent = max(np.ptp(points, axis=0)) * 0.6
    views = [(0, 90, 180), (22, 55, 180), (20, 135, 180)] if hand == "wuji" else [(0, 0, 0), (20, 35, 0), (20, 125, 0)]
    fig = plt.figure(figsize=(13, 6))
    for i, (elev, azim, roll) in enumerate(views, 1):
        ax = fig.add_subplot(1, 3, i, projection="3d")
        for triangles in world_meshes:
            ax.add_collection3d(
                Poly3DCollection(triangles, facecolor="#bac6cf", linewidth=0, alpha=0.15, rasterized=True)
            )
        ax.add_collection3d(Line3DCollection(circles, colors="#3286ad", linewidths=0.4, alpha=0.4))
        ax.scatter(*points.T, s=3, color="#105773", depthshade=False)
        ax.set(
            xlim=(center[0] - extent, center[0] + extent),
            ylim=(center[1] - extent, center[1] + extent),
            zlim=(center[2] - extent, center[2] + extent),
        )
        ax.set_box_aspect((1, 1, 1))
        ax.set_axis_off()
        ax.view_init(elev=elev, azim=azim, roll=roll)
    fig.suptitle(f"{hand} {manifest['profile']}: {manifest['active_count']} probes on palm and fingers")
    fig.text(0.5, 0.08, "Circles show sensing radii in each probe's tangent plane", ha="center")
    fig.tight_layout()
    fig.savefig(destination, dpi=130)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hand", choices=HANDS, nargs="+", default=HANDS)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    for hand in args.hand:
        for profile in BUDGETS:
            records, manifest = generate(hand, profile)
            folder = ROOT / "src/assets/sensors" / hand / profile
            files = {"probes_hand.json": records, "layout_metadata.json": manifest}
            if args.check:
                for name, value in files.items():
                    assert json.loads((folder / name).read_text()) == value, folder / name
            else:
                folder.mkdir(parents=True, exist_ok=True)
                for name, value in files.items():
                    (folder / name).write_text(json.dumps(value, indent=2) + "\n")
                preview(records, manifest, folder / "placement_preview.png")
            print(f"{hand}/{profile}: {manifest['active_count']} probes, {len(records)} contact patches", flush=True)


if __name__ == "__main__":
    main()
