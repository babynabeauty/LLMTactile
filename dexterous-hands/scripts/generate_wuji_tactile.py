"""Fit the Wuji Glove's regional taxel counts to the right Wuji Hand 2 mesh.

Run directly with the project Python; no Genesis initialization or device is needed.
Dense per-link grids preserve exact regional counts and approximate placement, not
hardware matrix indices. `actual` describes resolution, not measured robot taxel poses.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
HAND = ROOT / "src/assets/hands/wuji"
OUTPUT = ROOT / "src/assets/sensors/wuji/actual"
SOURCE_URL = "https://docs.wuji.tech/docs/en/wuji-glove/latest/sdk-data-reference/tactile/"
COUNTS = {"palm": 288, "thumb": 41, "index": 43, "middle": 58, "ring": 52, "pinky": 44}
# (prefix, distal grid, middle grid, proximal grid). All grids are dense and
# at least 2x2, including sensor types that cannot accept radius-zero padding.
FINGERS = {
    "thumb": ("r_thumb", (3, 4), (3, 5), (2, 7)),       # 12 + 15 + 14 = 41
    "index": ("r_index_finger", (2, 4), (3, 5), (4, 5)), # 8 + 15 + 20 = 43
    "middle": ("r_middle_finger", (4, 4), (3, 6), (4, 6)), # 16 + 18 + 24 = 58
    "ring": ("r_ring_finger", (2, 5), (3, 6), (4, 6)),  # 10 + 18 + 24 = 52
    "pinky": ("r_pinky", (3, 4), (4, 4), (4, 4)),      # 12 + 16 + 16 = 44
}

COLORS = dict(zip(COUNTS, ["#d99c24", "#db5571", "#48a46b", "#338ecc", "#9163bd", "#dd783b"]))


def source_mask() -> np.ndarray:
    """Read tile backgrounds, avoiding labels, from the AFTER half of the diagram."""
    path = OUTPUT / "reference/tactile-layout-change.png"
    pixels = np.asarray(Image.open(path).convert("RGB"))
    if pixels.shape != (928, 2088, 3):
        raise ValueError("Source diagram changed; revalidate the pixel calibration")
    scale = 2088 / 2048
    mask = np.zeros((24, 31), dtype=bool)
    for row in range(24):
        for col in range(31):
            x = round((1133 + col * 26.4 - 6) * scale)
            y = round((185 + row * 26.4 - 6) * scale)
            mask[row, col] = pixels[y, x].min() < 170
    if mask.sum() != 526:
        raise ValueError(f"Expected 526 source cells, got {mask.sum()}")
    return mask


def load_geometry():
    """Use the MJCF body frames and original STL coordinates, in metres."""
    tree = ET.parse(HAND / "mjcf/right.xml")
    bodies = {b.attrib["name"]: b for b in tree.findall(".//body")}
    meshes = {
        name: trimesh.load_mesh(HAND / f"meshes/right/{name}.STL")
        for name in bodies
        if (HAND / f"meshes/right/{name}.STL").exists()
    }
    return tree, bodies, meshes


@lru_cache(maxsize=32)
def query_mesh(mesh):
    # Trimesh's absolute triangle tolerances lose precision for small STL faces
    # in metre units. Query in mm, then serialize the result back in metres.
    return mesh.copy().apply_scale(1000)


def surface_probe(mesh, z: float, fraction: float):
    """Project a width-normalized location onto the exposed +Y surface.

    Cross-section bounds fit tapering pads and the palm outline. A 10% total
    width margin avoids grazing the mesh silhouette. No nearest-face fallback:
    a missed or back-facing ray is a generation error requiring a fit correction.
    """
    mesh = query_mesh(mesh)
    z *= 1000
    section = mesh.section(plane_normal=[0, 0, 1], plane_origin=[0, 0, z])
    if section is None:
        raise ValueError(f"Empty cross-section at z={z}")
    lo, hi = section.bounds[:, 0]
    x = lo + (hi - lo) * (0.05 + 0.90 * fraction)
    origin = np.array([x, mesh.bounds[1, 1] + 20.0, z])
    points, _, faces = mesh.ray.intersects_location([origin], [[0, -1, 0]], multiple_hits=True)
    if not len(points):
        raise ValueError(f"Projection missed at x={x}, z={z}")
    hit = np.argmax(points[:, 1])
    normal = mesh.face_normals[faces[hit]]
    if normal[1] <= 0.05:
        raise ValueError(f"Projection is not on the outward contact face: {normal}")
    return points[hit] / 1000, normal


def contact_strips(bodies, meshes):
    """Return link, region, grid shape, and axial bounds in fingertip-to-base order."""
    # The wrist above z=-54 mm contains open joint cutouts. Use continuous skin.
    yield "r_wrist", "palm", (16, 18), -0.054, -0.010
    for region, (prefix, tip_shape, middle_shape, proximal_shape) in FINGERS.items():
        distal, middle, proximal = (prefix + suffix for suffix in ("_tip_sensor_frame", "_middle", "_proximal_abd"))
        middle_length = -float(bodies[prefix + "_distal"].attrib["pos"].split()[2])
        proximal_length = -float(bodies[middle].attrib["pos"].split()[2])
        yield distal, region, tip_shape, float(meshes[distal].bounds[0, 2]) + 0.0015, -0.006
        # Thumb shells have distal slots; avoid these mechanical openings.
        yield middle, region, middle_shape, -0.026 if region == "thumb" else -middle_length + 0.003, -0.006
        yield proximal, region, proximal_shape, -0.025 if region == "thumb" else -proximal_length + 0.003, -0.006


def generate():
    assert source_mask().sum() == sum(COUNTS.values())
    _, bodies, meshes = load_geometry()
    records, layouts = [], []
    for link, region, (ny, nx), lo, hi in contact_strips(bodies, meshes):
        grid = []
        for row in range(ny):
            z = lo + (hi - lo) * (row + 0.5) / ny
            grid_row = []
            for col in range(nx):
                pos, normal = surface_probe(meshes[link], z, (col + 0.5) / nx)
                grid_row.append({"pos": pos.round(8).tolist(), "normal": normal.round(8).tolist()})
            grid.append(grid_row)
        positions = np.array([p["pos"] for row in grid for p in row])
        radius = round(float(np.median(cKDTree(positions).query(positions, k=2)[0][:, 1]) / 2), 8)
        for row in grid:
            for probe in row:
                probe["radius"] = radius
        records.append({"link_name": link, "kind": "grid", "probes": grid})
        layouts.append({"link_name": link, "region": region, "shape": [ny, nx], "count": ny * nx,
                        "radius_m": radius})
    manifest = {
        "source_url": SOURCE_URL, "active_count": 526, "region_counts": COUNTS,
        "geometry": "Wuji Hand 2 Beta 2 right; fitted positions, not measured robot hardware coordinates",
        "layout_note": ("Dense per-link grids preserve regional counts and rough placement. "
                        "No hardware row/column correspondence."),
        "radius_note": ("Half the median nearest-neighbor spacing per link; "
                        "simulation footprint, not vendor calibration."),
        "links": layouts,
        "source_sha256": {str(p.relative_to(OUTPUT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted((OUTPUT / "reference").glob("*.png"))},
        "mesh_sha256": {record["link_name"]: hashlib.sha256(
            (HAND / ("meshes/right/" + record["link_name"] + ".STL")).read_bytes()).hexdigest()
            for record in records},
        "mjcf_sha256": hashlib.sha256((HAND / "mjcf/right.xml").read_bytes()).hexdigest(),
    }
    return records, manifest


def validate(records, manifest):
    """Check exact counts, dense grids, geometry, and outward normals after serialization."""
    _, _, meshes = load_geometry()
    counts, worst = dict.fromkeys(COUNTS, 0), 0.0
    assert len(records) == len({r["link_name"] for r in records}) == 16
    for record, mapping in zip(records, manifest["links"], strict=True):
        assert record["link_name"] == mapping["link_name"]
        assert record["kind"] == "grid"
        ny, nx = mapping["shape"]
        assert ny >= 2 and nx >= 2 and len(record["probes"]) == ny
        assert all(len(row) == nx for row in record["probes"])
        positions, normals = [], []
        for row in record["probes"]:
            for probe in row:
                assert np.isfinite([*probe["pos"], *probe["normal"], probe["radius"]]).all()
                assert abs(np.linalg.norm(probe["normal"]) - 1) < 1e-6
                assert probe["radius"] == mapping["radius_m"] > 0
                counts[mapping["region"]] += 1
                positions.append(probe["pos"])
                normals.append(probe["normal"])
        positions, normals = np.array(positions), np.array(normals)
        mesh = meshes[record["link_name"]]
        _, distances, triangles = query_mesh(mesh).nearest.on_surface(positions * 1000)
        distances /= 1000
        worst = max(worst, float(distances.max()))
        assert distances.max() < 1e-7, (record["link_name"], distances.max())
        assert (normals[:, 1] > 0.05).all()
        assert (np.einsum("ij,ij->i", normals, mesh.face_normals[triangles]) > 0.99).all()
        assert cKDTree(positions).query(positions, k=2)[0][:, 1].min() > 1e-5
    assert counts == COUNTS, counts
    return {"active_count": sum(counts.values()), "region_counts": counts, "links": len(records),
            "max_surface_error_m": worst}


def body_transforms(tree, flexed=False):
    transforms = {}

    def walk(body, parent):
        transform = np.eye(4)
        q = np.fromstring(body.get("quat", "1 0 0 0"), sep=" ")
        transform[:3, :3] = Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()
        transform[:3, 3] = np.fromstring(body.get("pos", "0 0 0"), sep=" ")
        if flexed:
            for joint in body.findall("joint"):
                angle = 0.45 if not joint.get("name", "").endswith("abd") else 0.0
                axis = np.fromstring(joint.get("axis", "0 0 1"), sep=" ")
                rot = Rotation.from_rotvec(axis / np.linalg.norm(axis) * angle).as_matrix()
                anchor = np.fromstring(joint.get("pos", "0 0 0"), sep=" ")
                jt = np.eye(4)
                jt[:3, :3], jt[:3, 3] = rot, anchor - rot @ anchor
                transform = transform @ jt
        world = parent @ transform
        transforms[body.attrib["name"]] = world
        for child in body.findall("body"):
            walk(child, world)

    for body in tree.findall("./worldbody/body"):
        walk(body, np.eye(4))
    return transforms


def preview(records, manifest, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    tree, _, meshes = load_geometry()
    fig = plt.figure(figsize=(15, 7))
    for subplot, (flexed, elev, azim, title) in enumerate([
        (False, 0, 90, "Palm view"), (False, 22, 55, "Oblique view"), (True, 22, 55, "Flexed links")
    ], 1):
        ax = fig.add_subplot(1, 3, subplot, projection="3d")
        transforms = body_transforms(tree, flexed)
        for link, mesh in meshes.items():
            transform = transforms[link]
            vertices = trimesh.transform_points(mesh.vertices, transform) * 1000
            # Display original geometry; sensor-frame meshes intentionally overlap shells.
            poly = Poly3DCollection(vertices[mesh.faces], facecolor="#c4cbd2", linewidth=0,
                                    alpha=0.18, rasterized=True)
            ax.add_collection3d(poly)
        for record, mapping in zip(records, manifest["links"]):
            points = np.array([p["pos"] for row in record["probes"] for p in row if p["radius"] > 0])
            world = trimesh.transform_points(points, transforms[record["link_name"]]) * 1000
            ax.scatter(*world.T, s=9, color=COLORS[mapping["region"]], depthshade=False)
        ax.set(xlim=(-65, 65), ylim=(-40, 100), zlim=(-195, 10), title=title,
               xlabel="X (mm)", ylabel="Y (mm)", zlabel="Z (mm)")
        ax.set_box_aspect((130, 140, 205))
        ax.view_init(elev=elev, azim=azim, roll=180)
        ax.set_axis_off()
    fig.suptitle("Wuji actual · 526 active glove taxels fitted to robot meshes", fontsize=15)
    handles = [plt.Line2D([], [], marker="o", linestyle="", color=COLORS[r], label=f"{r}: {n}")
               for r, n in COUNTS.items()]
    fig.legend(handles=handles, loc="lower center", ncol=6)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig(destination, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="Validate and compare checked-in outputs without rewriting"
    )
    parser.add_argument("--preview", action="store_true", help="Render open and articulated placement previews")
    args = parser.parse_args()
    records, manifest = generate()
    outputs = {"probes_526_hand_wuji.json": records, "layout_metadata.json": manifest}
    if args.check:
        for name, value in outputs.items():
            if json.loads((OUTPUT / name).read_text()) != value:
                raise ValueError(f"{name} differs from deterministic generation")
    report = validate(records, manifest)
    if not args.check:
        for name, value in outputs.items():
            (OUTPUT / name).write_text(json.dumps(value, indent=2) + "\n")
    if args.preview:
        preview(records, manifest, OUTPUT / "placement_preview.png")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
