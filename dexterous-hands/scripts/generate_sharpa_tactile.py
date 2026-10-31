"""Fit dense link-local grids to the Sharpa Wave MJCF contact surfaces.

actual samples fingertip pads at nominal 1 mm projected pitch (spatial resolution,
not the native optical image pixels). med matches Wuji actual's regional counts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from generate_wuji_tactile import COLORS, COUNTS, query_mesh, surface_probe

ROOT = Path(__file__).resolve().parents[1]
HAND = ROOT / "src/assets/hands/sharpa"
OUTPUT = ROOT / "src/assets/sensors/sharpa"
MODEL = HAND / "right_sharpa_wave.xml"
SOURCE = "https://github.com/sharpa-robotics/sharpa-tactile-sensor-assets/tree/865530a98a0ca0e69d177f2121833f8bb3ed94de"
FILES = {"actual": "probes_1206_tips_sharpa.json", "med": "probes_526_hand_sharpa.json"}


def load_geometry():
    tree = ET.parse(MODEL)
    assets = {m.get("name"): m for m in tree.findall("./asset/mesh")}
    meshes, paths, display = {}, {}, {}
    for body in tree.findall(".//body"):
        name = body.get("name")
        geometries = []
        for geom in body.findall("geom"):
            if geom.get("mesh") is None or geom.get("contype") == "0":
                continue
            asset = assets[geom.get("mesh")]
            path = HAND / "meshes" / asset.get("file")
            mesh = trimesh.load_mesh(path)
            mesh.apply_scale(np.fromstring(asset.get("scale", "1 1 1"), sep=" "))
            transform = np.eye(4)
            q = np.fromstring(geom.get("quat", "1 0 0 0"), sep=" ")
            transform[:3, :3] = Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()
            transform[:3, 3] = np.fromstring(geom.get("pos", "0 0 0"), sep=" ")
            mesh.apply_transform(transform)
            geometries.append(mesh)
            # DP bodies also have a hard shell: sample only their elastomer geom.
            if not name.endswith("_DP") or "elastomer" in geom.get("mesh"):
                meshes[name], paths[name] = mesh, path
        if geometries:
            display[name] = trimesh.util.concatenate(geometries)
    return meshes, paths, display


def strips():
    """Link, region, medium shape, normal, length axis, axial interval (metres)."""
    yield "right_hand_C_MC", "palm", (16, 18), (1, 0, 0), (0, 0, 1), (.032, .081)
    shapes = {
        "thumb": ((3, 4), (3, 5), (2, 7)),
        "index": ((2, 4), (3, 5), (4, 5)),
        "middle": ((4, 4), (3, 6), (4, 6)),
        "ring": ((2, 5), (3, 6), (4, 6)),
        "pinky": ((3, 4), (3, 4), (3, 4)),
    }
    for region, (tip, middle, proximal) in shapes.items():
        prefix = f"right_{region}"
        yield prefix + "_DP", region, tip, (0, 1, 0), (1, 0, 0), ((.012, .030) if region == "thumb" else (.0085, .0265))
        if region == "thumb":
            yield prefix + "_PP", region, middle, (0, 0, 1), (1, 0, 0), (.011, .031)
            yield prefix + "_MC", region, proximal, (0, np.sqrt(3) / 2, .5), (1, 0, 0), (.024, .052)
        else:
            yield prefix + "_MP", region, middle, (0, 1, 0), (1, 0, 0), (.010, .026)
            yield prefix + "_PP", region, proximal, (0, 0, 1), (1, 0, 0), (.013, .037)
            if region == "pinky":
                yield prefix + "_MC", region, (2, 4), (-1, 0, 0), (0, 0, 1), (-.048, -.018)


def generate(profile):
    meshes, paths, _ = load_geometry()
    records, layouts = [], []
    counts = dict.fromkeys(COUNTS, 0)
    for link, region, shape, normal_axis, length_axis, (lo, hi) in strips():
        if profile == "actual" and not link.endswith("_DP"):
            continue
        basis = np.column_stack((np.cross(normal_axis, length_axis), normal_axis, length_axis))
        transform = np.eye(4)
        transform[:3, :3] = basis.T
        canonical = meshes[link].copy().apply_transform(transform)
        if profile == "actual":
            # Optical maps establish the pad footprint; the specification's 1 mm
            # spatial resolution sets simulation density, not its 240x240 image.
            shape = (int(np.ceil((hi - lo) / .001 - 1e-9)),
                     int(np.ceil(.90 * canonical.extents[0] / .001)))
        ny, nx = shape
        grid = []
        for row in range(ny):
            grid_row = []
            for col in range(nx):
                pos, normal = surface_probe(canonical, lo + (hi - lo) * (row + .5) / ny, (col + .5) / nx)
                grid_row.append({"pos": (basis @ pos).round(8).tolist(),
                                 "normal": (basis @ normal).round(8).tolist()})
            grid.append(grid_row)
        points = np.array([p["pos"] for row in grid for p in row])
        radius = .0005 if profile == "actual" else round(float(np.median(cKDTree(points).query(points, k=2)[0][:, 1]) / 2), 8)
        for row in grid:
            for probe in row:
                probe["radius"] = radius
        records.append({"link_name": link, "kind": "grid", "probes": grid})
        counts[region] += ny * nx
        layouts.append({"link_name": link, "region": region, "shape": list(shape), "count": ny * nx,
                        "radius_m": radius, "normal_axis": list(normal_axis), "length_axis": list(length_axis),
                        "axial_bounds_m": [lo, hi], "mesh": str(paths[link].relative_to(HAND))})
    reference = OUTPUT / "actual/reference"
    manifest = {
        "profile": profile, "active_count": sum(counts.values()), "region_counts": counts,
        "source_urls": ["https://www.sharpa.com/pages/wave", SOURCE],
        "geometry": "Sharpa Wave right MJCF HB1 contact meshes, in link-local metres",
        "layout_note": ("Fingertip-only, nominal 1 mm projected spacing; a simulation discretization of spatial resolution, "
                        "not the 240x240 optical image or a hardware taxel count. Edge margins omit the pad silhouette."
                        if profile == "actual" else
                        "Synthetic whole-hand coverage matching Wuji actual regional counts; Sharpa hardware senses fingertips only."),
        "radius_note": ("0.5 mm simulation footprint" if profile == "actual" else
                        "Half the median nearest-neighbor spacing per link; simulation footprint, not calibration"),
        "fit_note": "Dense curved grids, 5% width margin each side. Vendor wave_01 maps guide coverage; probes are projected onto the current HB1 meshes. No hardware row/column correspondence.",
        "links": layouts,
        "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(reference.glob("*.npy"))},
        "mesh_sha256": {r["link_name"]: hashlib.sha256(paths[r["link_name"]].read_bytes()).hexdigest() for r in records},
        "mjcf_sha256": hashlib.sha256(MODEL.read_bytes()).hexdigest(),
    }
    return records, manifest


def validate(records, manifest):
    meshes, _, _ = load_geometry()
    counts, worst = dict.fromkeys(COUNTS, 0), 0.0
    assert len(records) == len({r["link_name"] for r in records}) == (5 if manifest["profile"] == "actual" else 17)
    for record, mapping in zip(records, manifest["links"], strict=True):
        assert record["link_name"] == mapping["link_name"] and record["kind"] == "grid"
        ny, nx = mapping["shape"]
        assert min(ny, nx) >= 2 and len(record["probes"]) == ny
        assert all(len(row) == nx for row in record["probes"])
        probes = [p for row in record["probes"] for p in row]
        positions = np.array([p["pos"] for p in probes])
        normals = np.array([p["normal"] for p in probes])
        assert np.isfinite(positions).all() and np.isfinite(normals).all()
        assert all(p["radius"] == mapping["radius_m"] > 0 for p in probes)
        np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1, atol=1e-7)
        mesh = meshes[record["link_name"]]
        _, distances, triangles = query_mesh(mesh).nearest.on_surface(positions * 1000)
        error = float(distances.max()) / 1000
        assert error < 1e-7, (record["link_name"], error)
        assert (np.einsum("ij,ij->i", normals, mesh.face_normals[triangles]) > .99).all()
        assert (normals @ mapping["normal_axis"] > .05).all()
        assert cKDTree(positions).query(positions, k=2)[0][:, 1].min() > 1e-5
        counts[mapping["region"]] += len(probes)
        worst = max(worst, error)
    assert counts == manifest["region_counts"] and sum(counts.values()) == manifest["active_count"]
    if manifest["profile"] == "med":
        assert counts == COUNTS
    else:
        assert counts == {"palm": 0, "thumb": 270, "index": 234, "middle": 234, "ring": 234, "pinky": 234}
    return {"profile": manifest["profile"], "active_count": sum(counts.values()), "links": len(records), "max_surface_error_m": worst}


def preview(records, manifest, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import mujoco

    _, _, meshes = load_geometry()
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    fig = plt.figure(figsize=(15, 7))
    for subplot, (flexed, elev, azim, title) in enumerate([
        (False, 0, 0, "Palm view"), (False, 20, 35, "Oblique view"), (True, 20, 35, "Flexed links")
    ], 1):
        data = mujoco.MjData(model)
        if flexed:
            for joint in range(model.njnt):
                name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
                data.qpos[model.jnt_qposadr[joint]] = 0 if name.endswith("_AA") else (.15 if name == "right_pinky_CMC" else .45)
        mujoco.mj_forward(model, data)
        transforms = {}
        for link in meshes:
            body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, link)
            matrix = np.eye(4)
            matrix[:3, :3], matrix[:3, 3] = data.xmat[body].reshape(3, 3), data.xpos[body]
            transforms[link] = matrix
        ax = fig.add_subplot(1, 3, subplot, projection="3d")
        all_vertices = []
        for link, mesh in meshes.items():
            vertices = trimesh.transform_points(mesh.vertices, transforms[link]) * 1000
            all_vertices.append(vertices)
            ax.add_collection3d(Poly3DCollection(vertices[mesh.faces], facecolor="#b9c3cd", linewidth=0, alpha=.12, rasterized=True))
        for record, mapping in zip(records, manifest["links"], strict=True):
            points = np.array([p["pos"] for row in record["probes"] for p in row])
            world = trimesh.transform_points(points, transforms[record["link_name"]]) * 1000
            ax.scatter(*world.T, s=4 if manifest["profile"] == "actual" else 10, color=COLORS[mapping["region"]], depthshade=False)
        vertices = np.concatenate(all_vertices)
        lo, hi = vertices.min(0), vertices.max(0)
        for setter, a, b in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), lo, hi):
            setter(a - 5, b + 5)
        ax.set_box_aspect(hi - lo + 10)
        ax.view_init(elev=elev, azim=azim)
        ax.set_axis_off()
        ax.set_title(title)
    fig.suptitle(f"Sharpa {manifest['profile']} — {manifest['active_count']} probes / {len(records)} link grids")
    fig.text(.5, .04, "Nominal 1 mm fingertip sampling" if manifest["profile"] == "actual" else "Synthetic whole-hand coverage; Wuji-matched regional counts", ha="center")
    fig.savefig(destination, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify saved layouts without writing")
    args = parser.parse_args()
    for profile, filename in FILES.items():
        records, manifest = generate(profile)
        folder = OUTPUT / profile
        if args.check:
            assert json.loads((folder / filename).read_text()) == records
            assert json.loads((folder / "layout_metadata.json").read_text()) == manifest
        else:
            folder.mkdir(parents=True, exist_ok=True)
            (folder / filename).write_text(json.dumps(records, indent=2) + "\n")
            (folder / "layout_metadata.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(json.dumps(validate(records, manifest)))
        if not args.check:
            preview(records, manifest, folder / "placement_preview.png")


if __name__ == "__main__":
    main()
