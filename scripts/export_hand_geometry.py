#!/usr/bin/env python
"""Bake each hand and its probe layouts into one JSON the project page renders in 3D.

MuJoCo loads the hand's MJCF and evaluates forward kinematics at the rest pose, which puts every
link's meshes and its probes in one frame. The meshes are merged and decimated; the probes keep
their position, normal and radius. Run with the project's interpreter, which has mujoco and
trimesh:

    uv run --project dexterous-hands python scripts/export_hand_geometry.py
"""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[1]
HANDS = ROOT / "dexterous-hands/src/assets/hands"
SENSORS = ROOT / "dexterous-hands/src/assets/sensors"
OUT = ROOT / "docs/static/data/hands"

MODELS = {
    "wuji": "wuji/mjcf/right.xml",
    "allegro": "allegro/xela_v4/right_hand.xml",
    "sharpa": "sharpa/right_sharpa_wave.xml",
}
# Visual geoms are the non-colliding mesh geoms. The group number they sit in differs per
# model (Wuji 1, Allegro 2, Sharpa 1), so select on collision flags rather than group.


def rest_pose(path: Path):
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def hand_mesh(model, data, max_faces: int):
    """Every visual mesh geom, placed at the rest pose and merged into one decimated mesh."""
    parts = []
    for geom in range(model.ngeom):
        if model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        if model.geom_contype[geom] or model.geom_conaffinity[geom]:
            continue
        mesh_id = model.geom_dataid[geom]
        vstart, vnum = model.mesh_vertadr[mesh_id], model.mesh_vertnum[mesh_id]
        fstart, fnum = model.mesh_faceadr[mesh_id], model.mesh_facenum[mesh_id]
        verts = model.mesh_vert[vstart : vstart + vnum].reshape(-1, 3).astype(np.float64)
        faces = model.mesh_face[fstart : fstart + fnum].reshape(-1, 3)
        rot = data.geom_xmat[geom].reshape(3, 3)
        parts.append(trimesh.Trimesh(verts @ rot.T + data.geom_xpos[geom], faces, process=False))
    if not parts:
        return None
    merged = trimesh.util.concatenate(parts)
    if max_faces and len(merged.faces) > max_faces:
        merged = merged.simplify_quadric_decimation(face_count=max_faces)
    merged.merge_vertices()
    return merged


def probes(model, data, layout_path: Path):
    """Probe positions and normals, moved from their link frame into the rest-pose frame."""
    groups = json.loads(layout_path.read_text())
    out = []
    for group in groups:
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, group["link_name"])
        if body < 0:
            continue
        rot, origin = data.xmat[body].reshape(3, 3), data.xpos[body]
        rows = group["probes"] if isinstance(group["probes"][0], dict) else [p for r in group["probes"] for p in r]
        for probe in rows:
            if probe.get("radius", 0) <= 0:
                continue  # zero-radius probes only pad the layout for batching
            out.append(
                {
                    "p": np.asarray(probe["pos"]) @ rot.T + origin,
                    "n": np.asarray(probe["normal"]) @ rot.T,
                    "r": probe["radius"],
                    "link": group["link_name"],
                }
            )
    return out


def region_of(link: str, regions: dict) -> str:
    name = link.lower()
    for region in regions:
        if region.lower() in name:
            return region
    return "palm" if ("wrist" in name or "palm" in name or "base" in name) else "other"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--max-faces", type=int, default=24000)
    parser.add_argument("--layouts", nargs="*", default=["low", "actual", "high"])
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    index = []
    for hand, model_rel in MODELS.items():
        model, data = rest_pose(HANDS / model_rel)
        mesh = hand_mesh(model, data, args.max_faces)
        entry = {
            "id": hand,
            "mesh": {
                "vertices": [round(float(v), 5) for v in mesh.vertices.reshape(-1)],
                "indices": [int(i) for i in mesh.faces.reshape(-1)],
            },
            "layouts": {},
        }
        for name in args.layouts:
            folder = SENSORS / hand / name
            files = sorted(folder.glob("probes*.json"))
            if not files:
                continue
            meta = json.loads((folder / "layout_metadata.json").read_text())
            regions = meta.get("region_counts", {})
            # Some layout folders keep a superseded probe file beside the current one; pick the
            # file whose active probe count is the one layout_metadata.json describes.
            wanted = meta.get("active_count")
            found = None
            for candidate in files:
                counted = probes(model, data, candidate)
                if found is None or len(counted) == wanted:
                    found = counted
                if len(counted) == wanted:
                    break
            if wanted is not None and len(found) != wanted:
                print(f"  WARNING {hand}/{name}: {len(found)} probes, metadata says {wanted}")
            entry["layouts"][name] = {
                "count": len(found),
                "positions": [round(float(v), 5) for probe in found for v in probe["p"]],
                "normals": [round(float(v), 4) for probe in found for v in probe["n"]],
                "radii": [round(float(probe["r"]), 5) for probe in found],
                "regions": [region_of(probe["link"], regions) for probe in found],
            }
        path = args.out / f"{hand}.json"
        path.write_text(json.dumps(entry, separators=(",", ":")))
        index.append({"id": hand, "file": f"{hand}.json"})
        sizes = ", ".join(f"{k} {v['count']}" for k, v in entry["layouts"].items())
        print(f"{hand}: {len(mesh.faces):,} faces, {sizes} -> {path.stat().st_size / 1e6:.1f} MB")
    (args.out / "index.json").write_text(json.dumps(index, separators=(",", ":")))


if __name__ == "__main__":
    main()
