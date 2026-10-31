"""Reproduce the XELA V4 MJCF and 368-probe actual layout from pinned sources."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import operator
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
HAND = ROOT / "src/assets/hands/allegro/xela_v4"
REF = HAND / "reference"
OUTPUT = ROOT / "src/assets/sensors/allegro/actual"
MODEL = HAND / "right_hand.xml"
PROBES = OUTPUT / "probes_368_hand_allegro.json"
FINGERS = ("index", "middle", "ring", "thumb")
NS = "{http://www.ros.org/wiki/xacro}"
# Upstream materials are named for colors, so the sensor skin renders red, blue
# and orange. Structure goes mid-gray, sensor skin near-black.
STRUCTURE = ("black", "grey", "white")


def gray(name, rgba):
    level = "0.45" if name in STRUCTURE else "0.1"
    return f"{level} {level} {level} {rgba.split()[3]}"


def link_name(name):
    if name == "palm_link":
        return "base_link"
    if name[0].isdigit() and name.endswith("aftc_palm_link"):
        return FINGERS[int(name[0])] + "_3_tip"
    if name.startswith("link_"):
        number = name[5:].split("_")[0].rstrip("AB")
        finger, segment = divmod(int(number), 4)
        return FINGERS[finger] + "_" + str(segment) + name[5 + len(number) :]
    return name


def joint_name(name):
    finger, segment = divmod(int(name.removeprefix("joint_")), 4)
    suffix = (
        ("bend0", "roll", "bend1", "bend2")
        if finger == 3
        else ("roll", "bend0", "bend1", "bend2")
    )
    return FINGERS[finger] + "_" + suffix[segment]


def origin(element):
    if element is None:
        return {"pos": "0 0 0", "quat": "1 0 0 0"}
    q = Rotation.from_euler(
        "xyz", np.fromstring(element.get("rpy", "0 0 0"), sep=" ")
    ).as_quat()
    return {
        "pos": element.get("xyz", "0 0 0"),
        "quat": " ".join(f"{v:.16g}" for v in q[[3, 0, 1, 2]]),
    }


def source_tree():
    return ET.parse(REF / "ahrcpcpn.urdf")


def convert_model():
    source = source_tree()
    root = ET.Element("mujoco", model="allegro_xela_v4_right")
    ET.SubElement(
        root,
        "compiler",
        angle="radian",
        meshdir="mesh",
        autolimits="true",
        fusestatic="false",
        inertiafromgeom="false",
    )
    defaults = ET.SubElement(root, "default")
    ET.SubElement(defaults, "joint", damping="0.1", armature="0")
    ET.SubElement(defaults, "position", kp="1", kv="0")
    asset = ET.SubElement(root, "asset")
    mesh_ids = {}
    for mesh in source.findall(".//mesh"):
        path = mesh.get("filename").removeprefix("package://xela_models/mesh/")
        key = (path, mesh.get("scale", "1 1 1"))
        if key not in mesh_ids:
            mesh_ids[key] = f"mesh_{len(mesh_ids)}"
            ET.SubElement(asset, "mesh", name=mesh_ids[key], file=path, scale=key[1])
    colors = {
        m.get("name"): gray(m.get("name"), m.find("color").get("rgba"))
        for m in source.findall("./material")
        if m.find("color") is not None
    }
    links = {link.get("name"): link for link in source.findall("link")}
    children = {}
    for joint in source.findall("joint"):
        children.setdefault(joint.find("parent").get("link"), []).append(joint)
    world = ET.SubElement(root, "worldbody")
    actuators = ET.SubElement(root, "actuator")
    contact = ET.SubElement(root, "contact")

    def walk(name, parent, joint=None):
        body = ET.SubElement(
            parent,
            "body",
            name=link_name(name),
            **origin(joint.find("origin") if joint is not None else None),
        )
        if joint is not None and joint.get("type") != "fixed":
            limits = joint.find("limit")
            jname = joint_name(joint.get("name"))
            bounds = limits.get("lower") + " " + limits.get("upper")
            ET.SubElement(
                body,
                "joint",
                name=jname,
                type="hinge",
                axis=joint.find("axis").get("xyz"),
                range=bounds,
            )
            effort = float(limits.get("effort"))
            ET.SubElement(
                actuators,
                "position",
                name=jname + "_actuator",
                joint=jname,
                ctrlrange=bounds,
                forcerange=f"{-effort:g} {effort:g}",
            )
        link = links[name]
        inertial = link.find("inertial")
        if inertial is not None:
            tensor = inertial.find("inertia")
            # URDF tensors are expressed in the inertial frame. Rotate into body
            # coordinates; MuJoCo fullinertia cannot be combined with a quat.
            matrix = np.array(
                [
                    [
                        float(tensor.get("ixx")),
                        float(tensor.get("ixy")),
                        float(tensor.get("ixz")),
                    ],
                    [
                        float(tensor.get("ixy")),
                        float(tensor.get("iyy")),
                        float(tensor.get("iyz")),
                    ],
                    [
                        float(tensor.get("ixz")),
                        float(tensor.get("iyz")),
                        float(tensor.get("izz")),
                    ],
                ]
            )
            pose = origin(inertial.find("origin"))
            q = np.fromstring(pose["quat"], sep=" ")
            rot = Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()
            matrix = rot @ matrix @ rot.T
            values = (
                matrix[0, 0],
                matrix[1, 1],
                matrix[2, 2],
                matrix[0, 1],
                matrix[0, 2],
                matrix[1, 2],
            )
            ET.SubElement(
                body,
                "inertial",
                pos=pose["pos"],
                mass=inertial.find("mass").get("value"),
                fullinertia=" ".join(f"{v:.16g}" for v in values),
            )
        for kind in ("visual", "collision"):
            for i, geom in enumerate(link.findall(kind)):
                kwargs = dict(
                    origin(geom.find("origin")),
                    name=f"{link_name(name)}_{kind}_{i}",
                    mass="0",
                )
                kwargs.update(
                    contype="0" if kind == "visual" else "1",
                    conaffinity="0" if kind == "visual" else "1",
                    group="2" if kind == "visual" else "3",
                )
                material = geom.find("material")
                kwargs["rgba"] = (
                    colors.get(material.get("name"), ".6 .6 .6 1")
                    if material is not None
                    else ".6 .6 .6 1"
                )
                geometry = geom.find("geometry")[0]
                if geometry.tag == "mesh":
                    path = geometry.get("filename").removeprefix(
                        "package://xela_models/mesh/"
                    )
                    kwargs.update(
                        type="mesh",
                        mesh=mesh_ids[(path, geometry.get("scale", "1 1 1"))],
                    )
                elif geometry.tag == "box":
                    kwargs.update(
                        type="box",
                        size=" ".join(
                            str(float(v) / 2) for v in geometry.get("size").split()
                        ),
                    )
                elif geometry.tag == "sphere":
                    kwargs.update(type="sphere", size=geometry.get("radius"))
                elif geometry.tag == "cylinder":
                    kwargs.update(
                        type="cylinder",
                        size=f"{geometry.get('radius')} {float(geometry.get('length')) / 2}",
                    )
                else:
                    raise ValueError(geometry.tag)
                ET.SubElement(body, "geom", **kwargs)
        for child in children.get(name, []):
            walk(child.find("child").get("link"), body, child)
        return body

    # Discard only the empty ROS world link; keep the URDF's unrotated palm frame.
    for joint in children["world"]:
        walk(joint.find("child").get("link"), world, joint)
    # Match Menagerie's palm/base exclusions, in the preserved anatomical names.
    for name in ("index_0", "middle_0", "ring_0", "thumb_0", "thumb_1"):
        ET.SubElement(contact, "exclude", body1="base_link", body2=name)
    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode") + "\n"


def arithmetic(value):
    """Evaluate only numeric arithmetic in the vendor's taxel coordinates."""
    expression = ast.parse(value.removeprefix("${").removesuffix("}"), mode="eval").body
    operations = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
    }

    def visit(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in operations:
            return operations[type(node.op)](visit(node.left), visit(node.right))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -visit(node.operand)
        raise ValueError(f"Unsupported source expression: {value}")

    return float(visit(expression))


def template(kind):
    tree = ET.parse(REF / "xela.xacro")
    macro = next(
        m
        for m in tree.getroot()
        if m.get("name")
        == {"tip": "sensoraftc", "4x4": "sensor4x4", "4x6": "sensor4x6"}[kind]
    )
    taxels = list(macro.iter(NS + "taxel"))
    points = np.array(
        [[arithmetic(t.get(k, "0")) for k in ("x", "y", "z")] for t in taxels]
    )
    if kind != "tip":
        return points.reshape(4, 4 if kind == "4x4" else 6, 3)
    # Six native curved columns have 5,4,6,6,4,5 taxels. Resample each to
    # five points along its polyline, preserving the wrapped coverage and 30 count.
    columns, start = [], 0
    for count in (5, 4, 6, 6, 4, 5):
        col = points[start : start + count]
        distance = np.r_[0, np.cumsum(np.linalg.norm(np.diff(col, axis=0), axis=1))]
        columns.append(
            np.column_stack(
                [
                    np.interp(np.linspace(0, distance[-1], 5), distance, col[:, k])
                    for k in range(3)
                ]
            )
        )
        start += count
    return np.stack(columns, axis=1)


def sensor_specs():
    for link in source_tree().findall("link"):
        name = link.get("name")
        if "aftc_palm_link" in name:
            yield link_name(name), FINGERS[int(name[0])], "tip", "ft_c.stl"
        elif "_4x4_palm_link" in name:
            yield link_name(name), link_name(name).split("_")[0], "4x4", "4x4.stl"
        elif "_4x6_palm_link" in name:
            yield link_name(name), "palm", "4x6", "4x6.stl"


def generate():
    records, mapping = [], []
    for link, region, kind, filename in sensor_specs():
        mesh = trimesh.load_mesh(HAND / "mesh" / filename).apply_scale(1000)
        raw = template(kind)
        pos, distance, faces = mesh.nearest.on_surface(raw.reshape(-1, 3) * 1000)
        normals = mesh.face_normals[faces]
        # Avoid ambiguous normals at triangle edges after coordinate rounding.
        pos = 0.9999 * pos + 0.0001 * mesh.triangles_center[faces]
        distance = np.linalg.norm(pos - raw.reshape(-1, 3) * 1000, axis=1)
        pos /= 1000
        radius = round(float(np.median(cKDTree(pos).query(pos, k=2)[0][:, 1]) / 2), 8)
        rows, cols = raw.shape[:2]
        probes = [
            {
                "pos": p.round(8).tolist(),
                "normal": n.round(8).tolist(),
                "radius": radius,
            }
            for p, n in zip(pos, normals)
        ]
        records.append(
            {
                "link_name": link,
                "kind": "grid",
                "probes": [probes[i * cols : (i + 1) * cols] for i in range(rows)],
            }
        )
        mapping.append(
            {
                "link_name": link,
                "region": region,
                "sensor_type": kind,
                "shape": [rows, cols],
                "count": rows * cols,
                "radius_m": radius,
                "mesh": filename,
                "max_source_projection_m": float(distance.max()) / 1000,
            }
        )
    files = [
        REF / "ahrcpcpn.urdf",
        REF / "xela.xacro",
        REF / "menagerie_right_hand.xml",
        *sorted((HAND / "mesh").rglob("*")),
    ]
    manifest = {
        "active_count": 368,
        "region_counts": {
            "index": 78,
            "middle": 78,
            "ring": 78,
            "thumb": 62,
            "palm": 72,
        },
        "sources": json.loads((REF / "sources.json").read_text()),
        "links": mapping,
        "layout_note": "XELA hardware counts, dense per-pad grids. Curved tips resampled to 5x6; hardware indexing is not preserved. Link-local metres.",
        "radius_note": "Half the median nearest-neighbor spacing; simulation footprint, not calibrated receptive field.",
        "source_sha256": {
            str(p.relative_to(HAND)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in files
            if p.is_file()
        },
    }
    return records, manifest


def validate(records, manifest):
    counts = dict.fromkeys(manifest["region_counts"], 0)
    assert len(records) == len({r["link_name"] for r in records}) == 18
    worst = 0
    for record, meta in zip(records, manifest["links"], strict=True):
        assert record["kind"] == "grid" and record["link_name"] == meta["link_name"]
        assert len(record["probes"]) == meta["shape"][0] and all(
            len(row) == meta["shape"][1] for row in record["probes"]
        )
        probes = [p for row in record["probes"] for p in row]
        pos = np.array([p["pos"] for p in probes])
        normals = np.array([p["normal"] for p in probes])
        assert np.isfinite(pos).all() and np.isfinite(normals).all()
        np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1, atol=1e-7)
        assert all(p["radius"] == meta["radius_m"] > 0 for p in probes)
        mesh = trimesh.load_mesh(HAND / "mesh" / meta["mesh"]).apply_scale(1000)
        _, distance, faces = mesh.nearest.on_surface(pos * 1000)
        assert distance.max() < 1e-4
        assert (np.einsum("ij,ij->i", normals, mesh.face_normals[faces]) > 0.99).all()
        assert cKDTree(pos).query(pos, k=2)[0][:, 1].min() > 1e-5
        counts[meta["region"]] += len(probes)
        worst = max(worst, float(distance.max()) / 1000)
    assert counts == manifest["region_counts"] and sum(counts.values()) == 368
    return {"active_count": 368, "links": 18, "max_surface_error_m": worst}


def preview(records, destination):
    import mujoco
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from generate_wuji_tactile import COLORS

    model = mujoco.MjModel.from_xml_path(str(MODEL))
    meshes = {}
    for link in source_tree().findall("link"):
        visual = link.find("visual")
        if visual is None or visual.find("geometry/mesh") is None:
            continue
        path = (
            visual.find("geometry/mesh")
            .get("filename")
            .removeprefix("package://xela_models/")
        )
        mesh = trimesh.load_mesh(HAND / path)
        pose = origin(visual.find("origin"))
        q = np.fromstring(pose["quat"], sep=" ")
        transform = np.eye(4)
        transform[:3, :3] = Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()
        transform[:3, 3] = np.fromstring(pose["pos"], sep=" ")
        meshes[link_name(link.get("name"))] = mesh.apply_transform(transform)
    fig = plt.figure(figsize=(15, 7))
    for i, (flexed, elev, azim) in enumerate(
        [(False, 0, 0), (False, 20, 35), (True, 20, 35)], 1
    ):
        data = mujoco.MjData(model)
        for j in range(model.njnt):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
            angle = 0.45 if flexed and "bend" in name else 0
            data.qpos[model.jnt_qposadr[j]] = np.clip(angle, *model.jnt_range[j])
        mujoco.mj_forward(model, data)
        transforms = {}
        for name in meshes:
            b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            matrix = np.eye(4)
            matrix[:3, :3] = data.xmat[b].reshape(3, 3)
            matrix[:3, 3] = data.xpos[b]
            transforms[name] = matrix
        ax = fig.add_subplot(1, 3, i, projection="3d")
        all_points = []
        for name, mesh in meshes.items():
            points = trimesh.transform_points(mesh.vertices, transforms[name]) * 1000
            all_points.append(points)
            ax.add_collection3d(
                Poly3DCollection(
                    points[mesh.faces],
                    facecolor="#b9c3cd",
                    linewidth=0,
                    alpha=0.10,
                    rasterized=True,
                )
            )
        for record, (_, region, _, _) in zip(records, sensor_specs(), strict=True):
            points = np.array([p["pos"] for row in record["probes"] for p in row])
            world = (
                trimesh.transform_points(points, transforms[record["link_name"]]) * 1000
            )
            ax.scatter(*world.T, s=9, color=COLORS[region], depthshade=False)
        points = np.concatenate(all_points)
        lo, hi = points.min(0), points.max(0)
        for setter, a, b in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), lo, hi):
            setter(a - 5, b + 5)
        ax.set_box_aspect(hi - lo + 10)
        ax.view_init(elev=elev, azim=azim)
        ax.set_axis_off()
        ax.set_title(("Palm view", "Oblique view", "Flexed links")[i - 1])
    fig.suptitle("Allegro XELA V4 actual — 368 probes / 18 pad grids")
    fig.savefig(destination, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    xml = convert_model()
    records, manifest = generate()
    if args.check:
        assert MODEL.read_text() == xml
        assert json.loads(PROBES.read_text()) == records
        assert json.loads((OUTPUT / "layout_metadata.json").read_text()) == manifest
    else:
        OUTPUT.mkdir(parents=True, exist_ok=True)
        MODEL.write_text(xml)
        PROBES.write_text(json.dumps(records, indent=2) + "\n")
        (OUTPUT / "layout_metadata.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )
    print(validate(records, manifest))
    if not args.check:
        preview(records, OUTPUT / "placement_preview.png")


if __name__ == "__main__":
    main()
