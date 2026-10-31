"""Source fidelity, control and per-pad grid contracts for Allegro XELA V4."""

import json
import sys
from pathlib import Path

import eden  # noqa: F401
import numpy as np
import pytest
import mujoco
from scipy.spatial.transform import Rotation

from entities.robots.allegro_hand import AllegroHand
from task_mods import TactileSensorsMod

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from generate_allegro_tactile import (
    MODEL,
    OUTPUT,
    PROBES,
    REF,
    convert_model,
    generate,
    validate,
    source_tree,
    link_name,
    joint_name,
    origin,
)


def test_reproducible_assets_and_surface_geometry():
    assert MODEL.read_text() == convert_model()
    records, manifest = generate()
    assert json.loads(PROBES.read_text()) == records
    assert json.loads((OUTPUT / "layout_metadata.json").read_text()) == manifest
    assert validate(records, manifest)["active_count"] == 368


@pytest.mark.parametrize("kind", ["bool", "depth", "force", "proximity", "elastomer"])
@pytest.mark.parametrize(
    "subset,count,links",
    [
        ("hand", 368, 18),
        ("tips", 120, 4),
        ("midfinger", 176, 11),
        ("fingers", 296, 15),
        ("palm", 72, 3),
    ],
)
def test_sensor_options(kind, subset, count, links):
    options = TactileSensorsMod()._get_sensors_dict(
        AllegroHand(), f"actual-{subset}", kind
    )
    assert len(options) == links
    total = 0
    for option in options.values():
        shape = np.asarray(option.sensor.probe_local_pos).shape
        assert len(shape) == 3 and min(shape[:2]) >= 2 and shape[-1] == 3
        assert np.all(np.asarray(option.sensor.probe_radius) > 0)
        total += shape[0] * shape[1]
    assert total == count


def test_all_grids_support_acceleration(monkeypatch):
    import genesis as gs

    monkeypatch.setattr(gs, "np_float", np.float32, raising=False)
    monkeypatch.setattr(gs, "EPS", float(np.finfo(np.float32).eps))
    from genesis.engine.sensors.tactile_shared import normalize_grid_probe_layout

    for name, p in TactileSensorsMod._parse_probe_config(str(PROBES)).items():
        layout = normalize_grid_probe_layout(
            p["local_pos"], p["local_normal"], is_grid=True
        )
        assert layout[2], name
        assert np.isfinite(layout[-1]).all() and (layout[-1] > 0).all()
        assert abs(np.dot(layout[5], layout[6])) < 0.95


def test_controls_and_source_kinematics():
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    assert model.nv == model.nu == 16
    assert set(AllegroHand().dofs_name) == {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(16)
    }
    np.testing.assert_allclose(model.actuator_gainprm[:, 0], 1)
    np.testing.assert_allclose(model.actuator_biasprm[:, 1], -1)
    np.testing.assert_allclose(model.actuator_biasprm[:, 2], 0)
    np.testing.assert_allclose(model.dof_damping, 0.1)
    np.testing.assert_allclose(model.dof_armature, 0)
    np.testing.assert_allclose(model.actuator_forcerange, np.tile([-15, 15], (16, 1)))
    # Confirm the pinned Menagerie source still specifies the values transferred.
    import xml.etree.ElementTree as ET

    upstream = ET.parse(REF / "menagerie_right_hand.xml")
    default = upstream.find("./default/default[@class='allegro_right']")
    assert float(default.find("position").get("kp")) == 1
    assert float(default.find("joint").get("damping")) == 0.1
    data = mujoco.MjData(model)
    data.qpos[:] = np.clip(
        np.linspace(0.1, 0.7, 16), model.jnt_range[:, 0], model.jnt_range[:, 1]
    )
    mujoco.mj_forward(model, data)
    transforms = {"world": np.eye(4)}
    pending = source_tree().findall("joint")
    ordered = []
    known = {"world"}
    while pending:
        ready = [j for j in pending if j.find("parent").get("link") in known]
        assert ready, "Disconnected source kinematics"
        for joint in ready:
            pending.remove(joint)
            known.add(joint.find("child").get("link"))
            ordered.append(joint)
    for joint in ordered:
        pose = origin(joint.find("origin"))
        q = np.fromstring(pose["quat"], sep=" ")
        local = np.eye(4)
        local[:3, :3] = Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()
        local[:3, 3] = np.fromstring(pose["pos"], sep=" ")
        if joint.get("type") != "fixed":
            j = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, joint_name(joint.get("name"))
            )
            axis = np.fromstring(joint.find("axis").get("xyz"), sep=" ")
            local[:3, :3] = (
                local[:3, :3]
                @ Rotation.from_rotvec(
                    axis * data.qpos[model.jnt_qposadr[j]]
                ).as_matrix()
            )
        child = joint.find("child").get("link")
        transforms[child] = transforms[joint.find("parent").get("link")] @ local
        b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, link_name(child))
        assert b >= 0
        np.testing.assert_allclose(data.xpos[b], transforms[child][:3, 3], atol=1e-10)
        np.testing.assert_allclose(
            data.xmat[b].reshape(3, 3), transforms[child][:3, :3], atol=1e-10
        )
    names = {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b)
        for b in range(model.nbody)
    }
    assert set(AllegroHand().links_to_keep) <= names
    assert np.all(model.body_inertia[model.jnt_bodyid] > 0)


def test_rotation_task_has_no_allegro_control_override():
    from registry import get_task_config

    config = get_task_config(
        task_name="in_fingers_rotate",
        modifiers={
            "robot": "allegro",
            "obj": "primitives",
            "sensors": "actual-hand/bool",
            "stage": 1,
        },
    )
    robot = config.scene_options.robot
    assert robot.file.endswith("xela_v4/right_hand.xml")
    for field in (
        "default_dofs_kp",
        "default_dofs_kd",
        "default_dofs_armature",
        "default_dofs_force_limits",
    ):
        assert not getattr(robot, field)


def test_privileged_tip_positions_are_on_new_sensor_surface():
    import trimesh
    from generate_allegro_tactile import HAND

    mesh = trimesh.load_mesh(HAND / "mesh/ft_c.stl").apply_scale(1000)
    points = np.array(
        list(AllegroHand.metadata.priv_sensor_cfgs["fingertips"].values())
    )
    _, distances, _ = mesh.nearest.on_surface(points * 1000)
    assert distances.max() < 1e-4
