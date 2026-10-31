"""Geometry and runtime contracts for the fitted Wuji glove-resolution layout."""

import json
import sys
from pathlib import Path

import eden  # noqa: F401 -- initialize Eden before importing Genesis/project robots
import numpy as np
import pytest

from entities.robots.wuji_hand import WujiHand
from task_mods import TactileSensorsMod

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from generate_wuji_tactile import COUNTS, OUTPUT, generate, validate  # noqa: E402


def active_count(probe):
    shape = np.asarray(probe.probe_local_pos).shape[:-1]
    return int(np.broadcast_to(np.asarray(probe.probe_radius), shape).astype(bool).sum())


def test_generated_asset_is_reproducible_and_on_mesh():
    records = json.loads((OUTPUT / "probes_526_hand_wuji.json").read_text())
    mapping = json.loads((OUTPUT / "layout_metadata.json").read_text())
    expected_records, expected_mapping = generate()
    assert records == expected_records
    assert mapping == expected_mapping
    report = validate(records, mapping)
    assert report["region_counts"] == COUNTS
    assert report["links"] == 16


@pytest.mark.parametrize("sensor_type", ["bool", "depth", "force", "proximity", "elastomer"])
@pytest.mark.parametrize("subset,count,links", [
    ("hand", 526, 16), ("palm", 288, 1), ("tips", 58, 5),
    ("midfinger", 180, 10), ("fingers", 238, 15),
])
def test_runtime_sensor_options_preserve_grids_and_active_count(sensor_type, subset, count, links):
    options = TactileSensorsMod()._get_sensors_dict(WujiHand(), f"actual-{subset}", sensor_type)
    assert len(options) == links
    assert sum(active_count(value.sensor) for value in options.values()) == count
    for value in options.values():
        shape = np.asarray(value.sensor.probe_local_pos).shape
        assert len(shape) == 3 and shape[-1] == 3
        assert min(shape[:2]) >= 2


def test_every_link_has_a_usable_accelerated_grid(monkeypatch):
    import genesis as gs
    # This pure NumPy helper normally receives its dtype from gs.init().
    monkeypatch.setattr(gs, "np_float", np.float32, raising=False)
    monkeypatch.setattr(gs, "EPS", float(np.finfo(np.float32).eps))
    from genesis.engine.sensors.tactile_shared import normalize_grid_probe_layout

    path = WujiHand.metadata.tactile_probe_cfgs["actual"]
    for link, probe in TactileSensorsMod._parse_probe_config(path).items():
        layout = normalize_grid_probe_layout(probe["local_pos"], probe["local_normal"], is_grid=True)
        assert layout[2], f"{link} would fall back from the grid FFT path"
        assert np.isfinite(layout[-1]).all() and (layout[-1] > 0).all()
        # Row and column tangents must span a surface, even on curved grids.
        assert abs(np.dot(layout[5], layout[6])) < 0.95


def test_generator_body_frames_match_mujoco_forward_kinematics():
    import mujoco
    from generate_wuji_tactile import HAND, body_transforms, load_geometry

    model = mujoco.MjModel.from_xml_path(str(HAND / "mjcf/right.xml"))
    data = mujoco.MjData(model)
    for joint in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
        data.qpos[model.jnt_qposadr[joint]] = 0.0 if name.endswith("abd") else 0.45
    mujoco.mj_forward(model, data)
    tree, _, _ = load_geometry()
    for name, transform in body_transforms(tree, flexed=True).items():
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        np.testing.assert_allclose(transform[:3, 3], data.xpos[body], atol=1e-10)
        np.testing.assert_allclose(transform[:3, :3], data.xmat[body].reshape(3, 3), atol=1e-10)
