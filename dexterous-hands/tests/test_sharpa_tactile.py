"""Sharpa spatial-resolution and Wuji-comparable tactile layout contracts."""
import json
import sys
from pathlib import Path

import eden  # noqa: F401 -- initialize Eden before Genesis/project classes
import numpy as np
import pytest

from entities.robots.sharpa_hand import SharpaHand
from task_mods import TactileSensorsMod

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from generate_sharpa_tactile import FILES, OUTPUT, generate, validate  # noqa: E402


@pytest.mark.parametrize("profile", ["actual", "med"])
def test_reproducible_dense_surface_grids(profile):
    records = json.loads((OUTPUT / profile / FILES[profile]).read_text())
    manifest = json.loads((OUTPUT / profile / "layout_metadata.json").read_text())
    assert (records, manifest) == generate(profile)
    report = validate(records, manifest)
    assert report["active_count"] == (1206 if profile == "actual" else 526)
    if profile == "actual":
        assert {r["link_name"] for r in records} == set(SharpaHand.metadata.fingertip_links)
    else:
        assert {r["link_name"] for r in records} == set(SharpaHand.metadata.finger_links) | {SharpaHand.metadata.palm_link}


@pytest.mark.parametrize("sensor_type", ["bool", "depth", "force", "proximity", "elastomer"])
@pytest.mark.parametrize("selection,count,links", [
    ("actual-hand", 1206, 5), ("actual-tips", 1206, 5), ("actual-fingers", 1206, 5),
    ("med-hand", 526, 17), ("med-palm", 288, 1), ("med-tips", 58, 5),
    ("med-midfinger", 180, 11), ("med-fingers", 238, 16),
])
def test_runtime_options_preserve_shapes_and_counts(sensor_type, selection, count, links):
    options = TactileSensorsMod()._get_sensors_dict(SharpaHand(), selection, sensor_type)
    assert len(options) == links
    total = 0
    for value in options.values():
        shape = np.asarray(value.sensor.probe_local_pos).shape
        assert len(shape) == 3 and shape[-1] == 3 and min(shape[:2]) >= 2
        assert np.all(np.asarray(value.sensor.probe_radius) > 0)
        total += shape[0] * shape[1]
    assert total == count


@pytest.mark.parametrize("profile", ["actual", "med"])
def test_acceleration_eligible_grids(profile, monkeypatch):
    import genesis as gs
    monkeypatch.setattr(gs, "np_float", np.float32, raising=False)
    monkeypatch.setattr(gs, "EPS", float(np.finfo(np.float32).eps))
    from genesis.engine.sensors.tactile_shared import normalize_grid_probe_layout

    for link, probe in TactileSensorsMod._parse_probe_config(SharpaHand.metadata.tactile_probe_cfgs[profile]).items():
        layout = normalize_grid_probe_layout(probe["local_pos"], probe["local_normal"], is_grid=True)
        assert layout[2], f"{link} cannot use grid FFT"
        assert np.isfinite(layout[-1]).all() and (layout[-1] > 0).all()
        assert abs(np.dot(layout[5], layout[6])) < .95
