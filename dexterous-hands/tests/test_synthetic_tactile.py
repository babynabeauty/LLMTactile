"""Synthetic layouts retain coverage while changing sampling budget."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from generate_synthetic_tactile import generate


@pytest.mark.parametrize("hand", ["wuji", "allegro", "sharpa"])
def test_generated_layouts_reproducible_and_cover_same_regions(hand):
    counts, regions = [], []
    for profile in ("low", "high"):
        records, manifest = generate(hand, profile)  # Includes surface, normal, radius validation.
        folder = ROOT / "src/assets/sensors" / hand / profile
        assert json.loads((folder / "probes_hand.json").read_text()) == records
        assert json.loads((folder / "layout_metadata.json").read_text()) == manifest
        counts.append(manifest["active_count"])
        regions.append([(m["link_name"], m["region"]) for m in manifest["links"]])
        assert any(region == "palm" for _, region in regions[-1])
    assert regions[0] == regions[1]
    assert 100 <= counts[0] <= 180
    assert 450 <= counts[1] <= 600


@pytest.mark.parametrize("hand", ["wuji", "allegro", "sharpa"])
@pytest.mark.parametrize("profile", ["low", "high"])
def test_grid_acceleration_and_encoder_layout(hand, profile, monkeypatch):
    import genesis as gs
    from genesis.engine.sensors.tactile_shared import normalize_grid_probe_layout
    from task_mods import TactileSensorsMod

    monkeypatch.setattr(gs, "np_float", np.float32, raising=False)
    monkeypatch.setattr(gs, "EPS", float(np.finfo(np.float32).eps))
    path = ROOT / "src/assets/sensors" / hand / profile / "probes_hand.json"
    for link, probes in TactileSensorsMod._parse_probe_config(str(path)).items():
        layout = normalize_grid_probe_layout(probes["local_pos"], probes["local_normal"], is_grid=True)
        assert layout[2], link
        assert (layout[-1] > 0).all()
