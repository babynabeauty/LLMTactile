# Sharpa tactile layouts

- `actual`: **1,206 probes on five fingertip links**, sampled at nominal **1 mm projected spacing**. Thumb 18×15 (270); each other fingertip 18×13 (234).
- `med`: **526 probes on 17 links**, providing synthetic whole-hand sensing with exactly the same regional counts as Wuji `actual`.

Sharpa Wave has fingertip tactile sensing. [Sharpa's specification](https://www.sharpa.com/pages/wave) lists 1 mm spatial resolution and 240×240 tactile maps for its higher-resolution mode. The [official sensor assets](https://github.com/sharpa-robotics/sharpa-tactile-sensor-assets/tree/865530a98a0ca0e69d177f2121833f8bb3ed94de) provide position and normal maps. These are optical samples, not an enumerated set of discrete hardware taxels: the maps contain 46,119 valid samples for a non-thumb fingertip and 45,670 for the thumb, with zeros outside the footprint.

Here, `actual` means a **simulation discretization at the published spatial resolution**, not native image-pixel resolution or an exact hardware taxel count. Sampling uses the front pad surface, omits edge margins, and follows the mesh curvature. Surface distances can exceed the nominal projected spacing. Probe radius is 0.5 mm, a simulation footprint rather than a calibrated receptive field. Neither layout preserves hardware row/column indices.

The vendor wave_01 maps guide coverage; the repository's MJCF uses slightly different HB1 pads. Probes are projected onto those current collision meshes, including each geom's rotation into its owning DP link frame. The source maps (millimetres) and their Apache-2.0 license are retained in `actual/reference/`. Output positions and radii are in metres. Layout metadata records source and mesh hashes.

| Region | `actual` | `med` |
| --- | ---: | ---: |
| Palm | 0 | 288 |
| Thumb | 270 | 41 |
| Index | 234 | 43 |
| Middle | 234 | 58 |
| Ring | 234 | 52 |
| Pinky | 234 | 44 |
| Total | 1,206 | 526 |

`med` includes Sharpa's movable pinky metacarpal: its 44 probes split into DP 12, MP 12, PP 12, MC 8. Other finger allocations match Wuji's distal/middle/proximal totals. Its subset counts are tips 58, midfinger 180, fingers 238, palm 288. Non-tip sensing is synthetic; Sharpa does not advertise whole-hand tactile hardware.

Each link has a dense rectangular array with curved positions and surface normals. There is no zero-radius padding. This preserves the grid acceleration path and supports bool, depth, force, proximity, and elastomer sensor options. The `actual` layout has only tips, so use `actual-tips` or `actual-hand`; it has no palm or midfinger sensors.

The `med` setting now selects `med/probes_526_hand_sharpa.json`. The old 207-probe file remains unchanged at `med/probes_207_hand_sharpa.json`; existing policies expecting that observation size must continue using that file explicitly. `low` and `high` are unchanged.

Generate and inspect:

```bash
.venv/bin/python scripts/generate_sharpa_tactile.py
.venv/bin/python scripts/generate_sharpa_tactile.py --check
PYTHONPATH=src .venv/bin/pytest -q tests/test_sharpa_tactile.py
PYTHONPATH=src .venv/bin/python scripts/hand_tactile_sandbox.py --robot sharpa --sensors actual-tips/bool
PYTHONPATH=src .venv/bin/python scripts/hand_tactile_sandbox.py --robot sharpa --sensors med-hand/bool
```

[Actual placement](actual/placement_preview.png) · [Med placement](med/placement_preview.png)
