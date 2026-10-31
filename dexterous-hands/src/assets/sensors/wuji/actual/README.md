# Wuji `actual` tactile resolution

526 probes in 16 dense, link-local grids, fitted to the repository's right Wuji
Hand 2 Beta 2 meshes. The [Wuji Glove tactile reference](https://docs.wuji.tech/docs/en/wuji-glove/latest/sdk-data-reference/tactile/)
supplies the exact regional counts; placement is an anatomical approximation on
the robot. This is not the robot's built-in fingertip sensor specification or a
calibrated hardware coordinate/index mapping.

All records use the selector's `kind: grid` JSON format. Every cell is active;
there is no padding. Each link has one grid and one positive probe radius.

| Region | Tip grid | Middle grid | Proximal grid | Total |
| --- | --- | --- | --- | --- |
| Palm | 16 × 18 on `r_wrist` | — | — | 288 |
| Thumb | 3 × 4 | 3 × 5 | 2 × 7 | 41 |
| Index | 2 × 4 | 3 × 5 | 4 × 5 | 43 |
| Middle | 4 × 4 | 3 × 6 | 4 × 6 | 58 |
| Ring | 2 × 5 | 3 × 6 | 4 × 6 | 52 |
| Pinky | 3 × 4 | 4 × 4 | 4 × 4 | 44 |

Tip grids attach to `*_tip_sensor_frame`, middle grids to `*_middle`, and proximal
grids to `*_proximal_abd`. Subsets contain 526 (`hand`), 288 (`palm`), 58 (`tips`),
180 (`midfinger`), or 238 (`fingers`) active probes. Per-link factorization is a
simulation choice: neither grid dimensions nor row/column ordering match the SDK.

## Use and inspect

From the repository root, select `--robot wuji --sensors actual-hand/force` in a
task or viewer supporting those options. Other sensor types and the usual subsets
work, for example `actual-tips/bool` and `actual-palm/elastomer`.

Open the configuration in the mesh point selector:

```bash
PYTHONPATH=src .venv/bin/python scripts/sensor_probes_selector.py \
  --robot wuji --gui \
  --file src/assets/sensors/wuji/actual/probes_526_hand_wuji.json
```

Run the interactive contact sandbox:

```bash
PYTHONPATH=src .venv/bin/python scripts/hand_tactile_sandbox.py \
  --robot wuji --sensors actual-hand/bool --num_envs 1
```

![Open and flexed placement](placement_preview.png)

## Geometry and regeneration

The generator samples rows along each link's contact strip, fits columns to mesh
cross-sections, and ray-projects from the local +Y side. Each probe stores its
surface position in metres and an outward unit normal in the same link frame.
Geometry queries run in millimetres to avoid small-triangle tolerance errors, then
convert back to metres. The palm and thumb strips avoid mechanical openings.

Probe radius is half the median nearest-neighbor spacing on that link, a simulation
footprint assumption. Existing sensor-type overrides still apply (for example,
the proximity sensor's configured sensing radius).

The grids follow curved surfaces. Genesis can use its grid FFT path with averaged
spacing and normals; its warning about an irregular grid is expected. No claim is
made that FFT results on this fit equal a direct calculation on the curved surface.

```bash
.venv/bin/python scripts/generate_wuji_tactile.py --preview
.venv/bin/python scripts/generate_wuji_tactile.py --check
PYTHONPATH=src .venv/bin/pytest -q tests/test_wuji_tactile.py
```

`layout_metadata.json` records grid shapes, counts, radii, and source/model hashes.
The two reference images are from Wuji's linked documentation, retrieved
2026-09-09; they document the source resolution and coverage. The matrix image is
also checked for its 526 active tiles. Regeneration is offline and overwrites the
generated probe JSON, metadata, and (with `--preview`) placement image. Save manual
selector edits under another name if you want to keep them when regenerating.

Validation covers exact counts, dense per-link grids, mesh-surface agreement,
outward normals, all subsets across five sensor types, grid FFT eligibility, and
articulated frame transforms checked against MuJoCo. A headless Genesis contact
check also verified all 16 grids, a zero-contact baseline, positive palm readings
under sphere contact, and FFT selection on a curved fingertip elastomer sensor.
