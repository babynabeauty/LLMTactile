# Allegro XELA V4 actual tactile layout

`actual` reproduces the **368-taxel count** of the XELA curved whole-hand configuration using **18 dense per-pad grids**. It uses the source sensor meshes and attachment frames rather than placing probes on the bare rounded Allegro tips.

| Region | Grids | Probes |
| --- | --- | ---: |
| Index | 3 × 4×4 pads + 5×6 tip | 78 |
| Middle | 3 × 4×4 pads + 5×6 tip | 78 |
| Ring | 3 × 4×4 pads + 5×6 tip | 78 |
| Thumb | 2 × 4×4 pads + 5×6 tip | 62 |
| Palm | 3 × 4×6 pads | 72 |
| Total | 18 grids | 368 |

Subset counts: `actual-hand` 368; `actual-tips` 120; `actual-midfinger` 176; `actual-fingers` 296; `actual-palm` 72. Optional `palm_links` metadata groups the three palm pads for tactile selection; `palm_link` still identifies the anatomical base for positioning. Existing robots without `palm_links` retain their prior subset behavior.

[The XELA Xacro](https://github.com/mcsix/xela_models/blob/4cc8e177fedd4b66d29672536484be648eede7da/xela_models/urdf/xela.xacro) supplies flat-pad positions and 30 positions/orientations on each curved tip. The native curved columns contain 5, 4, 6, 6, 4 and 5 samples. We resample each column to five points along its polyline, yielding a dense 5×6 grid that wraps over the front, sides and rounded end. Hardware indexing is not preserved. All probes are projected onto their pad mesh; triangle-edge hits are moved slightly toward the triangle interior for unambiguous outward normals after serialization.

`actual` means exact hardware counts and approximate surface placement, not calibrated taxel poses. Positions and radii use link-local metres. Each pad has one radius, half its median nearest-neighbor spacing, representing a simulation footprint. No radius-zero padding is used. All grids support the existing acceleration eligibility check and bool, depth, force, proximity and elastomer options; FFT treatment of a strongly curved surface remains a fitted-plane approximation.

Sources, licensing and the reproducible model conversion are described in the [model README](../../../hands/allegro/xela_v4/README.md). The `layout_metadata.json` records source hashes, regional counts and projection distances.

```bash
PYTHONPATH=src .venv/bin/python scripts/hand_tactile_sandbox.py --robot allegro --sensors actual-hand/bool
PYTHONPATH=src .venv/bin/python scripts/hand_tactile_sandbox.py --robot allegro --sensors actual-tips/elastomer
```

![Placement in open and flexed poses](placement_preview.png)
