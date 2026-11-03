# Imported UR7e/XHand lab platform

Copied from the user-provided `ur7e_xhand/platform_meshes/base_link.STL` and
`ur7e_xhand/ur7e_xhand.urdf`. `manifest.json` pins both source files by SHA-256.
The source assembly is retained for provenance; only its platform is imported.
The current verified UR7e, XHand tactile model, and camera calibration are kept.

The supplied platform has a 1.2 × 0.8 m top, is 0.9 m tall, and comprises a
0.8 m solid body plus four 0.1 m feet. This is the source CAD geometry, including
its unusually thick body. The source OBJ is a simplified hull with different
height, so it is deliberately not used as a collider. Five boxes reproduce the
STL solid without filling the spaces between the feet. Source URDF has no
platform collision element.

`platform_asset.py` levels the CAD by replacing rounded -1.57 rad rotations
with exact -pi/2. The platform frame is translated (-0.1, 0, -0.7) m from the
source: its top remains at z=0.2 in the existing policy frame, and the current
object has margin from the left edge. Ground is z=-0.7. Robot and all three
observation camera poses are unchanged. Table bounds: x=[-0.72,0.48],
y=[-0.67,0.13], z=[-0.7,0.2] m. Existing trajectories must respect this new
footprint; the former table was larger in y.

From the repository root:

```bash
.venv/bin/python ur7e_xhand_sim/preview_environment.py --output outputs/platform_preview
```

This runs 100 physics steps, checks object support and 18 robot DOFs, then
exports the real renderer's overview, front, left, wrist and hand views,
`contact_sheet.png`, and `validation.json`. The overview is a diagnostic camera
only and does not change the three policy observation cameras.
