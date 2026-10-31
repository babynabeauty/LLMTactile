# Allegro V4 with XELA curved whole-hand sensing

The registered `allegro` robot uses `right_hand.xml`, converted from XELA's published V4 assembly. This is a sensor-equipped V4 model, not Menagerie's V3 geometry.

Sources are pinned in `reference/sources.json`:

- [XELA model repository](https://github.com/mcsix/xela_models/tree/4cc8e177fedd4b66d29672536484be648eede7da), by Matt (XELA Robotics software engineer). `reference/ahrcpcpn.urdf` is the full assembly; `reference/xela.xacro` supplies taxel positions. `reference/package.xml` declares GPLv3. The source meshes and derived model are distributed under that license; see `LICENSE` and the original README.
- [Menagerie Allegro](https://github.com/google-deepmind/mujoco_menagerie/tree/71f066ad0be9cd271f7ed58c030243ef157af9f4/wonik_allegro) supplies the control baseline. Its original model, README and BSD-2-Clause license are retained with `menagerie_` prefixes.

## Conversion and control

The converter retains the source link poses, joint limits, masses and inertias, maps anatomical names to the project's existing names, and removes the empty ROS world frame. The root stays in the existing URDF palm coordinate convention. All 18 sensor pads remain fixed bodies. Source pad links without their own inertias remain massless; the source Xacro marks these inertias as included in parent links. Collision and visual geoms contribute no additional mass. Source inertial tensors, rather than geometry-derived approximations, determine dynamics.

Position actuators have `kp=1`, velocity gain `kv=0`, passive joint damping `0.1`, and zero armature. Passive damping is not a velocity-feedback actuator gain. The ±15 Nm effort bound comes from the XELA URDF. These are source-model simulation parameters, not calibrated V4 motor specifications. Menagerie's palm/base collision exclusions are transferred by anatomical name.

The robot-level 120/10 gains and the in-fingers rotation task's Allegro-specific gains, armature and torque overrides have been removed. The tactile sandbox now preserves imported gains unless `--kp` or `--kv` is explicitly provided. Other robots' task tuning remains as configured.

The 16 action names and their configured order are preserved. The neutral thumb base starts at its 0.263 rad lower limit. Existing policies and grasp caches have not been regenerated: changed geometry and dynamics require revalidation before reuse. In particular, the existing Allegro rotation grasp cache still refers to the previous model; regenerate using `conf/sample_grasps/in_fingers_rotate_allegro.yaml` before training against a matching cache.

## Reproduction

```bash
.venv/bin/python scripts/generate_allegro_tactile.py
.venv/bin/python scripts/generate_allegro_tactile.py --check
PYTHONPATH=src .venv/bin/pytest -q tests/test_allegro_tactile.py
PYTHONPATH=src .venv/bin/python tests/check_allegro_runtime.py
```

No network or ROS installation is needed to regenerate the checked-in assets. The generator records hashes of the source definitions and meshes.

See [tactile layout](../../../sensors/allegro/actual/README.md) for counts and placement previews.
