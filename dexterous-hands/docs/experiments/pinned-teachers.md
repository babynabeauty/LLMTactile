# Pinned distillation teachers

Three F141 policies are frozen as the teachers for the distillation ablations.
`conf/experiments/tactile_distillation.yaml` points at the copies under `teachers/f141/`,
which sit outside version control; each directory holds the checkpoint and the teacher's
saved `eden_config.yaml`.

| Task | Hand | W&B | Checkpoint | Iteration |
|---|---|---|---|---|
| grasp_lift | allegro | `05g6r800` | `teachers/f141/grasp_lift-allegro/model_4999.pt` | 4999 of 5000 |
| in_hand_repose | wuji | `90x8btvt` | `teachers/f141/in_hand_repose-wuji/model_5999.pt` | 5999 of 6000 |
| in_fingers_rotate | sharpa | `cupwtv7u` | `teachers/f141/in_fingers_rotate-sharpa/model_5500.pt` | 5500 of 6000 |

The copy every student loads is byte-identical to the source:

| Teacher | SHA-256 |
|---|---|
| Allegro lift 4999 | `edc70bfea612c7c3583e880fe1a1cae944e3483ff4bdc07d7d0431b48aca70e5` |
| Wuji repose 5999 | `a7bd58658e9f62ed83fe7ab0e38ca9c8f87cb3f2ba10558ad3f4d7405c8689ad` |
| Sharpa rotation 5500 | `12ba80753550a0874b0d0328c0ff7fdfe6742ad1c6b319317b1f0d641fd7617a` |

Metrics at the pinned checkpoint, from each run's own `evaluation/` directory, 272 episodes
at 16 per resolved variant, clean and probe reported separately:

| Task | Hand | Retention clean/probe | Task metric clean/probe | Probe recovery |
|---|---|---|---|---|
| grasp_lift | allegro | 98.9% / 98.5% | settled 60.3% / 61.8%, lifted 99.6% / 99.6% | 55.5% |
| in_hand_repose | wuji | 98.2% / 98.5% | settled 93.8% / 94.1% | 86.7% |
| in_fingers_rotate | sharpa | 99.6% / 99.6% | retained axial speed 2.71 / 2.70 rad/s | 0.0% |

## The teacher reference line

`docs/experiments/teacher-scores.json` holds the value each figure draws as the dashed
teacher line. It must be computed in the same measure *and over the same evaluation set* as
the student curves it is drawn against, because the collector's `TASK_METRIC` reductions are
not all scale-free: `grasp_lift` uses `objects carried`, a count of distinct object variants,
so its value scales with how many variants the evaluation offers. The students' in-training
evaluations cover 17 variants, which caps their curves at 17; a teacher scored over a
50-variant set reads 50 on the same axis and is not comparable.

Recompute this file from the teacher's own clean evaluation at its pinned iteration, through
the collector's `score()`, whenever `TASK_METRIC` changes or the evaluation set changes. Over
the 17-variant evaluations the teachers read 17.00 objects, 15.52 goals per episode, and 8.55
turns.

## Provenance

All three are seed 42 on Genesis v1.4.1 (`a68b97b9`) with Eden `7faee0c`, trained by the
runs whose recipes are named below:

- `F141-lift-simple-rnd-v7-allegro-s42-...-f95a875ed8b5`, recipe
  `grasp_lift_allegro_fresh_v141_simple_contact_fast_v7`.
- `F141-repose-gentle025-rnd-v7-in_hand_repose-wuji-s42-...-adc6e41d1d24`, recipe
  `in_hand_repose_wuji_fresh_v141_limits_gentle025_rnd_fast_v7`.
- `F141-rotation-backtrackdebt-rnd-v8-in_fingers_rotate-sharpa-s42-...-797c951fa9e3`, recipe
  `in_fingers_rotate_sharpa_fresh_v141_hora_gain4_backtrackdebt_noforce_rnd_fast_v8`.

Lift and repose are pinned at their final checkpoint. Rotation is pinned at 5500 of its
6000 updates, the last checkpoint written before the students were launched; its score is
flat over the preceding thousand updates (2.704 at 4500, 2.709 at 5000, 2.707 at 5500), so
the remaining 499 sit inside run-to-run noise.

The rotation and repose runs were interrupted partway and resumed from their last
checkpoint into the same log directory; the lift run trained straight through. Resuming
re-ran the iterations W&B had already recorded, so that run's W&B history has a flat stretch
where the out-of-order points were dropped. TensorBoard keeps them.

Teachers were trained on a separate cluster from the students, so the three checkpoints
were copied into `teachers/f141/` rather than referenced across filesystems. The hashes above
are what the students actually loaded.

## Ablations run against these teachers

`<task>-<hand>` distils every tactile type at the actual layout, clean, alongside one
proprioception-only baseline. `<task>-<hand>-imperfection` holds the type at `force_torque`
and adds the mild and strong profiles, reusing the first entry's baseline.

### Surface sample points on the point-cloud sensors

`elastomer` and `proximity` build their reading by sampling the object surface, and the
first attempts at three cells faulted inside `point_cloud_tactile.py` -- an out-of-memory
abort on Allegro `elastomer`, illegal addresses on Wuji `elastomer` and Wuji `proximity`.
Lowering `n_sample_points` clears all three. The knob changes how densely the kernel queries
the surface; the taxel layout, the observation dimension, and the batch are unchanged, so
the comparison within a hand still holds.

| Hand | Sensor | `n_sample_points` |
|---|---|---|
| allegro | elastomer | 500 |
| wuji | elastomer | 250 |
| wuji | proximity | 500 |
| every other point-cloud cell | elastomer, proximity | 1000 |

The values differ per cell because the fix landed as a change to the shared default in
`conf/sensor/tactile_params.yaml` (1000 to 500 for both sensors) partway through the sweep,
so a cell runs at whatever the default was when its job started, except Wuji `elastomer`,
which takes 250 from `--tactile_sample_points`. Cells that had already finished at 1000 were
left alone rather than rerun. Read the point-cloud rows across hands with that in mind.

A consequence worth knowing before resuming any of these: `n_sample_points` is part of the
resolved config, and `validate_run_contract` compares the whole resolved config against
`run_contract.yaml`. A cell that started before the default changed will refuse to resume
with "Run configuration changed" unless it is given its original value back explicitly, as
`--tactile_sample_points=1000`. The flag is not part of `cli_run_identity`, so passing it
does not move the run directory.

Elastomer is also the slowest cell by a wide margin. Sharpa runs it at about 275 minutes per
500 updates, so a full 3000-update distillation needs roughly 27 hours and does not fit the
24-hour `#SBATCH --time` in `scripts/slurm/run.sh`; that cell timed out at ~2700 updates on
its first attempt. Submit elastomer with a longer `--time` on the sbatch command line, which
overrides the script's default.

Figures include a cell only once it reaches iteration 2999.
