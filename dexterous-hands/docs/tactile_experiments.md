# Preparing tactile student experiments

The matrix spans nine hand/task pairs: Wuji, Allegro, and Sharpa crossed with `in_hand_repose`, `in_fingers_rotate`, and `grasp_lift`. Each student learns from its own pair's teacher; there is no shared cross-hand policy. Three of the nine carry pinned teachers and are the pairs the paper reports: Wuji `in_hand_repose`, Sharpa `in_fingers_rotate`, and Allegro `grasp_lift`. The other six have no teacher and fail preflight.

## Layouts

| Hand | Actual | Synthetic low | Synthetic high |
|---|---:|---:|---:|
| Wuji | 526 | 140 | 507 |
| Allegro | 368 | 140 | 507 |
| Sharpa | 1,206 | 130 | 516 |

`actual` preserves the existing source-derived layout, including its coverage and provenance. Wuji's layout fits glove regional counts to robot meshes; it is not a calibrated robot hardware mapping. Allegro uses the existing XELA V4 layout. Sharpa's actual layout covers fingertips at nominal 1 mm sampling, not native image pixels.

`low` and `high` are synthetic **whole-hand contact-surface layouts**, covering palm and finger patches, not dorsal surfaces or joint gaps. Their labels denote approximately 128/512-probe budgets, not fewer/more probes than each device's actual layout. Counts vary through rectangular-grid rounding and a 2x2 minimum per patch. Allocations use patch area; both densities use exactly the same patches. Sharpa's actual-to-synthetic comparison therefore changes coverage as well as density.

Each layout records region counts, grid shapes, approximate patch areas, source hashes, and radii in `src/assets/sensors/<hand>/<layout>/layout_metadata.json`. `placement_preview.png` shows positions and sensing footprints. Source patches come from the existing whole-hand generators; no actual coordinates are overwritten. Reproduction:

```bash
python scripts/generate_synthetic_tactile.py --check
python scripts/generate_synthetic_tactile.py  # regenerate low/high and previews
```

Each link's radius is 1.05 times the greatest sampled surface-to-nearest-probe distance, including patch boundaries. Coverage validation samples at least 33x33 points and eight subdivisions per grid cell. This gives overlapping footprints rather than large gaps; it is a sampled geometric criterion, not a claim of calibrated physical receptive fields. Layout radii govern all probe modalities, including proximity. Legacy Sharpa `med` remains available to existing teacher recipes but is excluded from the new matrix.

## Imperfection profiles

Select `--tactile_imperfection=clean|mild|strong`; omission means clean. `/noisy` is a deprecated strong alias. Combining it with an explicit clean/mild profile is an error. These parameters affect measured tactile observations, not privileged teacher observations or contact physics.

The profiles are modality-specific engineering assumptions, shared across hands. They are **not calibrations** of Wuji, XELA, or Sharpa hardware. Piezoresistive hardware motivates explicit memory effects: [Tekscan's A201 specifications](https://www.tekscan.com/flexiforce-a201-sensor) give conditioned-sensor hysteresis below 4.5% of full scale and drift below 5% per logarithmic time. [Interlink's integration guide](https://www.interlinkelectronics.com/downloads/integration-guides/fsr-400-series-integration-guide.pdf) explains load-history and constant-load effects. Those figures use different definitions from the simulator's Maxwell-element hysteresis and additive random walk; the parameters below are not direct conversions of datasheet percentages.

| Parameter | Mild | Strong |
|---|---:|---:|
| Hysteresis strength, where supported | 0.05 | 0.25 |
| Relaxation time | 0.05 s | 0.15 s |
| Per-reset gain range | 0.95–1.05 | 0.85–1.15 |
| Per-reset dead-taxel probability | 1% | 5% |
| Additive noise/bias/drift/quantization/radius uncertainty | ¼ strong values | Existing modality baseline |
| Boolean contact/release thresholds | 0.5/0.425 mm | 0.5/0.2 mm |

The complete, explicit values live in `conf/sensor/tactile_params.yaml`. Clean adds no imperfection parameters. Unsupported effects are not injected into sensor classes that cannot model them. Boolean probes use threshold hysteresis and dead taxels; link contact sensors use their supported additive effects. Depth and depth-derived aggregate force use metre-valued perturbations. Elastomer uses metre-valued marker displacement. Force channels use newtons; force/torque caches use a 1 mm reference lever arm to convert additive force scales to N m for torque channels, independently of layout radius.

For example, strong force uses white noise 0.005 N, additive bias 0.01 N, random-walk increments of 0.002 N per sensor update, and quantization of 0.01 N. Strong depth white noise is 0.0002 m (200 µm). Genesis updates these per simulation step, not per policy step. At the current 0.005 s simulation step, a 10 s uninterrupted force-sensor random walk has approximately 0.089 N standard deviation. Changing simulation frequency changes this preset's drift per second and must be treated as an experiment change.

The installed hysteresis recurrence is `xi = exp(-dt/tau)*xi + x - previous_x`, followed by `y = x + strength*xi`. Strong is a deliberately broader robustness condition, not a claim of 25% datasheet hysteresis. State resets per environment. `src/sensor_compat.py` fixes the pinned simulator's indexed random-walk reset, where `.fill_()` on an advanced-indexed tensor otherwise changes a copy.

## Matrix and preflight

`conf/experiments/tactile_distillation.yaml` holds shared defaults and nine named entries. Entry fields replace defaults. Each entry accepts `teacher_checkpoint`, `teacher_config` (the teacher's saved `eden_config.yaml`), and optional `task_config` (shared task overrides compatible with that teacher). `scripts/make_distill_config.py` derives that override from the teacher's own stage-1 recipe: it restores `tactile_sensors` to the student observation group and drops the PPO exploration knobs the distillation runner rejects. Run it with `--check` to confirm a derived config still matches its recipe. The menu exposes modalities, layouts, imperfection profiles, seeds, model/encoders, temporal reduction, and iteration budget. The checked-in menu is the tactile-type ablation: all ten modalities at the `actual` layout, clean, seed 42, 3000 iterations, against the three teachers in [pinned-teachers.md](experiments/pinned-teachers.md). That is eleven jobs per pair including the baseline. Layouts and imperfection profiles are separate axes the same menu can sweep once the informative modalities are known.

```bash
python scripts/expand_distill_config.py in_fingers_rotate-allegro
python scripts/expand_distill_config.py --json
python scripts/expand_distill_config.py in_fingers_rotate-allegro --submit
```

Without `--submit`, these commands build and validate configurations without constructing a simulator scene, logging in to W&B, creating run directories, or submitting jobs. An entry with a `task_config` initializes Genesis on the CPU first, because Eden's override merge serializes surface options that read `gs.EPS`. Missing prerequisites produce a nonzero exit code. Submission preflights the entire selection before invoking `sbatch`; it never submits a partially valid matrix. Actual scheduler failures can still occur after some jobs have been accepted.

Each hand/task/teacher/seed gets one proprioception-only student, independent of tactile conditions. `tactile_encoder` selects the encoder for the `tactile_sensors` group: a named encoder applies to every modality, and `auto` gives aggregate and link modalities an RNN and grid modalities the ConvRNN. The type ablation sets `rnn` throughout, so the modality is the only thing that varies across its cells. The expander checks layout compatibility and student privilege separation. It compares the teacher's saved physical task, action/command definitions, timing, teacher observations, and network configuration against the student setup; exploration-distribution parameters are excluded from the deterministic teacher comparison, as are render-only `surface` fields and asset paths, which a saved post-build config reports differently from a freshly built one and which are fingerprinted by content instead. Tensor checkpoint structure is checked during preflight, and exact policy keys/shapes are checked against instantiated models before distillation loads weights.

Run identities include resolved configuration and input hashes. The saved `run_contract.yaml` records resolved parameters, seed, robot/layout file hashes, and checkpoint identity. Resume permits duration/logging changes but rejects incompatible configurations. A mismatch preserves existing data and requires the original configuration or a new run name. Legacy runs without a contract remain preserved and require deliberate migration rather than automatic reuse. `scripts/distill_status.py` reads local records; a recorded running state is not a liveness guarantee.

## Validation and current limits

Configuration tests cover all nine pairs, modalities/profiles, teacher compatibility, unique jobs, and preserved resume data. Layout checks cover reproducibility, mesh fit, normals, positive radii, sampled coverage, and grid compatibility. Sensor tests exercise the installed hysteresis and drift implementations, seeded behavior, and indexed resets.

Run `scripts/check_experiment_runtime.py --task=<task> --robot=<hand>` for a short two-environment GPU check, or add `--cpu`. It builds current task dynamics, uses strong low-resolution tactile sensing, and does not load policies or train.

Eden is pinned to Milotrince/Eden `7faee0c70560b9ed5c33f051e95c975baf5a84ab`, paired with Genesis `v1.4.1` (`a68b97b9363aa0be9ccb0c8291b021f3afdd807d`). Run `.venv/bin/python scripts/patch_eden.py` after installation to apply the rebased physics-event and terminal-state hooks required by smooth teachers.

Validation on Eden `7faee0c` with Genesis `58f25d87`, the pin preceding `v1.4.1`: 241 tests passed, and all nine hand/task combinations built, reset, and completed three GPU steps with finite observations and rewards using strong low-resolution tactile sensing. The 90-job preflight reports only the intentionally unset teacher checkpoint/config references. These short checks establish runtime compatibility, not task quality or learned performance. No training or job submission was performed.
