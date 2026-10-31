# Tactile Genesis: dexterous-hand experiments

The task, sensor and figure code behind the paper. Three hand-task pairs are studied:

| Pair | Task | Hand | Metric |
|---|---|---|---|
| `in_hand_repose-wuji` | reposing a held object to a goal orientation | Wuji | goals reached per episode |
| `in_fingers_rotate-sharpa` | continuous in-finger rotation | Sharpa | total turns per episode |
| `grasp_lift-allegro` | grasp and lift | Allegro | objects carried |

For each pair a privileged teacher supervises one student per tactile representation, over eight
representations plus a proprioception-only baseline, and over three force/torque imperfection
profiles. See [`docs/tactile_experiments.md`](docs/tactile_experiments.md) for the full matrix.

## Setup

```bash
uv sync   # installs eden, genesis-world and rsl-rl-lib from the sibling checkouts
uv run python main.py --task=in_hand_repose --robot=wuji --stage=1 --cpu \
  --config=conf/experiments/tiny.yaml --no_loaded_video --no_checkpoint_video
```

Copy `.env.example` to `.env` if you want Weights & Biases logging; training runs without it.

## Tactile sensors

`src/tactile_sensors.py` maps each policy observation type onto a Genesis sensor:

| Observation | Genesis sensor | Output per probe |
|---|---|---|
| `bool` | `ContactProbe` | contact bit |
| `agg_bool` | `ContactProbe`, summed per link | 1 per link |
| `depth` | `ContactDepthProbe` | penetration depth |
| `agg_force` | `ContactDepthProbe`, depth-weighted normals per link | 3 per link |
| `force` / `force_torque` | `KinematicTaxel` | 3 / 6 |
| `proximity` | `ProximityTaxel` | 6 |
| `elastomer` | `ElastomerTaxel` | 3 marker displacement |
| `link_bool` / `link_force` | `Contact` / `ContactForce` | 1 / 3 per link |

Probe layouts are decoupled from the readouts. Each hand carries an `actual` layout derived from
its real device (Wuji 526, Allegro's XELA V4 368, Sharpa 1206) and synthetic `low`/`high` layouts
at roughly 128 and 512 probes. Pass them as `--sensors=<layout>-<placement>/<modality>`, e.g.
`--sensors=actual-hand/force_torque`.

Measurement imperfections are selected with `--tactile_imperfection=clean|mild|strong`; the
explicit values live in `conf/sensor/tactile_params.yaml`.

## Running the sweep

```bash
python scripts/expand_distill_config.py                     # preview all nine entries
python scripts/expand_distill_config.py in_hand_repose-wuji # one pair
python scripts/expand_distill_config.py in_hand_repose-wuji --submit --partition=<your-partition>
python scripts/distill_status.py --task in_hand_repose      # progress
```

Preflight validates every cell before submitting any of it, and exits 2 if a teacher, its saved
config or a task override is missing. `--submit` launches through
[`scripts/slurm/run.sh`](scripts/slurm/run.sh); adjust its SBATCH preamble for your cluster.

## Figures

Every figure in the paper is regenerated from data in this repository, except where noted under
"Not reproducible here" below.

```bash
# Representation and imperfection study (paper Fig. 5)
python scripts/collect_distill_results.py logs -o results.json \
  --teacher teachers/f141/in_hand_repose-wuji/model_5999.pt   # re-collect from run logs
python scripts/plot_distill_ablations.py docs/experiments/distill-results-2999.json \
  -o ablations.pdf --teachers docs/experiments/teacher-scores.json --final-iteration 2500

# Throughput benchmark (paper Fig. 2)
DATA_DIR=data/sensor_benchmarks/5090 bash scripts/plot/plot_benchmark_sensor.sh

# Comparison with prior simulators (paper Fig. 3)
python scripts/plot/plot_tacsl_comparison.py
python scripts/plot/plot_tacmap_comparison.py
python scripts/plot/plot_hydroshear_comparison.py

# Material classification from force and temperature (paper Fig. 6)
python scripts/material_id/collect_slides.py          # regenerate data/material_id/slides.npz
python scripts/material_id/fit_svm.py --data data/material_id/slides.npz
```

The collected sweep results are committed at
[`docs/experiments/distill-results-2999.json`](docs/experiments/distill-results-2999.json), so the
study figure and the project page's interactive charts can be rebuilt without re-running training.

## Layout

| Path | Contents |
|---|---|
| `src/tactile_sensors.py` | sensor registry: modality to Genesis sensor, parameters, imperfections |
| `src/task_mods.py` | composes a task: robot, sensors, object set, runner |
| `src/tasks/` | the three task definitions and their reward/observation terms |
| `src/entities/robots/` | Wuji, Allegro and Sharpa hand definitions, including probe layouts |
| `src/models/` | tactile encoders (CNN, ConvRNN, canvas) and the pre-encode policy wrappers |
| `src/distill_experiments.py` | sweep expansion and preflight, independent of any scheduler |
| `src/assets/sensors/<hand>/` | probe layouts with `layout_metadata.json` and a placement preview |
| `conf/sensor/tactile_params.yaml` | sensor parameters and the clean/mild/strong profiles |
| `conf/experiments/tactile_distillation.yaml` | the sweep matrix |
| `data/sensor_benchmarks/` | raw throughput logs and the transcribed prior-work numbers |
| `docs/experiments/` | collected sweep results, teacher scores and teacher provenance |

## Tools

| Script | Purpose |
|---|---|
| `scripts/sensors_viewer.py` | visualize probe placements and live readings inside a task |
| `scripts/task_viewer.py` | step a task interactively with manual actions |
| `scripts/sensor_probes_selector.py` | place and select probes on a hand, writes the probe JSON |
| `scripts/generate_synthetic_tactile.py` | regenerate the `low`/`high` layouts (`--check` to verify) |
| `scripts/plot_taxel_handmap.py` | unrolled taxel maps like the paper's layout row |
| `scripts/build_ablation_widget.py` | reshape the sweep results into the project page's charts |
| `scripts/animate_tactile.py` | render a recorded tactile episode to video |

## Validation

```bash
PYTHONPATH=src pytest -q
python scripts/generate_synthetic_tactile.py --check
```

## Not reproducible from this repository

Five things need inputs that are not in this snapshot, and are documented rather than papered over:

- **Anything tracked by Git LFS.** `.gitattributes` routes every `*.pt` through LFS, which covers the
  cached grasp poses under `src/assets/grasps/`. Clone with `git lfs install` before `git clone`, or
  run `git lfs pull` afterwards; without it those files are 130-byte pointer stubs and the rotation
  recipes fail to load their grasp cache.
- **The three teacher checkpoints.** `conf/experiments/tactile_distillation.yaml` points at
  `teachers/f141/<pair>/model_*.pt`. Their SHA-256 digests and training provenance are in
  [`docs/experiments/pinned-teachers.md`](docs/experiments/pinned-teachers.md). Without them
  preflight exits 2 for every cell, and only the committed results JSON can be replotted.
- **The real GelSight marker data.** `scripts/fots_marker_compare.py` reads tracked markers from
  `reference/FOTS/marker_calib/data/<frame>.csv` (frames 0-5, plus the matching PNGs for the image
  overlay), which are FOTS's data and are not redistributed here. Everything else the comparison
  needs is in this repository: the FOTS constants and marker-motion model are transcribed into
  `fots_marker_compare.py`, and `hydroshear_ref.py` is a standalone port of HydroShear's marker
  model that imports only numpy and torch. Drop those CSVs in place and
  `bash scripts/run_fots_compare.sh` regenerates the figure; `hydroshear_ref.py` is a derived work
  carrying HydroShear's own licence.
- **The grasp-lift generalization scores.** The committed results JSON scores `grasp_lift` over the
  17 training objects, which every condition carries, so it cannot separate them. The paper's
  50-unseen-variant curves come from re-evaluating each student on rescaled variants:
  `scripts/submit_lift_generalization.py` generates them procedurally and writes
  `<dir>/<modality>-<level>[@<iter>]/metrics.json`, which
  `collect_distill_results.py --lift-generalization <dir>` then folds in. The script and its
  evaluator ship here; only the evaluation outputs are missing, and reproducing them needs the
  teacher, the trained students and a GPU.
- **The contact-audio sensor.** `../Genesis` is upstream v1.4.1, the revision
  `src/fresh_campaign.py` asserts and every `fresh_v141` recipe trained against. The tactile sensors
  are upstreamed there; the contact-audio sensor shown on the project page is not, so that
  demonstration cannot be reproduced from this snapshot. Nothing in this repository calls it.
