# Tactile Genesis

> Note: this is not the final version of the paper. We plan to update all experiments with the latest version of Genesis World.

Self-contained code snapshot reproducing [Tactile Genesis: Simulating Tactile Sensors at Scale to
Explore Dexterous Task Learning](https://arxiv.org/abs/2606.22332), with the project page at
[neuroagents-lab.github.io/tactile-genesis](https://neuroagents-lab.github.io/tactile-genesis/).

If you want tactile sensors for your own project, use the latest
[Genesis World](https://github.com/Genesis-Embodied-AI/genesis-world) physics engine, which includes
our tactile sensors, and optionally [Eden](https://github.com/embodied-ai-nexus/Eden) (not yet
available, but to be released before the end of 2026) if you need a managed learning framework.

## Structure

| Directory | Package | Role |
|---|---|---|
| [`dexterous-hands/`](dexterous-hands/) | `tactile_genesis` | Tactile sensors, the three studied hand-task pairs, the distillation sweep, and every figure pipeline. |
| [`Eden/`](Eden/) | `eden` | RL environment framework built on Genesis + rsl_rl. Declarative Pydantic configs, manager/term registries. |
| [`Genesis/`](Genesis/) | `genesis-world` | The physics-simulation backend (rigid/MPM/SPH/FEM/PBD) and the tactile sensor implementations. Upstream v1.4.1 (`a68b97b9`), the revision the paper's runs pin — see [`Genesis/VENDORED.md`](Genesis/VENDORED.md). |
| [`rsl_rl/`](rsl_rl/) | `rsl-rl-lib` | PPO + student/teacher distillation training loop. |
| [`docs/`](docs/) | &mdash; | The project page, served by GitHub Pages. |

Dependency topology: `dexterous-hands` → `Eden` → `Genesis`, with `rsl_rl` as the training loop.

## How the packages are wired

`dexterous-hands/pyproject.toml` resolves the three dependencies from the sibling checkouts via
`[tool.uv.sources]` (editable path installs). These sources override the package origin across the
whole resolution graph, so Eden's own transitive `genesis-world` / `rsl-rl-lib` requirements also
point at the local checkouts:

```toml
[tool.uv.sources]
eden          = { path = "../Eden",    editable = true }
genesis-world = { path = "../Genesis", editable = true }
rsl-rl-lib    = { path = "../rsl_rl",  editable = true }
```

## Quickstart

Everything is run from `dexterous-hands/`:

```bash
cd dexterous-hands
uv sync  # installs eden, genesis-world, rsl-rl-lib from local
uv run python main.py --task=in_hand_repose --robot=wuji --stage=1 --cpu \
  --config=conf/experiments/tiny.yaml --no_loaded_video --no_checkpoint_video
```

See [`dexterous-hands/README.md`](dexterous-hands/README.md) for the sensor interface, the
distillation sweep, the figure pipelines, and what is not reproducible from this snapshot.

## Project page

`docs/` is a static site with no build step. Its three interactive pieces — the throughput
benchmark, the prior-simulator comparison and the hand-layout browser — read JSON under
`docs/static/data/`, regenerated from the committed benchmark logs and sensor layouts by
`scripts/build_site_data.py` (which also copies the placement previews into
`docs/static/images/layouts/`). The distillation curves are injected into the page itself:

```bash
python scripts/build_site_data.py
python dexterous-hands/scripts/build_ablation_widget.py \
  dexterous-hands/docs/experiments/distill-results-2999.json docs/index.html \
  --teachers dexterous-hands/docs/experiments/teacher-scores-eval17.json --final-iteration 2500
```

Preview it with `python -m http.server -d docs 8000`.

## Licence

Apache 2.0 (see [`LICENSE`](LICENSE)). Redistributed robot models keep their own licences, listed
in [`NOTICE`](NOTICE) — note that the XELA V4 tactile model under
`dexterous-hands/src/assets/hands/allegro/xela_v4/` is GPL-3.0.

## Citation

If you use or reference our simulated tactile sensors in any way, please cite:

```bibtex
@article{chung2026tactilegenesis,
  title   = {Tactile Genesis: Simulating Tactile Sensors at Scale to Explore Dexterous Task Learning},
  author  = {Chung, Trinity and Yamazaki, Kashu and Patel, Dhruv and Duburcq, Alexis and Qiao, Yiling and Fragkiadaki, Katerina and Nayebi, Aran},
  journal = {arXiv preprint arXiv:2606.22332},
  year    = {2026},
  url     = {https://arxiv.org/abs/2606.22332}
}
```
