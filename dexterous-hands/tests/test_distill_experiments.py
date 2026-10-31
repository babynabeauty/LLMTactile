"""Experiment preparation is testable without a simulator scene or finalized teachers."""

from types import SimpleNamespace

import pytest
import yaml

from distill_experiments import ROOT, expand, init_genesis_for_overrides, modifiers, preflight
from experiment_runs import cli_run_identity, serialize, validate_run_contract
from registry import get_task_config
from tactile_sensors import TACTILE_SENSORS, imperfection_params, probe_imperfection_params
from task_mods import TactileSensorsMod

MATRIX = ROOT / "conf/experiments/tactile_distillation.yaml"


def test_matrix_and_baseline_deduplication():
    """Invariants of the menu, whatever sweep it currently selects."""
    jobs = expand(MATRIX)
    pairs = {(j["hand"], j["task"]) for j in jobs}
    assert len(pairs) == 9
    # One proprioception-only baseline per pair per seed, independent of tactile conditions.
    baselines = [(j["task"], j["hand"], j["seed"]) for j in jobs if j["sensors"] == "none"]
    assert len(baselines) == len(set(baselines)) == 9
    assert len({(j["task"], j["hand"], j["sensors"], j["imperfection"], j["seed"]) for j in jobs}) == len(jobs)


def test_baseline_student_drops_the_tactile_group():
    """A teacher recipe naming tactile_sensors must not follow it into the --sensors=none cell."""
    init_genesis_for_overrides()
    for job in expand(MATRIX):
        if job["sensors"] != "none" or not job["task_config"]:
            continue
        cfg = get_task_config(
            task_name=job["task"], modifiers=modifiers(job), config_override_path=job["task_config"]
        )
        assert "tactile_sensors" not in cfg.runner_options.obs_groups["student"]


def test_wandb_project_is_per_experiment_and_not_part_of_identity():
    """Each teacher's set reports to its own project, and moving it must not rename runs."""
    projects = {j["experiment"]: j["wandb_project"] for j in expand(MATRIX)}
    assert len(set(projects.values())) == len(projects)
    job = next(j for j in expand(MATRIX) if not j["task_config"] and j["sensors"] != "none")
    assert preflight(dict(job))["run_name"] == preflight({**job, "wandb_project": "tgs-elsewhere"})["run_name"]


def test_derived_task_configs_match_their_recipes():
    """Every `_distill.yaml` the matrix uses is still what its teacher recipe derives."""
    import subprocess
    import sys

    recipes = [
        ROOT / entry["task_config"].replace("_distill.yaml", ".yaml")
        for entry in yaml.safe_load(MATRIX.read_text())["experiments"].values()
        if entry.get("task_config")
    ]
    if not recipes:
        pytest.skip("No derived task configs in the matrix")
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/make_distill_config.py"), "--check", *map(str, recipes)],
        cwd=ROOT,
        check=True,
    )


@pytest.mark.parametrize("name", list(yaml.safe_load(MATRIX.read_text())["experiments"]))
def test_target_configs_and_missing_teachers(name):
    names = set()
    for job in expand(MATRIX, [name]):
        result = preflight(job)
        if job["teacher_checkpoint"]:
            if not (ROOT / job["teacher_checkpoint"]).exists():
                pytest.skip(f"pinned teacher weights are not in this checkout: {job['teacher_checkpoint']}")
            assert result["errors"] == []
        else:
            assert result["errors"] == ["teacher_checkpoint is not set", "teacher_config is not set"]
        assert result["run_name"] not in names
        names.add(result["run_name"])
        cfg = get_task_config(task_name=job["task"], modifiers=modifiers(job, stage=1))
        assert cfg.runner_options.obs_groups["actor"] == cfg.runner_options.obs_groups["teacher"]


def test_every_modality_and_profile_builds():
    from entities.robots.wuji_hand import WujiHand

    for kind, spec in TACTILE_SENSORS.items():
        placement = "links" if spec.placement == "link" else "low-hand"
        for level in ("clean", "mild", "strong"):
            options = TactileSensorsMod(tactile_imperfection=level)._get_sensors_dict(WujiHand(), placement, kind)
            assert options
            for option in options.values():
                import numpy as np

                expected = (
                    imperfection_params(kind, level)
                    if spec.placement == "link"
                    else probe_imperfection_params(kind, level, np.asarray(option.sensor.probe_local_pos).size // 3)
                )
                for key, value in expected.items():
                    actual = getattr(option.sensor, key)
                    # Genesis expands scalar channel parameters to tuples.
                    if isinstance(actual, tuple) and isinstance(value, (float, int)):
                        assert all(v == value for v in actual)
                    else:
                        np.testing.assert_allclose(actual, value)


def test_noisy_alias_and_conflict():
    from entities.robots.wuji_hand import WujiHand

    with pytest.warns(FutureWarning):
        old = TactileSensorsMod()._get_sensors_dict(WujiHand(), "low-hand", "force", noisy=True)
    new = TactileSensorsMod(tactile_imperfection="strong")._get_sensors_dict(WujiHand(), "low-hand", "force")
    assert serialize(old) == serialize(new)
    with pytest.raises(ValueError, match="conflicts"):
        TactileSensorsMod(tactile_imperfection="clean")._get_sensors_dict(WujiHand(), "low-hand", "force", noisy=True)


def test_run_contract_preserves_data(tmp_path):
    cfg = get_task_config(task_name="in_hand_repose", modifiers={"robot": "wuji", "stage": 1})
    args = SimpleNamespace(
        task="in_hand_repose",
        robot="wuji",
        sensors="none",
        checkpoint=None,
        actor_checkpoint=None,
        critic_checkpoint=None,
        resume=False,
    )
    validate_run_contract(cfg, args, tmp_path)
    checkpoint = tmp_path / "model_2.pt"
    checkpoint.write_bytes(b"precious research result")
    before = (tmp_path / "run_contract.yaml").read_bytes()
    with pytest.raises(ValueError, match="already exists"):
        validate_run_contract(cfg, args, tmp_path)
    args.resume = True
    validate_run_contract(cfg, args, tmp_path)
    cfg.runner_options.seed += 1
    with pytest.raises(ValueError, match="configuration changed"):
        validate_run_contract(cfg, args, tmp_path)
    assert checkpoint.read_bytes() == b"precious research result"
    assert (tmp_path / "run_contract.yaml").read_bytes() == before


def test_legacy_run_not_overwritten(tmp_path):
    cfg = get_task_config(task_name="in_hand_repose", modifiers={"robot": "wuji", "stage": 1})
    args = SimpleNamespace(
        task="in_hand_repose",
        robot="wuji",
        sensors="none",
        checkpoint=None,
        actor_checkpoint=None,
        critic_checkpoint=None,
        resume=True,
    )
    (tmp_path / "model_2.pt").write_bytes(b"old")
    with pytest.raises(ValueError, match="Cannot verify legacy"):
        validate_run_contract(cfg, args, tmp_path)
    assert not (tmp_path / "run_contract.yaml").exists()


def test_cli_identity_tracks_seed_model_and_inputs(tmp_path):
    config = tmp_path / "override.yaml"
    config.write_text("foo: 1")
    args = SimpleNamespace(config=str(config), seed=42, model="tac_mlp")
    original = cli_run_identity(args)
    args.seed = 43
    assert cli_run_identity(args) != original
    args.seed = 42
    args.model = "rnn"
    assert cli_run_identity(args) != original
    args.model = "tac_mlp"
    config.write_text("foo: 2")
    assert cli_run_identity(args) != original


def test_ready_teacher_roundtrip_and_mismatch(tmp_path):
    # An entry with no task_config, so the synthetic teacher below is built the same way.
    job = next(j for j in expand(MATRIX) if not j["task_config"] and j["sensors"] != "none")
    teacher = get_task_config(task_name=job["task"], modifiers=modifiers({**job, "sensors": "none"}, stage=1))
    teacher_path = tmp_path / "eden_config.yaml"
    # Same declared-field metadata format as saved Eden configs, without initializing Genesis.
    teacher_path.write_text(yaml.safe_dump(serialize(teacher)))
    checkpoint = tmp_path / "teacher.pt"
    import torch

    torch.save({"actor_state_dict": {"mlp.0.weight": torch.ones(2, 2)}}, checkpoint)
    job.update(teacher_config=str(teacher_path), teacher_checkpoint=str(checkpoint))
    result = preflight(job)
    assert result["errors"] == []
    teacher.action_options.dofs_pos_controller.scale *= 2
    teacher_path.write_text(yaml.safe_dump(serialize(teacher)))
    result = preflight(job)
    assert any("action_options" in error for error in result["errors"])


def test_checkpoint_validation_rejects_missing_and_wrong_weights(tmp_path):
    import torch
    from checkpoint_validation import read_teacher_checkpoint, validate_distillation_weights

    path = tmp_path / "teacher.pt"
    path.write_bytes(b"not a checkpoint")
    with pytest.raises(ValueError, match="Cannot read tensor checkpoint"):
        read_teacher_checkpoint(path)
    model = torch.nn.Linear(2, 3)
    runner = SimpleNamespace(alg=SimpleNamespace(teacher=model))
    weights = dict(model.state_dict())
    weights["distribution.std_param"] = torch.ones(3)
    torch.save({"actor_state_dict": weights}, path)
    validate_distillation_weights(runner, path)
    weights["weight"] = torch.zeros(3, 4)
    torch.save({"actor_state_dict": weights}, path)
    with pytest.raises(ValueError, match="shape mismatch"):
        validate_distillation_weights(runner, path)
    del weights["bias"]
    torch.save({"actor_state_dict": weights}, path)
    with pytest.raises(ValueError, match="missing="):
        validate_distillation_weights(runner, path)


def test_submit_preflights_all_jobs_before_sbatch(monkeypatch):
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location("experiment_cli", ROOT / "scripts/expand_distill_config.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    monkeypatch.setattr(sys, "argv", ["expand_distill_config.py", "--submit", "--json"])
    monkeypatch.setattr(cli, "expand", lambda *args: [{"errors": []}, {"errors": ["missing teacher"]}])
    monkeypatch.setattr(cli, "preflight", lambda job, cache: job)

    def should_not_submit(*args, **kwargs):
        pytest.fail("sbatch called despite invalid prerequisites")

    monkeypatch.setattr(cli.subprocess, "run", should_not_submit)
    assert cli.main() == 2
