from types import SimpleNamespace
import math
import pytest
import torch
from smooth_control import limited_target
from object_wrench import WrenchPulseState
from tasks.in_hand_repose.custom_terms import OrientationProgressReward, SettledTargetRotationCommand


@pytest.mark.parametrize("dt", [0.025, 0.05])
def test_limiter_bounds_and_reversals(dt):
    torch.manual_seed(3)
    q = torch.zeros(64, 7)
    v = torch.zeros_like(q)
    lower, upper = torch.full_like(q, -0.13), torch.full_like(q, 0.17)
    desired = q
    for i in range(400):
        desired = torch.randn_like(q) * 10 if i % 10 == 0 else desired
        new, nv = limited_target(desired, q, v, lower, upper, dt)
        assert (nv.abs() <= 0.8 + 1e-5).all()
        assert ((nv - v).abs() <= 8 * dt + 1e-5).all()
        assert ((new >= lower) & (new <= upper)).all()
        q, v = new, nv


@pytest.mark.parametrize("control_dt", [0.025, 0.05])
@pytest.mark.parametrize("duration", [0.1, 0.237, 0.5])
def test_pulse_impulse_and_flags(control_dt, duration):
    s = WrenchPulseState(2, "cpu", 1, 0)
    s.reset(slice(None))
    s.duration[0] = duration
    s.peak_force[0, 0] = 1
    impulse = 0
    for i in range(120):
        s.step(0.005, i % round(control_dt / 0.005) == 0)
        impulse += float(s.force[0, 0]) * 0.005
        assert s.active[1] == 0
        arrows = s.arrow_data(torch.zeros(2, 3), [0, 1])
        assert bool(arrows) == bool(s.force[0].any())
        if arrows:
            torch.testing.assert_close(arrows[0]["force"], s.force[0])
    assert impulse == pytest.approx(2 * duration / math.pi, abs=2e-7)
    assert s.active.sum() == 0


def test_pulse_reset_isolation_and_grace():
    s = WrenchPulseState(1000, "cpu", 1, 0.25)
    s.reset(slice(None))
    s.schedule(slice(None), torch.ones(1000), torch.ones(1000) * 0.02)
    assert s.duration.sum() == 0
    s.age[:] = 1
    s.schedule(slice(None), torch.ones(1000), torch.ones(1000) * 0.02)
    assert 650 < int((s.duration > 0).sum()) < 850
    assert s.peak_force.norm(dim=-1).max() <= 2.000001
    s.step(0.005, True)
    other = s.force[1:].clone()
    s.reset(torch.tensor([0]))
    assert s.active[0] == 0 and s.force[0].sum() == 0
    torch.testing.assert_close(s.force[1:], other)


def test_progress_sign_reset_and_goal_change():
    c = object.__new__(OrientationProgressReward)
    c._env = SimpleNamespace(env_options=SimpleNamespace(sim_dt=0.005, decimation=5))
    c._cmd = SimpleNamespace(quat=torch.tensor([[1.0, 0, 0, 0], [1.0, 0, 0, 0]]))
    c._prev_goal_quat = c._cmd.quat.clone()
    c._prev_error = torch.ones(2)
    c.negative_scale = 2.0
    c.clip = (-5.0, 5.0)
    c._progress_debt = torch.zeros(2)
    err = torch.tensor([0.95, 1.05])
    c._current_error = lambda: err
    r = c.compute()
    assert r[0] > 0 and r[1] < 0
    err = torch.ones(2)
    r += c.compute()
    assert (r <= 0).all()
    c._cmd.quat[0] = torch.tensor([0.0, 1, 0, 0])
    err = torch.tensor([3.0, 1.0])
    assert c.compute()[0] == 0
    c.reset(torch.tensor([0]))
    assert c.compute()[0] == 0


def test_settled_success_once_and_reset():
    c = object.__new__(SettledTargetRotationCommand)
    c._env = SimpleNamespace(dt=0.025, device="cpu", num_envs=2)
    c.quat = torch.tensor([[1.0, 0, 0, 0], [1.0, 0, 0, 0]])
    omega = torch.zeros(2, 3)
    c.object = SimpleNamespace(get_quat=lambda: c.quat, get_ang=lambda: omega)
    c.stats = {"orientation_error": torch.zeros(2), "consecutive_success": torch.zeros(2)}
    c._dwell = torch.zeros(2)
    c._awarded = torch.zeros(2, dtype=torch.bool)
    c.time_left = torch.ones(2)
    c.orientation_success_threshold = 0.1
    c.settle_time = 0.25
    c.max_success_angular_speed = 0.25
    total = torch.zeros(2)
    omega[1, 0] = 0.3
    for _ in range(20):
        c.post_physics_refresh()
        total += c._last_success
    torch.testing.assert_close(total, torch.tensor([1.0, 0.0]))
    assert c._dwell[1] == 0


def test_controller_partial_reset(monkeypatch):
    from smooth_control import SmoothPDController
    from eden.managers.terms.actions.joint_actions import ExplicitPDController

    monkeypatch.setattr(ExplicitPDController, "reset", lambda self, ids: None)
    c = object.__new__(SmoothPDController)
    c._entity = SimpleNamespace(get_dofs_pos=lambda **kw: torch.tensor([[0.1, 0.2], [0.3, 0.4]]))
    c.dofs_idx_local = torch.arange(2)
    c.lower = torch.zeros(2, 2)
    c.upper = torch.ones(2, 2)
    c.target = torch.ones(2, 2)
    c.target_velocity = torch.ones(2, 2)
    c.filtered_target = torch.ones(2, 2)
    c._processed_action = torch.ones(2, 2)
    c._accumulated_target = torch.ones(2, 2)
    c.action_saturation = torch.ones(2, 1)
    c.reset(torch.tensor([0]))
    torch.testing.assert_close(c.target, torch.tensor([[0.1, 0.2], [1.0, 1.0]]))
    torch.testing.assert_close(c.target_velocity, torch.tensor([[0.0, 0.0], [1.0, 1.0]]))
    torch.testing.assert_close(c.filtered_target, c.target)
    torch.testing.assert_close(c._accumulated_target, c.target)
    c.reset_target_source = "cached_grasp_target"
    with pytest.raises(RuntimeError, match="loaded grasp"):
        c.reset(torch.tensor([0]))
    c._grasp_reset_target = torch.tensor([[0.5, 0.6], [0.7, 0.8]])
    c.reset(torch.tensor([0]))
    torch.testing.assert_close(c.target, torch.tensor([[0.5, 0.6], [1.0, 1.0]]))
    torch.testing.assert_close(c.filtered_target, c.target)
    torch.testing.assert_close(c._accumulated_target, c.target)


def test_binary_normalization_preserves_flag():
    from smooth_checkpoint import BinaryTailNormalization

    normalizer = BinaryTailNormalization(3)
    x = torch.tensor([[10.0, 20.0, 0.0], [30.0, 40.0, 1.0]])
    normalizer.update(x)
    y = normalizer(x)
    torch.testing.assert_close(y[:, -1], x[:, -1])
    torch.testing.assert_close(normalizer.inverse(y), x)


def test_terminal_metrics_before_reset_and_once():
    from teacher_evaluation import EpisodeCollector

    pos = torch.zeros(2, 3)
    robot = SimpleNamespace(
        get_dofs_pos=lambda **kw: torch.zeros(2, 2),
        get_dofs_vel=lambda **kw: torch.zeros(2, 2),
        get_dofs_control_force=lambda **kw: torch.zeros(2, 2),
        get_dofs_force_range=lambda **kw: (-torch.ones(2, 2), torch.ones(2, 2)),
    )
    obj = SimpleNamespace(
        get_pos=lambda: pos,
        get_quat=lambda: torch.tensor([[1., 0., 0., 0.], [1., 0., 0., 0.]]),
        get_ang=lambda: torch.tensor([[0.0, 0.0, 0.5], [0.0, 0.0, 0.5]]),
        links=[SimpleNamespace(geoms=[SimpleNamespace(active_envs_idx=torch.arange(2))])],
    )
    controller = SimpleNamespace(dofs_idx_local=torch.arange(2), processed_action=torch.zeros(2, 2))
    env = SimpleNamespace(
        num_envs=2,
        device="cpu",
        dt=0.05,
        entities={"obj": obj, "robot": robot},
        action_manager=SimpleNamespace(get_term=lambda name: controller),
        command_manager=SimpleNamespace(
            active_terms=["rotation_axis"],
            get_term=lambda name: SimpleNamespace(command=torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])),
        ),
        reset_buf=torch.tensor([True, False]),
        reset_terminated=torch.tensor([True, False]),
    )
    collector = EpisodeCollector(env)
    pos[0, 2] = -1
    collector(env)
    pos[0, 2] = 0  # automatic reset, before the next callback
    env.reset_buf = torch.tensor([False, True])
    env.reset_terminated = torch.tensor([False, False])
    collector(env)
    assert collector.dropped.tolist() == [True, False]
    assert collector.steps.tolist() == [1, 2]
    assert collector.max_displacement.tolist() == [1, 0]
    assert collector.summary()["retention"] == 0.5
    # Reported angular velocity alone cannot manufacture quaternion turn progress.
    assert collector.summary()["per_variant_rotation"][0]["median_net_turns"] == 0.0
    q = torch.tensor([[1., 0., 0., 0.], [1., 0., 0., 0.]])
    obj.get_quat = lambda: q
    fast = EpisodeCollector(env)
    q[:, 0], q[:, 3] = math.cos(0.3), -math.sin(0.3)
    env.reset_buf[:] = True
    env.reset_terminated = torch.tensor([True, False])
    fast(env)
    # Fast backward motion counts; the same rotation on a drop step does not.
    torch.testing.assert_close(fast.turns, torch.tensor([0., -0.6 / (2 * math.pi)]))
    torch.testing.assert_close(fast.backward_turns, torch.tensor([0., 0.6 / (2 * math.pi)]))


def test_progress_fast_out_slow_back_cannot_profit():
    c = object.__new__(OrientationProgressReward)
    c._env = SimpleNamespace(env_options=SimpleNamespace(sim_dt=0.005, decimation=5))
    c._cmd = SimpleNamespace(quat=torch.tensor([[1.0, 0, 0, 0]]))
    c._prev_goal_quat = c._cmd.quat.clone()
    c._prev_error = torch.tensor([1.0])
    c._progress_debt = torch.zeros(1)
    c.negative_scale = 2.0
    c.clip = (-5.0, 5.0)
    total = 0
    for error in [2.0] + [2.0 - i * 0.05 for i in range(1, 21)]:
        c._current_error = lambda: torch.tensor([error])
        total += float(c.compute()[0]) * 0.025
    assert total <= 0


def test_normalizer_reload_after_inference_update():
    from smooth_checkpoint import BinaryTailNormalization

    normalizer = BinaryTailNormalization(3)
    with torch.inference_mode():
        normalizer.update(torch.ones(2, 3))
    normalizer.load_state_dict(normalizer.state_dict())
    assert not normalizer._std.is_inference()


def test_physics_hook_force_torque_flag_share_state():
    from object_wrench import ObjectWrenchPulse

    applied = []
    pulse = object.__new__(ObjectWrenchPulse)
    pulse._env = SimpleNamespace(
        training=True,
        rigid_solver=SimpleNamespace(
            apply_links_external_wrench=lambda **kw: applied.append(
                {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in kw.items()}
            )
        ),
    )
    pulse.state = WrenchPulseState(2, "cpu", 1, 0)
    pulse.state.duration[:] = 0.1
    pulse.state.peak_force[0, 0] = 1
    pulse.state.peak_torque[1, 2] = 0.01
    pulse.links_idx = torch.tensor([0])
    pulse.draw_envs = []
    pulse.pre_physics_step(0.005, True)
    assert pulse.state.active.tolist() == [[1.0], [1.0]]
    torch.testing.assert_close(applied[0]["force"][:, 0], pulse.state.force)
    torch.testing.assert_close(applied[0]["torque"][:, 0], pulse.state.torque)
    pulse.state.reset(torch.tensor([0]))
    assert pulse.state.active.tolist() == [[0.0], [1.0]]


def test_pulse_modes_and_offcenter_moment():
    torch.manual_seed(42)
    s = WrenchPulseState(3000, "cpu", 1, 0)
    s.reset(slice(None))
    s.age[:] = 1
    radius = torch.full((3000,), 0.04)
    s.schedule(slice(None), torch.full((3000,), 0.1), radius)
    for mode in range(3):
        assert 850 < int((s.mode == mode).sum()) < 1150
    mask = s.mode == 1
    torch.testing.assert_close(s.peak_torque[mask], torch.linalg.cross(s.lever[mask], s.peak_force[mask]))
    assert (s.lever.norm(dim=-1) <= radius * 0.5 + 1e-6).all()
    assert not s.peak_force[s.mode == 2].any()
    assert s.peak_torque[s.mode == 2].norm(dim=-1).max() <= 0.020001


def test_checkpoint_contract_rejects_legacy(tmp_path):
    from smooth_checkpoint import install_smooth_runner_contract
    from smooth_control import OBSERVATION_VERSION
    from rsl_rl.modules.normalization import EmpiricalNormalization

    c = SimpleNamespace(
        observation_version=OBSERVATION_VERSION,
        dofs_name=["joint"],
        reference_source="default",
        ema_alpha=0.8,
        max_target_velocity=0.8,
        max_target_acceleration=8.0,
        limit_targets=True,
        scale_ratio=0.125,
        scale=1.0,
    )
    env = SimpleNamespace(
        num_envs=2,
        unwrapped=SimpleNamespace(
            dt=0.025, action_manager=SimpleNamespace(active_terms=["dofs_pos_controller"], get_term=lambda name: c)
        ),
        runner_dict={
            "num_steps_per_env": 24,
            "obs_groups": {"actor": ["controller", "perturbation"], "critic": ["controller", "perturbation"]},
        },
    )
    saved = []
    loaded = []
    runner = SimpleNamespace(
        alg=SimpleNamespace(
            update=lambda: None,
            actor=SimpleNamespace(obs_normalizer=EmpiricalNormalization(3)),
            critic=SimpleNamespace(obs_normalizer=EmpiricalNormalization(3)),
        ),
        save=lambda path, infos: saved.append(infos),
        load=lambda path, *args, **kw: loaded.append(path),
    )
    install_smooth_runner_contract(runner, env)
    runner.alg.update()
    assert runner.smooth_training_env_steps == 48
    runner.save("unused")
    checkpoint = tmp_path / "model.pt"
    torch.save({"infos": saved[0]}, checkpoint)
    runner.load(checkpoint)
    assert loaded
    torch.save({"infos": {}}, checkpoint)
    with pytest.raises(ValueError, match="contract mismatch"):
        runner.load(checkpoint)


def test_training_checkpoint_evaluation_schedule(monkeypatch, tmp_path):
    import teacher_evaluation

    calls = []
    monkeypatch.setattr(teacher_evaluation, "snapshot_run", lambda *args: None)
    monkeypatch.setattr(teacher_evaluation.subprocess, "run", lambda args, **kw: calls.append(args))
    monkeypatch.setattr(teacher_evaluation, "publish_checkpoint_evaluation", lambda *args: None)
    saved = []
    media = []
    runner = SimpleNamespace(
        current_learning_iteration=499,
        save=lambda path, infos=None: saved.append(path),
        logger=SimpleNamespace(writer=SimpleNamespace(save_video=lambda path, step: media.append((path, step)))),
    )
    config = SimpleNamespace(
        runner_options=SimpleNamespace(seed=42, max_iterations=5000),
        scene_options=SimpleNamespace(obj=SimpleNamespace(grouped_entities=list(range(17)))),
        event_options=SimpleNamespace(object_pulse=SimpleNamespace(strength=0.25)),
    )
    teacher_evaluation.attach_training_evaluation(runner, config, tmp_path, "in_hand_repose", "wuji")
    runner.save("499.pt")
    assert not calls
    runner.current_learning_iteration = 500
    runner.save("500.pt")
    assert len(calls) == 3
    assert all("--video" not in call for call in calls)
    assert calls[2][1] == "scripts/record_teacher_checkpoint.py"
    assert calls[2][calls[2].index("--num_envs") + 1] == "17"
    assert calls[2][calls[2].index("--output") + 1] == "500.mp4"
    assert calls[0][calls[0].index("--task") + 1] == "in_hand_repose"
    assert calls[0][calls[0].index("--robot") + 1] == "wuji"
    assert calls[0][calls[0].index("--eval_envs") + 1] == "272"
    assert calls[0][calls[0].index("--pulse_strength") + 1] == "0.0"
    assert calls[1][calls[1].index("--pulse_strength") + 1] == "0.25"
    assert [(str(path), step) for path, step in media] == [("500.mp4", 500)]


def test_checkpoint_metrics_do_not_publish_diagnostic_videos(monkeypatch, tmp_path):
    import json
    import wandb
    from teacher_evaluation import publish_checkpoint_evaluation

    (tmp_path / "metrics.json").write_text(json.dumps({"retention": 0.95, "seed": 42}))
    logged = []
    monkeypatch.setattr(wandb, "run", object())
    monkeypatch.setattr(wandb, "log", lambda payload, **kwargs: logged.append(payload))
    publish_checkpoint_evaluation(tmp_path, "clean", 500)
    assert logged == [{"Evaluation/clean/retention": 0.95, "Evaluation/checkpoint_iteration": 500}]


def test_acceptance_requires_matched_policy_and_budget():
    from scripts.check_teacher_gates import check_gates

    baseline = dict(
        training_env_steps=100, motion_sample_rate_hz=200, acceleration_rms=1.0, high_frequency_velocity_rms=1.0,
        reset_rng_protocol="runner_rng_isolated_v1",
    )
    clean = dict(
        training_env_steps=100,
        motion_sample_rate_hz=200,
        acceleration_rms=0.4,
        high_frequency_velocity_rms=0.4,
        retention=0.95,
        settled_goal_fraction=0.8,
        median_retained_axial_speed=0.5,
        checkpoint_sha256="same",
        reset_rng_protocol="runner_rng_isolated_v1",
    )
    perturbed = dict(clean, retention=0.9, pulses=10, pulse_recovery_fraction=0.9, pulse_profile="probe")
    assert check_gates("in_hand_repose", baseline, clean, perturbed) == []
    assert check_gates("in_fingers_rotate", baseline, clean, perturbed) == []
    assert check_gates("in_hand_repose", baseline, clean, dict(perturbed, checkpoint_sha256="different"))
    assert check_gates("in_hand_repose", baseline, dict(clean, training_env_steps=99), perturbed)
    assert check_gates("in_hand_repose", baseline, dict(clean, fixed_target=True), perturbed)
    assert check_gates("in_hand_repose", baseline, dict(clean, goal_curriculum=True), perturbed)
    assert check_gates("grasp_lift", baseline, dict(clean, settled_goal_fraction=0.0), perturbed)
    lift_clean = dict(clean, settled_goal_fraction=0.95)
    lift_perturbed = dict(perturbed, settled_goal_fraction=0.85)
    assert check_gates("grasp_lift", baseline, lift_clean, lift_perturbed) == []
    assert check_gates("grasp_lift", baseline, dict(lift_clean, reset_rng_protocol=None), lift_perturbed)
    assert check_gates("grasp_lift", baseline, lift_clean, dict(lift_perturbed, pulse_profile="weight_scaled"))


def test_physics_rate_spectrum_detects_twenty_hz_chatter():
    from teacher_evaluation import motion_summary

    dt = 0.005
    time = torch.arange(400) * dt
    common = dict(
        live=torch.ones(400, 1, dtype=torch.bool),
        pulse=torch.zeros(400, 1, dtype=torch.bool),
        transition=torch.zeros(400, 1, dtype=torch.bool),
        reset=torch.zeros(400, 1, dtype=torch.bool),
    )
    slow = motion_summary(dict(common, velocity=torch.sin(2 * torch.pi * 2 * time)[:, None, None]), dt)
    chatter = motion_summary(dict(common, velocity=torch.sin(2 * torch.pi * 20 * time)[:, None, None]), dt)
    assert chatter["motion_sample_rate_hz"] == 200
    assert chatter["high_frequency_velocity_rms"] == pytest.approx(2**-0.5, abs=1e-5)
    assert slow["high_frequency_velocity_rms"] < 1e-5


def test_training_preferences():
    from training_preferences import apply_training_preferences

    args = SimpleNamespace(mode="train", obj="cube", no_checkpoint_video=True)
    apply_training_preferences(args)
    assert args.obj == "primitives" and not args.no_checkpoint_video
    args = SimpleNamespace(mode="play", obj="cube", no_checkpoint_video=True)
    apply_training_preferences(args)
    assert args.obj == "cube" and args.no_checkpoint_video


def test_retention_reward_cannot_pay_for_early_quitting():
    from tasks.in_fingers_rotate.custom_terms import retention_rotation_score

    speed = torch.linspace(-20, 20, 100)
    cost = torch.linspace(0, 1e6, 100)
    alive = torch.zeros(100, dtype=torch.bool)
    reward = retention_rotation_score(speed, cost, alive)
    assert (reward >= 0).all() and (reward <= 1.25).all()
    held = retention_rotation_score(torch.zeros(100), cost, alive)
    assert (held >= 0.5).all()
    assert not retention_rotation_score(speed, cost, ~alive).any()
    assert retention_rotation_score(torch.tensor([0.5]), torch.tensor([0.0]), alive[:1]) > held[0]


def test_stronger_rotation_reward_stays_signed_and_keeps_holding_viable():
    from tasks.in_fingers_rotate.custom_terms import retention_rotation_score

    speed = torch.tensor([0.0, 0.25, -0.25, 0.5, -0.5])
    cost = torch.full_like(speed, 1e6)
    reward = retention_rotation_score(speed, cost, torch.zeros_like(speed, dtype=torch.bool), rotation_gain=4)
    assert reward[0] >= 0.5
    assert reward[1] > reward[0] and reward[2] < reward[0]
    torch.testing.assert_close(reward[1] + reward[2], 2 * reward[0])
    torch.testing.assert_close(reward[3] + reward[4], 2 * reward[0])


def test_s2_objective_uses_an_enabled_reward():
    from pathlib import Path
    import yaml

    for hand in ("wuji", "sharpa"):
        c = yaml.safe_load(Path(f"conf/experiments/in_fingers_rotate_{hand}_retain_s2.yaml").read_text())
        for reward in c["metric_options"]["objective"]["params"]["reward_names"]:
            assert c["reward_options"][reward]["weight"] != 0


def test_accumulator_zero_action_preserves_grasp_outside_soft_range(monkeypatch):
    from smooth_control import SmoothPDController
    from eden.managers.terms.actions.joint_actions import ExplicitPDController

    monkeypatch.setattr(ExplicitPDController, "build", lambda self: None)
    c = object.__new__(SmoothPDController)
    q = torch.tensor([[0.98], [0.02]])
    c._env = SimpleNamespace(num_envs=2, device="cpu", dt=0.05)
    c._entity = SimpleNamespace(
        get_dofs_pos=lambda **kw: q.clone(), get_dofs_limit=lambda **kw: (torch.zeros(1), torch.ones(1))
    )
    c.dofs_idx_local = torch.arange(1)
    c.accumulator_hard_limits = True
    c._accumulated_target = q.clone()
    c._target_lower = torch.full_like(q, 0.05)
    c._target_upper = torch.full_like(q, 0.95)
    c._raw_action = torch.zeros_like(q)
    c._prev_action = torch.zeros_like(q)
    c._dofs_pos_offset = c._motion_ref_dofs_idx = c._modifier = None
    c._scale, c._offset = 1 / 24, 0.0
    c.build()
    for _ in range(10):
        c.compute(torch.zeros_like(q))
        torch.testing.assert_close(c.target, q)
        assert not c.target_velocity.any()
    c.compute(torch.ones_like(q) * 100)
    assert c.target.max() <= 1
    assert c.target_velocity.abs().max() <= 0.4 + 1e-5


def test_training_rejects_single_object_full_config():
    from training_preferences import validate_training_objects

    c = SimpleNamespace(scene_options=SimpleNamespace(obj=SimpleNamespace(grouped_entities=None)))
    with pytest.raises(ValueError, match="multiple-object"):
        validate_training_objects(c)
    c.scene_options.obj.grouped_entities = list(range(17))
    validate_training_objects(c)
