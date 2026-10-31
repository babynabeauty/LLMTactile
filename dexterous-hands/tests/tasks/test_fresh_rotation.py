import math
from types import SimpleNamespace
import torch
from tasks.in_fingers_rotate.custom_terms import (
    RetainedQuaternionRotationReward,
    SignedQuaternionProgressReward,
    retention_rotation_score,
    signed_quaternion_increment,
)


def quat(angle):
    a = torch.as_tensor(angle) / 2
    return torch.stack((a.cos(), torch.zeros_like(a), torch.zeros_like(a), a.sin()), -1)


def test_signed_progress_full_revolutions_and_no_accumulated_payment():
    angles = torch.linspace(0, 8 * math.pi, 1001)
    q = quat(angles)
    q[::3] *= -1  # Equivalent quaternion representation must not reverse progress.
    axis = torch.tensor([0., 0., 1.])
    increments = signed_quaternion_increment(q[1:], q[:-1], axis)
    torch.testing.assert_close(increments.sum(), torch.tensor(8 * math.pi))
    torch.testing.assert_close(signed_quaternion_increment(q[:-1], q[1:], axis), -increments)
    torch.testing.assert_close(signed_quaternion_increment(q[-1:], q[-1:], axis), torch.zeros(1))


def test_signed_reward_saturation_and_drop_exclusion():
    speed = torch.tensor([-2., -.5, 0., .5, 2.])
    result = retention_rotation_score(speed, torch.zeros(5), torch.zeros(5, dtype=torch.bool), 4)
    torch.testing.assert_close(result, torch.tensor([-1.25, -1.25, .75, 2.75, 2.75]))
    assert not retention_rotation_score(speed, torch.zeros(5), torch.ones(5, dtype=torch.bool), 4).any()


def test_reward_reset_isolates_teleport(monkeypatch):
    import tasks.in_fingers_rotate.custom_terms as terms
    term = object.__new__(RetainedQuaternionRotationReward)
    q = quat(torch.tensor([.3, .6]))
    term._obj = SimpleNamespace(get_quat=lambda: q)
    term._previous_quat = quat(torch.tensor([0., 0.]))
    term._env = SimpleNamespace(dt=.05, command_manager=SimpleNamespace(get_command=lambda _: torch.tensor([[0., 0., 1.]])))
    term.rotation_gain = 4.
    monkeypatch.setattr(terms, 'retained_rotation_reward', lambda env, gain, progress_speed: progress_speed)
    term.reset(torch.tensor([0]))
    q[0] = quat(torch.tensor(1.2))  # Reset event changes pose after manager.reset.
    term.post_reset(torch.tensor([0]))
    result = term.compute()
    torch.testing.assert_close(result, torch.tensor([0., 12.]))
    torch.testing.assert_close(term.compute(), torch.zeros(2))


def test_hora_progress_wobble_cancels_and_drop_has_no_credit():
    angles = torch.tensor([0.0, 0.02, 0.04, 0.02, 0.0, 0.02])
    state = {"i": 0, "dropped": False}
    term = object.__new__(SignedQuaternionProgressReward)
    term._obj = SimpleNamespace(get_quat=lambda: quat(angles[state["i"] : state["i"] + 1]))
    term._previous_quat = quat(angles[:1]).clone()
    term._env = SimpleNamespace(
        dt=0.04,
        command_manager=SimpleNamespace(get_command=lambda _: torch.tensor([[0.0, 0.0, 1.0]])),
        termination_manager=SimpleNamespace(
            get_term=lambda _: SimpleNamespace(compute=lambda: torch.tensor([state["dropped"]]))
        ),
    )
    term.command_name = "rotation_axis"
    term.angvel_clip_min = -0.5
    term.angvel_clip_max = 0.5
    rewards = []
    for i in range(1, 5):
        state["i"] = i
        rewards.append(term.compute())
    torch.testing.assert_close(torch.stack(rewards).sum(), torch.tensor(0.0))
    state["i"] = 5
    state["dropped"] = True
    torch.testing.assert_close(term.compute(), torch.zeros(1))


def test_instantaneous_rotation_reward_plateaus_without_speed_tracking_or_memory():
    angles = torch.tensor([0.0, 0.05, 0.05, -0.05])
    state = {"i": 0}
    term = object.__new__(SignedQuaternionProgressReward)
    term._obj = SimpleNamespace(get_quat=lambda: quat(angles[state["i"] : state["i"] + 1]))
    term._previous_quat = quat(angles[:1]).clone()
    term._env = SimpleNamespace(
        dt=0.025,
        command_manager=SimpleNamespace(get_command=lambda _: torch.tensor([[0.0, 0.0, 1.0]])),
        termination_manager=SimpleNamespace(get_term=lambda _: SimpleNamespace(compute=lambda: torch.tensor([False]))),
    )
    term.command_name = "rotation_axis"
    term.angvel_clip_min = -0.5
    term.angvel_clip_max = 0.5

    values = []
    for i in range(1, 4):
        state["i"] = i
        values.append(term.compute())
    # 2 rad/s and 4 rad/s both plateau at +/-0.5; a stationary step is zero.
    torch.testing.assert_close(torch.stack(values).flatten(), torch.tensor([0.5, 0.0, -0.5]))


def test_v9_rotation_recipes_use_only_capped_signed_velocity_signal():
    import genesis as gs
    from registry import _load_tasks, get_task_config
    if not gs._initialized:
        gs.init(backend=gs.cpu, logging_level='ERROR')
    _load_tasks()
    for hand in ('wuji', 'sharpa', 'allegro'):
        c = get_task_config(
            task_name='in_fingers_rotate',
            modifiers={'robot': hand, 'obj': 'primitives', 'sensors': 'none', 'stage': 1},
            config_override_path=f'conf/experiments/in_fingers_rotate_{hand}_fresh_v141_hora_gain4_cappedvel_noforce_rnd_fast_v9.yaml',
        )
        r = c.reward_options.rotation_reward
        assert r.name == 'signed_quaternion_progress_reward'
        assert r.weight == 4.0
        assert r.angvel_clip_min == -0.5 and r.angvel_clip_max == 0.5
        assert c.reward_options.steady_rotation_bonus.weight == 0.0
        assert c.reward_options.excess_axial_speed.weight == 0.0
        assert c.reward_options.off_axis_speed.weight == 0.0
        assert c.event_options.object_pulse.linear_force_scale == 0.0
