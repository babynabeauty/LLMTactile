from types import SimpleNamespace as NS

import pytest
import torch
from eden.utils.geom import quat_error_magnitude
from tasks.in_hand_repose.custom_terms import ReachableSettledTargetRotationCommand, default_pose_l1_penalty


@pytest.fixture(scope="module", autouse=True)
def init_genesis():
    import genesis as gs
    if not gs._initialized:
        gs.init(backend=gs.cpu, logging_level="warning")


def test_exact_angle_goals_anchor_actual_pose_and_isolate_resampling():
    c = object.__new__(ReachableSettledTargetRotationCommand)
    c._env = NS(device='cpu', num_envs=128)
    actual = torch.randn(128, 4)
    actual /= actual.norm(dim=-1, keepdim=True)
    c.object = NS(get_quat=lambda: actual)
    c.quat = torch.zeros_like(actual)
    c.rotation = torch.zeros(128, 6)
    c._dwell = torch.ones(128)
    c._awarded = torch.ones(128, dtype=torch.bool)
    c.min_angle = c.max_angle = .50
    c._sync_goal_vis_orientation = lambda ids: None
    ids = torch.arange(0, 128, 2)
    c._resample_command(ids)
    torch.testing.assert_close(quat_error_magnitude(actual[ids], c.quat[ids]), torch.full((64,), .5), atol=2e-6, rtol=0)
    # target * inverse(actual) is a world-frame increment with no Z axis.
    q, a = c.quat[ids], actual[ids]
    world_delta_z = -q[:, 0]*a[:, 3] - q[:, 1]*a[:, 2] + q[:, 2]*a[:, 1] + q[:, 3]*a[:, 0]
    torch.testing.assert_close(world_delta_z, torch.zeros_like(world_delta_z), atol=2e-7, rtol=0)
    assert not c._dwell[ids].any() and not c._awarded[ids].any()
    assert (c._dwell[1::2] == 1).all() and c._awarded[1::2].all()
    assert not c.quat[1::2].any()
    # Changing the actual pose does not walk from the previous goal.
    actual.copy_(torch.roll(actual, 1, dims=0))
    c._resample_command(ids)
    torch.testing.assert_close(quat_error_magnitude(actual[ids], c.quat[ids]), torch.full((64,), .5), atol=2e-6, rtol=0)


def test_posture_penalty_uses_configured_defaults_not_first_observation():
    q = torch.tensor([[.2, -.3], [.1, .5]])
    robot = NS(dofs_idx_local=[0, 1], get_dofs_pos=lambda ids: q, default_dofs_pos=torch.tensor([.1, .1]))
    env = NS(entities={'robot': robot})
    torch.testing.assert_close(default_pose_l1_penalty(env), torch.tensor([.5, .4]))


@pytest.mark.parametrize("hand", ["wuji", "sharpa", "allegro"])
def test_fresh_repose_resolved_recipe(hand):
    from registry import _load_tasks, get_task_config
    _load_tasks()
    c = get_task_config(task_name='in_hand_repose', modifiers={'robot':hand, 'obj':'primitives', 'sensors':'none', 'stage':1}, config_override_path=f'conf/experiments/in_hand_repose_{hand}_fresh_v141.yaml')
    assert len(c.scene_options.obj.grouped_entities) > 1
    assert c.command_options.goal_rot.min_angle == c.command_options.goal_rot.max_angle == .5
    assert c.runner_options.max_iterations == 6000
    assert c.event_options.object_pulse.strength == c.event_options.object_pulse.clean_fraction == .25
    if hand == "allegro":
        assert c.action_options.dofs_pos_controller.reference_source == "accumulate"
        assert c.action_options.dofs_pos_controller.scale == 1/24
        assert c.scene_options.robot.default_root_pos[2] == .43
    else:
        assert c.action_options.dofs_pos_controller.scale_ratio == .5
    assert c.reward_options.default_pose_penalty.weight == -.04
    active = {name for name, term in c.reward_options if getattr(term, "weight", 0)}
    assert active == {'success_bonus', 'track_orientation_inv_l2', 'action_rate_penalty', 'is_dropped', 'default_pose_penalty'}


def test_completion_excludes_same_step_drop():
    from tasks.in_hand_repose.custom_terms import settled_goal_completion
    command = NS(_last_success=torch.tensor([True, True]))
    env = NS(dt=.025, command_manager=NS(get_term=lambda name:command), reset_terminated=torch.tensor([True, False]))
    torch.testing.assert_close(settled_goal_completion(env) * .025 * 20, torch.tensor([0., 20.]))


def test_variant_reset_centers_preserve_height_and_other_environments(monkeypatch):
    from shared_terms import SetSampledBottomAlignedPos
    from tasks.in_hand_repose.custom_terms import VariantCenteredReposeReset
    positions = torch.tensor([[0., 0., .53], [0., 0., .533]])
    term = object.__new__(VariantCenteredReposeReset)
    term._env = NS(device='cpu', num_envs=2)
    term._centers = torch.tensor([[.01, -.01], [-.02, .02]])
    def set_pos(values, envs_idx):
        positions[envs_idx] = values
    term._entity = NS(get_pos=lambda:positions, set_pos=set_pos)
    monkeypatch.setattr(SetSampledBottomAlignedPos, 'compute', lambda self, ids:None)
    term.compute(torch.tensor([1]))
    torch.testing.assert_close(positions, torch.tensor([[0., 0., .53], [-.02, .02, .533]]))


def test_soft_limit_margin_is_zero_inside_and_smooth_at_margin():
    from tasks.in_hand_repose.custom_terms import joint_limit_margin_cost
    p = torch.tensor([[0.0], [.8999], [.9001], [.95], [1.0], [1.1]], requires_grad=True)
    cost = joint_limit_margin_cost(p, torch.tensor([-1.]), torch.tensor([1.]), .1)
    torch.testing.assert_close(cost.detach(), torch.tensor([0., 0., 1e-6, .25, 1., 4.]), atol=3e-6, rtol=1e-4)
    cost.sum().backward()
    assert p.grad[0] == 0 and p.grad[1] == 0
    assert 0 < p.grad[2] < .1


def test_soft_limit_handles_short_locked_and_unbounded_joints():
    from tasks.in_hand_repose.custom_terms import joint_limit_margin_cost
    lo = torch.tensor([0., 0., -float('inf')])
    hi = torch.tensor([.04, 0., float('inf')])
    torch.testing.assert_close(joint_limit_margin_cost(torch.tensor([[.02, 0., 5.]]), lo, hi), torch.zeros(1))
    torch.testing.assert_close(joint_limit_margin_cost(torch.tensor([[.04, 0., 5.]]), lo, hi), torch.ones(1))


@pytest.mark.parametrize('hand', ['wuji', 'sharpa', 'allegro'])
def test_limits_gentle_successor_has_only_requested_recipe_changes(hand):
    from registry import _load_tasks, get_task_config
    _load_tasks()
    c = get_task_config(task_name='in_hand_repose', modifiers={'robot':hand, 'obj':'primitives', 'sensors':'none', 'stage':1}, config_override_path=f'conf/experiments/in_hand_repose_{hand}_fresh_v141_limits_gentle.yaml')
    assert c.reward_options.joint_limits_penalty.name == 'repose_joint_limit_margin_penalty'
    assert c.reward_options.joint_limits_penalty.weight == -.04
    assert c.reward_options.joint_limits_penalty.params['margin'] == .10
    assert c.event_options.object_pulse.probe_wrench_scale == .5
    assert c.event_options.object_pulse.strength == c.event_options.object_pulse.clean_fraction == .25
    assert c.command_options.goal_rot.min_angle == c.command_options.goal_rot.max_angle == .5
    assert c.runner_options.max_iterations == 6000


@pytest.mark.parametrize('hand', ['wuji', 'sharpa', 'allegro'])
def test_strongsuccess_rnd10_recipe_preserves_protocol(hand):
    from registry import _load_tasks, get_task_config
    _load_tasks()
    suffix = 'strongsuccess_rnd10_centerfix_fast_v8' if hand == 'allegro' else 'strongsuccess_rnd10_fast_v8'
    c = get_task_config(
        task_name='in_hand_repose',
        modifiers={'robot': hand, 'obj': 'primitives', 'sensors': 'none', 'stage': 1},
        config_override_path=f'conf/experiments/in_hand_repose_{hand}_fresh_v141_{suffix}.yaml',
    )
    assert len(c.scene_options.obj.grouped_entities) == (11 if hand == 'allegro' else 17)
    assert c.runner_options.max_iterations == 6000
    assert c.runner_options.save_interval == 500
    assert c.reward_options.success_bonus.weight == 100.0
    assert c.reward_options.track_orientation_inv_l2.weight == 1.0
    assert c.reward_options.action_rate_penalty.weight == -0.0002
    assert c.reward_options.default_pose_penalty.weight == -0.04
    assert c.reward_options.joint_limits_penalty.weight == -0.04
    assert c.reward_options.is_dropped.weight == -100.0
    assert c.command_options.goal_rot.min_angle == c.command_options.goal_rot.max_angle == 0.5
    assert c.command_options.goal_rot.settle_time == 0.25
    assert c.command_options.goal_rot.max_success_angular_speed == 0.25
    assert c.event_options.object_pulse.strength == 0.25
    assert c.event_options.object_pulse.probe_wrench_scale == 0.25
    assert c.event_options.object_pulse.clean_fraction == 0.25
    if hand == 'allegro':
        assert c.event_options.place_obj.name == 'variant_centered_repose_reset'


def test_isaac_action_penalty_is_squared_action_sum():
    from tasks.in_hand_repose.custom_terms import action_l2_penalty
    action = torch.tensor([[1.0, -2.0], [0.5, 0.5]])
    env = NS(action_manager=NS(get_term=lambda _: NS(action=action)))
    torch.testing.assert_close(action_l2_penalty(env), torch.tensor([5.0, 0.5]))


@pytest.mark.parametrize('hand', ['wuji', 'sharpa', 'allegro'])
def test_isaac_direct_repose_recipe_is_immediate_and_full_range(hand):
    from registry import _load_tasks, get_task_config
    _load_tasks()
    suffix = 'isaac_direct_rnd10_centerfix_v9' if hand == 'allegro' else 'isaac_direct_rnd10_v9'
    c = get_task_config(
        task_name='in_hand_repose',
        modifiers={'robot': hand, 'obj': 'primitives', 'sensors': 'none', 'stage': 1},
        config_override_path=f'conf/experiments/in_hand_repose_{hand}_fresh_v141_{suffix}.yaml',
    )
    ctrl = c.action_options.dofs_pos_controller
    assert ctrl.reference_source == 'default'
    assert ctrl.scale_ratio == 1.0 and ctrl.scale == 1.0
    assert ctrl.limit_targets is False and ctrl.ema_alpha == 0.0
    assert c.reward_options.success_bonus.weight == 250.0
    assert c.reward_options.track_orientation_inv_l2.weight == 1.0
    assert c.reward_options.action_rate_penalty.weight == 0.0
    assert c.reward_options.action_penalty.name == 'action_l2_penalty'
    assert c.reward_options.action_penalty.weight == -0.0002
    assert c.reward_options.default_pose_penalty.weight == 0.0
    assert c.reward_options.joint_limits_penalty.weight == -0.04
    cmd = c.command_options.goal_rot
    assert cmd.min_angle == cmd.max_angle == 0.5
    assert cmd.settle_time == 0.0 and cmd.max_success_angular_speed == 1_000_000.0
    assert tuple(cmd.resampling_time_range) == (20.0, 20.0)
    assert c.env_options.episode_length_s == 20.0
