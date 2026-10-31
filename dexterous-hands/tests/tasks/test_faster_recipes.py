"""The versioned faster recipes retain bounded smooth control and task protocols."""

import pytest


@pytest.fixture(scope="module", autouse=True)
def init_genesis():
    import genesis as gs

    if not gs._initialized:
        gs.init(backend=gs.cpu, logging_level="error")


@pytest.mark.parametrize("task", ["in_fingers_rotate", "in_hand_repose", "grasp_lift"])
@pytest.mark.parametrize("hand", ["wuji", "sharpa", "allegro"])
def test_faster_recipe_is_a_small_bounded_controller_change(task, hand):
    from registry import _load_tasks, get_task_config

    suffix = {
        "in_fingers_rotate": "fresh_v141_faster",
        "in_hand_repose": "fresh_v141_limits_gentle_faster",
        "grasp_lift": "fresh_v141_plane_clearance_faster",
    }[task]
    _load_tasks()
    c = get_task_config(
        task_name=task,
        modifiers={"robot": hand, "obj": "primitives", "sensors": "none", "stage": 1},
        config_override_path=f"conf/experiments/{task}_{hand}_{suffix}.yaml",
    )
    control = c.action_options.dofs_pos_controller
    assert control.max_target_velocity == 1.0
    assert control.max_target_acceleration == 10.0
    assert control.ema_alpha == (0.0 if task == "in_fingers_rotate" else 0.7)
    assert c.env_options.episode_length_s == 20.0
    if task == "in_hand_repose":
        assert tuple(c.command_options.goal_rot.resampling_time_range) == (6.0, 8.0)
        assert c.event_options.object_pulse.probe_wrench_scale == 0.5
    if task == "grasp_lift":
        assert c.action_options.ee_controller.linear_acceleration_limit == 0.75
        assert c.action_options.ee_controller.angular_acceleration_limit == 5.0
        assert c.reward_options.plane_clearance.name == "plane_clearance_reward"


@pytest.mark.parametrize("task", ["in_fingers_rotate", "in_hand_repose", "grasp_lift"])
@pytest.mark.parametrize("hand", ["wuji", "sharpa", "allegro"])
def test_v4_recipes_are_responsive_and_keep_task_protocols(task, hand):
    from registry import _load_tasks, get_task_config

    suffix = {
        "in_fingers_rotate": "fresh_v141_hora_fast_v4",
        "in_hand_repose": "fresh_v141_limits_gentle_fast_v4",
        "grasp_lift": "fresh_v141_plane_lift_fast_v4",
    }[task]
    _load_tasks()
    c = get_task_config(
        task_name=task,
        modifiers={"robot": hand, "obj": "primitives", "sensors": "none", "stage": 1},
        config_override_path=f"conf/experiments/{task}_{hand}_{suffix}.yaml",
    )
    control = c.action_options.dofs_pos_controller
    assert control.max_target_velocity == 4.0
    assert control.max_target_acceleration == 80.0
    assert control.ema_alpha == 0.0
    assert c.env_options.episode_length_s == 20.0
    if task == "in_fingers_rotate":
        assert control.scale == pytest.approx(1 / 12)
        assert c.reward_options.rotation_reward.name == "signed_quaternion_progress_reward"
        assert c.reward_options.rotation_reward.weight == 1.0
        assert c.reward_options.steady_rotation_bonus.weight == 0.0
        assert c.reward_options.retained_rotation.weight == 0.0
    elif task == "in_hand_repose":
        assert c.reward_options.action_rate_penalty.weight == -0.001
        assert c.event_options.object_pulse.probe_wrench_scale == 0.5
        assert tuple(c.command_options.goal_rot.resampling_time_range) == (6.0, 8.0)
    else:
        root = c.action_options.ee_controller
        assert (root.linear_velocity_limit, root.linear_acceleration_limit) == (0.6, 8.0)
        assert (root.angular_velocity_limit, root.angular_acceleration_limit) == (3.0, 30.0)
        assert c.reward_options.lift.name == "sustained_plane_lift_reward"
        assert c.reward_options.lift.weight == 5.0
        assert c.reward_options.plane_clearance.weight == 0.0


@pytest.mark.parametrize("hand", ["wuji", "sharpa", "allegro"])
def test_v5_rotation_restores_established_signed_progress_gain(hand):
    from registry import _load_tasks, get_task_config

    _load_tasks()
    c = get_task_config(
        task_name="in_fingers_rotate",
        modifiers={"robot": hand, "obj": "primitives", "sensors": "none", "stage": 1},
        config_override_path=f"conf/experiments/in_fingers_rotate_{hand}_fresh_v141_hora_gain4_fast_v5.yaml",
    )
    assert c.reward_options.rotation_reward.name == "signed_quaternion_progress_reward"
    assert c.reward_options.rotation_reward.weight == 4.0
    assert c.reward_options.rotation_reward.angvel_clip_min == -0.5
    assert c.reward_options.rotation_reward.angvel_clip_max == 0.5
    assert c.reward_options.steady_rotation_bonus.weight == 0.0
    assert c.reward_options.retained_rotation.weight == 0.0


@pytest.mark.parametrize("hand", ["wuji", "sharpa", "allegro"])
def test_v6_repose_halves_the_object_wrench_only(hand):
    from registry import _load_tasks, get_task_config

    _load_tasks()
    c = get_task_config(
        task_name="in_hand_repose",
        modifiers={"robot": hand, "obj": "primitives", "sensors": "none", "stage": 1},
        config_override_path=f"conf/experiments/in_hand_repose_{hand}_fresh_v141_limits_gentle025_fast_v6.yaml",
    )
    pulse = c.event_options.object_pulse
    assert pulse.profile == "probe"
    assert pulse.strength == 0.25
    assert pulse.probe_wrench_scale == 0.25
    assert pulse.clean_fraction == 0.25
    assert c.reward_options.joint_limits_penalty.weight == -0.04
    assert tuple(c.command_options.goal_rot.resampling_time_range) == (6.0, 8.0)


@pytest.mark.parametrize("hand", ["wuji", "sharpa", "allegro"])
def test_v6_rotation_disables_external_linear_force_only(hand):
    from registry import _load_tasks, get_task_config

    _load_tasks()
    c = get_task_config(
        task_name="in_fingers_rotate",
        modifiers={"robot": hand, "obj": "primitives", "sensors": "none", "stage": 1},
        config_override_path=f"conf/experiments/in_fingers_rotate_{hand}_fresh_v141_hora_gain4_noforce_fast_v6.yaml",
    )
    pulse = c.event_options.object_pulse
    assert pulse.profile == "probe"
    assert pulse.strength == 0.25
    assert pulse.linear_force_scale == 0.0
    assert pulse.clean_fraction == 0.25
    assert c.reward_options.rotation_reward.name == "signed_quaternion_progress_reward"
    assert c.reward_options.rotation_reward.weight == 4.0


@pytest.mark.parametrize("hand", ["wuji", "sharpa", "allegro"])
def test_v7_lift_has_six_active_extrinsic_terms(hand):
    from registry import _load_tasks, get_task_config

    _load_tasks()
    c = get_task_config(
        task_name="grasp_lift",
        modifiers={"robot": hand, "obj": "primitives", "sensors": "none", "stage": 1},
        config_override_path=f"conf/experiments/grasp_lift_{hand}_fresh_v141_simple_contact_fast_v7.yaml",
    )
    active = {
        name: term
        for name, term in c.reward_options
        if hasattr(term, "weight") and term.weight != 0
    }
    assert set(active) == {
        "contact_proximity",
        "lift",
        "goal_tracking",
        "natural_motion",
        "action_rate_penalty",
        "dropped",
    }
    assert active["lift"].name == "sustained_plane_lift_reward"
    assert active["contact_proximity"].name == "surface_distance_reward"


@pytest.mark.parametrize(
    "task,config",
    [
        ("in_fingers_rotate", "conf/experiments/in_fingers_rotate_wuji_fresh_v141_hora_gain4_noforce_fast_v6.yaml"),
        ("in_hand_repose", "conf/experiments/in_hand_repose_wuji_fresh_v141_limits_gentle025_fast_v6.yaml"),
        ("grasp_lift", "conf/experiments/grasp_lift_wuji_fresh_v141_simple_contact_fast_v7.yaml"),
    ],
)
def test_fresh_runner_enables_normalized_rnd_for_every_task(task, config):
    from registry import _load_tasks, get_task_config

    _load_tasks()
    c = get_task_config(
        task_name=task,
        modifiers={"robot": "wuji", "obj": "primitives", "sensors": "none", "stage": 1, "rnd": 100.0},
        config_override_path=config,
    )
    rnd = c.runner_options.algorithm.rnd_cfg
    assert rnd.weight == 100.0
    assert rnd.learning_rate == 1e-3
    assert rnd.num_outputs == 8
    assert rnd.state_normalization is True
    assert rnd.reward_normalization is True


def test_v7_allegro_repose_tightens_only_reset_jitter():
    from registry import _load_tasks, get_task_config

    _load_tasks()
    c = get_task_config(
        task_name="in_hand_repose",
        modifiers={"robot": "allegro", "obj": "primitives", "sensors": "none", "stage": 1},
        config_override_path="conf/experiments/in_hand_repose_allegro_fresh_v141_limits_gentle025_rnd_tightreset_fast_v7.yaml",
    )
    reset = c.event_options.place_obj
    assert tuple(reset.range_x) == (-0.001, 0.001)
    assert tuple(reset.range_y) == (-0.001, 0.001)
    assert c.event_options.object_pulse.probe_wrench_scale == 0.25
    assert tuple(c.command_options.goal_rot.resampling_time_range) == (6.0, 8.0)


def test_v7_allegro_repose_repairs_fragile_variant_center():
    from registry import _load_tasks, get_task_config

    _load_tasks()
    c = get_task_config(
        task_name="in_hand_repose",
        modifiers={"robot": "allegro", "obj": "primitives", "sensors": "none", "stage": 1},
        config_override_path="conf/experiments/in_hand_repose_allegro_fresh_v141_limits_gentle025_rnd_centerfix_fast_v7.yaml",
    )
    reset = c.event_options.place_obj
    assert tuple(reset.variant_centers[9]) == (-0.015, -0.015)
    assert tuple(reset.range_x) == (-0.001, 0.001)
    assert tuple(reset.range_y) == (-0.001, 0.001)
