from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from tasks.grasp_lift.custom_terms import FingertipWrapReward, GoalTrackingReward, SettledCarryBonus

CONFIGS = Path(__file__).resolve().parents[2] / "conf" / "experiments"
GOAL = torch.tensor([[0.02, -0.01, 0.65], [0.02, -0.01, 0.65]])


def make_term(cls, **fields):
    term = object.__new__(cls)
    pos = torch.tensor([[0.0, 0.0, 0.5], [0.0, 0.0, 0.5]])
    vel = torch.zeros(2, 3)
    ang = torch.zeros(2, 3)
    term._obj = SimpleNamespace(
        get_pos=lambda: pos, get_vel=lambda: vel, get_ang=lambda: ang, get_AABB=lambda: torch.stack([pos, pos], dim=1)
    )
    term._cmd = SimpleNamespace(command=GOAL)
    term._init_z = pos[:, 2].clone()
    term._lifted = torch.zeros(2, dtype=torch.bool)
    term.lift_threshold = 0.05
    for name, value in fields.items():
        setattr(term, name, value)
    return term, pos, vel, ang


def test_settled_carry_pays_only_the_evaluation_condition():
    bonus, pos, vel, ang = make_term(
        SettledCarryBonus, tolerance=0.03, max_linear_speed=0.1, max_angular_speed=0.5
    )
    torch.testing.assert_close(bonus.compute(), torch.zeros(2))
    pos.copy_(GOAL)
    torch.testing.assert_close(bonus.compute(), torch.ones(2))
    pos[0, 0] += 0.04
    torch.testing.assert_close(bonus.compute(), torch.tensor([0.0, 1.0]))
    pos.copy_(GOAL)
    vel[1, 2] = 0.2
    torch.testing.assert_close(bonus.compute(), torch.tensor([1.0, 0.0]))
    vel.zero_()
    ang[0, 1] = 1.0
    torch.testing.assert_close(bonus.compute(), torch.tensor([0.0, 1.0]))


def test_settled_carry_ignores_the_goal_footprint_on_the_table():
    bonus, pos, *_ = make_term(SettledCarryBonus, tolerance=0.30, max_linear_speed=0.1, max_angular_speed=0.5)
    pos[:, :2] = GOAL[:, :2]
    torch.testing.assert_close(bonus.compute(), torch.zeros(2))


def test_goal_tracking_is_gaussian_in_distance_and_gated_on_lift():
    tracking, pos, *_ = make_term(GoalTrackingReward, sigma=0.05)
    torch.testing.assert_close(tracking.compute(), torch.zeros(2))
    pos.copy_(GOAL)
    torch.testing.assert_close(tracking.compute(), torch.ones(2))
    pos[1, 0] += 0.05
    torch.testing.assert_close(tracking.compute(), torch.tensor([1.0, torch.e**-1]))
    pos[:, 2] = 0.54
    torch.testing.assert_close(tracking.compute(), torch.zeros(2))


@pytest.mark.parametrize("stem", ["wuji_objaverse_m1", "sharpa_objaverse_m1", "allegro_objaverse_m1", "wuji_goal_l5", "sharpa_goal_l5", "allegro_goal_l5"])
def test_goal_configs_make_the_settled_goal_worth_more_than_hovering(stem):
    rewards = yaml.safe_load((CONFIGS / f"grasp_lift_{stem}.yaml").read_text())["reward_options"]
    assert rewards["goal_tracking"]["name"] == "goal_tracking_reward"
    assert rewards["settled_carry"]["name"] == "settled_carry_bonus"
    # Sustained lift saturates at 1.0/step anywhere above the threshold; L3/L4/P1 hovered there
    # with 0-4% settled goals. The goal terms must dominate that plateau.
    assert rewards["goal_tracking"]["weight"] + rewards["settled_carry"]["weight"] > 1.0


def test_goal_band_requires_a_full_lift_of_the_largest_object():
    from entities.objects import OBJECTS_10CM_16
    from tasks.grasp_lift.config import LIFT_THRESHOLD, TABLE_Z, GraspLiftConfig, hand_start_height, object_height

    low, high = GraspLiftConfig.model_fields["command_options"].default.goal_pos.z_range
    tallest = max(object_height(v) for v in OBJECTS_10CM_16)
    # The object centre starts half its height above the table; the goal must sit well above the
    # lift threshold measured from there.
    assert low - TABLE_Z > tallest / 2 + LIFT_THRESHOLD + 0.05
    assert high > low
    assert hand_start_height(GraspLiftConfig()) >= tallest


def test_goal_tracking_far_scale_pays_inside_the_dead_zone():
    fine, pos, _, _ = make_term(GoalTrackingReward, sigma=0.05, sigma_far=None)
    coarse, pos2, _, _ = make_term(GoalTrackingReward, sigma=0.05, sigma_far=0.25)
    for p in (pos, pos2):
        p[:, 2] = 0.56  # lifted 6 cm, still 9 cm below the goal band's centre
        p[:, 2] = GOAL[0, 2] - 0.09
    assert fine.compute().max() < 0.05
    assert 0.3 < coarse.compute().min() < 0.6
    pos2.copy_(GOAL)
    torch.testing.assert_close(coarse.compute(), torch.ones(2))


def test_fingertip_wrap_pays_antipodal_touching_pairs_only():
    term = object.__new__(FingertipWrapReward)
    term.sigma = 0.03
    obj = torch.tensor([[0.0, 0.0, 0.55]])
    term._obj = SimpleNamespace(get_pos=lambda: obj)
    term._env = SimpleNamespace(num_envs=1, device="cpu")

    def probe(tip, nearest):
        tip_t, near_t = torch.tensor([tip]), torch.tensor([[nearest]])
        return SimpleNamespace(read=lambda: None, get_pos=lambda: tip_t, sensor=SimpleNamespace(nearest_points=near_t))

    touching = lambda x, y, z: probe([x, y, z], [x, y, z])
    # Three tips lying on one face (+x side), as in the L10 videos: no opposition, no reward.
    term._sensors = [touching(0.05, 0.0, 0.52), touching(0.05, 0.02, 0.53), touching(0.05, -0.02, 0.51)]
    term._pairs = torch.triu_indices(3, 3, offset=1)
    torch.testing.assert_close(term.compute(), torch.zeros(1))
    # Thumb on -x, finger on +x, both touching: full credit.
    term._sensors = [touching(-0.05, 0.0, 0.55), touching(0.05, 0.0, 0.55), touching(0.0, 0.0, 0.60)]
    torch.testing.assert_close(term.compute(), torch.ones(1))
    # Same pair but the +x tip is 15 cm off the surface: nothing.
    term._sensors = [touching(-0.05, 0.0, 0.55), probe([0.20, 0.0, 0.55], [0.05, 0.0, 0.55]), touching(0.0, 0.0, 0.60)]
    assert term.compute().item() < 1e-6
    # Perpendicular faces (+x and +y): half credit at most.
    term._sensors = [touching(0.05, 0.0, 0.55), touching(0.0, 0.05, 0.55)]
    term._pairs = torch.triu_indices(2, 2, offset=1)
    torch.testing.assert_close(term.compute(), torch.zeros(1))
