from types import SimpleNamespace

import torch

from tasks.grasp_lift.custom_terms import PlaneClearanceReward, SustainedLiftReward, SustainedPlaneLiftReward


def make_reward():
    reward = object.__new__(SustainedLiftReward)
    pos = torch.tensor([[0.0, 0.0, 0.5], [0.0, 0.0, 0.5]])
    reward._obj = SimpleNamespace(get_pos=lambda: pos, get_AABB=lambda: torch.stack([pos, pos], dim=1))
    reward._init_z = pos[:, 2].clone()
    reward._lifted = torch.zeros(2, dtype=torch.bool)
    reward.lift_threshold = 0.05
    reward.lift_scale = 1.0
    reward.lift_bonus = 10.0
    return reward, pos


def test_sustained_height_survives_crossing_and_bonus_pays_once():
    reward, pos = make_reward()
    pos[:, 2] = 0.56
    torch.testing.assert_close(reward.compute(), torch.full((2,), 11.0))
    for _ in range(800):
        torch.testing.assert_close(reward.compute(), torch.ones(2))
    pos[:, 2] = 0.8
    torch.testing.assert_close(reward.compute(), torch.ones(2))


def test_putting_down_removes_height_credit_without_rearming_bonus():
    reward, pos = make_reward()
    pos[:, 2] = 0.56
    reward.compute()
    pos[:, 2] = 0.49
    torch.testing.assert_close(reward.compute(), torch.zeros(2))
    pos[:, 2] = 0.56
    torch.testing.assert_close(reward.compute(), torch.ones(2))


def test_partial_reset_rebases_height_and_preserves_other_episode():
    reward, pos = make_reward()
    pos[:, 2] = 0.56
    reward.compute()
    reward.reset(torch.tensor([0]))
    torch.testing.assert_close(reward.compute(), torch.tensor([0.0, 1.0]))
    pos[0, 2] += 0.06
    torch.testing.assert_close(reward.compute(), torch.tensor([11.0, 1.0]))


def test_full_lift_beats_hovering_below_threshold_over_twenty_seconds():
    reward, pos = make_reward()
    pos[:, 2] = torch.tensor([0.549, 0.551])
    total = torch.zeros(2)
    for _ in range(800):
        total += reward.compute() * 0.025
    assert total[1] > total[0]


def test_tilting_the_object_does_not_count_as_lifting():
    """A 10 cm cube tipped onto an edge raises its centre 2 cm; its lowest point stays on the table."""
    reward, pos = make_reward()
    half = torch.full((2, 3), 0.05)
    reward._obj = SimpleNamespace(get_pos=lambda: pos, get_AABB=lambda: torch.stack([pos - half, pos + half], dim=1))
    reward._init_z = (pos[:, 2] - half[:, 2]).clone()
    pos[:, 2] += 0.0207
    half[:, 2] += 0.0207
    torch.testing.assert_close(reward.compute(), torch.zeros(2))
    assert not reward._lifted.any()
    half[:, 2] -= 0.0207
    pos[:, 2] += 0.06
    torch.testing.assert_close(reward.compute(), torch.full((2,), 11.0))


def test_plane_clearance_is_zero_for_a_tip_and_bounded_for_a_lift():
    term = object.__new__(PlaneClearanceReward)
    pos = torch.tensor([[0.0, 0.0, 0.55], [0.0, 0.0, 0.55]])
    half_z = torch.tensor([0.05, 0.05])
    term._obj = SimpleNamespace(
        get_AABB=lambda: torch.stack(
            [pos - torch.stack([torch.zeros_like(half_z), torch.zeros_like(half_z), half_z], dim=-1),
             pos + torch.stack([torch.zeros_like(half_z), torch.zeros_like(half_z), half_z], dim=-1)],
            dim=1,
        )
    )
    term.plane_z = 0.5
    term.lift_threshold = 0.05
    term.lift_scale = 1.0
    torch.testing.assert_close(term.compute(), torch.zeros(2))
    pos[0, 2] += 0.0207  # centre rises while a lower edge stays on the table
    half_z[0] += 0.0207
    pos[1, 2] += 0.30  # complete object clears the table
    torch.testing.assert_close(term.compute(), torch.tensor([0.0, 1.0]))


def test_sustained_plane_lift_only_pays_after_complete_clearance():
    term = object.__new__(SustainedPlaneLiftReward)
    bottom = torch.tensor([0.5, 0.525])
    term._obj = SimpleNamespace(
        get_AABB=lambda: torch.stack(
            [torch.stack([torch.zeros_like(bottom), torch.zeros_like(bottom), bottom], -1),
             torch.stack([torch.zeros_like(bottom), torch.zeros_like(bottom), bottom + 0.1], -1)],
            1,
        )
    )
    term.plane_z = 0.5
    term.threshold = 0.05
    term.lift_bonus = 10.0
    term._lifted = torch.zeros(2, dtype=torch.bool)
    torch.testing.assert_close(term.compute(), torch.tensor([0.0, 0.5]))
    bottom[:] = torch.tensor([0.499, 0.56])
    torch.testing.assert_close(term.compute(), torch.tensor([0.0, 11.0]))
    torch.testing.assert_close(term.compute(), torch.tensor([0.0, 1.0]))
