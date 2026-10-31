"""grasp_pose_proximity_reward is bounded and anchored to the episode's sampled grasp."""

import math
from types import SimpleNamespace

import torch

from tasks.in_fingers_rotate.custom_terms import grasp_pose_proximity_reward


def _env(q, anchor):
    robot = SimpleNamespace(get_dofs_pos=lambda: q)
    return SimpleNamespace(entities={"robot": robot}, _sampled_grasp_joint_pos=anchor, num_envs=q.shape[0], device=q.device)


def test_one_at_grasp_pose_and_e_minus_one_at_tolerance():
    anchor = torch.tensor([[0.1, -0.2, 0.3, 0.0]])
    q = torch.cat([anchor, anchor + 0.5], dim=0)
    reward = grasp_pose_proximity_reward(_env(q, anchor.expand(2, -1)), tolerance=0.5)
    assert reward[0].item() == 1.0
    assert math.isclose(reward[1].item(), math.exp(-1), rel_tol=1e-5)


def test_far_pose_stays_bounded_and_missing_anchor_is_zero():
    q = torch.full((3, 6), 2.0)
    reward = grasp_pose_proximity_reward(_env(q, torch.zeros(3, 6)), tolerance=0.5)
    assert torch.all((reward >= 0) & (reward < 1e-6))
    env = _env(q, None)
    del env._sampled_grasp_joint_pos
    assert torch.equal(grasp_pose_proximity_reward(env), torch.zeros(3))
