from types import SimpleNamespace as NS

import torch

from tasks.grasp_lift.custom_terms import NaturalLiftMotionPenalty


def fixture():
    r = object.__new__(NaturalLiftMotionPenalty)
    r._env = NS(dt=0.025)
    vel = torch.zeros(2, 3)
    ang = torch.zeros_like(vel)
    pos = torch.tensor([[0., 0., .56], [0., 0., .56]])
    obj_vel = torch.zeros_like(vel)
    quat = torch.tensor([[2**-.5, 0., 2**-.5, 0.]]).repeat(2, 1)
    r._robot = NS(get_vel=lambda: vel, get_ang=lambda: ang,
                  get_pos=lambda: pos + torch.tensor([0., 0., .05]), get_quat=lambda: quat,
                  get_link=lambda name: NS(get_pos=lambda: pos))
    r._obj = NS(get_pos=lambda: pos, get_vel=lambda: obj_vel, get_ang=lambda: ang,
                get_AABB=lambda: torch.stack([pos, pos], dim=1))
    r._links = ['tip']
    r._palm_normal = torch.tensor([1.0, 0.0, 0.0])
    r._prev_vel = vel.clone()
    r._prev_ang = ang.clone()
    r._init_z = torch.full((2,), .5)
    r._lifted = torch.zeros(2, dtype=torch.bool)
    return r, vel, obj_vel, pos


def test_rigid_steady_carry_has_no_motion_cost():
    r, vel, obj_vel, _ = fixture()
    vel[:, 0] = .05
    obj_vel.copy_(vel)
    r._prev_vel.copy_(vel)
    torch.testing.assert_close(r.compute(), torch.zeros(2), atol=1e-6, rtol=0)


def test_abrupt_motion_and_slip_cost_more_but_are_bounded():
    r, vel, obj_vel, _ = fixture()
    vel[0, 0] = 1.
    obj_vel[1, 0] = 10.
    cost = r.compute()
    assert (cost > 0).all() and (cost <= .25).all()


def test_partial_reset_does_not_charge_teleport_velocity_or_clear_other_env():
    r, vel, obj_vel, _ = fixture()
    vel[:, 0] = .05
    obj_vel.copy_(vel)
    r.reset(torch.tensor([0]))
    torch.testing.assert_close(r._prev_vel[1], torch.zeros(3))
    cost = r.compute()
    assert cost[0] < 1e-6 and cost[1] > 0


def test_fast_near_table_approach_costs_more_than_gentle_approach():
    r, vel, _, pos = fixture()
    pos[:, 2] = .5
    vel[:, 2] = torch.tensor([-.04, -.12])
    r._prev_vel.copy_(vel)
    cost = r.compute()
    assert cost[0] < 1e-6 and cost[1] > 0
