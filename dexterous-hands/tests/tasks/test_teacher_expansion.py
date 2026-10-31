from types import SimpleNamespace
import torch
from eden.utils.geom import quat_error_magnitude


def test_repose_goals_anchor_to_object_and_isolate_resets(monkeypatch):
    import genesis as gs

    monkeypatch.setattr(gs, "tc_int", torch.int32, raising=False)
    monkeypatch.setattr(gs, "device", torch.device("cpu"), raising=False)
    from tasks.in_hand_repose.custom_terms import ReachableSettledTargetRotationCommand

    c = object.__new__(ReachableSettledTargetRotationCommand)
    c._env = SimpleNamespace(num_envs=3, device="cpu")
    live = torch.tensor([[0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    c.object = SimpleNamespace(get_quat=lambda: live)
    c.quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(3, 1)
    c.rotation = torch.zeros(3, 6)
    c._sync_goal_vis_orientation = lambda ids: None
    c._dwell = torch.ones(3)
    c._awarded = torch.ones(3, dtype=torch.bool)
    untouched = c.quat[1].clone()
    for _ in range(50):
        c._resample_command(torch.tensor([0, 2]))
        error = quat_error_magnitude(c.quat[[0, 2]], live[[0, 2]])
        assert (error >= 0.25 - 1e-5).all() and (error <= 0.60 + 1e-5).all()
        torch.testing.assert_close(c.quat[1], untouched)
        assert c._dwell.tolist() == [0, 1, 0]
        assert c._awarded.tolist() == [False, True, False]


def test_lift_evaluation_does_not_count_resting_on_table():
    from teacher_evaluation import EpisodeCollector

    pos = torch.tensor([[0.0, 0.0, 0.5], [0.0, 0.0, 0.5]])
    robot = SimpleNamespace(
        get_dofs_pos=lambda **kw: torch.zeros(2, 2),
        get_dofs_vel=lambda **kw: torch.zeros(2, 2),
        get_dofs_control_force=lambda **kw: torch.zeros(2, 2),
        get_dofs_force_range=lambda **kw: (-torch.ones(2, 2), torch.ones(2, 2)),
        get_link=lambda name: SimpleNamespace(get_quat=lambda: torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(2, 1)),
        options=SimpleNamespace(metadata=SimpleNamespace(palm_link="palm", palm_normal=(0.0, 0.0, -1.0))),
    )
    obj = SimpleNamespace(
        get_pos=lambda: pos,
        get_quat=lambda: torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(2, 1),
        get_AABB=lambda: torch.stack((pos - 0.04, pos + 0.04), dim=1),
        get_vel=lambda: torch.zeros(2, 3),
        get_ang=lambda: torch.zeros(2, 3),
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
            active_terms=["goal_pos"],
            get_term=lambda name: SimpleNamespace(command=torch.tensor([[0.0, 0.0, 0.5], [0.0, 0.0, 0.7]])),
        ),
        reset_buf=torch.zeros(2, dtype=torch.bool),
        reset_terminated=torch.zeros(2, dtype=torch.bool),
    )
    collector = EpisodeCollector(env)
    pos[1, 2] = 0.7
    for _ in range(4):
        collector(env)
    assert collector.successes.tolist() == [0, 0]
    for _ in range(8):
        collector(env)
    env.reset_buf[:] = True
    collector(env)
    report = collector.summary()
    assert report["retention"] == 1.0
    assert report["settled_goal_fraction"] == 0.5
    assert report["lifted_fraction"] == 0.5
    assert collector.successes.tolist() == [0, 1]


def test_root_limiter_and_partial_reset(monkeypatch):
    from lift_teacher import SmoothRootPoseController
    from shared_terms import RootPoseController

    c = object.__new__(SmoothRootPoseController)
    c._env = SimpleNamespace(num_envs=2, device="cpu", dt=0.025)
    q = torch.zeros(2, 6)
    c._entity = SimpleNamespace(get_dofs_pos=lambda **kw: q.clone())
    c._pos_min = torch.full((3,), -0.2)
    c._pos_max = torch.full((3,), 0.2)
    c._pos_center = torch.zeros(3)
    c._pos_half_extent = torch.full((3,), 0.2)
    c._euler_min_deg = torch.full((3,), -45.0)
    c._euler_max_deg = torch.full((3,), 45.0)
    c._euler_center_deg = torch.zeros(3)
    c._euler_half_extent_deg = torch.full((3,), 45.0)
    c._raw_action = torch.zeros(2, 6)
    c._prev_action = torch.zeros(2, 6)
    c._processed_action = q.clone()
    monkeypatch.setattr(RootPoseController, "build", lambda self: None)
    c.build()
    for i in range(80):
        old = c.target_velocity.clone()
        c.compute(torch.full((2, 6), 10.0 if i < 40 else -10.0))
        assert (c.target_velocity[:, :3].abs() <= 0.15 + 1e-5).all()
        assert (c.target_velocity[:, 3:].abs() <= 0.8 + 1e-5).all()
        assert ((c.target_velocity[:, :3] - old[:, :3]).abs() <= 0.025 + 1e-5).all()
        assert ((c.target_velocity[:, 3:] - old[:, 3:]).abs() <= 0.2 + 1e-5).all()
    other = c.target[1].clone()
    c.reset(torch.tensor([0]))
    torch.testing.assert_close(c.target[0], q[0])
    torch.testing.assert_close(c.target[1], other)
    assert not c.target_velocity[0].any()
