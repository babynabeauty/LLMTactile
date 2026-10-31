import math

import pytest
import torch

from object_wrench import WrenchPulseState


@pytest.mark.parametrize("strength,maximum", [(0.25, 2.0), (0.5, 3.0), (1.0, 5.0)])
def test_probe_force_floor_and_off_center_torque(strength, maximum):
    torch.manual_seed(42)
    state = WrenchPulseState(1024, "cpu", strength, 0, profile="probe")
    state.reset(slice(None))
    state.age[:] = 1
    # Physical poke strength must not vanish for lightweight objects.
    state.schedule(slice(None), torch.full((1024,), 0.005), torch.full((1024,), 0.03))
    pushes = state.mode != 2
    force = state.peak_force[pushes].norm(dim=-1)
    assert force.min() >= 1 - 1e-6
    assert force.max() <= maximum + 1e-6
    assert force.max() > maximum * 0.99
    assert (state.lever.norm(dim=-1) <= 0.015 + 1e-6).all()
    torch.testing.assert_close(state.peak_torque[pushes], torch.linalg.cross(state.lever, state.peak_force)[pushes])
    twists = state.mode == 2
    torque = state.peak_torque[twists].norm(dim=-1)
    assert torque.min() >= 0.02 - 1e-6
    assert torque.max() <= 0.02 + 0.06 * strength + 1e-6
    assert not state.peak_force[twists].any()


@pytest.mark.parametrize("control_dt", [0.025, 0.05])
def test_full_probe_impulse_flag_and_arrow_use_actual_force(control_dt):
    state = WrenchPulseState(1, "cpu", 1, 0, profile="probe")
    state.duration[0] = 0.237
    state.peak_force[0, 0] = 5
    impulse = 0.0
    for i in range(100):
        state.step(0.005, i % round(control_dt / 0.005) == 0)
        impulse += float(state.force[0, 0]) * 0.005
        arrows = state.arrow_data(torch.zeros(1, 3), [0])
        if state.force.any():
            assert state.active.item() == 1
            torch.testing.assert_close(arrows[0]["force"], state.force[0])
        else:
            assert not arrows
    assert impulse == pytest.approx(2 * 5 * 0.237 / math.pi, abs=1e-6)
    assert state.active.item() == 0


def test_disabled_probe_stays_clean_and_reset_clears_state():
    state = WrenchPulseState(32, "cpu", 0, 0, profile="probe")
    state.reset(slice(None))
    state.age[:] = 1
    state.schedule(slice(None), torch.ones(32), torch.ones(32))
    state.step(0.005, True)
    assert not state.force.any() and not state.torque.any() and not state.active.any()
    state.force[:] = 1
    state.reset(torch.tensor([0]))
    assert not state.force[0].any() and state.force[1:].all()


def test_lifted_gate_holds_pulses_until_the_object_is_off_the_table():
    from types import SimpleNamespace

    from object_wrench import ObjectWrenchPulse

    torch.manual_seed(0)
    pulse = object.__new__(ObjectWrenchPulse)
    pos = torch.tensor([[0.0, 0.0, 0.52], [0.0, 0.0, 0.52]])
    bottom = pos[:, 2].clone()
    def aabb():
        bounds = torch.zeros(2, 2, 3)
        bounds[:, 0, 2] = bottom
        return bounds
    pulse.obj = SimpleNamespace(get_pos=lambda: pos, get_AABB=aabb, get_mass=lambda: 0.03)
    pulse.only_lifted_above_m = 0.05
    pulse.draw_envs = []
    pulse.radius = torch.full((2,), 0.03)
    pulse._env = SimpleNamespace(num_envs=2, device=torch.device("cpu"))
    pulse.state = WrenchPulseState(2, "cpu", 0.25, 0.0, profile="probe")
    pulse._reset_z = pos[:, 2].clone()
    pulse.reset()
    pulse.state.age[:] = 5.0
    pulse.compute()
    assert not pulse.state.peak_force.any(), "a resting object must not be pulsed"
    pos[1, 2] += 0.08
    pulse.compute()
    assert not pulse.state.duration.any(), "centre rise alone must not make probes eligible"
    bottom[1] += 0.08
    pulse.compute()
    assert not pulse.state.peak_force[0].any() and pulse.state.duration[0] == 0
    assert pulse.state.duration[1] > 0, "the lifted object is scheduled"


def test_gentle_probe_scales_push_and_twist_without_changing_duration():
    torch.manual_seed(42)
    state = WrenchPulseState(1024, 'cpu', .25, 0., profile='probe', probe_wrench_scale=.5)
    state.reset(slice(None))
    state.age[:] = 1
    state.schedule(slice(None), torch.full((1024,), .03), torch.full((1024,), .03))
    pushes = state.mode != 2
    force = state.peak_force[pushes].norm(dim=-1)
    assert force.min() >= .5 - 1e-6 and force.max() <= 1.0 + 1e-6
    twists = state.mode == 2
    torque = state.peak_torque[twists].norm(dim=-1)
    assert torque.min() >= .01 - 1e-6 and torque.max() <= .0175 + 1e-6
    assert state.duration.min() >= .1 and state.duration.max() <= .5


def test_repose_quarter_scale_probe_is_point25_to_point5_newtons():
    torch.manual_seed(42)
    state = WrenchPulseState(1024, 'cpu', .25, 0., profile='probe', probe_wrench_scale=.25)
    state.reset(slice(None))
    state.age[:] = 1
    state.schedule(slice(None), torch.full((1024,), .03), torch.full((1024,), .03))
    pushes = state.mode != 2
    force = state.peak_force[pushes].norm(dim=-1)
    assert force.min() >= .25 - 1e-6 and force.max() <= .5 + 1e-6
    assert state.duration.min() >= .1 and state.duration.max() <= .5


def test_linear_force_can_be_disabled_without_changing_twist_magnitude():
    state = WrenchPulseState(
        4096, 'cpu', .25, 0., profile='probe', linear_force_scale=0.
    )
    state.reset(slice(None))
    state.age[:] = 1
    state.schedule(slice(None), torch.ones(4096), torch.ones(4096))
    assert not state.peak_force.any()
    twists = state.mode == 2
    torque = state.peak_torque[twists].norm(dim=-1)
    assert torque.numel() > 1000
    assert torque.min() >= .02 - 1e-6 and torque.max() <= .035 + 1e-6
