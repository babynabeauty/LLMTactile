"""Exercise Genesis's actual stateful kernels without building a simulation scene."""

import math
from types import SimpleNamespace

import numpy as np
import torch
from genesis.engine.sensors.base_sensor import SimpleSensor
from genesis.engine.sensors.tactile_shared import ViscoelasticHysteresisMixin

from sensor_compat import install_sensor_reset_fix
from tactile_sensors import imperfection_params


class TransformBase:
    @classmethod
    def _apply_transform(cls, *args, **kwargs):
        pass

    @classmethod
    def reset(cls, *args, **kwargs):
        pass


class Hysteresis(ViscoelasticHysteresisMixin, TransformBase):
    pass


def history(level):
    p = imperfection_params("force", level)
    strength, tau = p.get("hysteresis_strength", 0), p.get("hysteresis_tau", 0.05)
    state = SimpleNamespace(
        has_any_hysteresis=bool(strength),
        viscoelastic_strength_row=torch.tensor([strength]),
        viscoelastic_alpha_row=torch.tensor([math.exp(-0.005 / tau)]),
        viscoelastic_xi=torch.zeros(2, 1),
        viscoelastic_prev_input=torch.zeros(2, 1),
    )
    signal = torch.cat((torch.zeros(20), torch.ones(100), torch.zeros(100)))
    trace = []
    for value in signal:
        sample = torch.full((2, 1), value)
        Hysteresis._apply_transform(state, sample, None, is_measured=True)
        trace.append(sample[0, 0].item())
    return signal.numpy(), np.array(trace), state


def test_loading_hold_unloading_and_partial_reset():
    signal, clean, _ = history("clean")
    _, mild, _ = history("mild")
    _, strong, state = history("strong")
    np.testing.assert_allclose(clean, signal)
    assert strong[20] > mild[20] > clean[20]
    assert abs(strong[119] - 1) < abs(strong[20] - 1)
    assert strong[120] < mild[120] < clean[120]
    state.viscoelastic_xi[:] = 1
    state.viscoelastic_prev_input[:] = 1
    Hysteresis.reset(state, None, torch.tensor([0]))
    assert state.viscoelastic_xi[:, 0].tolist() == [0, 1]
    assert state.viscoelastic_prev_input[:, 0].tolist() == [0, 1]


def drift_trace(level, seed=42):
    p = imperfection_params("force", level)
    sigma = p.get("random_walk", 0)
    state = SimpleNamespace(
        random_walk=torch.full((2, 1), float(sigma)),
        _cur_random_walk=torch.zeros(2, 1),
        has_any_random_walk=bool(sigma),
        has_any_noise=False,
        has_any_bias=False,
        has_any_resolution=False,
    )
    torch.manual_seed(seed)
    trace = []
    for _ in range(200):
        value = torch.zeros(2, 1)
        SimpleSensor._apply_hardware_imperfections(state, value)
        trace.append(value.clone())
    return torch.stack(trace), state


def test_seeded_drift_and_per_environment_reset():
    clean, _ = drift_trace("clean")
    mild, _ = drift_trace("mild")
    strong, state = drift_trace("strong")
    assert not clean.any()
    torch.testing.assert_close(mild * 4, strong)
    torch.testing.assert_close(strong, drift_trace("strong")[0])
    assert not torch.equal(strong, drift_trace("strong", 43)[0])
    install_sensor_reset_fix()
    other = state._cur_random_walk[1].clone()
    SimpleSensor.reset(state, torch.zeros(2, 1), torch.tensor([0]))
    assert not state._cur_random_walk[0].any()
    torch.testing.assert_close(state._cur_random_walk[1], other)
