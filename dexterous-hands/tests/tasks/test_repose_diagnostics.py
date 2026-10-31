import pytest
import torch
import random
import numpy as np

from teacher_evaluation import repose_tracking_summary, preserve_evaluation_rng


@pytest.mark.parametrize("raise_error", [False, True])
def test_model_initialization_does_not_change_reset_rng(raise_error):
    def seed():
        random.seed(43)
        np.random.seed(43)
        torch.manual_seed(43)

    def draws():
        return random.random(), np.random.random(), torch.rand(3)

    seed()
    expected = draws()
    seed()
    try:
        with preserve_evaluation_rng():
            draws()
            torch.nn.Linear(12, 24)
            if raise_error:
                raise RuntimeError("load failure")
    except RuntimeError:
        assert raise_error
    actual = draws()
    assert actual[:2] == expected[:2]
    torch.testing.assert_close(actual[2], expected[2])


def test_repose_diagnostics_exclude_automatic_reset_and_distinguish_speed():
    error = torch.tensor([[0.05, 0.2], [0.04, 99.0]])
    omega = torch.zeros(2, 2, 3)
    omega[0, 0, 0] = 0.5
    live = torch.tensor([[True, True], [True, False]])
    result = repose_tracking_summary(error, omega, live)
    assert result['mean_orientation_error'] == pytest.approx(0.29 / 3)
    assert result['orientation_in_tolerance_fraction'] == pytest.approx(2 / 3)
    assert result['orientation_and_speed_in_tolerance_fraction'] == pytest.approx(1 / 3)
    assert result['speed_blocked_fraction_when_orientation_reached'] == 0.5


def test_repose_diagnostics_do_not_claim_speed_failure_when_goal_never_reached():
    result = repose_tracking_summary(torch.ones(2, 3), torch.zeros(2, 3, 3), torch.ones(2, 3, dtype=torch.bool))
    assert result['orientation_in_tolerance_fraction'] == 0
    assert result['speed_blocked_fraction_when_orientation_reached'] is None
