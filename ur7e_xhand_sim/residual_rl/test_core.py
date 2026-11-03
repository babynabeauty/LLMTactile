"""CPU-only tests for the shared timing, trajectory and hold contracts."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from .core import RateConfig, Trajectory, hold_success, interval_overlap


class TestCadence(unittest.TestCase):
    def test_four_distinct_tactile_samples_per_policy_decision(self):
        rates = RateConfig()
        self.assertEqual(rates.physics_hz, 300)
        self.assertEqual(rates.tactile_hz, 60)
        self.assertEqual(rates.action_hz, 15)
        self.assertEqual(rates.action_stride, 20)
        self.assertEqual(rates.tactile_stride, 5)
        self.assertEqual(rates.samples_per_action, 4)
        sample_ticks = list(range(rates.tactile_stride,
                                  rates.action_stride + 1,
                                  rates.tactile_stride))
        self.assertEqual(sample_ticks, [5, 10, 15, 20])
        self.assertAlmostEqual(rates.physics_dt * 20, rates.action_dt)
        self.assertAlmostEqual(rates.physics_dt * 5, 1 / 60)

    def test_incompatible_rates_are_rejected(self):
        for values in [(0, 60, 15), (300, -60, 15), (300, 60, 0),
                       (301, 60, 15), (300, 60, 16)]:
            with self.subTest(values=values), self.assertRaises(ValueError):
                RateConfig(*values)

    def test_fractional_rate_cannot_produce_float_tick_strides(self):
        for values in [(300.0, 60, 15), (300, 60.0, 15), (300, 60, 15.0)]:
            with self.subTest(values=values), self.assertRaises((ValueError, TypeError)):
                RateConfig(*values)


class TestTrajectoryInputValidation(unittest.TestCase):
    def write_demo(self, directory, times=(0.0, 0.025, 0.05)):
        path = Path(directory) / 'control.jsonl'
        rows = [dict(before={'sim_time_s': t}, after={'sim_time_s': t + .025},
                     sent_target_rad=[float(i), float(2 * i)], phase='hold')
                for i, t in enumerate(times)]
        path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
        return path

    def test_interpolation_and_endpoint_clamping_are_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            trajectory = Trajectory(self.write_demo(directory))
        np.testing.assert_allclose(trajectory.target(.0125), [.5, 1.])
        np.testing.assert_allclose(trajectory.target(-1), [0., 0.])
        np.testing.assert_allclose(trajectory.target(1), [2., 4.])
        self.assertAlmostEqual(trajectory.duration_s, .075)

    def test_nonmonotonic_source_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_demo(directory, (0, .05, .025))
            with self.assertRaises(ValueError):
                Trajectory(path)

    def test_modified_successful_reference_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_demo(directory)
            (Path(directory) / 'reference.json').write_text(json.dumps({
                'files': {'control.jsonl': '0' * 64}}))
            with self.assertRaises(ValueError):
                Trajectory(path)


class TestOutcome(unittest.TestCase):
    def test_final_hold_overlap_preserves_exact_three_seconds(self):
        ticks = np.arange(9579) / 300
        durations = interval_overlap(ticks[:-1], ticks[1:], 28.875, 31.875)
        self.assertAlmostEqual(float(durations.sum()), 3.0, places=12)
        self.assertAlmostEqual(float(interval_overlap(31.873333333333335,
                                                      31.876666666666665,
                                                      28.875, 31.875)),
                               1 / 600, places=12)
        self.assertEqual(float(interval_overlap(31.875, 31.933333333333334,
                                                 28.875, 31.875)), 0)

    def test_overlap_does_not_credit_lift_or_empty_intervals(self):
        for t0, t1 in [(0, 28.875), (31.875, 32), (29, 29)]:
            with self.subTest(interval=(t0, t1)):
                self.assertEqual(float(interval_overlap(t0, t1, 28.875, 31.875)), 0)
        for endpoints in [(30, 29, 28.875, 31.875),
                          (28, 30, 31.875, 28.875),
                          (28, np.inf, 28.875, 31.875)]:
            with self.subTest(endpoints=endpoints), self.assertRaises(ValueError):
                interval_overlap(*endpoints)

    def test_hold_requires_full_duration_and_never_timeout_alone(self):
        self.assertFalse(hold_success(0, .1))
        self.assertFalse(hold_success(2.999, .1))
        self.assertFalse(hold_success(100, 0))
        self.assertTrue(hold_success(3, .05))

    def test_disconnected_success_intervals_cannot_hide_a_drop(self):
        # An episode with 2 s good + a dip + 2 s good cannot be converted to
        # a successful 4 s hold: the interval minimum carries the violation.
        self.assertFalse(hold_success(4, .049))
        self.assertFalse(hold_success(3, .08, fault=True))
        np.testing.assert_array_equal(hold_success([3, 3, 2.5], [.06, .01, .08]),
                                      [True, False, False])


if __name__ == '__main__':
    unittest.main()
