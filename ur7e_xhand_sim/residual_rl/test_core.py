"""CPU-only regression tests for the residual-controller task contract.

Run from the repository root:
    .venv/bin/python -m unittest ur7e_xhand_sim.residual_rl.test_core -v

These tests establish timing/action/reward contracts, not learned grasp quality.
"""
import ast
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

from .core import (
    DEMO, RateConfig, Trajectory, gains_from_action, hold_success, interval_overlap,
    rate_limit_gains, reward_rate,
)


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


class TestDemonstrationTimeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.trajectory = Trajectory()
        # Independent source-row read: assertions do not reuse Trajectory.targets.
        cls.rows = []
        with (DEMO / 'control.jsonl').open() as stream:
            for line in stream:
                row = json.loads(line)
                cls.rows.append((row['before']['sim_time_s'],
                                 row['after']['sim_time_s'],
                                 row['phase'], row['sent_target_rad']))

    def test_preserves_original_duration_and_recorded_waypoints(self):
        self.assertEqual(len(self.rows), 1275)
        self.assertAlmostEqual(self.trajectory.duration_s, 31.875)
        times = np.asarray([r[0] for r in self.rows])
        expected = np.asarray([r[3] for r in self.rows])
        np.testing.assert_allclose(self.trajectory.target(times), expected,
                                   rtol=0, atol=1e-12)
        np.testing.assert_allclose(self.trajectory.target(31.875), expected[-1],
                                   rtol=0, atol=1e-12)

    def test_phase_boundaries_use_command_start_time(self):
        boundaries = [(0.625, 'settle', 'preshape'),
                      (5.25, 'preshape', 'approach'),
                      (13.15, 'approach', 'align'),
                      (16.925, 'align', 'close'),
                      (19.875, 'close', 'preload'),
                      (21.375, 'preload', 'test_lift'),
                      (24.125, 'test_lift', 'lift'),
                      (28.875, 'lift', 'hold')]
        for time_s, before, after in boundaries:
            with self.subTest(time_s=time_s):
                self.assertEqual(self.trajectory.phase(time_s - 1e-8), before)
                self.assertEqual(self.trajectory.phase(time_s + 1e-8), after)
        self.assertEqual(self.trajectory.phase(31.875), 'hold')

    def test_15hz_clock_does_not_consume_one_40hz_row_per_action(self):
        # At 20 seconds the command must address the 20-second source bracket,
        # regardless of whether a policy has made 300 or 800 decisions.
        for time_s in [19.866666666666667, 20.0, 21.4, 24.133333333333333,
                       28.866666666666667, 31.866666666666667]:
            index = max(i for i, r in enumerate(self.rows) if r[0] <= time_s)
            current = self.rows[index]
            if index + 1 < len(self.rows):
                following = self.rows[index + 1]
                weight = (time_s - current[0]) / (following[0] - current[0])
                expected = ((1 - weight) * np.asarray(current[3])
                            + weight * np.asarray(following[3]))
            else:
                expected = np.asarray(current[3])
            with self.subTest(time_s=time_s):
                np.testing.assert_allclose(self.trajectory.target(time_s),
                                           expected, rtol=0, atol=1e-12)

    def test_last_decision_does_not_extend_source_duration(self):
        # Full 15 Hz decision blocks run past the source end; the terminal
        # command may hold, but the source duration remains 31.875 seconds.
        self.assertAlmostEqual(479 / 15, 31.933333333333334)
        self.assertLess(self.trajectory.duration_s, 479 / 15)
        np.testing.assert_allclose(self.trajectory.target(479 / 15),
                                   self.rows[-1][3], rtol=0, atol=1e-12)
        self.assertAlmostEqual(self.trajectory.duration_s
                               - self.trajectory.phase_starts['hold'], 3.0)


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


class TestGainActionContract(unittest.TestCase):
    def test_policy_range_and_out_of_range_actions_are_bounded(self):
        np.testing.assert_array_equal(gains_from_action([-1, 1]), [1., 2.])
        np.testing.assert_array_equal(gains_from_action([0, 0]), [1.5, 1.5])
        np.testing.assert_array_equal(gains_from_action([-100, 100]), [1., 2.])

    def test_nonfinite_action_is_rejected_before_physics(self):
        for action in ([np.nan, 0], [0, np.inf], [-np.inf, 0]):
            with self.subTest(action=action), self.assertRaises(ValueError):
                gains_from_action(action)

    def test_rate_limit_applies_per_finger_and_per_decision(self):
        previous = np.asarray([1., 2.])
        requested = np.asarray([2., 1.])
        for expected in ([1.25, 1.75], [1.5, 1.5], [1.75, 1.25], [2., 1.]):
            actual = rate_limit_gains(previous, requested)
            self.assertTrue(np.all(np.abs(actual - previous) <= .25 + 1e-12))
            self.assertTrue(np.all((actual >= 1) & (actual <= 2)))
            np.testing.assert_allclose(actual, expected)
            previous = actual

    def test_batched_environments_do_not_share_gain_history(self):
        previous = np.asarray([[1., 2.], [2., 1.]])
        requested = np.asarray([[2., 1.], [1., 2.]])
        np.testing.assert_allclose(rate_limit_gains(previous, requested),
                                   [[1.25, 1.75], [1.75, 1.25]])

    def test_invalid_limiter_inputs_are_rejected(self):
        cases = [([np.nan, 1], [2, 2], .25),
                 ([1, 1], [np.inf, 2], .25),
                 ([1, 1], [2, 2], -.25),
                 ([1, 1], [2, 2], np.inf),
                 ([.5, 1], [2, 2], .25)]
        for previous, requested, delta in cases:
            with self.subTest(case=(previous, requested, delta)):
                with self.assertRaises(ValueError):
                    rate_limit_gains(previous, requested, max_delta=delta)


class TestOutcomeAndReward(unittest.TestCase):
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

    def test_no_contact_or_one_finger_cannot_earn_height_reward(self):
        for load in ([0., 0.], [8., 0.], [0., 8.], [.05, .05]):
            with self.subTest(load=load):
                self.assertLess(float(reward_rate(.10, load, 0)), 0)
                self.assertAlmostEqual(float(reward_rate(.10, load, 0)),
                                       float(reward_rate(0, load, 0)))

    def test_force_and_slip_costs_do_not_reward_excess(self):
        stable = float(reward_rate(.07, [4., 4.], 0))
        self.assertGreater(stable, 0)
        self.assertLess(float(reward_rate(.07, [8., 8.], 0)), stable)
        self.assertLess(float(reward_rate(.07, [4., 4.], .01)), stable)
        self.assertLess(float(reward_rate(0, [4., 4.], 0)), 0)

    def test_batched_reward_matches_independent_scenarios(self):
        height = np.asarray([.07, .08, 0])
        load = np.asarray([[4., 4.], [0., 0.], [6., 6.]])
        slip = np.asarray([0, 0, .02])
        actual = reward_rate(height, load, slip)
        expected = [reward_rate(h, f, s) for h, f, s in zip(height, load, slip)]
        np.testing.assert_allclose(actual, expected)


class TestEpisodeHistoryIsolation(unittest.TestCase):
    """Run actual env methods against CPU fixtures, without importing Genesis.

    Extracting the methods avoids initializing/importing the physical engine;
    their code is executed unchanged, while scene and sensors are test doubles.
    This verifies buffer/reset contracts, not physical state restoration.
    """
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch
        source = Path(__file__).with_name('env.py').read_text()
        tree = ast.parse(source)
        env_class = next(node for node in tree.body
                         if isinstance(node, ast.ClassDef)
                         and node.name == 'ResidualVecEnv')
        names = {'reset', '_obs', '_sample_tactile', 'step_async', 'step_wait'}
        methods = [node for node in env_class.body
                   if isinstance(node, ast.FunctionDef) and node.name in names]
        harness = ast.ClassDef(name='EnvContractHarness', bases=[], keywords=[],
                               body=methods, decorator_list=[])
        module = ast.fix_missing_locations(ast.Module(body=[harness], type_ignores=[]))
        namespace = dict(np=np, torch=torch,
                         cpu=lambda x: x.detach().cpu().numpy(),
                         rate_limit_gains=rate_limit_gains,
                         gains_from_action=gains_from_action,
                         reward_rate=reward_rate, hold_success=hold_success,
                         interval_overlap=interval_overlap)
        exec(compile(module, str(Path(__file__).with_name('env.py')), 'exec'), namespace)
        cls.harness_type = namespace['EnvContractHarness']

    def fixture(self):
        env = self.harness_type()
        env.num_envs = 2
        env.rates = RateConfig()
        env.start_tick, env.end_tick = 5960, 9580
        env.hold_start = 28.875
        env.prefix_checkpoint = object()
        env.reset_calls = []
        env.physics_operations = []
        def restore(checkpoint):
            env.reset_calls.append(checkpoint)
            env.physics_operations.append('restore_checkpoint')
        env.scene = SimpleNamespace(__setstate__=restore,
                                    step=lambda: env.physics_operations.append('advance_physics'))
        env._apply_materials = lambda: env.physics_operations.append('rewrite_materials')
        env._apply_gains = lambda: env.physics_operations.append('rewrite_gains')
        env.indices = list(range(18))
        env.robot = SimpleNamespace(
            control_dofs_position=lambda target, indices: env.physics_operations.append('rewrite_target'),
            get_dofs_position=lambda indices: self.torch.zeros((2, 18)),
            get_dofs_velocity=lambda indices: self.torch.zeros((2, 18)))
        env.trajectory = SimpleNamespace(
            target=lambda t: np.zeros(18), duration_s=31.875,
            phase=lambda t: 'hold' if t >= 28.875 else 'preload')
        env.mid, env.half = np.zeros(18), np.ones(18)
        env.gains = np.full((2, 2), 2., dtype=np.float32)
        env.prefix_history = np.arange(2 * 12 * 5 * 120 * 3, dtype=np.float32).reshape(
            2, 12, 5, 120, 3) / 1000
        env.history = np.full_like(env.prefix_history, -12345)
        env.sensors = [SimpleNamespace(read=lambda f=f: SimpleNamespace(
            force=self.torch.full((2, 120, 3), 100. + f))) for f in range(5)]
        env.tactile_noise_std = 0.
        env.episode_counter = 4
        env._pending_action = np.ones((2, 2))
        env._metrics = lambda: (np.full(2, .08), np.ones((2, 5)),
                               np.full(2, .001), np.zeros(2, dtype=bool))
        env.mass, env.mu = np.full(2, .2), np.full(2, .8)
        return env

    def test_reset_discards_previous_episode_and_returns_unaliased_history(self):
        env = self.fixture()
        expected = env.prefix_history.copy()
        observation = env.reset()
        self.assertEqual(env.reset_calls, [env.prefix_checkpoint])
        self.assertEqual(env.tick, env.start_tick)
        self.assertIsNone(env._pending_action)
        np.testing.assert_array_equal(env.gains, np.ones((2, 2)))
        np.testing.assert_array_equal(observation['tactile'], expected)
        self.assertFalse(np.shares_memory(env.history, env.prefix_history))
        observation['tactile'][:] = -900
        np.testing.assert_array_equal(env.history, expected)
        np.testing.assert_array_equal(env.prefix_history, expected)

    def test_checkpoint_reset_does_not_rewrite_restored_physics_or_advance_time(self):
        env = self.fixture()
        env.reset()
        # Mass/inertia, PD gains, targets and solver state belong to the
        # checkpoint. Reapplying them can disturb an exact continuation.
        self.assertEqual(env.physics_operations, ['restore_checkpoint'])
        self.assertEqual(env.tick, env.start_tick)
        self.assertEqual(env.elapsed, 0.)
        env.gains[:] = 2.
        env.tick = env.end_tick
        env.reset()
        self.assertEqual(env.physics_operations,
                         ['restore_checkpoint', 'restore_checkpoint'])
        self.assertEqual(env.reset_calls, [env.prefix_checkpoint, env.prefix_checkpoint])
        np.testing.assert_array_equal(env.gains, np.ones((2, 2)))
        self.assertEqual(env.tick, env.start_tick)

    def test_new_samples_do_not_modify_cached_prefix_for_next_reset(self):
        env = self.fixture()
        expected = env.prefix_history.copy()
        env.reset()
        for _ in range(4):
            env._sample_tactile()
        np.testing.assert_array_equal(env.history[:, :8], expected[:, 4:])
        self.assertTrue(np.all(env.history[:, 8:] >= 100.))
        np.testing.assert_array_equal(env.prefix_history, expected)
        env.reset()
        np.testing.assert_array_equal(env.history, expected)

    def test_auto_reset_keeps_terminal_history_separate_from_new_episode(self):
        env = self.fixture()
        expected = env.prefix_history.copy()
        env.reset()
        env.tick = env.end_tick - env.rates.action_stride
        env.step_async(np.full((2, 2), -1, dtype=np.float32))
        observation, _, done, infos = env.step_wait()
        self.assertTrue(done.all())
        self.assertEqual(env.physics_operations.count('advance_physics'), 20)
        self.assertEqual(env.physics_operations.count('rewrite_materials'), 0)
        self.assertEqual(env.physics_operations.count('rewrite_gains'), 1)
        self.assertEqual(env.physics_operations.count('rewrite_target'), 1)
        self.assertEqual(env.physics_operations[-1], 'restore_checkpoint')
        np.testing.assert_array_equal(observation['tactile'], expected)
        for i, info in enumerate(infos):
            terminal = info['terminal_observation']['tactile']
            np.testing.assert_array_equal(terminal[:8], expected[i, 4:])
            self.assertTrue(np.all(terminal[8:] >= 100.))
            self.assertFalse(np.shares_memory(terminal, env.history))
            terminal[:] = -500
        np.testing.assert_array_equal(env.history, expected)
        np.testing.assert_array_equal(env.prefix_history, expected)


if __name__ == '__main__':
    unittest.main()
