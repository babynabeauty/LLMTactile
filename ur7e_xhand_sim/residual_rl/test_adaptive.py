"""CPU contract checks for adaptive observations, reset, actions and outcomes.

The physical environment methods are executed unchanged through AST fixtures;
Genesis is never imported or initialized. These are not physical-grasp tests.
"""
import ast
from dataclasses import dataclass, asdict
from pathlib import Path
from types import SimpleNamespace
import unittest

import gymnasium as gym
import numpy as np
import torch

from .core import interval_overlap, hold_success
from .adaptive_policy import AdaptiveTactileExtractor


def environment_contract_types():
    directory = Path(__file__).parent
    base_tree = ast.parse((directory / 'env.py').read_text())
    base = next(n for n in base_tree.body if isinstance(n, ast.ClassDef)
                and n.name == 'ResidualVecEnv')
    base.body = [n for n in base.body if isinstance(n, ast.FunctionDef)
                 and n.name in {'reset', '_sample_tactile'}]
    base.bases = []
    adaptive_tree = ast.parse((directory / 'adaptive_env.py').read_text())
    config = next(n for n in adaptive_tree.body if isinstance(n, ast.ClassDef)
                  and n.name == 'AdaptiveConfig')
    adaptive = next(n for n in adaptive_tree.body if isinstance(n, ast.ClassDef)
                    and n.name == 'AdaptiveVecEnv')
    adaptive.body = [n for n in adaptive.body if isinstance(n, ast.FunctionDef)
                     and n.name != '__init__']
    namespace = dict(__name__=__name__, np=np, torch=torch, dataclass=dataclass,
                     asdict=asdict, interval_overlap=interval_overlap,
                     hold_success=hold_success,
                     cpu=lambda x: x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x))
    module = ast.fix_missing_locations(ast.Module(body=[base, config, adaptive], type_ignores=[]))
    exec(compile(module, str(directory / 'adaptive_env.py'), 'exec'), namespace)
    return namespace['AdaptiveConfig'], namespace['AdaptiveVecEnv']


Config, Harness = environment_contract_types()


class TestAdaptiveEnvironmentContract(unittest.TestCase):
    def test_invalid_config_cannot_make_gain_mapping_undefined(self):
        for values in ({'gain_min': 1., 'gain_max': 1.},
                       {'gain_min': 0.}, {'gain_max': .9},
                       {'force_weight': np.nan}, {'gain_rate_limit': np.inf},
                       {'slip_weight': -1.}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                Config(**values)

    def fixture(self, config=None):
        env = Harness()
        env.config = config or Config()
        env.num_envs = 2
        env.start_tick, env.end_tick = 5960, 9580
        env.hold_start = 28.875
        env.indices = list(range(18))
        env.mid, env.half = np.zeros(18), np.ones(18)
        env.sensor_tick = env.start_tick - 60
        env.tick = env.sensor_tick
        env.clock = env.sensor_tick
        env.gains = np.ones((2, 2), np.float32)
        env.history = np.zeros((2, 12, 5, 120, 3), np.float32)
        env.proprio_history = env.prefix_proprio = None
        env.prefix_checkpoint = object()
        env.operations = []
        def restore(checkpoint):
            self.assertIs(checkpoint, env.prefix_checkpoint)
            env.operations.append('restore')
            env.clock = env.start_tick
        def step():
            env.operations.append('physics')
            env.clock += 1
        env.scene = SimpleNamespace(__setstate__=restore, step=step)
        env._apply_materials = lambda: env.operations.append('materials')
        env._apply_gains = lambda: env.operations.append('gains')
        env.robot = SimpleNamespace(
            get_dofs_position=lambda indices: np.full((2, 18), env.clock / 300, np.float32),
            get_dofs_velocity=lambda indices: np.full((2, 18), env.clock / 600, np.float32),
            control_dofs_position=lambda target, indices: env.operations.append('target'))
        env.trajectory = SimpleNamespace(
            target=lambda t: np.full(18, t), duration_s=31.875,
            phase=lambda t: 'hold' if t >= 28.875 else 'preload')
        env.sensors = [SimpleNamespace(read=lambda: SimpleNamespace(
            force=torch.full((2, 120, 3), env.clock / 300))) for _ in range(5)]
        env.tactile_noise_std = 0.
        env.episode_counter = 0
        env.mass, env.mu = np.full(2, .2), np.full(2, .8)
        env._metrics = lambda: (np.full(2, .07), np.ones((2, 5)),
                               np.zeros(2), np.zeros(2, dtype=bool))
        # Mimic the last 12 physically sampled prefix frames. This does not
        # verify Genesis checkpoint internals, only application-clock history.
        for _ in range(12):
            env.clock += 5
            env._sample_tactile()
        env.prefix_history = env.history.copy()
        env.reset()
        env.operations.clear()
        return env

    def test_four_fresh_proprio_and_tactile_samples_share_physical_times(self):
        env = self.fixture()
        old = env.proprio_history.copy()
        env.step_async(np.zeros((2, 1), np.float32))
        obs, _, done, _ = env.step_wait()
        self.assertFalse(done.any())
        self.assertEqual(env.sensor_tick, env.tick)
        self.assertEqual(env.tick, env.clock)
        self.assertEqual(env.operations.count('physics'), 20)
        np.testing.assert_array_equal(obs['state'][:, :8], old[:, 4:])
        expected = (env.start_tick + np.arange(5, 21, 5)) / 300
        np.testing.assert_allclose(obs['state'][0, -4:, 0], expected, atol=2e-6)
        np.testing.assert_array_equal(obs['tactile'][:, -4:, 0, 0, 0], obs['state'][:, -4:, 0])
        self.assertEqual(env.last_interval_metrics['clearance_m'].shape, (4, 2))

    def test_reset_restores_same_proprio_and_tactile_without_physics_writes(self):
        env = self.fixture()
        prefix = env.prefix_proprio.copy()
        env.proprio_history[:] = -900
        env.history[:] = -800
        env.gains[:] = 3
        env.sensor_tick = 9999
        obs = env.reset()
        self.assertEqual(env.operations, ['restore'])
        self.assertEqual(env.sensor_tick, env.start_tick)
        self.assertEqual(env.tick, env.start_tick)
        np.testing.assert_array_equal(obs['state'], prefix)
        np.testing.assert_array_equal(obs['tactile'], env.prefix_history)
        obs['state'][:] = 88
        obs['tactile'][:] = 99
        np.testing.assert_array_equal(env.proprio_history, prefix)
        np.testing.assert_array_equal(env.prefix_proprio, prefix)
        np.testing.assert_array_equal(env.history, env.prefix_history)

    def test_actor_observation_has_no_material_or_physical_label_path(self):
        env = self.fixture()
        before = env._obs()
        env.mass[:] = 100
        env.mu[:] = 5
        env.last_metrics = {'clearance_m': np.full(2, 99), 'slip_speed_m_s': np.full(2, 99)}
        env.hold_contact_ok[:] = False
        after = env._obs()
        self.assertEqual(set(after), {'state', 'tactile'})
        for key in before:
            np.testing.assert_array_equal(before[key], after[key])

    def test_common_gain_bounds_rate_limit_and_inverse(self):
        env = self.fixture()
        for target in (.6, 1., 1.5, 3.):
            a = env.action_for_gain(target)
            reconstructed = env.config.gain_min * np.exp((a + 1) / 2 * np.log(
                env.config.gain_max / env.config.gain_min))
            self.assertAlmostEqual(float(reconstructed), target, places=5)
        for action in (100., 100., -100., -100.):
            previous = env.gains.copy()
            env.step_async(np.full((2, 1), action, np.float32))
            env.step_wait()
            self.assertTrue((abs(env.gains - previous) <= env.config.gain_rate_limit + 1e-6).all())
            self.assertTrue((env.gains >= env.config.gain_min).all())
            self.assertTrue((env.gains <= env.config.gain_max).all())
            np.testing.assert_array_equal(env.gains[:, 0], env.gains[:, 1])

    def test_nonfinite_or_wrong_shape_actions_rejected_before_physics(self):
        env = self.fixture()
        for action in (np.zeros((2, 2)), np.full((2, 1), np.nan), np.full((2, 1), np.inf)):
            with self.subTest(shape=action.shape), self.assertRaises(ValueError):
                env.step_async(action)
        self.assertEqual(env.operations, [])

    def test_zero_contact_cannot_earn_progress_even_while_above_height_threshold(self):
        for force in (0., .05):
            env = self.fixture()
            env._metrics = lambda: (np.full(2, .07), np.full((2, 5), force),
                                   np.zeros(2), np.zeros(2, dtype=bool))
            env.step_async(np.full((2, 1), env.action_for_gain(1.), np.float32))
            _, reward, _, _ = env.step_wait()
            np.testing.assert_array_equal(env.reward_components[:, 0], [0., 0.])
            self.assertTrue((reward <= 0).all())

    def test_single_finger_support_can_earn_progress_without_reinstating_strict_gate(self):
        env = self.fixture()
        env._metrics = lambda: (np.full(2, .07), np.tile([1., 0., 0., 0., 0.], (2, 1)),
                               np.zeros(2), np.zeros(2, dtype=bool))
        env.step_async(np.full((2, 1), env.action_for_gain(1.), np.float32))
        _, reward, _, _ = env.step_wait()
        self.assertTrue((env.reward_components[:, 0] > 0).all())
        self.assertTrue((env.reward_components[:, 3] < 0).all())
        self.assertTrue((reward > 0).all())

    def finish_hold(self, env):
        env.tick = env.clock = env.sensor_tick = 8660
        while True:
            env.step_async(np.full((2, 1), env.action_for_gain(1.), np.float32))
            obs, reward, done, infos = env.step_wait()
            if done.all():
                return obs, reward, infos

    def test_one_sample_support_loss_separates_task_and_strict_success(self):
        env = self.fixture()
        def measurements():
            load = np.ones((2, 5))
            if env.tick == 9500:
                load[0] = [0, 1, 0, 0, 0]
            return np.full(2, .07), load, np.zeros(2), np.zeros(2, dtype=bool)
        env._metrics = measurements
        obs, _, infos = self.finish_hold(env)
        a, b = [i['episode_metrics'] for i in infos]
        self.assertTrue(a['task_success'])
        self.assertFalse(a['strict_success'])
        self.assertFalse(a['success'])
        self.assertEqual(a['termination_reason'], 'contact_discontinuity')
        self.assertTrue(b['task_success'] and b['strict_success'])
        self.assertAlmostEqual(a['hold_duration_s'], 3., places=10)
        self.assertAlmostEqual(a['hold_max_unsupported_duration_s'], 1 / 60)
        self.assertAlmostEqual(a['hold_contact_fraction'], 1 - 1 / 180)
        for info in infos:
            m = info['episode_metrics']
            self.assertAlmostEqual(sum(m['reward_components']), m['return'], places=8)
            terminal = info['terminal_observation']
            self.assertFalse(np.shares_memory(terminal['state'], env.proprio_history))
            self.assertFalse(np.shares_memory(terminal['tactile'], env.history))
        np.testing.assert_array_equal(obs['state'], env.prefix_proprio)
        np.testing.assert_array_equal(obs['tactile'], env.prefix_history)
        self.assertEqual(env.sensor_tick, env.start_tick)

    def test_height_dip_or_fault_fails_both_criteria_even_at_timeout(self):
        env = self.fixture()
        def measurements():
            height, fault = np.full(2, .07), np.zeros(2, dtype=bool)
            if env.tick == 9500:
                height[0], fault[1] = .049, True
            return height, np.ones((2, 5)), np.zeros(2), fault
        env._metrics = measurements
        _, _, infos = self.finish_hold(env)
        for info in infos:
            metrics = info['episode_metrics']
            self.assertFalse(metrics['task_success'])
            self.assertFalse(metrics['strict_success'])


class TestAdaptiveEncoderIsolation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def fixture(self, **kwargs):
        space = gym.spaces.Dict({
            'tactile': gym.spaces.Box(-np.inf, np.inf, (12, 5, 120, 3), np.float32),
            'state': gym.spaces.Box(-np.inf, np.inf, (12, 79), np.float32)})
        torch.manual_seed(7)
        model = AdaptiveTactileExtractor(space, **kwargs).eval()
        observation = {'tactile': torch.randn(2, 12, 5, 120, 3) * .1,
                       'state': torch.randn(2, 12, 79)}
        return model, observation

    @torch.no_grad()
    def test_no_tactile_erases_raw_summary_and_derivative_paths(self):
        model, obs = self.fixture(use_tactile=False)
        changed = {'state': obs['state'].clone(), 'tactile': torch.randn_like(obs['tactile']) * 1000}
        torch.testing.assert_close(model(obs), model(changed), atol=0, rtol=0)
        self.assertEqual(torch.count_nonzero(model.tactile_summaries(changed['tactile'])).item(), 0)

    @torch.no_grad()
    def test_no_tactile_still_uses_past_proprio_history(self):
        model, obs = self.fixture(use_tactile=False)
        changed = {k: v.clone() for k, v in obs.items()}
        changed['state'][:, :6] += 10
        self.assertFalse(torch.allclose(model(obs), model(changed)))

    @torch.no_grad()
    def test_future_samples_cannot_change_earlier_encoded_frames(self):
        model, obs = self.fixture()
        changed = {k: v.clone() for k, v in obs.items()}
        changed['state'][:, 7:] += 100
        changed['tactile'][:, 7:] += 100
        torch.testing.assert_close(model.encode_sequence(obs)[:, :7],
                                   model.encode_sequence(changed)[:, :7], atol=0, rtol=0)

    def test_normal_projection_requires_static_calibration(self):
        with self.assertRaises(ValueError):
            self.fixture(use_shear=False)
        normals = np.zeros((5, 120, 3), np.float32)
        normals[..., 2] = 1
        model, obs = self.fixture(use_shear=False, normal_directions_link=normals)
        changed = {k: v.clone() for k, v in obs.items()}
        changed['tactile'][..., :2] += 100
        with torch.no_grad():
            torch.testing.assert_close(model(obs), model(changed), atol=0, rtol=0)


if __name__ == '__main__':
    unittest.main()
