"""CPU contracts for grouped, normalized gSDE exploration; no Genesis or rollouts."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3.common.vec_env import DummyVecEnv

from .v1_bases import TRAINING_DEFAULTS
from .v1_task import V1Config, action_exploration_stds, action_joints
from .v1_train import dry_run, make_model


class ActorEnv(gym.Env):
    def __init__(self, joints):
        self.config = V1Config()
        self.actor_conditioning = 'astra_level_one_hot_v1'
        self.action_joints = np.asarray(joints)
        self.action_space = gym.spaces.Box(-1., 1., (len(joints),), np.float32)
        self.observation_space = gym.spaces.Dict({
            'tactile': gym.spaces.Box(-np.inf, np.inf, (12, 5, 120, 3), np.float32),
            'state': gym.spaces.Box(-np.inf, np.inf, (12, 94), np.float32)})


class ExplorationStdTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_all_twelve_joints_and_pinch_only(self):
        groups = TRAINING_DEFAULTS['action_std_init']
        expected = (.03, .2, .2, .03, .1, .1, .1, .1, .1, .1, .1, .1)
        self.assertEqual(action_exploration_stds(action_joints(['wrap']), groups), expected)
        self.assertEqual(action_exploration_stds(action_joints(['pinch']), groups), expected[:6])

    def test_joint_ids_not_dimension_positions(self):
        self.assertEqual(action_exploration_stds((11, 2, 0, 4, 3, 1),
                         TRAINING_DEFAULTS['action_std_init']), (.1, .2, .03, .1, .03, .2))

    def test_scalar_requires_new_experiment_json(self):
        with self.assertRaisesRegex(ValueError, 'regenerate the experiment JSON'):
            action_exploration_stds(range(12), .03)
        env = DummyVecEnv([lambda: ActorEnv(range(6))])
        try:
            with self.assertRaisesRegex(ValueError, 'regenerate the experiment JSON'):
                make_model(env, dict(TRAINING_DEFAULTS, action_std_init=.03), 1, 'tactile')
        finally:
            env.close()

    def test_group_values_and_joint_ids_are_validated(self):
        for value in (0., -.1, np.inf, np.nan, True, [0.3]):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'finite and positive'):
                action_exploration_stds(range(6), dict(TRAINING_DEFAULTS['action_std_init'], thumb_flexion=value))
        with self.assertRaisesRegex(ValueError, 'must contain'):
            action_exploration_stds(range(6), {'other': .03})
        with self.assertRaisesRegex(ValueError, 'Unknown hand-local action joint'):
            action_exploration_stds((12,), TRAINING_DEFAULTS['action_std_init'])

    def test_fresh_model_columns_and_unit_feature_marginal_std(self):
        for joints in (tuple(range(12)), tuple(range(6)), (11, 2, 0, 4, 3, 1)):
            with self.subTest(joints=joints):
                env = DummyVecEnv([lambda: ActorEnv(joints)])
                try:
                    model = make_model(env, dict(TRAINING_DEFAULTS, n_steps=2, batch_size=2), 1, 'tactile')
                    policy = model.policy
                    expected = torch.tensor(action_exploration_stds(joints, TRAINING_DEFAULTS['action_std_init']))
                    self.assertEqual(tuple(policy.log_std.shape), (64, len(joints)))
                    torch.testing.assert_close(policy.log_std.exp(), expected.expand(64, -1))
                    self.assertEqual(torch.count_nonzero(policy.action_net.weight).item(), 0)
                    self.assertEqual(torch.count_nonzero(policy.action_net.bias).item(), 0)
                    self.assertEqual(model.sde_sample_freq, -1)
                    # Every unit basis feature has the requested marginal noise std. Native SB3
                    # adds epsilon to likelihood variance; keep that existing numerical floor.
                    distribution = policy._get_action_dist_from_latent(torch.eye(64))
                    variance = distribution.distribution.scale.square()
                    torch.testing.assert_close((variance-distribution.epsilon).sqrt(), expected.expand(64, -1))
                    torch.testing.assert_close(distribution.distribution.scale,
                                               (expected.square()+distribution.epsilon).sqrt().expand(64, -1))
                finally:
                    env.close()

    def test_resume_keeps_checkpoint_std_without_reading_scalar_setting(self):
        env = DummyVecEnv([lambda: ActorEnv(range(6))])
        try:
            settings = dict(TRAINING_DEFAULTS, n_steps=2, batch_size=2)
            model = make_model(env, settings, 1, 'tactile')
            with torch.no_grad():
                model.policy.log_std[:, 1].fill_(-1.7)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory)/'model.zip'
                model.save(path)
                restored = make_model(env, dict(settings, action_std_init=.12), 1, 'tactile', resume=path)
            torch.testing.assert_close(restored.policy.log_std, model.policy.log_std, rtol=0, atol=0)
        finally:
            env.close()

    def test_dry_run_reports_std_for_each_action_joint(self):
        experiment = dict(objects=[dict(object_id='cuboid_pinch', base_library='unused')],
                          lanes_per_object=4, training=TRAINING_DEFAULTS)
        library = dict(hand_shape='pinch', object_spec={})
        output = io.StringIO()
        with patch('ur7e_xhand_sim.residual_rl.v1_train.load_library', return_value=(library, [])), \
                contextlib.redirect_stdout(output):
            dry_run(experiment)
        report = json.loads(output.getvalue())
        self.assertEqual(report['action_std_init_by_joint'],
                         {'0': .03, '1': .2, '2': .2, '3': .03, '4': .1, '5': .1})


if __name__ == '__main__':
    unittest.main()
