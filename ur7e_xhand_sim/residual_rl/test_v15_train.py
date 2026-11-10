"""CPU-only verification that V1.5 changes exploration without resetting learning."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from stable_baselines3.common.vec_env import DummyVecEnv

from .test_v1_exploration_std import ActorEnv
from .tactile_provenance import current_tactile_model
from .v1_bases import TRAINING_DEFAULTS
from .v1_train import CPUTrainingPPO, make_model as make_v1_model
from .v15_train import assert_state_equal, load_finger_model, make_model
from .finger_sde_policy import FingerSDEMultiInputPolicy


class MigrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_weights_adam_mean_and_saved_resume_are_preserved(self):
        env = DummyVecEnv([lambda: ActorEnv(range(6))])
        settings = dict(TRAINING_DEFAULTS, n_steps=2, batch_size=2)
        try:
            source = make_v1_model(env, settings, 1, 'tactile')
            with torch.no_grad():
                source.policy.action_net.weight.normal_(std=.01)
                source.policy.action_net.bias.fill_(.02)
                source.policy.log_std[:, 1].add_(.07)
            # Populate genuine Adam moments instead of testing an empty optimizer.
            source.policy.optimizer.zero_grad()
            sum(p.square().sum() for p in source.policy.parameters()).backward()
            source.policy.optimizer.step()
            source.num_timesteps = 122880
            rng = np.random.default_rng(8)
            obs = {k: rng.normal(size=(2, *space.shape)).astype(np.float32)
                   for k, space in source.observation_space.spaces.items()}
            before = source.predict(obs, deterministic=True)[0]
            with tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                (directory/'manifest.json').write_text(json.dumps({'tactile_model': current_tactile_model()}))
                source.save(directory/'source.zip')
                migrated = make_model(env, settings, 1, 'tactile', resume=directory/'source.zip')
                self.assertIsInstance(migrated.policy, FingerSDEMultiInputPolicy)
                self.assertEqual(migrated.num_timesteps, 122880)
                self.assertEqual(migrated.policy.finger_noise_correlation, .8)
                assert_state_equal(source.policy.state_dict(), migrated.policy.state_dict())
                assert_state_equal(source.policy.optimizer.state_dict(), migrated.policy.optimizer.state_dict())
                np.testing.assert_array_equal(before, migrated.predict(obs, deterministic=True)[0])
                migrated.save(directory/'migrated.zip')
                restored = CPUTrainingPPO.load(directory/'migrated.zip', device='cpu')
                self.assertIsInstance(restored.policy, FingerSDEMultiInputPolicy)
                self.assertEqual(restored.policy.finger_noise_correlation, .8)
                assert_state_equal(migrated.policy.state_dict(), restored.policy.state_dict())
                assert_state_equal(migrated.policy.optimizer.state_dict(), restored.policy.optimizer.state_dict())
                np.testing.assert_array_equal(before, restored.predict(obs, deterministic=True)[0])
                with self.assertRaisesRegex(ValueError, 'silently change'):
                    load_finger_model(directory/'migrated.zip', correlation=.5)
                with self.assertRaisesRegex(ValueError, 'learning rate'):
                    make_model(env, dict(settings, learning_rate=.0001), 1, 'tactile', resume=directory/'source.zip')
        finally:
            env.close()

    def test_requires_shared_checkpoint(self):
        with self.assertRaisesRegex(ValueError, '--resume'):
            make_model(None, {}, 1, 'tactile')

    def test_state_comparison_catches_tensor_changes(self):
        with self.assertRaisesRegex(ValueError, 'tensor changed'):
            assert_state_equal({'x': torch.tensor([1.])}, {'x': torch.tensor([2.])})


if __name__ == '__main__':
    unittest.main()
