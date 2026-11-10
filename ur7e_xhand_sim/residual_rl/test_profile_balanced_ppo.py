"""CPU checks for profile weighting, temporal identity and real PPO updates."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import set_random_seed

from .profile_balanced_ppo import (ProfileBalancedDictRolloutBuffer, ProfileBalancedPPO,
                                   profile_weights, weighted_mean)
from .test_normalized_sde import MockDictEnv, spaces
from .v1_balanced_train import BalancedTrainingCallback, current_profile_ids, profile_key
from .v1_train import TrainingCallback


class TagSyntheticRollout(BaseCallback):
    def _on_step(self):
        buffer = self.model.rollout_buffer
        buffer.profile_ids[buffer.pos] = int(buffer.pos % 4 == 0)
        return True


class TagOneGroup(BaseCallback):
    def _on_step(self):
        self.model.rollout_buffer.profile_ids[self.model.rollout_buffer.pos] = 0
        return True


class BalancedPPOTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_unequal_groups_have_equal_loss_and_gradient_weight(self):
        ids = np.array([0, 0, 0, 1])
        weights, counts = profile_weights(ids, 2)
        np.testing.assert_array_equal(counts, [3, 1])
        x = torch.tensor([2., 2., 2., 8.], requires_grad=True)
        loss = weighted_mean(x, torch.tensor(weights))
        self.assertAlmostEqual(loss.item(), 5.)
        loss.backward()
        self.assertAlmostEqual(x.grad[:3].sum().item(), .5)
        self.assertAlmostEqual(x.grad[3].item(), .5)
        duplicate_weights, _ = profile_weights(np.array([0]*30+[1]), 2)
        self.assertAlmostEqual(weighted_mean(torch.tensor([2.]*30+[8.]),
                                             torch.tensor(duplicate_weights)).item(), 5., places=5)

    def test_global_weights_not_minibatch_renormalization(self):
        weights, _ = profile_weights(np.array([0]*6+[1]*2), 2)
        losses = torch.arange(8, dtype=torch.float32)
        all_loss = weighted_mean(losses, torch.tensor(weights))
        batches = [weighted_mean(losses[i:i+2], torch.tensor(weights[i:i+2])) for i in range(0, 8, 2)]
        torch.testing.assert_close(torch.stack(batches).mean(), all_loss)

    def test_missing_and_invalid_profiles_are_explicit(self):
        weights, counts = profile_weights(np.array([0, 0, 2]), 3)
        np.testing.assert_array_equal(counts, [2, 0, 1])
        self.assertAlmostEqual(float(weights.mean()), 1.)
        for ids in ([-1, 0], [0, 3], [0., 1.]):
            with self.assertRaises(ValueError):
                profile_weights(np.array(ids), 3)

    def test_flattening_keeps_tags_with_observations_across_epochs(self):
        obs, act = spaces()
        b = ProfileBalancedDictRolloutBuffer(3, obs, act, device='cpu', n_envs=2, n_profiles=6)
        for step in range(3):
            ids = np.array([step*2, step*2+1])
            b.profile_ids[b.pos] = ids
            b.add({'state': np.repeat(ids[:, None], 4, axis=1).astype(np.float32)},
                  np.zeros((2, 1)), np.zeros(2), np.zeros(2), torch.zeros(2), torch.zeros(2))
        for _ in range(2):
            seen = []
            for data in b.get(2):
                torch.testing.assert_close(data.observations['state'][:, 0].long(), data.profile_ids)
                torch.testing.assert_close(data.weights, torch.ones(2))
                seen.extend(data.profile_ids.tolist())
            self.assertEqual(sorted(seen), list(range(6)))
        b.reset()
        self.assertTrue((b.profile_ids == -1).all())

    def test_terminal_transition_keeps_old_profile_then_advances(self):
        profiles = [dict(mass_kg=.1, friction=.45, f_break_N=4.),
                    dict(mass_kg=.25, friction=.5, f_break_N=None)]
        scene = SimpleNamespace(episode=[profiles[1]])  # already auto-reset
        callback = object.__new__(BalancedTrainingCallback)
        callback.scenes = [scene]
        callback.profile_map = {profile_key(p): i for i, p in enumerate(profiles)}
        callback.transition_profiles = np.array([0])
        buffer = SimpleNamespace(pos=0, profile_ids=np.full((2, 1), -1))
        callback.model = SimpleNamespace(rollout_buffer=buffer)
        with patch.object(TrainingCallback, '_on_step', return_value=True):
            callback._on_step()
            buffer.pos = 1
            callback._on_step()
        np.testing.assert_array_equal(buffer.profile_ids[:, 0], [0, 1])
        np.testing.assert_array_equal(current_profile_ids([scene], callback.profile_map), [1])

    def test_real_update_save_resume_and_native_evaluation_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            model = ProfileBalancedPPO('MultiInputPolicy', MockDictEnv(), n_steps=8, batch_size=4,
                n_epochs=2, device='cpu', seed=7, use_sde=True,
                rollout_buffer_class=ProfileBalancedDictRolloutBuffer,
                rollout_buffer_kwargs={'n_profiles': 2}, policy_kwargs={'net_arch': [8]})
            model.balance_log_path = str(Path(directory)/'updates.jsonl')
            before = {k: v.clone() for k, v in model.policy.state_dict().items()}
            model.learn(16, callback=TagSyntheticRollout())
            self.assertTrue(any(not torch.equal(before[k], v) for k, v in model.policy.state_dict().items()))
            rows = [json.loads(line) for line in Path(model.balance_log_path).read_text().splitlines()]
            self.assertEqual([r['profile_counts'] for r in rows], [[6, 2], [6, 2]])
            self.assertTrue(all(r['optimizer_steps'] > 0 for r in rows))
            model.save(Path(directory)/'model')
            loaded = PPO.load(Path(directory)/'model', device='cpu')
            obs = model.get_env().reset()
            np.testing.assert_array_equal(model.predict(obs, deterministic=True)[0], loaded.predict(obs, deterministic=True)[0])
            resumed = ProfileBalancedPPO.load(Path(directory)/'model', env=MockDictEnv(), device='cpu')
            resumed.learn(8, callback=TagSyntheticRollout(), reset_num_timesteps=False)
            self.assertEqual(resumed.num_timesteps, 24)

    def test_uniform_weights_reproduce_native_ppo_update(self):
        kwargs = dict(n_steps=8, batch_size=4, n_epochs=2, seed=7, device='cpu',
                      use_sde=True, policy_kwargs={'net_arch': [8]})
        native = PPO('MultiInputPolicy', MockDictEnv(), **kwargs)
        balanced = ProfileBalancedPPO('MultiInputPolicy', MockDictEnv(), **kwargs,
            rollout_buffer_class=ProfileBalancedDictRolloutBuffer,
            rollout_buffer_kwargs={'n_profiles': 1})
        set_random_seed(7)
        native.learn(16)
        set_random_seed(7)
        balanced.learn(16, callback=TagOneGroup())
        for key, tensor in native.policy.state_dict().items():
            torch.testing.assert_close(tensor, balanced.policy.state_dict()[key], rtol=0, atol=1e-7)


if __name__ == '__main__':
    unittest.main()
