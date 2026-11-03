"""CPU checks that the no_tactile encoder ignores touch and the encoder is causal."""
import unittest

import gymnasium as gym
import numpy as np
import torch

from .adaptive_policy import AdaptiveTactileExtractor


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
