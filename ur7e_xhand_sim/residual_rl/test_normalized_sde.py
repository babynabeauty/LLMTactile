"""CPU-only contracts for optional normalized native gSDE, without Genesis.

Run: .venv/bin/python -m unittest ur7e_xhand_sim.residual_rl.test_normalized_sde -v
These tests check distribution/PPO semantics, not grasp success or sim-to-real.
"""
import math
from pathlib import Path
import tempfile
import unittest

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import MultiInputActorCriticPolicy

from .adaptive_policy import policy_kwargs
from .normalized_sde_policy import NormalizedSDEMultiInputPolicy


def spaces():
    return (gym.spaces.Dict({"state": gym.spaces.Box(-10., 10., (4,), dtype=np.float32)}),
            gym.spaces.Box(-1., 1., (1,), dtype=np.float32))


def build(policy_class=NormalizedSDEMultiInputPolicy, seed=13, **kwargs):
    torch.manual_seed(seed)
    observation, action = spaces()
    parameters = dict(use_sde=True, net_arch={"pi": [8], "vf": [8]},
                      log_std_init=math.log(.12))
    parameters.update(kwargs)
    with torch.device("cpu"):
        return policy_class(observation, action, lambda _: 3e-4, **parameters)


class MockDictEnv(gym.Env):
    """Small terminating environment that records actual executed actions."""
    def __init__(self):
        self.observation_space, self.action_space = spaces()
        self.tick = 0
        self.executed = []

    def observation(self):
        return {"state": np.asarray([1., self.tick / 10., .2, -.1], np.float32)}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.tick = 0
        return self.observation(), {}

    def step(self, action):
        self.executed.append(np.asarray(action).copy())
        self.tick += 1
        return self.observation(), -float(np.square(action).sum()), self.tick == 6, False, {}


class CheckRolloutProbabilities(BaseCallback):
    """Check stored probabilities before PPO changes the policy weights."""
    def __init__(self):
        super().__init__()
        self.checked = 0
        self.saw_unclipped = False

    def _on_step(self):
        return True

    def _on_rollout_end(self):
        buffer = self.model.rollout_buffer
        observations = {key: torch.as_tensor(value.reshape((-1, *value.shape[2:])),
                                              device="cpu")
                        for key, value in buffer.observations.items()}
        actions = torch.as_tensor(buffer.actions.reshape(-1, 1), device="cpu")
        with torch.no_grad():
            _, evaluated, _ = self.model.policy.evaluate_actions(observations, actions)
        np.testing.assert_allclose(evaluated.numpy(), buffer.log_probs.reshape(-1),
                                   atol=2e-5, rtol=2e-5)
        self.saw_unclipped |= bool((np.abs(buffer.actions) > 1.).any())
        self.checked += 1


class TestNormalizedSDE(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_same_seed_native_parameters_mean_and_value_unchanged(self):
        native = build(MultiInputActorCriticPolicy)
        normalized = build()
        self.assertEqual(native.state_dict().keys(), normalized.state_dict().keys())
        for name, parameter in native.state_dict().items():
            torch.testing.assert_close(parameter, normalized.state_dict()[name], rtol=0, atol=0)
        observation = {"state": torch.randn(5, 4)}
        native_actions, native_values, _ = native(observation, deterministic=True)
        normalized_actions, normalized_values, _ = normalized(observation, deterministic=True)
        torch.testing.assert_close(native_actions, normalized_actions, rtol=0, atol=0)
        torch.testing.assert_close(native_values, normalized_values, rtol=0, atol=0)

    def test_noise_distribution_and_fixed_noise_are_scale_invariant(self):
        policy = build()
        # Nonuniform learned weights still depend on feature direction, but not norm.
        with torch.no_grad():
            policy.log_std.copy_(torch.linspace(-3., -.5, 8).reshape(8, 1))
        latent = torch.randn(3, 8)
        policy.reset_noise(3)
        scales, noises = [], []
        for factor in (.01, 1., 100.):
            distribution = policy._get_action_dist_from_latent(latent * factor)
            torch.testing.assert_close(distribution.distribution.mean,
                                       policy.action_net(latent * factor), rtol=0, atol=0)
            scales.append(distribution.distribution.scale.detach().clone())
            noises.append((distribution.sample() - distribution.distribution.mean).detach().clone())
        for index in (1, 2):
            torch.testing.assert_close(scales[0], scales[index], atol=1e-7, rtol=1e-6)
            torch.testing.assert_close(noises[0], noises[index], atol=1e-6, rtol=1e-5)

    def test_uniform_weight_std_has_expected_marginal_std(self):
        policy = build()
        distribution = policy._get_action_dist_from_latent(torch.randn(9, 8))
        expected = math.sqrt(.12 ** 2 + distribution.epsilon)
        torch.testing.assert_close(distribution.distribution.scale,
                                   torch.full((9, 1), expected), atol=2e-8, rtol=1e-6)

    def test_zero_and_near_zero_latent_stay_finite_with_native_floor(self):
        policy = build(sde_normalization_eps=1e-8)
        for magnitude in (0., 1e-20, 1e-10, 1e-8):
            latent = torch.full((2, 8), magnitude)
            policy.reset_noise(2)
            distribution = policy._get_action_dist_from_latent(latent)
            sample = distribution.sample()
            self.assertTrue(torch.isfinite(sample).all())
            self.assertTrue(torch.isfinite(distribution.log_prob(sample)).all())
            if magnitude == 0:
                torch.testing.assert_close(sample, distribution.distribution.mean, rtol=0, atol=0)
                torch.testing.assert_close(distribution.distribution.scale,
                                           torch.full((2, 1), math.sqrt(distribution.epsilon)))

    def test_native_sampled_logprob_equals_evaluated_logprob(self):
        policy = build()
        observation = {"state": torch.randn(6, 4)}
        policy.reset_noise(6)
        with torch.no_grad():
            actions, values, sampled_logprob = policy(observation)
            evaluated_values, evaluated_logprob, entropy = policy.evaluate_actions(observation, actions)
        torch.testing.assert_close(sampled_logprob, evaluated_logprob, rtol=0, atol=0)
        torch.testing.assert_close(values, evaluated_values, rtol=0, atol=0)
        self.assertTrue(torch.isfinite(entropy).all())

    def test_paired_adaptive_initialization_differs_only_in_log_std(self):
        observation_space = gym.spaces.Dict({
            "state": gym.spaces.Box(-10., 10., (12, 6), dtype=np.float32),
            "tactile": gym.spaces.Box(-100., 100., (12, 5, 120, 3), dtype=np.float32),
        })
        policies = []
        for use_tactile, initial_std in ((True, .12), (False, .2)):
            torch.manual_seed(0)
            with torch.device("cpu"):
                policies.append(NormalizedSDEMultiInputPolicy(
                    observation_space, spaces()[1], lambda _: 3e-4, use_sde=True,
                    **policy_kwargs(use_tactile=use_tactile, action_std_init=initial_std)))
        first, second = (policy.state_dict() for policy in policies)
        self.assertEqual(first.keys(), second.keys())
        for name in first:
            if name == "log_std":
                self.assertFalse(torch.equal(first[name], second[name]))
            else:
                torch.testing.assert_close(first[name], second[name], rtol=0, atol=0)

    def test_reject_non_sde_and_invalid_normalization_floor(self):
        with self.assertRaisesRegex(ValueError, "use_sde=True"):
            build(use_sde=False)
        for epsilon in (0., -1., np.inf, np.nan, True, 1e-50, 1e100):
            with self.subTest(epsilon=epsilon), self.assertRaisesRegex(ValueError, "finite and positive"):
                build(sde_normalization_eps=epsilon)

    def test_standalone_policy_save_preserves_configuration_and_distribution(self):
        policy = build(sde_normalization_eps=2e-7, share_features_extractor=False)
        observation = {"state": torch.randn(2, 4)}
        before = policy.get_distribution(observation)
        mean, std = before.distribution.mean.detach().clone(), before.distribution.scale.detach().clone()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "policy.pt"
            policy.save(path)
            restored = NormalizedSDEMultiInputPolicy.load(path, device="cpu")
        restored.reset_noise(2)  # Native standalone load does not resample saved-noise matrices.
        self.assertEqual(restored.sde_normalization_eps, 2e-7)
        self.assertFalse(restored.share_features_extractor)
        after = restored.get_distribution(observation)
        torch.testing.assert_close(mean, after.distribution.mean, rtol=0, atol=0)
        torch.testing.assert_close(std, after.distribution.scale, rtol=0, atol=0)

    def test_ppo_native_buffer_clipping_save_load_and_resume(self):
        env = MockDictEnv()
        with torch.device("cpu"):
            model = PPO(NormalizedSDEMultiInputPolicy, env, device="cpu", seed=7,
                        n_steps=16, batch_size=8, n_epochs=1, use_sde=True, sde_sample_freq=4,
                        policy_kwargs={"net_arch": {"pi": [8], "vf": [8]},
                                       "log_std_init": math.log(.12), "sde_normalization_eps": 3e-7})
        # Deliberately exceed action bounds: native buffer must keep raw actions
        # while only the environment receives clipped actions.
        with torch.no_grad():
            model.policy.action_net.weight.zero_()
            model.policy.action_net.bias.fill_(2.)
        callback = CheckRolloutProbabilities()
        model.learn(32, callback=callback)
        self.assertEqual(callback.checked, 2)
        self.assertTrue(callback.saw_unclipped)
        self.assertTrue((np.abs(np.asarray(env.executed)) <= 1.).all())
        observation = env.reset()[0]
        prediction = model.predict(observation, deterministic=True)[0]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "ppo.zip"
            model.save(path)
            restored = PPO.load(path, env=MockDictEnv(), device="cpu")
            self.assertIsInstance(restored.policy, NormalizedSDEMultiInputPolicy)
            self.assertTrue(restored.policy.normalize_sde_latent)
            self.assertTrue(restored.use_sde)
            self.assertEqual(restored.policy.sde_normalization_eps, 3e-7)
            np.testing.assert_array_equal(prediction, restored.predict(observation, deterministic=True)[0])
            for name, parameter in model.policy.state_dict().items():
                torch.testing.assert_close(parameter, restored.policy.state_dict()[name], rtol=0, atol=0)
            restored.learn(16, reset_num_timesteps=False)
            self.assertEqual(restored.num_timesteps, 48)
            self.assertTrue(all(torch.isfinite(value).all() for value in restored.policy.state_dict().values()))


if __name__ == "__main__":
    unittest.main()
