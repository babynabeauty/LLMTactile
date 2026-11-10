"""CPU contracts for V1.5 finger-correlated gSDE; no Genesis simulation."""
import math
from pathlib import Path
import tempfile
import unittest

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.distributions import StateDependentNoiseDistribution

from .finger_sde_policy import FingerSDEMultiInputPolicy, FingerStateDependentNoiseDistribution
from .normalized_sde_policy import NormalizedSDEMultiInputPolicy


def spaces(n_actions=12):
    return (gym.spaces.Dict({"state": gym.spaces.Box(-10., 10., (4,), dtype=np.float32)}),
            gym.spaces.Box(-1., 1., (n_actions,), dtype=np.float32))


def build(policy_class=FingerSDEMultiInputPolicy, seed=17, **kwargs):
    torch.manual_seed(seed)
    joints = kwargs.pop("action_joints", list(range(12)))
    observation, action = spaces(len(joints))
    parameters = dict(use_sde=True, net_arch={"pi": [8], "vf": [8]},
                      log_std_init=math.log(.12))
    if issubclass(policy_class, FingerSDEMultiInputPolicy):
        parameters["action_joints"] = joints
    parameters.update(kwargs)
    with torch.device("cpu"):
        return policy_class(observation, action, lambda _: 3e-4, **parameters)


class MockDictEnv(gym.Env):
    def __init__(self):
        self.observation_space, self.action_space = spaces(6)
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
    def __init__(self):
        super().__init__()
        self.checked = 0
        self.saw_unclipped = False

    def _on_step(self):
        return True

    def _on_rollout_end(self):
        buffer = self.model.rollout_buffer
        observations = {key: torch.as_tensor(value.reshape((-1, *value.shape[2:])), device="cpu")
                        for key, value in buffer.observations.items()}
        actions = torch.as_tensor(buffer.actions.reshape(-1, 6), device="cpu")
        with torch.no_grad():
            _, evaluated, _ = self.model.policy.evaluate_actions(observations, actions)
        np.testing.assert_allclose(evaluated.numpy(), buffer.log_probs.reshape(-1),
                                   atol=3e-5, rtol=3e-5)
        self.saw_unclipped |= bool((np.abs(buffer.actions) > 1.).any())
        self.checked += 1


class TestFingerSDE(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_grouping_uses_joint_ids_in_full_pinch_and_reordered_subsets(self):
        for joints in (list(range(12)), list(range(6)), [5, 1, 11, 0, 4, 2, 8, 3]):
            with self.subTest(joints=joints):
                dist = FingerStateDependentNoiseDistribution(len(joints), action_joints=joints)
                correlation, _ = dist._matrices(torch.zeros(1))
                for i, first in enumerate(joints):
                    for j, second in enumerate(joints):
                        paired = frozenset((first, second)) in {
                            frozenset(pair) for pair in ((1, 2), (4, 5), (6, 7), (8, 9), (10, 11))}
                        expected = 1. if i == j else (.8 if paired else 0.)
                        self.assertAlmostEqual(float(correlation[i, j]), expected, places=7)

    def test_same_seed_parameter_layout_mean_value_and_rng_are_unchanged(self):
        native = build(NormalizedSDEMultiInputPolicy)
        native_rng = torch.random.get_rng_state().clone()
        policy = build()
        self.assertTrue(torch.equal(native_rng, torch.random.get_rng_state()))
        self.assertEqual(list(native.state_dict()), list(policy.state_dict()))
        self.assertEqual([name for name, _ in native.named_parameters()],
                         [name for name, _ in policy.named_parameters()])
        for name, value in native.state_dict().items():
            torch.testing.assert_close(value, policy.state_dict()[name], atol=0, rtol=0)
        observation = {"state": torch.randn(7, 4)}
        native_actions, native_values, _ = native(observation, deterministic=True)
        actions, values, _ = policy(observation, deterministic=True)
        torch.testing.assert_close(native_actions, actions, atol=0, rtol=0)
        torch.testing.assert_close(native_values, values, atol=0, rtol=0)
        self.assertEqual(native.optimizer.state_dict(), policy.optimizer.state_dict())

    def test_zero_correlation_matches_native_sampling_logprob_and_entropy_exactly(self):
        native = StateDependentNoiseDistribution(6)
        correlated = FingerStateDependentNoiseDistribution(
            6, action_joints=range(6), finger_noise_correlation=0.)
        log_std = torch.linspace(-3., -.5, 48).reshape(8, 6)
        mean = torch.randn(9, 6)
        latent = torch.nn.functional.normalize(torch.randn(9, 8), dim=1)
        for dist in (native, correlated):
            torch.manual_seed(13)
            dist.sample_weights(log_std, 9)
            dist.proba_distribution(mean, log_std, latent)
        for name in ("exploration_mat", "exploration_matrices"):
            torch.testing.assert_close(getattr(native, name), getattr(correlated, name), atol=0, rtol=0)
        actions = native.sample()
        torch.testing.assert_close(actions, correlated.sample(), atol=0, rtol=0)
        torch.testing.assert_close(native.log_prob(actions), correlated.log_prob(actions), atol=0, rtol=0)
        torch.testing.assert_close(native.entropy(), correlated.entropy(), atol=0, rtol=0)

    def test_nonuniform_covariance_marginals_and_empirical_samples(self):
        torch.manual_seed(31)
        joints = [5, 1, 4, 2, 0, 3]
        dist = FingerStateDependentNoiseDistribution(6, action_joints=joints)
        log_std = torch.linspace(-3., -1., 24, dtype=torch.float64).reshape(4, 6)
        latent = torch.nn.functional.normalize(torch.tensor([[1., -.7, .4, .2]], dtype=torch.float64), dim=1)
        dist.proba_distribution(torch.zeros(1, 6, dtype=torch.float64), log_std, latent)
        correlation, _ = dist._matrices(log_std)
        std = log_std.exp()
        expected = sum(latent[0, i].square() * torch.diag(std[i]) @ correlation @ torch.diag(std[i])
                       for i in range(4))
        observed_covariance = dist.distribution.covariance_matrix[0]
        torch.testing.assert_close(observed_covariance, expected + dist.epsilon * torch.eye(6),
                                   rtol=1e-6, atol=1e-12)
        native_variance = (latent.square() @ std.square())[0]
        torch.testing.assert_close(observed_covariance.diag(), native_variance + dist.epsilon)
        # Native gSDE sampling omits the likelihood epsilon. Check many independent
        # lane matrices against the covariance, not only a single shared matrix.
        dist.sample_weights(log_std, 60000)
        samples = dist.get_noise(latent.expand(60000, -1)).detach()
        empirical = torch.cov(samples.T)
        torch.testing.assert_close(empirical, expected, rtol=.04, atol=3e-4)
        self.assertTrue((samples.mean(0).abs() < .003).all())

    def test_fixed_noise_is_reused_and_only_resampled_explicitly(self):
        policy = build()
        policy.reset_noise(3)
        latent = torch.randn(3, 8)
        dist = policy._get_action_dist_from_latent(latent)
        noise = (dist.sample() - dist.distribution.mean).detach().clone()
        again = policy._get_action_dist_from_latent(latent * 5.)
        torch.testing.assert_close(again.sample() - again.distribution.mean, noise, rtol=2e-5, atol=1e-7)
        policy.reset_noise(3)
        refreshed = policy._get_action_dist_from_latent(latent)
        self.assertFalse(torch.equal(refreshed.sample() - refreshed.distribution.mean, noise))

    def test_sampled_evaluated_logprob_and_gradients(self):
        for squash_output in (False, True):
            with self.subTest(squash_output=squash_output):
                policy = build(squash_output=squash_output)
                observation = {"state": torch.randn(6, 4)}
                policy.reset_noise(6)
                actions, values, sampled_logprob = policy(observation)
                evaluated_values, evaluated_logprob, entropy = policy.evaluate_actions(observation, actions.detach())
                torch.testing.assert_close(sampled_logprob, evaluated_logprob, atol=0, rtol=0)
                torch.testing.assert_close(values, evaluated_values, atol=0, rtol=0)
                loss = -evaluated_logprob.mean() + evaluated_values.square().mean()
                if entropy is not None:
                    loss -= .01 * entropy.mean()
                loss.backward()
                for parameter in policy.parameters():
                    if parameter.grad is not None:
                        self.assertTrue(torch.isfinite(parameter.grad).all())
                self.assertGreater(float(policy.log_std.grad.abs().max()), 0.)

    def test_zero_latent_and_tiny_features_keep_native_likelihood_floor(self):
        policy = build()
        for magnitude in (0., 1e-20, 1e-10, 1e-8):
            latent = torch.full((2, 8), magnitude)
            policy.reset_noise(2)
            dist = policy._get_action_dist_from_latent(latent)
            actions = dist.sample()
            self.assertTrue(torch.isfinite(actions).all())
            self.assertTrue(torch.isfinite(dist.log_prob(actions)).all())
            self.assertTrue(torch.isfinite(dist.entropy()).all())
            if magnitude == 0:
                torch.testing.assert_close(actions, dist.distribution.mean, atol=0, rtol=0)
                expected = dist.epsilon * torch.eye(12).expand(2, -1, -1)
                torch.testing.assert_close(dist.distribution.covariance_matrix, expected, atol=0, rtol=0)

    def test_reject_invalid_configuration(self):
        for joints in ([], [0, 0], [-1], [12], [1.5], [True]):
            with self.subTest(joints=joints), self.assertRaisesRegex(ValueError, "joint ids"):
                build(action_joints=joints)
        for rho in (-.1, 1., np.inf, np.nan, True):
            with self.subTest(rho=rho), self.assertRaisesRegex(ValueError, "correlation"):
                build(finger_noise_correlation=rho)
        with self.assertRaisesRegex(ValueError, "length"):
            FingerStateDependentNoiseDistribution(12, action_joints=range(6))
        with self.assertRaisesRegex(ValueError, "use_sde=True"):
            build(use_sde=False)

    def test_standalone_policy_save_preserves_full_configuration(self):
        policy = build(action_joints=[5, 1, 4, 2, 0, 3], finger_noise_correlation=.6,
                       sde_normalization_eps=2e-7, share_features_extractor=False)
        observation = {"state": torch.randn(2, 4)}
        before = policy.get_distribution(observation)
        mean = before.distribution.mean.detach().clone()
        covariance = before.distribution.covariance_matrix.detach().clone()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "policy.pt"
            policy.save(path)
            restored = FingerSDEMultiInputPolicy.load(path, device="cpu")
        restored.reset_noise(2)
        self.assertEqual(restored.action_joints, (5, 1, 4, 2, 0, 3))
        self.assertEqual(restored.finger_noise_correlation, .6)
        self.assertEqual(restored.sde_normalization_eps, 2e-7)
        self.assertFalse(restored.share_features_extractor)
        after = restored.get_distribution(observation)
        torch.testing.assert_close(mean, after.distribution.mean, atol=0, rtol=0)
        torch.testing.assert_close(covariance, after.distribution.covariance_matrix, atol=0, rtol=0)

    def test_native_adam_state_can_be_copied_without_new_parameters(self):
        native = build(NormalizedSDEMultiInputPolicy)
        observation = {"state": torch.randn(3, 4)}
        actions = torch.randn(3, 12) * .1
        values, log_prob, _ = native.evaluate_actions(observation, actions)
        (values.square().mean() - log_prob.mean()).backward()
        native.optimizer.step()
        policy = build()
        policy.load_state_dict(native.state_dict(), strict=True)
        policy.optimizer.load_state_dict(native.optimizer.state_dict())
        expected, actual = native.optimizer.state_dict(), policy.optimizer.state_dict()
        self.assertEqual(expected["param_groups"], actual["param_groups"])
        for key, state in expected["state"].items():
            for name, value in state.items():
                torch.testing.assert_close(value, actual["state"][key][name], atol=0, rtol=0)
        policy.optimizer.zero_grad()
        values, log_prob, _ = policy.evaluate_actions(observation, actions)
        (values.square().mean() - log_prob.mean()).backward()
        policy.optimizer.step()
        self.assertTrue(all(torch.isfinite(value).all() for value in policy.state_dict().values()))

    def test_ppo_buffer_save_load_and_update(self):
        env = MockDictEnv()
        with torch.device("cpu"):
            model = PPO(FingerSDEMultiInputPolicy, env, device="cpu", seed=7,
                        n_steps=16, batch_size=8, n_epochs=1, use_sde=True, sde_sample_freq=-1,
                        policy_kwargs={"net_arch": {"pi": [8], "vf": [8]},
                                       "log_std_init": math.log(.12), "action_joints": list(range(6)),
                                       "finger_noise_correlation": .8, "sde_normalization_eps": 3e-7})
        with torch.no_grad():
            model.policy.action_net.weight.zero_()
            model.policy.action_net.bias.fill_(2.)
        callback = CheckRolloutProbabilities()
        model.learn(32, callback=callback)
        self.assertEqual(callback.checked, 2)
        self.assertTrue(callback.saw_unclipped)
        self.assertTrue((np.abs(np.asarray(env.executed)) <= 1.).all())
        prediction = model.predict(env.reset()[0], deterministic=True)[0]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "ppo.zip"
            model.save(path)
            restored = PPO.load(path, env=MockDictEnv(), device="cpu")
            self.assertIsInstance(restored.policy, FingerSDEMultiInputPolicy)
            self.assertEqual(restored.policy.action_joints, tuple(range(6)))
            self.assertEqual(restored.policy.finger_noise_correlation, .8)
            self.assertEqual(restored.policy.sde_normalization_eps, 3e-7)
            self.assertEqual(restored.sde_sample_freq, -1)
            np.testing.assert_array_equal(prediction, restored.predict(env.reset()[0], deterministic=True)[0])
            for name, value in model.policy.state_dict().items():
                torch.testing.assert_close(value, restored.policy.state_dict()[name], atol=0, rtol=0)
            restored.learn(16, reset_num_timesteps=False)
            self.assertEqual(restored.num_timesteps, 48)
            self.assertTrue(all(torch.isfinite(value).all() for value in restored.policy.state_dict().values()))


if __name__ == "__main__":
    unittest.main()
