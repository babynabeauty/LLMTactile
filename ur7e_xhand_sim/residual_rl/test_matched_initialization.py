"""Pure CPU contracts for independent PPO encoders; no simulation/training."""
import copy
import gc
import inspect
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import weakref

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.vec_env import DummyVecEnv

from .adaptive_policy import policy_kwargs
from .adaptive_train import parser, validate_resume_extractors
from .matched_initialization import (build_matched_separate_ppo, capture_rng_state,
                                     restore_rng_state)
from .normalized_sde_policy import NormalizedSDEMultiInputPolicy


class SyntheticEnv(gym.Env):
    observation_space = gym.spaces.Dict({
        "tactile": gym.spaces.Box(-np.inf, np.inf, (12, 5, 120, 3), np.float32),
        "state": gym.spaces.Box(-np.inf, np.inf, (12, 79), np.float32)})
    action_space = gym.spaces.Box(-1, 1, (1,), np.float32)

    def reset(self, *, seed=None, options=None):
        raise AssertionError("Model construction must not reset the environment")

    def step(self, action):
        raise AssertionError("No simulation or environment step is allowed")


class MatchedInitializationContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)

    def fixture(self, use_tactile=True):
        env = DummyVecEnv([SyntheticEnv, SyntheticEnv])
        self.addCleanup(env.close)
        pkwargs = policy_kwargs(use_tactile=use_tactile, action_std_init=.12)
        kwargs = dict(n_steps=8, batch_size=4, n_epochs=5, seed=0, gamma=1.,
                      gae_lambda=1., use_sde=True, sde_sample_freq=30, device="cpu")
        with torch.device("cpu"):
            shared = PPO(NormalizedSDEMultiInputPolicy, env, policy_kwargs=pkwargs, **kwargs)
        after_shared = capture_rng_state()
        action_rng = copy.deepcopy(env.action_space.np_random.bit_generator.state)
        separate, metadata = build_matched_separate_ppo(
            NormalizedSDEMultiInputPolicy, env, policy_kwargs=pkwargs, **kwargs)
        rng = np.random.default_rng(82)
        obs = {k: rng.normal(0, .03, (2, *space.shape)).astype(np.float32)
               for k, space in env.observation_space.spaces.items()}
        return shared, separate, metadata, after_shared, action_rng, obs

    def assert_nested_equal(self, left, right):
        if isinstance(left, torch.Tensor):
            torch.testing.assert_close(left, right, atol=0, rtol=0)
        elif isinstance(left, np.ndarray):
            np.testing.assert_array_equal(left, right)
        elif isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys())
            for key in left:
                self.assert_nested_equal(left[key], right[key])
        elif isinstance(left, (tuple, list)):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.assert_nested_equal(a, b)
        else:
            self.assertEqual(left, right)

    def test_initial_weights_latents_values_std_and_rng_match_shared(self):
        shared, separate, metadata, after_shared, action_rng, obs = self.fixture()
        self.assert_nested_equal(after_shared, capture_rng_state())
        self.assert_nested_equal(action_rng, separate.env.action_space.np_random.bit_generator.state)
        self.assertEqual(separate.env._seeds, shared.env._seeds)
        self.assert_nested_equal(shared.policy.state_dict(), separate.policy.state_dict())
        self.assertFalse(separate.policy.share_features_extractor)
        self.assertEqual(len(separate.policy.optimizer.state), 0)
        self.assertEqual(metadata["method"], "shared_template_strict_policy_state_copy")
        with torch.no_grad():
            tensors, _ = shared.policy.obs_to_tensor(obs)
            shared_features = shared.policy.extract_features(tensors)
            pi_features, vf_features = separate.policy.extract_features(tensors)
            self.assert_nested_equal(shared_features, pi_features)
            self.assert_nested_equal(shared_features, vf_features)
            self.assert_nested_equal(shared.policy.mlp_extractor.forward_actor(shared_features),
                                     separate.policy.mlp_extractor.forward_actor(pi_features))
            self.assert_nested_equal(shared.policy.mlp_extractor.forward_critic(shared_features),
                                     separate.policy.mlp_extractor.forward_critic(vf_features))
            self.assert_nested_equal(shared.policy.predict_values(tensors), separate.policy.predict_values(tensors))
            a = shared.policy.get_distribution(tensors).distribution
            b = separate.policy.get_distribution(tensors).distribution
            self.assert_nested_equal(a.mean, b.mean)
            self.assert_nested_equal(a.stddev, b.stddev)
            # A fresh native rollout resets exploration noise before sampling.
            restore_rng_state(after_shared)
            shared.policy.reset_noise(2)
            shared_result = shared.policy(tensors, deterministic=False)
            after_shared_sample = capture_rng_state()
            restore_rng_state(after_shared)
            separate.policy.reset_noise(2)
            separate_result = separate.policy(tensors, deterministic=False)
            self.assert_nested_equal(shared.policy.action_dist.exploration_matrices,
                                     separate.policy.action_dist.exploration_matrices)
            self.assert_nested_equal(shared_result, separate_result)
            self.assert_nested_equal(after_shared_sample, capture_rng_state())
        source = inspect.getsource(OnPolicyAlgorithm.collect_rollouts)
        self.assertLess(source.index("self.policy.reset_noise(env.num_envs)"),
                        source.index("actions, values, log_probs = self.policy(obs_tensor)"))

    def test_branch_storage_and_gradients_are_independent(self):
        _, separate, _, _, _, obs = self.fixture()
        policy = separate.policy
        actor = tuple(policy.pi_features_extractor.parameters())
        critic = tuple(policy.vf_features_extractor.parameters())
        self.assertFalse({id(p) for p in actor} & {id(p) for p in critic})
        self.assertFalse({p.data_ptr() for p in actor} & {p.data_ptr() for p in critic})
        tensors, _ = policy.obs_to_tensor(obs)
        distribution = policy.get_distribution(tensors)
        actions = distribution.distribution.mean.detach() + .08
        actor_loss = -distribution.log_prob(actions).mean()
        actor_grads = torch.autograd.grad(actor_loss, actor + critic, allow_unused=True)
        self.assertTrue(any(g is not None and torch.any(g != 0) for g in actor_grads[:len(actor)]))
        self.assertTrue(all(g is None for g in actor_grads[len(actor):]))
        critic_loss = (policy.predict_values(tensors) - 2.).square().mean()
        critic_grads = torch.autograd.grad(critic_loss, actor + critic, allow_unused=True)
        self.assertTrue(all(g is None for g in critic_grads[:len(actor)]))
        self.assertTrue(any(g is not None and torch.any(g != 0) for g in critic_grads[len(actor):]))
        self.assertTrue(all(p.grad is None for p in policy.parameters()))

    def test_save_load_preserves_both_branches_and_nonempty_optimizer(self):
        _, separate, _, _, _, obs = self.fixture()
        tensors, _ = separate.policy.obs_to_tensor(obs)
        distribution = separate.policy.get_distribution(tensors)
        actions = distribution.distribution.mean.detach() + .08
        loss = -distribution.log_prob(actions).mean() + (separate.policy.predict_values(tensors) - 2.).square().mean()
        # One synthetic optimizer step populates both branches' Adam states;
        # no PPO learning, rollout collection, or environment operation occurs.
        loss.backward()
        separate.policy.optimizer.step()
        separate.policy.optimizer.zero_grad(set_to_none=True)
        for branch in (separate.policy.pi_features_extractor, separate.policy.vf_features_extractor):
            self.assertTrue(all(p in separate.policy.optimizer.state for p in branch.parameters()))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.zip"
            separate.save(path)
            with torch.device("cpu"):
                loaded = PPO.load(path, env=separate.env, device="cpu")
            self.assertFalse(loaded.policy.share_features_extractor)
            self.assert_nested_equal(separate.policy.state_dict(), loaded.policy.state_dict())
            self.assert_nested_equal(separate.policy.optimizer.state_dict(), loaded.policy.optimizer.state_dict())
            self.assertFalse({p.data_ptr() for p in loaded.policy.pi_features_extractor.parameters()} &
                             {p.data_ptr() for p in loaded.policy.vf_features_extractor.parameters()})
            validate_resume_extractors(loaded, None, False)
            validate_resume_extractors(loaded, True, False)
            with self.assertRaisesRegex(ValueError, "must match"):
                validate_resume_extractors(loaded, False, False)

    def test_ablation_masks_both_encoders_and_resume_checks_both(self):
        _, separate, _, _, _, obs = self.fixture(use_tactile=False)
        self.assertFalse(separate.policy.pi_features_extractor.use_tactile)
        self.assertFalse(separate.policy.vf_features_extractor.use_tactile)
        changed = {**obs, "tactile": obs["tactile"] * -10000 + 57}
        with torch.no_grad():
            tensors, _ = separate.policy.obs_to_tensor(obs)
            other, _ = separate.policy.obs_to_tensor(changed)
            self.assert_nested_equal(separate.policy(tensors, deterministic=True),
                                     separate.policy(other, deterministic=True))
            first = separate.policy.get_distribution(tensors).distribution.stddev.clone()
            second = separate.policy.get_distribution(other).distribution.stddev
            self.assert_nested_equal(first, second)
        validate_resume_extractors(separate, True, True)
        with self.assertRaisesRegex(ValueError, "ablate-tactile"):
            validate_resume_extractors(separate, True, False)
        separate.policy.vf_features_extractor.use_tactile = True
        with self.assertRaisesRegex(ValueError, "vf_features_extractor"):
            validate_resume_extractors(separate, True, True)

    def test_template_is_released_and_cpu_context_does_not_leak(self):
        env = DummyVecEnv([SyntheticEnv, SyntheticEnv])
        self.addCleanup(env.close)
        refs = []

        def record(*args, **kwargs):
            model = PPO(*args, **kwargs)
            refs.append(weakref.ref(model))
            return model

        with patch("ur7e_xhand_sim.residual_rl.matched_initialization.PPO", side_effect=record):
            with torch.device("meta"):
                separate, _ = build_matched_separate_ppo(
                    NormalizedSDEMultiInputPolicy, env, policy_kwargs=policy_kwargs(action_std_init=.12),
                    n_steps=8, batch_size=4, seed=0, use_sde=True)
                self.assertEqual(str(torch.get_default_device()), "meta")
        gc.collect()
        self.assertIsNone(refs[0]())
        self.assertIs(refs[1](), separate)
        self.assertEqual({str(p.device) for p in separate.policy.parameters()}, {"cpu"})

    def test_cli_default_and_explicit_resume_contract(self):
        base = ["--conditions-json", "conditions.json", "--config-json", "config.json",
                "--output", "out", "--total-timesteps", "1"]
        self.assertIsNone(parser().parse_args(base).separate_feature_extractors)
        self.assertTrue(parser().parse_args(base + ["--separate-feature-extractors"]).separate_feature_extractors)
        self.assertFalse(parser().parse_args(base + ["--no-separate-feature-extractors"]).separate_feature_extractors)
        shared, _, _, _, _, _ = self.fixture()
        validate_resume_extractors(shared, None, False)
        validate_resume_extractors(shared, False, False)
        with self.assertRaisesRegex(ValueError, "must match"):
            validate_resume_extractors(shared, True, False)

    def test_native_diagonal_gaussian_path_matches_too(self):
        env = DummyVecEnv([SyntheticEnv, SyntheticEnv])
        self.addCleanup(env.close)
        kwargs = dict(n_steps=8, batch_size=4, seed=3, device="cpu")
        pkwargs = policy_kwargs(action_std_init=.08)
        with torch.device("cpu"):
            shared = PPO("MultiInputPolicy", env, policy_kwargs=pkwargs, **kwargs)
        after_shared = capture_rng_state()
        separate, _ = build_matched_separate_ppo("MultiInputPolicy", env, policy_kwargs=pkwargs, **kwargs)
        self.assert_nested_equal(shared.policy.state_dict(), separate.policy.state_dict())
        self.assert_nested_equal(after_shared, capture_rng_state())
        self.assertFalse(separate.use_sde)


if __name__ == "__main__":
    unittest.main()
