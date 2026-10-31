"""CPU-only noninterference and terminal Monte Carlo contracts; no Genesis."""
import ast
import copy
import inspect
import json
from pathlib import Path
import pickle
import random
import tempfile
import textwrap
import unittest
from unittest.mock import patch

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.buffers import DictRolloutBuffer
from stable_baselines3.common.logger import Logger
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.vec_env import DummyVecEnv

from .adaptive_policy import policy_kwargs
from .adaptive_train import AdaptiveTrainingCallback, parser
from .gradient_diagnostics import diagnose_rollout_gradients
from .normalized_sde_policy import NormalizedSDEMultiInputPolicy


class SyntheticEnv(gym.Env):
    observation_space = gym.spaces.Dict({
        "tactile": gym.spaces.Box(-np.inf, np.inf, (12, 5, 120, 3), np.float32),
        "state": gym.spaces.Box(-np.inf, np.inf, (12, 79), np.float32)})
    action_space = gym.spaces.Box(-1, 1, (1,), np.float32)

    def reset(self, *, seed=None, options=None):
        return {k: np.zeros(s.shape, np.float32) for k, s in self.observation_space.spaces.items()}, {}

    def step(self, action):
        raise AssertionError("No environment or physical steps are allowed")


class GradientDiagnosticContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)

    def fixture(self, share_features_extractor=True, squash_output=False):
        env = DummyVecEnv([SyntheticEnv, SyntheticEnv])
        self.addCleanup(env.close)
        kwargs = policy_kwargs(action_std_init=.12)
        kwargs.update(share_features_extractor=share_features_extractor,
                      squash_output=squash_output)
        with torch.device("cpu"):
            model = PPO(NormalizedSDEMultiInputPolicy, env,
                        policy_kwargs=kwargs, use_sde=True,
                        n_steps=8, batch_size=4, seed=7, gamma=1, gae_lambda=1, device="cpu")
        model.policy.set_training_mode(False)
        rng = np.random.default_rng(11)
        for step in range(8):
            obs = {k: rng.normal(0, .02, (2, *space.shape)).astype(np.float32)
                   for k, space in env.observation_space.spaces.items()}
            tensors, _ = model.policy.obs_to_tensor(obs)
            with torch.no_grad():
                # Fixed offsets give nonzero policy gradients without drawing
                # random actions, while retaining exact old/new log probabilities.
                actions, _, _ = model.policy(tensors, deterministic=True)
                actions = actions + torch.tensor([[-.05], [.07]])
                values, log_prob, _ = model.policy.evaluate_actions(tensors, actions)
            reward = np.array([1., -2.], np.float32) + (np.array([25., -75.]) if step == 7 else 0)
            model.rollout_buffer.add(obs, actions.numpy(), reward,
                                     np.full(2, step == 0), values, log_prob)
        model.rollout_buffer.compute_returns_and_advantage(torch.tensor([999., -999.]), np.ones(2, bool))
        model.num_timesteps, model._total_timesteps = 16, 64
        for index, parameter in enumerate(model.policy.parameters()):
            parameter.grad = torch.full_like(parameter, .01) if index % 2 else None
        parameter = next(model.policy.parameters())
        model.policy.optimizer.state[parameter] = {
            "step": torch.tensor(3.), "exp_avg": torch.full_like(parameter, .02),
            "exp_avg_sq": torch.full_like(parameter, .03)}
        return model

    def snapshot(self, model):
        return {
            "state_dict": {k: v.detach().clone() for k, v in model.policy.state_dict().items()},
            "gradients": [(p.grad, None if p.grad is None else p.grad.clone()) for p in model.policy.parameters()],
            "optimizer": copy.deepcopy(model.policy.optimizer.state_dict()),
            "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
            "torch_rng": torch.get_rng_state().clone(), "rollout": pickle.dumps(model.rollout_buffer),
            "modes": [m.training for m in model.policy.modules()],
            "distribution_cache": model.policy.action_dist.__dict__.copy(),
            "progress": model._current_progress_remaining,
        }

    def assert_nested_equal(self, expected, actual):
        if isinstance(expected, torch.Tensor):
            torch.testing.assert_close(expected, actual, atol=0, rtol=0)
        elif isinstance(expected, dict):
            self.assertEqual(expected.keys(), actual.keys())
            for key in expected:
                self.assert_nested_equal(expected[key], actual[key])
        elif isinstance(expected, (tuple, list)):
            self.assertEqual(len(expected), len(actual))
            for left, right in zip(expected, actual):
                self.assert_nested_equal(left, right)
        elif isinstance(expected, np.ndarray):
            np.testing.assert_array_equal(expected, actual)
        else:
            self.assertEqual(expected, actual)

    def assert_unchanged(self, before, model):
        self.assert_nested_equal(before["state_dict"], model.policy.state_dict())
        for (identity, value), parameter in zip(before["gradients"], model.policy.parameters()):
            self.assertIs(identity, parameter.grad)
            if value is not None:
                torch.testing.assert_close(value, parameter.grad, atol=0, rtol=0)
        self.assert_nested_equal(before["optimizer"], model.policy.optimizer.state_dict())
        self.assertEqual(before["python_rng"], random.getstate())
        self.assert_nested_equal(before["numpy_rng"], np.random.get_state())
        torch.testing.assert_close(before["torch_rng"], torch.get_rng_state(), atol=0, rtol=0)
        self.assertEqual(before["rollout"], pickle.dumps(model.rollout_buffer))
        self.assertEqual(before["modes"], [m.training for m in model.policy.modules()])
        self.assertEqual(before["progress"], model._current_progress_remaining)
        self.assertEqual(before["distribution_cache"].keys(), model.policy.action_dist.__dict__.keys())
        for key, value in before["distribution_cache"].items():
            self.assertIs(value, model.policy.action_dist.__dict__[key])

    def test_diagnostics_preserve_all_training_state_and_are_repeatable(self):
        model = self.fixture()
        before = self.snapshot(model)
        # Genesis changes the global device; diagnostics must still remain CPU.
        with torch.device("meta"):
            first = diagnose_rollout_gradients(model, max_samples=7)
        self.assert_unchanged(before, model)
        second = diagnose_rollout_gradients(model, max_samples=7)
        self.assert_unchanged(before, model)
        self.assertEqual(first, second)
        self.assertEqual(first["sample_count"], 7)
        self.assertLess(first["ratio_max_abs_error_from_one"], 2e-6)
        self.assertGreater(first["shared_policy_grad_norm"], 0)
        self.assertGreater(first["shared_weighted_value_grad_norm"], 0)
        self.assertTrue(first["share_features_extractor"])
        self.assertEqual(first["actor_encoder_policy_grad_norm"], first["shared_policy_grad_norm"])
        self.assertEqual(first["critic_encoder_weighted_value_grad_norm"], first["shared_weighted_value_grad_norm"])
        self.assertIsNone(first["cross_encoder_gradients_zero_or_none"])
        self.assertLessEqual(abs(first["shared_policy_value_grad_cosine"]), 1)
        self.assertGreater(first["global_total_grad_norm_preclip"], 0)
        self.assertEqual(first["progress_for_imminent_update"], .75)

    def test_distribution_cache_restored_on_failure(self):
        for shared in (True, False):
            with self.subTest(shared=shared):
                model = self.fixture(share_features_extractor=shared)
                before = self.snapshot(model)
                original = model.policy.evaluate_actions

                def broken(*args, **kwargs):
                    original(*args, **kwargs)
                    raise RuntimeError("injected diagnostic failure")

                with patch.object(model.policy, "evaluate_actions", side_effect=broken):
                    with self.assertRaisesRegex(RuntimeError, "injected"):
                        diagnose_rollout_gradients(model)
                self.assert_unchanged(before, model)

    def test_separate_encoders_preserve_state_and_report_disconnected_cross_gradients(self):
        model = self.fixture(share_features_extractor=False)
        before = self.snapshot(model)
        with torch.device("meta"):
            first = diagnose_rollout_gradients(model, max_samples=7)
        self.assert_unchanged(before, model)
        second = diagnose_rollout_gradients(model, max_samples=7)
        self.assert_unchanged(before, model)
        self.assertEqual(first, second)
        self.assertEqual(first["sample_flat_indices"], np.linspace(0, 15, 7, dtype=np.int64).tolist())
        self.assertFalse(first["share_features_extractor"])
        self.assertTrue(first["encoder_parameter_sets_disjoint"])
        self.assertTrue(first["cross_encoder_gradients_zero_or_none"])
        self.assertGreater(first["actor_encoder_policy_grad_norm"], 0)
        self.assertGreater(first["critic_encoder_weighted_value_grad_norm"], 0)
        self.assertGreater(first["global_total_grad_norm_preclip"], 0)
        self.assertGreater(first["estimated_global_clip_coefficient"], 0)
        for name in ("shared_policy_grad_norm", "shared_weighted_value_grad_norm",
                     "shared_value_to_policy_norm_ratio", "shared_policy_value_grad_cosine",
                     "shared_policy_value_grad_angle_deg"):
            self.assertIsNone(first[name], name)
        for name in ("actor_encoder_weighted_value_cross_grad_norm",
                     "critic_encoder_policy_cross_grad_norm", "critic_encoder_actor_objective_cross_grad_norm"):
            self.assertEqual(first[name], 0., name)
        for name in ("actor_encoder_value_cross_all_none", "critic_encoder_policy_cross_all_none",
                     "critic_encoder_actor_objective_cross_all_none"):
            self.assertTrue(first[name], name)

    def test_separate_encoder_forward_paths_do_not_share_feature_updates(self):
        model = self.fixture(share_features_extractor=False)
        observations = {key: torch.as_tensor(value[0].copy(), device="cpu")
                        for key, value in model.rollout_buffer.observations.items()}

        def outputs():
            with torch.no_grad():
                actions, values, _ = model.policy(observations, deterministic=True)
                distribution = model.policy.get_distribution(observations)
            return actions.clone(), values.clone(), distribution.distribution.scale.clone()

        original = outputs()
        with torch.no_grad():
            model.policy.vf_features_extractor.fusion[0].bias.add_(.2)
        critic_changed = outputs()
        torch.testing.assert_close(original[0], critic_changed[0], rtol=0, atol=0)
        torch.testing.assert_close(original[2], critic_changed[2], rtol=0, atol=0)
        self.assertFalse(torch.equal(original[1], critic_changed[1]))
        with torch.no_grad():
            model.policy.pi_features_extractor.fusion[0].bias.add_(.2)
        actor_changed = outputs()
        torch.testing.assert_close(critic_changed[1], actor_changed[1], rtol=0, atol=0)
        self.assertFalse(torch.equal(critic_changed[0], actor_changed[0]))

    def test_separate_encoder_parameter_alias_is_rejected_without_changes(self):
        model = self.fixture(share_features_extractor=False)
        # Different modules can still alias a single Parameter. Module identity
        # alone is not a sufficient structural-isolation check.
        model.policy.vf_features_extractor.fusion[0].weight = model.policy.pi_features_extractor.fusion[0].weight
        before = self.snapshot(model)
        with self.assertRaisesRegex(ValueError, "disjoint parameter identities"):
            diagnose_rollout_gradients(model)
        self.assert_unchanged(before, model)

    def test_nonzero_cross_loss_gradient_is_rejected_and_cache_is_restored(self):
        model = self.fixture(share_features_extractor=False)
        before = self.snapshot(model)
        original = model.policy.evaluate_actions

        def coupled(*args, **kwargs):
            values, log_prob, entropy = original(*args, **kwargs)
            value_dependency_on_actor = next(model.policy.pi_features_extractor.parameters()).sum() * .01
            return values + value_dependency_on_actor, log_prob, entropy

        with patch.object(model.policy, "evaluate_actions", side_effect=coupled):
            with self.assertRaisesRegex(ValueError, "nonzero cross-loss gradients"):
                diagnose_rollout_gradients(model)
        self.assert_unchanged(before, model)

    def test_separate_squashed_entropy_fallback_preserves_state(self):
        model = self.fixture(share_features_extractor=False, squash_output=True)
        model.ent_coef = .001
        before = self.snapshot(model)
        diagnostic = diagnose_rollout_gradients(model)
        self.assert_unchanged(before, model)
        self.assertTrue(diagnostic["critic_encoder_actor_objective_cross_all_none"])
        self.assertEqual(diagnostic["critic_encoder_actor_objective_cross_grad_norm"], 0.)
        self.assertTrue(np.isfinite(diagnostic["entropy_loss"]))
        self.assertGreater(diagnostic["global_total_grad_norm_preclip"], 0)

    def test_entropy_only_cross_dependency_is_rejected_without_state_changes(self):
        model = self.fixture(share_features_extractor=False)
        model.ent_coef = .001
        before = self.snapshot(model)
        original = model.policy.evaluate_actions

        def coupled_entropy(*args, **kwargs):
            values, log_prob, entropy = original(*args, **kwargs)
            entropy_dependency_on_critic = next(model.policy.vf_features_extractor.parameters()).sum() * .01
            return values, log_prob, entropy + entropy_dependency_on_critic

        with patch.object(model.policy, "evaluate_actions", side_effect=coupled_entropy):
            with self.assertRaisesRegex(ValueError, "nonzero cross-loss gradients"):
                diagnose_rollout_gradients(model)
        self.assert_unchanged(before, model)

    def test_separate_zero_main_gradients_are_valid(self):
        model = self.fixture(share_features_extractor=False)
        model.vf_coef = 0.
        model.rollout_buffer.advantages[:] = 1.
        before = self.snapshot(model)
        diagnostic = diagnose_rollout_gradients(model)
        self.assert_unchanged(before, model)
        self.assertEqual(diagnostic["actor_encoder_policy_grad_norm"], 0.)
        self.assertEqual(diagnostic["critic_encoder_weighted_value_grad_norm"], 0.)
        self.assertTrue(diagnostic["cross_encoder_gradients_zero_or_none"])

    def test_callback_logs_nonshared_fields_and_json_nulls(self):
        model = self.fixture(share_features_extractor=False)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            model.set_logger(Logger(directory, []))
            callback = AdaptiveTrainingCallback(output, 10000, gradient_diagnostics=True)
            self.addCleanup(callback.handle.close)
            callback.init_callback(model)
            callback.num_timesteps = model.num_timesteps
            before = self.snapshot(model)
            callback.on_rollout_end()
            self.assert_unchanged(before, model)
            row = json.loads((output / "gradient_diagnostics.jsonl").read_text())
            self.assertFalse(row["share_features_extractor"])
            self.assertIsNone(row["shared_value_to_policy_norm_ratio"])
            self.assertTrue(row["cross_encoder_gradients_zero_or_none"])
            self.assertIn("gradient/actor_encoder_policy_grad_norm", model.logger.name_to_value)
            self.assertIn("gradient/critic_encoder_weighted_value_grad_norm", model.logger.name_to_value)

    def test_callback_switch_logging_and_failure_are_explicit(self):
        model = self.fixture()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            model.set_logger(Logger(directory, []))
            for enabled in (False, True):
                callback = AdaptiveTrainingCallback(output, 10000, gradient_diagnostics=enabled)
                self.addCleanup(callback.handle.close)
                callback.init_callback(model)
                callback.num_timesteps = model.num_timesteps
                before = self.snapshot(model)
                callback.on_rollout_end()
                self.assert_unchanged(before, model)
                self.assertEqual((output / "gradient_diagnostics.jsonl").exists(), enabled)
            row = json.loads((output / "gradient_diagnostics.jsonl").read_text())
            self.assertEqual(row["status"], "ok")
            self.assertIn("gradient/global_total_grad_norm_preclip", model.logger.name_to_value)
            with patch("ur7e_xhand_sim.residual_rl.gradient_diagnostics.diagnose_rollout_gradients",
                       side_effect=ValueError("visible failure")):
                with self.assertRaisesRegex(RuntimeError, "gradient diagnostics failed"):
                    callback.on_rollout_end()
            rows = [json.loads(line) for line in (output / "gradient_diagnostics.jsonl").read_text().splitlines()]
            self.assertEqual(rows[-1]["status"], "failed")
            self.assertIn("visible failure", rows[-1]["error"])

    def test_gradient_diagnostic_flag_defaults_off(self):
        args = parser().parse_args(["--conditions-json", "unused", "--config-json", "unused",
                                    "--output", "unused", "--total-timesteps", "1"])
        self.assertFalse(args.gradient_diagnostics)


class MonteCarloTerminalContract(unittest.TestCase):
    def test_181_terminal_returns_are_exact_reward_to_go(self):
        steps, lanes = 181, 3
        observation_space = gym.spaces.Dict({"state": gym.spaces.Box(-np.inf, np.inf, (2,), np.float32)})
        buffer = DictRolloutBuffer(steps, observation_space, gym.spaces.Box(-1, 1, (1,), np.float32),
                                   device="cpu", gamma=1, gae_lambda=1, n_envs=lanes)
        rewards = ((np.arange(steps)[:, None] + np.arange(lanes)[None]) % 5 + 1).astype(np.float32)
        rewards[-1] += [250, -100, 50]
        for step in range(steps):
            buffer.add({"state": np.zeros((lanes, 2), np.float32)}, np.zeros((lanes, 1), np.float32),
                       rewards[step], np.full(lanes, step == 0),
                       torch.tensor((step % 11) + np.arange(lanes) * 2, dtype=torch.float32), torch.zeros(lanes))
        expected = np.flip(np.cumsum(np.flip(rewards, axis=0), axis=0), axis=0)
        for sentinel in ([1e9, -1e9, 123456.], [-777777., 333333., -1e9]):
            buffer.compute_returns_and_advantage(torch.tensor(sentinel), np.ones(lanes, bool))
            np.testing.assert_array_equal(buffer.returns, expected)
            np.testing.assert_array_equal(buffer.returns[0], rewards.sum(0))
        # Check the installed collector's explicit ordering, without copying or
        # invoking a training loop or a physical environment.
        tree = ast.parse(textwrap.dedent(inspect.getsource(OnPolicyAlgorithm.collect_rollouts)))
        calls = {node.func.attr: node.lineno for node in ast.walk(tree)
                 if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                 and node.func.attr in {"compute_returns_and_advantage", "on_rollout_end"}}
        self.assertLess(calls["compute_returns_and_advantage"], calls["on_rollout_end"])


if __name__ == "__main__":
    unittest.main()
