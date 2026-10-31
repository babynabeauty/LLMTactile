"""CPU optimizer-epsilon contracts; no simulator, rollout or training run."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from .adaptive_policy import policy_kwargs
from .adaptive_train import (optimizer_metadata, parser, training_policy_kwargs,
                             validate_inputs, validate_resume_optimizer)
from .matched_initialization import build_matched_separate_ppo, capture_rng_state
from .normalized_sde_policy import NormalizedSDEMultiInputPolicy
from .test_matched_initialization import SyntheticEnv


class AdamEpsilonContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)

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

    def model(self, kwargs, *, separate=False):
        env = DummyVecEnv([SyntheticEnv, SyntheticEnv])
        self.addCleanup(env.close)
        options = dict(n_steps=8, batch_size=4, seed=0, gamma=1., gae_lambda=1.,
                       use_sde=True, sde_sample_freq=30, device="cpu")
        with torch.device("cpu"):
            if separate:
                model, _ = build_matched_separate_ppo(
                    NormalizedSDEMultiInputPolicy, env, policy_kwargs=kwargs, **options)
            else:
                model = PPO(NormalizedSDEMultiInputPolicy, env, policy_kwargs=kwargs, **options)
        return model

    def test_omitted_epsilon_keeps_native_kwargs_weights_rng_and_optimizer(self):
        original_kwargs = policy_kwargs(use_tactile=True, action_std_init=.12)
        current_kwargs = training_policy_kwargs(use_tactile=True, action_std_init=.12)
        self.assertEqual(original_kwargs, current_kwargs)
        self.assertNotIn("optimizer_kwargs", current_kwargs)
        for separate in (False, True):
            with self.subTest(separate=separate):
                original = self.model(original_kwargs, separate=separate)
                original_rng = capture_rng_state()
                current = self.model(current_kwargs, separate=separate)
                self.assert_nested_equal(original_rng, capture_rng_state())
                self.assert_nested_equal(original.policy.state_dict(), current.policy.state_dict())
                self.assert_nested_equal(original.policy.optimizer.state_dict(), current.policy.optimizer.state_dict())
                self.assertEqual(optimizer_metadata(current), {
                    "optimizer_class": "torch.optim.adam.Adam", "optimizer_param_group_eps": [1e-5]})

    def test_explicit_epsilon_changes_only_optimizer_configuration(self):
        for separate in (False, True):
            with self.subTest(separate=separate):
                baseline = self.model(training_policy_kwargs(use_tactile=True, action_std_init=.12), separate=separate)
                original_rng = capture_rng_state()
                changed = self.model(training_policy_kwargs(use_tactile=True, action_std_init=.12, adam_eps=1e-8),
                                     separate=separate)
                self.assert_nested_equal(original_rng, capture_rng_state())
                self.assert_nested_equal(baseline.policy.state_dict(), changed.policy.state_dict())
                expected = copy.deepcopy(baseline.policy.optimizer.state_dict())
                for group in expected["param_groups"]:
                    group["eps"] = 1e-8
                self.assert_nested_equal(expected, changed.policy.optimizer.state_dict())
                self.assertEqual(optimizer_metadata(changed)["optimizer_param_group_eps"], [1e-8])
                self.assertEqual(baseline.policy.action_dist.exploration_matrices.tolist(),
                                 changed.policy.action_dist.exploration_matrices.tolist())

    def test_save_load_retains_epsilon_state_and_resume_never_overwrites(self):
        model = self.model(training_policy_kwargs(use_tactile=False, action_std_init=.12, adam_eps=1e-8), separate=True)
        parameter = next(model.policy.parameters())
        # Populate a synthetic Adam state without taking an optimizer step.
        model.policy.optimizer.state[parameter] = {
            "step": torch.tensor(3.), "exp_avg": torch.full_like(parameter, .02),
            "exp_avg_sq": torch.full_like(parameter, .03)}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.zip"
            model.save(path)
            with torch.device("cpu"):
                loaded = PPO.load(path, env=model.env, device="cpu")
            self.assert_nested_equal(model.policy.optimizer.state_dict(), loaded.policy.optimizer.state_dict())
            self.assertEqual(optimizer_metadata(loaded)["optimizer_param_group_eps"], [1e-8])
            before = copy.deepcopy(loaded.policy.optimizer.state_dict())
            validate_resume_optimizer(loaded, None)
            validate_resume_optimizer(loaded, 1e-8)
            with self.assertRaisesRegex(ValueError, "must match every saved"):
                validate_resume_optimizer(loaded, 1e-5)
            self.assert_nested_equal(before, loaded.policy.optimizer.state_dict())

    def test_resume_checks_each_group_and_optimizer_class(self):
        model = self.model(training_policy_kwargs(use_tactile=True, action_std_init=.12))
        parameters = list(model.policy.parameters())
        model.policy.optimizer = torch.optim.Adam([
            {"params": parameters[:1], "eps": 1e-8},
            {"params": parameters[1:], "eps": 1e-5}])
        self.assertEqual(optimizer_metadata(model)["optimizer_param_group_eps"], [1e-8, 1e-5])
        validate_resume_optimizer(model, None)
        with self.assertRaisesRegex(ValueError, "must match every saved"):
            validate_resume_optimizer(model, 1e-8)
        model.policy.optimizer = torch.optim.SGD(parameters, lr=.01)
        validate_resume_optimizer(model, None)
        with self.assertRaisesRegex(ValueError, "requires an Adam optimizer"):
            validate_resume_optimizer(model, 1e-8)

    def test_cli_default_and_invalid_values(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "conditions.json").write_text(json.dumps([[.1, .8], [.2, .8]]))
            (root / "config.json").write_text("{}")
            base = ["--conditions-json", str(root / "conditions.json"),
                    "--config-json", str(root / "config.json"),
                    "--output", str(root / "out"), "--total-timesteps", "256"]
            args = parser().parse_args(base)
            self.assertIsNone(args.adam_eps)
            validate_inputs(args)
            for value in ("0", "-1", "nan", "inf", "-inf"):
                with self.subTest(value=value), self.assertRaisesRegex(ValueError, "adam-eps"):
                    validate_inputs(parser().parse_args(base + ["--adam-eps=" + value]))
            args = parser().parse_args(base + ["--adam-eps=1e-8"])
            validate_inputs(args)
            self.assertEqual(args.adam_eps, 1e-8)
        for value in (0., -1., float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "adam-eps"):
                training_policy_kwargs(use_tactile=True, action_std_init=.12, adam_eps=value)


if __name__ == "__main__":
    unittest.main()
