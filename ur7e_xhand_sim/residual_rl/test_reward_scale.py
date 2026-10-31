"""Synthetic reward-unit and real SB3 collection contracts; no Genesis."""
import copy
import json
from pathlib import Path
import pickle
from types import SimpleNamespace
import tempfile
import unittest

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv

from .adaptive_train import AdaptiveTrainingCallback, parser
from .reward_scale import (TrainingRewardScale, callback_with_raw_rewards,
                           resolve_training_reward_scale, reward_scale_statistics,
                           validate_reward_scale, wrap_training_rewards)


class RewardFixture(VecEnv):
    """Deterministic two-lane rewards, including dense/collision/terminal terms."""
    def __init__(self, horizons=(181, 181)):
        self.horizons = np.asarray(horizons)
        self.steps = np.zeros(2, np.int64)
        self.raw_returns = np.zeros(2, np.float64)
        self.raw_history = []
        self.render_mode = None
        super().__init__(2, gym.spaces.Dict({"state": gym.spaces.Box(-1, 1, (1,), np.float32)}),
                         gym.spaces.Box(-1, 1, (1,), np.float32))

    def reset(self):
        self.steps[:] = 0
        self.raw_returns[:] = 0
        self.raw_history.clear()
        return {"state": np.zeros((2, 1), np.float32)}

    def step_async(self, actions):
        pass

    def step_wait(self):
        self.steps += 1
        dones = self.steps == self.horizons
        rewards = np.array([1., -2.], np.float32)  # Dense reward.
        rewards -= 20 * (self.steps == 80)         # Collision-like penalty.
        rewards += np.array([25., -75.], np.float32) * dones
        self.raw_returns += rewards
        infos = [{"sentinel": {"lane": i}} for i in range(2)]
        for i in range(2):
            if dones[i]:
                metrics = {"return": float(self.raw_returns[i]), "task_success": True,
                           "strict_success": True, "success": True}
                infos[i].update(episode_metrics=metrics, episode={"r": metrics["return"], "l": int(self.steps[i])},
                                **{"TimeLimit.truncated": False})
                self.steps[i] = 0
                self.raw_returns[i] = 0
        obs = {"state": np.zeros((2, 1), np.float32)}
        self.last_result = obs, rewards, dones, infos
        self.raw_history.append(rewards.copy())
        return self.last_result

    def close(self):
        pass

    def get_attr(self, attr_name, indices=None):
        return [getattr(self, attr_name)] * len(self._get_indices(indices))

    def set_attr(self, attr_name, value, indices=None):
        setattr(self, attr_name, value)

    def env_method(self, method_name, *args, indices=None, **kwargs):
        raise NotImplementedError

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * len(self._get_indices(indices))


class RewardScaleContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)

    def test_default_is_exact_passthrough_and_nonunit_never_mutates_arrays_or_infos(self):
        raw = RewardFixture((1, 1))
        self.assertIs(wrap_training_rewards(raw, 1.), raw)
        unit = TrainingRewardScale(raw, 1.)
        obs, rewards, dones, infos = unit.step_wait()
        for expected, actual in zip(raw.last_result, (obs, rewards, dones, infos)):
            self.assertIs(expected, actual)
        before = pickle.dumps(raw.last_result)
        self.assertEqual(before, pickle.dumps((obs, rewards, dones, infos)))
        scaled = TrainingRewardScale(raw, .01)
        obs, rewards, dones, infos = scaled.step_wait()
        raw_obs, raw_rewards, raw_dones, raw_infos = raw.last_result
        self.assertIs(obs, raw_obs)
        self.assertIs(dones, raw_dones)
        self.assertIs(infos, raw_infos)
        self.assertFalse(np.shares_memory(rewards, raw_rewards))
        self.assertFalse(np.shares_memory(scaled.last_raw_rewards, raw_rewards))
        raw_bytes = raw_rewards.tobytes()
        info_bytes = pickle.dumps(infos)
        np.testing.assert_array_equal(rewards, raw_rewards * .01)
        rewards[:] = 999
        self.assertEqual(raw_rewards.tobytes(), raw_bytes)
        self.assertEqual(pickle.dumps(infos), info_bytes)
        self.assertEqual(infos[0]["episode_metrics"]["return"], 26.)

    def test_callback_rebinding_restores_identity_for_false_and_exception(self):
        env = TrainingRewardScale(RewardFixture(), .01)
        _, scaled, dones, infos = env.step_wait()
        original = {"rewards": scaled, "dones": dones, "infos": infos}
        callback = SimpleNamespace(training_env=env, locals=original)
        before = pickle.dumps(original)

        def stop():
            with callback_with_raw_rewards(callback):
                self.assertIsNot(callback.locals, original)
                self.assertIs(callback.locals["infos"], infos)
                self.assertIs(callback.locals["rewards"], env.last_raw_rewards)
                return False

        self.assertFalse(stop())
        self.assertIs(callback.locals, original)
        self.assertEqual(before, pickle.dumps(original))
        with self.assertRaisesRegex(RuntimeError, "injected"):
            with callback_with_raw_rewards(callback):
                raise RuntimeError("injected")
        self.assertIs(callback.locals, original)
        self.assertEqual(before, pickle.dumps(original))

    def collect(self, scale, *, horizons=(181, 181), steps=181):
        raw = RewardFixture(horizons)
        env = wrap_training_rewards(raw, scale)
        with torch.device("cpu"):
            model = PPO("MultiInputPolicy", env, n_steps=steps, batch_size=2 * steps,
                        gamma=1., gae_lambda=1., seed=0, device="cpu")
        model.training_reward_scale = scale
        with torch.no_grad():
            model.policy.value_net.weight.zero_()
            model.policy.value_net.bias.fill_(7.)  # Nonzero V must not be scaled.
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        callback = AdaptiveTrainingCallback(Path(directory.name), 1000000)
        self.addCleanup(callback.handle.close)
        _, callback = model._setup_learn(2 * steps, callback=callback, reset_num_timesteps=True)
        callback.on_training_start({}, {})
        self.assertTrue(model.collect_rollouts(env, callback, model.rollout_buffer, n_rollout_steps=steps))
        self.assertEqual(model._n_updates, 0)  # Collection only, no optimizer update.
        records = [json.loads(s) for s in (Path(directory.name) / "episodes.jsonl").read_text().splitlines()]
        return model, raw, callback, records

    def test_real_181_step_mc_returns_and_advantages_with_nonzero_values(self):
        raw_model, raw, _, raw_records = self.collect(1.)
        raw_rewards = np.stack(raw.raw_history)
        reward_to_go = np.flip(np.cumsum(np.flip(raw_rewards, axis=0), axis=0), axis=0)
        np.testing.assert_array_equal(raw_model.rollout_buffer.returns, reward_to_go)
        for scale in (.125, .01):
            with self.subTest(scale=scale):
                model, env, callback, records = self.collect(scale)
                np.testing.assert_array_equal(np.stack(env.raw_history), raw_rewards)
                np.testing.assert_array_equal(model.rollout_buffer.rewards, raw_rewards * scale)
                np.testing.assert_array_equal(model.rollout_buffer.values, np.full((181, 2), 7.))
                expected = reward_to_go * scale
                if scale == .125:
                    np.testing.assert_array_equal(model.rollout_buffer.returns, expected)
                    np.testing.assert_array_equal(model.rollout_buffer.advantages, expected - 7.)
                else:
                    # Decimal 0.01 and delta=(r+V)-V incur float32 roundoff.
                    # Bound two value-sized ulps per timestep, not a relative
                    # tolerance that could hide terminal credit attenuation.
                    roundoff_bound = 181 * 2 * np.spacing(np.float32(7.))
                    np.testing.assert_allclose(model.rollout_buffer.returns, expected, atol=roundoff_bound, rtol=0)
                    np.testing.assert_allclose(model.rollout_buffer.advantages, expected - 7., atol=roundoff_bound, rtol=0)
                self.assertFalse(np.allclose(model.rollout_buffer.advantages,
                                             raw_model.rollout_buffer.advantages * scale))
                self.assertEqual(records, raw_records)
                self.assertTrue(all(r["return"] == r["episode_metrics"]["return"] for r in records))
                self.assertTrue(all(info["episode"]["r"] == info["episode_metrics"]["return"]
                                    for info in callback.locals["infos"]))
                self.assertEqual(list(model.ep_info_buffer), list(raw_model.ep_info_buffer))
                np.testing.assert_array_equal(callback.locals["rewards"], raw_rewards[-1] * scale)
                before = pickle.dumps(model.rollout_buffer)
                stats = reward_scale_statistics(model)
                self.assertEqual(before, pickle.dumps(model.rollout_buffer))
                self.assertEqual(stats["value_prediction_rms"], 7.)
                self.assertAlmostEqual(stats["scaled_return_target_rms"],
                    float(np.sqrt(np.mean(expected.astype(np.float64) ** 2))), places=4)

    def test_two_lane_asynchronous_done_preserves_raw_episode_logs(self):
        _, _, _, original = self.collect(1., horizons=(2, 3), steps=6)
        model, _, _, scaled = self.collect(.01, horizons=(2, 3), steps=6)
        self.assertEqual(original, scaled)
        self.assertEqual([r["env_index"] for r in scaled], [0, 1, 0, 0, 1])
        self.assertTrue(all(r["return"] == r["episode_metrics"]["return"] for r in scaled))
        self.assertTrue(all(abs(x["r"]) > 1 for x in model.ep_info_buffer))

    def test_save_load_resume_scale_and_wrapper_agree_including_legacy(self):
        raw = RewardFixture()
        with torch.device("cpu"):
            model = PPO("MultiInputPolicy", raw, n_steps=2, batch_size=4, seed=0, device="cpu")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.zip"
            model.training_reward_scale = .01
            model.save(path)
            with torch.device("cpu"):
                loaded = PPO.load(path, env=raw, device="cpu")
            scale = resolve_training_reward_scale(1., checkpoint_model=loaded, explicitly_requested=False)
            self.assertEqual(scale, .01)
            loaded.set_env(wrap_training_rewards(raw, scale))
            self.assertEqual(loaded.training_reward_scale, loaded.get_env().scale)
            self.assertEqual(resolve_training_reward_scale(.01, checkpoint_model=loaded, explicitly_requested=True), .01)
            with self.assertRaisesRegex(ValueError, "must match saved"):
                resolve_training_reward_scale(1., checkpoint_model=loaded, explicitly_requested=True)
            del model.training_reward_scale
            model.save(path)
            with torch.device("cpu"):
                legacy = PPO.load(path, env=raw, device="cpu")
            self.assertEqual(resolve_training_reward_scale(checkpoint_model=legacy), 1.)
            with self.assertRaisesRegex(ValueError, "must match saved"):
                resolve_training_reward_scale(.01, checkpoint_model=legacy, explicitly_requested=True)

    def test_cli_default_explicit_marker_and_invalid_scales(self):
        base = ["--conditions-json", "conditions.json", "--config-json", "config.json",
                "--output", "out", "--total-timesteps", "1"]
        default = parser().parse_args(base)
        explicit = parser().parse_args(base + ["--reward-scale", "1"])
        self.assertEqual(default.reward_scale, 1.)
        self.assertFalse(default.reward_scale_explicit)
        self.assertEqual(explicit.reward_scale, 1.)
        self.assertTrue(explicit.reward_scale_explicit)
        for value in (0., -1., float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "reward-scale"):
                validate_reward_scale(value)
            with self.subTest(wrapper=value), self.assertRaisesRegex(ValueError, "reward-scale"):
                TrainingRewardScale(RewardFixture(), value)


if __name__ == "__main__":
    unittest.main()
