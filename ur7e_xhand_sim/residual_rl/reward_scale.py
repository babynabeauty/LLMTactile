"""Fixed reward-unit conversion for PPO training only.

The simulator's reward definition, infos, episode metrics and evaluation remain
in raw units. This wrapper changes only the reward array returned to PPO and
keeps an independent raw copy for the existing episode logger. It does not
normalize observations, adapt the scale, change sampling or touch policy losses.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import math

import numpy as np
from stable_baselines3.common.vec_env import VecEnvWrapper


def validate_reward_scale(value: float) -> float:
    if isinstance(value, (bool, np.bool_)) or not math.isfinite(value) or value <= 0:
        raise ValueError("reward-scale must be finite and positive")
    return float(value)


class ExplicitRewardScaleAction(argparse.Action):
    """Distinguish an omitted default 1.0 from an explicit resume request."""
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)
        setattr(namespace, self.dest + "_explicit", True)


def resolve_training_reward_scale(requested: float = 1.0, *, checkpoint_model=None,
                                   explicitly_requested: bool = False) -> float:
    requested = validate_reward_scale(requested)
    if checkpoint_model is None:
        return requested
    saved = validate_reward_scale(getattr(checkpoint_model, "training_reward_scale", 1.0))
    if explicitly_requested and saved != requested:
        raise ValueError(f"Explicit --reward-scale={requested} must match saved training_reward_scale={saved}")
    return saved


class TrainingRewardScale(VecEnvWrapper):
    """Multiply all emitted reward components uniformly; pass infos unchanged."""
    def __init__(self, venv, scale: float):
        self.scale = validate_reward_scale(scale)
        self.last_raw_rewards = None
        super().__init__(venv)

    def reset(self):
        self.last_raw_rewards = None
        return self.venv.reset()

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        self.last_raw_rewards = rewards.copy()
        if self.scale == 1.0:
            return obs, rewards, dones, infos
        with np.errstate(over="raise", invalid="raise"):
            scaled = rewards * self.scale
        if not np.isfinite(scaled).all():
            raise ValueError("Reward scaling produced nonfinite training rewards")
        return obs, scaled, dones, infos


def wrap_training_rewards(env, scale: float):
    """Preserve the exact existing environment path at the default scale."""
    scale = validate_reward_scale(scale)
    if scale == 1.0:
        return env
    return TrainingRewardScale(env, scale)


@contextmanager
def callback_with_raw_rewards(callback):
    """Give only the parent episode logger raw rewards; never mutate PPO locals.

    SB3's callback locals can alias the live rollout collection dictionary.
    Rebinding this callback's attribute to a shallow copy preserves that object
    and its scaled reward array, including on exceptions or early termination.
    """
    env = callback.training_env
    if not isinstance(env, TrainingRewardScale):
        yield
        return
    if env.last_raw_rewards is None:
        raise RuntimeError("Raw reward logging requires a completed wrapped environment step")
    original = callback.locals
    callback.locals = {**original, "rewards": env.last_raw_rewards}
    try:
        yield
    finally:
        callback.locals = original


def reward_scale_statistics(model) -> dict[str, float]:
    """Read-only RMS diagnostics; targets/values are in training reward units."""
    scale = resolve_training_reward_scale(checkpoint_model=model)
    buffer = model.rollout_buffer

    def rms(values):
        array = np.asarray(values, dtype=np.float64)
        if not np.isfinite(array).all():
            raise ValueError("Nonfinite reward/target/value in reward-scale diagnostics")
        return float(np.sqrt(np.mean(array * array)))

    return {
        "training_reward_scale": scale,
        "raw_reward_rms": rms(buffer.rewards / scale),
        "scaled_reward_rms": rms(buffer.rewards),
        "scaled_return_target_rms": rms(buffer.returns),
        "return_target_raw_units_rms": rms(buffer.returns / scale),
        "value_prediction_rms": rms(buffer.values),
    }
