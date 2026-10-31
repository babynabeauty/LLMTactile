"""Small causal-window encoder for tactile residual PPO.

There is no object pose, friction, mass, or physics-label input.  Taxel identity
is preserved by the per-finger ordered-array MLP before temporal aggregation.
The temporal convolutions see a window ending at the current observation;
they never receive samples from after the decision time.
"""
from __future__ import annotations

import math

import gymnasium as gym
import torch
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class TactileWindowExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        features_dim: int = 128,
        force_scale_N: float = 0.1,
        use_tactile: bool = True,
    ) -> None:
        super().__init__(observation_space, features_dim)
        shape = observation_space.spaces["tactile"].shape
        if shape != (12, 5, 120, 3):
            raise ValueError(f"Expected tactile (12,5,120,3), received {shape}")
        state_shape = observation_space.spaces["state"].shape
        if len(state_shape) != 1:
            raise ValueError("state must be a flat, physically normalized vector")
        if not math.isfinite(force_scale_N) or force_scale_N <= 0:
            raise ValueError("force_scale_N must be finite and positive")
        self.force_scale_N = float(force_scale_N)
        self.use_tactile = bool(use_tactile)
        self.finger_encoder = nn.Sequential(
            nn.Linear(120 * 3, 64), nn.SiLU(), nn.Linear(64, 32), nn.SiLU(),
        )
        self.temporal_encoder = nn.Sequential(
            nn.Conv1d(5 * 32, 64, kernel_size=3, padding=1), nn.SiLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1), nn.SiLU(),
        )
        self.state_encoder = nn.Sequential(nn.Linear(state_shape[0], 64), nn.SiLU())
        self.fusion = nn.Sequential(nn.Linear(64 * 3, features_dim), nn.SiLU())

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        tactile = observations["tactile"].float()
        if not self.use_tactile:
            tactile = torch.zeros_like(tactile)
        tactile = torch.asinh(tactile / self.force_scale_N)
        batch, history, fingers, _, _ = tactile.shape
        spatial = self.finger_encoder(tactile.reshape(batch * history * fingers, -1))
        spatial = spatial.reshape(batch, history, -1).transpose(1, 2)
        temporal = self.temporal_encoder(spatial)
        state = self.state_encoder(observations["state"].float())
        return self.fusion(torch.cat((temporal[:, :, -1], temporal.mean(dim=2), state), dim=1))


def policy_kwargs(use_tactile: bool = True) -> dict:
    return {
        "features_extractor_class": TactileWindowExtractor,
        "features_extractor_kwargs": {"features_dim": 128, "force_scale_N": 0.1, "use_tactile": use_tactile},
        "net_arch": {"pi": [64, 64], "vf": [64, 64]},
        "activation_fn": nn.SiLU,
        "normalize_images": False,
        # Zero action maps to gain 1.5; use conservative exploration, not a
        # claim that a randomly initialized policy reproduces the baseline.
        "log_std_init": -1.0,
    }
