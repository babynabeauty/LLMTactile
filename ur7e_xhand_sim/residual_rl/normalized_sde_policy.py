"""Optional SB3 gSDE policy with unit-length exploration features.

Only the features passed to the native state-dependent noise distribution are
normalized. The action mean and value network retain their original features;
sampling, log probabilities, PPO buffers and action clipping remain SB3's own.
For unit-length features and equal noise-weight standard deviations ``s``, the
marginal action standard deviation is sqrt(s**2 + distribution.epsilon).
Learned unequal weights retain dependence on feature direction.

This is an explicit new policy class, never a patch to existing checkpoints.
"""
from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from stable_baselines3.common.distributions import StateDependentNoiseDistribution
from stable_baselines3.common.policies import MultiInputActorCriticPolicy


class NormalizedSDEMultiInputPolicy(MultiInputActorCriticPolicy):
    """Normalize only gSDE noise features; require an explicit ``use_sde=True``.

    ``sde_normalization_eps`` controls division near zero, independently of the
    native distribution's variance floor. It must be a positive normal float32
    value, matching SB3's policy dtype. Zero features stay zero: their actual
    sampled noise is zero, although SB3 retains its likelihood variance floor.
    Features below this norm have length below one, so exact noise-scale
    invariance only holds above the normalization floor. No random fallback is
    introduced. SB3's default ``learn_features=False`` detaches the noise path.

    Set ``log_std_init`` explicitly for each new experiment. Reusing a native
    gSDE weight std changes the effective exploration scale after normalization.
    """

    normalize_sde_latent = True

    def __init__(self, *args: Any, sde_normalization_eps: float = 1e-8, **kwargs: Any):
        if isinstance(sde_normalization_eps, bool):
            raise ValueError("sde_normalization_eps must be finite and positive")
        self.sde_normalization_eps = float(sde_normalization_eps)
        if not math.isfinite(self.sde_normalization_eps) or self.sde_normalization_eps <= 0:
            raise ValueError("sde_normalization_eps must be finite and positive")
        limits = torch.finfo(torch.float32)
        if not limits.tiny <= self.sde_normalization_eps <= limits.max:
            raise ValueError("sde_normalization_eps must be finite and positive in float32's normal range")
        super().__init__(*args, **kwargs)
        if not isinstance(self.action_dist, StateDependentNoiseDistribution):
            raise ValueError("NormalizedSDEMultiInputPolicy requires use_sde=True")

    def _get_action_dist_from_latent(self, latent_pi: torch.Tensor) -> StateDependentNoiseDistribution:
        if not isinstance(self.action_dist, StateDependentNoiseDistribution):
            raise TypeError("NormalizedSDEMultiInputPolicy requires a native gSDE distribution")
        mean_actions = self.action_net(latent_pi)
        noise_latent = F.normalize(latent_pi, p=2, dim=-1, eps=self.sde_normalization_eps)
        return self.action_dist.proba_distribution(mean_actions, self.log_std, noise_latent)

    def _get_constructor_parameters(self) -> dict[str, Any]:
        parameters = super()._get_constructor_parameters()
        parameters.update(
            sde_normalization_eps=self.sde_normalization_eps,
            # SB3 2.9 omits this in standalone Policy.save constructor data.
            share_features_extractor=self.share_features_extractor,
        )
        return parameters
