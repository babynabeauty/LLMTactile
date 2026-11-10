"""Opt-in, per-finger correlated exploration for normalized gSDE.

Only the exploration covariance changes: the actor still predicts every joint
independently, and the learned log-std parameter has SB3's original layout.
For each latent feature, the two flexion joints of a finger share correlation
``rho``; different fingers and joints 0/3 remain independent. Equivalently,
their noise is sqrt(rho) * shared + sqrt(1-rho) * independent. Marginal
variances are unchanged, including when learned log-std columns differ.

The full covariance is used for PPO likelihoods and entropy. As in native SB3,
the small likelihood variance floor is not added to sampled exploration noise.
This module never changes the existing normalized policy or its checkpoints.
"""
from __future__ import annotations

import math
from numbers import Integral
from typing import Any, Sequence

import torch
from stable_baselines3.common.distributions import StateDependentNoiseDistribution
from torch.distributions import MultivariateNormal

from .normalized_sde_policy import NormalizedSDEMultiInputPolicy


FINGER_FLEXION_PAIRS = ((1, 2), (4, 5), (6, 7), (8, 9), (10, 11))


def _validate_configuration(action_joints: Sequence[int], correlation: float) -> tuple[tuple[int, ...], float]:
    joints = tuple(action_joints)
    if (not joints or any(isinstance(joint, bool) or not isinstance(joint, Integral)
                          or not 0 <= joint <= 11 for joint in joints)
            or len(set(joints)) != len(joints)):
        raise ValueError("action_joints must contain unique integer hand joint ids in 0..11")
    if isinstance(correlation, bool):
        raise ValueError("finger_noise_correlation must be finite and in [0, 1)")
    rho = float(correlation)
    if not math.isfinite(rho) or not 0 <= rho < 1:
        raise ValueError("finger_noise_correlation must be finite and in [0, 1)")
    return tuple(int(joint) for joint in joints), rho


class FingerStateDependentNoiseDistribution(StateDependentNoiseDistribution):
    """Native gSDE temporal noise, with a fixed per-finger correlation matrix.

    If ``h`` is the normalized noise latent and ``s[i]`` the std vector for its
    i-th feature, action covariance is
    ``sum_i h[i]**2 * diag(s[i]) @ R @ diag(s[i]) + epsilon * I``.
    The matrices sampled by ``sample_weights`` omit only the epsilon term.
    """

    def __init__(self, action_dim: int, *, action_joints: Sequence[int],
                 finger_noise_correlation: float = .8, **kwargs: Any):
        super().__init__(action_dim, **kwargs)
        self.action_joints, self.finger_noise_correlation = _validate_configuration(
            action_joints, finger_noise_correlation)
        if len(self.action_joints) != action_dim:
            raise ValueError("action_joints length must match the action dimension")
        # Plain distribution data, not a policy parameter or persistent buffer.
        # Cache conversion/factorization by the actual log_std device and dtype.
        matrix = [[float(row == col) for col in range(action_dim)]
                  for row in range(action_dim)]
        positions = {joint: index for index, joint in enumerate(self.action_joints)}
        for first, second in FINGER_FLEXION_PAIRS:
            if first in positions and second in positions:
                i, j = positions[first], positions[second]
                matrix[i][j] = matrix[j][i] = self.finger_noise_correlation
        self._correlation_values = matrix
        self._matrix_cache: dict[tuple[torch.device, torch.dtype], tuple[torch.Tensor, torch.Tensor]] = {}

    def _matrices(self, reference: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        key = (reference.device, reference.dtype)
        if key not in self._matrix_cache:
            correlation = torch.tensor(self._correlation_values, device=reference.device,
                                       dtype=reference.dtype)
            self._matrix_cache[key] = correlation, torch.linalg.cholesky(correlation)
        return self._matrix_cache[key]

    def sample_weights(self, log_std: torch.Tensor, batch_size: int = 1) -> None:
        if self.finger_noise_correlation == 0:
            # Preserve native RNG use and arithmetic exactly for the control.
            return super().sample_weights(log_std, batch_size)
        std = self.get_std(log_std)
        _, correlation_cholesky = self._matrices(std)
        # Each latent row is independent. Multiplication by each row's std
        # occurs AFTER correlation, so unequal learned scales keep marginals.
        self.weights_dist = MultivariateNormal(
            torch.zeros_like(std), scale_tril=std.unsqueeze(-1) * correlation_cholesky)
        self.exploration_mat = self.weights_dist.rsample()
        self.exploration_matrices = self.weights_dist.rsample((batch_size,))

    def proba_distribution(self, mean_actions: torch.Tensor, log_std: torch.Tensor,
                           latent_sde: torch.Tensor) -> FingerStateDependentNoiseDistribution:
        if self.finger_noise_correlation == 0:
            return super().proba_distribution(mean_actions, log_std, latent_sde)
        self._latent_sde = latent_sde if self.learn_features else latent_sde.detach()
        std = self.get_std(log_std)
        correlation, _ = self._matrices(std)
        row_covariance = std.unsqueeze(-1) * std.unsqueeze(-2) * correlation
        covariance = torch.mm(self._latent_sde.square(), row_covariance.flatten(1)).reshape(
            -1, self.action_dim, self.action_dim)
        covariance = covariance + self.epsilon * torch.eye(
            self.action_dim, dtype=std.dtype, device=std.device)
        self.distribution = MultivariateNormal(mean_actions, covariance_matrix=covariance)
        return self

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        if self.finger_noise_correlation == 0:
            return super().log_prob(actions)
        gaussian_actions = self.bijector.inverse(actions) if self.bijector is not None else actions
        log_prob = self.distribution.log_prob(gaussian_actions)
        if self.bijector is not None:
            log_prob -= self.bijector.log_prob_correction(gaussian_actions).sum(dim=1)
        return log_prob

    def entropy(self) -> torch.Tensor | None:
        if self.finger_noise_correlation == 0:
            return super().entropy()
        return None if self.bijector is not None else self.distribution.entropy()


class FingerSDEMultiInputPolicy(NormalizedSDEMultiInputPolicy):
    """Normalized gSDE with independent fingers and correlated flexion pairs."""

    def __init__(self, *args: Any, action_joints: Sequence[int],
                 finger_noise_correlation: float = .8, **kwargs: Any):
        self.action_joints, self.finger_noise_correlation = _validate_configuration(
            action_joints, finger_noise_correlation)
        super().__init__(*args, **kwargs)

    def _build(self, lr_schedule: Any) -> None:
        if not isinstance(self.action_dist, StateDependentNoiseDistribution):
            raise ValueError("FingerSDEMultiInputPolicy requires use_sde=True")
        native = self.action_dist
        # Replace before the sole network/optimizer build. Neither the parameter
        # order nor the number/shape of RNG draws differs from native gSDE.
        self.action_dist = FingerStateDependentNoiseDistribution(
            native.action_dim, action_joints=self.action_joints,
            finger_noise_correlation=self.finger_noise_correlation,
            full_std=native.full_std, use_expln=native.use_expln,
            squash_output=native.bijector is not None,
            learn_features=native.learn_features, epsilon=native.epsilon)
        super()._build(lr_schedule)

    def _get_constructor_parameters(self) -> dict[str, Any]:
        parameters = super()._get_constructor_parameters()
        parameters.update(action_joints=list(self.action_joints),
                          finger_noise_correlation=self.finger_noise_correlation)
        return parameters
