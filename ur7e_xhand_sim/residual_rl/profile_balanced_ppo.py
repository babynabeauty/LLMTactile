"""On-policy PPO with equal material-profile loss weights.

GAE, observations, action distribution and rollout sampling stay unchanged.
Every transition is used once per epoch. Its weight is N/(G * N_group),
computed over the complete current rollout, not independently per minibatch.
Thus each represented profile has equal total actor/value/entropy/KL weight.
The usual SB3 minibatch advantage normalization is deliberately unchanged.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.buffers import DictRolloutBuffer
from stable_baselines3.common.utils import explained_variance


class BalancedSamples(NamedTuple):
    observations: dict
    actions: torch.Tensor
    old_values: torch.Tensor
    old_log_prob: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    profile_ids: torch.Tensor
    weights: torch.Tensor


def profile_weights(ids, n_profiles):
    ids = np.asarray(ids)
    if not np.issubdtype(ids.dtype, np.integer) or np.any(ids < 0) or np.any(ids >= n_profiles):
        raise ValueError('Every transition needs a valid pre-reset profile ID')
    counts = np.bincount(ids.reshape(-1), minlength=n_profiles)
    present = int(np.count_nonzero(counts))
    if present == 0:
        raise ValueError('Empty rollout')
    # A short smoke rollout may miss groups; never fabricate their samples.
    weights = ids.size / (present * counts[ids])
    return weights.astype(np.float32), counts


class ProfileBalancedDictRolloutBuffer(DictRolloutBuffer):
    def __init__(self, *args, n_profiles, **kwargs):
        self.n_profiles = int(n_profiles)
        if self.n_profiles < 1:
            raise ValueError('n_profiles must be positive')
        super().__init__(*args, **kwargs)

    def reset(self):
        super().reset()
        self.profile_ids = np.full((self.buffer_size, self.n_envs), -1, dtype=np.int64)
        self.profile_counts = None

    def get(self, batch_size=None):
        if not self.generator_ready:
            self.weights, self.profile_counts = profile_weights(self.profile_ids, self.n_profiles)
            self.weights = self.swap_and_flatten(self.weights).flatten()
            self.profile_ids = self.swap_and_flatten(self.profile_ids).flatten()
        yield from super().get(batch_size)

    def _get_samples(self, batch_inds, env=None):
        native = super()._get_samples(batch_inds, env)
        return BalancedSamples(*native, self.to_torch(self.profile_ids[batch_inds]),
                               self.to_torch(self.weights[batch_inds]))


def weighted_mean(values, weights):
    # Do NOT divide by minibatch weight.sum(): weights have a global normalizer.
    return (values * weights).mean()


class ProfileBalancedPPO(PPO):
    """SB3 PPO update with profile-balanced reductions; continuous V1 actions."""
    def train(self):
        with torch.device('cpu'):
            self._train_balanced()

    def _train_balanced(self):
        if not isinstance(self.rollout_buffer, ProfileBalancedDictRolloutBuffer):
            raise TypeError('ProfileBalancedPPO requires the tagged rollout buffer')
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)
        clip_vf = self.clip_range_vf(self._current_progress_remaining) if self.clip_range_vf else None
        entropy_losses, pg_losses, value_losses, fractions, kls = [], [], [], [], []
        n_groups = self.rollout_buffer.n_profiles
        kl_sum, kl_count = np.zeros(n_groups), np.zeros(n_groups, dtype=int)
        optimizer_steps, stopped = 0, False
        for epoch in range(self.n_epochs):
            for data in self.rollout_buffer.get(self.batch_size):
                values, log_prob, entropy = self.policy.evaluate_actions(data.observations, data.actions)
                values = values.flatten()
                advantages = data.advantages
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                ratio = torch.exp(log_prob - data.old_log_prob)
                policy_loss = -weighted_mean(torch.minimum(advantages * ratio,
                    advantages * torch.clamp(ratio, 1-clip_range, 1+clip_range)), data.weights)
                values_pred = values if clip_vf is None else data.old_values + torch.clamp(
                    values-data.old_values, -clip_vf, clip_vf)
                value_loss = weighted_mean((data.returns-values_pred).square(), data.weights)
                entropy_loss = weighted_mean(log_prob if entropy is None else -entropy, data.weights)
                loss = policy_loss + self.ent_coef*entropy_loss + self.vf_coef*value_loss
                pg_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy_loss.item())
                fractions.append(weighted_mean((torch.abs(ratio-1)>clip_range).float(), data.weights).item())
                with torch.no_grad():
                    log_ratio = log_prob-data.old_log_prob
                    per_sample_kl = torch.exp(log_ratio)-1-log_ratio
                    kl = weighted_mean(per_sample_kl, data.weights).item()
                    kls.append(kl)
                    for group in range(n_groups):
                        mask = data.profile_ids == group
                        kl_sum[group] += per_sample_kl[mask].sum().item()
                        kl_count[group] += int(mask.sum().item())
                if not torch.isfinite(loss) or not np.isfinite(kl):
                    raise FloatingPointError('Nonfinite balanced PPO loss or KL')
                if self.target_kl is not None and kl > 1.5*self.target_kl:
                    stopped = True
                    if self.verbose:
                        print(f'Early stopping at epoch {epoch}, balanced KL {kl:.4f}', flush=True)
                    break
                self.policy.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()
                optimizer_steps += 1
            self._n_updates += 1
            if stopped:
                break
        metrics = dict(entropy_loss=float(np.mean(entropy_losses)), policy_gradient_loss=float(np.mean(pg_losses)),
            value_loss=float(np.mean(value_losses)), approx_kl=float(np.mean(kls)),
            clip_fraction=float(np.mean(fractions)), loss=loss.item(), optimizer_steps=optimizer_steps,
            early_stopped=stopped, n_updates=self._n_updates,
            explained_variance=float(explained_variance(self.rollout_buffer.values.flatten(),
                                                        self.rollout_buffer.returns.flatten())),
            std=self.policy.log_std.exp().mean().item(), clip_range=clip_range)
        for key, value in metrics.items():
            self.logger.record('train/'+key, value)
        counts = self.rollout_buffer.profile_counts.tolist()
        for group, count in enumerate(counts):
            self.logger.record(f'balance/profile_{group}_transitions', count)
            self.logger.record(f'balance/profile_{group}_kl', kl_sum[group]/max(kl_count[group], 1))
        if getattr(self, 'balance_log_path', None):
            with Path(self.balance_log_path).open('a') as handle:
                handle.write(json.dumps(dict(step=self.num_timesteps, **metrics, profile_counts=counts,
                    profile_kl=(kl_sum/np.maximum(kl_count, 1)).tolist(),
                    missing_profiles=[i for i, c in enumerate(counts) if not c]))+'\n')
