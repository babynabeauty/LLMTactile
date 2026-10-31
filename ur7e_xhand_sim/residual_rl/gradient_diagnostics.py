"""Read-only, pre-update PPO gradient attribution on a deterministic subset.

This is one loss evaluation, not an optimizer/training implementation. It does
not call RolloutBuffer.get(), sample actions, reset exploration noise, backward,
clip gradients, or update parameters. Norms describe this subset, not every
minibatch that the subsequent PPO update will visit.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch.nn import functional as F


def _norm(gradients) -> float:
    return math.sqrt(sum(float(g.detach().double().square().sum())
                         for g in gradients if g is not None))


def diagnose_rollout_gradients(model, max_samples: int = 64) -> dict:
    """Attribute shared or separate encoder gradients without changing PPO state.

    Existing module modes remain untouched. Stateful/stochastic training-mode
    layers are rejected; the current adaptive encoder has neither. Native SB3
    distribution caches are restored, including the stored gSDE noise matrices.
    The caller must invoke this after returns/advantages have been computed.
    """
    policy, buffer = model.policy, model.rollout_buffer
    if not isinstance(max_samples, int) or isinstance(max_samples, bool) or max_samples < 2:
        raise ValueError("Gradient diagnostics require max_samples >= 2")
    if not buffer.full or buffer.generator_ready or not isinstance(buffer.observations, dict):
        raise ValueError("Expected a full, unflattened DictRolloutBuffer before its first PPO update")
    if tuple(model.action_space.shape) != (1,):
        raise ValueError("Gradient diagnostics currently support the scalar adaptive gain action")
    guarded = (torch.nn.modules.batchnorm._BatchNorm, torch.nn.Dropout,
               torch.nn.Dropout1d, torch.nn.Dropout2d, torch.nn.Dropout3d,
               torch.nn.AlphaDropout, torch.nn.FeatureAlphaDropout)
    if any(module.training and isinstance(module, guarded) for module in policy.modules()):
        raise ValueError("Training-mode stochastic/stateful layers are unsupported by read-only diagnostics")
    parameters = tuple(p for p in policy.parameters() if p.requires_grad)
    shared_mode = bool(policy.share_features_extractor)
    actor_encoder = tuple(p for p in policy.pi_features_extractor.parameters() if p.requires_grad)
    critic_encoder = tuple(p for p in policy.vf_features_extractor.parameters() if p.requires_grad)
    if not actor_encoder or not critic_encoder or not parameters or any(p.device.type != "cpu" for p in parameters):
        raise ValueError("Gradient diagnostics require trainable encoders and CPU policy parameters")
    actor_ids, critic_ids = {id(p) for p in actor_encoder}, {id(p) for p in critic_encoder}
    if shared_mode and actor_ids != critic_ids:
        raise ValueError("Shared encoder flag disagrees with actor/critic parameter identities")
    if not shared_mode and actor_ids & critic_ids:
        raise ValueError("Separate actor/critic encoders must have disjoint parameter identities")
    # features_extractor aliases the actor even in SB3's nonshared mode. Use
    # both explicit branches, while global parameters stay natively deduplicated.
    encoders = actor_encoder if shared_mode else actor_encoder + critic_encoder

    size = buffer.buffer_size * buffer.n_envs
    indices = np.linspace(0, size - 1, min(max_samples, size), dtype=np.int64)
    steps, lanes = indices // buffer.n_envs, indices % buffer.n_envs

    def tensor(values):
        chosen = values[steps, lanes].copy()
        if not np.isfinite(chosen).all():
            raise ValueError("Nonfinite rollout data in gradient diagnostic subset")
        return torch.as_tensor(chosen, device=model.device)

    observations = {key: tensor(values) for key, values in buffer.observations.items()}
    actions, old_log_prob = tensor(buffer.actions), tensor(buffer.log_probs)
    advantages, returns, old_values = tensor(buffer.advantages), tensor(buffer.returns), tensor(buffer.values)
    if model.normalize_advantage and len(indices) > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    # collect_rollouts calls the callback before learn updates the progress
    # counter. Evaluate the schedule at the imminent update without assigning it.
    progress = (1 - model.num_timesteps / model._total_timesteps
                if model._total_timesteps > 0 else model._current_progress_remaining)
    clip_range = float(model.clip_range(progress))
    clip_range_vf = None if model.clip_range_vf is None else float(model.clip_range_vf(progress))
    distribution = policy.action_dist
    distribution_state = distribution.__dict__.copy()
    try:
        with torch.enable_grad():
            values, log_prob, entropy = policy.evaluate_actions(observations, actions)
            values = values.flatten()
            ratio = (log_prob - old_log_prob).exp()
            policy_loss = -torch.minimum(advantages * ratio,
                                        advantages * ratio.clamp(1 - clip_range, 1 + clip_range)).mean()
            predicted = (values if clip_range_vf is None else
                         old_values + (values - old_values).clamp(-clip_range_vf, clip_range_vf))
            value_loss = model.vf_coef * F.mse_loss(predicted, returns)
            entropy_loss = log_prob.mean() if entropy is None else -entropy.mean()
            total_loss = policy_loss + value_loss + model.ent_coef * entropy_loss
            if not torch.isfinite(total_loss):
                raise ValueError("Nonfinite gradient diagnostic loss")
            policy_encoder_grad = torch.autograd.grad(policy_loss, encoders, retain_graph=True, allow_unused=True)
            value_encoder_grad = torch.autograd.grad(value_loss, encoders, retain_graph=True, allow_unused=True)
            # The fallback entropy estimate can depend on actor features. Check
            # that complete actor objective as well as the pure policy loss.
            actor_objective_cross = (() if shared_mode else torch.autograd.grad(
                policy_loss + model.ent_coef * entropy_loss, critic_encoder,
                retain_graph=True, allow_unused=True))
            total_grad = torch.autograd.grad(total_loss, parameters, allow_unused=True)
        if any(not torch.isfinite(g).all() for gradients in
               (policy_encoder_grad, value_encoder_grad, actor_objective_cross, total_grad)
               for g in gradients if g is not None):
            raise ValueError("Nonfinite gradient diagnostic derivative")
        actor_count = len(actor_encoder)
        policy_grad = policy_encoder_grad[:actor_count]
        value_grad = value_encoder_grad if shared_mode else value_encoder_grad[actor_count:]
        value_cross = () if shared_mode else value_encoder_grad[:actor_count]
        policy_cross = () if shared_mode else policy_encoder_grad[actor_count:]
        cross_zero = (None if shared_mode else all(g is None or bool((g == 0).all())
            for gradients in (value_cross, policy_cross, actor_objective_cross) for g in gradients))
        if cross_zero is False:
            raise ValueError("Separate encoders have nonzero cross-loss gradients")
        policy_norm, value_norm, total_norm = map(_norm, (policy_grad, value_grad, total_grad))
        dot = (sum(float((a.detach().double() * b.detach().double()).sum())
                   for a, b in zip(policy_grad, value_grad) if a is not None and b is not None)
               if shared_mode else 0.)
        cosine = (max(-1., min(1., dot / (policy_norm * value_norm)))
                  if shared_mode and policy_norm > 0 and value_norm > 0 else None)
        ratio_values = ratio.detach().double()
        log_ratio = (log_prob - old_log_prob).detach().double()
        return {
            "status": "ok", "sample_count": len(indices), "rollout_transition_count": size,
            "selection": "evenly spaced step-major indices; no RNG or buffer.get",
            "sample_flat_indices": indices.tolist(), "policy_training_mode": bool(policy.training),
            "progress_for_imminent_update": float(progress), "clip_range": clip_range,
            "clip_range_vf": clip_range_vf, "normalize_advantage_on_subset": bool(model.normalize_advantage),
            "policy_loss": float(policy_loss.detach()), "weighted_value_loss": float(value_loss.detach()),
            "entropy_loss": float(entropy_loss.detach()), "total_loss": float(total_loss.detach()),
            "vf_coef": float(model.vf_coef), "ent_coef": float(model.ent_coef),
            "share_features_extractor": shared_mode,
            "encoder_gradient_scope": ("same shared encoder parameter space" if shared_mode else
                "disjoint actor/critic encoder spaces; shared ratio and angle are inapplicable; global clipping still couples gradient scales"),
            "actor_encoder_policy_grad_norm": policy_norm,
            "critic_encoder_weighted_value_grad_norm": value_norm,
            "encoder_parameter_sets_disjoint": not bool(actor_ids & critic_ids),
            "actor_encoder_weighted_value_cross_grad_norm": None if shared_mode else _norm(value_cross),
            "critic_encoder_policy_cross_grad_norm": None if shared_mode else _norm(policy_cross),
            "critic_encoder_actor_objective_cross_grad_norm": None if shared_mode else _norm(actor_objective_cross),
            "actor_encoder_value_cross_all_none": None if shared_mode else all(g is None for g in value_cross),
            "critic_encoder_policy_cross_all_none": None if shared_mode else all(g is None for g in policy_cross),
            "critic_encoder_actor_objective_cross_all_none": None if shared_mode else all(g is None for g in actor_objective_cross),
            "cross_encoder_gradients_zero_or_none": cross_zero,
            "cross_gradient_scope": "None means graph-disconnected; a zero tensor alone does not prove structural isolation",
            "shared_policy_grad_norm": policy_norm if shared_mode else None,
            "shared_weighted_value_grad_norm": value_norm if shared_mode else None,
            "shared_value_to_policy_norm_ratio": value_norm / policy_norm if shared_mode and policy_norm > 0 else None,
            "shared_policy_value_grad_cosine": cosine,
            "shared_policy_value_grad_angle_deg": math.degrees(math.acos(cosine)) if cosine is not None else None,
            "global_total_grad_norm_preclip": total_norm, "max_grad_norm": float(model.max_grad_norm),
            "estimated_global_clip_coefficient": min(1., model.max_grad_norm / (total_norm + 1e-6)),
            "ratio_mean": float(ratio_values.mean()), "ratio_min": float(ratio_values.min()),
            "ratio_max": float(ratio_values.max()), "ratio_max_abs_error_from_one": float((ratio_values - 1).abs().max()),
            "clip_fraction": float(((ratio_values - 1).abs() > clip_range).double().mean()),
            "approx_reverse_kl": float(((log_ratio.exp() - 1) - log_ratio).mean()),
            "scope": "pre-update deterministic subset diagnostic; not the full rollout gradient or the later shuffled PPO minibatch gradients",
        }
    finally:
        # evaluate_actions updates this cache even though it does not sample.
        distribution.__dict__.clear()
        distribution.__dict__.update(distribution_state)
