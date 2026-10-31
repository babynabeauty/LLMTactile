"""Match separate actor/critic initialization to an untrained shared PPO.

SB3 changes orthogonal-initialization order when feature extractors are split.
The same seed alone therefore does not produce the same initial function. This
helper copies a shared template's complete policy state into independent native
SB3 extractors, then restores the random streams immediately after template
construction. It never copies an optimizer or a distribution's cached graph.
"""
from __future__ import annotations

import copy
import random
from typing import Any

import numpy as np
import torch
from stable_baselines3 import PPO


def capture_rng_state() -> dict[str, Any]:
    """Snapshot existing streams without initializing CUDA or drawing samples."""
    return {
        "python": random.getstate(),
        "numpy": copy.deepcopy(np.random.get_state()),
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda": ([state.clone() for state in torch.cuda.get_rng_state_all()]
                       if torch.cuda.is_initialized() else None),
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if state["torch_cuda"] is not None:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def build_matched_separate_ppo(policy_class, env, *, policy_kwargs: dict,
                               **ppo_kwargs) -> tuple[PPO, dict]:
    """Return a native separate-extractor PPO with the shared seed's weights.

    The target optimizer is new and empty. Shared state_dict aliases include
    both ``pi_features_extractor`` and ``vf_features_extractor``; strict loading
    copies those identical values into different Parameter objects. CPU context
    is explicit even when Genesis has changed Torch's global default device.

    The constructor's gSDE caches are intentionally not copied. SB3 resets noise
    at collection start before the first policy sample; restored RNG streams
    make that reset match the shared template. No environment step/reset occurs
    here. Native PPO constructors schedule the same supplied seed on the env
    twice; actual environment reset remains owned by ``learn``.
    """
    if str(ppo_kwargs.get("device", "cpu")) != "cpu":
        raise ValueError("Matched separate initialization requires CPU construction")
    if policy_kwargs.get("share_features_extractor", True) is not True:
        raise ValueError("Pass shared-template policy_kwargs; the helper splits the target")
    ppo_kwargs = {**ppo_kwargs, "device": "cpu"}
    shared_kwargs = copy.deepcopy(policy_kwargs)
    shared_kwargs["share_features_extractor"] = True
    separate_kwargs = copy.deepcopy(shared_kwargs)
    separate_kwargs["share_features_extractor"] = False
    with torch.device("cpu"):
        template = PPO(policy_class, env, policy_kwargs=shared_kwargs, **ppo_kwargs)
        rng_after_template = capture_rng_state()
        try:
            target = PPO(policy_class, env, policy_kwargs=separate_kwargs, **ppo_kwargs)
            target.policy.load_state_dict(template.policy.state_dict(), strict=True)
            actor = target.policy.pi_features_extractor
            critic = target.policy.vf_features_extractor
            if actor is critic or {id(p) for p in actor.parameters()} & {id(p) for p in critic.parameters()}:
                raise RuntimeError("Separate feature extractors unexpectedly share parameters")
            if target.policy.optimizer.state:
                raise RuntimeError("Matched initialization requires a fresh empty optimizer")
            metadata = {
                "method": "shared_template_strict_policy_state_copy",
                "template_seed": ppo_kwargs.get("seed"),
                "template_policy_class": type(template.policy).__name__,
                "template_share_features_extractor": True,
                "target_share_features_extractor": False,
                "strict_state_dict_copy": True,
                "template_unique_parameter_count": sum(p.numel() for p in template.policy.parameters()),
                "target_unique_parameter_count": sum(p.numel() for p in target.policy.parameters()),
                "construction_device_context": "cpu",
                "rng_restored_to": "immediately after shared template construction",
                "rng_streams_restored": ["python", "numpy", "torch_cpu"] +
                    (["torch_cuda_all"] if rng_after_template["torch_cuda"] is not None else []),
                "optimizer": "new target optimizer with empty state; template optimizer not copied",
                "sde_cache": "not copied; native collect_rollouts resets noise before sampling using restored RNG",
                "environment": "no step/reset; both native constructors schedule the same supplied seed",
                "template_lifetime": "released before returning; no template graph or cache retained",
            }
        finally:
            restore_rng_state(rng_after_template)
            del template
    return target, metadata
