"""Train common-gain adaptive PPO from explicit frozen material conditions.

Example:
  .venv/bin/python -m ur7e_xhand_sim.residual_rl.adaptive_train \
      --conditions-json conditions.json --config-json config.json \
      --output outputs/adaptive_train_seed0 --total-timesteps 65536 --seed 0

This command trains only. Evaluate frozen checkpoints in an independent process.
No Genesis import/initialization occurs during --help or JSON argument checks.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import sys
import traceback

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.logger import HumanOutputFormat, Logger
from stable_baselines3.common.vec_env import VecCheckNan

from .adaptive_policy import AdaptiveTactileExtractor, policy_kwargs
from .reward_scale import (ExplicitRewardScaleAction, callback_with_raw_rewards,
                           resolve_training_reward_scale, reward_scale_statistics,
                           validate_reward_scale, wrap_training_rewards)
from .train import (DEMO, ROOT, JsonMetricsWriter, TrainingCallback, configure_seed,
                    digest, space_spec, write_json)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    result.add_argument("--conditions-json", required=True, type=Path,
                        help="JSON list of one [mass_kg, friction] pair per parallel lane")
    result.add_argument("--config-json", required=True, type=Path, help="JSON object passed to AdaptiveConfig")
    result.add_argument("--output", required=True, type=Path, help="New output directory, never overwritten")
    result.add_argument("--total-timesteps", required=True, type=int)
    result.add_argument("--seed", type=int, default=0)
    result.add_argument("--n-steps", type=int, default=128)
    result.add_argument("--batch-size", type=int, default=256)
    result.add_argument("--n-epochs", type=int, default=5)
    result.add_argument("--action-std-init", type=float, default=.08,
                        help="Initial action std, or latent noise-weight std when --use-sde is enabled")
    result.add_argument("--use-sde", action=argparse.BooleanOptionalAction, default=None,
                        help="Use SB3 generalized state-dependent exploration; defaults off for new models, saved setting for resume")
    result.add_argument("--normalize-sde-latent", action=argparse.BooleanOptionalAction, default=None,
                        help="L2-normalize only gSDE noise features in new models; requires --use-sde; resume keeps the saved policy")
    result.add_argument("--sde-sample-freq", type=int, default=30,
                        help="gSDE resampling interval in vector steps; -1 means rollout starts only; saved setting wins on resume")
    result.add_argument("--initial-gain", type=float, default=None,
                        help="Initialize the deterministic policy to this constant common gain; new models only")
    result.add_argument("--learning-rate", type=float, default=3e-4)
    result.add_argument("--adam-eps", type=float, default=None,
                        help="Optional positive Adam epsilon for new models; omitted keeps SB3 default (1e-5); resume preserves saved values and rejects conflicts")
    result.set_defaults(reward_scale_explicit=False)
    result.add_argument("--reward-scale", type=float, default=1.0, action=ExplicitRewardScaleAction,
                        help="Fixed positive training reward multiplier; raw episode metrics/evaluation stay unchanged; resume preserves saved scale and rejects explicit conflicts")
    result.add_argument("--gamma", type=float, default=.995,
                        help="Reward discount factor in (0,1]; saved checkpoint value wins on resume")
    result.add_argument("--gae-lambda", type=float, default=.95,
                        help="GAE trace coefficient in [0,1]; saved checkpoint value wins on resume")
    result.add_argument("--gradient-diagnostics", action="store_true",
                        help="Log read-only shared policy/value gradient attribution on 64 deterministic rollout samples")
    result.add_argument("--separate-feature-extractors", action=argparse.BooleanOptionalAction, default=None,
                        help="Use independent actor/value encoders with matched shared-template initialization; defaults shared for new models, saved setting for resume")
    result.add_argument("--ablate-tactile", action="store_true")
    result.add_argument("--checkpoint-every", type=int, default=10000)
    result.add_argument("--resume", type=Path,
                        help="Restore an adaptive checkpoint and optimizer; saved PPO hyperparameters/std take precedence")
    return result


def validate_inputs(args) -> tuple[np.ndarray, dict]:
    """Validate CLI and material/config documents without importing Genesis."""
    if args.seed < 0:
        raise ValueError("seed must be nonnegative")
    counts = (args.total_timesteps, args.n_steps, args.batch_size, args.n_epochs, args.checkpoint_every)
    if min(counts) < 1 or args.batch_size < 2:
        raise ValueError("Training counts must be positive; PPO batch-size must be >= 2")
    if args.sde_sample_freq != -1 and args.sde_sample_freq < 1:
        raise ValueError("sde-sample-freq must be positive or -1 for rollout-start resampling")
    if not math.isfinite(args.gamma) or not 0 < args.gamma <= 1:
        raise ValueError("gamma must be finite and in (0, 1]")
    if not math.isfinite(args.gae_lambda) or not 0 <= args.gae_lambda <= 1:
        raise ValueError("gae-lambda must be finite and in [0, 1]")
    validate_adam_eps(args.adam_eps)
    validate_reward_scale(args.reward_scale)
    if args.resume is None and args.normalize_sde_latent and not args.use_sde:
        raise ValueError("--normalize-sde-latent requires --use-sde for a new model")
    for name in ("action_std_init", "learning_rate"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if args.initial_gain is not None:
        if not math.isfinite(args.initial_gain):
            raise ValueError("initial-gain must be finite")
        if args.resume is not None:
            raise ValueError("--initial-gain cannot be combined with --resume")
    args.conditions_json = args.conditions_json.expanduser().resolve()
    args.config_json = args.config_json.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    if args.resume is not None:
        args.resume = args.resume.expanduser().resolve()
        if not args.resume.is_file():
            raise FileNotFoundError(args.resume)
    raw_conditions = json.loads(args.conditions_json.read_text())
    if not isinstance(raw_conditions, list) or not raw_conditions:
        raise ValueError("conditions-json must contain a nonempty list of [mass_kg, friction] pairs")
    if any(not isinstance(pair, list) or len(pair) != 2 or
           any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in pair)
           for pair in raw_conditions):
        raise ValueError("Every material condition must contain exactly two numeric values")
    conditions = np.asarray(raw_conditions, dtype=np.float64)
    if not np.isfinite(conditions).all() or np.any(conditions[:, 0] <= 0):
        raise ValueError("Material values must be finite, with mass > 0 kg")
    if np.any((conditions[:, 1] < .01) | (conditions[:, 1] > 5)):
        raise ValueError("Friction must lie in Genesis [0.01,5]")
    # A loaded checkpoint owns its rollout shape. New runs must have full mini-batches.
    if args.resume is None and (args.n_steps * len(conditions)) % args.batch_size:
        raise ValueError("n_steps * number_of_conditions must be divisible by batch_size")
    config = json.loads(args.config_json.read_text())
    if not isinstance(config, dict):
        raise ValueError("config-json must contain an AdaptiveConfig object")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0
           for v in config.values()):
        raise ValueError("AdaptiveConfig values must be finite nonnegative numbers")
    lo, hi = config.get("gain_min", .6), config.get("gain_max", 3.)
    if not (0 < lo <= 1 <= hi and lo < hi):
        raise ValueError("Gain bounds must include 1 and have positive logarithmic range")
    if args.initial_gain is not None and not lo <= args.initial_gain <= hi:
        raise ValueError(f"initial-gain must lie within configured [{lo}, {hi}]")
    return conditions, config


def snapshot_manifest(args, conditions, requested_config) -> dict:
    """Capture sources before environment construction; snapshots stay immutable."""
    source_paths = set((ROOT / "ur7e_xhand_sim").rglob("*.py"))
    source_paths.update((ROOT / "Genesis/genesis").rglob("*.py"))
    source_paths.add(Path(__file__).parent / "requirements.txt")
    hashes = {}
    for source in sorted(source_paths):
        if "__pycache__" in source.parts or not source.is_file():
            continue
        relative = source.relative_to(ROOT)
        destination = args.output / "source_snapshot" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        data = source.read_bytes()
        destination.write_bytes(data)
        hashes[str(relative)] = digest(destination)
    for name, source in (("conditions.input.json", args.conditions_json), ("config.input.json", args.config_json)):
        (args.output / name).write_bytes(source.read_bytes())
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "arguments": vars(args),
        "variant": "adaptive_common_gain", "n_envs": len(conditions),
        "conditions_kg_mu": conditions.tolist(), "requested_config": requested_config,
        "input_sha256": {"conditions": digest(args.output / "conditions.input.json"),
                         "config": digest(args.output / "config.input.json")},
        "source_sha256": hashes,
        "source_scope": "all Python under ur7e_xhand_sim and Genesis/genesis, plus residual RL requirements",
        "demonstration": str(DEMO),
        "demo_sha256": {name: digest(DEMO / name) for name in ("reference.json", "control.jsonl")},
        "input_checkpoint_sha256": digest(args.resume) if args.resume else None,
        "python": platform.python_version(),
        "versions": {p: importlib.metadata.version(p) for p in ("torch", "numpy", "gymnasium", "stable-baselines3")},
        "policy_device": "cpu", "torch_threads": 4,
        "optimizer": {
            "adam_eps_requested": args.adam_eps,
            "epsilon_contract": "Omitted leaves native SB3 optimizer construction unchanged (Adam eps=1e-5); explicit positive eps affects only new optimizer configuration. Resume preserves actual saved param-group eps and rejects an explicit mismatch.",
        },
        "training_reward": {
            "requested_scale": args.reward_scale,
            "explicitly_requested": args.reward_scale_explicit,
            "scope": "TrainingRewardScale changes only the reward array passed to PPO; all dense, collision, and terminal components use the same multiplier. Scale 1 skips the wrapper.",
            "raw_logging": "infos, episode_metrics, episode.r, episodes.jsonl return and rollout/ep_rew_mean remain original raw rewards; the parent episode callback receives a separate raw-reward locals view.",
            "target_units": "PPO rollout rewards, value targets, advantages and value predictions use training reward units. RMS raw-unit targets are recovered by dividing by the fixed scale; they need not equal unbootstrapped Monte Carlo returns outside gamma=lambda=1 full episodes.",
            "resume": "Saved model.training_reward_scale is authoritative; missing legacy attribute means 1. Omitted CLI scale preserves it, explicit mismatches fail.",
        },
        "exploration": {
            "use_sde_requested": args.use_sde,
            "normalize_sde_latent_requested": args.normalize_sde_latent,
            "sde_sample_freq_requested": args.sde_sample_freq,
            "action_std_init_semantics": "With gSDE this initializes latent noise-weight std, not marginal action std; actual loaded distribution is recorded in actual_model_config.",
            "sampling": "SB3 native policy sampling/log_prob/rollout buffer; no additional action noise or post-policy random offset.",
            "gSDE_schedule": "Resample at every rollout start and local vector steps 0,f,2f,... when f>0; no automatic episode-done resampling. Each lane has its own noise matrix."},
        "policy": {"encoder": "AdaptiveTactileExtractor", "uses_tactile": not args.ablate_tactile,
                   "uses_shear": True, "features_dim": 128,
                   "separate_feature_extractors_requested": args.separate_feature_extractors,
                   "feature_initialization": "Separate models copy an untrained same-seed shared policy strictly, retain a fresh optimizer, and restore post-template RNG; shared default is unchanged.",
                   "history": "12 synchronous 60Hz tactile and robot-state frames; identical state history for ablation",
                   "input_boundary": "measured tactile, robot state, executed gains and nominal plan; no object ground truth",
                   "force_transform": "fixed asinh scales and backward differences; no VecNormalize",
                   "ablation": "zero tactile before raw/summary/difference paths" if args.ablate_tactile else None},
        "rates_hz": {"physics": 300, "tactile": 60, "action": 15},
        "episode_metrics": {
            "task_success": "original final 3s hold clearance >=5cm at every 60Hz sample, no robot collision",
            "strict_success": "task_success plus >=2 contact-bearing fingertip links throughout hold",
            "success": "legacy top-level success means strict_success; inspect both labels",
            "training_terminal_reward": "task_success"},
        "resume_note": "Weights/optimizer and saved PPO hyperparameters (including feature sharing, gamma, gae_lambda and exploration settings) resume; explicit conflicting feature-sharing or Adam-eps flags are rejected, and CLI gamma/gae-lambda do not override the checkpoint. Environment uses this run's explicit config/conditions; no old rollout buffer is reused.",
        "evaluation_note": "Training metrics are not held-out evidence; evaluate frozen models independently with matched conditions.",
    }
    write_json(args.output / "manifest.json", manifest)
    return manifest


def make_env(args, conditions, requested_config):
    from .adaptive_env import AdaptiveConfig, AdaptiveVecEnv
    config = AdaptiveConfig(**requested_config)
    env = AdaptiveVecEnv(n_envs=len(conditions), seed=args.seed,
                         output_dir=args.output / "environment", config=config,
                         material_conditions=conditions, mode="train")
    return VecCheckNan(env, raise_exception=True, check_inf=True), config


def initialize_constant_gain(model, env, config, gain: float) -> None:
    """Set the Gaussian mean using the environment's exact gain-to-action map.

    Only the final actor layer changes. Exploration standard deviation remains
    unchanged, and executed gains still obey the environment's rate limit.
    """
    if not math.isfinite(gain) or not config.gain_min <= gain <= config.gain_max:
        raise ValueError("initial-gain must be finite and within resolved config bounds")
    action = np.asarray(env.unwrapped.action_for_gain(gain), dtype=np.float32).reshape(-1)
    head = model.policy.action_net
    if action.size != 1 or not np.isfinite(action).all() or np.any(np.abs(action) > 1 + 1e-6):
        raise ValueError("Environment gain mapping must produce one finite action in [-1, 1]")
    if not isinstance(head, torch.nn.Linear) or head.out_features != 1 or head.bias is None:
        raise ValueError("Constant common gain requires a scalar linear policy action head")
    with torch.no_grad():
        head.weight.zero_()
        head.bias.copy_(torch.as_tensor(action, dtype=head.bias.dtype, device=head.bias.device))


def validate_resume_exploration(model, requested_use_sde: bool | None,
                                requested_normalize_sde_latent: bool | None = None) -> None:
    """Saved exploration settings win; an explicit incompatible mode is an error."""
    if requested_use_sde is not None and bool(model.use_sde) != requested_use_sde:
        raise ValueError("Explicit --use-sde/--no-use-sde must match the resumed checkpoint")
    normalized = bool(getattr(model.policy, "normalize_sde_latent", False))
    if requested_normalize_sde_latent is not None and normalized != requested_normalize_sde_latent:
        raise ValueError("Explicit --normalize-sde-latent/--no-normalize-sde-latent must match the resumed checkpoint")


def validate_resume_extractors(model, requested_separate: bool | None,
                               requested_ablate_tactile: bool) -> None:
    """Validate both encoders; never silently change a checkpoint's topology."""
    shared = bool(model.policy.share_features_extractor)
    if requested_separate is not None and shared == requested_separate:
        raise ValueError("Explicit --separate-feature-extractors/--no-separate-feature-extractors must match the resumed checkpoint")
    for branch in ("pi_features_extractor", "vf_features_extractor"):
        extractor = getattr(model.policy, branch)
        if not isinstance(extractor, AdaptiveTactileExtractor):
            raise ValueError(f"resume requires AdaptiveTactileExtractor in {branch}")
        if extractor.use_tactile != (not requested_ablate_tactile):
            raise ValueError(f"--ablate-tactile must match both resumed encoders ({branch})")
        if not extractor.use_shear:
            raise ValueError(f"This CLI expects full measured XYZ in {branch}; a normal-only checkpoint is incompatible")


def validate_adam_eps(value: float | None) -> None:
    if value is not None and (not math.isfinite(value) or value <= 0):
        raise ValueError("adam-eps must be finite and positive when specified")


def training_policy_kwargs(*, use_tactile: bool, action_std_init: float,
                           adam_eps: float | None = None) -> dict:
    """Leave SB3's default optimizer kwargs absent unless explicitly requested."""
    validate_adam_eps(adam_eps)
    kwargs = policy_kwargs(use_tactile=use_tactile, action_std_init=action_std_init)
    if adam_eps is not None:
        kwargs["optimizer_kwargs"] = {"eps": float(adam_eps)}
    return kwargs


def optimizer_metadata(model) -> dict:
    optimizer = model.policy.optimizer
    cls = type(optimizer)
    return {
        "optimizer_class": f"{cls.__module__}.{cls.__qualname__}",
        "optimizer_param_group_eps": [float(group["eps"]) if "eps" in group else None
                                      for group in optimizer.param_groups],
    }


def validate_resume_optimizer(model, requested_adam_eps: float | None) -> None:
    """Saved optimizer state owns epsilon; never silently overwrite its groups."""
    validate_adam_eps(requested_adam_eps)
    if requested_adam_eps is None:
        return
    optimizer = model.policy.optimizer
    if not isinstance(optimizer, torch.optim.Adam):
        raise ValueError("Explicit --adam-eps requires an Adam optimizer in the resumed checkpoint")
    actual = optimizer_metadata(model)["optimizer_param_group_eps"]
    if not actual or any(value != requested_adam_eps for value in actual):
        raise ValueError(f"Explicit --adam-eps={requested_adam_eps} must match every saved optimizer param-group eps: {actual}")


def distribution_metadata(model) -> dict:
    """Record parameter std separately from state-dependent marginal action std."""
    policy, use_sde = model.policy, bool(model.use_sde)
    distribution = policy.action_dist
    parameter_std = distribution.get_std(policy.log_std) if use_sde else policy.log_std.exp()
    result = {
        "class": type(distribution).__name__, "use_sde": use_sde,
        "sde_sample_freq": model.sde_sample_freq,
        "squash_output": bool(policy.squash_output),
        "log_std_shape": list(policy.log_std.shape),
        "log_std": policy.log_std.detach().cpu().tolist(),
        "parameter_std": parameter_std.detach().cpu().tolist(),
        "parameter_std_semantics": "latent noise-weight std" if use_sde else "marginal action std",
        "normalize_sde_latent": bool(getattr(policy, "normalize_sde_latent", False)),
        "sde_normalization_eps": getattr(policy, "sde_normalization_eps", None),
    }
    if use_sde:
        result.update(latent_sde_dim=distribution.latent_sde_dim,
                      full_std=distribution.full_std, use_expln=distribution.use_expln,
                      learn_features=distribution.learn_features, epsilon=distribution.epsilon,
                      noise_latent="L2-normalized actor latent" if result["normalize_sde_latent"] else "unmodified actor latent",
                      marginal_action_std="sqrt(sum(noise_latent**2 * noise_weight_std**2) + epsilon); equal weights give fixed std for unit-length noise features")
    return result


class AdaptiveTrainingCallback(TrainingCallback):
    def __init__(self, output, checkpoint_every, gradient_diagnostics=False):
        super().__init__(output, checkpoint_every)
        self.gradient_diagnostics = gradient_diagnostics
        self.task_successes = 0
        self.strict_successes = 0

    def _on_step(self) -> bool:
        for done, info in zip(self.locals["dones"], self.locals["infos"]):
            if done:
                metrics = info["episode_metrics"]
                self.task_successes += bool(metrics["task_success"])
                self.strict_successes += bool(metrics["strict_success"])
        with callback_with_raw_rewards(self):
            return super()._on_step()

    def _on_rollout_end(self) -> None:
        if self.gradient_diagnostics:
            from .gradient_diagnostics import diagnose_rollout_gradients
            try:
                diagnostic = diagnose_rollout_gradients(self.model)
            except Exception as exc:
                failure = {"status": "failed", "timesteps": self.num_timesteps,
                           "error": repr(exc), "traceback": traceback.format_exc()}
                with (self.output / "gradient_diagnostics.jsonl").open("a") as handle:
                    handle.write(json.dumps(failure, allow_nan=False) + "\n")
                self.logger.record("gradient/diagnostic_failed", 1)
                raise RuntimeError("Read-only gradient diagnostics failed; see gradient_diagnostics.jsonl") from exc
            diagnostic["timesteps"] = self.num_timesteps
            with (self.output / "gradient_diagnostics.jsonl").open("a") as handle:
                handle.write(json.dumps(diagnostic, allow_nan=False) + "\n")
            for name, value in diagnostic.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    self.logger.record("gradient/" + name, value)
        self.logger.record("rollout/cumulative_task_success_rate", self.task_successes / max(self.episode_count, 1))
        self.logger.record("rollout/cumulative_strict_success_rate", self.strict_successes / max(self.episode_count, 1))
        # Only this new mode adds unit diagnostics; the default scale-1 logging
        # and training path remain unchanged.
        if getattr(self.model, "training_reward_scale", 1.) != 1.:
            for name, value in reward_scale_statistics(self.model).items():
                self.logger.record("rollout/" + name, value)
        write_json(self.output / "status.json", {
            "status": "running", "timesteps": self.num_timesteps, "completed_episodes": self.episode_count,
            "task_successes": self.task_successes, "strict_successes": self.strict_successes,
            "updated_utc": datetime.now(timezone.utc).isoformat()})


def run(args, conditions, requested_config) -> None:
    env, callback, logger, model = None, None, None, None
    try:
        manifest = snapshot_manifest(args, conditions, requested_config)
        configure_seed(args.seed, 4)
        env, config = make_env(args, conditions, requested_config)
        if env.observation_space.spaces["tactile"].shape != (12, 5, 120, 3) or \
           env.observation_space.spaces["state"].shape != (12, 79) or env.action_space.shape != (1,):
            raise ValueError("Adaptive environment must provide tactile(12,5,120,3), state(12,79), action(1,)")
        manifest.update(observation_space=space_spec(env.observation_space), action_space=space_spec(env.action_space),
                        config=asdict(config), zero_action_gain=math.sqrt(config.gain_min * config.gain_max))
        write_json(args.output / "config.resolved.json", asdict(config))
        write_json(args.output / "conditions.resolved.json", conditions.tolist())
        default_device_before_policy = str(torch.get_default_device())
        if args.resume:
            # Genesis sets Torch's default device to GPU. PPO moves the policy
            # after construction, so device="cpu" alone does not fix init RNG.
            with torch.device("cpu"):
                model = PPO.load(args.resume, env=env, device="cpu")
            validate_resume_exploration(model, args.use_sde, args.normalize_sde_latent)
            validate_resume_extractors(model, args.separate_feature_extractors, args.ablate_tactile)
            validate_resume_optimizer(model, args.adam_eps)
            training_reward_scale = resolve_training_reward_scale(
                args.reward_scale, checkpoint_model=model, explicitly_requested=args.reward_scale_explicit)
            if training_reward_scale != 1.:
                env = wrap_training_rewards(env, training_reward_scale)
                model.set_env(env, force_reset=True)
            initialization = {"method": "checkpoint_restore", "checkpoint_sha256": digest(args.resume),
                              "optimizer": "restored from checkpoint", "topology": "preserved from checkpoint"}
            model.set_random_seed(args.seed)
            if model.n_steps * len(conditions) % model.batch_size:
                raise ValueError("Saved rollout/batch sizes must divide the new lane count")
        else:
            training_reward_scale = resolve_training_reward_scale(args.reward_scale)
            env = wrap_training_rewards(env, training_reward_scale)
            policy_class = "MultiInputPolicy"
            if args.normalize_sde_latent:
                from .normalized_sde_policy import NormalizedSDEMultiInputPolicy
                policy_class = NormalizedSDEMultiInputPolicy
            construction_kwargs = dict(
                n_steps=args.n_steps, batch_size=args.batch_size, n_epochs=args.n_epochs,
                learning_rate=args.learning_rate, gamma=args.gamma, gae_lambda=args.gae_lambda, clip_range=.2,
                ent_coef=.001, vf_coef=.5, max_grad_norm=.5, target_kl=.03,
                use_sde=bool(args.use_sde), sde_sample_freq=args.sde_sample_freq,
                seed=args.seed, device="cpu", verbose=1)
            extractor_kwargs = training_policy_kwargs(use_tactile=not args.ablate_tactile,
                                                      action_std_init=args.action_std_init, adam_eps=args.adam_eps)
            with torch.device("cpu"):
                if args.separate_feature_extractors:
                    from .matched_initialization import build_matched_separate_ppo
                    model, initialization = build_matched_separate_ppo(
                        policy_class, env, policy_kwargs=extractor_kwargs, **construction_kwargs)
                else:
                    model = PPO(policy_class, env, policy_kwargs=extractor_kwargs, **construction_kwargs)
                    initialization = {"method": "native_shared_policy_construction",
                                      "optimizer": "new empty optimizer", "construction_device_context": "cpu"}
            if args.initial_gain is not None:
                initialize_constant_gain(model, env, config, args.initial_gain)
        model.training_reward_scale = training_reward_scale
        initial_timesteps = model.num_timesteps
        model.save(args.output / "initial_model")
        actual_model = {
            "parameter_count": sum(p.numel() for p in model.policy.parameters()),
            "device": str(model.device), "uses_tactile": model.policy.features_extractor.use_tactile,
            "share_features_extractor": bool(model.policy.share_features_extractor),
            "separate_feature_extractors_requested": args.separate_feature_extractors,
            "extractor_uses_tactile": {branch: getattr(model.policy, branch).use_tactile
                                      for branch in ("pi_features_extractor", "vf_features_extractor")},
            "initialization": initialization,
            "adam_eps_requested": args.adam_eps,
            "reward_scale_requested": args.reward_scale,
            "reward_scale_explicitly_requested": args.reward_scale_explicit,
            "training_reward_scale": model.training_reward_scale,
            "episode_return_units": "raw unscaled environment rewards; rollout/ep_rew_mean is also raw",
            "value_target_units": "training rewards multiplied by training_reward_scale",
            **optimizer_metadata(model),
            "optimization_coupling": "Native single PPO optimizer and global gradient-norm clipping remain unchanged; separate extractors isolate representation gradients only.",
            "torch_default_device_before_policy": default_device_before_policy,
            "policy_construction_device_context": "cpu",
            "torch_default_dtype_at_policy_construction": str(torch.get_default_dtype()),
            "policy_parameter_devices_at_run_start": sorted({str(p.device) for p in model.policy.parameters()}),
            "use_sde": bool(model.use_sde), "sde_sample_freq": model.sde_sample_freq,
            "normalize_sde_latent": bool(getattr(model.policy, "normalize_sde_latent", False)),
            "action_std_init_requested": args.action_std_init,
            "action_std_at_run_start": None if model.use_sde else model.policy.log_std.detach().exp().cpu().tolist(),
            "policy_distribution": distribution_metadata(model),
            "initial_gain_requested": args.initial_gain,
            "initial_action_bias": model.policy.action_net.bias.detach().cpu().tolist(),
            "initial_action_weight_zeroed": args.initial_gain is not None,
            "n_steps": model.n_steps, "batch_size": model.batch_size, "n_epochs": model.n_epochs,
            "learning_rate": model.learning_rate, "gamma_requested": args.gamma,
            "gae_lambda_requested": args.gae_lambda,
            "gamma": model.gamma, "gae_lambda": model.gae_lambda, "ent_coef": model.ent_coef,
            "vf_coef": model.vf_coef, "max_grad_norm": model.max_grad_norm,
            "gradient_diagnostics": bool(args.gradient_diagnostics),
            "gradient_diagnostic_samples": 64 if args.gradient_diagnostics else None,
            "resumed": bool(args.resume), "timesteps_at_run_start": initial_timesteps,
            "initial_checkpoint_sha256": digest(args.output / "initial_model.zip"),
        }
        manifest["actual_model_config"] = actual_model
        write_json(args.output / "model_config.json", actual_model)
        write_json(args.output / "manifest.json", manifest)
        logger = Logger(str(args.output), [HumanOutputFormat(sys.stdout), JsonMetricsWriter(args.output / "training_metrics.jsonl")])
        model.set_logger(logger)
        (args.output / "checkpoints").mkdir()
        callback = AdaptiveTrainingCallback(args.output, args.checkpoint_every, args.gradient_diagnostics)
        model.learn(total_timesteps=args.total_timesteps, callback=callback, log_interval=1,
                    reset_num_timesteps=not bool(args.resume), progress_bar=False)
        model.save(args.output / "final_model")
        write_json(args.output / "status.json", {
            "status": "completed", "timesteps": model.num_timesteps,
            "requested_additional_timesteps": args.total_timesteps,
            "actual_additional_timesteps": model.num_timesteps - initial_timesteps,
            "completed_episodes": callback.episode_count,
            "task_successes": callback.task_successes, "strict_successes": callback.strict_successes,
            "final_checkpoint_sha256": digest(args.output / "final_model.zip")})
    except BaseException as exc:
        save_error = None
        if model is not None:
            try:
                model.save(args.output / "interrupted_model")
            except Exception as checkpoint_error:
                save_error = repr(checkpoint_error)
        write_json(args.output / "status.json", {
            "status": "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
            "error": repr(exc), "traceback": traceback.format_exc(),
            "interrupted_checkpoint_error": save_error,
            "timesteps": model.num_timesteps if model is not None else 0})
        raise
    finally:
        if callback is not None:
            callback.handle.close()
        if logger is not None:
            logger.close()
        if env is not None:
            env.close()


def main(argv=None):
    args = parser().parse_args(argv)
    conditions, config = validate_inputs(args)
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "status.json", {"status": "initializing", "n_envs": len(conditions), "arguments": vars(args)})
    run(args, conditions, config)


if __name__ == "__main__":
    main()
