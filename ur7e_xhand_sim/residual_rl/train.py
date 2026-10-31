"""Train/evaluate tactile residual PPO; all experiments are simulation-only.

Examples (run from repository root):
  .venv/bin/python -m ur7e_xhand_sim.residual_rl.train train --output outputs/rl_smoke --total-timesteps 128 --n-envs 2 --n-steps 64 --batch-size 64
  .venv/bin/python -m ur7e_xhand_sim.residual_rl.train eval --output outputs/rl_eval --model outputs/rl_smoke/final_model.zip --episodes 8
  .venv/bin/python -m ur7e_xhand_sim.residual_rl.train eval --output outputs/rl_fixed --controller nominal --episodes 8

Evaluation uses a separate process/environment, deterministic policy actions,
and per-environment episode quotas. It does not update policy or normalization.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import random
import sys
import time
import traceback

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import HumanOutputFormat, KVWriter, Logger
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import VecCheckNan

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from ur7e_xhand_sim.residual_rl.policy import policy_kwargs
from ur7e_xhand_sim.residual_rl.baselines import ReactiveGripController

DEMO = ROOT / "outputs/rule_demos/teacher_pinch_v2_20261024"


def serializable(value):
    if isinstance(value, dict):
        return {str(k): serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return serializable(value.tolist())
    if isinstance(value, np.generic):
        return serializable(value.item())
    if isinstance(value, torch.Tensor):
        return serializable(value.detach().cpu().numpy())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    return repr(value)


def write_json(path: Path, value) -> None:
    # Atomic status/manifest updates avoid partial records during live review.
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(serializable(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def space_spec(space: gym.Space) -> dict:
    if isinstance(space, gym.spaces.Dict):
        return {"type": "Dict", "spaces": {k: space_spec(v) for k, v in space.spaces.items()}}
    if isinstance(space, gym.spaces.Box):
        return {"type": "Box", "shape": list(space.shape), "dtype": str(space.dtype),
                "low_min": float(np.min(space.low)), "high_max": float(np.max(space.high))}
    return {"type": type(space).__name__, "repr": repr(space)}


def provenance(output: Path, args: argparse.Namespace, env) -> dict:
    source_files = sorted(Path(__file__).parent.glob("*.py"))
    source_files += [Path(__file__).parent / "requirements.txt", ROOT / "ur7e_xhand_sim/environment.py",
                     ROOT / "ur7e_xhand_sim/build_assets.py",
                     ROOT / "Genesis/genesis/engine/sensors/kinematic_tactile.py"]
    source_hashes = {}
    for file in source_files:
        if not file.is_file():
            continue
        relative = str(file.relative_to(ROOT))
        source_hashes[relative] = digest(file)
        destination = output / "source_snapshot" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(file.read_bytes())
    demo_hashes = {name: digest(DEMO / name) for name in ("reference.json", "control.jsonl")}
    versions = {}
    for package in ("stable-baselines3", "gymnasium", "numpy", "torch"):
        versions[package] = importlib.metadata.version(package)
    model_path = getattr(args, "model", None) or getattr(args, "resume", None)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "arguments": vars(args),
        "python": platform.python_version(), "versions": versions,
        "source_sha256": source_hashes, "demonstration": str(DEMO), "demo_sha256": demo_hashes,
        "input_checkpoint_sha256": digest(Path(model_path)) if model_path else None,
        "observation_space": space_spec(env.observation_space), "action_space": space_spec(env.action_space),
        "policy": {"class": "MultiInputPolicy", "encoder": "TactileWindowExtractor",
                   "history_frames": 12, "force_transform": "asinh(force_N / 0.1)",
                   "features_dim": 128, "hidden_pi_vf": [64, 64],
                   "uses_object_ground_truth": False, "uses_risk_head": False,
                   "normalization": "fixed environment state scaling; no VecNormalize"},
        "rates_hz": {"physics": 300, "tactile": 60, "policy": 15},
        "reproducibility_note": "Seeds and sources recorded; GPU physical simulation may not be bitwise deterministic.",
        "evaluation_note": "Training returns are not held-out evaluation. Run eval in a separate process with independent seeds.",
    }
    write_json(output / "manifest.json", manifest)
    return manifest


class JsonMetricsWriter(KVWriter):
    def __init__(self, path: Path):
        self.handle = path.open("w", buffering=1)

    def write(self, key_values, key_excluded, step=0):
        values = {k: v for k, v in key_values.items() if "json" not in key_excluded.get(k, ())}
        record = {"step": int(step), "wall_time_s": time.time(), **values}
        self.handle.write(json.dumps(serializable(record), ensure_ascii=False, allow_nan=False) + "\n")

    def close(self):
        self.handle.close()


def episode_record(info: dict, env_index: int, episode_index: int, reward: float, steps: int) -> dict:
    metrics = info.get("episode_metrics", {})
    return {"episode_index": episode_index, "env_index": env_index, "return": float(reward),
            "policy_steps": int(steps), "duration_s": float(steps / 15),
            "episode_metrics": metrics,
            "success": bool(metrics["success"]) if "success" in metrics else None,
            "truncated": bool(info.get("TimeLimit.truncated", False))}


class TrainingCallback(BaseCallback):
    def __init__(self, output: Path, checkpoint_every: int):
        super().__init__()
        self.output = output
        self.checkpoint_every = checkpoint_every
        self.next_checkpoint = checkpoint_every
        self.episode_count = 0
        self.handle = (output / "episodes.jsonl").open("w", buffering=1)

    def _on_training_start(self) -> None:
        self.returns = np.zeros(self.training_env.num_envs, dtype=np.float64)
        self.lengths = np.zeros(self.training_env.num_envs, dtype=np.int64)
        self.next_checkpoint = self.num_timesteps + self.checkpoint_every

    def _on_step(self) -> bool:
        self.returns += self.locals["rewards"]
        self.lengths += 1
        for i, done in enumerate(self.locals["dones"]):
            if done:
                record = episode_record(self.locals["infos"][i], i, self.episode_count,
                                        self.returns[i], self.lengths[i])
                record["training_timesteps"] = self.num_timesteps
                self.handle.write(json.dumps(serializable(record), ensure_ascii=False, allow_nan=False) + "\n")
                self.episode_count += 1
                self.returns[i] = 0
                self.lengths[i] = 0
        if self.num_timesteps >= self.next_checkpoint:
            self.model.save(self.output / "checkpoints" / f"ppo_{self.num_timesteps:09d}")
            self.next_checkpoint = self.num_timesteps + self.checkpoint_every
        return True

    def _on_rollout_end(self) -> None:
        write_json(self.output / "status.json", {"status": "running", "timesteps": self.num_timesteps,
                   "completed_episodes": self.episode_count, "updated_utc": datetime.now(timezone.utc).isoformat()})


def configure_seed(seed: int, torch_threads: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(torch_threads)
    set_random_seed(seed, using_cuda=torch.cuda.is_available())
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def record_model_inputs(output, model):
    manifest = json.loads((output / "manifest.json").read_text())
    manifest["policy"]["uses_tactile"] = bool(model.policy.features_extractor.use_tactile)
    manifest["policy"]["ablation"] = None if manifest["policy"]["uses_tactile"] else "all tactile tensors replaced with zeros inside encoder; same architecture"
    manifest["policy"]["action_std_on_load_or_initialization"] = model.policy.log_std.detach().exp().cpu().tolist()
    write_json(output / "manifest.json", manifest)


def make_env(args):
    # Import Genesis only for actual simulation, not --help or encoder tests.
    from ur7e_xhand_sim.residual_rl.env import ResidualVecEnv
    env = ResidualVecEnv(n_envs=args.n_envs, seed=args.seed, output_dir=args.output / "environment",
                         mass_range_kg=tuple(args.mass_range_kg), friction_range=tuple(args.friction_range),
                         mode="train" if args.command == "train" else "eval")
    return VecCheckNan(env, raise_exception=True, check_inf=True)


def train(args) -> None:
    if args.n_steps * args.n_envs % args.batch_size:
        raise ValueError("n_steps * n_envs must be divisible by batch_size")
    if args.batch_size <= 1 or args.n_steps * args.n_envs <= 1:
        raise ValueError("PPO requires more than one rollout/batch sample")
    env = None
    callback = None
    logger = None
    model = None
    try:
        env = make_env(args)
        provenance(args.output, args, env)
        if args.resume:
            model = PPO.load(args.resume, env=env, device=args.device)
            model.set_random_seed(args.seed)
        else:
            network_kwargs = policy_kwargs(use_tactile=not args.ablate_tactile)
            if args.action_std_init is not None:
                network_kwargs["log_std_init"] = math.log(args.action_std_init)
            model = PPO("MultiInputPolicy", env, policy_kwargs=network_kwargs,
                        learning_rate=args.learning_rate, n_steps=args.n_steps, batch_size=args.batch_size,
                        n_epochs=args.n_epochs, gamma=args.gamma, gae_lambda=.95, clip_range=.2,
                        ent_coef=args.ent_coef, vf_coef=.5, max_grad_norm=.5, target_kl=.03,
                        seed=args.seed, device=args.device, verbose=1)
        record_model_inputs(args.output, model)
        model.save(args.output / "initial_model")
        write_json(args.output / "model_config.json", {
            "parameter_count": sum(p.numel() for p in model.policy.parameters()),
            "trainable_parameter_count": sum(p.numel() for p in model.policy.parameters() if p.requires_grad),
            "learning_rate": model.learning_rate, "n_steps": model.n_steps, "batch_size": model.batch_size,
            "gamma": model.gamma, "n_epochs": model.n_epochs, "ent_coef": model.ent_coef,
            "device": str(model.device), "resumed": bool(args.resume),
            "uses_tactile": bool(model.policy.features_extractor.use_tactile),
            "action_std_init_requested": args.action_std_init,
            "action_std_at_run_start": model.policy.log_std.detach().exp().cpu().tolist(),
            "initial_checkpoint_sha256": digest(args.output / "initial_model.zip"),
        })
        logger = Logger(str(args.output), [HumanOutputFormat(sys.stdout), JsonMetricsWriter(args.output / "training_metrics.jsonl")])
        model.set_logger(logger)
        (args.output / "checkpoints").mkdir()
        callback = TrainingCallback(args.output, args.checkpoint_every)
        model.learn(total_timesteps=args.total_timesteps, callback=callback, log_interval=1,
                    reset_num_timesteps=not bool(args.resume), progress_bar=False)
        model.save(args.output / "final_model")
        write_json(args.output / "status.json", {"status": "completed", "timesteps": model.num_timesteps,
                   "requested_additional_timesteps": args.total_timesteps, "completed_episodes": callback.episode_count,
                   "final_checkpoint_sha256": digest(args.output / "final_model.zip")})
    except BaseException as exc:
        if model is not None:
            model.save(args.output / "interrupted_model")
        write_json(args.output / "status.json", {"status": "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                   "error": repr(exc), "traceback": traceback.format_exc(),
                   "timesteps": model.num_timesteps if model else 0})
        raise
    finally:
        if callback is not None:
            callback.handle.close()
        if logger is not None:
            logger.close()
        if env is not None:
            env.close()


def evaluate(args) -> None:
    if args.controller == "model" and not args.model:
        raise ValueError("--controller model requires --model")
    if args.controller != "model" and args.model:
        raise ValueError("--model cannot be combined with a fixed controller")
    env = None
    trace = []
    try:
        env = make_env(args)
        base_env = env
        while hasattr(base_env, "venv"):
            base_env = base_env.venv
        provenance(args.output, args, env)
        model = PPO.load(args.model, env=env, device=args.device) if args.controller == "model" else None
        if model is not None:
            record_model_inputs(args.output, model)
        reactive = None
        if args.controller == "reactive":
            reactive = ReactiveGripController(
                [layout["positions_link_m"] for layout in base_env.layouts[:2]], args.n_envs,
                base_env.start_tick / 300, base_env.trajectory.phase_starts["test_lift"])
            write_json(args.output / "reactive_config.json", reactive.config)
        env.seed(args.seed)
        obs = env.reset()
        quotas = np.full(args.n_envs, args.episodes // args.n_envs, dtype=int)
        quotas[: args.episodes % args.n_envs] += 1
        counts = np.zeros(args.n_envs, dtype=int)
        returns = np.zeros(args.n_envs, dtype=np.float64)
        lengths = np.zeros(args.n_envs, dtype=int)
        records = []
        with (args.output / "episodes.jsonl").open("w", buffering=1) as handle, \
             (args.output / "step_trace.jsonl").open("w", buffering=1) as step_handle, \
             (args.output / "reactive_events.jsonl").open("w", buffering=1) as event_handle:
            for vector_step in range(args.max_vector_steps):
                if model is not None:
                    actions, _ = model.predict(obs, deterministic=True)
                elif reactive is not None:
                    actions, events = reactive.predict(obs)
                    for event in events:
                        event.update(episode_index_in_lane=int(counts[event["env_index"]]), vector_step=vector_step)
                        event_handle.write(json.dumps(serializable(event), allow_nan=False) + "\n")
                else:
                    # -1 -> original gain 1; +1 -> stronger gain 2. Zero is 1.5.
                    value = {"nominal": -1., "strong": 1., "mid": 0.}[args.controller]
                    actions = np.full((args.n_envs, *env.action_space.shape), value, dtype=np.float32)
                tactile_norm = np.linalg.norm(obs["tactile"][:, -1], axis=-1)
                before_state = obs["state"].copy()
                obs, rewards, dones, infos = env.step(actions)
                physical = {k: np.array(v, copy=True) for k, v in base_env.last_metrics.items()}
                row = {"vector_step": vector_step + 1, "episode_index_in_lane": counts.copy(),
                       "elapsed_sim_s": np.array([info["elapsed_sim_s"] for info in infos]),
                       "actions": np.array(actions, copy=True),
                       "applied_gains": np.array([info["applied_gains"] for info in infos]),
                       "rewards": np.array(rewards, copy=True), "dones": np.array(dones, copy=True),
                       "policy_state_before": before_state,
                       "tactile_total_N_before": tactile_norm.sum(-1),
                       "tactile_peak_N_before": tactile_norm.max(-1),
                       "tactile_active_taxels_before": (tactile_norm > 1e-5).sum(-1),
                       **physical}
                trace.append(row)
                step_handle.write(json.dumps(serializable(row), allow_nan=False) + "\n")
                returns += rewards
                lengths += 1
                for i, done in enumerate(dones):
                    if not done:
                        continue
                    if counts[i] < quotas[i]:
                        record = episode_record(infos[i], i, len(records), returns[i], lengths[i])
                        record["controller"] = args.controller
                        record["vector_step"] = vector_step + 1
                        if reactive is not None:
                            record["reactive_trigger_sample_time_s"] = reactive.trigger_sample_time_s[i]
                            record["reactive_trigger_action_time_s"] = reactive.trigger_action_time_s[i]
                        handle.write(json.dumps(serializable(record), ensure_ascii=False, allow_nan=False) + "\n")
                        records.append(record)
                        counts[i] += 1
                        write_json(args.output / "status.json", {"status": "running", "controller": args.controller,
                                   "completed_episodes": len(records), "requested_episodes": args.episodes})
                    returns[i] = 0
                    lengths[i] = 0
                if reactive is not None:
                    reactive.reset(dones)
                if np.all(counts >= quotas):
                    break
            else:
                raise RuntimeError(f"Evaluation exceeded {args.max_vector_steps} vector steps; completed {len(records)}/{args.episodes}")
        known = [r for r in records if r["success"] is not None]
        summary = {"status": "completed", "controller": args.controller, "deterministic": True,
                   "seed": args.seed, "episodes": len(records), "per_env_quotas": quotas,
                   "successes": sum(r["success"] for r in known), "episodes_with_success_metric": len(known),
                   "success_rate": sum(r["success"] for r in known) / len(known) if known else None,
                   "mean_return": float(np.mean([r["return"] for r in records])),
                   "mean_policy_steps": float(np.mean([r["policy_steps"] for r in records])),
                   "note": "Per-condition/single-seed results are exploratory; inspect episodes and compare held-out matched-seed baselines."}
        write_json(args.output / "evaluation.json", summary)
        write_json(args.output / "status.json", summary)
        print(json.dumps(serializable(summary), ensure_ascii=False, indent=2))
    except BaseException as exc:
        write_json(args.output / "status.json", {"status": "failed", "error": repr(exc), "traceback": traceback.format_exc()})
        raise
    finally:
        if trace:
            np.savez_compressed(args.output / "trace.npz", **{
                key: np.stack([row[key] for row in trace]) for key in trace[0]})
        if env is not None:
            env.close()


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = result.add_subparsers(dest="command", required=True)
    for name in ("train", "eval"):
        command = commands.add_parser(name)
        command.add_argument("--output", required=True, type=Path, help="New output directory; existing directories are rejected")
        command.add_argument("--n-envs", type=int, default=8)
        command.add_argument("--seed", type=int, default=0 if name == "train" else 10000)
        command.add_argument("--mass-range-kg", nargs=2, type=float, default=(.1, .2), metavar=("MIN", "MAX"))
        command.add_argument("--friction-range", nargs=2, type=float, default=(.8, .8), metavar=("MIN", "MAX"))
        command.add_argument("--device", default="cpu", choices=("cpu", "cuda", "auto"))
        command.add_argument("--torch-threads", type=int, default=4)
        if name == "train":
            command.add_argument("--total-timesteps", type=int, default=100000)
            command.add_argument("--n-steps", type=int, default=128)
            command.add_argument("--batch-size", type=int, default=128)
            command.add_argument("--n-epochs", type=int, default=5)
            command.add_argument("--learning-rate", type=float, default=3e-4)
            command.add_argument("--gamma", type=float, default=.995)
            command.add_argument("--ent-coef", type=float, default=.001)
            command.add_argument("--checkpoint-every", type=int, default=10000)
            command.add_argument("--ablate-tactile", action="store_true", help="Train same architecture with all tactile input zeroed")
            command.add_argument("--action-std-init", type=float, default=None,
                                 help="Initial Gaussian action standard deviation; omitted preserves exp(-1). Cannot override a resumed checkpoint.")
            command.add_argument("--resume", type=Path, help="Resume weights/optimizer and saved PPO hyperparameters into a NEW directory")
        else:
            command.add_argument("--model", type=Path)
            command.add_argument("--controller", choices=("model", "nominal", "strong", "mid", "reactive"), default="model")
            command.add_argument("--episodes", type=int, default=32)
            command.add_argument("--max-vector-steps", type=int, default=100000)
    return result


def main(argv=None) -> None:
    args = parser().parse_args(argv)
    if args.n_envs < 1 or args.torch_threads < 1:
        raise ValueError("n-envs and torch-threads must be positive")
    if args.seed < 0:
        raise ValueError("seed must be nonnegative")
    if not (0 < args.mass_range_kg[0] <= args.mass_range_kg[1]):
        raise ValueError("Mass range must be positive and ordered")
    if not (.01 <= args.friction_range[0] <= args.friction_range[1] <= 5):
        raise ValueError("Friction range must be within Genesis [0.01,5]")
    if args.command == "train":
        if min(args.total_timesteps, args.n_steps, args.batch_size, args.n_epochs, args.checkpoint_every) < 1:
            raise ValueError("Training counts must be positive")
        if not (0 < args.learning_rate and 0 < args.gamma <= 1 and args.ent_coef >= 0):
            raise ValueError("Invalid PPO scalar hyperparameter")
        if args.action_std_init is not None:
            if not math.isfinite(args.action_std_init) or args.action_std_init <= 0:
                raise ValueError("action-std-init must be finite and positive")
            if args.resume:
                raise ValueError("action-std-init cannot override a resumed checkpoint; start a new model")
    elif min(args.episodes, args.max_vector_steps) < 1:
        raise ValueError("Evaluation counts must be positive")
    args.output = args.output.expanduser().resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    configure_seed(args.seed, args.torch_threads)
    write_json(args.output / "status.json", {"status": "initializing", "arguments": vars(args)})
    (train if args.command == "train" else evaluate)(args)


if __name__ == "__main__":
    main()
