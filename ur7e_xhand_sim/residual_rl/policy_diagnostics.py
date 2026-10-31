"""CPU-only policy distribution audit on recorded, causal reference observations.

Example:
  python -m ur7e_xhand_sim.residual_rl.policy_diagnostics \
    --trace outputs/.../validation_baselines/fixed_2/trace.npz \
    --models initial=outputs/.../initial_model.zip trained=outputs/.../checkpoints/ppo_000070112.zip \
    --output outputs/.../diagnostics/reference_policy_distributions.json

No environment is constructed and no simulator/privileged arrays are read.
These are counterfactual policy predictions on another controller's observations,
not actual rollouts, success rates, or measured exploration clipping frequencies.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import time

import numpy as np


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ReferenceObservations:
    """Reconstruct obs[j] using prefix and before[1:j+1], never after[j]."""

    def __init__(self, arrays: dict[str, np.ndarray]):
        self.series = {}
        self.n_steps = self.n_lanes = None
        for name in ("tactile", "state"):
            prefix = np.asarray(arrays[f"prefix_{name}"])
            before = np.asarray(arrays[f"{name}_before"])
            if prefix.ndim < 3 or prefix.shape[1] != 12:
                raise ValueError(f"prefix_{name} must be (lanes,12,...)")
            if before.ndim != prefix.ndim + 1 or before.shape[2] != 4 or before.shape[0] < 1:
                raise ValueError(f"{name}_before must be (steps,lanes,4,...)")
            if before.shape[1] != prefix.shape[0] or before.shape[3:] != prefix.shape[2:]:
                raise ValueError(f"Incompatible prefix/{name}_before shapes")
            if not np.array_equal(prefix[:, -4:], before[0]):
                raise ValueError(f"{name}_before[0] must equal the prefix's last four frames")
            if not np.isfinite(prefix).all() or not np.isfinite(before).all():
                raise ValueError(f"Nonfinite {name} observation")
            shape = before.shape[:2]
            if self.n_steps is not None and shape != (self.n_steps, self.n_lanes):
                raise ValueError("Tactile and state step/lane counts differ")
            self.n_steps, self.n_lanes = shape
            # before[0] duplicates the prefix suffix. Each j>=1 supplies four
            # NEW frames ending at decision j, not frames after action j.
            new = np.swapaxes(before[1:], 0, 1).reshape(
                self.n_lanes, (self.n_steps - 1) * 4, *prefix.shape[2:])
            self.series[name] = np.concatenate([prefix, new], axis=1)
        self.count = self.n_steps * self.n_lanes

    @classmethod
    def from_npz(cls, path: Path):
        with np.load(path, allow_pickle=False) as data:
            # Deliberately do not load height, contacts, friction, or other truth.
            arrays = {name: data[name] for name in (
                "prefix_tactile", "prefix_state", "tactile_before", "state_before")}
        return cls(arrays)

    def batch(self, flat_indices) -> dict[str, np.ndarray]:
        indices = np.asarray(flat_indices, dtype=np.int64).reshape(-1)
        if ((indices < 0) | (indices >= self.count)).any():
            raise IndexError("Reference observation index outside trace")
        steps, lanes = indices // self.n_lanes, indices % self.n_lanes
        frames = steps[:, None] * 4 + np.arange(12)[None]
        return {name: values[lanes[:, None], frames] for name, values in self.series.items()}

    def batches(self, batch_size: int):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        for start in range(0, self.count, batch_size):
            indices = np.arange(start, min(start + batch_size, self.count))
            yield indices, self.batch(indices)


def reconstruction_self_test() -> dict:
    """Sentinel timestamps exercise first/second/last windows and no-future use."""
    steps, lanes = 5, 2
    prefix = np.stack([np.arange(-11, 1) + 100 * lane for lane in range(lanes)])[:, :, None]
    before = np.stack([np.stack([
        np.arange(4 * step - 3, 4 * step + 1) + 100 * lane for lane in range(lanes)])
        for step in range(steps)])[:, :, :, None]
    arrays = {"prefix_tactile": prefix, "tactile_before": before,
              "prefix_state": prefix + 1000, "state_before": before + 1000}
    reference = ReferenceObservations(arrays)
    for step in (0, 1, steps - 1):
        obs = reference.batch(np.arange(step * lanes, (step + 1) * lanes))
        expected = np.stack([np.arange(4 * step - 11, 4 * step + 1) + 100 * lane
                             for lane in range(lanes)])[:, :, None]
        np.testing.assert_array_equal(obs["tactile"], expected)
        np.testing.assert_array_equal(obs["state"], expected + 1000)
    altered = {name: value.copy() for name, value in arrays.items()}
    for name in ("tactile_before", "state_before"):
        altered[name][2:] += 99999
    no_future = ReferenceObservations(altered)
    for name in ("tactile", "state"):
        np.testing.assert_array_equal(reference.batch([0, 1, 2, 3])[name],
                                      no_future.batch([0, 1, 2, 3])[name])
    all_indices, pieces = [], {"tactile": [], "state": []}
    for indices, obs in reference.batches(3):
        all_indices.extend(indices.tolist())
        for name in pieces:
            pieces[name].append(obs[name])
    assert all_indices == list(range(steps * lanes))
    for name in pieces:
        np.testing.assert_array_equal(np.concatenate(pieces[name]), reference.batch(all_indices)[name])
    invalid = {name: value.copy() for name, value in arrays.items()}
    invalid["state_before"][0] += 1
    try:
        ReferenceObservations(invalid)
    except ValueError:
        pass
    else:
        raise AssertionError("Mismatched prefix must be rejected")
    return {"passed": True, "checks": ["first_window", "second_window", "last_window",
            "future_perturbation_invariance", "all_lanes_and_partial_batch_order", "prefix_suffix_validation"]}


def summary(values) -> dict:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("Distribution diagnostics must contain finite values")
    quantiles = [0, .01, .05, .25, .5, .75, .95, .99, 1]
    labels = ["min", "p01", "p05", "p25", "p50", "p75", "p95", "p99", "max"]
    return {"count": len(values), "mean": float(values.mean()),
            **dict(zip(labels, np.quantile(values, quantiles).tolist()))}


def diagnose_model(path, reference, gain_bounds, batch_size, latency_samples, latency_warmup):
    import torch
    from stable_baselines3 import PPO

    model_sha = sha256(path)
    model = PPO.load(path, device="cpu")
    policy = model.policy
    policy.set_training_mode(False)
    if policy.squash_output:
        raise ValueError("This clipping audit requires unsquashed Gaussian actions, not tanh-squashed actions")
    if tuple(model.action_space.shape) != (1,) or not (
        np.array_equal(model.action_space.low, [-1.]) and np.array_equal(model.action_space.high, [1.])):
        raise ValueError("Expected the adaptive common-gain action space Box(-1,1,(1,))")
    values = {k: [] for k in ("action_mean_unclipped", "marginal_action_std",
              "theoretical_lower_clip_probability", "theoretical_upper_clip_probability",
              "deterministic_action_unclipped")}
    distribution_class = None
    began = time.perf_counter()
    with torch.inference_mode():
        for _, obs in reference.batches(batch_size):
            tensor_obs, _ = policy.obs_to_tensor(obs)
            distribution = policy.get_distribution(tensor_obs)
            base = distribution.distribution
            if not isinstance(base, torch.distributions.Normal):
                raise TypeError(f"Expected a Gaussian marginal, received {type(base).__name__}")
            mean, std = base.mean, base.stddev
            if mean.ndim != 2 or mean.shape[1] != 1 or not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std <= 0).any():
                raise ValueError("Invalid one-dimensional Gaussian mean/std")
            distribution_class = type(distribution).__module__ + "." + type(distribution).__name__
            lower = torch.special.ndtr((-1 - mean) / std)
            upper = torch.special.ndtr((mean - 1) / std)
            data = (mean, std, lower, upper, distribution.get_actions(deterministic=True))
            for name, item in zip(values, data):
                values[name].append(item.detach().cpu().numpy().reshape(-1))
    inference_seconds = time.perf_counter() - began
    values = {k: np.concatenate(v) for k, v in values.items()}
    values["theoretical_any_clip_probability"] = (values["theoretical_lower_clip_probability"] +
                                                   values["theoretical_upper_clip_probability"])
    clipped = np.clip(values["deterministic_action_unclipped"], -1, 1)
    lo, hi = gain_bounds
    values["deterministic_requested_gain"] = lo * np.exp((clipped + 1) * .5 * math.log(hi / lo))
    metrics = {k: summary(v) for k, v in values.items()}
    metrics["deterministic_clipped_fraction"] = float(np.mean(np.abs(values["deterministic_action_unclipped"]) > 1))
    metrics["deterministic_lower_bound_fraction"] = float(np.mean(clipped <= -1))
    metrics["deterministic_upper_bound_fraction"] = float(np.mean(clipped >= 1))
    # The last observed state's phase is part of the actor input, not a truth label.
    last_states = reference.series["state"][:, 11::4]
    phases = np.moveaxis(last_states, 0, 1).reshape(reference.count, -1)
    by_phase = {}
    if phases.shape[1] == 79:
        phase_bits = phases[:, 74:78]
        labels = np.where(phase_bits.max(-1) > .5, phase_bits.argmax(-1), 4)
        for idx, name in enumerate(("preload", "test_lift", "lift", "hold", "other")):
            mask = labels == idx
            if mask.any():
                by_phase[name] = {k: summary(v[mask]) for k, v in values.items()}
    result = {
        "path": str(path), "sha256": model_sha,
        "policy_class": type(policy).__module__ + "." + type(policy).__name__,
        "distribution_class": distribution_class, "model_num_timesteps": int(model.num_timesteps),
        "uses_tactile": getattr(policy.features_extractor, "use_tactile", None),
        "use_sde": bool(model.use_sde), "sde_sample_freq": int(model.sde_sample_freq),
        "gamma": float(model.gamma), "gae_lambda": float(model.gae_lambda),
        "action_space": {"low": [-1.], "high": [1.]}, "requested_gain_bounds": [lo, hi],
        "method": "policy.get_distribution(obs).distribution.mean/stddev; no manually reconstructed latent/noise formula",
        "theoretical_clip_probability_note": "Gaussian marginal probability at each recorded state, averaged over reference observations; not empirical trajectory clipping. Native gSDE samples are temporally correlated.",
        "requested_gain_note": "Exponential map of clipped deterministic policy action; the environment gain slew limit is not applied and this is not the actual applied gain.",
        "reference_observation_metrics": metrics, "by_observed_phase": by_phase,
        "batch_inference_wall_seconds": inference_seconds,
    }
    if latency_samples:
        indices = np.linspace(0, reference.count - 1, latency_samples, dtype=np.int64)
        timings = []
        with torch.inference_mode():
            for i in range(latency_warmup):
                policy.predict(reference.batch([int(indices[i % len(indices)])]), deterministic=True)
            for index in indices:
                one = reference.batch([int(index)])
                began = time.perf_counter_ns()
                policy.predict(one, deterministic=True)
                timings.append((time.perf_counter_ns() - began) / 1e6)
        result["single_lane_cpu_predict_latency_ms"] = summary(timings)
        result["latency_method"] = {"warmup_calls": latency_warmup, "measured_calls": latency_samples,
            "observation_indices": indices.tolist(), "timed": "policy.predict deterministic=True including tensor conversion and action clipping; observation slicing excluded", 
            "caveat": "Wall-clock CPU timings may include competing processes; no hard real-time guarantee"}
    if sha256(path) != model_sha:
        raise RuntimeError("Checkpoint changed during diagnosis")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--models", nargs="+", help="Distinct name=checkpoint.zip entries")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--latency-samples", type=int, default=0)
    parser.add_argument("--latency-warmup", type=int, default=20)
    parser.add_argument("--gain-min", type=float)
    parser.add_argument("--gain-max", type=float)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    checks = reconstruction_self_test()
    if args.self_test and args.trace is None:
        print(json.dumps(checks, indent=2))
        return
    if args.trace is None or not args.models or args.output is None:
        parser.error("--trace, --models and --output are required unless only --self-test is requested")
    if min(args.batch_size, args.cpu_threads) < 1 or min(args.latency_samples, args.latency_warmup) < 0:
        parser.error("Invalid batch size, thread count, or latency sample count")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing diagnosis: {output}")
    trace = args.trace.resolve()
    manifest_path = trace.parent.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    if (args.gain_min is None) != (args.gain_max is None):
        parser.error("Specify --gain-min and --gain-max together")
    if args.gain_min is None:
        if not all(k in manifest.get("config", {}) for k in ("gain_min", "gain_max")):
            parser.error("Reference manifest has no gain bounds; pass --gain-min and --gain-max")
        gain_bounds = (manifest["config"]["gain_min"], manifest["config"]["gain_max"])
        gain_source = {"path": str(manifest_path), "sha256": sha256(manifest_path)}
    else:
        gain_bounds = (args.gain_min, args.gain_max)
        gain_source = {"source": "explicit CLI bounds"}
    if not (all(math.isfinite(v) for v in gain_bounds) and 0 < gain_bounds[0] < gain_bounds[1]):
        parser.error("Gain bounds must be finite, positive and increasing")
    models = {}
    for item in args.models:
        name, separator, value = item.partition("=")
        if not separator or not name or name in models:
            parser.error("Models must have distinct nonempty name=checkpoint entries")
        models[name] = Path(value).resolve()
    reference_sha = sha256(trace)
    reference = ReferenceObservations.from_npz(trace)
    import torch
    import stable_baselines3
    torch.set_num_threads(args.cpu_threads)
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "kind": "CPU reference-observation policy-distribution diagnosis, NOT a model rollout",
        "trace": str(trace), "trace_sha256": reference_sha,
        "reference_controller": trace.parent.name, "reference_split": manifest.get("split"),
        "observation_steps": reference.n_steps, "lanes": reference.n_lanes,
        "reference_observation_count": reference.count, "observation_order": "step-major, then lane",
        "history_reconstruction": "j=0: prefix[0:12]; j>=1: append before[j]'s 4 CURRENT frames and drop oldest4. before[0] duplicates prefix suffix and is never appended.",
        "causality": "Each window ends at decision j. No future actions, after-action physics, object state, mass or friction are read as observations.",
        "reconstruction_tests": checks, "gain_mapping_source": gain_source,
        "tool_sha256": sha256(Path(__file__)), "torch_version": torch.__version__,
        "stable_baselines3_version": stable_baselines3.__version__, "numpy_version": np.__version__,
        "platform": platform.platform(), "cpu_threads": torch.get_num_threads(),
        "batch_size": args.batch_size, "models": {},
    }
    for name, path in models.items():
        report["models"][name] = diagnose_model(path, reference, gain_bounds, args.batch_size,
                                                 args.latency_samples, args.latency_warmup)
    if sha256(trace) != reference_sha:
        raise RuntimeError("Reference trace changed during diagnosis")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(output), "reference_observations": reference.count,
        "models": {name: {"action_std_p50": item["reference_observation_metrics"]["marginal_action_std"]["p50"],
                          "theoretical_clip_probability_mean": item["reference_observation_metrics"]["theoretical_any_clip_probability"]["mean"],
                          "requested_gain_p50": item["reference_observation_metrics"]["deterministic_requested_gain"]["p50"]}
                   for name, item in report["models"].items()}}, indent=2))


if __name__ == "__main__":
    main()
