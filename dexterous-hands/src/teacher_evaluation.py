"""First-episode accounting, called by Eden before its automatic reset."""

import hashlib
import json
import subprocess
from pathlib import Path
from contextlib import contextmanager
import random
import numpy as np
import torch


@contextmanager
def preserve_evaluation_rng():
    """Keep model construction/loading from changing randomized evaluation resets."""
    python_state, numpy_state = random.getstate(), np.random.get_state()
    try:
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def snapshot_run(config, output, checkpoint=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    config.save_as_file(str(output / "resolved.yaml"), lock_config=False)
    manifest = {"checkpoint": None if checkpoint is None else {"path": str(checkpoint), "sha256": sha256(checkpoint)}}
    manifest["git_revision"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    (output / "working-tree.patch").write_bytes(subprocess.check_output(["git", "diff", "HEAD"]))
    source_files = [*Path("src").rglob("*.py"), *Path("scripts").rglob("*.py"), Path("main.py")]
    manifest["source_hashes"] = {str(p): sha256(p) for p in source_files}
    manifest["grasp_caches"] = {str(p): sha256(p) for p in Path("src/assets/grasps").glob("*.pt")}
    manifest["eden_patch"] = sha256("patches/eden/physics-events-v1.patch")
    import importlib.metadata as metadata

    manifest["dependencies"] = {
        d.metadata["Name"]: {"version": d.version, "source": d.read_text("direct_url.json")}
        for name in sorted({d.metadata["Name"] for d in metadata.distributions() if d.metadata["Name"]})
        for d in [metadata.distribution(name)]
    }
    import tarfile

    with tarfile.open(output / "source.tar.gz", "w:gz") as archive:
        for pattern in (
            "src/**/*.py",
            "conf/**/*.yaml",
            "patches/eden/*",
            "scripts/**/*.py",
            "main.py",
            "uv.lock",
            "pyproject.toml",
        ):
            for path in Path(".").glob(pattern):
                if path.is_file():
                    archive.add(path, arcname=str(path))
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def motion_summary(t, dt):
    valid = t["live"].clone()
    valid[:2] = False
    if "reset" in t:
        valid &= ~t["reset"]
    # Exclude recovery for two seconds after each pulse and goal transitions.
    recovery = torch.zeros(valid.shape[1], dtype=torch.long, device="cpu")
    transition = torch.zeros_like(recovery)
    recovery_mask = torch.zeros_like(valid)
    transition_mask = torch.zeros_like(valid)
    for i in range(len(valid)):
        recovery = torch.where(t["pulse"][i], round(2 / dt), (recovery - 1).clamp_min(0))
        transition = torch.where(t["transition"][i], round(0.25 / dt), (transition - 1).clamp_min(0))
        recovery_mask[i] = t["live"][i] & (recovery > 0)
        transition_mask[i] = t["live"][i] & (transition > 0) & ~recovery_mask[i]
        valid[i] &= (recovery == 0) & (transition == 0)
    acceleration = torch.diff(t["velocity"], dim=0) / dt
    acceleration_rms = acceleration[valid[1:]].square().mean().sqrt().item() if valid[1:].any() else None
    phase_acceleration = {}
    for name, mask in [
        ("ordinary", valid),
        ("goal_transition", transition_mask),
        ("pulse_recovery", recovery_mask),
    ]:
        mask = mask[1:] & t["live"][:-1]
        phase_acceleration[name] = float(acceleration[mask].square().mean().sqrt()) if mask.any() else None
    # Independent contiguous one-second windows; never FFT across resets/gaps.
    hf = []
    width = round(1 / dt)
    for start in range(0, len(valid) - width + 1, width):
        ids = valid[start : start + width].all(0)
        if ids.any():
            x = t["velocity"][start : start + width, ids]
            x = x - x.mean(0)
            window = torch.hann_window(width, device="cpu")
            spectrum = torch.fft.rfft(x * window[:, None, None], dim=0, norm="ortho")
            frequencies = torch.fft.rfftfreq(width, dt, device="cpu")
            weights = torch.full_like(frequencies, 2.0)
            weights[0] = weights[-1] = 1.0
            power = spectrum.abs().square() * weights[:, None, None]
            hf.append(power[frequencies > 5].sum(0).mean() / window.square().sum())
    return dict(
        acceleration_rms=acceleration_rms,
        phase_acceleration_rms=phase_acceleration,
        high_frequency_velocity_rms=None if not hf else float(torch.stack(hf).mean().sqrt()),
        motion_sample_rate_hz=1 / dt,
    )


def repose_tracking_summary(error, omega, live):
    """Separate angular accuracy from settling, excluding post-terminal resets."""
    reached = (error < 0.1) & live
    slow = omega.norm(dim=-1) < 0.25
    samples = live.sum().clamp_min(1)
    return dict(
        mean_orientation_error=float(error[live].mean()) if live.any() else None,
        orientation_in_tolerance_fraction=float(reached.sum() / samples),
        orientation_and_speed_in_tolerance_fraction=float((reached & slow).sum() / samples),
        speed_blocked_fraction_when_orientation_reached=(
            float((reached & ~slow).sum() / reached.sum()) if reached.any() else None
        ),
    )


class EpisodeCollector:
    def __init__(self, env):
        self.finished = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.dropped = torch.zeros_like(self.finished)
        self.steps = torch.zeros(env.num_envs, device=env.device)
        self.turns = torch.zeros_like(self.steps)
        self.backward_turns = torch.zeros_like(self.steps)
        self.rotation_path = torch.zeros_like(self.steps)
        self.sustained_steps = torch.zeros_like(self.steps)
        self.previous_quat = env.entities["obj"].get_quat().clone()
        self.is_rotation = "rotation_axis" in env.command_manager.active_terms
        self.goals = torch.ones_like(self.steps)
        self.successes = torch.zeros_like(self.steps)
        self.lifted = torch.zeros_like(self.finished)
        self.goal_dwell = torch.zeros_like(self.steps)
        self.max_lift = torch.zeros_like(self.steps)
        self.max_displacement = torch.zeros_like(self.steps)
        self.start_pos = env.entities["obj"].get_pos().clone()
        self.start_bottom = (env.entities["obj"].get_AABB()[:, 0, 2].clone()
                             if "goal_pos" in env.command_manager.active_terms else None)
        self.was_active = torch.zeros_like(self.finished)
        self.recovery_time = torch.full_like(self.steps, -1)
        self.recovery_dwell = torch.zeros_like(self.steps)
        self.pulses = torch.zeros_like(self.steps)
        self.recovered = torch.zeros_like(self.steps)
        self.variant = torch.zeros_like(self.steps, dtype=torch.long)
        for variant, geom in enumerate(env.entities["obj"].links[0].geoms):
            self.variant[geom.active_envs_idx] = variant
        self.trace = []
        self.dt = env.dt
        self.physics_dt = getattr(env, "sim_dt", env.dt)
        self.physics_trace = []

    def capture_physics(self, env):
        """Sample the previous completed physics state before the next wrench is applied.

        The pre-step hook samples at 200 Hz, avoiding aliasing actuator chatter
        into the 20/40 Hz control trace. Fresh reset intervals are explicitly masked.
        """
        controller = env.action_manager.get_term("dofs_pos_controller")
        pulse = env.object_wrench.state
        transition = torch.zeros_like(self.finished)
        if "goal_rot" in env.command_manager.active_terms:
            cmd = env.command_manager.get_term("goal_rot")
            transition = cmd._last_success | cmd._last_timeout_reset
        self.physics_trace.append(
            dict(
                velocity=controller.entity.get_dofs_vel(dofs_idx_local=controller.dofs_idx_local).cpu(),
                live=(~self.finished).cpu(),
                reset=(env.episode_length_buf == 0).cpu(),
                pulse=((pulse.force.abs().sum(-1) + pulse.torque.abs().sum(-1)) > 0).cpu(),
                transition=transition.clone().cpu(),
            )
        )
        if "ee_controller" in getattr(env.action_manager, "active_terms", []):
            self.physics_trace[-1]["root_linear_velocity"] = env.entities["robot"].get_vel().clone().cpu()
            self.physics_trace[-1]["root_angular_velocity"] = env.entities["robot"].get_ang().clone().cpu()

    def __call__(self, env):
        live = ~self.finished
        obj = env.entities["obj"]
        robot = env.entities["robot"]
        c = env.action_manager.get_term("dofs_pos_controller")
        omega = obj.get_ang().clone()
        axis_cmd = (
            env.command_manager.get_term("rotation_axis")
            if "rotation_axis" in env.command_manager.active_terms
            else None
        )
        axial = (omega * axis_cmd.command).sum(-1) if axis_cmd is not None else torch.zeros_like(self.steps)
        if axis_cmd is not None:
            from tasks.in_fingers_rotate.custom_terms import signed_quaternion_increment
            quat = obj.get_quat()
            increment = signed_quaternion_increment(quat, self.previous_quat, axis_cmd.command)
            valid_rotation = live & ~env.reset_terminated
            increment = torch.where(valid_rotation, increment, 0.0)
            self.turns += increment / (2 * torch.pi)
            self.backward_turns += (-increment).clamp_min(0) / (2 * torch.pi)
            self.rotation_path += increment.abs() / (2 * torch.pi)
            off_axis = omega - axial[:, None] * axis_cmd.command
            self.sustained_steps += valid_rotation & (axial >= 0.25) & (axial <= 0.75) & (off_axis.norm(dim=-1) < 0.25)
            self.previous_quat.copy_(quat)
        self.steps += live
        self.max_displacement = torch.maximum(
            self.max_displacement, (obj.get_pos() - self.start_pos).norm(dim=-1) * live
        )
        pulse = getattr(env, "object_wrench", None)
        force = torch.zeros(env.num_envs, 3, device=env.device) if pulse is None else pulse.state.force.clone()
        torque = torch.zeros_like(force) if pulse is None else pulse.state.torque.clone()
        active = torch.zeros_like(live) if pulse is None else pulse.state.active[:, 0].bool().clone()
        transition = torch.zeros_like(live)
        lift_task = "goal_pos" in env.command_manager.active_terms
        if axis_cmd is None and not lift_task:
            cmd = env.command_manager.get_term("goal_rot")
            success = cmd._last_success & live & ~env.reset_terminated
            timeout = cmd._last_timeout_reset & live
            self.successes += success
            self.goals += (success | timeout) & ~env.reset_buf
            transition = success | timeout
        if lift_task:
            cmd = env.command_manager.get_term("goal_pos")
            height = obj.get_AABB()[:, 0, 2] - self.start_bottom
            self.max_lift = torch.maximum(self.max_lift, height * live)
            self.lifted |= (height > 0.05) & live
            settled = (height > 0.05) & ((obj.get_pos() - cmd.command).norm(dim=-1) < 0.03)
            settled &= (obj.get_vel().norm(dim=-1) < 0.1) & (omega.norm(dim=-1) < 0.5)
            self.goal_dwell = torch.where(live & settled, self.goal_dwell + self.dt, 0.0)
            self.successes += (self.goal_dwell >= 0.25 - 1e-6) & (self.successes == 0) & live & ~env.reset_terminated
        onset = active & ~self.was_active & live
        self.pulses += onset
        ended = ~active & self.was_active & live
        self.recovery_time[ended] = 0
        self.recovery_dwell[ended] = 0
        self.was_active = active
        pending = (self.recovery_time >= 0) & live
        if lift_task:
            pass  # Already computed the lifted, settled carry-goal condition above.
        elif axis_cmd is None:
            settled = (cmd.stats["orientation_error"] < 0.1) & (omega.norm(dim=-1) < 0.25)
        else:
            off_axis = omega - axial[:, None] * axis_cmd.command
            settled = (axial >= 0.25) & (axial <= 0.75) & (off_axis.norm(dim=-1) < 0.25)
        self.recovery_time[pending] += self.dt
        self.recovery_dwell = torch.where(pending & settled, self.recovery_dwell + self.dt, 0.0)
        recovered = (
            pending & (self.recovery_time <= 2 + 1e-6) & (self.recovery_dwell >= 0.25 - 1e-6) & ~env.reset_terminated
        )
        self.recovered += recovered
        self.recovery_time[recovered | (self.recovery_time > 2) | env.reset_buf] = -1
        q = robot.get_dofs_pos(dofs_idx_local=c.dofs_idx_local)
        qv = robot.get_dofs_vel(dofs_idx_local=c.dofs_idx_local)
        actual_torque = robot.get_dofs_control_force(dofs_idx_local=c.dofs_idx_local)
        lo, hi = robot.get_dofs_force_range(dofs_idx_local=c.dofs_idx_local)
        self.trace.append(
            dict(
                live=live.cpu(),
                position=q.cpu(),
                velocity=qv.cpu(),
                target=c.processed_action.detach().cpu(),
                target_velocity=getattr(c, "target_velocity", torch.zeros_like(q)).cpu(),
                object_position=obj.get_pos().clone().cpu(),
                omega=omega.cpu(),
                axial=axial.cpu(),
                force=force.cpu(),
                torque=torque.cpu(),
                pulse_duration=(torch.zeros_like(self.steps) if pulse is None else pulse.state.duration).cpu(),
                pulse_remaining=(
                    torch.zeros_like(self.steps)
                    if pulse is None
                    else (pulse.state.duration - pulse.state.elapsed).clamp_min(0)
                ).cpu(),
                application_point=(
                    obj.get_pos()
                    if pulse is None
                    else env.rigid_solver.get_links_root_COM(pulse.links_idx)[:, 0] + pulse.state.lever
                ).cpu(),
                pulse=active.cpu(),
                transition=transition.cpu(),
                work=(actual_torque * qv).abs().sum(-1).cpu(),
                saturation=((actual_torque <= lo + 0.001) | (actual_torque >= hi - 0.001)).float().mean(-1).cpu(),
            )
        )
        if hasattr(c, "lower"):
            margin = torch.minimum(torch.full_like(c.lower, .1), (c.upper - c.lower) * .25)
            near_limit = (q <= c.lower + margin) | (q >= c.upper - margin)
            self.trace[-1]["joint_limit_proximity_fraction"] = near_limit.float().mean(-1).cpu()
        if lift_task:
            from genesis.utils.geom import transform_by_quat
            palm = robot.get_link(robot.options.metadata.palm_link)
            quat = palm.get_quat()
            normal = torch.as_tensor(robot.options.metadata.palm_normal, device=quat.device, dtype=quat.dtype)
            normal = (normal / normal.norm()).expand(env.num_envs, -1)
            down_dot = -transform_by_quat(normal, quat)[:, 2]
            self.trace[-1]["palm_tilt_degrees"] = torch.rad2deg(torch.acos(down_dot.clamp(-1, 1))).cpu()
        if "goal_rot" in env.command_manager.active_terms:
            self.trace[-1]["orientation_error"] = (
                env.command_manager.get_term("goal_rot").stats["orientation_error"].clone().cpu()
            )
        self.dropped |= env.reset_terminated & live
        self.finished |= env.reset_buf

    def summary(self):
        t = {k: torch.stack([row[k] for row in self.trace]) for k in self.trace[0]}
        if self.physics_trace:
            physics = {k: torch.stack([row[k] for row in self.physics_trace]) for k in self.physics_trace[0]}
            motion = motion_summary(physics, self.physics_dt)
            for prefix in ("root_linear", "root_angular"):
                if prefix + "_velocity" in physics:
                    root_motion = motion_summary(dict(physics, velocity=physics[prefix + "_velocity"]), self.physics_dt)
                    for key in ("acceleration_rms", "high_frequency_velocity_rms"):
                        motion[prefix + "_" + key] = root_motion[key]
        else:
            motion = motion_summary(t, self.dt)
        held = ~self.dropped & self.finished
        return dict(
            episodes=int(self.finished.sum()),
            retention=float(held.float().mean()),
            **motion,
            **({"joint_limit_proximity_fraction": float(t["joint_limit_proximity_fraction"][t["live"]].mean())}
               if "joint_limit_proximity_fraction" in t else {}),
            **({"mean_palm_tilt_degrees": float(t["palm_tilt_degrees"][t["live"]].mean()),
                "p95_palm_tilt_degrees": float(torch.quantile(t["palm_tilt_degrees"][t["live"]], .95))}
               if "palm_tilt_degrees" in t else {}),
            **(repose_tracking_summary(t["orientation_error"], t["omega"], t["live"])
               if "orientation_error" in t else {}),
            median_retained_axial_speed=(
                None
                if not held.any()
                else float((self.turns[held] * 2 * torch.pi / (self.steps[held] * self.dt)).median())
            ),
            **({"per_variant_rotation": [
                {
                    "variant": int(i),
                    "episodes": int((self.variant == i).sum()),
                    "median_net_turns": float(self.turns[self.variant == i].median()),
                    "median_backward_turns": float(self.backward_turns[self.variant == i].median()),
                    "median_net_to_path_ratio": float((self.turns / self.rotation_path.clamp_min(1e-8))[self.variant == i].median()),
                    "mean_sustained_rotation_fraction": float((self.sustained_steps / self.steps.clamp_min(1))[self.variant == i].mean()),
                }
                for i in self.variant.unique()
            ]} if self.is_rotation else {}),
            settled_goal_fraction=float(self.successes.sum() / self.goals.sum()),
            lifted_fraction=float(self.lifted.float().mean()),
            mean_max_lift=float(self.max_lift.mean()),
            pulses=int(self.pulses.sum()),
            max_observed_force_n=float(t["force"][t["live"]].norm(dim=-1).max()),
            max_observed_torque_nm=float(t["torque"][t["live"]].norm(dim=-1).max()),
            pulse_recovery_fraction=None if self.pulses.sum() == 0 else float(self.recovered.sum() / self.pulses.sum()),
            per_episode={
                "variant": self.variant.cpu().tolist(),
                "retained": held.cpu().tolist(),
                "steps": self.steps.cpu().tolist(),
                "turns": self.turns.cpu().tolist(),
                "backward_turns": self.backward_turns.cpu().tolist(),
                "rotation_path_turns": self.rotation_path.cpu().tolist(),
                "net_to_path_ratio": (self.turns / self.rotation_path.clamp_min(1e-8)).cpu().tolist(),
                "sustained_rotation_fraction": (self.sustained_steps / self.steps.clamp_min(1)).cpu().tolist(),
                "successes": self.successes.cpu().tolist(),
                "goals": self.goals.cpu().tolist(),
                "lifted": self.lifted.cpu().tolist(),
                "max_lift": self.max_lift.cpu().tolist(),
            },
            torque_saturation=float(t["saturation"][t["live"]].mean()),
            mechanical_work=float((t["work"] * t["live"] * self.dt).sum(0).mean()),
        )


def attach_training_evaluation(runner, config, log_dir, task, robot, record_video=True):
    """Evaluate new teacher checkpoints synchronously every 500 iterations.

    A subprocess keeps deterministic evaluation resets out of training state. This
    is opt-in by the smooth teacher controller; legacy runs keep their lifecycle.
    ``record_video`` carries ``--no_checkpoint_video``: the metrics above are the point
    of a distillation sweep, and a recorder that aborts should not take the run with it.
    """
    from functools import wraps
    import os
    import sys

    if getattr(runner, "gpu_global_rank", 0) != 0:
        return
    evaluation_env = os.environ.copy()
    for key in (
        "RANK",
        "LOCAL_RANK",
        "WORLD_SIZE",
        "LOCAL_WORLD_SIZE",
        "GROUP_RANK",
        "ROLE_RANK",
        "ROLE_WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
    ):
        evaluation_env.pop(key, None)
    snapshot_run(config, Path(log_dir) / "provenance")
    save = runner.save
    seed = config.runner_options.seed
    variants = len(config.scene_options.obj.grouped_entities or [config.scene_options.obj])
    strength = config.event_options.object_pulse.strength

    @wraps(save)
    def save_and_evaluate(path, infos=None):
        iteration = runner.current_learning_iteration
        fresh = any(part.startswith("F141-") for part in Path(log_dir).parts)
        if fresh and iteration == 0:
            return  # Fresh campaign saves only at 500-step intervals and completion.
        save(path, infos)
        if iteration <= 0 or (iteration % 500 and iteration < config.runner_options.max_iterations - 1):
            return
        for label, pulse_strength in [("clean", 0.0), ("perturbed", max(0.25, strength))]:
            output = Path(log_dir) / "evaluation" / f"{iteration:05d}-{label}"
            subprocess.run(
                [
                    sys.executable,
                    "scripts/eval_smooth_teacher.py",
                    "--task",
                    task,
                    "--robot",
                    robot,
                    "--full_config",
                    str(Path(log_dir) / "eden_config.yaml"),
                    "--checkpoint",
                    str(path),
                    "--stage",
                    "1",
                    "--eval_seed",
                    str(seed),
                    "--eval_envs",
                    str(variants * 16),
                    "--pulse_strength",
                    str(pulse_strength),
                    "--no_traces",
                    "--output",
                    str(output),
                ],
                check=True,
                env=evaluation_env,
            )
            publish_checkpoint_evaluation(output, label, iteration)
        if not record_video:
            return
        # Use the original camera recorder and logger's Media/video upload path. The
        # recorder aborts for some sensor configurations, and the metrics above are the
        # point of the run, so a failed recording is reported and stepped over rather than
        # allowed to end training that has already produced its evaluation.
        try:
            subprocess.run(
                [
                    sys.executable,
                    "scripts/record_teacher_checkpoint.py",
                    "--full_config",
                    str(Path(log_dir) / "eden_config.yaml"),
                    "--checkpoint",
                    str(path),
                    "--output",
                    str(Path(path).with_suffix(".mp4")),
                    "--num_envs",
                    str(variants),
                ],
                check=True,
                env=evaluation_env,
            )
        except subprocess.CalledProcessError as exc:
            print(f"Checkpoint video failed at iteration {iteration} ({exc}); continuing training.")
            return
        writer = getattr(getattr(runner, "logger", None), "writer", None)
        if writer is not None and hasattr(writer, "save_video"):
            writer.save_video(Path(path).with_suffix(".mp4"), iteration)
        if any(part.startswith("F141-") for part in Path(log_dir).parts):
            subprocess.run([sys.executable, "scripts/update_fresh_teacher_report.py"], check=True, env=evaluation_env)

    runner.save = save_and_evaluate


def publish_checkpoint_evaluation(output, label, iteration):
    """Publish evaluation scalars; checkpoint videos use the original Media logger."""
    import wandb

    if wandb.run is None:
        return
    metrics = json.loads((Path(output) / "metrics.json").read_text())
    payload = {
        f"Evaluation/{label}/{key}": value
        for key, value in metrics.items()
        if isinstance(value, (int, float)) and key not in {"seed", "fixed_target"}
    }
    payload["Evaluation/checkpoint_iteration"] = iteration
    for key in ("pulse_profile", "reset_rng_protocol"):
        if key in metrics:
            payload[f"Evaluation/{label}/{key}"] = metrics[key]
    wandb.log(payload, step=iteration, commit=False)
