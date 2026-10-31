"""Checkpoint compatibility and literal binary observation for smooth teachers."""

from functools import wraps
import torch
from rsl_rl.modules.normalization import EmpiricalNormalization
from smooth_control import OBSERVATION_VERSION


def materialize_inference_buffers(module):
    """Make normalization buffers writable before loading a checkpoint.

    rsl_rl updates RND normalization statistics inside ``inference_mode`` and
    can therefore replace ordinary buffers with inference tensors. PyTorch
    refuses to copy checkpoint values into those tensors outside inference
    mode. Replacing only those registered buffers preserves their values and
    lets the normal state-dict loader restore RND exactly.
    """
    if module is None:
        return
    with torch.inference_mode(False):
        for child in module.modules():
            for name, value in tuple(child._buffers.items()):
                if value is not None and value.is_inference():
                    child._buffers[name] = value.clone()


class BinaryTailNormalization(EmpiricalNormalization):
    """Normalize continuous features while preserving the final binary feature."""

    def forward(self, x):
        y = super().forward(x)
        return torch.cat((y[..., :-1], x[..., -1:]), dim=-1)

    @torch.jit.unused
    def update(self, x):
        # PPO updates statistics inside inference_mode. Keep saved buffers ordinary
        # tensors so a checkpoint can also be reloaded outside inference_mode.
        with torch.inference_mode(False):
            super().update(x)

    def inverse(self, y):
        x = super().inverse(y)
        return torch.cat((x[..., :-1], y[..., -1:]), dim=-1)


def contract_mismatch(expected, actual, teacher_load):
    """Controller semantics must always match; the observation layout usually must too.

    Distilling from a stage-1 teacher is the one load where the two configs are meant to
    disagree, since the student sees tactile observations the teacher never had. The
    teacher-side layout is still checked there: ``validate_distillation_weights`` compares
    keys and shapes against the instantiated models, and preflight compares group names.
    """
    if actual is None:
        return True
    if teacher_load:
        return {key: value for key, value in expected.items() if key != "obs_groups"} != {
            key: value for key, value in actual.items() if key != "obs_groups"
        }
    return expected != actual


def warm_start_contract_error(loaded, current):
    """Why ``loaded``'s policy cannot warm-start a runner under ``current``, or None.

    Stage 3 puts the student's observations on the actor, so the two contracts differ in
    ``obs_groups['actor']`` by construction and plain equality always rejects. What has to
    hold is that the weights see the observations they were trained on -- the checkpoint's
    student group -- and that every other observation group and controller term is unchanged.
    """
    if current is None:
        return None
    if not loaded:
        return "checkpoint carries no observation/controller contract"
    expected = current.get("obs_groups", {}).get("actor")
    trained_on = loaded.get("obs_groups", {}).get("student")
    if expected != trained_on:
        return f"actor observes {expected}, checkpoint's student was trained on {trained_on}"
    for label, key in (("controller terms", None), ("observation groups", "obs_groups")):
        a = loaded.get(key, {}) if key else {k: v for k, v in loaded.items() if k != "obs_groups"}
        b = current.get(key, {}) if key else {k: v for k, v in current.items() if k != "obs_groups"}
        if key:
            a = {k: v for k, v in a.items() if k != "actor"}
            b = {k: v for k, v in b.items() if k != "actor"}
        differing = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
        if differing:
            return f"{label} differ: " + ", ".join(differing)
    return None


def install_smooth_runner_contract(runner, env):
    if "dofs_pos_controller" not in env.unwrapped.action_manager.active_terms:
        return
    c = env.unwrapped.action_manager.get_term("dofs_pos_controller")
    if getattr(c, "observation_version", None) != OBSERVATION_VERSION:
        return
    contract = dict(
        observation_version=OBSERVATION_VERSION,
        control_dt=env.unwrapped.dt,
        dofs=list(c.dofs_name),
        reference_source=str(c.reference_source),
        ema_alpha=c.ema_alpha,
        max_target_velocity=c.max_target_velocity,
        max_target_acceleration=c.max_target_acceleration,
        limit_targets=c.limit_targets,
        scale_ratio=c.scale_ratio,
        scale=c.scale,
        obs_groups=env.runner_dict["obs_groups"],
    )
    if getattr(c, "accumulator_hard_limits", False):
        contract["accumulator_limit_source"] = "hard_joint_limits"
    if getattr(c, "reset_target_source", "joint_position") != "joint_position":
        contract["reset_target_source"] = c.reset_target_source
    if "ee_controller" in env.unwrapped.action_manager.active_terms:
        root = env.unwrapped.action_manager.get_term("ee_controller")
        contract["root_controller"] = {
            "class": type(root).__name__,
            **{
                key: getattr(root, key)
                for key in (
                    "linear_velocity_limit",
                    "linear_acceleration_limit",
                    "angular_velocity_limit",
                    "angular_acceleration_limit",
                    "pos_x_range",
                    "pos_y_range",
                    "pos_z_range",
                    "euler_x_range",
                    "euler_y_range",
                    "euler_z_range",
                )
            },
        }
    # PPO carries actor/critic; distillation carries student/teacher. Each algorithm
    # names its models after their observation groups, and only a group that feeds the
    # binary perturbation feature needs the tail preserved -- a proprioception-and-tactile
    # student has no such feature, while a distilled teacher inherits one from training.
    for name in ("actor", "critic", "student", "teacher"):
        group = env.runner_dict["obs_groups"].get(name)
        model = getattr(runner.alg, name, None)
        if group is None or model is None or "perturbation" not in group:
            continue
        if group[-1] != "perturbation":
            raise ValueError("Smooth teacher binary perturbation group must be last")
        old = model.obs_normalizer
        if isinstance(old, EmpiricalNormalization):
            normalizer = BinaryTailNormalization(old.mean.shape, eps=old.eps, until=old.until).to(old.mean.device)
            normalizer.load_state_dict(old.state_dict())
            model.obs_normalizer = normalizer
    runner.smooth_training_env_steps = 0
    update = getattr(runner.alg, "update", None)
    if update is not None:

        @wraps(update)
        def counted_update(*args, **kwargs):
            if runner.smooth_training_env_steps is None:
                raise ValueError("Checkpoint lacks training-step provenance; start a fresh budgeted experiment")
            result = update(*args, **kwargs)
            runner.smooth_training_env_steps += (
                env.num_envs * env.runner_dict["num_steps_per_env"] * getattr(runner, "gpu_world_size", 1)
            )
            return result

        runner.alg.update = counted_update
    save, load = runner.save, runner.load

    @wraps(save)
    def save_versioned(path, infos=None):
        save(
            path,
            dict(infos or {}, smooth_teacher_contract=contract, training_env_steps=runner.smooth_training_env_steps),
        )

    @wraps(load)
    def load_versioned(path, *args, **kwargs):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        actual = (checkpoint.get("infos") or {}).get("smooth_teacher_contract")
        teacher_load = "actor_state_dict" in checkpoint and hasattr(runner.alg, "teacher")
        if contract_mismatch(contract, actual, teacher_load):
            raise ValueError("Checkpoint observation/controller contract mismatch; train a new smooth teacher")
        materialize_inference_buffers(getattr(runner.alg, "rnd", None))
        result = load(path, *args, **kwargs)
        runner.smooth_training_env_steps = (checkpoint.get("infos") or {}).get("training_env_steps")
        return result

    runner.save, runner.load = save_versioned, load_versioned
    runner.smooth_teacher_contract = contract
