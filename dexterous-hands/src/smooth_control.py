"""Bounded target control. Versioned separately from legacy teacher interfaces."""

import torch
from eden.managers import (
    ACTION_TERM_REGISTRY,
    OBSERVATION_TERM_REGISTRY,
    REWARD_TERM_REGISTRY,
    METRIC_TERM_REGISTRY,
    MetricTerm,
    RewardTerm,
)
from eden.managers.terms.actions.joint_actions import ExplicitPDController

OBSERVATION_VERSION = "smooth_teacher_v1"


def limited_target(desired, position, velocity, lower, upper, dt, max_velocity=0.8, max_acceleration=8.0):
    """Discrete velocity/acceleration limits, with braking before joint boundaries.

    The safe speed solves the exact discrete stopping-distance inequality, including
    this step. It permits an abrupt reversal request while keeping the target viable.
    """
    dv = max_acceleration * dt

    def safe_speed(distance):
        d = distance.clamp_min(0) / dt
        n = torch.floor((torch.sqrt(1 + 8 * d / dv) - 1) / 2)
        return (d + dv * n * (n + 1) / 2) / (n + 1)

    requested = (desired.clamp(lower, upper) - position) / dt
    lo = torch.maximum(torch.full_like(velocity, -max_velocity), -safe_speed(position - lower))
    hi = torch.minimum(torch.full_like(velocity, max_velocity), safe_speed(upper - position))
    new_velocity = requested.clamp(velocity - dv, velocity + dv).clamp(lo, hi)
    target = (position + dt * new_velocity).clamp(lower, upper)
    return target, (target - position) / dt


@ACTION_TERM_REGISTRY.register()
class SmoothPDController(ExplicitPDController):
    max_target_velocity: float = 0.8
    max_target_acceleration: float = 8.0
    ema_alpha: float = 0.0
    observation_version: str = OBSERVATION_VERSION
    limit_targets: bool = True
    accumulator_hard_limits: bool = False
    reset_target_source: str = "joint_position"

    def build(self):
        super().build()
        if self.observation_version != OBSERVATION_VERSION:
            raise ValueError("Unsupported smooth controller observation version")
        if self.reset_target_source not in {"joint_position", "cached_grasp_target"}:
            raise ValueError("Unknown reset target source")
        if not 0 <= self.ema_alpha < 1 or min(self.max_target_velocity, self.max_target_acceleration) <= 0:
            raise ValueError("Invalid smoothing limits")
        self.lower, self.upper = self.entity.get_dofs_limit(dofs_idx_local=self.dofs_idx_local)
        self.target = self.entity.get_dofs_pos(dofs_idx_local=self.dofs_idx_local).clone()
        self.lower = self.lower.expand_as(self.target)
        self.upper = self.upper.expand_as(self.target)
        if self.accumulator_hard_limits and self._accumulated_target is not None:
            # Cached grasps may lie outside Eden's 90% soft range. Zero actions
            # must preserve them; use the final limiter's actual joint bounds.
            self._target_lower = self.lower
            self._target_upper = self.upper
        self.target_velocity = torch.zeros_like(self.target)
        self.filtered_target = self.target.clone()
        self.action_saturation = torch.zeros(self.num_envs, 1, device=self.device)

    def reset(self, envs_idx=None):
        ids = slice(None) if envs_idx is None else envs_idx
        super().reset(ids)
        if self.reset_target_source == "cached_grasp_target":
            if not hasattr(self, "_grasp_reset_target"):
                raise RuntimeError("Cached-target reset requires a loaded grasp")
            reset_target = self._grasp_reset_target
        else:
            reset_target = self.entity.get_dofs_pos(dofs_idx_local=self.dofs_idx_local)
        target = reset_target[ids].clamp(self.lower[ids], self.upper[ids])
        self.target[ids] = target
        self.filtered_target[ids] = target
        self.target_velocity[ids] = 0
        self.action_saturation[ids] = 0
        self._processed_action[ids] = target
        if self._accumulated_target is not None:
            self._accumulated_target[ids] = target

    def compute(self, actions):
        self.action_saturation.copy_((actions.abs() >= 1).float().mean(-1, keepdim=True))
        super().compute(actions.clamp(-1, 1))
        self.filtered_target.mul_(self.ema_alpha).add_(self._processed_action, alpha=1 - self.ema_alpha)
        if self.limit_targets:
            target, velocity = limited_target(
                self.filtered_target,
                self.target,
                self.target_velocity,
                self.lower,
                self.upper,
                self._env.dt,
                self.max_target_velocity,
                self.max_target_acceleration,
            )
        else:
            target = self.filtered_target.clamp(self.lower, self.upper)
            velocity = (target - self.target) / self._env.dt
        self.target.copy_(target)
        self.target_velocity.copy_(velocity)
        self._processed_action = self.target.clone()
        if self._accumulated_target is not None:
            self._accumulated_target.copy_(self.target)


@OBSERVATION_TERM_REGISTRY.register()
def controller_state(env):
    c = env.action_manager.get_term("dofs_pos_controller")
    # EMA state is included because it too affects the next applied target.
    return torch.cat((c.target, c.target_velocity, c.filtered_target), dim=-1)


@REWARD_TERM_REGISTRY.register()
def excess_axial_speed(env, command_name="rotation_axis", entity_name="obj"):
    axis = env.command_manager.get_term(command_name).command
    speed = (env.entities[entity_name].get_ang() * axis).sum(-1)
    return (speed.abs() - 0.5).clamp_min(0).square()


@REWARD_TERM_REGISTRY.register()
def off_axis_speed(env, command_name="rotation_axis", entity_name="obj"):
    axis = env.command_manager.get_term(command_name).command
    omega = env.entities[entity_name].get_ang()
    return (omega - (omega * axis).sum(-1, keepdim=True) * axis).square().sum(-1)


@REWARD_TERM_REGISTRY.register()
def steady_rotation_bonus(env, command_name="rotation_axis", entity_name="obj", speed_range=(0.25, 0.75), max_off_axis=0.25):
    """1 while the object turns inside the acceptance band without wobble, else 0."""
    axis = env.command_manager.get_term(command_name).command
    omega = env.entities[entity_name].get_ang()
    axial = (omega * axis).sum(-1)
    off_axis = (omega - axial[:, None] * axis).norm(dim=-1)
    steady = (axial >= speed_range[0]) & (axial <= speed_range[1]) & (off_axis < max_off_axis)
    return steady.float()


def prepare_smooth_config(config):
    """Remove XHand-derived disturbances only for the explicit new interface."""
    c = getattr(config.action_options, "dofs_pos_controller", None)
    if c is None or c.name != "smooth_pd_controller":
        return
    for name in ("ext_force", "kick", "torque_noise", "deadband", "backlash", "motor_strength", "dofs_randomize_bias"):
        if name in config.event_options.term_keys():
            config.event_options.disable_term(name)
    # Config deep merging can retain the legacy modifier chain; the new controller
    # owns its EMA and target limiter, so this must be removed explicitly.
    c.modifier = None
    if config.event_options.obj_mass.name == "randomize_mass_shift_with_obs":
        config.event_options.obj_mass.name = "randomize_object_mass"


@METRIC_TERM_REGISTRY.register()
class SmoothMotionMetric(MetricTerm):
    quantity: str = "joint_velocity"

    def build(self):
        self.previous_velocity = None
        self.valid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def reset(self, envs_idx=None):
        ids = slice(None) if envs_idx is None else envs_idx
        self.valid[ids] = False

    def compute(self, envs_idx=None):
        c = self._env.action_manager.get_term("dofs_pos_controller")
        velocity = c.entity.get_dofs_vel(dofs_idx_local=c.dofs_idx_local)
        if self.previous_velocity is None:
            self.previous_velocity = velocity.clone()
        if self.quantity == "joint_acceleration":
            value = ((velocity - self.previous_velocity) / self._env.dt).square().mean(-1).sqrt() * self.valid
        elif self.quantity == "joint_velocity":
            value = velocity.square().mean(-1).sqrt()
        elif self.quantity == "target_velocity":
            value = c.target_velocity.square().mean(-1).sqrt()
        elif self.quantity == "action_saturation":
            value = c.action_saturation[:, 0]
        else:
            raise ValueError(f"Unknown motion metric {self.quantity}")
        self.previous_velocity.copy_(velocity)
        self.valid[:] = True
        return value


@REWARD_TERM_REGISTRY.register()
class JointAccelerationPenalty(RewardTerm):
    """Mean squared actual joint acceleration, in units of ``scale`` rad/s^2."""

    scale: float = 100.0

    def build(self):
        self.previous_velocity = None
        self.valid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def reset(self, envs_idx=None):
        ids = slice(None) if envs_idx is None else envs_idx
        self.valid[ids] = False

    def compute(self, envs_idx=None):
        c = self._env.action_manager.get_term("dofs_pos_controller")
        velocity = c.entity.get_dofs_vel(dofs_idx_local=c.dofs_idx_local)
        if self.previous_velocity is None:
            self.previous_velocity = velocity.clone()
        acceleration = (velocity - self.previous_velocity) / (self._env.dt * self.scale)
        value = acceleration.square().mean(-1) * self.valid
        self.previous_velocity.copy_(velocity)
        self.valid[:] = True
        return value


@REWARD_TERM_REGISTRY.register()
def current_rotation_reward(
    env, entity_name="obj", command_name="rotation_axis", angvel_clip_min=-0.5, angvel_clip_max=0.5
):
    axis = env.command_manager.get_command(command_name)
    return (env.entities[entity_name].get_ang() * axis).sum(-1).clamp(angvel_clip_min, angvel_clip_max)
