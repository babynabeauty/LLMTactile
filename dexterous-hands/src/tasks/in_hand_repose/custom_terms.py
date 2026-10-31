from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from eden.managers import COMMAND_TERM_REGISTRY, EVENT_TERM_REGISTRY, REWARD_TERM_REGISTRY, RewardTerm
from eden.utils.geom import quat_error_magnitude

from shared_terms import SetSampledBottomAlignedPos, TargetRotationCommand

if TYPE_CHECKING:
    from eden.envs.base import EnvBase


@REWARD_TERM_REGISTRY.register()
def default_pose_l1_penalty(env: "EnvBase", *, entity_name: str = "robot") -> torch.Tensor:
    """L1 joint deviation from the configured reset/controller default pose."""
    robot = env.entities[entity_name]
    return (robot.get_dofs_pos(robot.dofs_idx_local) - robot.default_dofs_pos).abs().sum(dim=-1)


@REWARD_TERM_REGISTRY.register()
def action_l2_penalty(env: "EnvBase", *, action_name: str = "dofs_pos_controller") -> torch.Tensor:
    """Sum of squared policy actions, matching IsaacLab's direct hand task."""
    action = env.action_manager.get_term(action_name).action
    return action.square().sum(dim=-1)


@REWARD_TERM_REGISTRY.register()
def settled_goal_completion(env: "EnvBase", *, command_name: str = "goal_rot") -> torch.Tensor:
    """Unit impulse per settled completion, independent of reward-manager dt scaling.

    The command owns the dwell requirement and once-per-goal latch. This term
    only converts its event to an impulse so the configured weight is the actual
    reward per completed goal, rather than weight times the control timestep.
    """
    success = env.command_manager.get_term(command_name)._last_success
    # Terminations are computed after command refresh and before this reward.
    # A coincident settled orientation and drop must never earn completion credit.
    dropped = getattr(env, "reset_terminated", torch.zeros_like(success))
    return (success & ~dropped).float() / env.dt


@REWARD_TERM_REGISTRY.register()
class OrientationProgressReward(RewardTerm):
    """Reward per-second reduction in orientation error to the active goal."""

    entity_name: str | None = None
    command_name: str = "goal_rot"
    clip: tuple[float, float] = (-10.0, 10.0)
    negative_scale: float = 1.0

    def build(self) -> None:
        super().build()
        if self.negative_scale < 1.0:
            raise ValueError("negative_scale must be >= 1 to prevent profitable out-and-back motion")
        self._cmd = self._env.command_manager.get_term(self.command_name)
        resolved_entity_name = (
            self.entity_name if self.entity_name is not None else getattr(self._cmd, "entity_name", "obj")
        )
        self._obj = self._env.entities[resolved_entity_name]

        self._prev_goal_quat = self._cmd.quat.detach().clone()
        self._prev_error = self._current_error().detach().clone()
        self._progress_debt = torch.zeros_like(self._prev_error)

    def _current_error(self) -> torch.Tensor:
        return quat_error_magnitude(self._obj.get_quat(), self._cmd.quat)

    def reset(self, envs_idx: slice | torch.Tensor | None = None) -> None:
        if envs_idx is None:
            envs_idx = slice(None)
        self._prev_goal_quat[envs_idx] = self._cmd.quat[envs_idx].detach().clone()
        self._prev_error[envs_idx] = self._current_error()[envs_idx].detach().clone()
        self._progress_debt[envs_idx] = 0

    def compute(self, envs_idx: slice | torch.Tensor | None = None) -> torch.Tensor:
        current_goal = self._cmd.quat
        current_error = self._current_error()
        goal_changed = torch.any(torch.abs(current_goal - self._prev_goal_quat) > 1e-6, dim=-1)

        dt = self._env.env_options.sim_dt * self._env.env_options.decimation
        reward = (self._prev_error - current_error) / dt
        reward[reward < 0.0] *= self.negative_scale
        # Symmetric output clipping must not erase a fast wrong-way excursion
        # that a slow return could profit from. Carry only unpaid negative progress;
        # clipped positive progress is discarded, never banked for later reward.
        raw = reward + self._progress_debt / dt
        reward = torch.clamp(raw, min=self.clip[0], max=self.clip[1])
        self._progress_debt.copy_(torch.minimum((raw - reward) * dt, torch.zeros_like(raw)))
        self._progress_debt[goal_changed] = 0
        reward = torch.where(goal_changed, torch.zeros_like(reward), reward)

        self._prev_goal_quat.copy_(current_goal.detach())
        self._prev_error.copy_(current_error.detach())
        return reward


@COMMAND_TERM_REGISTRY.register()
class TimeoutTrackingTargetRotationCommand(TargetRotationCommand):
    """``TargetRotationCommand`` that flags envs whose goal was just reset by *timeout*.

    Captures the timeout mask before ``super().compute(dt)`` runs the resample
    (which refills ``time_left``), so ``_last_timeout_reset`` is True for one
    step on the env(s) whose resampling timer expired. Success-driven goal
    resets fire from ``_update_command`` after the timeout resample and do not
    set this flag.
    """

    def build(self) -> None:
        super().build()
        self._last_timeout_reset = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def compute(self, dt: float) -> None:
        self._last_timeout_reset = (self.time_left - dt) <= 0.0
        super().compute(dt)


@REWARD_TERM_REGISTRY.register()
def target_timeout_reset_penalty(env: "EnvBase", *, command_name: str = "goal_rot") -> torch.Tensor:
    """1.0 on the step the orientation goal was reset by the resampling timeout."""
    cmd = env.command_manager.get_term(command_name)
    return cmd._last_timeout_reset.to(torch.float32)


@COMMAND_TERM_REGISTRY.register()
class SettledTargetRotationCommand(TimeoutTrackingTargetRotationCommand):
    """Grade the current goal after physics; resample only after rewards consume it."""

    settle_time: float = 0.25
    max_success_angular_speed: float = 0.25

    def build(self):
        super().build()
        self._dwell = torch.zeros(self.num_envs, device=self.device)
        self._awarded = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._last_timeout_reset = torch.zeros_like(self._awarded)

    def _resample_command(self, envs_idx):
        super()._resample_command(envs_idx)
        if hasattr(self, "_dwell"):
            self._dwell[envs_idx] = 0
            self._awarded[envs_idx] = False

    def reset(self, envs_idx=None):
        ids = slice(None) if envs_idx is None else envs_idx
        from eden.managers import CommandTerm

        extras = CommandTerm.reset(self, ids)
        self.stats["orientation_error"][ids] = quat_error_magnitude(self.object.get_quat(), self.quat)[ids]
        self._last_success[ids] = False
        self._last_timeout_reset[ids] = False
        return extras

    def post_physics_refresh(self):
        error = quat_error_magnitude(self.object.get_quat(), self.quat)
        self.stats["orientation_error"].copy_(error)
        settled = (error < self.orientation_success_threshold) & (
            self.object.get_ang().norm(dim=-1) < self.max_success_angular_speed
        )
        self._dwell.copy_(torch.where(settled, self._dwell + self._env.dt, 0.0))
        self._last_success = settled & (self._dwell >= self.settle_time - 1e-6) & ~self._awarded
        self._awarded |= self._last_success
        self.stats["consecutive_success"] += self._last_success.float()
        self._last_timeout_reset = (self.time_left <= self._env.dt) & ~self._last_success

    def compute(self, dt):
        # Reset environments have already received a fresh goal this step.
        live = self._env.episode_length_buf > 0
        self.time_left -= dt * live
        change = live & ((self.time_left <= 0) | (self._last_success & self.update_goal_on_success))
        self._resample(change)

    def _update_command(self):
        pass


@COMMAND_TERM_REGISTRY.register()
class ReachableSettledTargetRotationCommand(SettledTargetRotationCommand):
    """Bounded rotations from the current pose; equal bounds give exact-angle goals.

    Legacy relative targets walk from the previous goal, including across resets.
    This command anchors each new challenge to the actual object instead.
    """

    min_angle: float = 0.25
    max_angle: float = 0.60

    def _resample_command(self, envs_idx):
        import genesis.utils.geom as gu
        from eden.utils.geom import quat_to_rot6d
        from eden.utils.misc import sanitize_envs_idx

        ids, n = sanitize_envs_idx(envs_idx, self.num_envs, return_n_envs=True)
        if not n:
            return
        if not 0.1 < self.min_angle <= self.max_angle <= 3.141593:
            raise ValueError("Goal angles must exceed the success tolerance and be <= pi")
        axis = torch.randn(n, 3, device=self.device)
        axis[:, 2] = 0  # Random world-XY axes, including for fixed-angle goals.
        axis /= axis.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        angle = torch.empty(n, device=self.device).uniform_(self.min_angle, self.max_angle)
        self.quat[ids] = gu.transform_quat_by_quat(self.object.get_quat()[ids], gu.axis_angle_to_quat(angle, axis))
        self.rotation[ids] = quat_to_rot6d(self.quat[ids])
        self._sync_goal_vis_orientation(ids)
        if hasattr(self, "_dwell"):
            self._dwell[ids] = 0
            self._awarded[ids] = False


@EVENT_TERM_REGISTRY.register()
class VariantCenteredReposeReset(SetSampledBottomAlignedPos):
    """Fixed object-specific cradle centers with the existing random XY jitter."""

    variant_centers: list[tuple[float, float]] = []

    def build(self):
        super().build()
        memberships = []
        for link in self._entity.links:
            for geom in link.geoms:
                ids = geom.active_envs_idx
                if ids is not None and len(ids):
                    key = tuple(torch.as_tensor(ids).cpu().tolist())
                    if key not in memberships:
                        memberships.append(key)
        if len(memberships) != len(self.variant_centers):
            raise ValueError("Reset centers must match resolved primitive variant groups")
        self._centers = torch.zeros(self.num_envs, 2, device=self.device)
        for ids, center in zip(memberships, self.variant_centers, strict=True):
            self._centers[list(ids)] = torch.tensor(center, device=self.device)

    def compute(self, envs_idx=None):
        from eden.utils.misc import sanitize_envs_idx
        ids, n = sanitize_envs_idx(envs_idx, self.num_envs, return_n_envs=True)
        if not n:
            return
        super().compute(ids)
        pos = self._entity.get_pos()[ids].clone()
        pos[:, :2] += self._centers[ids]
        self._entity.set_pos(pos, envs_idx=ids)


def joint_limit_margin_cost(position, lower, upper, margin: float = 0.10):
    """Smooth quadratic cost inside mechanical-limit margins; one at a bound."""
    if margin <= 0:
        raise ValueError("Joint-limit margin must be positive")
    width = upper - lower
    active = torch.isfinite(width) & (width > 0)
    safe_margin = torch.minimum(torch.full_like(width, margin), width * .25)
    safe_margin = torch.where(active, safe_margin, torch.ones_like(width))
    penetration = ((lower + safe_margin - position).clamp_min(0)
                   + (position - upper + safe_margin).clamp_min(0)) / safe_margin
    return torch.where(active, penetration.square(), torch.zeros_like(penetration)).sum(dim=-1)


@REWARD_TERM_REGISTRY.register()
def repose_joint_limit_margin_penalty(env: "EnvBase", *, entity_name: str = "robot", margin: float = .10):
    """Discourage resting at actual DOF limits without clipping policy actions."""
    robot = env.entities[entity_name]
    lower, upper = robot.get_dofs_limit(dofs_idx_local=robot.dofs_idx_local)
    position = robot.get_dofs_pos(dofs_idx_local=robot.dofs_idx_local)
    return joint_limit_margin_cost(position, lower, upper, margin)
