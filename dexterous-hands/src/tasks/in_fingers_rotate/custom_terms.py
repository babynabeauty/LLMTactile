"""Custom terms for the in-fingers rotation task.

Holds the two HORA pieces with no shared equivalent: a pose-diff penalty anchored
on the grasp sampled for the current episode, and an accumulated-turns metric that
counts rotation only while the object is still held.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from eden.constants import MetricDirection, MetricMode
from eden.managers import (
    METRIC_TERM_REGISTRY,
    REWARD_TERM_REGISTRY,
    TERMINATION_TERM_REGISTRY,
    MetricTerm,
    RewardTerm,
    TerminationTerm,
)
from eden.utils.geom import axis_angle_from_quat, inv_quat, quat_mul
from eden.utils.misc import sanitize_envs_idx

if TYPE_CHECKING:
    from eden.envs.base import EnvBase
    from eden.options import MetricTermOptions

TWO_PI = 2.0 * math.pi


# ================== REWARDS ==================


@REWARD_TERM_REGISTRY.register()
class GraspPoseDiffPenalty(RewardTerm):
    """Squared joint deviation from the grasp sampled for the current episode.

    HORA's ``pose_diff_penalty`` measures against ``init_pose_buf``, which is
    re-seeded per env on every reset from the grasp cache. The shared
    :func:`shared_terms.pose_diff_penalty` snapshots one pose on its first call
    and holds it for the whole run, which is a different quantity once grasps are
    sampled per episode -- read ``LoadGraspPose``'s cache instead.

    This is the term that produces a repeating gait: the fingers pay for every
    radian they hold away from the grasp, so a stroke that returns is cheaper
    than one that drifts.
    """

    entity_name: str = "robot"

    def build(self) -> None:
        super().build()
        self._robot = self._env.entities[self.entity_name]

    def compute(self, envs_idx: slice | torch.Tensor | None = None) -> torch.Tensor:
        anchor = getattr(self._env, "_sampled_grasp_joint_pos", None)
        dofs_pos = self._robot.get_dofs_pos(self._robot.dofs_idx_local)
        if anchor is None:
            return torch.zeros(self._env.num_envs, device=self._env.device)
        return ((dofs_pos - anchor) ** 2).sum(dim=-1)


@REWARD_TERM_REGISTRY.register()
def grasp_pose_proximity_reward(env, entity_name="robot", tolerance=0.5):
    """Bounded closeness to the episode's sampled grasp pose.

    exp(-mean squared joint deviation / tolerance^2): 1 at the grasp pose, e^-1 at an
    RMS deviation of ``tolerance`` rad. Bounded so it shapes the gait toward a compact,
    natural hand without being able to outweigh the rotation credit the way an
    unbounded pose penalty does.
    """
    anchor = getattr(env, "_sampled_grasp_joint_pos", None)
    if anchor is None:
        return torch.zeros(env.num_envs, device=env.device)
    q = env.entities[entity_name].get_dofs_pos()
    return torch.exp(-((q - anchor) ** 2).mean(-1) / tolerance**2)


# ================== TERMINATIONS ==================


@TERMINATION_TERM_REGISTRY.register()
class DroppedBelowGraspStart(TerminationTerm):
    """Terminate when the object falls ``margin`` below where its grasp placed it.

    HORA can use an absolute height (0.645 against a 0.65 start) because every one of
    its grasps holds the same block at the same place. Here the object set spans
    5-8 cm, so the sampled grasps start anywhere in a ~2 cm band and a fixed threshold
    gives the tallest objects almost no margin while the shortest get double.
    Measuring against ``LoadGraspPose``'s own start height gives every episode the
    same margin.
    """

    entity_name: str = "obj"
    margin: float = 0.02
    fallback_height: float = 0.5

    def build(self) -> None:
        super().build()
        self._obj = self._env.entities[self.entity_name]

    def compute(self, envs_idx: slice | torch.Tensor | None = None) -> torch.Tensor:
        start = getattr(self._env, "_sampled_grasp_obj_pos", None)
        if start is None:
            start_z = torch.full((self._env.num_envs,), self.fallback_height, device=self._env.device)
        else:
            start_z = start[:, 2]
        return self._obj.get_pos()[:, 2] < (start_z - self.margin)


# ================== METRICS ==================


@METRIC_TERM_REGISTRY.register(
    direction=MetricDirection.HIB, metric_mode=MetricMode.RESET, contributes_to_success=False
)
class AxisTurnsMetric(MetricTerm):
    """Net turns completed about the command axis this episode.

    Integrates the signed per-step rotation about the axis and reports the running
    total, so the value logged at reset is the episode's turn count. Increments are
    gated on the object still being within ``hold_radius`` of its start position:
    an object that has left the hand can keep spinning for a while before the drop
    termination fires, and that spin is not the behaviour being measured.

    ``per_step_clip`` drops physically impossible single-step jumps, which is what
    a finite-difference angular velocity reports when the object teleports on
    reset or a contact resolves explosively.
    """

    entity_name: str = "obj"
    command_name: str = "rotation_axis"
    hold_radius: float = 0.06
    per_step_clip: float = 0.5

    def __init__(self, env: EnvBase, options: MetricTermOptions):
        super().__init__(env, options)
        self._turns = torch.zeros(self.num_envs, device=self.device)
        self._prev_quat: torch.Tensor | None = None
        self._start_pos: torch.Tensor | None = None

    def build(self) -> None:
        super().build()
        self._obj = self._env.entities[self.entity_name]

    def compute(self, envs_idx: slice | torch.Tensor | None = None) -> torch.Tensor:
        quat = self._obj.get_quat()
        pos = self._obj.get_pos()
        if self._prev_quat is None:
            self._prev_quat = quat.clone()
            self._start_pos = pos.clone()
            return self._turns

        axis_angle = axis_angle_from_quat(quat_mul(quat, inv_quat(self._prev_quat)))
        self._prev_quat[:] = quat

        axis = self._env.command_manager.get_term(self.command_name).command
        delta = (axis_angle * axis).sum(dim=-1)
        held = torch.norm(pos - self._start_pos, dim=-1) < self.hold_radius
        delta = torch.where(delta.abs() < self.per_step_clip, delta, torch.zeros_like(delta))
        self._turns += torch.where(held, delta, torch.zeros_like(delta)) / TWO_PI
        return self._turns

    def reset(self, envs_idx: torch.Tensor | None = None) -> None:
        envs_idx = sanitize_envs_idx(envs_idx, self.num_envs) if envs_idx is not None else slice(None)
        self._turns[envs_idx] = 0.0
        if self._prev_quat is not None:
            self._prev_quat[envs_idx] = self._obj.get_quat()[envs_idx]
            self._start_pos[envs_idx] = self._obj.get_pos()[envs_idx]


def retention_rotation_score(axial_speed, regularization_cost, dropped, rotation_gain=1.0):
    """Bound running costs; preserve symmetric signed rotation shaping.

    S2's default is nonnegative on every retained step. Higher gains keep a
    positive stationary-hold reward but penalize sustained reverse rotation.
    """
    cost = regularization_cost.clamp_min(0)
    bounded_cost = 0.25 * cost / (0.25 + cost)
    score = 0.75 + rotation_gain * axial_speed.clamp(-0.5, 0.5) - bounded_cost
    return torch.where(dropped, torch.zeros_like(score), score)


@REWARD_TERM_REGISTRY.register()
def retained_rotation_reward(env, rotation_gain=1.0, progress_speed=None):
    """Preserve directional/speed shaping with a bounded effort cost.

    Keep the separate terminal drop penalty. At the default gain, the 0.75/s
    retention credit covers the directional and regularization costs. S3 raises
    only the signed rotation gain, retaining positive credit for stationary holds.
    """
    from shared_terms import work_penalty, torque_penalty, object_linvel_penalty, dofs_pos_limits_penalty

    obj = env.entities["obj"]
    axis = env.command_manager.get_command("rotation_axis")
    omega = obj.get_ang()
    axial = (omega * axis).sum(-1)
    off_axis = (omega - axial[:, None] * axis).square().sum(-1)
    q = env.entities["robot"].get_dofs_pos()
    pose = (q - env._sampled_grasp_joint_pos).square().sum(-1)
    cost = (
        0.3 * object_linvel_penalty(env, entity_name="obj")
        + 0.3 * pose
        + 0.1 * torque_penalty(env, entity_name="robot")
        + 2 * work_penalty(env, entity_name="robot")
        + dofs_pos_limits_penalty(env, entity_name="robot")
        + (axial.abs() - 0.5).clamp_min(0).square()
        + 0.1 * off_axis
    )
    dropped = env.termination_manager.get_term("obj_dropped").compute()
    return retention_rotation_score(axial if progress_speed is None else progress_speed, cost, dropped, rotation_gain)


def signed_quaternion_increment(quat, previous_quat, axis):
    """Shortest signed world-axis increment, invariant to quaternion sign."""
    delta = quat_mul(quat, inv_quat(previous_quat))
    delta = torch.where(delta[..., :1] < 0, -delta, delta)
    return (axis_angle_from_quat(delta) * axis).sum(-1)


@REWARD_TERM_REGISTRY.register()
class RetainedQuaternionRotationReward(RewardTerm):
    """S14 costs and gain, paying only this step's signed quaternion progress.

    Dividing by dt preserves the previous rad/s gain and ±0.5 saturation.
    Reset copies the actual reset orientation, so teleports earn no credit.
    """

    rotation_gain: float = 4.0

    def build(self):
        super().build()
        self._obj = self._env.entities["obj"]
        self._previous_quat = self._obj.get_quat().clone()

    def reset(self, envs_idx=None):
        ids = slice(None) if envs_idx is None else envs_idx
        self._previous_quat[ids] = self._obj.get_quat()[ids]

    def post_reset(self, envs_idx):
        # LoadGraspPose runs after reward-manager reset; use its final orientation.
        self.reset(envs_idx)

    def compute(self, envs_idx=None):
        quat = self._obj.get_quat()
        axis = self._env.command_manager.get_command("rotation_axis")
        speed = signed_quaternion_increment(quat, self._previous_quat, axis) / self._env.dt
        self._previous_quat.copy_(quat)
        return retained_rotation_reward(self._env, self.rotation_gain, progress_speed=speed)


@REWARD_TERM_REGISTRY.register()
class SignedQuaternionProgressReward(RewardTerm):
    """HORA-style instantaneous signed axial velocity with a reward plateau.

    Every step derives angular velocity from the actual quaternion increment and
    clips it to the configured interval.  Motion at or above the positive cap
    earns the same reward, so the policy is not asked to track one exact speed.
    There is no reward history or accumulated-angle payment.  Reverse motion is
    negative in the current step and dropped objects receive no rotation credit.
    """

    entity_name: str = "obj"
    command_name: str = "rotation_axis"
    angvel_clip_min: float = -0.5
    angvel_clip_max: float = 0.5

    def build(self):
        super().build()
        if self.angvel_clip_min >= self.angvel_clip_max:
            raise ValueError("Rotation velocity clip must have increasing bounds")
        self._obj = self._env.entities[self.entity_name]
        self._previous_quat = self._obj.get_quat().detach().clone()

    def reset(self, envs_idx=None):
        ids = slice(None) if envs_idx is None else envs_idx
        self._previous_quat[ids] = self._obj.get_quat()[ids].detach()

    def post_reset(self, envs_idx):
        self.reset(envs_idx)

    def compute(self, envs_idx=None):
        quat = self._obj.get_quat()
        axis = self._env.command_manager.get_command(self.command_name)
        increment = signed_quaternion_increment(quat, self._previous_quat, axis)
        self._previous_quat.copy_(quat.detach())
        dropped = self._env.termination_manager.get_term("obj_dropped").compute()
        reward = (increment / self._env.dt).clamp(self.angvel_clip_min, self.angvel_clip_max)
        return torch.where(dropped, torch.zeros_like(reward), reward)
