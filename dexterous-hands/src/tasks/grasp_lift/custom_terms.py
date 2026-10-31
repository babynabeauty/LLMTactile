"""Staged grasp-and-lift rewards: reach, then lift, then carry.

Ported from https://github.com/ruoyiqiao/mjlab_hand (its `grasp/mdp/rewards.py`). The three
progress terms are *episodic best-so-far*: each keeps a per-env best and pays only when the policy
beats it, so a partial improvement is banked and cannot be farmed by oscillating across the same
distance. That is what lets the task train from an open hand without a grasp cache -- there is a
gradient during the reach, during the lift, and during the carry, rather than a single sparse
success.
"""

from __future__ import annotations

import torch
from eden.managers import OBSERVATION_TERM_REGISTRY, REWARD_TERM_REGISTRY, ObservationTerm, RewardTerm
from eden.utils.string import resolve_matching_names
from genesis.utils import geom as gu


@OBSERVATION_TERM_REGISTRY.register()
class HandCentricSurfaceVectors(ObservationTerm):
    """Nearest object-surface vectors from hand links, expressed in the palm frame."""

    sensor_names: tuple[str, ...] = ("priv_shape_distance_*",)
    robot_name: str = "robot"

    def build(self) -> None:
        super().build()
        names = list(self._env.sensors)
        _, matched = resolve_matching_names(self.sensor_names, names, preserve_order=True)
        if not matched:
            raise ValueError(f"{type(self).__name__}: sensor_names={self.sensor_names} matched no sensors")
        self._sensors = [self._env.sensors[name] for name in matched]
        self._robot = self._env.entities[self.robot_name]
        self._palm = self._robot.get_link(self._robot.options.metadata.palm_link)

    def compute(self, *args, **kwargs) -> torch.Tensor:
        vectors = []
        for sensor in self._sensors:
            sensor.read()
            vectors.append(sensor.sensor.nearest_points[:, 0] - sensor.get_pos())
        world_vectors = torch.stack(vectors, dim=1)
        palm_quat_inv = gu.inv_quat(self._palm.get_quat()).repeat_interleave(len(self._sensors), dim=0)
        return gu.transform_by_quat(world_vectors.reshape(-1, 3), palm_quat_inv).reshape(self.num_envs, -1)


def _keypoint_positions(robot, link_names: list[str]) -> torch.Tensor:
    """World positions of the named links, shape ``(num_envs, len(link_names), 3)``."""
    return torch.stack([robot.get_link(name).get_pos() for name in link_names], dim=1)


class _LiftGated(RewardTerm):
    """Shared lift bookkeeping: resting height at reset, and a latched 'has lifted' flag.

    Both the approach and lift terms need to know the object's height at the start of the episode
    and whether the lift threshold has ever been crossed since. Neither can read it from the other,
    so the state lives here and both subclass it.

    Height is the object's lowest point (AABB bottom), not its centre: tipping a 10 cm cube onto
    an edge raises the centre by 2 cm without leaving the table, and policies farmed that.
    """

    object_name: str = "obj"
    lift_threshold: float = 0.05

    def build(self) -> None:
        super().build()
        self._obj = self._env.entities[self.object_name]
        n = self._env.num_envs
        self._init_z = self._bottom_z().detach().clone()
        self._lifted = torch.zeros(n, dtype=torch.bool, device=self.device)

    def reset(self, envs_idx: slice | torch.Tensor | None = None) -> None:
        if envs_idx is None:
            envs_idx = slice(None)
        self._init_z[envs_idx] = self._bottom_z()[envs_idx].detach()
        self._lifted[envs_idx] = False

    def _bottom_z(self) -> torch.Tensor:
        return self._obj.get_AABB()[:, 0, 2]

    def _z_lift(self) -> torch.Tensor:
        return self._bottom_z() - self._init_z


@REWARD_TERM_REGISTRY.register()
class ApproachProgressReward(_LiftGated):
    """Per-keypoint closest-approach progress, paid only before the object is lifted.

    Each fingertip banks its own closest distance to the object this episode and is paid for
    beating it. Once the object is off the table the term switches off entirely: the reach is
    solved, and continuing to pay for it would reward letting go and reaching again.
    """

    robot_name: str = "robot"
    clip: float = 10.0

    def build(self) -> None:
        super().build()
        self._robot = self._env.entities[self.robot_name]
        self._links = list(self._robot.options.metadata.fingertip_links)
        self._closest = torch.full(
            (self._env.num_envs, len(self._links)), float("inf"), device=self.device, dtype=torch.float32
        )

    def _distances(self) -> torch.Tensor:
        kp = _keypoint_positions(self._robot, self._links)
        return torch.norm(kp - self._obj.get_pos().unsqueeze(1), dim=-1)

    def reset(self, envs_idx: slice | torch.Tensor | None = None) -> None:
        super().reset(envs_idx)
        if envs_idx is None:
            envs_idx = slice(None)
        self._closest[envs_idx] = self._distances()[envs_idx].detach()

    def compute(self, envs_idx: slice | torch.Tensor | None = None) -> torch.Tensor:
        dist = self._distances()
        gain = torch.clamp(self._closest - dist, min=0.0, max=self.clip)
        self._closest.copy_(torch.minimum(self._closest, dist).detach())
        self._lifted |= self._z_lift() > self.lift_threshold
        return gain.sum(dim=-1) * (~self._lifted).float()


@REWARD_TERM_REGISTRY.register()
class LiftProgressReward(_LiftGated):
    """Height above the resting pose, plus a one-shot bonus the first time it clears threshold.

    The height term saturates at ``lift_threshold`` so it cannot be farmed by lifting ever higher,
    and it is gated off once the bonus has paid, leaving the goal-progress term to take over.
    """

    lift_bonus: float = 10.0
    lift_scale: float = 1.0

    def compute(self, envs_idx: slice | torch.Tensor | None = None) -> torch.Tensor:
        z_lift = self._z_lift()
        was_lifted = self._lifted.clone()
        self._lifted |= z_lift > self.lift_threshold
        height = torch.clamp(z_lift, min=0.0, max=self.lift_threshold) * (~was_lifted).float() * self.lift_scale
        return height + self.lift_bonus * (self._lifted & ~was_lifted).float()


@REWARD_TERM_REGISTRY.register()
class SustainedLiftReward(LiftProgressReward):
    """L2: reward normalized current height throughout the hold and carry.

    The legacy term stops paying height after the first lift. This experiment
    keeps height credit after crossing the threshold, saturates at that threshold,
    and removes credit immediately when the object returns to its resting height.
    The threshold-crossing bonus still pays only once per episode.
    """

    def build(self) -> None:
        super().build()
        if self.lift_threshold <= 0:
            raise ValueError("Sustained lift requires a positive height threshold")

    def compute(self, envs_idx: slice | torch.Tensor | None = None) -> torch.Tensor:
        z_lift = self._z_lift()
        was_lifted = self._lifted.clone()
        self._lifted |= z_lift > self.lift_threshold
        height = (z_lift / self.lift_threshold).clamp(0.0, 1.0) * self.lift_scale
        return height + self.lift_bonus * (self._lifted & ~was_lifted).float()


@REWARD_TERM_REGISTRY.register()
class NaturalLiftMotionPenalty(_LiftGated):
    """Bounded cost for abrupt wrist motion and unstable pickup/carry.

    Uses measured velocities, not policy action changes. The total cost saturates
    so an external push cannot create an unbounded incentive to drop the object.
    """

    robot_name: str = "robot"
    palm_normal: tuple[float, float, float] | None = None
    max_cost: float = 0.25

    def build(self):
        super().build()
        self._robot = self._env.entities[self.robot_name]
        self._links = list(self._robot.options.metadata.fingertip_links)
        local_normal = self.palm_normal or self._robot.options.metadata.palm_normal
        self._palm_normal = torch.tensor(local_normal, device=self.device, dtype=torch.float32)
        self._prev_vel = self._robot.get_vel().detach().clone()
        self._prev_ang = self._robot.get_ang().detach().clone()

    def reset(self, envs_idx=None):
        super().reset(envs_idx)
        ids = slice(None) if envs_idx is None else envs_idx
        self._prev_vel[ids] = self._robot.get_vel()[ids].detach()
        self._prev_ang[ids] = self._robot.get_ang()[ids].detach()

    def compute(self, envs_idx=None):
        vel, ang = self._robot.get_vel(), self._robot.get_ang()
        acceleration = (vel - self._prev_vel) / self._env.dt
        angular_acceleration = (ang - self._prev_ang) / self._env.dt
        self._prev_vel.copy_(vel.detach())
        self._prev_ang.copy_(ang.detach())
        obj_pos = self._obj.get_pos()
        lifted = self._z_lift() > self.lift_threshold
        distance = (_keypoint_positions(self._robot, self._links) - obj_pos[:, None]).norm(dim=-1).amin(-1)
        near_pickup = (distance < 0.06) & ~lifted
        fast_approach = ((vel.norm(dim=-1) - 0.08).clamp_min(0) / 0.08).square()
        # Rigid carry includes the velocity induced by wrist rotation.
        carry_velocity = vel + torch.linalg.cross(ang, obj_pos - self._robot.get_pos(), dim=-1)
        slip = ((self._obj.get_vel() - carry_velocity).norm(dim=-1) / 0.10).square()
        spin = ((self._obj.get_ang() - ang).norm(dim=-1) / 0.5).square()
        quat = self._robot.get_quat()
        normal = self._palm_normal.to(dtype=quat.dtype).expand_as(vel)
        down = -gu.transform_by_quat(normal, quat)[:, 2]
        tilt = ((0.8660254 - down).clamp_min(0) / 0.5).square()
        cost = (
            0.25 * acceleration.square().sum(-1)
            + 0.1 * (angular_acceleration / 4.0).square().sum(-1)
            + near_pickup * fast_approach
            + lifted * (slip + 0.1 * spin)
            + 0.1 * tilt
        )
        return self.max_cost * cost / (1.0 + cost)


@REWARD_TERM_REGISTRY.register()
class GoalProgressReward(RewardTerm):
    """Best-so-far progress toward the goal position, plus a bonus inside the success tolerance.

    ``_best`` resets when the goal moves as well as on episode reset, so a resampled goal starts a
    fresh budget instead of paying out the jump in distance.
    """

    object_name: str = "obj"
    command_name: str = "goal_pos"
    scale: float = 500.0
    success_tolerance: float = 0.03
    success_bonus: float = 10.0

    def build(self) -> None:
        super().build()
        self._obj = self._env.entities[self.object_name]
        self._cmd = self._env.command_manager.get_term(self.command_name)
        self._prev_goal = self._goal().detach().clone()
        self._best = self._distance().detach().clone()

    def _goal(self) -> torch.Tensor:
        # `UniformT3Command` exposes the sampled target as `command`; it is world-frame here
        # because the task leaves `marker_frame_entity` unset.
        return self._cmd.command

    def _distance(self) -> torch.Tensor:
        return torch.norm(self._obj.get_pos() - self._goal(), dim=-1)

    def reset(self, envs_idx: slice | torch.Tensor | None = None) -> None:
        if envs_idx is None:
            envs_idx = slice(None)
        self._prev_goal[envs_idx] = self._goal()[envs_idx].detach()
        self._best[envs_idx] = self._distance()[envs_idx].detach()

    def compute(self, envs_idx: slice | torch.Tensor | None = None) -> torch.Tensor:
        goal = self._goal()
        dist = self._distance()
        moved = torch.any(torch.abs(goal - self._prev_goal) > 1e-6, dim=-1)
        gain = torch.clamp(self._best - dist, min=0.0) * self.scale
        gain = torch.where(moved, torch.zeros_like(gain), gain)
        self._best.copy_(torch.where(moved, dist, torch.minimum(self._best, dist)).detach())
        self._prev_goal.copy_(goal.detach())
        return gain + self.success_bonus * (dist < self.success_tolerance).float()


@REWARD_TERM_REGISTRY.register()
class PlaneClearanceReward(RewardTerm):
    """Bounded clearance of the complete object above a horizontal support plane.

    The object's lowest world-space AABB point must rise above ``plane_z`` before this term pays.
    Tilting can raise the centre and upper edge while the object remains supported by a lower edge;
    that motion receives exactly zero reward here.
    """

    object_name: str = "obj"
    plane_z: float = 0.5
    threshold: float = 0.05

    def build(self) -> None:
        super().build()
        if self.threshold <= 0:
            raise ValueError("Plane-clearance threshold must be positive")
        self._obj = self._env.entities[self.object_name]

    def compute(self, envs_idx: slice | torch.Tensor | None = None) -> torch.Tensor:
        bottom_z = self._obj.get_AABB()[:, 0, 2]
        return ((bottom_z - self.plane_z) / self.threshold).clamp(0.0, 1.0)


@REWARD_TERM_REGISTRY.register()
class SustainedPlaneLiftReward(SustainedLiftReward):
    """Dense sustained lift reward based only on clearance above the plane.

    Clearance begins at zero while any part of the object touches or penetrates
    the support plane.  It rises linearly to one and stays there while the whole
    object is clear.  A one-shot bonus marks full threshold clearance.  Object
    centre rise and tilt cannot activate either component.
    """

    object_name: str = "obj"
    plane_z: float = 0.5
    def compute(self, envs_idx=None) -> torch.Tensor:
        clearance = (self._obj.get_AABB()[:, 0, 2] - self.plane_z).clamp_min(0.0)
        was_lifted = self._lifted.clone()
        self._lifted |= clearance > self.lift_threshold
        sustained = (clearance / self.lift_threshold).clamp(0.0, 1.0) * self.lift_scale
        return sustained + self.lift_bonus * (self._lifted & ~was_lifted).float()


@REWARD_TERM_REGISTRY.register()
class FingertipWrapReward(RewardTerm):
    """Best antipodal pair: two fingertips touching the object on opposite sides.

    Palm proximity is maximised by resting on top of the object, and a plain fingertip-contact term by
    lying flat against one face (the L10 videos). A lift needs something to squeeze against, so this
    pays, per step, the best pair of tips weighted by how opposed their directions from the object's
    centre are: ``touch_i * touch_j * clamp(-u_i . u_j, 0, 1)`` with ``touch = exp(-d^2 / sigma^2)``.
    """

    robot_name: str = "robot"
    object_name: str = "obj"
    sensor_prefix: str = "priv_surface_distance_"
    sigma: float = 0.03

    def build(self) -> None:
        super().build()
        robot = self._env.entities[self.robot_name]
        self._obj = self._env.entities[self.object_name]
        names = [self.sensor_prefix + link for link in robot.options.metadata.fingertip_links]
        missing = [n for n in names if n not in self._env.sensors]
        if missing:
            raise ValueError(f"{type(self).__name__}: missing sensors {missing}")
        self._sensors = [self._env.sensors[n] for n in names]
        n = len(self._sensors)
        self._pairs = torch.triu_indices(n, n, offset=1, device=self.device)

    def compute(self, envs_idx: slice | torch.Tensor | None = None) -> torch.Tensor:
        centre = self._obj.get_pos()
        tips, touch = [], []
        for sensor in self._sensors:
            sensor.read()
            tip = sensor.get_pos()
            dist_sq = (sensor.sensor.nearest_points[:, 0] - tip).square().sum(dim=-1)
            tips.append(tip)
            touch.append(torch.exp(-dist_sq / self.sigma**2))
        tips = torch.stack(tips, dim=1)  # (E, n, 3)
        touch = torch.stack(touch, dim=1)  # (E, n)
        u = torch.nn.functional.normalize(tips - centre[:, None], dim=-1)
        a, b = self._pairs
        opposed = (-(u[:, a] * u[:, b]).sum(dim=-1)).clamp(0.0, 1.0)
        return (touch[:, a] * touch[:, b] * opposed).amax(dim=-1)


@REWARD_TERM_REGISTRY.register()
class GoalTrackingReward(_LiftGated):
    """Gaussian of the object-to-goal distance, paid while the object is off the table.

    The best-so-far progress term pays each metre of approach once, so hovering anywhere pays as
    much as sitting at the goal. This keeps paying for staying there.
    """

    command_name: str = "goal_pos"
    sigma: float = 0.05
    # A second, coarse Gaussian averaged with the fine one. With the goal 15-30 cm above the
    # resting object the fine scale alone is zero everywhere a freshly lifted object can be.
    sigma_far: float | None = None

    def build(self) -> None:
        super().build()
        self._cmd = self._env.command_manager.get_term(self.command_name)

    def compute(self, envs_idx: slice | torch.Tensor | None = None) -> torch.Tensor:
        dist_sq = (self._obj.get_pos() - self._cmd.command).square().sum(dim=-1)
        lifted = self._z_lift() > self.lift_threshold
        tracking = torch.exp(-dist_sq / self.sigma**2)
        if self.sigma_far is not None:
            tracking = 0.5 * (tracking + torch.exp(-dist_sq / self.sigma_far**2))
        return tracking * lifted.float()


@REWARD_TERM_REGISTRY.register()
class SettledCarryBonus(_LiftGated):
    """Pays the evaluation's settled-carry condition: lifted, inside tolerance, and nearly still."""

    command_name: str = "goal_pos"
    tolerance: float = 0.03
    max_linear_speed: float = 0.1
    max_angular_speed: float = 0.5

    def build(self) -> None:
        super().build()
        self._cmd = self._env.command_manager.get_term(self.command_name)

    def compute(self, envs_idx: slice | torch.Tensor | None = None) -> torch.Tensor:
        near = (self._obj.get_pos() - self._cmd.command).norm(dim=-1) < self.tolerance
        still = (self._obj.get_vel().norm(dim=-1) < self.max_linear_speed) & (
            self._obj.get_ang().norm(dim=-1) < self.max_angular_speed
        )
        return ((self._z_lift() > self.lift_threshold) & near & still).float()


@REWARD_TERM_REGISTRY.register()
class PalmDownPenalty(RewardTerm):
    """Yaw-invariant soft palm tilt cost, zero down and two up."""

    robot_name: str = "robot"
    palm_normal: tuple[float, float, float] | None = None

    def build(self):
        super().build()
        self._robot = self._env.entities[self.robot_name]
        self._palm = self._robot.get_link(self._robot.options.metadata.palm_link)
        normal = self.palm_normal or self._robot.options.metadata.palm_normal
        self._normal = torch.tensor(normal, device=self.device, dtype=torch.float32)
        self._normal = self._normal / self._normal.norm()

    def compute(self, envs_idx=None):
        quat = self._palm.get_quat()
        normal = self._normal.to(quat).expand(quat.shape[0], -1)
        down_dot = -gu.transform_by_quat(normal, quat)[:, 2]
        return 1.0 - down_dot.clamp(-1.0, 1.0)
