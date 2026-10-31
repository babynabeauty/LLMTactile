"""Bounded wrist control for the multiple-object grasp-lift teachers."""

import torch
import object_wrench  # noqa: F401
from eden.managers import ACTION_TERM_REGISTRY, OBSERVATION_TERM_REGISTRY
from shared_terms import RootPoseController
from smooth_control import limited_target


@ACTION_TERM_REGISTRY.register()
class SmoothRootPoseController(RootPoseController):
    linear_velocity_limit: float = 0.15
    linear_acceleration_limit: float = 1.0
    angular_velocity_limit: float = 0.8
    angular_acceleration_limit: float = 8.0

    def build(self):
        super().build()
        self.target = self.entity.get_dofs_pos(dofs_idx_local=slice(0, 6)).clone()
        self.target_velocity = torch.zeros_like(self.target)
        self.lower = torch.cat((self._pos_min, torch.deg2rad(self._euler_min_deg))).expand_as(self.target)
        self.upper = torch.cat((self._pos_max, torch.deg2rad(self._euler_max_deg))).expand_as(self.target)

    def reset(self, envs_idx=None):
        ids = slice(None) if envs_idx is None else envs_idx
        super().reset(ids)
        self.target[ids] = self.entity.get_dofs_pos(dofs_idx_local=slice(0, 6))[ids].clamp(
            self.lower[ids], self.upper[ids]
        )
        self.target_velocity[ids] = 0
        self._processed_action[ids] = self.target[ids]

    def compute(self, actions):
        super().compute(actions)
        for indices, speed, accel in [
            (slice(0, 3), self.linear_velocity_limit, self.linear_acceleration_limit),
            (slice(3, 6), self.angular_velocity_limit, self.angular_acceleration_limit),
        ]:
            q, v = limited_target(
                self._processed_action[:, indices],
                self.target[:, indices],
                self.target_velocity[:, indices],
                self.lower[:, indices],
                self.upper[:, indices],
                self._env.dt,
                speed,
                accel,
            )
            self.target[:, indices] = q
            self.target_velocity[:, indices] = v
        self._processed_action.copy_(self.target)


@OBSERVATION_TERM_REGISTRY.register()
def root_controller_state(env):
    c = env.action_manager.get_term("ee_controller")
    robot = c.entity
    return torch.cat(
        (c.target, c.target_velocity, robot.get_pos(), robot.get_quat(), robot.get_vel(), robot.get_ang()), -1
    )
