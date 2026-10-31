"""Multi-object residual PPO environment, preserving the legacy actor contract.

One scene has one immutable object geometry and demonstration. Its lanes may
have different masses/frictions; every lane's prefix is physically replayed at
that material before any learned action. No material truth enters the actor.
"""
import json

import numpy as np
import genesis as gs

from .adaptive_env import AdaptiveVecEnv
from .env import TABLE_TOP_Z_M
from .multiobject_geometry import ObjectSpec
from .solver_profile import enforce_trajectory_solver_profile
from .tactile_provenance import trajectory_tactile_evidence


class MultiObjectVecEnv(AdaptiveVecEnv):
    def __init__(self, *args, object_spec='cube', trajectory_path=None, **kwargs):
        if trajectory_path is None:
            raise ValueError('Multi-object experiments require an explicit verified demonstration')
        _,self.source_solver_profile_evidence=enforce_trajectory_solver_profile(trajectory_path,kwargs)
        self.object_spec = ObjectSpec.from_value(object_spec)
        if 'cube_size_m' in kwargs:
            raise ValueError('Specify geometry using object_spec, not cube_size_m')
        if 'terminal_hold_duration_s' in kwargs and kwargs['terminal_hold_duration_s'] != 3.:
            raise ValueError('Multi-object experiments always evaluate the final 3 seconds')
        kwargs['terminal_hold_duration_s'] = 3.
        if kwargs.get('replay_source_prefix_gains',True) is not True:
            raise ValueError('Multi-object experiments must replay recorded source prefix gains')
        kwargs['replay_source_prefix_gains']=True
        kwargs.setdefault('cube_size_m', float(self.object_spec.bounds_size_m[0]))
        super().__init__(*args, trajectory_path=trajectory_path, **kwargs)
        path = self.output_dir/'environment_protocol.json'
        protocol = json.loads(path.read_text())
        # Remove cube-only physical claims while keeping cube_* trace aliases
        # for compatibility with the established trace and replay tooling.
        for key in ('cube_size_m', 'cube_size_readback_m', 'cube_geometry_per_lane',
                    'cube_build_density_kg_m3', 'cube_initial_bottom_readback_m'):
            protocol.pop(key, None)
        protocol.update(
            variant='multiobject_common_gain_v1', object_spec=self.object_spec.protocol(),
            **trajectory_tactile_evidence(trajectory_path,protocol['tactile_model']),
            source_solver_profile_evidence=self.source_solver_profile_evidence,
            object_bounds_readback_m=self.cube_size_readback_m.tolist(),
            object_initial_position_m=self.cube_initial_position.tolist(),
            object_initial_bottom_readback_m=(self.cube_initial_position_readback[:,2]-
                self.object_spec.bounds_size_m[2]/2).tolist(),
            object_build_density_kg_m3=self.cube_build_density_kg_m3,
            density_kg_m3_per_lane=(self.mass/self.object_volume_m3).tolist(),
            density_readback_definition='solver mass divided by analytic primitive volume; not bounding-box volume',
            task_success='original final 3s hold: exact object lowest surface >=5cm above table at every 60Hz sample; no robot collision',
            strict_success='task_success AND >=2 tactile-bearing links with load>.05N at every hold sample',
            success='strict_success; task_success is separately retained',
            force_reward='all_robot_contact_load_N: sum of contact-force vector magnitudes over every robot-object contact',
            mean_contact_force_N='legacy five sensor-bearing-link total; use mean_all_robot_contact_force_N for the new reward metric',
            normal_force='sum(abs(dot(force_a, unit contact normal))) over all robot-object contacts',
            hold_force_window='same final 3s as task success; reported for every episode, interpret alongside success',
            trace_aliases='cube_position_m and cube_quaternion_wxyz refer to the chosen rigid object regardless of shape')
        protocol['read_only_contact_diagnostics']['all_robot_normal_load_N'] = protocol['normal_force']
        path.write_text(json.dumps(protocol, indent=2))

    def _configure_object_geometry(self):
        self.object_volume_m3 = self.object_spec.volume_m3
        self.object_inertia_per_kg = self.object_spec.inertia_per_kg
        self.object_bounds_size_m = self.object_spec.bounds_size_m
        self.cube_initial_position = self.object_spec.initial_position(
            xy=self.cube_initial_position[:2], table_top_z=TABLE_TOP_Z_M)

    def _create_grasp_object(self):
        spec = self.object_spec
        common = {'pos':tuple(self.cube_initial_position)}
        if spec.shape in ('cube','cuboid'):
            morph = gs.morphs.Box(size=tuple(spec.bounds_size_m), **common)
        elif spec.shape=='hex_prism':
            asset=spec.mesh_asset(create=True)
            morph=gs.morphs.Mesh(file=asset['path'],convexify=True,decimate=False,**common)
        elif spec.shape == 'cylinder':
            morph = gs.morphs.Cylinder(radius=spec.radius_m, height=spec.height_m, **common)
        else:
            morph = gs.morphs.Sphere(radius=spec.radius_m, **common)
        return self.scene.add_entity(morph,
            material=gs.materials.Rigid(rho=self.cube_build_density_kg_m3, friction=.8),
            surface=gs.surfaces.Default(color=(.95,.36,.10,1)))

    def object_vertical_extent(self, quaternion_wxyz):
        return self.object_spec.vertical_extent(quaternion_wxyz)

    def reset(self):
        obs = super().reset()
        self.all_force_integral = np.zeros(self.num_envs)
        self.all_normal_integral = np.zeros(self.num_envs)
        self.all_force_peak = np.zeros(self.num_envs)
        self.all_normal_peak = np.zeros(self.num_envs)
        self.hold_all_force_integral = np.zeros(self.num_envs)
        self.hold_all_normal_integral = np.zeros(self.num_envs)
        self.hold_all_force_peak = np.zeros(self.num_envs)
        self.hold_all_normal_peak = np.zeros(self.num_envs)
        self.hold_sensor_force_integral = np.zeros(self.num_envs)
        self.hold_sensor_force_peak = np.zeros(self.num_envs)
        self.extra_sample_duration = 0.
        self.extra_hold_duration = 0.
        return obs

    def _force_penalty_load(self, load):
        return self.last_contact_diagnostics['all_robot_contact_load_N']

    def _accumulate_extra_metrics(self, dt, hold_dt, load):
        total = self.last_contact_diagnostics['all_robot_contact_load_N']
        normal = self.last_contact_diagnostics['all_robot_normal_load_N']
        self.all_force_integral += total*dt
        self.all_normal_integral += normal*dt
        self.all_force_peak = np.maximum(self.all_force_peak, total)
        self.all_normal_peak = np.maximum(self.all_normal_peak, normal)
        self.extra_sample_duration += dt
        if hold_dt:
            self.hold_all_force_integral += total*hold_dt
            self.hold_all_normal_integral += normal*hold_dt
            self.hold_all_force_peak = np.maximum(self.hold_all_force_peak, total)
            self.hold_all_normal_peak = np.maximum(self.hold_all_normal_peak, normal)
            sensor_force = load.sum(-1)
            self.hold_sensor_force_integral += sensor_force*hold_dt
            self.hold_sensor_force_peak = np.maximum(self.hold_sensor_force_peak, sensor_force)
            self.extra_hold_duration += hold_dt

    def _extra_episode_metrics(self, lane):
        duration = max(self.extra_sample_duration, 1e-12)
        hold_duration = max(self.extra_hold_duration, 1e-12)
        return {
            'object_shape':self.object_spec.shape,
            'object_volume_m3':float(self.object_volume_m3),
            'density_kg_m3':float(self.density_readback_kg_m3_per_lane[lane]),
            'max_clearance_m':float(self.max_clearance[lane]),
            'mean_all_robot_contact_force_N':float(self.all_force_integral[lane]/duration),
            'mean_all_robot_normal_force_N':float(self.all_normal_integral[lane]/duration),
            'peak_all_robot_contact_force_N':float(self.all_force_peak[lane]),
            'peak_all_robot_normal_force_N':float(self.all_normal_peak[lane]),
            'hold_mean_all_robot_contact_force_N':float(self.hold_all_force_integral[lane]/hold_duration),
            'hold_mean_all_robot_normal_force_N':float(self.hold_all_normal_integral[lane]/hold_duration),
            'hold_peak_all_robot_contact_force_N':float(self.hold_all_force_peak[lane]),
            'hold_peak_all_robot_normal_force_N':float(self.hold_all_normal_peak[lane]),
            'hold_mean_sensor_contact_force_N':float(self.hold_sensor_force_integral[lane]/hold_duration),
            'hold_peak_sensor_contact_force_N':float(self.hold_sensor_force_peak[lane]),
            'force_measurement_duration_s':float(self.extra_sample_duration),
            'hold_force_measurement_duration_s':float(self.extra_hold_duration)}
