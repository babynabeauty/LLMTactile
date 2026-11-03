"""v1 environment: residual finger control from the first 'close' command.

One scene holds one object geometry and the recorded bases of that object.
Every base prefix is replayed physically once, in parallel lanes, to obtain
the robot and object state at takeover; a prefix that touches the object is
excluded. Each lane then runs its own episodes: reset places robot and object
in a base's takeover state, samples material and fragility, and continues the
base with a bounded residual on the hand-shape joints. Mass, friction and
F_break never enter the actor; only the (possibly wrong) Astra level does.
"""
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
import xml.etree.ElementTree as ET
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
os.environ.setdefault('OMP_NUM_THREADS', '4')
import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv
import genesis as gs

from .core import ROOT, RateConfig, hold_success, interval_overlap
from .multiobject_geometry import ObjectSpec
from .solver_profile import normalize_solver_profile, read_solver_readback, validate_solver_readback
from .tactile_provenance import current_tactile_model
from .v1_task import (V1Config, SamplerConfig, BreakMonitor, SlipTracker, REGIONS, FINGERS, PHASES,
    HAND_SHAPES, YAW_PERIOD_DEG, base_draw_weights, hand_shape_joints, link_region, limit_residual,
    break_soft_rate, object_offsets, opposing_support, sample_episode, wrap_yaw)
sys.path.insert(0, str(ROOT/'ur7e_xhand_sim'))
from build_assets import build, ARM_JOINTS, HAND_JOINTS
from platform_asset import build_platform, FLOOR_Z, PLATFORM_POSITION, TABLE_TOP_Z

PREFIX_CONTACT_LIMIT_N = .05
STATE_FIELDS = ('q_normalized[18]', 'qd_div_3[18]', 'command_minus_q_div_.2[18]',
    'next_nominal_minus_nominal_div_.1[18]', 'grip_gain_minus_1[1]', f'phase_one_hot{list(PHASES)}',
    'episode_progress[1]', 'astra_level_one_hot[4]', f'hand_shape_finger_mask{list(FINGERS)}',
    'previous_residual_div_bound[action_joints]')
REWARD_TERMS = ('progress', 'force', 'break_soft', 'slip', 'residual_change', 'terminal')


def cpu(x):
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def quaternion_multiply(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1*w2-x1*x2-y1*y2-z1*z2, w1*x2+x1*w2+y1*z2-z1*y2,
                     w1*y2-x1*z2+y1*w2+z1*x2, w1*z2+x1*y2-y1*x2+z1*w2])


def quaternion_matrix(q):
    w, x, y, z = np.moveaxis(np.asarray(q, dtype=float), -1, 0)
    return np.stack([
        np.stack([1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)], -1),
        np.stack([2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)], -1),
        np.stack([2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)], -1)], -2)


class V1VecEnv(VecEnv):
    def __init__(self, object_spec, bases, *, n_envs=12, seed=0, output_dir=None, object_id='object',
                 hand_shape='pinch', action_joints=None, config=None, sampler=None, use_level=True,
                 joint_friction_Nm=(0., 0.), tactile_noise_N=0., fixed_gain=None, augment=False):
        self.render_mode = None
        self.rates = RateConfig()
        self.config = config or V1Config()
        self.sampler = sampler or SamplerConfig()
        self.object_id = object_id
        self.object_spec = ObjectSpec.from_value(object_spec)
        self.use_level = bool(use_level)
        self.joint_friction_Nm = tuple(float(v) for v in joint_friction_Nm)
        if len(self.joint_friction_Nm) != 2 or not 0 <= self.joint_friction_Nm[0] <= self.joint_friction_Nm[1]:
            raise ValueError('joint_friction_Nm must be an ordered nonnegative [low, high] range')
        self.tactile_noise_N = float(tactile_noise_N)
        # None follows each base's recorded gain; a number replaces it after
        # takeover (fixed-gain baselines, residual left at zero).
        self.fixed_gain = None if fixed_gain is None else float(fixed_gain)
        self.rng = np.random.default_rng(seed)
        self.seed_value = seed
        self.output_dir = Path(output_dir or ROOT/'outputs/v1_runtime')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.all_bases = list(bases)
        if not self.all_bases:
            raise ValueError('At least one base is required')
        self.solver_profile = normalize_solver_profile(self.all_bases[0].solver_profile)
        # The action covers the joint union over the experiment's hand shapes so
        # every scene shares one policy; joints outside this scene's shape stay 0.
        if hand_shape not in HAND_SHAPES:
            raise ValueError(f'Unknown hand shape {hand_shape}')
        self.hand_shape = hand_shape
        own = hand_shape_joints(hand_shape)
        self.action_joints = np.asarray(own if action_joints is None else action_joints, dtype=int)
        if not set(own) <= set(self.action_joints.tolist()):
            raise ValueError('action_joints must include every joint of this hand shape')
        self.active = np.isin(self.action_joints, own).astype(float)
        self.controlled = self.action_joints+6
        self.gain_joints = np.asarray(own)+6
        self.finger_mask = np.array([f in HAND_SHAPES[hand_shape] for f in FINGERS], np.float32)
        self.state_dim = 18*4+1+len(PHASES)+1+4+len(FINGERS)+len(self.action_joints)
        observation_space = spaces.Dict({
            'tactile': spaces.Box(-np.inf, np.inf, (12, 5, 120, 3), np.float32),
            'state': spaces.Box(-np.inf, np.inf, (12, self.state_dim), np.float32)})
        super().__init__(n_envs, observation_space, spaces.Box(-1., 1., (len(self.action_joints),), np.float32))
        self._build_scene(seed)
        self._capture_takeover_states(augment)
        self.specs, self.next_spec = None, 0
        self.episode_counter = 0
        self._closed = False
        self._pending_action = None
        self._write_protocol()
        self.reset()

    # ------------------------------------------------------------------ scene
    def _object_morph(self, position):
        spec, common = self.object_spec, {'pos': tuple(position)}
        if spec.shape in ('cube', 'cuboid'):
            return gs.morphs.Box(size=tuple(spec.bounds_size_m), **common)
        if spec.shape == 'hex_prism':
            return gs.morphs.Mesh(file=spec.mesh_asset(create=True)['path'], convexify=True, decimate=False, **common)
        if spec.shape == 'cylinder':
            return gs.morphs.Cylinder(radius=spec.radius_m, height=spec.height_m, **common)
        return gs.morphs.Sphere(radius=spec.radius_m, **common)

    def _build_scene(self, seed):
        model, self.layouts = build()
        self.model_path = Path(model)
        if not gs._initialized:
            gs.init(backend=gs.gpu, precision='32', seed=seed, logging_level='warning',
                    use_deterministic_algorithms=True)
        if not gs.use_deterministic_algorithms:
            raise RuntimeError('v1 training requires deterministic Genesis solver selection')
        self.device = gs.device
        self.scene = gs.Scene(sim_options=gs.options.SimOptions(dt=self.rates.physics_dt, substeps=1),
            rigid_options=gs.options.RigidOptions(enable_self_collision=True, enable_joint_limit=True,
                enable_neutral_collision=True, batch_links_info=True, batch_dofs_info=True,
                noslip_iterations=self.solver_profile['noslip_iterations'],
                noslip_tolerance=self.solver_profile['noslip_tolerance']),
            show_viewer=False)
        # Same floor and imported platform as environment.Sandbox, where demonstrations and bases are recorded.
        self.scene.add_entity(gs.morphs.Plane(pos=(0, 0, FLOOR_Z)))
        self.table = self.scene.add_entity(gs.morphs.URDF(file=str(build_platform()[0]), pos=PLATFORM_POSITION,
                                                          fixed=True), material=gs.materials.Rigid(friction=.8))
        self.robot = self.scene.add_entity(gs.morphs.URDF(file=str(model), fixed=True, pos=(0, 0, .2),
            merge_fixed_links=True, links_to_keep=['right_hand_link', 'tool0']),
            material=gs.materials.Rigid(friction=.8))
        self.cube = self.scene.add_entity(self._object_morph(self.all_bases[0].object_position),
            material=gs.materials.Rigid(rho=.1/self.object_spec.volume_m3, friction=.8),
            surface=gs.surfaces.Default(color=(.95, .36, .10, 1)))
        self.sensors = []
        for layout in self.layouts:
            link = self.robot.get_link(layout['link_name'])
            self.sensors.append(self.scene.add_sensor(gs.sensors.KinematicTaxel(
                entity_idx=self.robot.idx, link_idx_local=link.idx_local,
                probe_local_pos=layout['positions_link_m'], probe_radius=.0015, history_length=0,
                normal_stiffness=500., normal_damping=1., normal_exponent=1.2,
                shear_scalar=2., twist_scalar=0., draw_debug=False)))
        self.scene.build(n_envs=self.num_envs)
        self.names = ARM_JOINTS+HAND_JOINTS
        self.indices = [self.robot.get_joint(n).dofs_idx_local[0] for n in self.names]
        self.finger_links = [self.robot.get_link(l['link_name']).idx for l in self.layouts]
        self.palm = self.robot.get_link('right_hand_link')
        self.solver = self.scene.rigid_solver
        self.solver_readback = validate_solver_readback(self.solver_profile, read_solver_readback(self.solver))
        lo, hi = self.robot.get_dofs_limit(self.indices)
        self.lower, self.upper = cpu(lo).reshape(-1, 18)[0], cpu(hi).reshape(-1, 18)[0]
        self.mid, self.half = (self.lower+self.upper)/2, (self.upper-self.lower)/2
        self.base_kp = np.asarray([4500, 4500, 3500, 2000, 2000, 2000]+[3.]*12, np.float32)
        self.base_kv = np.asarray([450, 450, 350, 200, 200, 200]+[.08]*12, np.float32)
        xml = ET.parse(model).getroot()
        self.effort = np.array([float(xml.find(f"joint[@name='{n}']/limit").get('effort')) for n in self.names], np.float32)
        self.robot.set_dofs_force_range(-self.effort, self.effort, self.indices)
        self.home = np.r_[np.deg2rad([0, -89.95, 89.96, -89.96, -89.95, -44.98]), np.zeros(12)]
        n_links = max(link.idx for entity in self.scene.entities for link in entity.links)+1
        region = torch.full((n_links,), -1, dtype=torch.long, device=self.device)
        self.link_regions = {}
        for link in self.robot.links:
            region[link.idx] = link_region(link.name)
            self.link_regions[link.name] = REGIONS[int(region[link.idx])]
        self.region_of_link = region
        self.gains = np.full(self.num_envs, np.nan, np.float32)
        self.mass = np.full(self.num_envs, .1)
        self.mu = np.full(self.num_envs, .8)

    def _set_materials(self, lanes, mass, mu):
        lanes = np.asarray(lanes)
        self.mass[lanes], self.mu[lanes] = mass, mu
        self.cube.set_mass(np.asarray(mass, np.float32)[:, None], envs_idx=lanes)
        ratio = (np.asarray(mu, np.float32)/.8)[:, None]
        self.cube.set_friction_ratio(ratio, envs_idx=lanes)
        self.robot.set_friction_ratio(np.repeat(ratio, self.robot.n_links, axis=1), envs_idx=lanes)
        readback = cpu(self.cube.get_mass(envs_idx=lanes)).reshape(-1)
        np.testing.assert_allclose(readback, mass, rtol=1e-5, err_msg='Object mass readback differs')

    def _set_gains(self, lanes, gain):
        """Recorded grip gain scales the hand-shape finger joints."""
        lanes, gain = np.asarray(lanes), np.asarray(gain, np.float32)
        changed = self.gains[lanes] != gain
        if not changed.any():
            return
        lanes, gain = lanes[changed], gain[changed]
        self.gains[lanes] = gain
        kp = np.repeat(self.base_kp[None], len(lanes), axis=0)
        kv = np.repeat(self.base_kv[None], len(lanes), axis=0)
        kp[:, self.gain_joints] *= gain[:, None]
        kv[:, self.gain_joints] *= np.sqrt(gain)[:, None]
        self.robot.set_dofs_kp(kp, self.indices, envs_idx=lanes)
        self.robot.set_dofs_kv(kv, self.indices, envs_idx=lanes)

    # ---------------------------------------------------------------- contact
    def _contacts(self):
        c = self.robot.get_contacts(is_padded=True)
        a, b, valid = c['link_a'].long(), c['link_b'].long(), c['valid_mask']
        obj = self.cube.base_link.idx
        pair = valid & ((a == obj) | (b == obj))
        other = torch.where(a == obj, b, a)
        normal = c['normal']/torch.linalg.vector_norm(c['normal'], dim=-1, keepdim=True).clamp_min(1e-9)
        normal_force = torch.abs((c['force_a']*normal).sum(-1))
        return c, a, b, valid, pair, other, normal_force

    def _region_force(self, pair, other, normal_force):
        region = self.region_of_link[other.clamp(0, len(self.region_of_link)-1)]
        use = pair & (region >= 0)
        out = torch.zeros((self.num_envs, len(REGIONS)), device=normal_force.device, dtype=normal_force.dtype)
        out.scatter_add_(1, region.clamp_min(0), torch.where(use, normal_force, 0.))
        return out

    def region_normal_force(self):
        """Normal resultant per region (N), every physics step; F_break uses its max."""
        _, _, _, _, pair, other, normal_force = self._contacts()
        return cpu(self._region_force(pair, other, normal_force))

    def _metrics(self):
        c, a, b, valid, pair, other, normal_force = self._contacts()
        magnitude = torch.linalg.vector_norm(c['force_a'], dim=-1)
        robot_a = (a >= self.robot.link_start) & (a < self.robot.link_end)
        robot_b = (b >= self.robot.link_start) & (b < self.robot.link_end)
        self_contact = (valid & robot_a & robot_b & (magnitude > .01)).any(-1)
        table = self.table.base_link.idx
        table_normal = torch.where(valid & ((a == table) | (b == table)), normal_force, 0.).sum(-1)
        force = torch.where(pair, magnitude, 0.)
        finger = torch.stack([torch.where(pair & (other == link), force, 0.).sum(-1) for link in self.finger_links], -1)
        robot_pair = pair & (other >= self.robot.link_start) & (other < self.robot.link_end)
        all_load = torch.where(robot_pair, magnitude, 0.).sum(-1)
        region = self._region_force(pair, other, normal_force)
        position, quaternion = self.cube.get_pos(), self.cube.get_quat()
        clearance = position[:, 2]-self.object_spec.vertical_extent(quaternion)-TABLE_TOP_Z
        # Every hand/object contact must carry this lane's sampled friction.
        from genesis.utils.misc import qd_to_torch
        state = self.solver.collider.collider_state
        friction = qd_to_torch(state.contact_data.friction, transpose=True)
        order = qd_to_torch(state.contact_sort_idx, transpose=True).long()
        friction = friction.gather(1, order.clamp(0, friction.shape[1]-1))[:, :pair.shape[1]]
        mu = torch.as_tensor(self.mu, device=friction.device, dtype=friction.dtype)[:, None]
        friction_error = torch.where(pair, (friction-mu).abs(), 0.).max(-1).values
        out = {k: cpu(v).astype(np.float64) for k, v in dict(clearance_m=clearance, finger_load_N=finger,
            all_robot_load_N=all_load, region_normal_N=region, table_normal_N=table_normal,
            object_position_m=position, object_quaternion_wxyz=quaternion,
            palm_position_m=self.palm.get_pos(), palm_quaternion_wxyz=self.palm.get_quat(),
            friction_error=friction_error).items()}
        out['self_contact'] = cpu(self_contact).astype(bool)
        if not all(np.isfinite(v).all() for v in out.values()):
            raise FloatingPointError('Nonfinite physical state')
        if (out['friction_error'] > 1e-5).any():
            raise AssertionError('Contact friction differs from the lane material')
        rotation = quaternion_matrix(out['palm_quaternion_wxyz'])
        out['object_in_palm_m'] = np.einsum('nji,nj->ni', rotation, out['object_position_m']-out['palm_position_m'])
        return out

    def _sample_tactile(self):
        value = cpu(torch.stack([s.read().force for s in self.sensors], dim=1))
        if value.shape != (self.num_envs, 5, 120, 3):
            raise RuntimeError(f'Unexpected tactile dimensions: {value.shape}')
        if not np.isfinite(value).all():
            raise FloatingPointError('Nonfinite tactile')
        if self.tactile_noise_N:
            value = value+self.rng.normal(0, self.tactile_noise_N, value.shape).astype(np.float32)
        self.history[:, :-1] = self.history[:, 1:]
        self.history[:, -1] = value

    # ---------------------------------------------------------------- prefix
    def _replay_prefixes(self, jobs):
        """Replay (base, object position, object quaternion) prefixes once, in parallel lanes.

        Lanes whose takeover comes earlier hold their takeover command until the
        group finishes. Contact is only audited before each lane's own takeover.
        """
        n, records = self.num_envs, []
        lanes = np.arange(n)
        for first in range(0, len(jobs), n):
            group = jobs[first:first+n]
            lane_jobs = [group[min(i, len(group)-1)] for i in range(n)]
            lane_bases = [job[0] for job in lane_jobs]
            takeover = np.array([b.takeover_tick for b in lane_bases])
            self.robot.set_dofs_position(np.repeat(self.home[None], n, axis=0), self.indices, zero_velocity=True)
            self.cube.set_pos(np.stack([job[1] for job in lane_jobs]), zero_velocity=True)
            self.cube.set_quat(np.stack([job[2] for job in lane_jobs]))
            self._set_materials(lanes, np.full(n, .05), np.full(n, .8))
            contact = np.zeros(n)
            print(f'PREFIX {first}-{first+len(group)-1} of {len(jobs)}, {takeover.max()} ticks', flush=True)
            for tick in range(int(takeover.max())):
                if tick % self.rates.action_stride == 0:
                    t = np.minimum(tick, takeover)/self.rates.physics_hz
                    self._set_gains(lanes, [b.trajectory.grip_gain_at(ti) for b, ti in zip(lane_bases, t)])
                    self.robot.control_dofs_position(np.stack([b.trajectory.target(ti)
                        for b, ti in zip(lane_bases, t)]), self.indices)
                self.scene.step()
                if (tick+1) % self.rates.tactile_stride == 0:
                    contact = np.maximum(contact, np.where(tick < takeover, self.region_normal_force().sum(-1), 0.))
            q = cpu(self.robot.get_dofs_position(self.indices)).reshape(n, 18)
            position = cpu(self.cube.get_pos()).reshape(n, 3)
            quaternion = cpu(self.cube.get_quat()).reshape(n, 4)
            palm = cpu(self.palm.get_quat()).reshape(n, 4)
            for i in range(len(group)):
                records.append(dict(q=q[i].copy(), object_position=position[i].copy(),
                    object_quaternion=quaternion[i].copy(), palm_quaternion=palm[i].copy(),
                    prefix_contact_N=float(contact[i])))
        return records

    def _capture_takeover_states(self, augment):
        started = time.monotonic()
        records = self._replay_prefixes([(b, b.object_position, b.object_quaternion) for b in self.all_bases])
        keep = [r['prefix_contact_N'] <= PREFIX_CONTACT_LIMIT_N for r in records]
        self.excluded_bases = [dict(base_id=b.base_id, prefix_contact_N=r['prefix_contact_N'])
                               for b, r, k in zip(self.all_bases, records, keep) if not k]
        self.bases = [b for b, k in zip(self.all_bases, keep) if k]
        self.takeover_states = [r for r, k in zip(records, keep) if k]
        if self.excluded_bases:
            print('EXCLUDED bases touching the object before close:', self.excluded_bases, flush=True)
        if not self.bases:
            raise RuntimeError('Every base touches the object before close; takeover must move earlier')
        self.base_p = base_draw_weights([b.label for b in self.bases], self.sampler.r_max_fraction)
        self.variants = [[] for _ in self.bases]
        if augment:
            self._shift_s_bases()
        self.prefix_wall_s = time.monotonic()-started

    def _shift_s_bases(self):
        """S-base variants whose object starts shifted (PLAN_v1 section 6).

        Composite deviations are drawn from this scene's S/R deviations within the
        offset bounds; a variant whose prefix touches the object is dropped.
        """
        s = self.sampler
        period = YAW_PERIOD_DEG.get(self.object_spec.shape)
        pool = [(d['palm_frame_m'], d['yaw_deg']) for d in (b.metadata.get('deviation') for b in self.bases)
                if d and d['distance_m'] <= s.offset_max_m and d['yaw_deg'] <= s.offset_max_yaw_deg]
        if not pool:
            return
        jobs, owners = [], []
        for k, (b, state) in enumerate(zip(self.bases, self.takeover_states)):
            if b.label != 'S' or 'deviation' not in b.metadata:
                continue
            own = np.asarray(b.metadata['deviation']['palm_frame_m'], float)
            yaw_now = wrap_yaw(np.degrees(2*np.arctan2(b.object_quaternion[3], b.object_quaternion[0])), period)
            for shift, turn in object_offsets(self.rng, pool, own, yaw_now, period,
                                              quaternion_matrix(state['palm_quaternion']), s.s_offset_variants, s):
                half = np.radians(turn)/2
                jobs.append((b, b.object_position+shift,
                             quaternion_multiply([np.cos(half), 0., 0., np.sin(half)], b.object_quaternion)))
                owners.append((k, shift, turn))
        for (k, shift, turn), record in zip(owners, self._replay_prefixes(jobs)):
            if record['prefix_contact_N'] <= PREFIX_CONTACT_LIMIT_N:
                self.variants[k].append(dict(record, offset_m=shift, turn_deg=turn))

    # --------------------------------------------------------------- episodes
    def set_episode_specs(self, specs):
        """Evaluation: lanes consume these in order; afterwards episodes are marked discard."""
        ids = {b.base_id for b in self.bases}
        if any(s['base_id'] not in ids for s in specs):
            raise ValueError('Every evaluation spec must name an included base_id')
        self.specs, self.next_spec = list(specs), 0

    def _targets(self, t):
        return np.stack([self.bases[k].trajectory.target(ti) for k, ti in zip(self.lane_base, t)])

    def _base_gain(self, t):
        if self.fixed_gain is None:
            return np.array([self.bases[k].trajectory.grip_gain_at(ti) for k, ti in zip(self.lane_base, t)])
        return np.full(self.num_envs, float(self.fixed_gain))

    def _draw(self):
        if self.specs is not None and self.next_spec < len(self.specs):
            spec = dict(self.specs[self.next_spec], spec_index=self.next_spec, discard=False)
            self.next_spec += 1
            base = next(i for i, b in enumerate(self.bases) if b.base_id == spec['base_id'])
            return base, spec, self.takeover_states[base]
        k = int(self.rng.choice(len(self.bases), p=self.base_p))
        state = self.takeover_states[k]
        if self.variants[k] and self.rng.random() < self.sampler.s_offset_fraction:
            state = self.variants[k][self.rng.integers(len(self.variants[k]))]
        spec = dict(sample_episode(self.rng, self.sampler), spec_index=None, discard=self.specs is not None,
                    object_offset_m=float(np.linalg.norm(state.get('offset_m', 0.))),
                    object_turn_deg=float(state.get('turn_deg', 0.)))
        return k, spec, state

    def _reset_lanes(self, lanes):
        lanes = np.asarray(lanes, dtype=int)
        if not len(lanes):
            return
        draws = [self._draw() for _ in lanes]
        self.lane_base[lanes] = [d[0] for d in draws]
        for lane, (_, spec, _) in zip(lanes, draws):
            self.episode[lane] = spec
        states = [d[2] for d in draws]
        self.lane_start[lanes] = [self.bases[k].takeover_tick for k in self.lane_base[lanes]]
        self.lane_end[lanes] = [self.bases[k].end_tick for k in self.lane_base[lanes]]
        self.lane_tick[lanes] = self.lane_start[lanes]
        self.f_break[lanes] = [d[1]['f_break_N'] for d in draws]
        self.astra_level[lanes] = [d[1]['astra_level'] for d in draws]
        self.robot.set_dofs_position(np.stack([s['q'] for s in states]), self.indices,
                                     envs_idx=lanes, zero_velocity=True)
        # Without zero_velocity the object keeps the motion it had when the previous episode ended.
        self.cube.set_pos(np.stack([s['object_position'] for s in states]), envs_idx=lanes, zero_velocity=True)
        self.cube.set_quat(np.stack([s['object_quaternion'] for s in states]), envs_idx=lanes)
        self._set_materials(lanes, np.array([d[1]['mass_kg'] for d in draws]),
                            np.array([d[1]['friction'] for d in draws]))
        lo, hi = self.joint_friction_Nm
        if hi > 0:
            loss = self.rng.uniform(lo, hi, (len(lanes), 1))
            self.joint_friction[lanes] = loss[:, 0]
            self.robot.set_dofs_frictionloss(np.repeat(loss, 12, axis=1), self.indices[6:], envs_idx=lanes)
        self.residual[lanes] = 0.
        t = self.lane_tick/self.rates.physics_hz
        gains = self._base_gain(t)
        self._set_gains(lanes, gains[lanes])
        self.command[lanes] = self._targets(t)[lanes]
        self.robot.control_dofs_position(self.command[lanes], self.indices, envs_idx=lanes)
        self.breaks.reset(lanes)
        self.slip.reset(lanes)
        for name in ('returns', 'max_clearance', 'hold_duration', 'load_integral', 'peak_all_load',
                     'peak_region', 'hold_peak_region', 'residual_abs_integral', 'elapsed'):
            getattr(self, name)[lanes] = 0.
        self.hold_min[lanes] = np.inf
        self.hold_support_ok[lanes] = True
        self.terms[lanes] = 0.
        self.history[lanes] = 0.
        state = self._state()
        self.proprio[lanes] = state[lanes, None]
        self.episode_counter += len(lanes)

    def reset(self):
        n = self.num_envs
        self.lane_base = np.zeros(n, dtype=int)
        self.lane_start, self.lane_end, self.lane_tick = (np.zeros(n, dtype=int) for _ in range(3))
        self.f_break = np.full(n, np.inf)
        self.astra_level = np.full(n, 4, dtype=int)
        self.joint_friction = np.zeros(n)
        self.episode = [None]*n
        self.residual = np.zeros((n, len(self.action_joints)))
        self.command = np.zeros((n, 18))
        self.breaks = BreakMonitor(n, self.config.break_consecutive_steps)
        self.slip = SlipTracker(n)
        for name in ('returns', 'max_clearance', 'hold_duration', 'load_integral', 'peak_all_load',
                     'peak_region', 'hold_peak_region', 'residual_abs_integral', 'elapsed', 'hold_min'):
            setattr(self, name, np.zeros(n))
        self.hold_support_ok = np.ones(n, dtype=bool)
        self.terms = np.zeros((n, len(REWARD_TERMS)))
        self.history = np.zeros((n, 12, 5, 120, 3), np.float32)
        self.proprio = np.zeros((n, 12, self.state_dim), np.float32)
        self._reset_lanes(np.arange(n))
        self._pending_action = None
        return self._obs()

    def _state(self):
        n = self.num_envs
        q = cpu(self.robot.get_dofs_position(self.indices)).reshape(n, 18)
        qd = cpu(self.robot.get_dofs_velocity(self.indices)).reshape(n, 18)
        t = self.lane_tick/self.rates.physics_hz
        nominal, future = self._targets(t), self._targets(t+self.rates.action_dt)
        phase = np.array([[self.bases[k].trajectory.phase(ti) == p for p in PHASES]
                          for k, ti in zip(self.lane_base, t)], np.float32)
        progress = (self.lane_tick-self.lane_start)/np.maximum(self.lane_end-self.lane_start, 1)
        level = np.eye(4, dtype=np.float32)[self.astra_level-1]*self.use_level
        state = np.concatenate([(q-self.mid)/self.half, np.clip(qd/3, -10, 10), (self.command-q)/.2,
            (future-nominal)/.1, (self.gains-1)[:, None], phase, progress[:, None], level,
            np.repeat(self.finger_mask[None], n, axis=0), self.residual/self.config.residual_bound_rad],
            axis=1).astype(np.float32)
        if not np.isfinite(state).all():
            raise FloatingPointError('Nonfinite joint state')
        return state

    def _obs(self):
        return {'tactile': self.history.copy(), 'state': self.proprio.copy()}

    # ------------------------------------------------------------------ step
    def step_async(self, actions):
        a = np.asarray(actions, dtype=np.float32)
        if a.shape != (self.num_envs, len(self.action_joints)) or not np.isfinite(a).all():
            raise ValueError('Expected one finite residual action per lane')
        self._pending_action = a.copy()

    def step_wait(self):
        if self._pending_action is None:
            raise RuntimeError('step_async must be called first')
        c, n = self.config, self.num_envs
        old = self.residual.copy()
        self.residual = (np.zeros_like(old) if self.fixed_gain is not None
                         else limit_residual(self._pending_action, old, c)*self.active)
        t = self.lane_tick/self.rates.physics_hz
        self._set_gains(np.arange(n), self._base_gain(t))
        command = self._targets(t)
        command[:, self.controlled] += self.residual
        self.command = np.clip(command, self.lower, self.upper)
        self.robot.control_dofs_position(self.command, self.indices)
        reward = np.zeros(n)
        change = c.residual_change_weight*np.square((self.residual-old)/c.residual_rate_rad).sum(-1)
        reward -= change
        self.terms[:, 4] -= change
        failed = np.zeros(n, dtype=bool)
        reason = np.array(['']*n, dtype=object)
        for sub in range(self.rates.action_stride):
            self.scene.step()
            self.lane_tick += 1
            region = self.region_normal_force()
            self.peak_region = np.maximum(self.peak_region, region.max(-1))
            broken = self.breaks.update(region, self.f_break, ~failed)
            reason[broken] = 'break'
            failed |= broken
            if (sub+1) % self.rates.tactile_stride:
                continue
            self._sample_tactile()
            m = self._metrics()
            dt = 1/self.rates.tactile_hz
            now = self.lane_tick/self.rates.physics_hz
            hold_start = np.array([self.bases[k].hold_start_s for k in self.lane_base])
            lift_start = np.array([self.bases[k].trajectory.phase_starts['test_lift'] for k in self.lane_base])
            end = np.array([self.bases[k].trajectory.duration_s for k in self.lane_base])
            hold_dt = interval_overlap(now-dt, now, hold_start, end)
            in_hold = hold_dt > 0
            any_support = (m['region_normal_N'][:, :6] > c.support_threshold_N).any(-1)
            # Slip counts from the first lifting command: sliding while lifting means too little grip.
            slip_m = self.slip.update(m['object_in_palm_m'], (now >= lift_start) & ~failed)
            terms = np.stack([
                c.progress_weight*np.clip(m['clearance_m']/.07, 0, 1)*any_support*dt,
                -c.force_weight*m['all_robot_load_N']*dt,
                -break_soft_rate(m['region_normal_N'], self.f_break, c)*dt,
                -c.slip_weight_per_mm*slip_m*1000.], axis=1)*~failed[:, None]
            reward += terms.sum(-1)
            self.terms[:, :4] += terms
            fault = (m['table_normal_N'] > c.table_fault_N) | m['self_contact']
            slipped = self.slip.distance > c.slip_fail_m
            for flag, name in ((fault, 'collision'), (slipped, 'slip')):
                new = flag & ~failed
                reason[new] = name
                failed |= new
            self.max_clearance = np.maximum(self.max_clearance, m['clearance_m'])
            self.load_integral += m['all_robot_load_N']*dt
            self.peak_all_load = np.maximum(self.peak_all_load, m['all_robot_load_N'])
            self.hold_duration += hold_dt
            self.hold_min = np.where(in_hold, np.minimum(self.hold_min, m['clearance_m']), self.hold_min)
            self.hold_support_ok &= ~in_hold | opposing_support(m['region_normal_N'], c.support_threshold_N)
            self.hold_peak_region = np.where(in_hold, np.maximum(self.hold_peak_region, m['region_normal_N'].max(-1)),
                                             self.hold_peak_region)
            state = self._state()
            self.proprio[:, :-1] = self.proprio[:, 1:]
            self.proprio[:, -1] = state
        self.elapsed += self.rates.action_dt
        self.residual_abs_integral += np.abs(self.residual).mean(-1)*self.rates.action_dt
        ended = self.lane_tick >= self.lane_end
        done = failed | ended
        success = (ended & ~failed & hold_success(self.hold_duration, self.hold_min)
                   & self.hold_support_ok)
        terminal = np.where(success, c.success_bonus, -c.failure_penalty)*done
        reward += terminal
        self.terms[:, 5] += terminal
        self.returns += reward
        obs = self._obs()
        infos = [{} for _ in range(n)]
        lanes = np.flatnonzero(done)
        for i in lanes:
            if not reason[i]:
                reason[i] = 'success' if success[i] else ('no_opposing_support' if self.hold_duration[i] >= 3-1e-6
                    and self.hold_min[i] >= .05 else 'hold_failed')
            infos[i] = dict(episode_metrics=self._episode_metrics(i, bool(success[i]), reason[i]),
                episode={'r': float(self.returns[i]), 'l': int(round(self.elapsed[i]*15))},
                terminal_observation={k: v[i].copy() for k, v in obs.items()}, **{'TimeLimit.truncated': False})
        if len(lanes):
            self._reset_lanes(lanes)
            obs = self._obs()
        self._pending_action = None
        return obs, reward.astype(np.float32), done, infos

    def _episode_metrics(self, i, success, reason):
        base, spec = self.bases[self.lane_base[i]], self.episode[i]
        duration = max(self.elapsed[i], 1e-9)
        return dict(success=success, task_success=success, strict_success=success, termination_reason=reason,
            object_id=self.object_id, base_id=base.base_id, base_label=base.label,
            edge_placement=bool(base.metadata.get('edge')), unseen_object=bool(base.metadata.get('unseen_object')),
            mass_kg=spec['mass_kg'], friction=spec['friction'], level=spec['level'],
            astra_level=spec['astra_level'], f_break_N=None if np.isinf(spec['f_break_N']) else spec['f_break_N'],
            object_offset_m=spec.get('object_offset_m', 0.), object_turn_deg=spec.get('object_turn_deg', 0.),
            f_min_N=spec['f_min_N'], window_ratio=None if np.isinf(spec['window_ratio']) else spec['window_ratio'],
            narrow_window=spec['narrow_window'], spec_index=spec['spec_index'], discard=spec['discard'],
            hand_shape=self.hand_shape, joint_friction_Nm=float(self.joint_friction[i]), fixed_gain=self.fixed_gain,
            max_clearance_m=float(self.max_clearance[i]), hold_duration_s=float(self.hold_duration[i]),
            hold_min_clearance_m=float(self.hold_min[i]) if np.isfinite(self.hold_min[i]) else None,
            hold_opposing_support=bool(self.hold_support_ok[i]),
            slip_distance_m=float(self.slip.distance[i]),
            peak_region_normal_N=float(self.peak_region[i]), hold_peak_region_normal_N=float(self.hold_peak_region[i]),
            peak_to_break_ratio=None if np.isinf(spec['f_break_N']) else float(self.peak_region[i]/spec['f_break_N']),
            mean_all_robot_contact_force_N=float(self.load_integral[i]/duration),
            peak_all_robot_contact_force_N=float(self.peak_all_load[i]),
            mean_abs_residual_rad=float(self.residual_abs_integral[i]/duration),
            reward_terms=dict(zip(REWARD_TERMS, self.terms[i].tolist())), **{'return': float(self.returns[i])},
            sim_duration_s=float(self.elapsed[i]))

    # -------------------------------------------------------------- protocol
    def _write_protocol(self):
        protocol = dict(variant='v1_takeover_residual', object_id=self.object_id,
            object_spec=self.object_spec.protocol(), physics_hz=300, tactile_hz=60, action_hz=15,
            config=asdict(self.config), sampler=asdict(self.sampler), use_level=self.use_level,
            joint_friction_Nm=list(self.joint_friction_Nm), tactile_noise_N=self.tactile_noise_N,
            fixed_gain=self.fixed_gain, tactile_model=current_tactile_model(),
            solver_profile=self.solver_profile, solver_readback=self.solver_readback,
            actor_state_fields=list(STATE_FIELDS), state_dim=self.state_dim, privileged_actor_input=False,
            hand_shape=self.hand_shape, hand_shape_fingers=list(HAND_SHAPES[self.hand_shape]),
            action_joints=self.action_joints.tolist(), active_action_joints=self.active.tolist(),
            action='absolute residual on hand-local action_joints; [-1,1] -> +-bound, rate limited per 15 Hz '
                   'action; joints outside this hand shape stay at the base command',
            grip_gain='recorded grip gain scales kp (kv by sqrt) of the hand-shape finger joints',
            takeover="first 'close' command floored to the 15 Hz grid",
            reset='robot joints and object pose set to the replayed takeover state; material, fragility and joint '
                  'friction resampled per episode; no contact exists at takeover',
            base_draw='uniform over bases, R capped at sampler.r_max_fraction of draws; a sampler.s_offset_fraction '
                      'share of S episodes starts from a shifted-object variant (prefix replayed, no contact)',
            base_draw_probability=dict(zip([b.base_id for b in self.bases], self.base_p.tolist())),
            s_offset_variants={b.base_id: [dict(offset_m=v['offset_m'].tolist(), turn_deg=v['turn_deg'])
                               for v in vs] for b, vs in zip(self.bases, self.variants) if vs},
            break_rule='normal resultant per region (5 fingers, palm, arm) above F_break for consecutive physics steps',
            slip_rule='largest object displacement in the palm frame since the first test_lift command',
            success='final 3 s: object lowest point >=5 cm at every 60 Hz sample, thumb or palm plus another '
                    'finger in contact (normal >= support threshold), no break, no collision, slip within limit',
            link_regions=self.link_regions, bases=[b.summary() for b in self.bases],
            excluded_bases=self.excluded_bases, prefix_wall_s=self.prefix_wall_s,
            takeover_states=[dict(base_id=b.base_id, prefix_contact_N=s['prefix_contact_N'],
                object_position_m=s['object_position'].tolist()) for b, s in zip(self.bases, self.takeover_states)],
            seed=self.seed_value, model_path=str(self.model_path))
        (self.output_dir/'environment_protocol.json').write_text(json.dumps(protocol, indent=2))

    # ---------------------------------------------------------------- VecEnv
    def close(self):
        if not self._closed:
            gs.destroy()
            self._closed = True

    def get_attr(self, attr_name, indices=None):
        return [getattr(self, attr_name) for _ in self._get_indices(indices)]

    def set_attr(self, attr_name, value, indices=None):
        if indices is not None:
            raise NotImplementedError('Only shared attributes supported')
        setattr(self, attr_name, value)

    def env_method(self, method_name, *args, indices=None, **kwargs):
        raise NotImplementedError('Use batched environment methods directly')

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False for _ in self._get_indices(indices)]


class V1ComposedVecEnv(VecEnv):
    """Concatenate one scene per object; each scene resets its own lanes."""
    def __init__(self, envs):
        self.envs = list(envs)
        self.offsets = np.r_[0, np.cumsum([e.num_envs for e in self.envs])]
        for e in self.envs:
            if e.observation_space != self.envs[0].observation_space or e.action_space != self.envs[0].action_space:
                raise ValueError('All scenes must share observation and action spaces')
        self.render_mode = None
        self.config = self.envs[0].config
        self._closed = False
        super().__init__(int(self.offsets[-1]), self.envs[0].observation_space, self.envs[0].action_space)

    def reset(self):
        return self._concat([e.reset() for e in self.envs])

    @staticmethod
    def _concat(observations):
        return {k: np.concatenate([o[k] for o in observations], axis=0) for k in observations[0]}

    def step_async(self, actions):
        actions = np.asarray(actions)
        for i, e in enumerate(self.envs):
            e.step_async(actions[self.offsets[i]:self.offsets[i+1]])

    def step_wait(self):
        out = [e.step_wait() for e in self.envs]
        return (self._concat([o[0] for o in out]), np.concatenate([o[1] for o in out]),
                np.concatenate([o[2] for o in out]), [info for o in out for info in o[3]])

    def close(self):
        if not self._closed:
            self.envs[0].close()
            for e in self.envs:
                e._closed = True
            self._closed = True

    def get_attr(self, attr_name, indices=None):
        return [getattr(self.envs[0], attr_name) for _ in self._get_indices(indices)]

    def set_attr(self, attr_name, value, indices=None):
        for e in self.envs:
            e.set_attr(attr_name, value, indices)

    def env_method(self, method_name, *args, indices=None, **kwargs):
        raise NotImplementedError('Call batched methods on child scenes')

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False for _ in self._get_indices(indices)]
