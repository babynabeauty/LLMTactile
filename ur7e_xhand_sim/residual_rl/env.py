"""Batched rigid Genesis environment. No hardware, web bridge, or privileged actor input."""
import json
import os
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as ET
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
os.environ.setdefault('OMP_NUM_THREADS', '4')
import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv
import genesis as gs

from .core import (ROOT, DEMO, RateConfig, Trajectory, gains_from_action,
                   rate_limit_gains, hold_success, reward_rate, interval_overlap)
from .tactile_provenance import current_tactile_model
sys.path.insert(0, str(ROOT / 'ur7e_xhand_sim'))
from build_assets import build, ARM_JOINTS, HAND_JOINTS


DEFAULT_CUBE_SIZE_M = .045
TABLE_TOP_Z_M = .2


def cpu(x):
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def cube_geometry(cube_size_m=DEFAULT_CUBE_SIZE_M, initial_position=None):
    """One shared side length per scene; keep the original training pose intact."""
    size = np.asarray(cube_size_m)
    if size.ndim != 0 or size.dtype.kind not in 'iuf' or not np.isfinite(size) or size <= 0:
        raise ValueError('cube_size_m must be one finite positive scalar side length')
    size = float(size)
    position = np.asarray(
        [-.60, -.1333, .225] if initial_position is None else initial_position, dtype=float).copy()
    if position.shape != (3,) or not np.isfinite(position).all():
        raise ValueError('Cube initial position must contain three finite coordinates')
    # The original 45 mm scene intentionally retains its 2.5 mm initial gap.
    # New sizes are placed with their bottom on the same physical tabletop.
    if size != DEFAULT_CUBE_SIZE_M:
        position[2] = TABLE_TOP_Z_M + size / 2
    return size, position


def cube_vertical_extent(quaternion_wxyz, side_m):
    """World-Z half extent of an oriented cube, including corners after rotation."""
    w, x, y, z = quaternion_wxyz.unbind(-1)
    return (side_m / 2) * (torch.abs(2*(x*z-w*y)) + torch.abs(2*(y*z+w*x))
                           + torch.abs(1-2*(x*x+y*y)))


def trajectory_hold_window(trajectory, terminal_hold_duration_s=None):
    """Validate recorded phases without relabelling or removing any command.

    The legacy default requires exactly three recorded hold seconds. The
    explicit new protocol permits a longer uninterrupted recorded hold and
    evaluates only its final three seconds, including all inserted recovery
    commands in the nominal trajectory.
    """
    required = ('preload','test_lift','lift','hold')
    if any(phase not in trajectory.phase_starts for phase in required):
        raise ValueError('Residual trajectory requires preload/test_lift/lift/hold phases')
    starts = [trajectory.phase_starts[phase] for phase in required]
    if np.any(np.diff(starts) <= 0):
        raise ValueError('This benchmark requires ordered phases and a final 3-second hold')
    recorded_start = starts[-1]
    recorded_duration = trajectory.duration_s-recorded_start
    if terminal_hold_duration_s is None:
        if not np.isclose(recorded_duration,3.,rtol=0,atol=1e-8):
            raise ValueError('This benchmark requires ordered phases and a final 3-second hold')
        evaluation_start = recorded_start
    else:
        value = np.asarray(terminal_hold_duration_s)
        if (value.ndim != 0 or value.dtype.kind not in 'iuf' or not np.isfinite(value)
                or not np.isclose(value,3.,rtol=0,atol=1e-12)):
            raise ValueError('The terminal hold success criterion is fixed at exactly 3.0 seconds')
        if recorded_duration < 3.-1e-8:
            raise ValueError('Recorded final hold is shorter than the required 3 seconds')
        evaluation_start = trajectory.duration_s-3.
    if np.any(trajectory.phases[trajectory.times >= recorded_start] != 'hold'):
        raise ValueError('Final recorded hold must be uninterrupted')
    # Also check the command whose interval straddles the evaluation boundary.
    index = max(0,int(np.searchsorted(trajectory.times,evaluation_start+1e-8,side='right'))-1)
    if np.any(trajectory.phases[index:] != 'hold'):
        raise ValueError('Every command overlapping the final evaluation window must be hold')
    return float(recorded_start),float(evaluation_start)


class ResidualVecEnv(VecEnv):
    """Synchronous episodes from a physically replayed prefix; two gain actions.

    Each lane has a fixed, seeded material sampled BEFORE replaying its prefix.
    Reset restores that same material and physical prefix state. There is no
    teleportation, attachment or object pose correction during an episode.
    This pilot intentionally does not claim broad initial-state diversity.
    """
    def __init__(self, n_envs=8, seed=0, output_dir=None,
                 mass_range_kg=(.1, .2), friction_range=(.8, .8), mode='train',
                 tactile_noise_std=0., render=False, material_conditions=None,
                 trajectory_path=None, cube_initial_position=None, audit_prefix=False,
                 cube_size_m=DEFAULT_CUBE_SIZE_M, terminal_hold_duration_s=None,
                 replay_source_prefix_gains=False, noslip_iterations=0, noslip_tolerance=1e-6):
        from .solver_profile import SCHEMA, normalize_solver_profile
        self.solver_profile=normalize_solver_profile({'schema':SCHEMA,
            'noslip_iterations':noslip_iterations,'noslip_tolerance':noslip_tolerance})
        self.render_mode = None
        self.mode = mode
        self.rates = RateConfig()
        self.trajectory = Trajectory(trajectory_path or DEMO / 'control.jsonl')
        self.replay_source_prefix_gains=bool(replay_source_prefix_gains)
        self.prefix_gain_events=[]
        self.cube_size_m, self.cube_initial_position = cube_geometry(cube_size_m, cube_initial_position)
        self._configure_object_geometry()
        if self.trajectory.targets.shape[1] != 18 or abs(self.trajectory.times[0]) > 1e-8:
            raise ValueError('Residual trajectory requires 18 ordered joints and time zero start')
        self.recorded_hold_start, self.hold_start = trajectory_hold_window(
            self.trajectory,terminal_hold_duration_s)
        self.terminal_hold_duration_s = terminal_hold_duration_s
        self.rng = np.random.default_rng(seed)
        self.seed_value = seed
        self.tactile_noise_std = tactile_noise_std
        self.audit_prefix = bool(audit_prefix)
        self.prefix_audit_summary = {'enabled':self.audit_prefix}
        self.output_dir = Path(output_dir or ROOT / 'outputs/residual_rl_runtime')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.start_tick = int(self.trajectory.phase_starts['preload'] * 15) * 20
        self.end_tick = int(np.ceil(self.trajectory.duration_s * 15)) * 20
        if self.replay_source_prefix_gains:
            prefix_values=self.trajectory.command_grip_gains[self.trajectory.times<self.start_tick/300]
            lower=getattr(getattr(self,'config',None),'gain_min',1.)
            upper=getattr(getattr(self,'config',None),'gain_max',2.)
            if ((prefix_values<lower)|(prefix_values>upper)).any():
                raise ValueError('Source prefix gain is outside this residual controller absolute gain range')
        self.mass = self.rng.uniform(*mass_range_kg, size=n_envs).astype(np.float32)
        self.mu = self.rng.uniform(*friction_range, size=n_envs).astype(np.float32)
        if material_conditions is not None:
            conditions = np.asarray(material_conditions, dtype=np.float32)
            if conditions.shape != (n_envs, 2) or not np.isfinite(conditions).all():
                raise ValueError('material_conditions must contain one finite (kg, friction) pair per lane')
            self.mass, self.mu = conditions[:, 0].copy(), conditions[:, 1].copy()
        if (self.mass <= 0).any() or (self.mu < .01).any() or (self.mu > 5).any():
            raise ValueError('Invalid material range')
        self.cube_build_density_kg_m3 = (
            .2 if self.cube_size_m == DEFAULT_CUBE_SIZE_M else float(self.mass[0])) / getattr(
                self, 'object_volume_m3', self.cube_size_m**3)
        self.state_fields = ['q_normalized[18]', 'qd_div_3[18]',
            'nominal_minus_q_div_.2[18]', 'next_nominal_minus_nominal_div_.1[18]',
            'applied_gain_minus_1[2]', 'phase_one_hot[4]', 'local_progress[1]']
        obs_space = spaces.Dict({
            'tactile': spaces.Box(-np.inf, np.inf, (12, 5, 120, 3), np.float32),
            'state': spaces.Box(-np.inf, np.inf, (79,), np.float32)})
        super().__init__(n_envs, obs_space, spaces.Box(-1., 1., (2,), np.float32))
        model, self.layouts = build()
        if not gs._initialized:
            gs.init(backend=gs.gpu, precision='32', seed=seed, logging_level='warning',
                    use_deterministic_algorithms=True)
        if not gs.use_deterministic_algorithms:
            raise RuntimeError('Residual training requires deterministic Genesis solver selection')
        self.device = gs.device
        self.scene = gs.Scene(sim_options=gs.options.SimOptions(dt=self.rates.physics_dt, substeps=1),
            rigid_options=gs.options.RigidOptions(enable_self_collision=True, enable_joint_limit=True,
                enable_neutral_collision=True, batch_links_info=True, batch_dofs_info=True,
                noslip_iterations=self.solver_profile['noslip_iterations'],
                noslip_tolerance=self.solver_profile['noslip_tolerance']),
            show_viewer=False)
        self.scene.add_entity(gs.morphs.Plane())
        self.table = self.scene.add_entity(gs.morphs.Box(size=(1.3,1.,.04),pos=(-.4,0,.18),fixed=True),
                                           material=gs.materials.Rigid(friction=.8))
        self.robot = self.scene.add_entity(gs.morphs.URDF(file=str(model),fixed=True,pos=(0,0,.2),
            merge_fixed_links=True,links_to_keep=['right_hand_link','tool0']),
            material=gs.materials.Rigid(friction=.8))
        self.cube = self._create_grasp_object()
        self.sensors = []
        for layout in self.layouts:
            link = self.robot.get_link(layout['link_name'])
            self.sensors.append(self.scene.add_sensor(gs.sensors.KinematicTaxel(
                entity_idx=self.robot.idx,link_idx_local=link.idx_local,
                probe_local_pos=layout['positions_link_m'],probe_radius=.0015,history_length=0,
                normal_stiffness=500.,normal_damping=1.,normal_exponent=1.2,
                shear_scalar=2.,twist_scalar=0.,draw_debug=False)))
        self.camera = None
        if render:
            from cameras import load_camera_views, camera_kwargs
            views, _ = load_camera_views()
            self.camera = self.scene.add_camera(**camera_kwargs(views['scene_front']))
        self.scene.build(n_envs=n_envs)
        self.cube_initial_position_readback = cpu(self.cube.get_pos()).reshape(n_envs,3)
        np.testing.assert_allclose(self.cube_initial_position_readback,
            np.broadcast_to(self.cube_initial_position,(n_envs,3)),rtol=0,atol=1e-7)
        self.cube_size_readback_m = np.ptp(cpu(self.cube.base_link.geoms[0].init_verts),axis=0)
        np.testing.assert_allclose(self.cube_size_readback_m,
            getattr(self, 'object_bounds_size_m', self.cube_size_m),rtol=0,atol=1e-7)
        self.names = ARM_JOINTS + HAND_JOINTS
        self.indices = [self.robot.get_joint(n).dofs_idx_local[0] for n in self.names]
        self.finger_links = [self.robot.get_link(l['link_name']).idx for l in self.layouts]
        self.solver = self.scene.rigid_solver
        from .solver_profile import read_solver_readback, validate_solver_readback
        self.solver_readback=validate_solver_readback(self.solver_profile,read_solver_readback(self.solver))
        lo, hi = self.robot.get_dofs_limit(self.indices)
        self.lower = cpu(lo).reshape(-1,18)[0]
        self.upper = cpu(hi).reshape(-1,18)[0]
        if ((self.trajectory.targets < self.lower-1e-6) | (self.trajectory.targets > self.upper+1e-6)).any():
            raise ValueError('Recorded command exceeds a robot joint limit')
        self.mid = (self.lower+self.upper)/2
        self.half = (self.upper-self.lower)/2
        self.base_kp = np.asarray([4500,4500,3500,2000,2000,2000]+[3.]*12, np.float32)
        self.base_kv = np.asarray([450,450,350,200,200,200]+[.08]*12, np.float32)
        xml = ET.parse(model).getroot()
        self.effort = np.array([float(xml.find(f"joint[@name='{n}']/limit").get('effort'))
                                for n in self.names], np.float32)
        self.home = np.r_[np.deg2rad([0,-89.95,89.96,-89.96,-89.95,-44.98]),np.zeros(12)]
        self.history = np.zeros((n_envs,12,5,120,3), np.float32)
        self.gains = np.ones((n_envs,2), np.float32)
        self._apply_materials()
        self._apply_gains()
        self.robot.set_dofs_force_range(-self.effort,self.effort,self.indices)
        self.robot.set_dofs_position(self.home,self.indices,zero_velocity=True)
        self.robot.control_dofs_position(self.home,self.indices)
        started = time.monotonic()
        prefix_metrics = []
        print(f'PREFIX replay 0 -> {self.start_tick/300:.6f}s, lanes={n_envs}', flush=True)
        for tick in range(self.start_tick):
            self._source_prefix_gain_tick(tick)
            if tick % self.rates.action_stride == 0:
                self.robot.control_dofs_position(self.trajectory.target(tick/300), self.indices)
            self.scene.step()
            if (tick+1) % self.rates.tactile_stride == 0:
                self._sample_tactile()
                if self.audit_prefix:
                    clearance,load,slip,fault = self._metrics()
                    prefix_metrics.append({'sample_time_s':(tick+1)/300,
                        'clearance_m':clearance.copy(),'contact_load_N':load.copy(),
                        'slip_speed_m_s':slip.copy(),'fault':fault.copy(),
                        'applied_gains':self.gains.copy(),
                        **{k:v.copy() for k,v in self.last_contact_diagnostics.items()}})
            if (tick+1) % 1500 == 0:
                print(f'PREFIX {tick+1}/{self.start_tick}, wall={time.monotonic()-started:.1f}s',flush=True)
        # SimState/get_state is intentionally too small for exact continuation:
        # it omits solver warm-start and derived state. Keep the full checkpoint
        # in this SAME scene (constructing a new scene would omit taxel sensors).
        self.prefix_gains=self.gains.copy()
        if self.replay_source_prefix_gains:
            self.prefix_readbacks=self._prefix_physics_readbacks()
            self._assert_prefix_gain_readbacks(self.prefix_readbacks)
        self.prefix_checkpoint = self.scene.__getstate__()
        if self.audit_prefix:
            prefix_trace = {k:np.asarray([r[k] for r in prefix_metrics]) for k in prefix_metrics[0]}
            np.savez_compressed(self.output_dir/'prefix_physics_trace.npz',**prefix_trace)
            self.prefix_audit_summary.update(samples=len(prefix_metrics),
                collision_per_lane=prefix_trace['fault'].any(axis=0).tolist(),
                path='prefix_physics_trace.npz',
                interpretation='Read-only pre-policy physical audit; reported separately from unchanged residual task labels')
        self.prefix_history = self.history.copy()
        self.episode_counter = 0
        self._closed = False
        self._pending_action = None
        self._write_protocol(model, started)
        self.reset()

    def _configure_object_geometry(self):
        """Subclass hook; the original cube path and its initial pose stay unchanged."""

    def _create_grasp_object(self):
        return self.scene.add_entity(
            gs.morphs.Box(size=(self.cube_size_m,)*3,pos=tuple(self.cube_initial_position)),
            material=gs.materials.Rigid(rho=self.cube_build_density_kg_m3,friction=.8),
            surface=gs.surfaces.Default(color=(.95,.36,.10,1)))

    def _apply_materials(self):
        self.cube.set_mass(self.mass[:,None])
        self.cube.set_friction_ratio((self.mu/.8)[:,None])
        self.robot.set_friction_ratio(np.broadcast_to((self.mu/.8)[:,None],
                                                      (self.num_envs,self.robot.n_links)).copy())
        self.mass_readback_kg_per_lane = cpu(self.cube.get_mass()).reshape(-1).copy()
        np.testing.assert_allclose(self.mass_readback_kg_per_lane,self.mass,rtol=1e-5)
        inertia = cpu(self.cube.get_links_inertia()).reshape(self.num_envs,1,3,3)[:,0]
        unit_inertia = getattr(self, 'object_inertia_per_kg', self.cube_size_m**2/6*np.eye(3))
        expected = self.mass[:,None,None] * unit_inertia[None]
        np.testing.assert_allclose(inertia,expected,rtol=1e-4,atol=1e-9)
        self.inertia_readback_kg_m2_per_lane = inertia.copy()
        self.density_readback_kg_m3_per_lane = self.mass_readback_kg_per_lane / getattr(
            self, 'object_volume_m3', np.prod(self.cube_size_readback_m))

    def _apply_gains(self):
        kp = np.broadcast_to(self.base_kp,(self.num_envs,18)).copy()
        kv = np.broadcast_to(self.base_kv,(self.num_envs,18)).copy()
        for j, sl in enumerate((slice(6,9),slice(9,12))):
            kp[:,sl] *= self.gains[:,j,None]
            kv[:,sl] *= np.sqrt(self.gains[:,j,None])
        self.robot.set_dofs_kp(kp,self.indices)
        self.robot.set_dofs_kv(kv,self.indices)

    def _source_prefix_gain_tick(self,tick):
        if not self.replay_source_prefix_gains:return
        requested=float(self.trajectory.grip_gain_at(tick/self.rates.physics_hz))
        changed=not np.all(self.gains==requested)
        if changed:
            self.gains[:]=requested
            self._apply_gains()
        if changed or tick==0:
            readbacks=self._prefix_physics_readbacks()
            self._assert_prefix_gain_readbacks(readbacks)
            self.prefix_gain_events.append(dict(physics_tick=int(tick),
                time_s=tick/self.rates.physics_hz,common_absolute_gain=requested,
                kp_readback=readbacks['kp'].tolist(),kv_readback=readbacks['kv'].tolist(),
                mass_readback_kg=readbacks['mass'].tolist()))

    def _prefix_physics_readbacks(self):
        lower,upper=self.robot.get_dofs_force_range(self.indices)
        return dict(kp=cpu(self.robot.get_dofs_kp(self.indices)).reshape(-1,18).copy(),
            kv=cpu(self.robot.get_dofs_kv(self.indices)).reshape(-1,18).copy(),
            effort_lower=cpu(lower).reshape(-1,18).copy(),
            effort_upper=cpu(upper).reshape(-1,18).copy(),
            mass=cpu(self.cube.get_mass()).reshape(-1).copy(),
            inertia=cpu(self.cube.get_links_inertia()).reshape(self.num_envs,1,3,3)[:,0].copy())

    def _assert_prefix_gain_readbacks(self,readbacks):
        if not all(np.isfinite(value).all() for value in readbacks.values()):
            raise FloatingPointError('Nonfinite prefix physical readback')
        kp=np.broadcast_to(self.base_kp,(self.num_envs,18)).copy()
        kv=np.broadcast_to(self.base_kv,(self.num_envs,18)).copy()
        for j,sl in enumerate((slice(6,9),slice(9,12))):
            kp[:,sl]*=self.gains[:,j,None]
            kv[:,sl]*=np.sqrt(self.gains[:,j,None])
        for key,expected in (('kp',kp),('kv',kv),('mass',self.mass)):
            np.testing.assert_allclose(readbacks[key],expected,rtol=1e-5,atol=1e-7,
                err_msg='Prefix '+key+' differs from actual requested state')
        np.testing.assert_allclose(readbacks['inertia'],self.inertia_readback_kg_m2_per_lane,rtol=1e-5,atol=1e-9)
        for key,sign in (('effort_lower',-1),('effort_upper',1)):
            np.testing.assert_allclose(readbacks[key],np.broadcast_to(sign*self.effort,readbacks[key].shape),rtol=1e-5,atol=1e-7)

    def _sample_tactile(self):
        value = cpu(torch.stack([s.read().force for s in self.sensors],dim=1))
        if value.shape != (self.num_envs,5,120,3):
            raise RuntimeError(f'Unexpected tactile dimensions: {value.shape}')
        if not np.isfinite(value).all(): raise FloatingPointError('Nonfinite tactile')
        self.history[:,:-1] = self.history[:,1:]
        self.history[:,-1] = value

    def _metrics(self):
        """Physical contact labels at 60 Hz; never passed to the actor."""
        c = self.robot.get_contacts(is_padded=True)
        a,b,valid = c['link_a'].long(),c['link_b'].long(),c['valid_mask']
        is_robot_a = (a >= self.robot.link_start) & (a < self.robot.link_end)
        is_robot_b = (b >= self.robot.link_start) & (b < self.robot.link_end)
        bad = valid & ((is_robot_a & is_robot_b) | (a == self.table.base_link.idx) | (b == self.table.base_link.idx))
        bad &= torch.linalg.vector_norm(c['force_a'],dim=-1) > .01
        cubeid = self.cube.base_link.idx
        pair = valid & ((a == cubeid) | (b == cubeid))
        other = torch.where(a == cubeid,b,a)
        p,normal = c['position'],c['normal']
        force = torch.where(pair,torch.linalg.vector_norm(c['force_a'],dim=-1),0.)
        pos = self.solver.get_links_pos(relative=False)
        vel = self.solver.get_links_vel(relative=False)
        ang = self.solver.get_links_ang()
        batch = torch.arange(self.num_envs,device=self.device)[:,None]
        other_safe = other.clamp(0,pos.shape[1]-1)
        vc = vel[:,cubeid,None,:]+torch.cross(ang[:,cubeid,None,:].expand_as(p),p-pos[:,cubeid,None,:],dim=-1)
        vh = vel[batch,other_safe]+torch.cross(ang[batch,other_safe],p-pos[batch,other_safe],dim=-1)
        rel = vc-vh
        normal = normal / torch.linalg.vector_norm(normal,dim=-1,keepdim=True).clamp_min(1e-9)
        tangent = rel-(rel*normal).sum(-1,keepdim=True)*normal
        speed = torch.linalg.vector_norm(tangent,dim=-1)
        per_finger = torch.stack([torch.where(pair & (other==link),force,0.).sum(-1)
                                  for link in self.finger_links],dim=-1)
        sensor_pair = pair & torch.stack([other==link for link in self.finger_links]).any(0)
        weight = torch.where(sensor_pair,force,0.)
        slip = (torch.where(sensor_pair,speed,0.)*weight).sum(-1)/weight.sum(-1).clamp_min(1e-9)
        slip = torch.where(weight.sum(-1)>.05,slip,0.)
        cp = self.cube.get_pos()
        q = self.cube.get_quat()
        extent_fn = getattr(self, 'object_vertical_extent', None)
        extent = cube_vertical_extent(q,self.cube_size_m) if extent_fn is None else extent_fn(q)
        clearance = cp[:,2]-extent-TABLE_TOP_Z_M
        # Audit the actual collider coefficient on every valid hand/cube contact.
        st = self.solver.collider.collider_state
        from genesis.utils.misc import qd_to_torch
        friction = qd_to_torch(st.contact_data.friction,transpose=True)
        order = qd_to_torch(st.contact_sort_idx,transpose=True).long()
        friction = friction.gather(1,order.clamp(0,friction.shape[1]-1))
        friction = friction[:,:pair.shape[1]]
        err = torch.where(pair,(friction-torch.as_tensor(self.mu,device=self.device)[:,None]).abs(),0.).max(-1).values
        out = cpu(torch.cat([clearance[:,None],per_finger,slip[:,None],bad.any(-1)[:,None],err[:,None]],dim=1))
        if not np.isfinite(out).all(): raise FloatingPointError('Nonfinite physical state')
        if (out[:,8]>1e-5).any(): raise AssertionError('Contact friction differs from requested coefficient')
        robot_pair = pair & (other >= self.robot.link_start) & (other < self.robot.link_end)
        self.last_contact_diagnostics = {
            'cube_position_m':cpu(cp).copy(),
            'cube_quaternion_wxyz':cpu(q).copy(),
            'contact_friction_error':out[:,8].copy(),
            'all_robot_contact_load_N':cpu(torch.where(robot_pair,force,0.).sum(-1)).copy(),
            'all_robot_normal_load_N':cpu(torch.where(robot_pair,
                torch.abs((c['force_a']*normal).sum(-1)),0.).sum(-1)).copy()}
        if not all(np.isfinite(v).all() for v in self.last_contact_diagnostics.values()):
            raise FloatingPointError('Nonfinite read-only contact diagnostics')
        return out[:,0],out[:,1:6],out[:,6],out[:,7].astype(bool)

    def _obs(self):
        q = cpu(self.robot.get_dofs_position(self.indices))
        qd = cpu(self.robot.get_dofs_velocity(self.indices))
        nominal = self.trajectory.target(self.tick/300)
        future = self.trajectory.target(self.tick/300+self.rates.action_dt)
        phase = self.trajectory.phase(self.tick/300)
        phases = np.array([phase==p for p in ('preload','test_lift','lift','hold')],np.float32)
        state = np.concatenate([(q-self.mid)/self.half,np.clip(qd/3,-10,10),
            (nominal-q)/.2,np.broadcast_to((future-nominal)/.1,q.shape),self.gains-1,
            np.broadcast_to(phases,(self.num_envs,4)),
            np.full((self.num_envs,1),(self.tick-self.start_tick)/(self.end_tick-self.start_tick))],axis=1)
        tactile = self.history.copy()
        if self.tactile_noise_std:
            tactile += self.rng.normal(0,self.tactile_noise_std,tactile.shape).astype(np.float32)
        if not np.isfinite(state).all(): raise FloatingPointError('Nonfinite joint state')
        return {'tactile':tactile,'state':state.astype(np.float32)}

    def reset(self):
        self.scene.__setstate__(self.prefix_checkpoint)
        self.gains[:] = getattr(self,'prefix_gains',1.)
        if getattr(self,'replay_source_prefix_gains',False):
            restored=self._prefix_physics_readbacks()
            self._assert_prefix_gain_readbacks(restored)
            for key,expected in self.prefix_readbacks.items():
                np.testing.assert_array_equal(restored[key],expected,
                    err_msg='Checkpoint did not exactly restore prefix '+key)
        # Mass, inertia, gains, targets, contact caches and simulation clock are
        # restored together. Do not recompute them after the exact restore.
        self.history[:] = self.prefix_history
        self.tick = self.start_tick
        self.returns = np.zeros(self.num_envs)
        self.max_clearance = np.full(self.num_envs,-np.inf)
        self.hold_min = np.full(self.num_envs,np.inf)
        self.hold_duration = np.zeros(self.num_envs)
        self.hold_contact_ok = np.ones(self.num_envs,dtype=bool)
        self.fault = np.zeros(self.num_envs,dtype=bool)
        self.load_integral = np.zeros(self.num_envs)
        self.peak_load = np.zeros(self.num_envs)
        self.slip_integral = np.zeros(self.num_envs)
        self.gain_integral = np.zeros((self.num_envs,2))
        self.elapsed = 0.
        self.episode_counter += 1
        self._pending_action = None
        return self._obs()

    def step_async(self, actions):
        a = np.asarray(actions,dtype=np.float32)
        if a.shape != (self.num_envs,2): raise ValueError('Expected one pair of actions per environment')
        self._pending_action = a.copy()

    def step_wait(self):
        if self._pending_action is None: raise RuntimeError('step_async must be called first')
        old = self.gains.copy()
        self.gains = rate_limit_gains(old,gains_from_action(self._pending_action))
        self._apply_gains()
        self.robot.control_dofs_position(self.trajectory.target(self.tick/300),self.indices)
        reward = -.02*np.square(self.gains-old).sum(-1)
        new_fault = np.zeros(self.num_envs,dtype=bool)
        for sub in range(self.rates.action_stride):
            self.scene.step()
            self.tick += 1
            if self.tick % self.rates.tactile_stride == 0:
                self._sample_tactile()
                clearance,load,slip,fault = self._metrics()
                dt = 1/self.rates.tactile_hz
                reward += reward_rate(clearance,load,slip)*dt
                self.max_clearance = np.maximum(self.max_clearance,clearance)
                new_fault |= fault & ~self.fault
                self.fault |= fault
                self.load_integral += load.sum(-1)*dt
                self.peak_load = np.maximum(self.peak_load,load.sum(-1))
                self.slip_integral += slip*dt
                t = self.tick/300
                hold_dt = float(interval_overlap(t-dt,t,self.hold_start,self.trajectory.duration_s))
                if hold_dt:
                    self.hold_duration += hold_dt
                    self.hold_min = np.minimum(self.hold_min,clearance)
                    self.hold_contact_ok &= (load>.05).sum(-1)>=2
                self.last_metrics = {'clearance_m':clearance.copy(),'contact_load_N':load.copy(),
                                     'slip_speed_m_s':slip.copy(),'fault':fault.copy(),
                                     **{k:v.copy() for k,v in getattr(self,'last_contact_diagnostics',{}).items()}}
        self.elapsed += self.rates.action_dt
        self.gain_integral += self.gains*self.rates.action_dt
        reward -= 20*new_fault
        done = self.tick >= self.end_tick
        infos = [{'elapsed_sim_s':self.elapsed,'tactile_samples':4,
                  'applied_gains':self.gains[i].copy()} for i in range(self.num_envs)]
        self.returns += reward
        obs = self._obs()
        if done:
            success = hold_success(self.hold_duration,self.hold_min,self.fault) & self.hold_contact_ok
            bonus = np.where(success,20.,-10.)
            reward += bonus
            self.returns += bonus
            for i in range(self.num_envs):
                metrics = {'success':bool(success[i]),'mass_kg':float(self.mass[i]),'friction':float(self.mu[i]),
                    'max_clearance_m':float(self.max_clearance[i]),'hold_min_clearance_m':float(self.hold_min[i]),
                    'hold_duration_s':float(self.hold_duration[i]),'hold_contact_ok':bool(self.hold_contact_ok[i]),
                    'mean_contact_force_N':float(self.load_integral[i]/self.elapsed),
                    'peak_contact_force_N':float(self.peak_load[i]),'slip_distance_proxy_m':float(self.slip_integral[i]),
                    'mean_gains':(self.gain_integral[i]/self.elapsed).tolist(),
                    'termination_reason':'collision' if self.fault[i] else 'success' if success[i] else 'hold_failed',
                    'return':float(self.returns[i]),'episode_index':self.episode_counter,
                    'sim_duration_s':self.elapsed,'steps':int((self.end_tick-self.start_tick)/20)}
                infos[i].update(episode_metrics=metrics,episode={'r':float(self.returns[i]),'l':metrics['steps']},
                    terminal_observation={k:v[i].copy() for k,v in obs.items()},**{'TimeLimit.truncated':False})
            obs = self.reset()
        self._pending_action = None
        return obs,reward.astype(np.float32),np.full(self.num_envs,done),infos

    def _write_protocol(self, model, started):
        import hashlib
        protocol = {'physics_hz':300,'tactile_hz':60,'action_hz':15,'tactile_samples_per_action':4,
            'tactile_model':current_tactile_model(),
            'solver_profile':self.solver_profile,'solver_readback':self.solver_readback,
            'rigid_options_actual':self.solver._options.model_dump(mode='json'),
            'history_frames':12,'actor_state_fields':self.state_fields,'privileged_actor_input':False,
            'action':'two bounded absolute pinch gain multipliers; [-1,1] -> [1,2]; max change .25/action',
            'kv':'base * sqrt(kp_multiplier)','effort_limits_unchanged':True,
            'target_resampling':'linear interpolation by original time, sampled and held at 15 Hz',
            'source_prefix_gain_replay':self.replay_source_prefix_gains,
            'prefix_gain_sampling':'source command grip_gain (missing=1) zero-order held; queried before each 300Hz physics step; changes applied at first grid time >= source command start; residual actions remain absolute gains',
            'prefix_gain_events':self.prefix_gain_events,
            'prefix_gains_per_lane':self.prefix_gains.tolist(),
            'demonstration_duration_s':self.trajectory.duration_s,'start_time_s':self.start_tick/300,
            'end_physics_time_s':self.end_tick/300,'hold_interval_s':[self.hold_start,self.trajectory.duration_s],
            'recorded_hold_interval_s':[self.recorded_hold_start,self.trajectory.duration_s],
            'terminal_hold_duration_s':self.terminal_hold_duration_s,
            'hold_window_policy':'legacy exact recorded 3s' if self.terminal_hold_duration_s is None else 'final 3s of uninterrupted recorded hold; all earlier commands and phase labels preserved',
            'success':'at each 60Hz sample during original final 3s hold: cube bottom >=.05m, >=2 tactile-bearing links with physical contacts, no robot table/self collision',
            'reward_measurement_hz':60,'slip_label':'force-weighted relative material-point tangent velocity, solver truth; no risk head',
            'force_metric':'sum of contact-force magnitudes, not normal force or calibrated taxel force',
            'mass_kg_per_lane':self.mass.tolist(),'friction_per_lane':self.mu.tolist(),
            'cube_size_m':self.cube_size_m,'cube_size_readback_m':self.cube_size_readback_m.tolist(),
            'table_top_z_m':TABLE_TOP_Z_M,
            'cube_geometry_per_lane':'shared scalar side length; different sizes require a fresh scene',
            'cube_build_density_kg_m3':self.cube_build_density_kg_m3,
            'density_kg_m3_per_lane':(self.mass/self.cube_size_m**3).tolist(),
            'mass_readback_kg_per_lane':self.mass_readback_kg_per_lane.tolist(),
            'inertia_readback_kg_m2_per_lane':self.inertia_readback_kg_m2_per_lane.tolist(),
            'density_readback_kg_m3_per_lane':self.density_readback_kg_m3_per_lane.tolist(),
            'density_readback_definition':'solver mass divided by collision-mesh bounding-box volume; not stale build-time material rho',
            'read_only_contact_diagnostics':{
                'cube_position_m':'world position, one xyz vector per lane',
                'cube_quaternion_wxyz':'world orientation, one wxyz quaternion per lane',
                'contact_friction_error':'maximum absolute collider friction error over valid robot/cube contacts; zero when none',
                'all_robot_contact_load_N':'sum of force_a magnitudes over valid cube contacts whose other link is in [robot.link_start,robot.link_end); includes arm and hand, excludes table',
                'actor_input':False,
                'legacy_contact_load_N':'unchanged sum by the five sensor-bearing links only'},
            'mass_and_inertia_readback_verified':True,'material_rho_is_build_time_only':True,
            'deterministic_algorithms':bool(gs.use_deterministic_algorithms),
            'prefer_decomposed_solver':int(self.solver.rigid_config.prefer_decomposed_solver),
            'randomization':'materials sampled once per lane before full physical prefix; fixed per lane across resets',
            'reset':'restore full SceneCheckpoint including material/control/warmstart/derived state in same scene; restore application tactile history separately',
            'prefix_wall_seconds':time.monotonic()-started,'seed':self.seed_value,
            'model_sha256':hashlib.sha256(Path(model).read_bytes()).hexdigest(),
            'demonstration_sha256':self.trajectory.sha256,
            'cube_initial_position_m':self.cube_initial_position.tolist(),
            'cube_initial_position_readback_m':self.cube_initial_position_readback.tolist(),
            'cube_initial_bottom_readback_m':(self.cube_initial_position_readback[:,2]-self.cube_size_m/2).tolist(),
            'prefix_physics_audit':self.prefix_audit_summary,
            'demonstration_path':str(self.trajectory.path),
            'nominal_commands':'frozen trajectory targets; no live RuleSupportController or IK replanning'}
        (self.output_dir/'environment_protocol.json').write_text(json.dumps(protocol,indent=2))

    def close(self):
        if not self._closed:
            gs.destroy()
            self._closed=True

    def get_attr(self, attr_name, indices=None):
        return [getattr(self,attr_name) for _ in self._get_indices(indices)]
    def set_attr(self, attr_name, value, indices=None):
        if indices is not None: raise NotImplementedError('Only shared attributes supported')
        setattr(self,attr_name,value)
    def env_method(self, method_name, *args, indices=None, **kwargs):
        raise NotImplementedError('Use batched environment methods directly')
    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False for _ in self._get_indices(indices)]
