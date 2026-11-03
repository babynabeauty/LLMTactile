"""Causal 60 Hz actor adapter shared by the live skill executor.

Only measured joint state, raw link-frame tactile and nominal commands enter
this adapter. It never reads object state, contacts, mass or friction.
"""
from dataclasses import asdict
from pathlib import Path
import hashlib
import json
import numpy as np

from residual_rl.control_protocol import (V1_RATES, HISTORY_SAMPLES,
    current_control_protocol, elapsed_feature, require_checkpoint_control_compatibility)
from residual_rl.tactile_provenance import require_checkpoint_tactile_compatibility
from residual_rl.v1_task import (V1Config, FINGERS, PHASES, HAND_SHAPES, hand_shape_joints,
    limit_residual, REGIONS, link_region, opposing_support)


def actor_state(q, qd, command, nominal, future, lower, upper, gain, phase,
                elapsed_s, astra_level, use_level, finger_mask, residual, bound):
    """One V1 actor row, in exactly the training environment's feature order."""
    lower, upper = np.asarray(lower), np.asarray(upper)
    mid, half = (lower+upper)/2, (upper-lower)/2
    value = np.concatenate([(np.asarray(q)-mid)/half, np.clip(np.asarray(qd)/3, -10, 10),
        (np.asarray(command)-q)/.2, (np.asarray(future)-nominal)/.1, [gain-1],
        [phase == p for p in PHASES], [float(elapsed_feature(elapsed_s))],
        np.eye(4)[astra_level-1]*use_level, finger_mask, np.asarray(residual)/bound]).astype(np.float32)
    if not np.isfinite(value).all():
        raise FloatingPointError('Nonfinite online actor state')
    return value


class OnlineResidual:
    def __init__(self, sim, hand_shape, gain, *, checkpoint=None, zero=False, astra_level=4,
                 model=None, metadata=None):
        if not np.isclose(sim.control_dt_s, V1_RATES.action_dt, rtol=0, atol=1e-10):
            raise ValueError('Online residual requires a 60 Hz Sandbox control step')
        if checkpoint is not None and zero:
            raise ValueError('Choose a checkpoint or explicit zero residual')
        if not (zero or checkpoint is not None or model is not None):
            raise ValueError('Online mode needs a 60 Hz checkpoint or explicit zero residual')
        if type(astra_level) is not int or astra_level not in range(1, 5):
            raise ValueError('astra_level must be an integer from 1 to 4')
        self.sim, self.hand_shape, self.gain = sim, hand_shape, float(gain)
        self.zero, self.checkpoint = bool(zero), str(Path(checkpoint).resolve()) if checkpoint else None
        if self.checkpoint and not Path(self.checkpoint).is_file() and Path(self.checkpoint).suffix != '.zip':
            self.checkpoint = str(Path(self.checkpoint).with_suffix('.zip'))
        if checkpoint is not None:
            metadata = require_checkpoint_control_compatibility(checkpoint)
            require_checkpoint_tactile_compatibility(checkpoint)
            from stable_baselines3 import PPO
            import torch
            # A tiny single-lane network avoids CPU threadpool launch overhead;
            # the separate training process retains its own thread settings.
            torch.set_num_threads(1)
            model = PPO.load(checkpoint, device='cpu')
        metadata = metadata or {}
        if model is not None and metadata.get('control_protocol') != current_control_protocol():
            raise ValueError('Online actor metadata must declare the current 60 Hz protocol')
        self.model = model
        self.config = V1Config(**metadata.get('v1_config', {}))
        self.action_joints = np.asarray(metadata.get('action_joints', list(range(12))), dtype=int)
        own = hand_shape_joints(hand_shape)
        if (self.action_joints.ndim != 1 or len(set(self.action_joints.tolist())) != len(self.action_joints)
                or not set(self.action_joints.tolist()) <= set(range(12))
                or not set(own) <= set(self.action_joints.tolist())):
            raise ValueError('Checkpoint action_joints do not cover this hand shape')
        self.active = np.isin(self.action_joints, own).astype(float)
        self.controlled = self.action_joints+6
        self.mask = np.asarray([f in HAND_SHAPES[hand_shape] for f in FINGERS], np.float32)
        self.use_level = bool(metadata.get('use_level', True))
        self.astra_level = astra_level
        self.residual = np.zeros(len(self.action_joints))
        self.history = np.zeros((HISTORY_SAMPLES, 5, 120, 3), np.float32)
        self.proprio = None
        self.started = False
        self.start_step = None
        self.inferences = 0
        self.command = np.asarray(sim.target, float).copy()
        self.last_raw_action = np.zeros(len(self.action_joints))
        self.last_observation = None
        if self.model is not None:
            expected_state = 18*4+1+len(PHASES)+1+4+len(FINGERS)+len(self.action_joints)
            if self.model.action_space.shape != (len(self.action_joints),):
                raise ValueError('Checkpoint action dimension differs from its joint metadata')
            spaces = self.model.observation_space.spaces
            if (spaces['tactile'].shape != self.history.shape
                    or spaces['state'].shape != (HISTORY_SAMPLES, expected_state)):
                raise ValueError('Checkpoint observation dimensions differ from the V1 actor')

    def protocol(self):
        sources = [Path(__file__), Path(__file__).with_name('base_policy_v1.py'),
                   Path(__file__).parent/'residual_rl'/'v1_task.py',
                   Path(__file__).parent/'residual_rl'/'control_protocol.py']
        return dict(control_protocol=current_control_protocol(), checkpoint=self.checkpoint,
            checkpoint_sha256=hashlib.sha256(Path(self.checkpoint).read_bytes()).hexdigest() if self.checkpoint else None,
            source_sha256={str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
            zero_residual=self.zero, action_joints=self.action_joints.tolist(),
            active_action_joints=self.active.tolist(), config=asdict(self.config),
            astra_level=self.astra_level, use_level=self.use_level,
            tactile_frame='raw link-frame sensor force, matching V1VecEnv sensors.read().force',
            state_time_feature='elapsed simulated seconds since first close / 10, clipped to [0,1]',
            between_skills='hold nominal command; tactile and actor continue; next target delta is zero',
            inference_device='cpu', inference_cpu_threads=1 if self.checkpoint else None,
            privileged_actor_input=False)

    def apply(self, nominal, future, phase):
        nominal, future = np.asarray(nominal), np.asarray(future)
        if not self.started and phase != 'close':
            self.command = nominal.copy()
            return self.command.copy()
        first = not self.started
        if first:
            self.started, self.start_step = True, self.sim.steps
            # V1 reset fills proprio with the first state and starts tactile at zero.
            self.command = nominal.copy()
        observed = self.sim.observe(full=True)
        q = np.asarray([observed['joint_positions_rad'][n] for n in self.sim.names])
        qd = np.asarray([observed['joint_velocities_rad_s'][n] for n in self.sim.names])
        state = actor_state(q, qd, self.command, nominal, future, self.sim.lower, self.sim.upper,
            self.gain, phase, (self.sim.steps-self.start_step)*self.sim.control_dt_s,
            self.astra_level, self.use_level, self.mask, self.residual, self.config.residual_bound_rad)
        if first:
            self.proprio = np.repeat(state[None], HISTORY_SAMPLES, axis=0)
        else:
            self.proprio[:-1] = self.proprio[1:]
            self.proprio[-1] = state
        self.last_observation = {'tactile': self.history[None].copy(), 'state': self.proprio[None].copy()}
        if self.zero:
            raw = np.zeros(len(self.action_joints))
        else:
            raw = np.asarray(self.model.predict(self.last_observation, deterministic=True)[0]).reshape(-1)
        if raw.shape != self.residual.shape or not np.isfinite(raw).all():
            raise ValueError('Online actor returned an invalid residual action')
        self.residual = limit_residual(raw, self.residual, self.config)*self.active
        self.last_raw_action = raw.copy()
        self.command = nominal.copy()
        self.command[self.controlled] += self.residual
        self.command = np.clip(self.command, self.sim.lower, self.sim.upper)
        self.inferences += 1
        return self.command.copy()

    def after_step(self):
        if not self.started:
            return
        # V1 training reads .force directly (link frame), not force_sensor_N.
        value = np.stack([self.sim.array(self.sim.sensors[f].read().force).reshape(-1, 120, 3)[0]
                          for f in FINGERS]).astype(np.float32)
        if value.shape != (5, 120, 3) or not np.isfinite(value).all():
            raise FloatingPointError('Invalid online tactile sample')
        self.history[:-1] = self.history[1:]
        self.history[-1] = value


def strict_online_metrics(control_path, evaluations, control_dt_s, config, fault=None):
    """Post-run rigid-body checks; privileged evidence never enters the actor.

    The online Sandbox has no fragility sampler/breakage monitor. These results
    therefore cover slip, clearance and opposing support only, and say so.
    """
    from scipy.spatial.transform import Rotation
    observations = {}
    with Path(control_path).open() as stream:
        for line in stream:
            row = json.loads(line)
            observations[row['step']] = row['observation']
    samples = round(3/control_dt_s)
    hold = [e for e in evaluations if e['phase'] == 'hold'][-samples:]
    hold_steps = {e['step'] for e in hold}
    reference, previous_relative, max_slip, peak_region = None, None, 0., 0.
    support_ok, collision = True, bool(fault)
    for e in evaluations:
        o = observations[e['step']]
        q = np.asarray(o['wrist_quaternion_wxyz'])
        rotation = Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()
        relative = rotation.T@(np.asarray(e['object_position_m'])-o['wrist_position_m'])
        if reference is None and e['phase'] == 'test_lift':
            # V1 begins tracking at the command boundary, before its first lift
            # interval. The preceding recorded post-step state is that boundary.
            reference = relative if previous_relative is None else previous_relative
        if reference is not None:
            max_slip = max(max_slip, float(np.linalg.norm(relative-reference)))
        previous_relative = relative
        region = np.zeros(len(REGIONS))
        for contact in e['solver_contacts']:
            entity_a, entity_b = contact['entity_a'], contact['entity_b']
            robot_link = (contact['link_b'] if entity_a == 'cube' and entity_b == 'ur7e_xhand'
                          else contact['link_a'] if entity_b == 'cube' and entity_a == 'ur7e_xhand' else None)
            if robot_link is not None:
                index = link_region(robot_link)
                if index >= 0:
                    region[index] += contact['normal_force_N']
        peak_region = max(peak_region, float(region.max()))
        if e['step'] in hold_steps:
            support_ok &= bool(opposing_support(region[None], config.support_threshold_N)[0])
        collision |= bool(e.get('contact_guard', {}).get('stop'))
    complete = len(hold) == samples
    min_clearance = min((e['bottom_clearance_m'] for e in hold), default=None)
    clearance_ok = complete and min_clearance >= .05
    slip_ok = reference is not None and max_slip <= config.slip_fail_m
    success = bool(clearance_ok and support_ok and slip_ok and not collision)
    reason = ('collision_or_execution_fault' if collision else 'missing_test_lift' if reference is None
              else 'slip' if not slip_ok else 'hold_failed' if not clearance_ok
              else 'no_opposing_support' if not support_ok else 'success')
    return dict(strict_success=success, termination_reason=reason, slip_distance_m=max_slip,
        slip_limit_m=config.slip_fail_m, hold_duration_s=len(hold)*control_dt_s,
        hold_min_clearance_m=min_clearance, hold_opposing_support=bool(complete and support_ok),
        peak_region_normal_N=peak_region, contact_guard_fault=bool(collision),
        fragility_not_simulated=True, f_break_N=None, evaluation_sample_hz=1/control_dt_s,
        evaluation_scope='rigid unbreakable object; no fragility/breakage validation',
        slip_reference='state before first test_lift command, in palm frame',
        support_rule='thumb or palm and at least one other finger above support threshold',
        support_threshold_N=config.support_threshold_N)
