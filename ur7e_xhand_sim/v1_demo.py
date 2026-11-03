"""v1 demonstration: one object x one hand shape, privileged scripted grasp in simulation.

The spec JSON fixes the grasp: palm offset from the object center, palm
orientation, open and closed hand postures and grip gain. Close, test_lift and
lift use BasePolicyV1's timing (straight lines, one target per 60 Hz tick, no
tactile feedback), so a successful demonstration predicts the bases. A
successful run writes, next to the base-format commands (control.jsonl):
  reference.json  Astra success reference (7 keyframes x 3 views)
  demo.json       grasp parameters for BasePolicyV1
  bases.json      a one-base library (this demonstration) for smoke tests
--scale runs the same spec on the object scaled by that factor (id suffix _x085 etc.):
a size check before collection, not a demonstration.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import traceback

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from environment import Sandbox
from build_assets import HAND_JOINTS, ARM_JOINTS
from contact_guard import assess_contacts
from residual_rl.v1_task import (BASE_MOTION, HAND_SHAPES, hand_shape_joints, link_region, opposing_support,
                                 scaled_object)

VIEWS = ('scene_front', 'scene_left', 'wrist')
CONTACT_N = .05
TABLE_LIMIT_N = 5.


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def compact(observation):
    result = dict(observation)
    result['tactile'] = {f: {k: v for k, v in row.items() if k not in ('force_sensor_N', 'force_link_N')}
                         for f, row in observation['tactile'].items()}
    return result


def object_normals(evaluation):
    """Normal force on the object per REGIONS entry from one evaluation sample."""
    regions = np.zeros(7)
    for c in evaluation['solver_contacts']:
        if {c['entity_a'], c['entity_b']} == {'cube', 'ur7e_xhand'}:
            regions[link_region(c['link_a'] if c['entity_a'] == 'ur7e_xhand' else c['link_b'])] += c['normal_force_N']
    return regions


def validate_spec(spec):
    if not str(spec.get('object_id', '')).isidentifier() or spec.get('hand_shape') not in HAND_SHAPES:
        raise ValueError('spec needs an identifier object_id and hand_shape pinch/tripod/wrap')
    for key, size in (('palm_offset_m', 3), ('palm_quaternion_wxyz', 4), ('open_hand_rad', 12), ('closed_hand_rad', 12)):
        value = np.asarray(spec.get(key), dtype=float)
        if value.shape != (size,) or not np.isfinite(value).all():
            raise ValueError(f'{key} must hold {size} finite numbers')
    if not .2 <= float(spec.get('grip_gain', 1.)) <= 3.:
        raise ValueError('grip_gain must lie in [.2, 3]')


class SpecDemo:
    """300 Hz physics, 60 Hz records, 15 Hz targets; the object is never moved by the script."""
    def __init__(self, sim, output, spec):
        validate_spec(spec)
        self.sim, self.spec = sim, dict(spec)
        self.out = Path(output).resolve()
        self.out.mkdir(parents=True, exist_ok=False)
        self.hand_shape = spec['hand_shape']
        self.gain_joints = np.asarray(hand_shape_joints(self.hand_shape))+6
        self.failed, self.correction, self.evals, self.stages = None, np.zeros(3), [], []
        self.record = (self.out/'control.jsonl').open('w', buffering=1)
        self.eval_record = (self.out/'evaluation.jsonl').open('w', buffering=1)
        sim.render_enabled = False
        sim.reset()
        self.home_quat = sim.array(sim.wrist.get_quat()).reshape(4)
        self.home_pos = sim.array(sim.wrist.get_pos()).reshape(3)
        self.set_gain(1.)
        write(self.out/'manifest.json', {'schema': 'ur7e_xhand_v1_demo', 'spec': self.spec,
            'object_spec': sim.object_spec.protocol(), 'solver_profile': sim.solver_profile,
            'robot_joint_order': sim.names, 'control_dt_s': 1/60, 'physics_dt_s': 1/300, 'target_update_hz': 15,
            'provenance': 'Scripted grasp from a fixed spec with privileged object center; no Astra inference.',
            'closure': 'fixed interpolation from open to closed hand, no tactile feedback',
            'pd_scaling': 'hand-shape finger kp multiplied by grip gain, kv by sqrt(gain); effort limits unchanged',
            'collision_rule': f'any robot self contact or robot/table normal >{TABLE_LIMIT_N} N stops the run'})

    def set_gain(self, gain):
        s = self.sim
        kp, kv = np.asarray(s.kp, float).copy(), np.asarray(s.kv, float).copy()
        kp[self.gain_joints] *= gain
        kv[self.gain_joints] *= np.sqrt(gain)
        s.robot.set_dofs_kp(kp, s.indices)
        s.robot.set_dofs_kv(kv, s.indices)
        np.testing.assert_allclose(s.array(s.robot.get_dofs_kp(s.indices)).reshape(-1), kp, rtol=1e-6)
        self.grip_gain = float(gain)

    def servo(self, pos, quat, hand, phase, ticks=4):
        """One target held for `ticks` 60 Hz samples (4 = one 15 Hz update), each logged."""
        s = self.sim
        pos, quat, hand = (np.asarray(v, float) for v in (pos, quat, hand))
        # Integral IK correction for arm sag; gain per call scales with ticks so its rate per second is fixed.
        self.correction = np.clip(self.correction+.03*ticks*(pos-s.array(s.wrist.get_pos()).reshape(3)), -.025, .025)
        q, err = s.robot.inverse_kinematics(s.wrist, pos=pos+self.correction, quat=quat,
            dofs_idx_local=s.arm_indices, return_error=True, max_solver_iters=100, seed=42)
        err = s.array(err).reshape(-1)
        if np.linalg.norm(err[:3]) > .003 or np.linalg.norm(err[3:]) > .03:
            raise ValueError('Demo IK failed')
        target = s.array(q).reshape(-1)[s.indices]
        target[6:] = hand
        if np.any(target < s.lower-1e-6) or np.any(target > s.upper+1e-6) or not np.isfinite(target).all():
            raise ValueError('Demo joint target outside limits')
        target = np.clip(target, s.lower, s.upper)
        payload = {'actions': dict(zip(s.names, ((target-s.offset)/s.scale).tolist())), 'steps': 1}
        for _ in range(ticks):
            before = s.observe(full=False)
            s.step(payload)
            after = s.observe(full=True)
            e = s.evaluate()
            e.update(step=s.steps, phase=phase, sample_time_s=s.steps/60)
            self.record.write(json.dumps({'step': s.steps, 'phase': phase, 'before': before, 'after': compact(after),
                'sent_target_rad': target.tolist(), 'grip_gain': self.grip_gain}, allow_nan=False)+'\n')
            self.eval_record.write(json.dumps(e, allow_nan=False)+'\n')
            self.evals.append(e)
            guard = assess_contacts(e['solver_contacts'], TABLE_LIMIT_N)
            if guard['stop']:
                raise ValueError(guard['reason'])

    def move(self, phase, pos, quat, hand, duration_s):
        s = self.sim
        start, q0, h0 = s.array(s.wrist.get_pos()).reshape(3), s.array(s.wrist.get_quat()).reshape(4), s.target[6:].copy()
        slerp = Slerp([0, 1], Rotation.from_quat(np.stack([q0[[1, 2, 3, 0]], np.asarray(quat)[[1, 2, 3, 0]]])))
        n = max(math.ceil(duration_s*15), math.ceil(np.linalg.norm(pos-start)/.002), math.ceil(np.max(np.abs(hand-h0))/.03))
        for i in range(1, n+1):
            u = i/n
            f = u*u*(3-2*u)
            self.servo(start+f*(pos-start), slerp(f).as_quat()[[3, 0, 1, 2]], h0+f*(hand-h0), phase)
        for _ in range(8):
            self.servo(pos, quat, hand, phase)

    def linear(self, phase, pos, quat, hand, on_tick=None):
        """BasePolicyV1.move timing: straight line, one target per 60 Hz tick, then settle ticks."""
        s, m = self.sim, BASE_MOTION
        start, h0 = s.array(s.wrist.get_pos()).reshape(3), s.target[6:].copy()
        n = max(m['min_ticks'], math.ceil(np.linalg.norm(pos-start)/m['position_m_per_tick']),
                math.ceil(np.max(np.abs(hand-h0))/m['hand_rad_per_tick']))
        for i in range(1, n+1):
            self.servo(start+(pos-start)*i/n, quat, h0+(hand-h0)*i/n, phase, ticks=1)
            if on_tick:
                on_tick()
        for _ in range(m['settle_ticks']):
            self.servo(pos, quat, hand, phase, ticks=1)

    def touching(self, evals):
        return max(object_normals(e)[:6].sum() for e in evals) > CONTACT_N

    def capture_stage(self, name):
        s = self.sim
        s.render_enabled = True
        s.render()
        s.render_enabled = False
        folder = self.out/f'stage_{name}'
        folder.mkdir()
        for view in VIEWS:
            (folder/f'{view}.jpg').write_bytes(s.image_bytes[view])
        o = s.observe(full=True)
        palm = np.eye(4)
        palm[:3, :3] = Rotation.from_quat(np.asarray(o['wrist_quaternion_wxyz'])[[1, 2, 3, 0]]).as_matrix()
        palm[:3, 3] = o['wrist_position_m']
        self.stages.append({'name': name, 'step': s.steps, 'sample_time_s': s.steps/60, 'image_dir': folder.name,
            'observation': o, 'sent_joint_target_rad': s.target.tolist(), 'palm_pose_world': palm.tolist(),
            'object_position_m': list(s.evaluate()['object_position_m']), 'grip_gain': self.grip_gain})
        write(self.out/'stages.json', self.stages)

    def grasp(self):
        s, c = self.sim, self.spec
        quat = np.asarray(c['palm_quaternion_wxyz'], float)
        open_hand, closed = np.asarray(c['open_hand_rad'], float), np.asarray(c['closed_hand_rad'], float)
        self.initial_object = (s.array(s.cube.get_pos()).reshape(3).tolist(), s.array(s.cube.get_quat()).reshape(4).tolist())
        for _ in range(10):
            self.servo(self.home_pos, self.home_quat, s.home[6:], 'settle')
        align = np.asarray(self.initial_object[0])+np.asarray(c['palm_offset_m'], float)
        pre = align+np.array([0, 0, .12])
        high = pre.copy()
        high[2] = max(pre[2], self.home_pos[2])
        self.move('preshape', self.home_pos, quat, open_hand, 2.)
        self.capture_stage('open')
        self.move('approach', high, quat, open_hand, 3.)
        self.move('approach', pre, quat, open_hand, 1.)
        self.move('align', align, quat, open_hand, 2.)
        self.capture_stage('aligned')
        if self.touching(self.evals):
            raise ValueError('Hand touches the object before close')
        self.set_gain(float(c.get('grip_gain', 1.)))

        def first_contact():
            if not any(row['name'] == 'light_contact' for row in self.stages) and self.touching(self.evals[-1:]):
                self.capture_stage('light_contact')
        self.linear('close', align, quat, closed, first_contact)
        if not any(row['name'] == 'light_contact' for row in self.stages):
            raise ValueError('No contact with the object during close')
        self.capture_stage('closed')
        self.linear('test_lift', align+np.array([0, 0, .02]), quat, closed)
        self.capture_stage('test_lifted')
        lifted = align+np.array([0, 0, .10])
        self.linear('lift', lifted, quat, closed)
        self.capture_stage('lifted')
        for _ in range(45):
            self.servo(lifted, quat, closed, 'hold')
        self.capture_stage('held')

    def run(self):
        try:
            self.grasp()
        except Exception as exc:
            self.failed = f'{type(exc).__name__}: {exc}'
            (self.out/'error.txt').write_text(traceback.format_exc())
        self.record.close()
        self.eval_record.close()
        hold = [e for e in self.evals if e['phase'] == 'hold'][-180:]
        lifted = len(hold) == 180 and min(e['bottom_clearance_m'] for e in hold) >= .05 and not self.failed
        support = bool(hold) and bool(opposing_support(np.stack([object_normals(e) for e in hold]), CONTACT_N).all())
        score = {'success': bool(lifted and support), 'lifted_and_held': bool(lifted), 'opposing_support_in_hold': support,
                 'fault': self.failed, 'control_steps': len(self.evals), 'hand_shape': self.hand_shape,
                 'hold_min_bottom_clearance_m': min((e['bottom_clearance_m'] for e in hold), default=None)}
        write(self.out/'score.json', score)
        if score['success']:
            self.export(score)
        return score

    def export(self, score):
        c, stages = self.spec, {row['name']: row for row in self.stages}
        # Before closing: BasePolicyV1 places the open hand here and v1_label compares close start with it.
        palm = np.asarray(stages['aligned']['palm_pose_world'])
        center_in_palm = palm[:3, :3].T@(np.asarray(stages['aligned']['object_position_m'])-palm[:3, 3])
        files = ['manifest.json', 'stages.json', 'control.jsonl', 'evaluation.jsonl', 'score.json']
        files += [f"{row['image_dir']}/{view}.jpg" for row in self.stages for view in VIEWS]
        write(self.out/'reference.json', {'schema': 'ur7e_xhand_genesis_success_reference_v1',
            'review_status': 'simulation_verified', 'run_dir': str(self.out), 'validation': score,
            'hand_shape': self.hand_shape, 'grasp_quaternion_wxyz': c['palm_quaternion_wxyz'],
            'open_hand_target_rad': c['open_hand_rad'], 'closed_hand_target_rad': c['closed_hand_rad'],
            'light_hand_target_rad': stages['light_contact']['sent_joint_target_rad'][6:],
            'hand_joint_order': HAND_JOINTS, 'arm_joint_order': ARM_JOINTS, 'not_hardware_calibration': True,
            'files': {name: sha(self.out/name) for name in files}})
        trajectory = sha(self.out/'control.jsonl')
        write(self.out/'trajectory_metadata.json', {'trajectory_sha256': trajectory, 'solver_profile': self.sim.solver_profile})
        write(self.out/'demo.json', {'schema': 'v1_demo', 'object_id': c['object_id'], 'object_spec': c['object_spec'],
            'hand_shape': self.hand_shape, 'palm_offset_m': c['palm_offset_m'],
            'palm_quaternion_wxyz': c['palm_quaternion_wxyz'], 'open_hand_rad': c['open_hand_rad'],
            'closed_hand_rad': c['closed_hand_rad'], 'grip_gain': float(c.get('grip_gain', 1.)),
            'object_center_in_palm_m': center_in_palm.tolist(), 'object_initial_position_m': self.initial_object[0],
            'object_initial_quaternion_wxyz': self.initial_object[1], 'trajectory': 'control.jsonl',
            'trajectory_sha256': trajectory, 'reference': 'reference.json',
            'reference_sha256': sha(self.out/'reference.json'), 'solver_profile': self.sim.solver_profile})
        write(self.out/'bases.json', {'schema': 'v1_base_library', 'object_id': c['object_id'],
            'object_spec': c['object_spec'], 'hand_shape': self.hand_shape, 'source': str(self.out),
            'bases': [{'base_id': f"{c['object_id']}_{self.hand_shape}_demo", 'trajectory_path': 'control.jsonl',
                'trajectory_sha256': trajectory, 'object_initial_position_m': self.initial_object[0],
                'object_initial_quaternion_wxyz': self.initial_object[1], 'label': 'S', 'split': 'train',
                'source': 'v1_demo'}]})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--spec', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--scale', type=float, default=1., help='object scale for a size check (not a demonstration)')
    args = p.parse_args()
    spec = json.loads(args.spec.read_text())
    if args.scale != 1:
        object_id, object_spec = scaled_object(spec['object_id'], spec['object_spec'], args.scale)
        spec = dict(spec, object_id=object_id, object_spec=object_spec)
    validate_spec(spec)
    sim = Sandbox(output_dir=args.output.parent/f'runtime_{args.output.name}', render=False, cube_mass_kg=.05,
        cube_friction=.8, robot_friction=.8, object_spec=spec['object_spec'], physics_dt_s=1/300,
        control_substeps=5, noslip_iterations=5, deterministic=True)
    score = SpecDemo(sim, args.output, spec).run()
    print(json.dumps(score), flush=True)
    raise SystemExit(0 if score['success'] else 1)


if __name__ == '__main__':
    main()
