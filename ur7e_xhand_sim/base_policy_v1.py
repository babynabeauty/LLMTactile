"""BasePolicyV1: Astra skill executor without rule tightening, used to collect v1 bases.

Astra decides where to grasp (object box, candidate offset, skill sequence);
one v1 demonstration (v1_demo.py) fixes the hand: palm orientation, open and
closed hand, grasp center and grip gain. close interpolates straight to the
demonstrated closed hand. There is no rule controller, no Astra force choice
and no support gate before lifting. The object starts at an explicit
placement. Only robot targets are commanded; the object is never moved.
"""
from pathlib import Path
import json
import math
import sys
import types

import numpy as np
import torch
from scipy.spatial.transform import Rotation, Slerp

from build_assets import HERE, ARM_JOINTS, HAND_JOINTS
from contact_guard import assess_contacts
from residual_rl.multiobject_geometry import ObjectSpec
from residual_rl.v1_task import FINGERS, FINGER_JOINTS, HAND_SHAPES, hand_shape_joints

VLM_ROOT = Path('/home/sai/zsq/vlm-policy')
GENDEX = Path('/home/sai/zsq/bench2dex-work/third_party/GenDexGrasp')
if 'vlm_policy_local' not in sys.modules:
    package = types.ModuleType('vlm_policy_local')
    package.__path__ = [str(VLM_ROOT)]
    sys.modules['vlm_policy_local'] = package
from vlm_policy_local.real_variants.prior import XHandContactPrior
from vlm_policy_local.real_variants.config import SkillAct as ProjectSkillAct
from vlm_policy_local.urdf_frames import URDFFrames

HAND_SHAPE_TEXT = {'pinch': 'two-finger pinch (thumb and index)', 'tripod': 'three-finger grasp (thumb, index, middle)',
                   'wrap': 'whole-hand wrap (not every finger needs contact)'}
INSTRUCTIONS = '''You control a simulated UR7e + XHAND1 through the BasePolicyV1 skill executor.
Task: grasp the orange {shape}, lift it at least 5 cm clear of the table and hold it for 3 simulated
seconds. Physics pauses during your decisions. No hardware is connected. Return only SkillAct JSON.
Use the current observation_id and the sequence locate -> approach -> align -> close -> test_lift
-> lift -> hold -> done.
One verified successful simulation demonstration of this object with a {hand} is supplied below.
The executor copies its palm orientation, open and closed hand and grip stiffness, and places the
palm relative to the object center you localize. You decide where to grasp: the object box in
locate and the candidate offset in approach. Grasp force is not your decision.
For locate, give a tight bounding box around the orange object in scene_front ONLY, using the
width/height in stack.image_sizes_px, in [x1,y1,x2,y2] order; exclude the hand. For approach select
an exact supplied candidate offset; for align and close preserve it. Other offsets and non-locate
boxes must be zero.
close moves the fingers straight to the demonstrated closed hand. No tactile correction runs in this
executor and lifting is never blocked. Always continue through test_lift, lift and hold even if the
grasp looks poor; an independent evaluation scores the result. Return done with skill=none after
hold, or abort with skill=none only after an executor error. Do not repeat skills invalid in the
current phase. Tactile values are uncalibrated simulated estimates, not hardware units.
'''


class SkillAct(ProjectSkillAct):
    @classmethod
    def model_json_schema(cls, *args, **kwargs):
        # The API's strict response schema requires every property to be required.
        schema = super().model_json_schema(*args, **kwargs)
        schema['required'] = list(schema['properties'])
        return schema


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False))


def instructions(demo):
    return INSTRUCTIONS.format(shape=demo['object_spec']['shape'], hand=HAND_SHAPE_TEXT[demo['hand_shape']])


def yaw_quaternion(yaw_deg):
    half = math.radians(yaw_deg)/2
    return np.array([math.cos(half), 0., 0., math.sin(half)])


class BasePolicyV1:
    def __init__(self, sim, label, demo_path, placement):
        demo = json.loads(Path(demo_path).read_text())
        if demo.get('schema') != 'v1_demo':
            raise ValueError('BasePolicyV1 needs a v1 demonstration (demo.json)')
        # An unseen test object reuses the nearest seen object's demonstration.
        if not placement.get('unseen_object') and ObjectSpec.from_value(demo['object_spec']) != sim.object_spec:
            raise ValueError('Bridge object differs from the demonstration object')
        if not label or any(not (c.isalnum() or c in '_-') for c in label):
            raise ValueError('label must be a new alphanumeric run name')
        self.sim, self.demo, self.demo_path = sim, demo, str(Path(demo_path).resolve())
        self.out = HERE.parent/'outputs/base_policy_v1'/label
        self.out.mkdir(parents=True, exist_ok=False)
        self.phase, self.proposals, self.offset = 'unregistered', None, np.zeros(3)
        self.log, self.evaluation, self.index, self.finished, self.failed = [], [], 0, False, None
        self.hand_shape = demo['hand_shape']
        self.open = np.clip(demo['open_hand_rad'], sim.lower[6:], sim.upper[6:])
        self.closed = np.clip(demo['closed_hand_rad'], sim.lower[6:], sim.upper[6:])
        self.grasp_quat = np.asarray(demo['palm_quaternion_wxyz'], float)
        self.rotation = Rotation.from_quat(self.grasp_quat[[1, 2, 3, 0]]).as_matrix()
        # locate places the palm so the object center sits where it did in the demonstration.
        self.grasp_center = np.asarray(demo['object_center_in_palm_m'], float)
        self.grip_gain = float(demo['grip_gain'])
        with torch.device('cpu'):
            self.prior = XHandContactPrior(GENDEX, HERE/'generated/ur7e_xhand.urdf', HERE/'sources/xhand/meshes',
                                           self.closed, [0]*6)
        joints = dict(zip(HAND_JOINTS, self.closed))
        joints.update(dict.fromkeys(ARM_JOINTS, 0.))
        self.tool_to_palm = URDFFrames(HERE/'generated/ur7e_xhand.urdf').transform('tool0', 'right_hand_link', joints)
        sim.reset()
        gain_joints = np.asarray(hand_shape_joints(self.hand_shape))+6
        kp, kv = np.asarray(sim.kp, float).copy(), np.asarray(sim.kv, float).copy()
        kp[gain_joints] *= self.grip_gain
        kv[gain_joints] *= np.sqrt(self.grip_gain)
        sim.robot.set_dofs_kp(kp, sim.indices)
        sim.robot.set_dofs_kv(kv, sim.indices)
        position = sim._initial_object_position()
        position[:2] = np.asarray(demo['object_initial_position_m'][:2], float)+np.asarray(
            placement.get('offset_xy_m', [0., 0.]), float)
        quaternion = yaw_quaternion(float(placement.get('yaw_deg', 0.)))
        sim.cube.set_pos(position)
        sim.cube.set_quat(quaternion)
        self.placement = dict(placement, object_initial_position_m=position.tolist(),
                              object_initial_quaternion_wxyz=quaternion.tolist())
        self.record = open(self.out/'control.jsonl', 'w', buffering=1)
        self.eval_record = open(self.out/'evaluation.jsonl', 'w', buffering=1)
        write(self.out/'manifest.json', dict(method='BasePolicyV1', v1_demo=self.demo_path,
            hand_shape=self.hand_shape, placement=self.placement, grip_gain=self.grip_gain,
            open_hand=self.open.tolist(), closed_hand=self.closed.tolist(),
            grasp_quaternion_wxyz=self.grasp_quat.tolist(), object_center_in_palm_m=self.grasp_center.tolist(),
            object_spec=sim.object_spec.protocol(), solver_profile=sim.solver_profile,
            camera_views=sim.camera_views, control_dt_s=sim.control_dt_s,
            rule_controller=None, lift_support_gate=None, astra_force_choice=None,
            closure='fixed interpolation from the open to the demonstrated closed hand',
            success='object bottom > table + .05 m for the final 3 simulated seconds, no execution fault'))
        self.capture()

    # ------------------------------------------------------------ observation
    def state(self):
        s = self.sim
        o = s.observe(full=True)
        return {'read_monotonic_s': s.steps*s.control_dt_s, 'hand_q': [o['joint_positions_rad'][n] for n in HAND_JOINTS],
                'hand_tactile_raw_force': [o['tactile'][l['finger']]['force_sensor_N'] for l in s.layouts]}

    def support(self):
        """Observation only: contact evidence of the hand-shape fingers; never gates a skill."""
        state = self.state()
        peaks = np.linalg.norm(np.asarray(state['hand_tactile_raw_force']), axis=-1).max(axis=1)
        q, target = np.asarray(state['hand_q']), self.sim.target[6:]
        return {finger: {'tactile_peak_nominal_N': float(peaks[i]), 'joint_deflection_rad': float(max(0, np.max(
            target[list(FINGER_JOINTS[finger])]-q[list(FINGER_JOINTS[finger])])))}
            for i, finger in enumerate(FINGERS) if finger in HAND_SHAPES[self.hand_shape]}

    def capture(self):
        s = self.sim
        s.render_enabled = True
        s.render()
        folder = self.out/f'view_{self.index:03d}'
        folder.mkdir(exist_ok=True)
        for name, data in s.image_bytes.items():
            if name != 'hand':
                (folder/f'{name}.jpg').write_bytes(data)
        # Observed RGB-D only; no segmentation IDs or object state.
        pc, mask = s.camera.render_pointcloud(world_frame=True)
        width, height = s.camera.res
        self.cloud_image = s.array(pc).reshape(height, width, 3)
        self.mask = s.array(mask).reshape(height, width)
        view = s.camera_views['scene_front']
        forward = Rotation.from_quat(view['optical_quaternion_xyzw']).as_matrix()[:, 2]
        depth = (self.cloud_image-np.asarray(view['position_m']))@forward
        self.mask = self.mask & (depth > view['near_m']) & (depth < view['far_m'])
        self.stamp, self.current_folder = (s.episode, s.steps), folder
        return self.observe()

    def observe(self):
        s = self.sim
        return {'observation_id': f'{s.episode}:{s.steps}:{self.index}', 'phase': self.phase,
                'observation': s.observe(), 'support': self.support(), 'proposals': self.proposals,
                'selected_offset_xyz_m': self.offset.tolist(),
                'alignment_palm_world_m': self.alignment.tolist() if hasattr(self, 'alignment') else None,
                'image_dir': str(self.current_folder), 'run_dir': str(self.out), 'fault': self.failed}

    # ---------------------------------------------------------------- motion
    def _tick(self, target, phase):
        s = self.sim
        s.step({'actions': dict(zip(s.names, ((target-s.offset)/s.scale).tolist())), 'steps': 1})
        self.record.write(json.dumps({'step': s.steps, 'phase': phase, 'target': target.tolist(),
                                      'observation': s.observe(full=True)})+'\n')
        # Privileged data goes only to a separate evaluation stream, never policy observations.
        e = s.evaluate()
        e.update(step=s.steps, phase=phase)
        guard = assess_contacts(e['solver_contacts'], s.table_contact_force_limit_N)
        e['contact_guard'] = guard
        self.evaluation.append(e)
        self.eval_record.write(json.dumps(e)+'\n')
        if guard['stop']:
            write(self.out/'contact_guard_stop.json', dict(step=s.steps, phase=phase, **guard))
            raise ValueError(guard['reason'])

    def move(self, position, hand, phase):
        """Straight palm path while turning to the demonstrated grasp orientation."""
        s = self.sim
        start, q0 = s.array(s.wrist.get_pos()).reshape(3), s.array(s.wrist.get_quat()).reshape(4)
        hand0 = s.target[6:].copy()
        rotations = Rotation.from_quat(np.stack([q0[[1, 2, 3, 0]], self.grasp_quat[[1, 2, 3, 0]]]))
        turn, slerp = (rotations[0].inv()*rotations[1]).magnitude(), Slerp([0, 1], rotations)
        count = max(20, int(np.ceil(np.linalg.norm(position-start)/.0015)),
                    int(np.ceil(np.max(np.abs(hand-hand0))/.012)), int(np.ceil(turn/.01)))
        s.render_enabled = False
        try:
            for i in range(1, count+1):
                f = i/count
                q, err = s.robot.inverse_kinematics(s.wrist, pos=start+(position-start)*f,
                    quat=slerp(f).as_quat()[[3, 0, 1, 2]], dofs_idx_local=s.arm_indices, return_error=True,
                    max_solver_iters=80, seed=42)
                err = s.array(err).reshape(-1)
                if np.linalg.norm(err[:3]) > .004 or np.linalg.norm(err[3:]) > .03:
                    raise ValueError('IK failed along skill path')
                target = s.array(q).reshape(-1)[s.indices]
                target[6:] = hand0+(hand-hand0)*f
                self._tick(target, phase)
            for _ in range(12):
                self._tick(target, phase)
        finally:
            s.render_enabled = True

    def locate(self, roi):
        x1, y1, x2, y2 = roi
        height, width = self.cloud_image.shape[:2]
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError('ROI outside scene image')
        cloud = self.cloud_image[y1:y2, x1:x2][self.mask[y1:y2, x1:x2]]
        cloud = cloud[np.isfinite(cloud).all(1) & (cloud[:, 2] > .212) & (cloud[:, 2] < .40)]
        if len(cloud) < 40 or np.ptp(cloud, axis=0).max() > .15:
            raise ValueError('Insufficient/unbounded observed object surface')
        top = cloud[cloud[:, 2] > np.percentile(cloud[:, 2], 95)-.003]
        # Upright tabletop object: visible top and the calibrated table give the center height.
        xy = (np.percentile(top[:, :2], 5, axis=0)+np.percentile(top[:, :2], 95, axis=0))/2
        self.anchor = np.r_[xy, (np.median(top[:, 2])+.20)/2]
        self.alignment = self.anchor-self.rotation@self.grasp_center
        palm = np.eye(4)
        palm[:3, :3], palm[:3, 3] = self.rotation, self.alignment
        tool = palm@np.linalg.inv(self.tool_to_palm)
        toolpose = np.r_[tool[:3, 3], Rotation.from_matrix(tool[:3, :3]).as_rotvec()]
        with torch.device('cpu'):
            self.proposals = self.prior.propose(cloud, toolpose, .20)
        np.save(self.out/'observed_object_cloud.npy', cloud)
        write(self.out/'registration.json', {'roi': roi, 'surface_count': len(cloud),
            'estimated_center_world_m': self.anchor.tolist(), 'nominal_palm_world_m': self.alignment.tolist(),
            'proposals': self.proposals})
        return {'status': 'located', 'point_count': len(cloud)}

    def execute(self, payload):
        if self.finished:
            raise ValueError('Trial finished; start a new unique run')
        action = SkillAct.model_validate(payload)
        if action.observation_id != self.observe()['observation_id'] or self.stamp != (self.sim.episode, self.sim.steps):
            raise ValueError('Stale observation; refresh policy observation')
        if self.failed:
            raise ValueError('Trial stopped after execution fault')
        entry, result = {'action': action.model_dump(), 'before': self.observe()}, {}
        try:
            if action.status != 'continue':
                self.finished, self.phase, result = True, action.status, {'status': 'model_'+action.status}
            elif action.skill == 'locate':
                if self.phase != 'unregistered':
                    raise ValueError('Already registered')
                result = self.locate(action.object_roi_xyxy)
                self.phase = 'located'
            else:
                allowed = {'approach': ['located'], 'align': ['approached'], 'close': ['aligned'],
                           'test_lift': ['contact'], 'lift': ['test_lifted'], 'hold': ['lifted', 'holding']}
                if self.phase not in allowed.get(action.skill, []):
                    raise ValueError('Skill invalid in phase '+self.phase)
                if action.skill == 'approach':
                    if not any(np.allclose(action.offset_xyz_m, c['offset_xyz_m'], atol=1e-8)
                               for c in self.proposals['candidates']):
                        raise ValueError('Select a supplied GenDex offset')
                    self.offset = np.array(action.offset_xyz_m)
                elif action.skill in ('align', 'close') and not np.allclose(action.offset_xyz_m, self.offset):
                    raise ValueError('Preserve selected offset')
                position = self.alignment+self.offset
                if action.skill == 'approach':
                    # Move horizontally above the workpiece before descending.
                    high = position.copy()
                    high[2] = max(self.sim.values(self.sim.wrist.get_pos())[2], position[2]+.10)
                    self.move(high, self.open, 'approach')
                    self.move(position+np.array([0, 0, .10]), self.open, 'approach')
                    self.phase = 'approached'
                elif action.skill == 'align':
                    self.move(position, self.open, 'align')
                    self.phase = 'aligned'
                elif action.skill == 'close':
                    self.move(position, self.closed, 'close')
                    self.phase = 'contact'
                elif action.skill in ('test_lift', 'lift'):
                    pos = self.sim.array(self.sim.wrist.get_pos()).reshape(3)
                    pos[2] += .02 if action.skill == 'test_lift' else .08
                    self.move(pos, self.closed, action.skill)
                    self.phase = 'test_lifted' if action.skill == 'test_lift' else 'lifted'
                else:
                    self.sim.render_enabled = False
                    try:
                        for _ in range(round(3/self.sim.control_dt_s)):
                            self._tick(self.sim.target.copy(), 'hold')
                    finally:
                        self.sim.render_enabled = True
                    self.phase = 'holding'
                result['status'] = 'executed'
        except Exception as exc:
            self.failed = str(exc)
            result = {'status': 'failed', 'error': str(exc)}
        self.index += 1
        after = self.capture()
        entry.update(result=result, after=after)
        self.log.append(entry)
        write(self.out/'interactions.json', self.log)
        return {'receipt': result, 'policy_observation': after}

    def score(self):
        samples = round(3/self.sim.control_dt_s)
        hold = [e for e in self.evaluation if e['phase'] == 'hold'][-samples:]
        success = bool(len(hold) == samples and min(e['bottom_clearance_m'] for e in hold) >= .05 and not self.failed)
        score = {'success': success, 'fault': self.failed, 'phase': self.phase, 'control_steps': len(self.evaluation),
                 'hold_min_bottom_clearance_m': min((e['bottom_clearance_m'] for e in hold), default=None),
                 'run_dir': str(self.out), 'evaluation_only': True}
        self.record.close()
        self.eval_record.close()
        write(self.out/'score.json', score)
        return score
