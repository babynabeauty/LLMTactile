"""Direct simulator demonstrations: privileged teacher, no model/API decisions.

Only robot joint targets and PD gains are commanded. Object state is read for
teacher planning and evaluation; it is never repositioned after episode reset.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import shutil
import traceback

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from environment import Sandbox
from build_assets import HERE, HAND_JOINTS, ARM_JOINTS
from residual_rl.tactile_provenance import current_tactile_model, source_paths as tactile_source_paths


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def compact(observation):
    result=dict(observation)
    result['tactile']={finger:{k:v for k,v in row.items()
        if k not in ('force_sensor_N','force_link_N')} for finger,row in observation['tactile'].items()}
    return result


class DirectTeacher:
    """Shared 300Hz physics / 60Hz record / 15Hz command teacher engine."""
    def __init__(self, sim, output, candidate):
        self.sim=sim
        if not np.isclose(sim.control_dt_s,1/60) or not np.isclose(sim.physics_dt_s,1/300):
            raise ValueError('Direct teacher requires explicit 300Hz physics and 60Hz sampling')
        self.out=Path(output).resolve();self.out.mkdir(parents=True,exist_ok=False)
        self.candidate=dict(candidate);self.phase='initial';self.failed=None
        self.correction=np.zeros(3);self.evals=[];self.raw=[];self.stages=[]
        self.record=(self.out/'control.jsonl').open('w',buffering=1)
        self.eval_record=(self.out/'evaluation.jsonl').open('w',buffering=1)
        self.grip_gain=1.;self.pd_readback=None;self.pending_pd=True
        self.sim.render_enabled=False;self.sim.reset()
        self.home_quat=sim.array(sim.wrist.get_quat()).reshape(4)
        self.home_pos=sim.array(sim.wrist.get_pos()).reshape(3)
        self.set_gain(1.)
        self.manifest={'schema':'ur7e_xhand_direct_teacher_v1','driver':'direct_teacher_search',
            'privileged_teacher':True,'actual_astra_api':False,
            'provenance':'Robot IK/PD plus bounded hand-shape search directly in simulator; no Astra inference.',
            'teacher_inputs':'Object pose, geometry and physical contacts may guide this teacher. These are not residual actor observations.',
            'no_attachment_or_object_pose_commands':True,'parameters':self.candidate,
            'object_spec':sim.object_spec.protocol(),'robot_joint_order':sim.names,
            'solver_profile':sim.solver_profile,'solver_readback':sim.solver_readback,
            'control_dt_s':1/60,'physics_dt_s':1/300,'target_update_hz':15,
            'tactile_hz':60,'sensor_finger_order':list(sim.sensors),
            'tactile_model':current_tactile_model(),
            'tactile_semantics':'Uncalibrated kinematic taxel estimates, nominal N; independent solver contact forces recorded separately.',
            'success_criterion':'Lowest physical object surface >=5cm above table throughout final3s; no robot/table/self collision.',
            'pd_scaling':'Thumb/index kp multiplied by absolute gain, kv by sqrt(gain); effort limits unchanged.',
            'source_sha256':{str(p.resolve()):sha(p) for p in [Path(__file__),HERE/'environment.py',
                HERE/'residual_rl/multiobject_geometry.py',HERE/'residual_rl/solver_profile.py',
                HERE/'generated/ur7e_xhand.urdf',*tactile_source_paths()]}}
        write(self.out/'manifest.json',self.manifest)
        for original in self.manifest['source_sha256']:
            source=Path(original);destination=self.out/'source_snapshot'/source.relative_to(HERE.parent)
            destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,destination)
        write(self.out/'object_scene.json',sim.describe())

    def set_gain(self, gain):
        if not np.isfinite(gain) or not .2<=gain<=3.:
            raise ValueError('Teacher common gain must be in [.2,3]')
        s=self.sim;kp=np.asarray(s.kp,float).copy();kv=np.asarray(s.kv,float).copy()
        kp[6:12]*=gain;kv[6:12]*=np.sqrt(gain)
        s.robot.set_dofs_kp(kp,s.indices);s.robot.set_dofs_kv(kv,s.indices)
        actual_kp=s.array(s.robot.get_dofs_kp(s.indices)).reshape(-1)
        actual_kv=s.array(s.robot.get_dofs_kv(s.indices)).reshape(-1)
        lo,hi=s.robot.get_dofs_force_range(s.indices)
        lo=s.array(lo).reshape(-1);hi=s.array(hi).reshape(-1)
        np.testing.assert_allclose(actual_kp,kp,rtol=1e-6)
        np.testing.assert_allclose(actual_kv,kv,rtol=1e-6)
        np.testing.assert_allclose(lo,-s.effort,rtol=1e-6)
        np.testing.assert_allclose(hi,s.effort,rtol=1e-6)
        self.grip_gain=float(gain)
        self.pd_readback={'common_thumb_index_gain':float(gain),'kp_readback':actual_kp.tolist(),
            'kv_readback':actual_kv.tolist(),'effort_lower_readback_Nm':lo.tolist(),
            'effort_upper_readback_Nm':hi.tolist(),'effort_limits_unchanged':True}
        self.pending_pd=True

    def observe_teacher(self):
        return {'observation':self.sim.observe(full=True),'evaluation':self.sim.evaluate()}

    def finger_loads(self):
        e=self.evals[-1] if self.evals else self.sim.evaluate()
        loads={f:0. for f in self.sim.sensors}
        for c in e['solver_contacts']:
            if 'cube' not in (c['entity_a'],c['entity_b']):continue
            for finger,layout in zip(self.sim.sensors,self.sim.layouts):
                if layout['link_name'] in (c['link_a'],c['link_b']):
                    loads[finger]+=float(np.linalg.norm(c['force_on_a_world_N']))
        return loads

    def servo(self, pos, quat, hand, phase, gain=None):
        """One 15Hz target update, followed by four independently logged samples."""
        s=self.sim;self.phase=phase
        if gain is not None and not np.isclose(gain,self.grip_gain):self.set_gain(gain)
        pos=np.asarray(pos,float);quat=np.asarray(quat,float);hand=np.asarray(hand,float)
        self.correction=np.clip(self.correction+.12*(pos-s.array(s.wrist.get_pos()).reshape(3)),-.025,.025)
        q,err=s.robot.inverse_kinematics(s.wrist,pos=pos+self.correction,quat=quat,
            dofs_idx_local=s.arm_indices,return_error=True,max_solver_iters=100,seed=42)
        err=s.array(err).reshape(-1)
        if np.linalg.norm(err[:3])>.003 or np.linalg.norm(err[3:])>.03:
            raise ValueError('Teacher IK failed')
        target=s.array(q).reshape(-1)[s.indices];target[6:]=hand
        if np.any(target<s.lower-1e-6) or np.any(target>s.upper+1e-6) or not np.isfinite(target).all():
            raise ValueError('Teacher joint target outside limits')
        target=np.clip(target,s.lower,s.upper)
        payload={'actions':dict(zip(s.names,((target-s.offset)/s.scale).tolist())),'steps':1}
        for _ in range(4):
            before=s.observe(full=False);s.step(payload);after=s.observe(full=True)
            e=s.evaluate();e.update(step=s.steps,phase=phase,sample_time_s=s.steps/60)
            row={'step':s.steps,'sample_time_s':s.steps/60,'phase':phase,'before':before,
                'sent_target_rad':target.tolist(),'after':compact(after),'grip_gain':self.grip_gain,
                'desired_palm_world_m':pos.tolist(),'desired_quaternion_wxyz':quat.tolist(),
                'cartesian_correction_m':self.correction.tolist()}
            if self.pending_pd:
                row['grip_gain_controller']=self.pd_readback;self.pending_pd=False
            self.record.write(json.dumps(row,allow_nan=False)+'\n')
            self.eval_record.write(json.dumps(e,allow_nan=False)+'\n');self.evals.append(e)
            self.raw.append({'sample_time_s':s.steps/60,
                'tactile':np.asarray([after['tactile'][f]['force_sensor_N'] for f in s.sensors],np.float32),
                'joint_positions_rad':np.asarray([after['joint_positions_rad'][n] for n in s.names],np.float32),
                'sent_target_rad':target.copy(),'grip_gain':self.grip_gain})
            bad=[c for c in e['solver_contacts'] if ('ur7e_xhand' in (c['entity_a'],c['entity_b']) and
                ('table' in (c['entity_a'],c['entity_b']) or c['entity_a']==c['entity_b']))]
            if bad:raise ValueError('Robot self/table contact: '+str([(c['link_a'],c['link_b']) for c in bad]))
            if s.steps%30==0:self.capture_frames()

    def capture_frames(self):
        s=self.sim;s.render_enabled=True;s.render();s.render_enabled=False
        folder=self.out/'frames';folder.mkdir(exist_ok=True)
        for name in ('scene_front','scene_left','wrist','hand_palm','hand_thumb'):
            if name in s.image_bytes:(folder/f'{s.steps:06d}_{name}.jpg').write_bytes(s.image_bytes[name])

    def stage(self, name):
        self.capture_frames()
        row={'name':name,'step':self.sim.steps,'sample_time_s':self.sim.steps/60,
            'observation':compact(self.sim.observe(full=True)),
            'sent_joint_target_rad':self.sim.target.tolist(),'grip_gain':self.grip_gain}
        self.stages.append(row);write(self.out/'stages.json',self.stages)

    def move(self, phase, pos, quat, hand, duration_s=2.):
        s=self.sim;start=s.array(s.wrist.get_pos()).reshape(3);q0=s.array(s.wrist.get_quat()).reshape(4)
        h0=s.target[6:].copy();hand=np.asarray(hand);pos=np.asarray(pos);quat=np.asarray(quat)
        interp=Slerp([0,1],Rotation.from_quat(np.stack([q0[[1,2,3,0]],quat[[1,2,3,0]]])))
        n=max(math.ceil(duration_s*15),math.ceil(np.linalg.norm(pos-start)/.002),
            math.ceil(np.max(np.abs(hand-h0))/.03))
        for i in range(1,n+1):
            u=i/n;f=u*u*(3-2*u);q=interp(f).as_quat()
            self.servo(start+f*(pos-start),q[[3,0,1,2]],h0+f*(hand-h0),phase)
        for _ in range(8):self.servo(pos,quat,hand,phase)

    def run_reference_pinch(self):
        s=self.sim;c=self.candidate
        ref=json.loads((HERE.parent/'outputs/rule_demos/teacher_pinch_v2_20261024/reference.json').read_text())
        quat=np.asarray(c.get('palm_quaternion_wxyz',ref['grasp_quaternion_wxyz']))
        hand=np.asarray(c.get('hand_open_rad',ref['open_hand_target_rad']),float)
        for _ in range(10):self.servo(self.home_pos,self.home_quat,s.home[6:],'settle')
        center=s.array(s.cube.get_pos()).reshape(3)
        align=center+np.asarray(c.get('palm_offset_m',[.096,.027,.108]))
        self.alignment=align;self.grasp_quat=quat;self.open_hand=hand.copy()
        write(self.out/'registration.json',{'source':'privileged simulator object center for direct teacher',
            'object_center_world_m':center.tolist(),'align_palm_world_m':align.tolist()})
        pre=align+np.array([0,0,.12]);high=pre.copy();high[2]=max(pre[2],self.home_pos[2])
        self.move('preshape',self.home_pos,quat,hand,2.);self.stage('open')
        self.move('approach',high,quat,hand,3.);self.move('approach',pre,quat,hand,1.)
        self.move('align',align,quat,hand,2.);self.stage('aligned')
        self.set_gain(c.get('grip_gain',1.));confirmed=0
        for _ in range(120):
            loads=self.finger_loads()
            if min(loads['thumb'],loads['index'])>=.05:
                confirmed+=1
                if confirmed>=2:break
            else:
                confirmed=0;hand[4]=min(hand[4]+c.get('close_increment_rad',.012),.85)
            self.servo(align,quat,hand,'close')
        else:raise ValueError('No persistent physical thumb/index contact at bounded closure')
        self.stage('light_contact');start=hand.copy();goal=hand.copy()
        goal[4]=min(goal[4]+c.get('preload_rad',.20),.85)
        for i in range(1,16):
            hand=start+(goal-start)*i/15;self.servo(align,quat,hand,'preload')
        for _ in range(10):self.servo(align,quat,hand,'preload')
        self.stage('closed')
        if min(self.finger_loads()[f] for f in ('thumb','index'))<.05:
            raise ValueError('Physical contact lost during preload')
        pre_lift_center_z=self.evals[-1]['object_position_m'][2]
        self.move('test_lift',align+np.array([0,0,.02]),quat,hand,2.);self.stage('test_lifted')
        # A long object can pivot while its center follows this short test lift.
        # Final success still uses the true lowest surface, never center height.
        if (self.evals[-1]['object_position_m'][2]-pre_lift_center_z<.005
                or min(self.finger_loads()[f] for f in ('thumb','index'))<.05):
            raise ValueError('Object did not follow test lift with retained physical contact')
        lifted=align+np.array([0,0,.10])
        self.move('lift',lifted,quat,hand,3.);self.stage('lifted')
        for _ in range(45):self.servo(lifted,quat,hand,'hold')
        self.stage('held')

    def finalize(self):
        hold=[e for e in self.evals if e['phase']=='hold'];window=hold[-180:]
        success=bool(len(window)==180 and min(e['bottom_clearance_m'] for e in window)>=.05 and not self.failed)
        score={'success':success,'task_success':success,'fault':self.failed,'control_steps':len(self.evals),
            'hold_seconds':len(hold)/60,'hold_min_bottom_clearance_m':min((e['bottom_clearance_m'] for e in window),default=None),
            'max_bottom_clearance_m':max((e['bottom_clearance_m'] for e in self.evals),default=0),
            'no_attachment_or_object_pose_commands':True,'simulated_success_only':True,
            'actual_astra_api':False,'driver':'direct_teacher_search','grip_gain':self.grip_gain}
        write(self.out/'score.json',score)
        if self.raw:
            np.savez_compressed(self.out/'raw_tactile.npz',**{k:np.asarray([r[k] for r in self.raw]) for k in self.raw[0]})
        self.record.close();self.eval_record.close()
        write(self.out/'artifact_sha256.json',{p.name:sha(p) for p in self.out.iterdir() if p.is_file() and p.name!='artifact_sha256.json'})
        return score

    def run(self, strategy=None):
        try:
            (strategy or (lambda t:t.run_reference_pinch()))(self)
        except Exception as exc:
            self.failed=f'{type(exc).__name__}: {exc}'
            (self.out/'error.txt').write_text(traceback.format_exc())
        finally:
            result=self.finalize()
        return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--shape',choices=['cuboid','hex_prism','sphere','cylinder'],required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--candidate-json',type=Path)
    p.add_argument('--gain',type=float,default=1.)
    p.add_argument('--preload',type=float,default=.20)
    p.add_argument('--noslip-iterations',type=int,choices=[0,5],default=0)
    args=p.parse_args()
    candidate=json.loads(args.candidate_json.read_text()) if args.candidate_json else {
        'name':'reference_pinch','grip_gain':args.gain,'preload_rad':args.preload}
    spec={'shape':args.shape,**({'height_m':.09} if args.shape=='cylinder' else {})}
    sim=Sandbox(output_dir=args.output.parent/f'runtime_{args.shape}',render=False,
        cube_mass_kg=.05,cube_friction=.8,robot_friction=.8,object_spec=spec,physics_dt_s=1/300,control_substeps=5,
        noslip_iterations=args.noslip_iterations)
    teacher=DirectTeacher(sim,args.output,candidate)
    result=teacher.run();print(json.dumps(result),flush=True)
    raise SystemExit(0 if result['success'] else 1)


if __name__=='__main__':main()
