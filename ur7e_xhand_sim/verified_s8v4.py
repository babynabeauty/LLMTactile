"""Demonstration-registered UR/XHand S8V4: learned contact prior + Astra + support rule.

Reuses the project's UR/XHand port, not the native Panda/Allegro embodiment.
Scene initialization/evaluation truth never enters model or registration inputs.
"""
import math
import json
import shutil
import numpy as np
import torch
from scipy.spatial.transform import Rotation,Slerp
from s8v4 import S8V4,XHandContactPrior,RuleSupportController,SkillAct,URDFFrames,HERE,GENDEX,VLM_ROOT,ARM_JOINTS,HAND_JOINTS,write,sha
from success_context import load_reference

class VerifiedS8V4(S8V4):
    def __init__(self,sim,label,reference):
        if not label or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in label):raise ValueError('Invalid label')
        self.ref,root=load_reference(reference)
        self.sim=sim;self.out=HERE.parent/'outputs/s8v4_genesis'/label;self.out.mkdir(parents=True,exist_ok=False)
        self.phase='unregistered';self.proposals=None;self.offset=np.zeros(3);self.log=[];self.evaluation=[]
        self.index=0;self.finished=False;self.failed=None;self.correction=np.zeros(3);self.support_history=[]
        self.open=np.array(self.ref['open_hand_target_rad']);self.closed=np.array(self.ref['closed_hand_target_rad'])
        if self.ref['hand_joint_order']!=HAND_JOINTS:raise ValueError('Demonstration joint order mismatch')
        if np.any(self.closed<sim.lower[6:]) or np.any(self.closed>sim.upper[6:]):raise ValueError('Demo targets outside limits')
        self.controller=RuleSupportController(sim.lower[6:],sim.upper[6:],fingers=(0,1),threshold=.01,period=1/15,max_speed=.25)
        with torch.device('cpu'):
            self.prior=XHandContactPrior(GENDEX,HERE/'generated/ur7e_xhand.urdf',HERE/'sources/xhand/meshes',self.closed,[0]*6)
        fk=URDFFrames(HERE/'generated/ur7e_xhand.urdf')
        self.tool_to_palm=fk.transform('tool0','right_hand_link',dict.fromkeys(ARM_JOINTS+HAND_JOINTS,0.))
        self.grasp_quat=np.array(self.ref['grasp_quaternion_wxyz'])
        self.rotation=Rotation.from_quat(self.grasp_quat[[1,2,3,0]]).as_matrix()
        self.relative_alignment=np.array(self.ref['desired_align_palm_world_m'])-self.ref['reference_anchor_world_m']
        self.closure=[]
        for line in (root/'control.jsonl').open():
            row=json.loads(line)
            if row['phase'] in ('close','preload'):self.closure.append(row)
        self.record=(self.out/'control.jsonl').open('w',buffering=1);self.eval_record=(self.out/'evaluation.jsonl').open('w',buffering=1)
        write(self.out/'scenario_evaluation_only.json',sim.scenario)
        shutil.copy2(reference,self.out/'demonstration_reference.json')
        write(self.out/'manifest.json',{'method':'UR7e/XHand S8V4 demonstration registered port',
            'high_level_driver':'vlm_policy_CodexAppServerPolicy','reference_sha256':sha(reference),
            'components':['verified historical multimodal context','live RGB-D registration','GenDex CMap-CVAE candidate scoring','independent Astra SkillAct','recorded close/preload trajectory','RuleSupportController'],
            'prior':self.prior.provenance,'camera_source':sim.camera_source,
            'alignment_reference':'demonstrated desired palm minus demonstrated visible RGB-D anchor',
            'local_motion':'same smooth Cartesian IK/PD and bounded .12 integral compensation as rule teacher',
            'support_gate':'both thumb/index tactile >= .01 nominal N for last 5 samples before upward motion; deflection is controller-only auxiliary evidence',
            'support_rule':'original UR/XHand RuleSupportController; active after recorded closure, bounded +.25 rad',
            'limits':'UR/XHand port; fixed-orientation translation transfer and bounded contact candidates, not full native Allegro optimizer',
            'source_hashes':{str(p):sha(p) for p in [HERE/'verified_s8v4.py',HERE/'s8v4.py',HERE/'success_context.py',HERE/'run_s8v4.py',HERE/'environment.py',VLM_ROOT/'real_variants/prior.py',VLM_ROOT/'tactile_policy/runtime.py']}})
        sim.reset();self.home_quat=np.array(sim.values(sim.wrist.get_quat()));self.home_pos=np.array(sim.values(sim.wrist.get_pos()))
        # Only physical settling at HOME before the first RGB-D/model observation.
        for _ in range(25):self.servo(self.home_pos,self.home_quat,sim.home[6:],'settle')
        self.capture()

    def support(self):
        result=super().support()
        if self.ref.get('execution_mode') == 'recorded_tripod_closure_v1':
            state=self.state();raw=np.asarray(state['hand_tactile_raw_force'])
            peak=float(np.linalg.norm(raw[2],axis=-1).max())
            result['middle']={'tactile_peak_nominal_N':peak,'supported':peak>=.01}
        for v in result.values():v['supported']=v['tactile_peak_nominal_N']>=.01
        return result

    def observe(self):
        result=super().observe()
        result.update(demonstration_reference_sha256=sha(self.out/'demonstration_reference.json'),
            registration_translation_m=getattr(self,'translation',None).tolist() if hasattr(self,'translation') else None,
            local_feedback=self.controller.last_diagnostic,
            lift_gate_ready=len(self.support_history)>=5 and all(self.support_history[-5:]))
        return result

    def servo(self,pos,quat,hand,phase):
        s=self.sim;self.correction=np.clip(self.correction+.12*(np.asarray(pos)-s.array(s.wrist.get_pos()).reshape(3)),-.025,.025)
        q,err=s.robot.inverse_kinematics(s.wrist,pos=np.asarray(pos)+self.correction,quat=quat,dofs_idx_local=s.arm_indices,return_error=True,max_solver_iters=100,seed=42)
        err=s.array(err).reshape(-1)
        if np.linalg.norm(err[:3])>.003 or np.linalg.norm(err[3:])>.03:raise ValueError('IK failed along registered skill')
        target=s.array(q).reshape(-1)[s.indices];target[6:]=hand
        if np.any(target<s.lower-1e-6) or np.any(target>s.upper+1e-6):raise ValueError('Joint limit in registered skill')
        s.render_enabled=False
        self._tick(np.clip(target,s.lower,s.upper),phase)
        self.support_history.append(all(v['supported'] for v in self.support().values()))
        if self.evaluation:
            bad=[c for c in self.evaluation[-1]['solver_contacts'] if c['entity_a'] in s.distractors or c['entity_b'] in s.distractors]
            if bad:raise ValueError('Robot contacted a distractor')
        if s.steps%12==0:
            s.render_enabled=True;s.render();s.render_enabled=False
            folder=self.out/'multiview_frames';folder.mkdir(exist_ok=True)
            for name in ('scene_front','scene_left','wrist'):(folder/f'{s.steps:06d}_{name}.jpg').write_bytes(s.image_bytes[name])

    def move(self,position,hand,phase,steps=None,quat=None):
        s=self.sim;start=s.array(s.wrist.get_pos()).reshape(3);h0=s.target[6:].copy();q0=s.array(s.wrist.get_quat()).reshape(4)
        quat=self.grasp_quat if quat is None else quat
        interp=Slerp([0,1],Rotation.from_quat(np.stack([q0[[1,2,3,0]],np.asarray(quat)[[1,2,3,0]]])))
        n=max(40,math.ceil(np.linalg.norm(position-start)/.001),math.ceil(np.max(np.abs(hand-h0))/.01),steps or 0)
        try:
            for i in range(1,n+1):
                u=i/n;f=u*u*(3-2*u);q=interp(f).as_quat()
                self.servo(start+f*(position-start),q[[3,0,1,2]],h0+f*(hand-h0),phase)
            for _ in range(30):self.servo(position,quat,hand,phase)
        finally:s.render_enabled=True

    def execute(self,payload):
        if self.finished or self.failed:raise ValueError('Trial already ended')
        a=SkillAct.model_validate(payload)
        self._validate_preload_capability(a)
        if a.observation_id!=self.observe()['observation_id'] or self.stamp!=(self.sim.episode,self.sim.steps):raise ValueError('Stale observation')
        entry={'action':a.model_dump(),'before':self.observe()};result={}
        try:
            if a.status!='continue':
                if a.status=='done' and self.phase!='holding':raise ValueError('done requires a completed hold')
                self.finished=True;self.phase=a.status;result={'status':'model_'+a.status}
            elif a.skill=='locate':
                if self.phase!='unregistered':raise ValueError('Already registered')
                x1,y1,x2,y2=a.object_roi_xyxy;h,w=self.cloud_image.shape[:2]
                if not (0<=x1<x2<=w and 0<=y1<y2<=h):raise ValueError('ROI outside image')
                cloud=self.cloud_image[y1:y2,x1:x2][self.mask[y1:y2,x1:x2]]
                cloud=cloud[np.isfinite(cloud).all(1)&(cloud[:,2]>.212)&(cloud[:,2]<.40)]
                if len(cloud)<40 or np.ptp(cloud,axis=0).max()>.15:raise ValueError('Insufficient or mixed-object RGB-D ROI')
                top=cloud[cloud[:,2]>np.percentile(cloud[:,2],95)-.003]
                xy=(np.percentile(top[:,:2],5,axis=0)+np.percentile(top[:,:2],95,axis=0))/2
                self.anchor=np.r_[xy,(np.median(top[:,2])+.20)/2]
                self.translation=self.anchor-self.ref['reference_anchor_world_m']
                if np.linalg.norm(self.translation)>.25:raise ValueError('Outside frozen 25 cm simulation registration range')
                self.alignment=self.anchor+self.relative_alignment
                palm=np.eye(4);palm[:3,:3]=self.rotation;palm[:3,3]=self.alignment;tool=palm@np.linalg.inv(self.tool_to_palm)
                with torch.device('cpu'):self.proposals=self.prior.propose(cloud,np.r_[tool[:3,3],Rotation.from_matrix(tool[:3,:3]).as_rotvec()],.20)
                # Keep the demonstrated baseline explicit alongside learned proposals,
                # as in the native harness's unrestricted bounded residual interface.
                if not any(np.allclose(c['offset_xyz_m'],0) for c in self.proposals['candidates']):
                    self.proposals['candidates'].append({'offset_xyz_m':[0.,0.,0.],'energy':None,
                        'source':getattr(self,'baseline_candidate_source','verified demonstration baseline; not GenDex ranked')})
                np.save(self.out/'observed_object_cloud.npy',cloud)
                write(self.out/'registration.json',{'roi':a.object_roi_xyxy,'estimated_center_world_m':self.anchor.tolist(),'translation_m':self.translation.tolist(),'nominal_palm_world_m':self.alignment.tolist(),'proposals':self.proposals})
                self.phase='located';result={'status':'located','point_count':len(cloud)}
            else:
                allowed={'approach':'located','align':'approached','close':'aligned','test_lift':'contact','lift':'test_lifted','hold':'lifted'}
                if self.phase!=allowed.get(a.skill):raise ValueError('Skill invalid in phase '+self.phase)
                if a.skill=='approach':
                    if not any(np.allclose(a.offset_xyz_m,c['offset_xyz_m'],atol=1e-8) for c in self.proposals['candidates']):raise ValueError('Choose supplied candidate')
                    self.offset=np.array(a.offset_xyz_m)
                elif a.skill in ('align','close') and not np.allclose(a.offset_xyz_m,self.offset):raise ValueError('Preserve offset')
                elif a.skill not in ('align','close') and not np.allclose(a.offset_xyz_m,0):raise ValueError('Other skill offsets must be zero')
                p=self.alignment+self.offset
                if a.skill=='approach':
                    self.move(self.home_pos,self.open,'preshape')
                    pre=p+np.array([0,0,.12]);high=pre.copy();high[2]=max(self.home_pos[2],pre[2])
                    self.move(high,self.open,'approach');self.move(pre,self.open,'approach');self.phase='approached'
                elif a.skill=='align':
                    self.move(p,self.open,'align')
                    if np.linalg.norm(self.sim.array(self.sim.wrist.get_pos()).reshape(3)-p)>.003:raise ValueError('Alignment outside 3 mm')
                    self.phase='aligned'
                elif a.skill=='close':
                    # Use the demonstrated sequence, never re-run the teacher's online closure decisions.
                    for row in self.closure:self.servo(p,self.grasp_quat,np.array(row['sent_target_rad'][6:]),row['phase'])
                    self.controller.activate(self.sim.target[6:]);self.phase='contact'
                    if not all(self.support_history[-5:]):result['warning']='Missing persistent two-finger tactile; upward motion blocked'
                elif a.skill in ('test_lift','lift'):
                    if len(self.support_history)<5 or not all(self.support_history[-5:]):raise ValueError('No persistent two-finger tactile; lift blocked')
                    self.move(p+np.array([0,0,.02 if a.skill=='test_lift' else .10]),self.sim.target[6:],a.skill,steps=80 if a.skill=='test_lift' else 160)
                    self.phase='test_lifted' if a.skill=='test_lift' else 'lifted'
                elif a.skill=='hold':
                    for _ in range(round(3/self.sim.control_dt_s)):self.servo(p+np.array([0,0,.10]),self.grasp_quat,self.sim.target[6:],'hold')
                    self.phase='holding'
                result['status']='executed'
        except Exception as exc:
            self.failed=str(exc);result={'status':'failed','error':str(exc)}
        self.index+=1;after=self.capture();entry.update(result=result,after=after);self.log.append(entry);write(self.out/'interactions.json',self.log)
        return {'receipt':result,'policy_observation':after}
