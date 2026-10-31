"""Agent-selected squeeze/release skills for a FEM material experiment, not S8V4 scoring."""
import json,hashlib
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from rule_demo import RuleDemo, write
from s8v4 import S8V4, SkillAct, URDFFrames, HERE
from fem_diagnostics import mesh_quality,contact_diagnostics


class SiliconeSqueeze(RuleDemo):
    def __init__(self, sim, label):
        if sim.silicone is None: raise ValueError('Start scene with --silicone')
        if not label or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in label):
            raise ValueError('Unique alphanumeric label required')
        self.sim=sim; self.out=HERE.parent/'outputs/silicone_squeeze'/label
        self.out.mkdir(parents=True,exist_ok=False)
        self.phase='initial';self.finished=False;self.failed=None;self.index=0
        self.frames=[];self.evals=[];self.stages=[];self.correction=np.zeros(3);self.last_desired=None
        self.cfg={'servo_integral_gain':.12,'servo_correction_limit_m':.025}
        ref=json.loads((HERE.parent/'outputs/rule_demos/teacher_pinch_v2_20261024/reference.json').read_text())
        self.open_hand=np.array(ref['open_hand_target_rad']);self.hand=self.open_hand.copy()
        self.release_hand=np.clip(np.zeros(12),sim.lower[6:],sim.upper[6:])
        self.grasp_quat=np.array(ref['grasp_quaternion_wxyz'])
        self.relative_alignment=np.array(ref['desired_align_palm_world_m'])-ref['reference_anchor_world_m']
        self.fk=URDFFrames(HERE/'generated/ur7e_xhand.urdf')
        self.record=(self.out/'control.jsonl').open('w',buffering=1)
        self.eval_record=(self.out/'evaluation.jsonl').open('w',buffering=1)
        self.soft_record=(self.out/'deformation.jsonl').open('w',buffering=1)
        sim.reset();self.home_quat=np.array(sim.values(sim.wrist.get_quat()))
        self.rest=sim.array(sim.silicone.init_positions).reshape(-1,3).copy()
        self.tets=sim.array(sim.silicone.get_el2v()).astype(int)
        x=self.rest[self.tets];self.inv_rest=np.linalg.inv(np.stack([x[:,i]-x[:,3] for i in range(3)],axis=-1))
        self.rest_volumes=np.abs(np.linalg.det(np.stack([x[:,i]-x[:,3] for i in range(3)],axis=-1)))/6
        write(self.out/'mesh_quality.json',mesh_quality(self.rest,self.tets))
        np.savez_compressed(self.out/'mesh.npz',rest=self.rest,tets=self.tets,boundary=sim.silicone._surface_tri_np)
        self.metrics=[]
        sim.render_enabled=False
        for _ in range(20):sim.step({'steps':1})
        sim.render_enabled=True
        write(self.out/'manifest.json',{'task':'strong squeeze then release pink silicone block on table',
            'driver':'independent Astra API -> bridge -> bounded IK/PD skills',
            'not_full_S8V4':True,'soft_tactile_connected':False,
            'feedback':'live RGB-D plus privileged FEM strain diagnostics, NOT hardware tactile',
            'material':sim.silicone_parameters,'max_index_target_rad':.85,
            'deformation_threshold_stops':False,
            'release_hand_target_rad':self.release_hand.tolist(),
            'source_hashes':{f:hashlib.sha256((HERE/f).read_bytes()).hexdigest() for f in ('silicone_squeeze.py','environment.py','fem_diagnostics.py','ipc_support.py')},
            'joint_effort_limits_Nm':sim.effort.tolist(),
            'no_object_attachment_or_pose_writes':True,'reference_usage':'rigid demo palm offset and preshape only; no soft success demonstration'})
        self.capture_stage('initial')

    def capture_stage(self,name):
        result=super().capture_stage(name)
        p=self.sim.array(self.sim.silicone.get_state().pos).reshape(-1,3)
        np.savez_compressed(self.current_folder/'fem_positions.npz',positions=p)
        write(self.current_folder/'fem_contacts.json',contact_diagnostics(self.sim,p))
        if self.sim.fem_pcg_stats is not None:
            write(self.current_folder/'fem_pcg_stats.json',self.sim.fem_pcg_stats)
        coupler=self.sim.scene.sim.coupler
        if hasattr(coupler,'_abd_data_by_link'):
            comparisons=[]
            for link,entries in coupler._abd_data_by_link.items():
                if link.entity is not self.sim.robot:continue
                T=entries[0].transform
                pos=self.sim.array(link.get_pos(relative=False)).reshape(3)
                comparisons.append({'link':link.name,
                    'translation_difference_m':float(np.linalg.norm(T[:3,3]-pos)),
                    'max_affine_stretch_error':float(np.abs(np.linalg.svd(T[:3,:3],compute_uv=False)-1).max())})
            write(self.current_folder/'ipc_coupling.json',{'robot_links':comparisons,
                'max_translation_difference_m':max((r['translation_difference_m'] for r in comparisons),default=0),
                'note':'FEM contact uses IPC body poses; SDF proximity diagnostics use Genesis rigid poses.'})
        return result

    def deformation(self):
        st=self.sim.silicone.get_state();p=self.sim.array(st.pos).reshape(-1,3)
        if not np.isfinite(p).all():raise ValueError('Nonfinite FEM positions')
        x=p[self.tets];F=np.stack([x[:,i]-x[:,3] for i in range(3)],axis=-1)@self.inv_rest
        singular=np.linalg.svd(F,compute_uv=False);J=np.linalg.det(F)
        return {'step':self.sim.steps,'phase':self.phase,
            'principal_compression_p05':float(1-np.percentile(singular[:,-1],5)),
            'principal_compression_max':float(1-singular[:,-1].min()),
            'volume_ratio_min':float(J.min()),'extent_m':np.ptp(p,axis=0).tolist(),
            'minimum_J_element_id':int(J.argmin()),'inverted_element_count':int((J<0).sum()),
            'total_signed_volume_ratio':float((J*self.rest_volumes).sum()/self.rest_volumes.sum()),
            'bounds_min_m':p.min(0).tolist(),'bounds_max_m':p.max(0).tolist(),
            'velocity_available':self.sim.silicone_parameters['coupler'] != 'IPCCoupler',
            'max_vertex_speed_m_s':float(np.linalg.norm(self.sim.array(st.vel).reshape(-1,3),axis=1).max())
                if self.sim.silicone_parameters['coupler'] != 'IPCCoupler' else None}

    def observe(self):
        allowed={'initial':['locate'],'registered':['approach'],'approached':['align'],
            'aligned':['close','release'],'closed':['hold','release'],'held':['release'],
            'released':['retreat'],'retreated':[]}.get(self.phase,[])
        return {'observation_id':f'{self.sim.episode}:{self.sim.steps}:{self.index}',
            'phase':self.phase,'observation':self.sim.observe(),'image_dir':str(self.current_folder),
            'run_dir':str(self.out),'fault':self.failed,'allowed_skills':allowed,
            'experimental_FEM_diagnostics':self.deformation(),
            'soft_tactile_supported':False,'deformation_threshold_stops':False}

    def tick(self,pos,quat,hand):
        super().tick(pos,quat,hand)
        m=self.deformation();self.metrics.append(m);self.soft_record.write(json.dumps(m)+'\n')
        if self.sim.steps%12==0:
            np.savez_compressed(self.out/'frames'/f'{self.sim.steps:06d}_fem.npz',
                positions=self.sim.array(self.sim.silicone.get_state().pos).reshape(-1,3))
        e=self.evals[-1]
        if any(c['entity_a'] in self.sim.distractors or c['entity_b'] in self.sim.distractors for c in e['solver_contacts']):
            raise ValueError('Robot touched a distractor')

    def execute(self,payload):
        if self.finished:raise ValueError('Trial already finished')
        a=SkillAct.model_validate(payload)
        if a.observation_id!=self.observe()['observation_id']:raise ValueError('Stale observation')
        if any(a.offset_xyz_m):raise ValueError('This experiment accepts zero offsets only')
        before=self.sim.steps;error=None
        try:
            if a.status!='continue':
                if a.status=='done' and self.phase!='retreated':raise ValueError('Release and retreat before done')
                self.finished=True
                if a.status=='abort':self.failed=a.explanation
            elif a.skill not in self.observe()['allowed_skills']:raise ValueError('Skill invalid in current phase')
            elif a.skill=='locate':
                x1,y1,x2,y2=a.object_roi_xyxy;h,w=self.mask.shape
                if not (0<=x1<x2<=w and 0<=y1<y2<=h):raise ValueError('ROI out of bounds')
                cloud=self.cloud_image[y1:y2,x1:x2][self.mask[y1:y2,x1:x2]]
                cloud=cloud[np.isfinite(cloud).all(1)&(cloud[:,2]>.211)&(cloud[:,2]<.30)]
                if len(cloud)<35:raise ValueError('Insufficient visible soft-block points')
                top=cloud[cloud[:,2]>np.percentile(cloud[:,2],95)-.003]
                xy=(np.percentile(top[:,:2],5,axis=0)+np.percentile(top[:,:2],95,axis=0))/2
                self.anchor=np.r_[xy,(np.median(top[:,2])+.20)/2]
                self.align_pos=self.anchor+self.relative_alignment
                write(self.out/'registration.json',{'roi':a.object_roi_xyxy,'RGBD_anchor':self.anchor.tolist(),'alignment':self.align_pos.tolist()})
                self.capture_stage('registered')
            elif a.skill=='approach':
                pos=np.array(self.sim.values(self.sim.wrist.get_pos()))
                for _ in self.trajectory('preshape',pos,self.grasp_quat,self.hand):pass
                high=self.align_pos+np.array([0,0,.12]);high[2]=max(high[2],pos[2])
                for _ in self.trajectory('approach',high,self.grasp_quat,self.hand):pass
                for _ in self.trajectory('approach',self.align_pos+np.array([0,0,.12]),self.grasp_quat,self.hand):pass
                self.capture_stage('approached')
            elif a.skill=='align':
                for _ in self.trajectory('align',self.align_pos,self.grasp_quat,self.hand):pass
                self.capture_stage('aligned')
            elif a.skill=='close':
                self.phase='squeeze';start=self.hand.copy();goal=start.copy();goal[4]=min(.85,self.sim.upper[10])
                for i in range(1,171):
                    self.hand=start+(goal-start)*i/170
                    self.tick(self.align_pos,self.grasp_quat,self.hand)
                for _ in range(20):self.tick(self.align_pos,self.grasp_quat,self.hand)
                self.capture_stage('closed')
            elif a.skill=='hold':
                self.phase='hold'
                for _ in range(40):self.tick(self.align_pos,self.grasp_quat,self.hand)
                self.capture_stage('held')
            elif a.skill=='release':
                for _ in self.trajectory('release',self.align_pos,self.grasp_quat,self.open_hand,steps=80):pass
                for _ in self.trajectory('release',self.align_pos,self.grasp_quat,self.release_hand,steps=100):pass
                self.hand=self.release_hand.copy()
                for _ in range(40):self.tick(self.align_pos,self.grasp_quat,self.hand)
                self.capture_stage('released')
            elif a.skill=='retreat':
                for _ in self.trajectory('retreat',self.align_pos+np.array([0,0,.12]),self.grasp_quat,self.release_hand):pass
                self.phase='recovery'
                for _ in range(80):self.tick(self.last_desired,self.grasp_quat,self.release_hand)
                self.capture_stage('retreated')
        except Exception as exc:
            error=str(exc);self.failed=error;self.finished=True
            self.capture_stage('failed')
        finally:
            self.sim.render_enabled=True;self.sim.render()
        receipt={'skill':a.skill,'status':a.status,'steps_executed':self.sim.steps-before,'error':error}
        with (self.out/'actions.jsonl').open('a') as f:f.write(json.dumps({'action':a.model_dump(),'receipt':receipt})+'\n')
        if self.finished:
            self.score={'completed_sequence':self.phase=='retreated' and not self.failed,'fault':self.failed,
                'peak_compression_p05':max((m['principal_compression_p05'] for m in self.metrics),default=0),
                'final_deformation':self.deformation(),'steps':self.sim.steps,'no_attachment_or_object_pose_writes':True}
            write(self.out/'score.json',self.score)
            self.record.close();self.eval_record.close();self.soft_record.close()
        return {'receipt':receipt,'policy_observation':self.observe(),'finished':self.finished}
