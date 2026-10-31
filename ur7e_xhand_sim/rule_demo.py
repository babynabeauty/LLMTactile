"""Deterministic RGB-D / contact-feedback teacher; simulation demonstration only."""
import hashlib,json,math
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation,Slerp
from PIL import Image
from build_assets import HERE,ARM_JOINTS,HAND_JOINTS
from s8v4 import S8V4,URDFFrames


def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class RuleDemo:
    def __init__(self,sim,label,options=None):
        if not isinstance(label,str) or not label or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in label):raise ValueError('Unique alphanumeric label required')
        self.sim=sim;self.out=HERE.parent/'outputs/rule_demos'/label;self.out.mkdir(parents=True,exist_ok=False)
        defaults={'pitch_deg':20.,'palm_offset_m':[.096,.027,.108],
            'thumb':[1.07,.4,1.535],'index_distal':1.55,'index_start':0.,'index_limit':.85,
            'close_increment_rad':.004,'preload_rad':.05,'contact_threshold_N':.01,
            'servo_integral_gain':.12,'servo_correction_limit_m':.025,'lift_m':.10}
        if options:
            if set(options)-set(defaults):raise ValueError('Unknown teacher parameter')
            defaults.update(options)
        self.cfg=defaults;self.phase='initial';self.finished=False;self.failed=None;self.index=0
        self.frames=[];self.evals=[];self.stages=[];self.correction=np.zeros(3);self.last_desired=None
        self.record=(self.out/'control.jsonl').open('w',buffering=1);self.eval_record=(self.out/'evaluation.jsonl').open('w',buffering=1)
        sim.reset();self.home_quat=np.array(sim.values(sim.wrist.get_quat()));R=Rotation.from_quat(self.home_quat[[1,2,3,0]])
        self.grasp_rotation=R*Rotation.from_euler('y',self.cfg['pitch_deg'],degrees=True)
        q=self.grasp_rotation.as_quat();self.grasp_quat=q[[3,0,1,2]]
        self.open_hand=np.array([*self.cfg['thumb'],0,self.cfg['index_start'],self.cfg['index_distal'],0,.087,0,.087,0,.087],float)
        if np.any(self.open_hand<sim.lower[6:]) or np.any(self.open_hand>sim.upper[6:]):raise ValueError('Teacher preshape exceeds joint limits')
        self.hand=self.open_hand.copy();self.registration=None
        self.fk=URDFFrames(HERE/'generated/ur7e_xhand.urdf')
        write(self.out/'manifest.json',{'schema':'ur7e_xhand_rule_teacher_v1','driver':'deterministic_rule_controller',
            'parameters':self.cfg,'camera_source':sim.camera_source,'camera_views':sim.camera_views,
            'robot_joint_order':sim.names,'control_dt_s':.025,'physics_dt_s':.005,
            'scene':'same UR7e/XHand scene, cube .045m rho500 friction .8; unchanged PD/torque limits and collisions',
            'perception':'orange RGB mask + front RGB-D; table height .20m known; no runtime cube pose in teacher feedback',
            'evaluation':'privileged cube pose and solver contacts for independent scoring and forbidden-contact stop only',
            'source_sha256':{str(p):sha(p) for p in [Path(__file__),HERE/'environment.py',HERE/'generated/ur7e_xhand.urdf',HERE/'cameras.py']}})
        self.capture_stage('initial');self.generator=self.run()

    def observe(self):
        return {'phase':self.phase,'finished':self.finished,'fault':self.failed,'steps':self.sim.steps,
                'run_dir':str(self.out),'image_dir':str(getattr(self,'current_folder',self.out)),
                'tactile':self.sim.observe()['tactile']}

    def capture_stage(self,name):
        self.phase=name
        S8V4.capture(self)
        o=self.sim.observe(full=True);j=o['joint_positions_rad']
        palm=np.eye(4);palm[:3,:3]=Rotation.from_quat(np.array(o['wrist_quaternion_wxyz'])[[1,2,3,0]]).as_matrix();palm[:3,3]=o['wrist_position_m']
        tool=palm@self.fk.transform('right_hand_link','tool0',j)
        entry={'name':name,'step':self.sim.steps,'image_dir':str(self.current_folder),'observation':o,
            'sent_joint_target_rad':self.sim.target.tolist(),'palm_pose_world':palm.tolist(),'tool0_pose_world':tool.tolist()}
        self.stages.append(entry);write(self.out/'stages.json',self.stages);self.index+=1
        return entry

    def tactile_support(self):
        tactile=self.sim.observe()['tactile']
        return [tactile[f]['peak_force_N']>=self.cfg['contact_threshold_N'] for f in ('thumb','index')]

    def tick(self,pos,quat,hand):
        s=self.sim;before=s.observe(full=True)
        actual=np.array(before['wrist_position_m']);error=np.asarray(pos)-actual
        # Cartesian integral correction compensates simulation PD gravity error.
        # It is bounded, reset only at the initial episode, and never writes state.
        self.correction=np.clip(self.correction+self.cfg['servo_integral_gain']*error,
                                -self.cfg['servo_correction_limit_m'],self.cfg['servo_correction_limit_m'])
        q,ikerr=s.robot.inverse_kinematics(s.wrist,pos=np.asarray(pos)+self.correction,quat=quat,
            dofs_idx_local=s.arm_indices,return_error=True,max_solver_iters=100,seed=42)
        ikerr=s.array(ikerr).reshape(-1)
        if np.linalg.norm(ikerr[:3])>.003 or np.linalg.norm(ikerr[3:])>.03:raise ValueError('Teacher IK failed')
        target=s.array(q).reshape(-1)[s.indices];target[6:]=hand
        if np.any(target<s.lower-1e-6) or np.any(target>s.upper+1e-6):raise ValueError('Teacher joint limit')
        target=np.clip(target,s.lower,s.upper)
        s.render_enabled=False;s.step({'actions':dict(zip(s.names,((target-s.offset)/s.scale).tolist())),'steps':1})
        after=s.observe(full=True)
        self.record.write(json.dumps({'step':s.steps,'phase':self.phase,'before':before,'sent_target_rad':target.tolist(),
            'desired_palm_world_m':np.asarray(pos).tolist(),'desired_quaternion_wxyz':np.asarray(quat).tolist(),
            'cartesian_correction_m':self.correction.tolist(),'after':after})+'\n')
        e=s.evaluate();r=Rotation.from_quat(np.array(e['cube_quaternion_wxyz'])[[1,2,3,0]]).as_matrix()
        e.update(step=s.steps,phase=self.phase,bottom_clearance_m=float(e['cube_position_m'][2]-.0225*np.abs(r[2]).sum()-.20))
        self.evals.append(e);self.eval_record.write(json.dumps(e)+'\n')
        bad=[c for c in e['solver_contacts'] if (c['entity_a']==c['entity_b']=='ur7e_xhand') or 'table' in (c['entity_a'],c['entity_b'])]
        if bad:raise ValueError('Robot self/table contact: '+str([(c['link_a'],c['link_b']) for c in bad]))
        if s.steps%12==0:
            s.render_enabled=True;s.render();s.render_enabled=False
            folder=self.out/'frames';folder.mkdir(exist_ok=True)
            for name in ('scene_front','scene_left','wrist'):(folder/f'{s.steps:06d}_{name}.jpg').write_bytes(s.image_bytes[name])
        self.last_desired=np.asarray(pos).copy()

    def trajectory(self,phase,pos,quat,hand,steps=None):
        self.phase=phase;s=self.sim;start=np.array(s.values(s.wrist.get_pos()));q0=np.array(s.values(s.wrist.get_quat()));h0=s.target[6:].copy()
        rotations=Rotation.from_quat(np.stack([q0[[1,2,3,0]],np.asarray(quat)[[1,2,3,0]]]))
        interp=Slerp([0,1],rotations)
        n=max(40,math.ceil(np.linalg.norm(np.asarray(pos)-start)/.001),math.ceil(np.max(np.abs(hand-h0))/.01),steps or 0)
        for i in range(1,n+1):
            u=i/n;f=u*u*(3-2*u);q=interp(f).as_quat();self.tick(start+f*(np.asarray(pos)-start),q[[3,0,1,2]],h0+f*(hand-h0));yield
        for _ in range(30):self.tick(pos,quat,hand);yield

    def run(self):
        s=self.sim
        # Let the block settle physically before registration.
        pos=np.array(s.values(s.wrist.get_pos()))
        self.phase='settle'
        for _ in range(25):self.tick(pos,self.home_quat,s.home[6:]);yield
        self.capture_stage('registered')
        rgb=np.asarray(Image.open(self.current_folder/'scene_front.jpg'))
        mask=(rgb[:,:,0]>180)&(rgb[:,:,1]>55)&(rgb[:,:,1]<160)&(rgb[:,:,2]<70)
        yy,xx=np.where(mask)
        if len(xx)<30:raise ValueError('Orange block not visible')
        roi=[int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)];x1,y1,x2,y2=roi
        cloud=self.cloud_image[y1:y2,x1:x2][self.mask[y1:y2,x1:x2]]
        cloud=cloud[np.isfinite(cloud).all(1)&(cloud[:,2]>.212)&(cloud[:,2]<.40)]
        if len(cloud)<40:raise ValueError('RGB-D object cloud insufficient')
        top=cloud[cloud[:,2]>np.percentile(cloud[:,2],95)-.003]
        xy=(np.percentile(top[:,:2],5,axis=0)+np.percentile(top[:,:2],95,axis=0))/2
        self.anchor=np.r_[xy,(np.median(top[:,2])+.20)/2]
        self.registration={'roi':roi,'anchor_world_m':self.anchor.tolist(),'surface_points':len(cloud),'source':'RGB mask plus measured visible depth'}
        write(self.out/'registration.json',self.registration);np.save(self.out/'observed_object_cloud.npy',cloud)
        self.align_pos=self.anchor+np.asarray(self.cfg['palm_offset_m'])
        pre=self.align_pos+np.array([0,0,.12]);high=pre.copy();high[2]=max(pos[2],pre[2])
        yield from self.trajectory('preshape',pos,self.grasp_quat,self.hand)
        self.capture_stage('open')
        yield from self.trajectory('approach',high,self.grasp_quat,self.hand)
        yield from self.trajectory('approach',pre,self.grasp_quat,self.hand)
        self.capture_stage('approached')
        yield from self.trajectory('align',self.align_pos,self.grasp_quat,self.hand)
        if np.linalg.norm(np.array(s.values(s.wrist.get_pos()))-self.align_pos)>.003:raise ValueError('Actual alignment outside 3 mm tolerance')
        self.capture_stage('aligned')
        self.phase='close';confirmed=0
        for _ in range(300):
            support=self.tactile_support()
            if all(support):
                confirmed+=1
                # Hold the command on contact rather than sweeping through the cube.
                if confirmed>=5:break
            else:
                confirmed=0;self.hand[4]=min(self.hand[4]+self.cfg['close_increment_rad'],self.cfg['index_limit'])
            self.tick(self.align_pos,self.grasp_quat,self.hand);yield
        else:raise ValueError('No persistent thumb/index contact at bounded closure')
        self.capture_stage('light_contact')
        # Add bounded preload after both contact signals have persisted.
        self.phase='preload';goal=self.hand.copy();goal[4]=min(goal[4]+self.cfg['preload_rad'],self.cfg['index_limit'])
        start=self.hand.copy()
        for i in range(1,31):
            self.hand=start+(goal-start)*i/30;self.tick(self.align_pos,self.grasp_quat,self.hand);yield
        for _ in range(30):self.tick(self.align_pos,self.grasp_quat,self.hand);yield
        if not all(self.tactile_support()):raise ValueError('Lost contact after preload')
        self.capture_stage('closed')
        self.phase='test_lift';test=self.align_pos+np.array([0,0,.02])
        yield from self.trajectory('test_lift',test,self.grasp_quat,self.hand,steps=80)
        self.capture_stage('test_lifted')
        if not all(self.tactile_support()):raise ValueError('Lost tactile support after test lift')
        lifted=self.align_pos+np.array([0,0,self.cfg['lift_m']])
        yield from self.trajectory('lift',lifted,self.grasp_quat,self.hand,steps=160)
        self.capture_stage('lifted');self.phase='hold'
        for _ in range(120):self.tick(lifted,self.grasp_quat,self.hand);yield
        self.capture_stage('held')

    def advance(self,count=12):
        if self.finished:return self.observe()
        try:
            for _ in range(count):next(self.generator)
        except StopIteration:
            self.finished=True;self.finalize()
        except Exception as exc:
            self.failed=str(exc);self.finished=True
            self.capture_stage('failed');self.finalize()
        return self.observe()

    def finalize(self):
        hold=[e for e in self.evals if e['phase']=='hold'];success=bool(len(hold)==120 and min(e['bottom_clearance_m'] for e in hold)>=.05 and not self.failed)
        score={'success':success,'fault':self.failed,'control_steps':len(self.evals),'hold_seconds':len(hold)*.025,
            'hold_min_bottom_clearance_m':min([e['bottom_clearance_m'] for e in hold],default=None),
            'max_bottom_clearance_m':max([e['bottom_clearance_m'] for e in self.evals],default=0),
            'no_attachment_or_object_pose_commands':True,'simulated_success_only':True}
        self.score=score;write(self.out/'score.json',score)
        if success:
            stages={e['name']:e for e in self.stages};closed=stages['closed'];aligned=stages['aligned'];light=stages['light_contact']
            ref={'schema':'ur7e_xhand_genesis_success_reference_v1','review_status':'simulation_verified',
                'run_dir':str(self.out),'validation':score,'registration':self.registration,
                'reference_anchor_world_m':self.anchor.tolist(),'align_palm_pose_world':aligned['palm_pose_world'],
                'align_tool0_pose_world':aligned['tool0_pose_world'],'desired_align_palm_world_m':self.align_pos.tolist(),
                'grasp_quaternion_wxyz':self.grasp_quat.tolist(),'open_hand_target_rad':self.open_hand.tolist(),
                'light_hand_target_rad':light['sent_joint_target_rad'][6:],'closed_hand_target_rad':closed['sent_joint_target_rad'][6:],
                'hand_joint_order':HAND_JOINTS,'arm_joint_order':ARM_JOINTS,'camera_source':self.sim.camera_source,
                'not_hardware_calibration':True,'files':{n:sha(self.out/n) for n in ['manifest.json','stages.json','control.jsonl','evaluation.jsonl','score.json']}}
            write(self.out/'reference.json',ref)
        self.record.close();self.eval_record.close();self.sim.render_enabled=True;self.sim.render()
