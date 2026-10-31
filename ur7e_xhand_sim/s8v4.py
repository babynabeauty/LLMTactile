"""Genesis adaptation of vlm-policy S8V4; no hardware imports or device connection.

Reuses released XHandContactPrior, RuleSupportController and SkillAct. Nominal
palm alignment is constructed from visible tabletop RGB-D and pad geometry;
it is NOT a verified real-world demonstration or a replay of an old success.
"""
from contact_guard import assess_contacts
from pathlib import Path
from typing import Literal
import sys, types, json, time, hashlib, shutil
import numpy as np
import torch
from pydantic import Field, model_validator
from scipy.spatial.transform import Rotation
from build_assets import HERE, ARM_JOINTS, HAND_JOINTS

VLM_ROOT=Path('/home/sai/zsq/vlm-policy')
GENDEX=Path('/home/sai/zsq/bench2dex-work/third_party/GenDexGrasp')
if 'vlm_policy_local' not in sys.modules:
    package=types.ModuleType('vlm_policy_local');package.__path__=[str(VLM_ROOT)]
    sys.modules['vlm_policy_local']=package
from vlm_policy_local.real_variants.prior import XHandContactPrior
from vlm_policy_local.real_variants.config import SkillAct as ProjectSkillAct
from vlm_policy_local.tactile_policy.runtime import RuleSupportController
from vlm_policy_local.urdf_frames import URDFFrames


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False))

class SkillAct(ProjectSkillAct):
    """Local, backwards-compatible bounded preload choice for shape studies."""
    preload_increment_rad: float = Field(default=0., ge=0., le=.20,
        description='Shape-study close only: 0 uses the default; choose .05, .10, .15, or .20 rad. All other skills must use 0.')
    preshape_id: Literal['keep','reference','reference_hold_face',
                         'opposed_mild_hold_face','opposed_wide_hold_face'] = Field(default='keep',
        description='Shape-study locate only: choose an offered hand preshape before GenDex scoring. All other skills use keep; keep at locate preserves reference.')
    grip_gain: float = Field(default=0., ge=0., le=3.,
        description='Shape-study close only: choose common thumb/index PD gain 1, 2, or 3; 0 uses default 1. It persists through lift and hold. Other skills must use 0.')

    @classmethod
    def model_json_schema(cls, *args, **kwargs):
        # The API's strict response schema requires every property to be
        # required. Local legacy payload validation still supplies the default.
        schema = super().model_json_schema(*args, **kwargs)
        schema['required'] = list(schema['properties'])
        return schema

    @model_validator(mode='after')
    def valid_preload(self):
        if self.preload_increment_rad not in (0., .05, .10, .15, .20):
            raise ValueError('preload_increment_rad must be 0, .05, .10, .15, or .20')
        if self.preload_increment_rad and (self.status != 'continue' or self.skill != 'close'):
            raise ValueError('Only the close skill may select a nonzero preload_increment_rad')
        if self.preshape_id != 'keep' and (self.status != 'continue' or self.skill != 'locate'):
            raise ValueError('Only the locate skill may select a preshape_id; other skills require keep')
        if self.grip_gain not in (0.,1.,2.,3.):
            raise ValueError('grip_gain must be 0, 1, 2, or 3')
        if self.grip_gain and (self.status != 'continue' or self.skill != 'close'):
            raise ValueError('Only the close skill may select a nonzero grip_gain')
        return self


class S8V4:
    supports_preload_selection = False
    supports_preshape_selection = False
    supports_grip_gain_selection = False

    def _validate_preload_capability(self, action):
        if action.preload_increment_rad and not self.supports_preload_selection:
            raise ValueError('Nonzero preload_increment_rad requires explicit shape-study S8V4')
        if action.preshape_id != 'keep' and not self.supports_preshape_selection:
            raise ValueError('Selecting preshape_id requires explicit shape-study S8V4')
        if action.grip_gain and not self.supports_grip_gain_selection:
            raise ValueError('Nonzero grip_gain requires explicit shape-study S8V4')

    def __init__(self,sim,label,driver='vlm_policy_CodexAppServerPolicy'):
        if not isinstance(label,str) or not label or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in label):
            raise ValueError('label must be a new alphanumeric run name')
        self.sim=sim;self.out=HERE.parent/'outputs'/'s8v4_genesis'/label
        self.out.mkdir(parents=True,exist_ok=False)
        self.phase='unregistered';self.proposals=None;self.offset=np.zeros(3);self.log=[];self.evaluation=[]
        self.index=0;self.finished=False;self.failed=None
        demo=json.loads((VLM_ROOT/'configs/real_variants/reference.example.json').read_text())
        self.open=np.clip(demo['open_hand_target'],sim.lower[6:],sim.upper[6:])
        self.closed=np.clip(demo['closed_hand_target'],sim.lower[6:],sim.upper[6:])
        # Flex the distal thumb above the table but keep its base spread during
        # descent. A fully opposed thumb lands on the cube before fingers close.
        self.open[1:3]=self.closed[1:3]
        self.open[0]=0.
        self.controller=RuleSupportController(sim.lower[6:],sim.upper[6:],fingers=(0,1),
            threshold=.01,period=1/15,max_speed=.25)
        # Genesis changes torch's default device. The reused prior uses a CPU
        # random generator and numpy outputs, so keep its entire inference on CPU.
        with torch.device('cpu'):
            self.prior=XHandContactPrior(GENDEX,HERE/'generated/ur7e_xhand.urdf',HERE/'sources/xhand/meshes',self.closed,[0]*6)
        fk=URDFFrames(HERE/'generated/ur7e_xhand.urdf')
        joints=dict(zip(HAND_JOINTS,self.closed));joints.update(dict.fromkeys(ARM_JOINTS,0.))
        self.tool_to_palm=fk.transform('tool0','right_hand_link',joints)
        centers=[]
        for layout in sim.layouts[:2]:
            t=fk.transform('right_hand_link',layout['link_name'],joints)
            points=np.asarray(layout['positions_link_m'])@t[:3,:3].T+t[:3,3]
            centers.append(np.median(points,axis=0))
        self.pinch_center=np.mean(centers,axis=0)
        # Keep fingers a little above the object's center initially, avoiding table strike.
        self.clearance=.018
        self.sim.reset();self.home_quat=self.sim.array(sim.wrist.get_quat()).reshape(4)
        self.rotation=Rotation.from_quat(self.home_quat[[1,2,3,0]]).as_matrix()
        self.record=open(self.out/'control.jsonl','w',buffering=1)
        self.eval_record=open(self.out/'evaluation.jsonl','w',buffering=1)
        write(self.out/'manifest.json',{'method':'vlm-policy s8v4 Genesis adaptation',
            'reused_components':['XHandContactPrior','RuleSupportController','SkillAct'],
            'high_level_driver':driver,
            'camera_source':sim.camera_source,'camera_views':sim.camera_views,
            'prior':self.prior.provenance,'source_hashes':{str(p):sha(p) for p in [Path(__file__),VLM_ROOT/'real_variants/prior.py',VLM_ROOT/'tactile_policy/runtime.py',VLM_ROOT/'codex_policy.py',HERE/'environment.py']},
            'nominal_reference':'Candidate demo hand targets; palm alignment from observed partial RGB-D and hand pad FK. Not a verified hardware demonstration.',
            'pinch_center_palm_m':self.pinch_center.tolist(),'open_hand':self.open.tolist(),'closed_hand':self.closed.tolist(),
            'tactile_threshold_nominal_N':.01,'tactile_units':'uncalibrated simulation estimate, not SDK raw units',
            'mass_and_friction':'Uncalibrated existing scene values: rho=500, cube size=.045, friction=.8',
            'success':'rotated cube bottom > table + .05 m for 3 continuous simulation seconds',
            'high_level_clock':'physics pauses while VLM thinks; rule runs on simulation time during skills'})
        self.capture()

    def state(self):
        s=self.sim;o=s.observe(full=True)
        return {'read_monotonic_s':s.steps*s.control_dt_s,'hand_q':[o['joint_positions_rad'][n] for n in HAND_JOINTS],
            # Compatibility field name only. Units remain nominal N and are recorded explicitly.
            'hand_tactile_raw_force':[o['tactile'][l['finger']]['force_sensor_N'] for l in s.layouts]}

    def support(self):
        state=self.state();raw=np.asarray(state['hand_tactile_raw_force']);q=np.asarray(state['hand_q'])
        target=self.controller.sent if self.controller.active else self.sim.target[6:]
        peaks=np.linalg.norm(raw,axis=-1).max(axis=1)
        return {finger:{'tactile_peak_nominal_N':float(peaks[i]),'joint_deflection_rad':float(max(0,np.max(target[js]-q[js]))),
            'supported':bool(peaks[i]>=.01 or np.max(target[js]-q[js])>.04)} for i,finger,js in [(0,'thumb',[1,2]),(1,'index',[4,5])]}

    def capture(self):
        s=self.sim;s.render_enabled=True;s.render()
        folder=self.out/f'view_{self.index:03d}';folder.mkdir(exist_ok=True)
        for name,data in s.image_bytes.items():
            if name!='hand':(folder/f'{name}.jpg').write_bytes(data)
        # Observed RGB-D only; no segmentation IDs or object state.
        pc,mask=s.camera.render_pointcloud(world_frame=True)
        width,height=s.camera.res
        self.cloud_image=s.array(pc).reshape(height,width,3);self.mask=s.array(mask).reshape(height,width)
        view=s.camera_views['scene_front']
        forward=Rotation.from_quat(view['optical_quaternion_xyzw']).as_matrix()[:,2]
        depth=(self.cloud_image-np.asarray(view['position_m']))@forward
        self.mask=self.mask & (depth>view['near_m']) & (depth<view['far_m'])
        np.savez_compressed(folder/'rgbd.npz',world_points=self.cloud_image,valid=self.mask,
            intrinsics=s.camera.intrinsics,transform=s.camera.transform)
        write(folder/'camera_params.json',{name:{'intrinsics':s.array(cam.intrinsics).tolist(),
            'transform':s.array(cam.transform).tolist()} for name,cam in
            [('scene_front',s.camera),('scene_left',s.rear_camera),('wrist',s.wrist_camera)]})
        self.stamp=(s.episode,s.steps);self.current_folder=folder
        return self.observe()

    def observe(self):
        s=self.sim
        return {'observation_id':f'{s.episode}:{s.steps}:{self.index}', 'phase':self.phase,
            'observation':s.observe(),'support':self.support(),'proposals':self.proposals,
            'selected_offset_xyz_m':self.offset.tolist(),
            'alignment_palm_world_m':getattr(self,'alignment',None).tolist() if hasattr(self,'alignment') else None,
            'image_dir':str(self.current_folder),'run_dir':str(self.out),'fault':self.failed}

    def _tick(self,target,phase):
        s=self.sim
        if self.controller.active:
            target=target.copy();target[6:]=self.controller.propose(self.state(),target[6:])
        if hasattr(s,'online_ppo'):s.online_ppo.before(target,phase)
        s.step({'actions':dict(zip(s.names,((target-s.offset)/s.scale).tolist())),'steps':1})
        if hasattr(s,'online_ppo'):
            s.online_ppo.after()
            if hasattr(self,'grip_gain_controller'):
                self.grip_gain=float(s.online_ppo.gain)
                self.grip_gain_controller.update(common_thumb_index_gain=self.grip_gain,
                    source='online PPO' if s.online_ppo.active else 'initial Astra/default',
                    kp_readback=s.array(s.robot.get_dofs_kp(s.indices)).reshape(-1).tolist(),
                    kv_readback=s.array(s.robot.get_dofs_kv(s.indices)).reshape(-1).tolist())
        if self.controller.active:self.controller.acknowledge(target[6:])
        command={'step':s.steps,'phase':phase,'target':target.tolist(),'observation':s.observe(full=True),'local_feedback':self.controller.last_diagnostic}
        if hasattr(self,'_command_metadata'):command.update(self._command_metadata())
        self.record.write(json.dumps(command)+'\n')
        # Privileged data goes only to a separate evaluation stream, never policy observations.
        e=s.evaluate();p=np.asarray(e['cube_position_m']);q=np.asarray(e['cube_quaternion_wxyz']);r=Rotation.from_quat(q[[1,2,3,0]]).as_matrix()
        e.update(step=s.steps,phase=phase,bottom_clearance_m=float(
            e.get('bottom_clearance_m',p[2]-.0225*np.abs(r[2]).sum()-.20)))
        guard=assess_contacts(e['solver_contacts'],getattr(s,'table_contact_force_limit_N',5.0))
        e['contact_guard']=guard
        self.evaluation.append(e);self.eval_record.write(json.dumps(e)+'\n')
        if guard['stop']:
            write(self.out/'contact_guard_stop.json',dict(step=s.steps,phase=phase,**guard))
            raise ValueError(guard['reason'])
        if s.steps%20==0:
            s.render_enabled=True;s.render();s.render_enabled=False
            (self.out/'frames').mkdir(exist_ok=True)
            (self.out/'frames'/f'{s.steps:06d}.jpg').write_bytes(s.image_bytes['scene'])

    def move(self,position,hand,phase,seconds=None):
        s=self.sim;start=s.array(s.wrist.get_pos()).reshape(3);hand0=s.target[6:].copy()
        distance=np.linalg.norm(position-start)
        count=max(20,int(np.ceil(distance/.0015)),int(np.ceil(np.max(np.abs(hand-hand0))/.012)))
        if seconds is not None:count=max(count,int(seconds/self.sim.control_dt_s))
        s.render_enabled=False
        try:
            for i in range(1,count+1):
                fraction=i/count;pos=start+(position-start)*fraction
                q,err=s.robot.inverse_kinematics(s.wrist,pos=pos,quat=self.home_quat,
                    dofs_idx_local=s.arm_indices,return_error=True,max_solver_iters=80,seed=42)
                err=s.array(err).reshape(-1)
                if np.linalg.norm(err[:3])>.004 or np.linalg.norm(err[3:])>.03:raise ValueError('IK failed along skill path')
                target=s.array(q).reshape(-1)[s.indices];target[6:]=hand0+(hand-hand0)*fraction
                self._tick(target,phase)
            for _ in range(12):self._tick(target,phase)
        finally:s.render_enabled=True

    def execute(self,payload):
        if self.finished:raise ValueError('Trial finished; start a new unique run')
        action=SkillAct.model_validate(payload)
        self._validate_preload_capability(action)
        expected=self.observe()['observation_id']
        if action.observation_id!=expected or self.stamp!=(self.sim.episode,self.sim.steps):raise ValueError('Stale observation; refresh policy observation')
        if self.failed:raise ValueError('Trial stopped after execution fault')
        entry={'action':action.model_dump(),'before':self.observe()};result={}
        try:
            if action.status!='continue':
                self.finished=True;self.phase=action.status;result={'status':'model_'+action.status}
            elif action.skill=='locate':
                if self.phase!='unregistered':raise ValueError('Already registered')
                x1,y1,x2,y2=action.object_roi_xyxy
                height,width=self.cloud_image.shape[:2]
                if not(0<=x1<x2<=width and 0<=y1<y2<=height):raise ValueError('ROI outside scene image')
                cloud=self.cloud_image[y1:y2,x1:x2][self.mask[y1:y2,x1:x2]]
                cloud=cloud[np.isfinite(cloud).all(1)&(cloud[:,2]>.212)&(cloud[:,2]<.40)]
                if len(cloud)<40 or np.ptp(cloud,axis=0).max()>.15:raise ValueError('Insufficient/unbounded observed object surface')
                top=cloud[cloud[:,2]>np.percentile(cloud[:,2],95)-.003]
                # Upright tabletop object assumption, inferred visible top and calibrated table.
                xy=(np.percentile(top[:,:2],5,axis=0)+np.percentile(top[:,:2],95,axis=0))/2
                self.anchor=np.r_[xy,(np.median(top[:,2])+.20)/2]
                self.alignment=self.anchor-self.rotation@self.pinch_center;self.alignment[2]+=self.clearance
                palm=np.eye(4);palm[:3,:3]=self.rotation;palm[:3,3]=self.alignment
                tool=palm@np.linalg.inv(self.tool_to_palm)
                toolpose=np.r_[tool[:3,3],Rotation.from_matrix(tool[:3,:3]).as_rotvec()]
                with torch.device('cpu'):
                    self.proposals=self.prior.propose(cloud,toolpose,.20)
                np.save(self.out/'observed_object_cloud.npy',cloud)
                write(self.out/'registration.json',{'roi':action.object_roi_xyxy,'surface_count':len(cloud),'estimated_center_world_m':self.anchor.tolist(),'nominal_palm_world_m':self.alignment.tolist(),'proposals':self.proposals})
                self.phase='located';result={'status':'located','point_count':len(cloud)}
            else:
                allowed={'approach':['located'],'align':['approached'],'close':['aligned'],'test_lift':['contact'],'lift':['test_lifted'],'hold':['lifted','holding']}
                if self.phase not in allowed.get(action.skill,[]):raise ValueError('Skill invalid in phase '+self.phase)
                if action.skill=='approach':
                    if not any(np.allclose(action.offset_xyz_m,c['offset_xyz_m'],atol=1e-8) for c in self.proposals['candidates']):raise ValueError('Select a supplied GenDex offset')
                    self.offset=np.array(action.offset_xyz_m)
                elif action.skill in ('align','close') and not np.allclose(action.offset_xyz_m,self.offset):raise ValueError('Preserve selected offset')
                position=self.alignment+self.offset
                if action.skill=='approach':
                    # Move horizontally above the workpiece before descending.
                    high=position.copy();high[2]=max(self.sim.values(self.sim.wrist.get_pos())[2],position[2]+.10)
                    self.move(high,self.open,'approach');position=position.copy();position[2]+=.10
                    self.move(position,self.open,'approach');self.phase='approached'
                elif action.skill=='align':self.move(position,self.open,'align');self.phase='aligned'
                elif action.skill=='close':
                    self.move(position,self.closed,'close');self.phase='contact'
                    if all(x['supported'] for x in self.support().values()):self.controller.activate(self.sim.target[6:])
                    else:result['warning']='No two-finger support; inspect before lifting'
                elif action.skill in ('test_lift','lift'):
                    if not all(x['supported'] for x in self.support().values()):raise ValueError('No two-finger support; lift blocked')
                    pos=self.sim.array(self.sim.wrist.get_pos()).reshape(3);pos[2]+=.02 if action.skill=='test_lift' else .08
                    self.move(pos,self.sim.target[6:],action.skill);self.phase='test_lifted' if action.skill=='test_lift' else 'lifted'
                elif action.skill=='hold':
                    self.sim.render_enabled=False
                    try:
                        for _ in range(round(3/self.sim.control_dt_s)):self._tick(self.sim.target.copy(),'hold')
                    finally:self.sim.render_enabled=True
                    self.phase='holding'
                result['status']='executed'
        except Exception as exc:
            self.failed=str(exc);result={'status':'failed','error':str(exc)}
        self.index+=1;after=self.capture();entry.update(result=result,after=after)
        self.log.append(entry);write(self.out/'interactions.json',self.log)
        return {'receipt':result,'policy_observation':after}

    def score(self):
        hold_samples=round(3/self.sim.control_dt_s)
        hold=[e for e in self.evaluation if e['phase']=='hold']
        consecutive=best=0
        for e in self.evaluation:
            consecutive=consecutive+1 if e['bottom_clearance_m']>=.05 else 0;best=max(best,consecutive)
        score={'success':bool(len(hold)>=hold_samples and min(e['bottom_clearance_m'] for e in hold[-hold_samples:])>=.05 and not self.failed),
            'continuous_clearance_above_5cm_s':best*self.sim.control_dt_s,'max_bottom_clearance_m':max([e['bottom_clearance_m'] for e in self.evaluation],default=0),
            'hold_min_bottom_clearance_m':min([e['bottom_clearance_m'] for e in hold[-hold_samples:]],default=None),
            'hold_required_samples':hold_samples,'control_dt_s':self.sim.control_dt_s,
            'fault':self.failed,'phase':self.phase,'control_steps':len(self.evaluation),'run_dir':str(self.out),
            'evaluation_only':True,'not_hardware_or_tactile_benefit_validation':True}
        write(self.out/'score.json',score);return score
