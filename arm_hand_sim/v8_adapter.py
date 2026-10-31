"""Stage4 v8 control structure adapted to Panda/Wuji; no LEAP demo replay."""
import hashlib
import json
import time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
SOURCE = Path('/home/sai/zsq/vlm-policy/experiments/stage4/native_source_v8/bench2dex_soap.py')


class V8Adapter:
    def __init__(self, sim):
        self.s = sim
        self.run = None
        self.turn = 0
        self.protected = {}
        self.last_end = time.monotonic()

    def perceive(self, payload):
        s=self.s
        clouds=[]
        for cam in (s.camera,s.close_camera):
            rgb=s.array(cam.render(rgb=True)[0]).reshape(cam.res[1],cam.res[0],3).astype(float)
            pc,valid=cam.render_pointcloud()
            pc=np.asarray(pc).reshape(*rgb.shape[:2],3)
            valid=np.asarray(valid).reshape(rgb.shape[:2])
            mask=valid & (rgb[...,0]>100) & (rgb[...,0]>rgb[...,1]*1.6) & (rgb[...,1]>rgb[...,2]*1.5)
            points=pc[mask]
            points=points[np.isfinite(points).all(axis=1)]
            if len(points)>10:clouds.append(points)
        if not clouds:return {'valid':False,'reason':'orange cube not visible'}
        points=np.concatenate(clouds)
        low,high=np.quantile(points,[.01,.99],axis=0)
        result={'valid':True,'surface_bounds_m':[low.tolist(),high.tolist()],
                'surface_center_m':((low+high)/2).tolist(),'points':len(points),
                'source':'two RGB-D cameras, orange segmentation; no object state'}
        if self.run:
            np.savez_compressed(self.run/f'perception_{self.turn:02d}.npz',points=points)
        return result

    def start(self,payload):
        if self.run:raise ValueError('An experiment is already active')
        self.run=HERE/'runs'/time.strftime('%Y%m%d_%H%M%S_stage4_v8_wuji')
        self.run.mkdir(parents=True,exist_ok=False)
        (self.run/'frames').mkdir()
        self.turn=0
        manifest={'scheme':'stage4 v8 adaptation','max_turns':20,
          'source':str(SOURCE),'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
          'retained':'smooth skills; alignment contact guard; per-finger joint relief; fixed wrist closure; test lift; hold',
          'changed':'Panda/Wuji geometric preshape replaces embodiment-specific LEAP/soap demonstration',
          'controller':'current conversation; one decision per fresh feedback',
          'object_state':'only independent evaluation file, never policy observation',
          'start_observation':self.s.observe()}
        (self.run/'manifest.json').write_text(json.dumps(manifest,indent=2))
        self.s.render()
        self.snapshot()
        return {'run_dir':str(self.run),'max_turns':20,'observation':self.s.observe()}

    def snapshot(self):
        s=self.s
        (self.run/'frames'/f'{s.steps:06d}.jpg').write_bytes(s.image_bytes['scene'])
        (self.run/'frames'/f'{s.steps:06d}_hand.jpg').write_bytes(s.image_bytes['hand'])

    def skill(self,p):
        s=self.s
        if self.run is None:raise ValueError('Start the experiment first')
        if self.turn>=20:raise ValueError('20-turn decision budget exhausted')
        if p.get('expected_step')!=s.steps:raise ValueError('Fresh expected_step required')
        kind=p.get('skill')
        if kind not in ('approach','align','close_fixed','test_lift','retreat','hold','reopen'):
            raise ValueError('Unknown v8 skill')
        duration=float(p.get('seconds',2.0))
        if not np.isfinite(duration) or not .25<=duration<=8:raise ValueError('Duration must be .25..8 seconds')
        q0=s.array(s.robot.get_dofs_position(s.indices)).reshape(-1)
        goal=s.target.copy()
        if 'position_m' in p:
            if kind in ('close_fixed','hold','reopen'):raise ValueError('Fixed-wrist skill cannot move arm')
            pos=np.array(p['position_m'],dtype=float)
            if pos.shape!=(3,) or not np.isfinite(pos).all() or not(-.1<=pos[0]<=.85 and -.4<=pos[1]<=.4 and .24<=pos[2]<=.95):
                raise ValueError('Pose outside workspace')
            quat=np.array(p.get('quaternion_wxyz',s.values(s.wrist.get_quat())),dtype=float)
            if quat.shape!=(4,) or not np.isfinite(quat).all() or abs(np.linalg.norm(quat)-1)>.001:raise ValueError('Invalid quaternion')
            q,err=s.robot.inverse_kinematics(s.wrist,pos=pos,quat=quat,dofs_idx_local=s.arm_indices,
                return_error=True,max_solver_iters=100,seed=42)
            err=s.array(err).reshape(-1)
            if np.linalg.norm(err[:3])>.003 or np.linalg.norm(err[3:])>.03:raise ValueError('IK did not converge')
            goal[:7]=s.array(q).reshape(-1)[s.indices][:7]
        for name,value in p.get('hand_rad',{}).items():
            if name not in s.hand_names:raise ValueError('Unknown finger joint')
            i=s.names.index(name)
            if not np.isfinite(value) or not s.lower[i]+.01<=value<=s.upper[i]-.01:raise ValueError('Joint target out of interior bounds')
            goal[i]=value
        if kind in ('close_fixed','reopen'):goal[:7]=q0[:7]
        if kind in ('approach','reopen'):self.protected={}
        duration=max(duration,1.5*float(np.max(np.abs(goal[:7]-q0[:7])))/.8)
        if duration>12:raise ValueError('Arm displacement too large')
        self.turn+=1
        before=s.observe()
        (self.run/f'turn_{self.turn:02d}_before.json').write_text(json.dumps(before))
        events=[];guard=None;sustained=0;start=s.steps
        waiting=time.monotonic()-self.last_end
        n=max(1,round(duration/.025))
        h0=s.target.copy()
        for t in range(n+30):
            a=min((t+1)/n,1.);a=a*a*(3-2*a)
            target=q0+a*(goal-q0)
            target[7:]=h0[7:]+a*(goal[7:]-h0[7:])
            for indices,values in self.protected.values():target[indices]=values
            if guard:target[:7]=hold_arm
            s.robot.control_dofs_position(target,s.indices)
            for _ in range(5):s.scene.step()
            s.steps+=1
            actual=s.array(s.robot.get_dofs_position(s.indices)).reshape(-1)
            if not np.isfinite(actual).all():
                s.fault='Nonfinite joint state';break
            margin=np.minimum(actual-s.lower,s.upper-actual)
            danger=(margin<0)|((margin<.03)&(np.abs(target-actual)>.04))
            for finger in ('thumb','index','middle','ring','pinky'):
                indices=np.array([i for i,name in enumerate(s.names) if name.startswith('r_') and finger in name])
                if danger[indices].any():
                    relieved=actual[indices]+np.clip(-actual[indices],-.025,.025)
                    relieved=np.clip(relieved,s.lower[indices]+.01,s.upper[indices]-.01)
                    self.protected[finger]=(indices,relieved)
                    target[indices]=relieved
                    events.append({'step':s.steps,'finger':finger,'response':'relieve toward open preshape'})
            deviation=float(np.max(np.abs(target[7:]-actual[7:])))
            sustained=sustained+1 if deviation>.10 else 0
            if kind=='align' and sustained>=3 and guard is None:
                guard={'step':s.steps,'finger_tracking_error_rad':deviation}
                hold_arm=actual[:7].copy()
                target[:7]=hold_arm
            s.target=target
            s.actions=((target-s.offset)/s.scale).tolist()
            obs=s.observe(full=True)
            with (self.run/'trace.jsonl').open('a') as f:
                f.write(json.dumps({'turn':self.turn,'target_rad':target.tolist(),'observation':obs})+'\n')
            # Ground truth never included in action/observation responses.
            with (self.run/'evaluation_only.jsonl').open('a') as f:
                f.write(json.dumps({'turn':self.turn,'step':s.steps,'time_s':s.steps*.025,
                    'cube_position_m':s.values(s.cube.get_pos()),'cube_quaternion_wxyz':s.values(s.cube.get_quat()),
                    'wrist_position_m':s.values(s.wrist.get_pos())})+'\n')
            if t%10==0 or t==n+29:
                s.render();self.snapshot()
            if guard and t>=n:break
        s.robot.control_dofs_position(s.target,s.indices)
        s.render();self.snapshot()
        result={'turn':self.turn,'remaining_turns':20-self.turn,'skill':kind,'contact_guard':guard,
                'joint_guard_events':len(events),'simulation_duration_s':(s.steps-start)*.025,
                'decision_wait_s':waiting,'observation':s.observe(),'run_dir':str(self.run)}
        (self.run/f'turn_{self.turn:02d}.json').write_text(json.dumps({'action':p,'result':result,'guard_events':events},indent=2))
        self.last_end=time.monotonic()
        return result

    def evaluate(self,p):
        rows=[json.loads(line) for line in (self.run/'evaluation_only.jsonl').read_text().splitlines()]
        from scipy.spatial.transform import Rotation
        for row in rows:
            q=row['cube_quaternion_wxyz']
            r=Rotation.from_quat([q[1],q[2],q[3],q[0]]).as_matrix()
            row['bottom_clearance_m']=row['cube_position_m'][2]-.0225*np.abs(r[2]).sum()-.20
        last=rows[-1]['time_s']
        tail=[r for r in rows if r['time_s']>=last-2]
        clearance=min(r['bottom_clearance_m'] for r in tail)
        offsets=np.array([np.array(r['cube_position_m'])-r['wrist_position_m'] for r in tail])
        drift=float(np.max(np.linalg.norm(offsets-offsets[0],axis=1)))
        report={'turns':self.turn,'max_turns':20,'last_2s_min_bottom_clearance_m':clearance,
                'last_2s_relative_drift_m':drift,'success':bool(clearance>=.05 and drift<=.02),
                'criterion':'cube bottom >=5 cm above table for final2s, wrist-relative drift <=2cm',
                'max_bottom_clearance_m':max(r['bottom_clearance_m'] for r in rows)}
        (self.run/'evaluation.json').write_text(json.dumps(report,indent=2))
        return report
