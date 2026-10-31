"""200 Hz tactile monitoring and causal reactive grip experiment, simulation only."""
import argparse
from collections import deque
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import replay_material_sweep as replay

CASES={
 'mass200_monitor':{'cube_mass_kg':.2,'cube_friction':.8,'robot_friction':.8},
 'lowmu_monitor':{'cube_density':500.,'cube_friction':.01,'robot_friction':.01},
 'mass100_monitor':{'cube_mass_kg':.1,'cube_friction':.8,'robot_friction':.8},
 'mass200_reactive':{'cube_mass_kg':.2,'cube_friction':.8,'robot_friction':.8},
 'lowmu_reactive':{'cube_density':500.,'cube_friction':.01,'robot_friction':.01},
}
CONFIG={'sensor_hz':200,'command_hz':40,'window_s':.05,'centroid_speed_threshold_m_s':.020,
 'signal_drop_threshold':.25,'min_signal_N':.01,'debounce_samples':3,
 'response_kp_multiplier':2.,'response_ramp_s':.05,
 'gt_slip_threshold_m_s':.005,'gt_slip_persistence_samples':3,
 'gt_note':'Backward finite difference of material-point relative tangential velocity at current solver contact; evaluation only, not a hardware slip label.'}

class TactileTrigger:
    """Input boundary: measured tactile + static probe coordinates + armed phase only."""
    def __init__(self,positions):self.positions=positions;self.history=deque(maxlen=11);self.count=0;self.triggered=False
    def update(self,force,armed):
        w=np.linalg.norm(force[:2],axis=-1);total=w.sum(-1)
        center=(w[:,:,None]*self.positions).sum(1)/np.maximum(total[:,None],1e-10)
        peak=w.max(-1);active=(w>1e-5).sum(-1)
        self.history.append((total,center));speed=np.zeros(2);drop=np.zeros(2);candidate=False
        if len(self.history)==11 and armed:
            old,c0=self.history[0];speed=np.linalg.norm(center-c0,axis=-1)/.05
            drop=1-total/np.maximum(old,1e-10)
            valid=(old>=CONFIG['min_signal_N'])&(total>=CONFIG['min_signal_N'])
            candidate=bool(np.any(valid&((speed>CONFIG['centroid_speed_threshold_m_s'])|(drop>CONFIG['signal_drop_threshold']))))
        self.count=self.count+1 if candidate else 0
        event=not self.triggered and self.count>=CONFIG['debounce_samples']
        if event:self.triggered=True
        return {'total_N':total.tolist(),'peak_N':peak.tolist(),'active_taxels':active.tolist(),
            'centroid_link_m':center.tolist(),'centroid_speed_m_s':speed.tolist(),'drop_fraction':drop.tolist(),
            'armed':bool(armed),'candidate':candidate,'event':event,'debounce':self.count}

def run_case(out,case):
    from scipy.spatial.transform import Rotation
    from environment import Sandbox
    rows=[json.loads(x) for x in (replay.DEMO/'control.jsonl').read_text().splitlines()]
    armed_step=max(i+1 for i,r in enumerate(rows) if r['phase']=='preload')
    reactive=case.endswith('_reactive')
    instances=[]
    class SampledSandbox(Sandbox):
        def __init__(self,**kw):
            super().__init__(**kw)
            self.sample_idx=0;self.trigger_sample=None;self.multiplier=1.
            self.base_kp=np.array(self.kp);self.base_kv=np.array(self.kv)
            self.links=[self.robot.get_link(l['link_name']) for l in self.layouts]
            self.trigger=TactileTrigger(np.array([l['positions_link_m'] for l in self.layouts[:2]]))
            self.prev_poses=self.poses();self.inputs=[];self.labels=[]
            self.log=(out/'highfreq_events.jsonl').open('w',buffering=1)
            instances.append(self)
        def poses(self):
            ids=[l.idx_local for l in self.links]
            pp=self.array(self.robot.get_links_pos(ids)).reshape(5,3)
            qq=self.array(self.robot.get_links_quat(ids)).reshape(5,4)
            result={}
            for l,p,q in zip(self.links,pp,qq):result[l.idx]=(p,Rotation.from_quat(q[[1,2,3,0]]).as_matrix())
            q=np.array(self.values(self.cube.get_quat()));result[self.cube.base_link.idx]=(np.array(self.values(self.cube.get_pos())),Rotation.from_quat(q[[1,2,3,0]]).as_matrix())
            return result
        def evaluation_labels(self):
            now=self.poses();cubeid=self.cube.base_link.idx;cp,cr=now[cubeid]
            raw=self.scene.rigid_solver.collider.get_contacts(as_tensor=True,to_torch=False)
            a=np.asarray(raw['link_a']).reshape(-1);b=np.asarray(raw['link_b']).reshape(-1)
            pts=np.asarray(raw['position']).reshape(-1,3);normals=np.asarray(raw['normal']).reshape(-1,3)
            forces=np.asarray(raw['force']).reshape(-1,3)
            per=[[] for _ in self.links];ids={l.idx:i for i,l in enumerate(self.links)}
            for la,lb,p,n,f in zip(a,b,pts,normals,forces):
                other=lb if la==cubeid else la if lb==cubeid else -1
                if other not in ids:continue
                velocities=[]
                for body in (cubeid,other):
                    pos,rot=now[body];oldpos,oldrot=self.prev_poses[body]
                    previous_point=oldpos+oldrot@(rot.T@(p-pos))
                    velocities.append((p-previous_point)/.005)
                rel=velocities[0]-velocities[1];n=n/max(np.linalg.norm(n),1e-10)
                speed=float(np.linalg.norm(rel-n*np.dot(rel,n)))
                per[ids[other]].append((speed,float(np.linalg.norm(f))))
            self.prev_poses=now
            speed=[];load=[];counts=[]
            for contacts in per:
                weights=np.array([x[1] for x in contacts]);loads=float(weights.sum())
                speed.append(float(np.dot([x[0] for x in contacts],weights)/loads) if loads>.05 else 0.)
                load.append(loads);counts.append(len(contacts))
            return {'cube_position_m':cp.tolist(),'cube_rotation':cr.tolist(),
                'bottom_clearance_m':float(cp[2]-.0225*abs(cr[2]).sum()-.20),
                'contact_tangent_speed_m_s':speed,'contact_force_norm_sum_N':load,'contact_count':counts}
        def step(self,payload):
            actions=np.array([payload['actions'][n] for n in self.names]);self.actions=actions.tolist()
            self.target=self.offset+actions*self.scale;self.robot.control_dofs_position(self.target,self.indices)
            phase=rows[self.steps]['phase']
            for sub in range(5):
                if reactive and self.trigger_sample is not None:
                    u=min(1.,(self.sample_idx+1-self.trigger_sample)/10.)
                    self.multiplier=1+(CONFIG['response_kp_multiplier']-1)*(u*u*(3-2*u))
                    kp=self.base_kp.copy();kv=self.base_kv.copy();kp[6:12]*=self.multiplier;kv[6:12]*=np.sqrt(self.multiplier)
                    self.robot.set_dofs_kp(kp,self.indices);self.robot.set_dofs_kv(kv,self.indices)
                self.scene.step();self.sample_idx+=1
                force=np.stack([self.array(s.read().force).reshape(-1,120,3)[0].copy() for s in self.sensors.values()])
                features=self.trigger.update(force,self.steps>=armed_step)
                if features['event']:
                    self.trigger_sample=self.sample_idx
                    print('TACTILE_EVENT '+json.dumps({'case':case,'time_s':self.sample_idx*.005,'features':features,'reactive':reactive}),flush=True)
                # Evaluation follows the detector and never feeds its decision.
                label=self.evaluation_labels()
                q=self.values(self.robot.get_dofs_position(self.indices))
                self.inputs.append((force,np.array(q),self.target.copy(),self.multiplier))
                self.labels.append(label)
                self.log.write(json.dumps({'sample':self.sample_idx,'time_s':self.sample_idx*.005,'phase':phase,
                    'gain_applied':self.multiplier,'features':features,'evaluation_only':label})+'\n')
            self.steps+=1
            if not np.isfinite(q).all():self.fault='Nonfinite joint state'
            self.render();return self.observe()
    replay.CASES=CASES
    replay.run_case(out,case,sandbox_cls=SampledSandbox)
    sim=instances[0];sim.log.close()
    np.savez_compressed(out/'inputs_200hz.npz',force_link_N=np.array([x[0] for x in sim.inputs],dtype=np.float32),
        q_rad=np.array([x[1] for x in sim.inputs],dtype=np.float32),target_q_rad=np.array([x[2] for x in sim.inputs],dtype=np.float32),
        gain_applied=np.array([x[3] for x in sim.inputs],dtype=np.float32),time_s=np.arange(1,len(sim.inputs)+1)*.005,
        finger_order=np.array(list(sim.sensors)),joint_order=np.array(sim.names))
    np.savez_compressed(out/'labels_200hz.npz',**{k:np.array([x[k] for x in sim.labels]) for k in sim.labels[0]})
    for file in ('manifest.json','score.json'):
        p=out/file;j=replay.load(p);j.update(sensor_hz=200,command_hz=40,reactive=reactive,
            trigger_time_s=None if sim.trigger_sample is None else sim.trigger_sample*.005,trigger_config=CONFIG,
            detector_inputs='Current/past tactile and static sensor coordinates; phase only arms detection',
            gain_at_end=sim.multiplier,effort_limits_unchanged=True,
            interpretation='Exact target replay; tactile evaluated each 5ms. Reactive groups ramp pinch PD stiffness x2 over 50ms after a causal event; evaluation truth is separate.')
        if file=='manifest.json':j['driver']='200 Hz causal tactile monitor / reactive gains; 40 Hz stored targets'
        replay.write(p,j)

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--case',choices=CASES);a=p.parse_args()
    if a.case:run_case(a.output,a.case);return
    a.output.mkdir(parents=True,exist_ok=False)
    paths=[Path(__file__),Path(replay.__file__),replay.ROOT/'ur7e_xhand_sim/environment.py',replay.ROOT/'Genesis/genesis/engine/sensors/kinematic_tactile.py']
    replay.write(a.output/'protocol.json',{'cases':CASES,'detector':CONFIG,'frozen_before_runs':True,'one_attempt_per_case':True,
        'demonstration':str(replay.DEMO),'trajectory_sha256':replay.sha(replay.DEMO/'control.jsonl'),
        'reference_sha256':replay.sha(replay.DEMO/'reference.json'),
        'source_sha256':{str(x):replay.sha(x) for x in paths},
        'scope':'Thresholds informed by historical 40Hz runs; these are exploratory replay checks, not held-out model validation.'})
    snap=a.output/'source_snapshot';snap.mkdir()
    for x in paths:(snap/x.name).write_bytes(x.read_bytes())
    results=[]
    for case in CASES:
        print('START '+case,flush=True)
        with (a.output/(case+'.log')).open('w') as log:
            r=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--output',str((a.output/case).resolve()),'--case',case],stdout=log,stderr=subprocess.STDOUT,cwd=replay.ROOT)
        score=a.output/case/'score.json';results.append(replay.load(score) if score.exists() else {'case':case,'execution_error':True,'returncode':r.returncode})
        replay.write(a.output/'results.json',results);print(json.dumps(results[-1]),flush=True)
    if any(x.get('execution_error') for x in results):raise SystemExit(1)
if __name__=='__main__':main()
