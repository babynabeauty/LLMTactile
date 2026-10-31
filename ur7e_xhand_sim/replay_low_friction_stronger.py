"""Two requested replay experiments: mu .01 and stronger 200g pinch."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import replay_material_sweep as base

CASES={'friction_001':{'cube_density':500.,'cube_friction':.01,'robot_friction':.01},
       'mass_200g_stronger':{'cube_mass_kg':.2,'cube_friction':.8,'robot_friction':.8}}

def run(out,case):
    import numpy as np
    from environment import Sandbox
    rows=[json.loads(s) for s in (base.DEMO/'control.jsonl').read_text().splitlines()]
    pre=[i for i,r in enumerate(rows) if r['phase']=='preload']
    maximum=2. if case=='mass_200g_stronger' else 1.
    class GainSandbox(Sandbox):
        def __init__(self,**kwargs):
            self.gain_multiplier=1.
            super().__init__(**kwargs)
            self.initial_kp=np.array(self.kp);self.initial_kv=np.array(self.kv)
        def step(self,payload):
            u=np.clip((self.steps-pre[0]+1)/len(pre),0.,1.)
            self.gain_multiplier=1+(maximum-1)*(u*u*(3-2*u))
            kp=self.initial_kp.copy();kv=self.initial_kv.copy()
            kp[6:12]*=self.gain_multiplier;kv[6:12]*=np.sqrt(self.gain_multiplier)
            self.robot.set_dofs_kp(kp,self.indices);self.robot.set_dofs_kv(kv,self.indices)
            np.testing.assert_allclose(self.values(self.robot.get_dofs_kp(self.indices)),kp,rtol=1e-6,atol=1e-6)
            np.testing.assert_allclose(self.values(self.robot.get_dofs_kv(self.indices)),kv,rtol=1e-6,atol=1e-6)
            return super().step(payload)
        def evaluate(self):
            result=super().evaluate()
            result['pinch_kp_multiplier']=self.gain_multiplier
            result['pinch_kv_multiplier']=self.gain_multiplier**.5
            return result
    base.CASES=CASES
    base.run_case(out,case,sandbox_cls=GainSandbox)
    for name in ('manifest.json','score.json'):
        p=out/name;j=base.load(p)
        j.update(pinch_kp_multiplier=maximum,pinch_kv_multiplier=maximum**.5,
            gain_schedule={'start_step':pre[0]+1,'end_step':pre[-1]+1,'curve':'smoothstep; hold increased gains thereafter'},
            increased_joints=base.load(base.DEMO/'reference.json')['hand_joint_order'][:6],
            effort_limits_unchanged=True,
            interpretation='Exact recorded 18-joint targets; unchanged effort limits; thumb/index gains ramp during preload. Tactile is recorded only.')
        if name=='manifest.json':j['driver']='Recorded target replay with explicitly configured thumb/index PD gains'
        base.write(p,j)

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--case',choices=CASES);a=p.parse_args()
    if a.case:run(a.output,a.case);return
    a.output.mkdir(parents=True,exist_ok=False)
    paths=[Path(__file__),Path(base.__file__),base.ROOT/'ur7e_xhand_sim/environment.py',base.ROOT/'Genesis/genesis/engine/sensors/kinematic_tactile.py']
    base.write(a.output/'protocol.json',{'cases':CASES,'scenario':'original; no distractors','demonstration':str(base.DEMO),
        'reference_sha256':base.sha(base.DEMO/'reference.json'),'trajectory_sha256':base.sha(base.DEMO/'control.jsonl'),
        'one_attempt_per_case':True,'stronger_grip':'Thumb/index kp x2, kv x sqrt(2), smooth ramp during recorded preload; same targets and effort limits',
        'comparison_run':str(base.ROOT/'outputs/material_replay_20261024_v1/mass_200g'),
        'source_sha256':{str(x):base.sha(x) for x in paths}})
    snapshot=a.output/'source_snapshot';snapshot.mkdir()
    for x in paths:(snapshot/x.name).write_bytes(x.read_bytes())
    results=[]
    for case in CASES:
        print('START '+case,flush=True)
        with (a.output/(case+'.log')).open('w') as log:
            ret=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--output',str((a.output/case).resolve()),'--case',case],stdout=log,stderr=subprocess.STDOUT,cwd=base.ROOT)
        score=a.output/case/'score.json'
        results.append(base.load(score) if score.exists() else {'case':case,'execution_error':True,'returncode':ret.returncode})
        base.write(a.output/'results.json',results);print(json.dumps(results[-1]),flush=True)
    if any(x.get('execution_error') for x in results):raise SystemExit(1)
if __name__=='__main__':main()
