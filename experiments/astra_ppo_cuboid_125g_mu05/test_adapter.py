"""CPU-only adapter contract tests using an actual frozen model, no Astra."""
import json,tempfile
from pathlib import Path
import numpy as np
from online_ppo import OnlinePPO
class Robot:
 def __init__(self):self.kp=np.array([4500,4500,3500,2000,2000,2000]+[3.]*12);self.kv=np.array([450,450,350,200,200,200]+[.08]*12)
 def get_dofs_kp(self,i):return self.kp
 def get_dofs_kv(self,i):return self.kv
 def set_dofs_kp(self,v,i):self.kp=v.copy()
 def set_dofs_kv(self,v,i):self.kv=v.copy()
 def get_dofs_force_range(self,i):return -np.ones(18)*10,np.ones(18)*10
class Sim:
 def __init__(self,p):
  self.output_dir=p;self.physics_dt_s=1/300;self.control_substeps=5;self.robot=Robot();self.steps=0;self.names=list(map(str,range(18)));self.indices=list(range(18));self.layouts=[{'finger':x} for x in ['thumb','index','middle','ring','pinky']];self.lower=-np.ones(18)*2;self.upper=np.ones(18)*2;self.kp=self.robot.kp.copy();self.kv=self.robot.kv.copy();self.effort=np.ones(18)*10
 def array(self,v):return np.asarray(v)
 def observe(self,full=False):return {'joint_positions_rad':dict.fromkeys(self.names,0.),'joint_velocities_rad_s':dict.fromkeys(self.names,0.),'tactile':{x['finger']:{'force_link_N':np.zeros((120,3)).tolist()} for x in self.layouts}}
with tempfile.TemporaryDirectory() as tmp:
 s=Sim(Path(tmp));p=OnlinePPO(s,Path(__file__).parent/'config.json')
 for _ in range(12):p.before(np.zeros(18),'close');s.steps+=1;p.after()
 assert p.updates==0
 for _ in range(12):p.before(np.zeros(18),'preload');s.steps+=1;p.after()
 assert p.updates==3 and p.history.shape==(12,5,120,3) and p.states.shape==(12,79)
 assert np.isfinite(p.states).all();np.testing.assert_array_equal(s.robot.kp[12:],s.kp[12:]);np.testing.assert_array_equal(s.robot.kp[:6],s.kp[:6])
 p.log.close();rows=[json.loads(x) for x in (Path(tmp)/'online_ppo.jsonl').read_text().splitlines()]
 assert [x['step'] for x in rows]==[12,16,20]
 assert max(abs(np.diff([1.]+[x['gain'] for x in rows])))<=.200001
 print('PASS: real frozen tactile PPO; takeover at preload; 60Hz history/15Hz actions; rate limit; arm/other fingers unchanged; no Astra')
from s8v4 import S8V4
from types import SimpleNamespace
with tempfile.TemporaryDirectory() as tmp:
 for dt,n,expected in [(1/60,120,False),(1/60,180,True),(.025,120,True)]:
  fake=SimpleNamespace(sim=SimpleNamespace(control_dt_s=dt),evaluation=[{'phase':'hold','bottom_clearance_m':.06}]*n,failed=None,phase='holding',out=Path(tmp))
  assert S8V4.score(fake)['success']==expected
 print('PASS: 3-second hold criterion at both old40Hz and online60Hz; 120 samples at60Hz correctly fail')
