"""Opt-in causal PPO gain controller for live S8V4. No policy training or Astra calls."""
import hashlib,json
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
from residual_rl.core import Trajectory

class OnlinePPO:
    def __init__(self,sim,config_path):
        self.sim=sim;self.config=json.loads(Path(config_path).read_text())
        c=self.config
        for path,key in [('checkpoint','checkpoint_sha256'),('trajectory','trajectory_sha256')]:
            if hashlib.sha256(Path(c[path]).read_bytes()).hexdigest()!=c[key]:raise ValueError('Frozen input hash mismatch: '+path)
        if not np.isclose(sim.physics_dt_s,1/300) or sim.control_substeps!=5:raise ValueError('Online PPO requires 300/60Hz')
        torch.set_num_threads(4);self.model=PPO.load(c['checkpoint'],device='cpu')
        if not self.model.policy.features_extractor.use_tactile:raise ValueError('Tactile model required')
        self.demo=Trajectory(c['trajectory']);self.history=np.zeros((12,5,120,3),np.float32);self.states=None
        self.gain=np.float32(1);self.active=False;self.phase=None;self.phase_ticks=0;self.updates=0
        self.log=(sim.output_dir/'online_ppo.jsonl').open('w',buffering=1)
        self.meta={'enabled':True,'checkpoint_sha256':c['checkpoint_sha256'],'seed':int(self.model.seed),'action_hz':15,'tactile_hz':60,
            'causal_lookahead':'zero-order hold: unknown future target equals current nominal target',
            'progress':'reference phase start + elapsed phase time, clipped at next reference phase boundary; normalized to demo preload/end',
            'gain_owner':'Astra may choose initial close gain; PPO takes over at preload; original support target corrections remain active',
            'experimental_online_distribution_shift':True}

    def before(self,target,phase):
        s=self.sim
        if phase!=self.phase:self.phase=phase;self.phase_ticks=0
        # Actual gains are read before takeover, including Astra's initial close choice.
        self.gain=np.float32(s.array(s.robot.get_dofs_kp(s.indices)).reshape(-1)[6]/3.)
        if phase in ('preload','test_lift','lift','hold') and not self.active:
            self.active=True;self.takeover_step=s.steps
        if self.active and (s.steps-self.takeover_step)%4==0:
            if self.states is None:raise ValueError('No proprioceptive history before PPO takeover')
            obs={'tactile':self.history[None].copy(),'state':self.states[None].copy()}
            action,_=self.model.predict(obs,deterministic=True)
            requested=.2*np.exp((np.clip(action[0,0],-1,1)+1)*.5*np.log(15.))
            self.gain=np.float32(np.clip(self.gain+np.clip(requested-self.gain,-.2,.2),.2,3.))
            kp=np.asarray(s.kp,np.float32).copy();kv=np.asarray(s.kv,np.float32).copy()
            kp[6:12]*=self.gain;kv[6:12]*=np.sqrt(self.gain)
            s.robot.set_dofs_kp(kp,s.indices);s.robot.set_dofs_kv(kv,s.indices)
            np.testing.assert_allclose(s.array(s.robot.get_dofs_kp(s.indices)).reshape(-1),kp,rtol=1e-5)
            np.testing.assert_allclose(s.array(s.robot.get_dofs_kv(s.indices)).reshape(-1),kv,rtol=1e-5)
            lower,upper=s.robot.get_dofs_force_range(s.indices)
            np.testing.assert_allclose(s.array(lower).reshape(-1),-s.effort,rtol=1e-5)
            np.testing.assert_allclose(s.array(upper).reshape(-1),s.effort,rtol=1e-5)
            self.updates+=1
            self.log.write(json.dumps({'step':s.steps,'phase':phase,'action':float(action[0,0]),'gain':float(self.gain),'updates':self.updates,'observation':{k:v[0].tolist() for k,v in obs.items()}})+'\n')
        self.nominal=np.asarray(target).copy()

    def after(self):
        s=self.sim;o=s.observe(full=True)
        force=np.asarray([o['tactile'][l['finger']]['force_link_N'] for l in s.layouts],np.float32)
        self.history[:-1]=self.history[1:];self.history[-1]=force
        q=np.asarray([o['joint_positions_rad'][n] for n in s.names],np.float32);qd=np.asarray([o['joint_velocities_rad_s'][n] for n in s.names],np.float32)
        mid=(s.lower+s.upper)/2;half=(s.upper-s.lower)/2
        starts=self.demo.phase_starts;phase=self.phase if self.phase in starts else 'settle'
        begin=starts.get(phase,0.);next_starts=[v for v in starts.values() if v>begin]
        end=min(next_starts) if next_starts else self.demo.duration_s
        local=min(begin+self.phase_ticks/60,end)
        origin=int(starts['preload']*15)/15;duration=np.ceil(self.demo.duration_s*15)/15-origin
        state=np.concatenate([(q-mid)/half,np.clip(qd/3,-10,10),(self.nominal-q)/.2,np.zeros(18),
            [self.gain-1,self.gain-1],[self.phase==p for p in ('preload','test_lift','lift','hold')],[(local-origin)/duration]]).astype(np.float32)
        if self.states is None:self.states=np.repeat(state[None],12,axis=0)
        else:self.states[:-1]=self.states[1:];self.states[-1]=state
        self.phase_ticks+=1
