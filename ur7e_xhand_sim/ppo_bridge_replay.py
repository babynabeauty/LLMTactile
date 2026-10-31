"""Explicit diagnostic replay over the real Sandbox; no offline VecEnv stepping."""
import numpy as np
from residual_rl.core import Trajectory

class PPOBridgeReplay:
    def __init__(self, sim, payload):
        self.sim=sim
        if not np.isclose(sim.physics_dt_s,1/300) or sim.control_substeps!=5:
            raise ValueError('PPO replay requires explicit 300Hz physics / 60Hz sensor configuration')
        self.trajectory=Trajectory(payload['trajectory_path'])
        self.start=int(self.trajectory.phase_starts['preload']*15)*20
        self.end=int(np.ceil(self.trajectory.duration_s*15))*20
        self.tick=0;self.gain=np.float32(1);self.history=np.zeros((12,5,120,3),np.float32);self.states=None
        self.gain_min=.2;self.gain_max=3.;self.rate=.2
        sim.reset()
        self.finished=False
        self.names=[l['finger'] for l in sim.layouts]
        for t in range(self.start):
            self.set_gain(np.float32(self.trajectory.grip_gain_at(t/300)))
            if t%20==0:self.target(t)
            sim.scene.step();self.tick+=1
            if self.tick%5==0:self.sample()
        sim.steps=self.tick//5

    def set_gain(self,g):
        s=self.sim;self.gain=np.float32(g)
        kp=np.asarray(s.kp,np.float32).copy();kv=np.asarray(s.kv,np.float32).copy()
        kp[6:12]*=self.gain;kv[6:12]*=np.sqrt(self.gain)
        s.robot.set_dofs_kp(kp,s.indices);s.robot.set_dofs_kv(kv,s.indices)
        np.testing.assert_allclose(s.array(s.robot.get_dofs_kp(s.indices)).reshape(-1),kp,rtol=1e-5)
        np.testing.assert_allclose(s.array(s.robot.get_dofs_kv(s.indices)).reshape(-1),kv,rtol=1e-5)
        low,high=s.robot.get_dofs_force_range(s.indices)
        np.testing.assert_allclose(s.array(high).reshape(-1),s.effort,rtol=1e-5)
        np.testing.assert_allclose(s.array(low).reshape(-1),-s.effort,rtol=1e-5)

    def target(self,tick):
        s=self.sim;s.target=self.trajectory.target(tick/300)
        s.robot.control_dofs_position(s.target,s.indices)
        s.actions=((s.target-s.offset)/s.scale).tolist()

    def sample(self):
        s=self.sim;s.steps=self.tick//5;o=s.observe(full=True)
        forces=np.asarray([o['tactile'][f]['force_link_N'] for f in self.names],np.float32)
        direct=np.stack([s.array(s.sensors[f].read().force).reshape(-1,120,3)[0] for f in self.names])
        np.testing.assert_array_equal(forces,direct)
        self.history[:-1]=self.history[1:];self.history[-1]=forces
        q=np.asarray([o['joint_positions_rad'][n] for n in s.names],np.float32)
        qd=np.asarray([o['joint_velocities_rad_s'][n] for n in s.names],np.float32)
        mid=(s.lower+s.upper)/2;half=(s.upper-s.lower)/2
        nom=self.trajectory.target(self.tick/300);future=self.trajectory.target(self.tick/300+1/15)
        phase=self.trajectory.phase(self.tick/300)
        state=np.concatenate([(q-mid)/half,np.clip(qd/3,-10,10),(nom-q)/.2,(future-nom)/.1,
            np.array([self.gain-1,self.gain-1]),np.array([phase==p for p in ('preload','test_lift','lift','hold')]),
            [(self.tick-self.start)/(self.end-self.start)]]).astype(np.float32)
        if self.states is None:self.states=np.repeat(state[None],12,axis=0)
        else:self.states[:-1]=self.states[1:];self.states[-1]=state

    def observe(self):
        return {'tick':self.tick,'finished':self.finished,'gain':float(self.gain),
            'observation':{'tactile':self.history.tolist(),'state':self.states.tolist()},
            'finger_order':self.names,'frame':'attachment_link_XYZ','hold_interval_s':[self.trajectory.duration_s-3,self.trajectory.duration_s]}

    def advance(self,payload):
        if self.finished:raise ValueError('Replay finished')
        if payload.get('expected_tick')!=self.tick:raise ValueError('Stale PPO action')
        a=np.asarray(payload['action'],np.float32)
        if a.size!=1 or not np.isfinite(a).all():raise ValueError('One finite PPO action required')
        requested=np.float32(self.gain_min)*np.exp((np.clip(a.reshape(1),-1,1)+1)*.5*np.log(self.gain_max/self.gain_min))
        self.set_gain(np.clip(self.gain+np.clip(requested[0]-self.gain,-self.rate,self.rate),self.gain_min,self.gain_max))
        self.target(self.tick);samples=[]
        for _ in range(20):
            self.sim.scene.step();self.tick+=1
            if self.tick%5==0:
                self.sample();e=self.sim.evaluate();e['sample_time_s']=self.tick/300;samples.append(e)
        self.finished=self.tick>=self.end
        result=self.observe();result['evaluation_samples']=samples
        return result
