"""Material-adaptive common-gain residual control, sharing the v2 physical scene.

Both task success and the unchanged v2 strict contact criterion are recorded.
All reward coefficients and gain bounds must be frozen before held-out tests.
"""
from dataclasses import dataclass, asdict
import json
import numpy as np
from gymnasium import spaces
from .env import ResidualVecEnv, cpu
from .core import interval_overlap, hold_success


@dataclass(frozen=True)
class AdaptiveConfig:
    gain_min: float = .6
    gain_max: float = 3.
    gain_rate_limit: float = .2
    progress_weight: float = 3.
    force_weight: float = .12
    slip_weight: float = 1.
    unsupported_weight: float = .5
    action_change_weight: float = .05
    success_bonus: float = 25.
    failure_penalty: float = 25.

    def __post_init__(self):
        if not (0 < self.gain_min <= 1 <= self.gain_max and self.gain_min < self.gain_max):
            raise ValueError('Gain range must include the physically replayed nominal gain 1')
        if not all(np.isfinite(v) and v >= 0 for v in asdict(self).values()):
            raise ValueError('Configuration values must be finite and nonnegative')


class AdaptiveVecEnv(ResidualVecEnv):
    def __init__(self, *args, config=None, **kwargs):
        self.config = config or AdaptiveConfig()
        self.proprio_history = None
        self.prefix_proprio = None
        self.sensor_tick = 0
        super().__init__(*args, **kwargs)
        self.observation_space = spaces.Dict({
            'tactile': spaces.Box(-np.inf,np.inf,(12,5,120,3),np.float32),
            'state': spaces.Box(-np.inf,np.inf,(12,79),np.float32)})
        self.action_space = spaces.Box(-1.,1.,(1,),np.float32)
        protocol_path = self.output_dir/'environment_protocol.json'
        protocol = json.loads(protocol_path.read_text())
        protocol.update(variant='adaptive_common_gain_v3', config=asdict(self.config),
            action='one common thumb/index gain; exponential interpolation of configured range; bounded change per 15Hz step',
            state_history='12 samples at 60Hz; identical proprioceptive history for tactile and no-tactile policies',
            task_success='original final 3s hold: cube bottom >=5cm at each 60Hz sample; no robot collision',
            strict_success='task_success AND unchanged v2 criterion: >=2 tactile-bearing links with load>.05N at every hold sample',
            training_terminal_reward='task_success, not strict_success; uninterrupted contact quality remains a continuous penalty and separately logged old criterion',
            randomization='explicit or seeded material condition per lane; all prefixes physically replayed at that material')
        protocol_path.write_text(json.dumps(protocol,indent=2))

    def _state_at(self, tick):
        q = cpu(self.robot.get_dofs_position(self.indices))
        qd = cpu(self.robot.get_dofs_velocity(self.indices))
        t = tick/300
        nominal = self.trajectory.target(t)
        future = self.trajectory.target(t+1/15)
        phase = self.trajectory.phase(t)
        phases = np.array([phase==p for p in ('preload','test_lift','lift','hold')],np.float32)
        return np.concatenate([(q-self.mid)/self.half,np.clip(qd/3,-10,10),
            (nominal-q)/.2,np.broadcast_to((future-nominal)/.1,q.shape),self.gains-1,
            np.broadcast_to(phases,(self.num_envs,4)),
            np.full((self.num_envs,1),(tick-self.start_tick)/(self.end_tick-self.start_tick))],axis=1).astype(np.float32)

    def _sample_tactile(self):
        super()._sample_tactile()
        self.sensor_tick += 5
        state = self._state_at(self.sensor_tick)
        if self.proprio_history is None:
            self.proprio_history = np.repeat(state[:,None],12,axis=1)
        else:
            self.proprio_history[:,:-1] = self.proprio_history[:,1:]
            self.proprio_history[:,-1] = state

    def _obs(self):
        tactile = self.history.copy()
        if self.tactile_noise_std:
            tactile += self.rng.normal(0,self.tactile_noise_std,tactile.shape).astype(np.float32)
        return {'tactile':tactile,'state':self.proprio_history.copy()}

    def reset(self):
        if self.prefix_proprio is None:
            self.prefix_proprio = self.proprio_history.copy()
        self.proprio_history[:] = self.prefix_proprio
        self.sensor_tick = self.start_tick
        obs = super().reset()
        self.hold_supported_time = np.zeros(self.num_envs)
        self.unsupported_run = np.zeros(self.num_envs)
        self.max_unsupported_run = np.zeros(self.num_envs)
        self.reward_components = np.zeros((self.num_envs,6))
        return obs

    def action_for_gain(self, gain):
        gain = np.asarray(gain,dtype=np.float32)
        c = self.config
        return 2*np.log(np.clip(gain,c.gain_min,c.gain_max)/c.gain_min)/np.log(c.gain_max/c.gain_min)-1

    def step_async(self, actions):
        a = np.asarray(actions,dtype=np.float32)
        if a.shape != (self.num_envs,1) or not np.isfinite(a).all():
            raise ValueError('Expected one finite common gain action per lane')
        self._pending_action = a.copy()

    def _force_penalty_load(self, load):
        """Legacy five-link sum; new protocols can cover every robot contact."""
        return load.sum(-1)

    def _accumulate_extra_metrics(self, dt, hold_dt, load):
        """Read-only reporting hook, called exactly once per physical sample."""

    def _extra_episode_metrics(self, lane):
        return {}

    def step_wait(self):
        if self._pending_action is None: raise RuntimeError('step_async must be called first')
        c = self.config
        requested = c.gain_min*np.exp((np.clip(self._pending_action,-1,1)+1)*.5*np.log(c.gain_max/c.gain_min))
        old = self.gains.copy()
        self.gains = np.clip(old+np.clip(requested-old,-c.gain_rate_limit,c.gain_rate_limit),c.gain_min,c.gain_max)
        self._apply_gains()
        self.robot.control_dofs_position(self.trajectory.target(self.tick/300),self.indices)
        change_cost = c.action_change_weight*np.square(self.gains-old).sum(-1)
        reward = -change_cost
        self.reward_components[:,4] -= change_cost
        new_fault = np.zeros(self.num_envs,dtype=bool)
        interval = []
        for sub in range(20):
            self.scene.step()
            self.tick += 1
            if self.tick % 5 == 0:
                self._sample_tactile()
                clearance,load,slip,fault = self._metrics()
                dt = 1/60
                supported = (load>.05).sum(-1)>=2
                any_support = (load>.05).any(-1)
                components = np.stack([
                    c.progress_weight*np.clip(clearance/.07,0,1)*any_support,
                    -c.force_weight*self._force_penalty_load(load),
                    -c.slip_weight*np.clip(slip/.02,0,5),
                    -c.unsupported_weight*(~supported)*(clearance>.02)],axis=1)*dt
                reward += components.sum(-1)
                self.reward_components[:,:4] += components
                self.max_clearance = np.maximum(self.max_clearance,clearance)
                new_fault |= fault & ~self.fault
                self.fault |= fault
                self.load_integral += load.sum(-1)*dt
                self.peak_load = np.maximum(self.peak_load,load.sum(-1))
                self.slip_integral += slip*dt
                t = self.tick/300
                hold_dt = float(interval_overlap(t-dt,t,self.hold_start,self.trajectory.duration_s))
                self._accumulate_extra_metrics(dt, hold_dt, load)
                if hold_dt:
                    self.hold_duration += hold_dt
                    self.hold_min = np.minimum(self.hold_min,clearance)
                    self.hold_contact_ok &= supported
                    self.hold_supported_time += hold_dt*supported
                    self.unsupported_run = np.where(supported,0,self.unsupported_run+hold_dt)
                    self.max_unsupported_run = np.maximum(self.max_unsupported_run,self.unsupported_run)
                interval.append({'clearance_m':clearance.copy(),'contact_load_N':load.copy(),
                    'slip_speed_m_s':slip.copy(),'fault':fault.copy(),
                    **{k:v.copy() for k,v in getattr(self,'last_contact_diagnostics',{}).items()}})
        self.last_metrics = interval[-1]
        self.last_interval_metrics = {k:np.stack([r[k] for r in interval]) for k in interval[0]}
        self.elapsed += 1/15
        self.gain_integral += self.gains/15
        reward -= 20*new_fault
        self.reward_components[:,5] -= 20*new_fault
        done = self.tick >= self.end_tick
        infos = [{'elapsed_sim_s':self.elapsed,'tactile_samples':4,
                  'applied_gains':self.gains[i].copy()} for i in range(self.num_envs)]
        self.returns += reward
        obs = self._obs()
        if done:
            task = hold_success(self.hold_duration,self.hold_min,self.fault)
            strict = task & self.hold_contact_ok
            bonus = np.where(task,c.success_bonus,-c.failure_penalty)
            reward += bonus
            self.returns += bonus
            self.reward_components[:,5] += bonus
            for i in range(self.num_envs):
                metrics = {'success':bool(strict[i]),'strict_success':bool(strict[i]),'task_success':bool(task[i]),
                    'mass_kg':float(self.mass[i]),'friction':float(self.mu[i]),
                    'hold_min_clearance_m':float(self.hold_min[i]),'hold_duration_s':float(self.hold_duration[i]),
                    'hold_contact_ok':bool(self.hold_contact_ok[i]),
                    'hold_contact_fraction':float(self.hold_supported_time[i]/max(self.hold_duration[i],1e-9)),
                    'hold_max_unsupported_duration_s':float(self.max_unsupported_run[i]),
                    'mean_contact_force_N':float(self.load_integral[i]/self.elapsed),
                    'peak_contact_force_N':float(self.peak_load[i]),
                    'slip_distance_proxy_m':float(self.slip_integral[i]),
                    'mean_gains':(self.gain_integral[i]/self.elapsed).tolist(),
                    'return':float(self.returns[i]),'reward_components':self.reward_components[i].tolist(),
                    'termination_reason':'collision' if self.fault[i] else 'success' if strict[i] else 'contact_discontinuity' if task[i] else 'hold_failed',
                    'steps':int((self.end_tick-self.start_tick)/20)}
                metrics.update(self._extra_episode_metrics(i))
                infos[i].update(episode_metrics=metrics,episode={'r':float(self.returns[i]),'l':metrics['steps']},
                    terminal_observation={k:v[i].copy() for k,v in obs.items()},**{'TimeLimit.truncated':False})
            obs = self.reset()
        self._pending_action = None
        return obs,reward.astype(np.float32),np.full(self.num_envs,done),infos
