"""Matched evaluation of frozen policies, sharing one physically replayed prefix.

Run in a process separate from training. Reset restores all lanes before each
controller; neither physics labels nor reward are fed back to learned actors.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time
import traceback
import numpy as np
import torch
from stable_baselines3 import PPO
from .env import ResidualVecEnv
from .baselines import ReactiveGripController
from .train import serializable, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--tactile-run',type=Path,required=True)
    p.add_argument('--no-tactile-run',type=Path,required=True)
    p.add_argument('--mass-range-kg',nargs=2,type=float,default=[.1,.2])
    p.add_argument('--seed',type=int,default=10000)
    p.add_argument('--n-envs',type=int,default=8)
    a = p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    checkpoints = {'initial':a.tactile_run/'initial_model.zip',
                   'ppo_tactile':a.tactile_run/'final_model.zip',
                   'ppo_no_tactile':a.no_tactile_run/'final_model.zip'}
    for path in checkpoints.values():
        if not path.is_file(): raise FileNotFoundError(path)
    models = {name:PPO.load(path,device='cpu') for name,path in checkpoints.items()}
    source = a.output/'source_snapshot'
    source.mkdir()
    hashes = {}
    for f in Path(__file__).parent.glob('*.py'):
        shutil.copy2(f,source/f.name)
        hashes[f.name]=hashlib.sha256(f.read_bytes()).hexdigest()
    write_json(a.output/'manifest.json',{'seed':a.seed,'mass_range_kg':a.mass_range_kg,
        'source_hashes':hashes,'checkpoints':{n:{'path':str(p.resolve()),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
            for n,p in checkpoints.items()},
        'scope':'one fixed initial condition per mass lane; matched reset for every controller; independent evaluation seed',
        'no_training':True,'tactile_only_policy_inputs':False,
        'actor_inputs':'tactile plus measured robot state and nominal plan; no privileged object state'})
    env = ResidualVecEnv(n_envs=a.n_envs,seed=a.seed,output_dir=a.output/'environment',
                         mass_range_kg=tuple(a.mass_range_kg),friction_range=(.8,.8),mode='eval')
    results=[]
    try:
        # Repeat nominal last to verify that previous policies did not contaminate reset.
        for name in ['nominal','mid','strong','reactive',*models,'nominal_reset_check']:
            out=a.output/name
            out.mkdir()
            obs=env.reset()
            initial_history=obs['tactile'].copy()
            initial_state=obs['state'].copy()
            trigger=ReactiveGripController([l['positions_link_m'] for l in env.layouts[:2]],
                env.num_envs,env.start_tick/300,env.trajectory.phase_starts['test_lift']) if name=='reactive' else None
            trace=[];events=[];start=time.monotonic()
            while True:
                if name in models:
                    action,_=models[name].predict(obs,deterministic=True)
                elif trigger:
                    action,new_events=trigger.predict(obs);events.extend(new_events)
                else:
                    action=np.full((env.num_envs,2),{'nominal':-1.,'nominal_reset_check':-1.,'mid':0.,'strong':1.}[name],np.float32)
                tactile_before=obs['tactile'][:,-4:].copy()
                state_before=obs['state'].copy()
                obs,reward,done,infos=env.step(action)
                trace.append({'actions':action.copy(),'applied_gains':np.array([i['applied_gains'] for i in infos]),
                    'reward':reward.copy(),'elapsed_sim_s':np.array([i['elapsed_sim_s'] for i in infos]),
                    'force_link_N_before':tactile_before,'state_before':state_before,
                    **{k:v.copy() for k,v in env.last_metrics.items()}})
                if done.all(): break
            metrics=[i['episode_metrics'] for i in infos]
            row={'controller':name,'successes':sum(m['success'] for m in metrics),
                'conditions':env.num_envs,'mean_return':float(np.mean([m['return'] for m in metrics])),
                'mean_contact_load_N':float(np.mean([m['mean_contact_force_N'] for m in metrics])),
                'mean_slip_distance_proxy_m':float(np.mean([m['slip_distance_proxy_m'] for m in metrics])),
                'wall_seconds':time.monotonic()-start,'episodes':metrics}
            results.append(row)
            np.savez_compressed(out/'trace.npz',**{k:np.stack([r[k] for r in trace]) for k in trace[0]},
                prefix_tactile=initial_history,prefix_state=initial_state)
            write_json(out/'evaluation.json',row)
            write_json(out/'reactive_events.json',events)
            write_json(a.output/'results.json',results)
            print(json.dumps({k:v for k,v in row.items() if k!='episodes'}),flush=True)
        first=np.load(a.output/'nominal/trace.npz')
        last=np.load(a.output/'nominal_reset_check/trace.npz')
        reset_check={k:float(np.max(np.abs(first[k].astype(float)-last[k].astype(float))))
            for k in ('clearance_m','contact_load_N','force_link_N_before','state_before',
                      'prefix_tactile','prefix_state')}
        write_json(a.output/'reset_consistency.json',reset_check)
        np.testing.assert_array_equal(first['prefix_tactile'],last['prefix_tactile'])
        np.testing.assert_allclose(first['prefix_state'],last['prefix_state'],atol=1e-7,rtol=0)
        if reset_check['clearance_m']>1e-5:
            raise AssertionError('Same-controller reset failed repeatability check')
        if max(reset_check[k] for k in ('contact_load_N','force_link_N_before','state_before'))>1e-5:
            raise AssertionError('Same-controller force/observation reset failed repeatability check')
        write_json(a.output/'status.json',{'status':'completed','controllers':len(results),'reset_check':reset_check})
    except BaseException as exc:
        write_json(a.output/'status.json',{'status':'failed','error':repr(exc),'traceback':traceback.format_exc()})
        raise
    finally:
        env.close()


if __name__=='__main__': main()
