"""Freeze a gain/material controllability scan before PPO model selection."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
from .adaptive_env import AdaptiveVecEnv, AdaptiveConfig
from .train import write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--masses',type=float,nargs='+',default=[.05,.1,.15,.2])
    p.add_argument('--frictions',type=float,nargs='+',default=[.2,.4,.6,.8])
    p.add_argument('--gains',type=float,nargs='+',default=[.6,.8,1,1.25,1.5,2,2.5,3])
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    conditions=np.array([(m,mu) for m in a.masses for mu in a.frictions])
    cfg=AdaptiveConfig()
    write_json(a.output/'manifest.json',{'conditions_kg_mu':conditions.tolist(),
        'gains':a.gains,'config':asdict(cfg),'role':'development/calibration, not held-out test',
        'classification':'common fixed gain with bounded initial ramp'})
    snap=a.output/'source_snapshot';snap.mkdir()
    for path in Path(__file__).parent.glob('*.py'): shutil.copy2(path,snap/path.name)
    env=AdaptiveVecEnv(n_envs=len(conditions),seed=123,material_conditions=conditions,
        output_dir=a.output/'environment',config=cfg)
    results=[];first=None
    try:
        for j,g in enumerate([*a.gains,a.gains[0]]):
            obs=env.reset(); prefix={k:v.copy() for k,v in obs.items()}
            action=np.full((len(conditions),1),env.action_for_gain(g),np.float32)
            metrics=[];summaries=[];state=[]
            while True:
                summaries.append(np.linalg.norm(obs['tactile'][:,-4:],axis=-1).sum(-1))
                state.append(obs['state'][:,-1].copy())
                obs,reward,done,infos=env.step(action)
                metrics.append({k:v.copy() for k,v in env.last_interval_metrics.items()})
                if done.all():break
            trace={k:np.concatenate([x[k] for x in metrics]) for k in metrics[0]}
            if first is None:first={k:v.copy() for k,v in trace.items()}
            name=f'gain_{g:g}' if j<len(a.gains) else 'reset_check'
            np.savez_compressed(a.output/f'{name}.npz',**trace,
                tactile_load_proxy_before=np.stack(summaries),state_before=np.stack(state),
                prefix_tactile=prefix['tactile'],prefix_state=prefix['state'])
            records=[i['episode_metrics'] for i in infos]
            result={'gain':g,'name':name,'task_successes':sum(r['task_success'] for r in records),
                'strict_successes':sum(r['strict_success'] for r in records),
                'mean_load_N':float(np.mean([r['mean_contact_force_N'] for r in records])),
                'mean_return':float(np.mean([r['return'] for r in records])), 'episodes':records}
            results.append(result)
            write_json(a.output/'results.json',results)
            print(json.dumps({k:v for k,v in result.items() if k!='episodes'}),flush=True)
            if j==len(a.gains):
                diffs={k:float(np.max(np.abs(first[k].astype(float)-v.astype(float)))) for k,v in trace.items()}
                write_json(a.output/'reset_consistency.json',diffs)
                if max(diffs.values())>1e-5:raise AssertionError('Adaptive scene reset did not reproduce fixed controller')
        write_json(a.output/'status.json',{'status':'completed','controllers':len(results)})
    except BaseException as exc:
        write_json(a.output/'status.json',{'status':'failed','error':repr(exc)})
        raise
    finally:env.close()


if __name__=='__main__':main()
