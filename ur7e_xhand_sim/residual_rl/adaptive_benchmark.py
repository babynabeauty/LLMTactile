"""Paired multi-material evaluation, including frozen tactile interventions."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import time
import traceback
import numpy as np
import torch
from stable_baselines3 import PPO
from .adaptive_env import AdaptiveVecEnv, AdaptiveConfig
from .baselines import ReactiveGripController
from .train import write_json
from .core import DEMO, Trajectory
from .env import ARM_JOINTS, HAND_JOINTS


def _same_finite_shape(value, expected, atol=1e-8):
    actual, expected = np.asarray(value,dtype=float), np.asarray(expected,dtype=float)
    return (actual.shape == expected.shape and np.isfinite(actual).all()
            and np.allclose(actual,expected,rtol=0,atol=atol))


def freeze_trajectory_input(path, destination):
    """Freeze commands and reject external scenes unsupported by this benchmark."""
    path = Path(path or DEMO/'control.jsonl').expanduser().resolve()
    trajectory = Trajectory(path)
    metadata = None
    cube_position = [-.6,-.1333,.225]
    metadata_path = path.parent/'trajectory_metadata.json'
    if path != (DEMO/'control.jsonl').resolve():
        if not metadata_path.is_file():
            raise ValueError('External trajectory requires trajectory_metadata.json')
        metadata = json.loads(metadata_path.read_text())
        if metadata.get('trajectory_sha256') != trajectory.sha256:
            raise ValueError('External trajectory hash does not match metadata')
        if metadata.get('joint_order') != ARM_JOINTS+HAND_JOINTS:
            raise ValueError('External trajectory joint order differs from the robot')
        scene = metadata['scene']
        supported_translations = {'original':[0.,0.,0.], 'large':[.15,.10,0.]}
        scenario_name = scene.get('scenario_name','original')
        if scenario_name not in supported_translations:
            raise ValueError('Only original and large-translation scenes are supported')
        translation = supported_translations[scenario_name]
        if not _same_finite_shape(scene.get('translation_m',[0.,0.,0.]),translation):
            raise ValueError('External trajectory translation differs from named scene')
        cube_position = (np.asarray(cube_position)+translation).tolist()
        for key, expected in {'target_initial_position_m':cube_position,
                              'robot_base_position_m':[0,0,.2], 'table_top_z_m':.2,
                              'cube_size_m':[.045]*3}.items():
            if not _same_finite_shape(scene[key], expected):
                raise ValueError('External trajectory scene differs: '+key)
        if scene.get('distractors'):
            raise ValueError('This benchmark has no distractors')
        home = np.r_[np.deg2rad([0,-89.95,89.96,-89.96,-89.95,-44.98]),np.zeros(12)]
        if not _same_finite_shape(metadata['initial_state']['home_joint_positions_rad'],home,atol=1e-6):
            raise ValueError('External trajectory initial home differs')
        if not _same_finite_shape(metadata['initial_state']['home_joint_velocities_rad_s'],np.zeros(18)):
            raise ValueError('External trajectory initial velocity differs')
        for key, expected in {'kp':[4500,4500,3500,2000,2000,2000]+[3.]*12,
                              'kv':[450,450,350,200,200,200]+[.08]*12}.items():
            if not _same_finite_shape(metadata['source_pd'][key],expected,atol=1e-6):
                raise ValueError('External trajectory baseline PD differs: '+key)
    destination.mkdir()
    frozen = destination/'control.jsonl'
    shutil.copy2(path, frozen)
    if metadata is not None:
        shutil.copy2(metadata_path, destination/metadata_path.name)
    if hashlib.sha256(frozen.read_bytes()).hexdigest() != trajectory.sha256:
        raise ValueError('Trajectory changed while freezing input')
    return frozen, {'source_path':str(path), 'sha256':trajectory.sha256,
                    'duration_s':trajectory.duration_s, 'phase_starts':trajectory.phase_starts,
                    'cube_initial_position_m':cube_position,
                    'metadata':metadata}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--conditions-json',type=Path,required=True)
    p.add_argument('--config-json',type=Path,required=True)
    p.add_argument('--trajectory',type=Path,help='External converted command trajectory; default is original teacher')
    p.add_argument('--cube-size-m',type=float,help='Explicit rigid cube size transfer; preserve trajectory XY and place bottom on table')
    p.add_argument('--models-only',action='store_true',help='Evaluate exactly the listed frozen models, without fixed controls or repeat episodes')
    p.add_argument('--audit-prefix',action='store_true',help='Record pre-policy physical contacts/collisions separately without changing task labels')
    p.add_argument('--models',nargs='*',default=[],help='name=/absolute/or/relative/checkpoint.zip')
    p.add_argument('--fixed-gains',type=float,nargs='+',default=[.8,1.,1.25,1.5,2.,2.5,3.])
    p.add_argument('--intervene-model',help='Name of one frozen model to evaluate with zero/permuted tactile')
    p.add_argument('--interventions',nargs='+',choices=['zero','permuted','frozen_after_probe','gain_hold_after_probe'],
        default=['zero','permuted'])
    p.add_argument('--reactive',action='store_true',help='Include the frozen old tactile detector, requesting gain 3 after alarm')
    p.add_argument('--seed',type=int,default=7000)
    p.add_argument('--split',choices=['development','validation','heldout','trajectory_transfer','size_transfer','preflight'],required=True)
    a=p.parse_args()
    if a.cube_size_m is not None and (not np.isfinite(a.cube_size_m) or a.cube_size_m <= 0):
        raise ValueError('Cube edge must be finite and positive')
    if a.models_only and (not a.models or a.reactive or a.intervene_model):
        raise ValueError('models-only requires listed models and no reactive/intervention controllers')
    conditions=np.asarray(json.loads(a.conditions_json.read_text()),np.float32)
    config=AdaptiveConfig(**json.loads(a.config_json.read_text()))
    a.output.mkdir(parents=True,exist_ok=False)
    frozen_trajectory, trajectory_info = freeze_trajectory_input(a.trajectory,a.output/'trajectory_input')
    if a.cube_size_m is not None:
        trajectory_info['source_cube_initial_position_m']=trajectory_info['cube_initial_position_m'].copy()
        trajectory_info['cube_initial_position_m'][2]=.2+a.cube_size_m/2
        trajectory_info['cube_size_m']=a.cube_size_m
        trajectory_info['geometry_override']='Explicit rigid size transfer; commands unchanged, XY unchanged, bottom initially at table z=.2'
    shutil.copy2(a.conditions_json,a.output/'conditions.input.json')
    shutil.copy2(a.config_json,a.output/'config.input.json')
    write_json(a.output/'status.json',{'status':'initializing','trajectory':trajectory_info['sha256']})
    torch.set_num_threads(4)
    model_paths={}
    for spec in a.models:
        name,path=spec.split('=',1)
        if not name.isidentifier() or name in model_paths:raise ValueError('Model names must be distinct identifiers')
        model_paths[name]=Path(path)
    models={name:PPO.load(path,device='cpu') for name,path in model_paths.items()}
    if a.intervene_model and a.intervene_model not in models:raise ValueError('Unknown intervention model')
    source=a.output/'source_snapshot';source.mkdir()
    hashes={}
    for f in Path(__file__).parent.glob('*.py'):
        shutil.copy2(f,source/f.name);hashes[f.name]=hashlib.sha256(f.read_bytes()).hexdigest()
    write_json(a.output/'manifest.json',{'conditions_kg_mu':conditions,'config':asdict(config),
        'split':a.split,'seed':a.seed,'source_hashes':hashes,'trajectory':trajectory_info,
        'material_test_interpretation':'Previously evaluated material conditions; new command-source comparison, not fresh material holdout' if a.split=='trajectory_transfer' else a.split,
        'fixed_gains_requested':[] if a.models_only else a.fixed_gains, 'reactive':a.reactive,'audit_prefix':a.audit_prefix,
        'models_only':a.models_only,'cube_size_m':a.cube_size_m if a.cube_size_m is not None else .045,
        'models':{k:{'path':str(v.resolve()),'sha256':hashlib.sha256(v.read_bytes()).hexdigest()} for k,v in model_paths.items()},
        'checkpoint_metadata':{k:{'training_steps':getattr(m,'num_timesteps',None),
            'uses_tactile':getattr(getattr(getattr(m,'policy',None),'features_extractor',None),'use_tactile',None)} for k,m in models.items()},
        'interventions':'zero: all touch becomes 0; permuted: fixed cyclic shift across lanes; frozen_after_probe: tactile window frozen after test_lift; gain_hold_after_probe: keep last actual gain after test_lift. Diagnostic, not standalone proof.',
        'requested_interventions':a.interventions if a.intervene_model else [],
        'actor_inputs':'same 60Hz 12-frame measured tactile/proprio histories, no materials or object truth',
        'training':False})
    geometry_kwargs={} if a.cube_size_m is None else {'cube_size_m':a.cube_size_m}
    env=AdaptiveVecEnv(n_envs=len(conditions),seed=a.seed,material_conditions=conditions,
        output_dir=a.output/'environment',config=config,mode='eval',trajectory_path=frozen_trajectory,
        cube_initial_position=trajectory_info['cube_initial_position_m'],audit_prefix=a.audit_prefix,**geometry_kwargs)
    controllers=[] if a.models_only else [(f'fixed_{g:g}','fixed',g) for g in a.fixed_gains]
    if a.reactive:controllers += [('reactive_3','reactive',None)]
    controllers += [(n,'model',n) for n in models]
    if a.intervene_model:
        controllers += [(a.intervene_model+'_'+s,s,a.intervene_model) for s in a.interventions]
    if not a.models_only:controllers += [('reset_check','fixed',a.fixed_gains[0])]
    results=[];first=None;first_prefix=None
    try:
        for name,kind,value in controllers:
            out=a.output/name;out.mkdir()
            obs=env.reset();prefix={k:v.copy() for k,v in obs.items()}
            if first_prefix is None:first_prefix={k:v.copy() for k,v in prefix.items()}
            prefix_diff={k:float(np.max(np.abs(v-first_prefix[k]))) for k,v in prefix.items()}
            write_json(out/'prefix_consistency.json',prefix_diff)
            if any(value != 0 for value in prefix_diff.values()):
                raise AssertionError('Controller did not start from identical observation history')
            reactive=ReactiveGripController([l['positions_link_m'] for l in env.layouts[:2]],
                env.num_envs,env.start_tick/300,env.trajectory.phase_starts['test_lift']) if kind=='reactive' else None
            frozen_touch=None; held_gain=None
            physical=[];actions=[];gains=[];states=[];tactile=[];start=time.monotonic()
            while True:
                states.append(obs['state'][:,-4:].copy())
                tactile.append(obs['tactile'][:,-4:].copy())
                if kind=='fixed':
                    action=np.full((env.num_envs,1),env.action_for_gain(value),np.float32)
                elif reactive is not None:
                    old_action,_=reactive.predict(obs)
                    action=env.action_for_gain(np.where(old_action[:,:1]>0,3.,1.)).astype(np.float32)
                else:
                    actor_obs=obs
                    if kind in ('zero','permuted'):
                        actor_obs={k:v.copy() for k,v in obs.items()}
                        actor_obs['tactile'] = np.zeros_like(obs['tactile']) if kind=='zero' else np.roll(obs['tactile'],1,axis=0)
                    after_probe=env.tick/300>=env.trajectory.phase_starts['lift']
                    if kind=='frozen_after_probe' and after_probe:
                        if frozen_touch is None:frozen_touch=obs['tactile'].copy()
                        actor_obs={k:v.copy() for k,v in obs.items()}
                        actor_obs['tactile']=frozen_touch.copy()
                    if kind=='gain_hold_after_probe' and after_probe:
                        if held_gain is None:held_gain=env.gains[:,:1].copy()
                        action=env.action_for_gain(held_gain).astype(np.float32)
                    else:
                        action,_=models[value].predict(actor_obs,deterministic=True)
                obs,reward,done,infos=env.step(action)
                physical.append({k:v.copy() for k,v in env.last_interval_metrics.items()})
                actions.append(action.copy());gains.append(np.array([i['applied_gains'] for i in infos]))
                if done.all():break
            trace={k:np.concatenate([x[k] for x in physical]) for k in physical[0]}
            trace.update(actions=np.stack(actions),applied_gains=np.stack(gains),
                sample_time_s=env.start_tick/300+np.arange(1,len(trace['clearance_m'])+1)/60,
                state_before=np.stack(states),tactile_before=np.stack(tactile),
                prefix_tactile=prefix['tactile'],prefix_state=prefix['state'])
            if first is None:first={k:v.copy() for k,v in trace.items()}
            np.savez_compressed(out/'trace.npz',**trace)
            episodes=[i['episode_metrics'] for i in infos]
            row={'controller':name,'task_successes':sum(e['task_success'] for e in episodes),
                'strict_successes':sum(e['strict_success'] for e in episodes),'conditions':env.num_envs,
                'mean_load_N':float(np.mean([e['mean_contact_force_N'] for e in episodes])),
                'mean_return':float(np.mean([e['return'] for e in episodes])),
                'wall_seconds':time.monotonic()-start,'episodes':episodes}
            results.append(row);write_json(out/'evaluation.json',row);write_json(a.output/'results.json',results)
            print(json.dumps({k:v for k,v in row.items() if k!='episodes'}),flush=True)
            if name=='reset_check':
                diff={k:float(np.max(np.abs(first[k].astype(float)-v.astype(float)))) for k,v in trace.items()}
                write_json(a.output/'reset_consistency.json',diff)
                if max(diff.values())>1e-5:raise AssertionError('Repeated fixed controller did not reproduce')
        write_json(a.output/'status.json',{'status':'completed','controllers':len(results)})
    except BaseException as exc:
        write_json(a.output/'status.json',{'status':'failed','error':repr(exc),'traceback':traceback.format_exc()})
        raise
    finally:env.close()


if __name__=='__main__':main()
