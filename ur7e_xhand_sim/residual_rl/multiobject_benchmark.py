"""Frozen paired evaluation of shared multi-object PPO and fixed-gain controls.

One shape scene at a time limits GPU memory. Every controller reuses the same
saved physical prefix within each material batch; equality is checked exactly.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import re
import time
import traceback

import numpy as np
import torch
from stable_baselines3 import PPO

from .multiobject_train import load_experiment, source_snapshot
from .train import digest, write_json
from .tactile_provenance import current_tactile_model, tactile_models_match, legacy_tactile_model


def model_specs(protocol, training_root=None, requested=(), checkpoint_stage=None):
    specs = []
    if training_root is not None:
        final_stage = checkpoint_stage or protocol['training']['stages'][-1]['name']
        if final_stage not in [s['name'] for s in protocol['training']['stages']]:
            raise ValueError('training-root requires a named frozen training stage')
        for seed in protocol['training']['seeds']:
            for method in protocol['training']['methods']:
                name = f'seed{seed}_{method}'
                specs.append((name, Path(training_root)/f'{name}_{final_stage}'/'final_model.zip'))
    for item in requested:
        name, path = item.split('=',1)
        specs.append((name,Path(path).expanduser().resolve()))
    names = [n for n,_ in specs]
    if len(set(names)) != len(names) or any(not n.isidentifier() for n in names):
        raise ValueError('Controller names must be distinct identifiers')
    return specs


def controller_metadata(name, model):
    match = re.fullmatch(r'seed(\d+)_(tactile|no_tactile)(?:_stage\d+)?',name)
    tactile = bool(model.policy.features_extractor.use_tactile)
    method = 'tactile' if tactile else 'no_tactile'
    if match and match[2] != method:
        raise ValueError('Controller name contradicts saved tactile ablation')
    seed = int(match[1]) if match else int(model.seed)
    if seed != model.seed:
        raise ValueError('Controller seed contradicts saved training seed')
    return seed, method


def read_selection(path, protocol, experiment_path):
    value=json.loads(Path(path).read_text())
    if value.get('selection_split')!='validation' or value.get('test_outcomes_read') is not False:
        raise ValueError('Selection must be based only on validation, without test outcomes')
    if value.get('protocol_sha256')!=digest(experiment_path):
        raise ValueError('Selection protocol hash differs from the frozen experiment')
    selected=value.get('selected_models')
    allowed={f'seed{seed}_{method}' for seed in protocol['training']['seeds']
             for method in protocol['training']['methods']}
    if not isinstance(selected,dict) or not selected or set(selected)-allowed:
        raise ValueError('Selected models must have recognized safe seed/modality names')
    for name,record in selected.items():
        if record.get('checkpoint_stage') not in [s['name'] for s in protocol['training']['stages']]:
            raise ValueError('Selected checkpoint has no valid frozen training stage')
        if digest(Path(record['path']))!=record.get('sha256'):
            raise ValueError('Selected checkpoint hash changed')
    return selected


def evaluate_controller(env, *, name, model, fixed_gain, output, conditions,
                        object_id, split, training_seed, method, prefix_reference=None,
                        save_tactile=True, checkpoint_stage=None):
    output.mkdir(parents=True,exist_ok=False)
    obs = env.reset()
    prefix = {key:values.copy() for key,values in obs.items()}
    difference = {key:float(np.max(np.abs(values-prefix_reference[key])))
                  if prefix_reference is not None else 0. for key,values in prefix.items()}
    write_json(output/'prefix_consistency.json', difference)
    if any(value != 0 for value in difference.values()):
        raise AssertionError('Controllers have different physical/observation prefix')
    physical, actions, gains, states, tactile = [], [], [], [], []
    started = time.monotonic()
    while True:
        states.append(obs['state'][:,-4:].copy())
        if save_tactile: tactile.append(obs['tactile'][:,-4:].copy())
        if model is not None:
            action,_ = model.predict(obs,deterministic=True)
        else:
            action = np.full((env.num_envs,1),env.action_for_gain(fixed_gain),np.float32)
        obs, reward, done, infos = env.step(action)
        physical.append({key:values.copy() for key,values in env.last_interval_metrics.items()})
        actions.append(action.copy())
        gains.append(np.asarray([i['applied_gains'] for i in infos]))
        if done.all(): break
    trace = {key:np.concatenate([interval[key] for interval in physical]) for key in physical[0]}
    trace.update(actions=np.stack(actions),applied_gains=np.stack(gains),state_before=np.stack(states),
        sample_time_s=env.start_tick/300+np.arange(1,len(trace['clearance_m'])+1)/60,
        prefix_state=prefix['state'],prefix_tactile=prefix['tactile'])
    if save_tactile: trace['tactile_before'] = np.stack(tactile)
    episodes = []
    for lane,(condition,info) in enumerate(zip(conditions,infos)):
        metrics = dict(info['episode_metrics'])
        metrics.update(object_id=object_id,split=split,condition_id=condition['condition_id'],
            training_seed=training_seed,method=method,lane=lane,controller=name,
            checkpoint_stage=checkpoint_stage)
        episodes.append(metrics)
    np.savez_compressed(output/'trace.npz',**trace)
    result = {'controller':name,'object_id':object_id,'object_shape':env.object_spec.shape,
        'split':split,'training_seed':training_seed,'method':method,
        'checkpoint_stage':checkpoint_stage,
        'conditions':conditions,'hold_interval_s':[env.hold_start,env.trajectory.duration_s],
        'trajectory':{'path':str(env.trajectory.path),'sha256':env.trajectory.sha256,
            'duration_s':env.trajectory.duration_s,'phase_starts':env.trajectory.phase_starts},
        'task_successes':sum(e['task_success'] for e in episodes),
        'strict_successes':sum(e['strict_success'] for e in episodes),
        'num_episodes':len(episodes),'wall_seconds':time.monotonic()-started,'episodes':episodes,
        'trace_sha256':digest(output/'trace.npz'),
        'environment_protocol_path':str((env.output_dir/'environment_protocol.json').resolve()),
        'environment_protocol_sha256':digest(env.output_dir/'environment_protocol.json'),
        'trace_robot_alignment':'sample i uses flattened state_before[i+4]; final four physical samples have no saved post-action q',
        'tactile_saved':save_tactile,'training':False}
    write_json(output/'evaluation.json',result)
    return result,prefix


def run(args, protocol):
    from .adaptive_env import AdaptiveConfig
    from .multiobject_env import MultiObjectVecEnv
    from .multiobject_protocol import validate_protocol
    if args.split!='preflight':validate_protocol(protocol,require_demonstrations=True)
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    specifications = model_specs(protocol,args.training_root,args.models,args.checkpoint_stage)
    selected={}
    if args.split=='test' and specifications and not args.selection_json:
        raise ValueError('Model test requires --selection-json from the declared validation procedure')
    if args.selection_json:
        if args.training_root or args.models:raise ValueError('selection-json supplies all model paths; do not combine model flags')
        selected=read_selection(args.selection_json,protocol,args.experiment_json)
        specifications=[]
        for name,record in selected.items():
            path=Path(record['path']).expanduser().resolve()
            if digest(path)!=record['sha256']:raise ValueError('Selected checkpoint hash changed')
            specifications.append((name,path))
    models = {}
    checkpoint_tactile_models = {}
    for name,path in specifications:
        with torch.device('cpu'):
            model = PPO.load(path,device='cpu')
        model_manifest_path=path.parent/'manifest.json'
        saved_manifest=json.loads(model_manifest_path.read_text()) if model_manifest_path.is_file() else {}
        if saved_manifest.get('experiment_sha256') != digest(args.experiment_json):
            raise ValueError('Checkpoint must originate from this exact frozen experiment')
        saved_stage=saved_manifest.get('stage',{}).get('name')
        if name in selected and saved_stage!=selected[name]['checkpoint_stage']:
            raise ValueError('Selected stage contradicts checkpoint provenance')
        checkpoint_tactile_models[name]=saved_manifest.get('tactile_model',legacy_tactile_model())
        models[name]=(model,*controller_metadata(name,model),saved_stage)
    gains = protocol['evaluation']['fixed_gains'] if args.fixed_gains is None else args.fixed_gains
    controllers = [(name,'model',name) for name in models]
    controllers += [(f'fixed_{g:g}'.replace('.','p'),'fixed',g) for g in gains]
    if not controllers: raise ValueError('At least one model or fixed gain must be evaluated')
    conditions = (protocol['splits'][args.split] if args.split != 'preflight' else
        [{'condition_id':'demonstration_easy',**protocol['demonstration_condition']}])
    objects = [o for o in protocol['objects'] if args.object_id is None or o['object_id']==args.object_id]
    if not objects: raise ValueError('Unknown object_id')
    config = AdaptiveConfig(**protocol['training']['stages'][-1]['config'])
    for gain in gains:
        if not config.gain_min <= gain <= config.gain_max: raise ValueError('Fixed gain outside action bounds')
    manifest = {'experiment_path':str(args.experiment_json.resolve()),
        'experiment_sha256':digest(args.experiment_json),'experiment':protocol,'split':args.split,
        'config':asdict(config),'physics_seed':args.physics_seed,'source_sha256':source_snapshot(args.output),
        'current_tactile_model':current_tactile_model(),
        'checkpoint_sensor_semantics_all_match':all(tactile_models_match(value,current_tactile_model()) for value in checkpoint_tactile_models.values()),
        'sensor_semantics_comparison_policy':'A missing or mismatched checkpoint sensor profile makes its evaluation a cross-version diagnostic, not a validated same-semantics model comparison.',
        'models':{name:{'path':str(path.resolve()),'sha256':digest(path),
            'checkpoint_tactile_model':checkpoint_tactile_models[name],
            'checkpoint_and_current_tactile_models_match':tactile_models_match(checkpoint_tactile_models[name],current_tactile_model()),
            'training_steps':models[name][0].num_timesteps,'training_seed':models[name][1],
            'method':models[name][2],'checkpoint_stage':models[name][3]} for name,path in specifications},
        'fixed_gains':gains,'training':False,'checkpoint_selection':protocol['training']['checkpoint_selection']}
    if args.selection_json:
        manifest['selection']={'path':str(args.selection_json.resolve()),'sha256':digest(args.selection_json)}
    write_json(args.output/'manifest.json',manifest)
    results, env = [], None
    try:
        for obj in objects:
            for chunk_idx,start in enumerate(range(0,len(conditions),args.lanes_per_scene)):
                rows = conditions[start:start+args.lanes_per_scene]
                chunk_name=f'chunk_{chunk_idx:03d}'
                scene_path=args.output/'environment'/obj['object_id']/chunk_name
                write_json(args.output/'status.json',{'status':'initializing','object_id':obj['object_id'],'chunk':chunk_idx})
                env=MultiObjectVecEnv(object_spec=obj['object_spec'],trajectory_path=obj['trajectory_path'],
                    n_envs=len(rows),seed=args.physics_seed,mode='eval',output_dir=scene_path,
                    config=config,material_conditions=[[c['mass_kg'],c['friction']] for c in rows],audit_prefix=True)
                reference=None
                for name,kind,value in controllers:
                    write_json(args.output/'status.json',{'status':'running','object_id':obj['object_id'],
                        'chunk':chunk_idx,'controller':name,'completed_batches':len(results)})
                    model,seed,method,checkpoint_stage=models[value] if kind=='model' else (None,0,name,None)
                    out=args.output/name/f"shape_{obj['object_id']}"/chunk_name
                    result,prefix=evaluate_controller(env,name=name,model=model,
                        fixed_gain=value if kind=='fixed' else None,output=out,conditions=rows,
                        object_id=obj['object_id'],split=args.split,training_seed=seed,method=method,
                        prefix_reference=reference,save_tactile=not args.omit_tactile_trace,
                        checkpoint_stage=checkpoint_stage)
                    if reference is None: reference=prefix
                    results.append(result)
                    write_json(args.output/'results.json',results)
                    print(json.dumps({k:v for k,v in result.items() if k not in ('episodes','conditions','trajectory')}),flush=True)
                env.close();env=None
        write_json(args.output/'status.json',{'status':'completed','batches':len(results),
            'episodes':sum(r['num_episodes'] for r in results)})
    except BaseException as exc:
        write_json(args.output/'status.json',{'status':'failed','error':repr(exc),'traceback':traceback.format_exc()})
        raise
    finally:
        if env is not None:env.close()


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment-json',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--split',choices=['validation','test','preflight'],required=True)
    parser.add_argument('--training-root',type=Path,help='Loads all final-stage seed/modality checkpoints')
    parser.add_argument('--checkpoint-stage',choices=['stage1','stage2'],help='Training-root stage for validation; selected test uses selection-json')
    parser.add_argument('--selection-json',type=Path,help='Frozen validation selection with selected_models paths, hashes and stages')
    parser.add_argument('--models',nargs='*',default=[],help='Explicit name=checkpoint.zip, e.g. seed1_tactile=/...')
    parser.add_argument('--fixed-gains',nargs='*',type=float,default=None,
        help='Omit flag for protocol fixed controls; supply flag without values to evaluate only models')
    parser.add_argument('--object-id',help='Evaluate only this shape, mainly for physical preflight')
    parser.add_argument('--lanes-per-scene',type=int,default=12)
    parser.add_argument('--physics-seed',type=int,default=7000)
    parser.add_argument('--omit-tactile-trace',action='store_true',help='Save physics/proprio/action but omit raw tactile to reduce disk')
    args=parser.parse_args(argv)
    if args.lanes_per_scene<1:parser.error('lanes-per-scene must be positive')
    run(args,load_experiment(args.experiment_json,object_ids=[args.object_id] if args.object_id else None))


if __name__=='__main__':main()
