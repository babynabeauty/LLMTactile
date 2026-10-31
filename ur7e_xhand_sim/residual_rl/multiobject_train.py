"""Shared-policy multi-shape PPO, with independent per-shape physical clocks.

The experiment document owns trajectories, split membership, reward stages and
optimizer settings. Train and evaluation never infer conditions from filenames.
``--dry-run`` validates inputs and reports rollout RAM without importing Genesis.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import traceback

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.logger import HumanOutputFormat, Logger
from stable_baselines3.common.vec_env import VecEnv, VecCheckNan

from .adaptive_train import (AdaptiveTrainingCallback, distribution_metadata,
    initialize_constant_gain, optimizer_metadata, training_policy_kwargs,
    validate_resume_exploration, validate_resume_extractors, validate_resume_optimizer)
from .reward_scale import wrap_training_rewards, resolve_training_reward_scale
from .train import ROOT, JsonMetricsWriter, configure_seed, digest, write_json
from .tactile_provenance import current_tactile_model, require_checkpoint_tactile_compatibility


class CPUTrainingPPO(PPO):
    """Keep lazy optimizer scalars on CPU without changing Genesis stepping."""
    def train(self):
        with torch.device('cpu'):
            return super().train()


def load_experiment(path, *, require_trajectories=True, object_ids=None):
    from .solver_profile import normalize_solver_profile,trajectory_solver_profile
    path = Path(path).expanduser().resolve()
    protocol = json.loads(path.read_text())
    expected_solver=normalize_solver_profile(protocol.get('solver_profile'))
    if not isinstance(protocol.get('objects'), list) or not protocol['objects']:
        raise ValueError('experiment objects must be a nonempty list')
    ids = [obj['object_id'] for obj in protocol['objects']]
    if len(set(ids)) != len(ids) or any(not str(name).isidentifier() for name in ids):
        raise ValueError('object_id values must be distinct safe identifiers')
    all_pairs = set()
    for split in ('train', 'validation', 'test'):
        rows = protocol['splits'][split]
        if not rows:
            raise ValueError(f'{split} has no conditions')
        ids = [r['condition_id'] for r in rows]
        if len(set(ids)) != len(ids):
            raise ValueError('Duplicate condition IDs')
        pairs = [(float(r['mass_kg']), float(r['friction'])) for r in rows]
        if len(set(pairs)) != len(pairs) or set(pairs) & all_pairs:
            raise ValueError('Material splits must be disjoint without duplicates')
        if not np.isfinite(pairs).all() or any(m <= 0 or not .01 <= mu <= 5 for m, mu in pairs):
            raise ValueError('Invalid physical condition')
        all_pairs.update(pairs)
    for obj in protocol['objects']:
        required = require_trajectories and (object_ids is None or obj['object_id'] in object_ids)
        if not obj.get('trajectory_path'):
            if required:
                raise ValueError(f"{obj['object_id']} has no verified demonstration")
            continue
        source = Path(obj['trajectory_path']).expanduser()
        if not source.is_absolute():
            source = path.parent/source
        obj['trajectory_path'] = str(source.resolve())
        if required:
            expected = obj.get('trajectory_sha256')
            if not expected or digest(source) != expected:
                raise ValueError(f"Missing/mismatched demonstration hash: {obj['object_id']}")
            actual_solver,_=trajectory_solver_profile(source)
            if actual_solver!=expected_solver:
                raise ValueError(f"Demonstration solver profile differs from the uniform experiment profile: {obj['object_id']}")
    return protocol


def concatenate_observations(observations):
    return {key:np.concatenate([obs[key] for obs in observations], axis=0)
            for key in observations[0]}


class ComposedShapeVecEnv(VecEnv):
    """Concatenate native batched scenes without synchronizing their endings.

    Each child already supplies terminal_observation and auto-resets its own
    lanes. Other shapes continue uninterrupted when one child reaches hold end.
    No truth about shape/material is appended to actor observations.
    """
    def __init__(self, envs, object_ids, conditions, *, training_seed, method, split):
        if not envs or len(envs) != len(object_ids):
            raise ValueError('One object ID is required per child scene')
        self.envs, self.object_ids = envs, list(object_ids)
        self.conditions = list(conditions)
        self.training_seed, self.method, self.split = training_seed, method, split
        self.offsets = np.r_[0, np.cumsum([env.num_envs for env in envs])]
        for env in envs:
            if env.num_envs != len(conditions):
                raise ValueError('All shapes must have the same material lane count')
            if env.observation_space != envs[0].observation_space or env.action_space != envs[0].action_space:
                raise ValueError('All scene observation/action spaces must match')
        self.render_mode = None
        self.config = envs[0].config
        self._closed = False
        super().__init__(int(self.offsets[-1]), envs[0].observation_space, envs[0].action_space)

    def action_for_gain(self, gain):
        return self.envs[0].action_for_gain(gain)

    def reset(self):
        return concatenate_observations([env.reset() for env in self.envs])

    def step_async(self, actions):
        values = np.asarray(actions)
        if values.shape != (self.num_envs, *self.action_space.shape):
            raise ValueError('Wrong composed action shape')
        for i, env in enumerate(self.envs):
            env.step_async(values[self.offsets[i]:self.offsets[i+1]])

    def step_wait(self):
        outputs = [env.step_wait() for env in self.envs]
        infos = []
        for object_id, (_, _, _, group_infos) in zip(self.object_ids, outputs):
            for condition, info in zip(self.conditions, group_infos):
                if 'episode_metrics' in info:
                    info['episode_metrics'].update(object_id=object_id,
                        condition_id=condition['condition_id'], split=self.split,
                        training_seed=self.training_seed, method=self.method)
                infos.append(info)
        return (concatenate_observations([o[0] for o in outputs]),
                np.concatenate([o[1] for o in outputs]),
                np.concatenate([o[2] for o in outputs]), infos)

    def close(self):
        if not self._closed:
            # Native scenes use a process-wide Genesis runtime. Destroy it once
            # after every child is done; repeated child.close() destroys twice.
            self.envs[0].close()
            for env in self.envs:
                env._closed = True
            self._closed = True

    def _child_indices(self, indices):
        for index in self._get_indices(indices):
            child = int(np.searchsorted(self.offsets[1:], index, side='right'))
            yield child, int(index-self.offsets[child])

    def get_attr(self, attr_name, indices=None):
        return [self.envs[child].get_attr(attr_name, indices=[local])[0]
                for child, local in self._child_indices(indices)]

    def set_attr(self, attr_name, value, indices=None):
        if indices is not None:
            raise NotImplementedError('Only shared attributes can be set')
        for env in self.envs:
            env.set_attr(attr_name, value)

    def env_method(self, method_name, *args, indices=None, **kwargs):
        raise NotImplementedError('Call explicitly batched methods on child scenes')

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False for _ in self._get_indices(indices)]


def build_environment(protocol, stage, output, *, seed, method, max_conditions=None):
    from .adaptive_env import AdaptiveConfig
    from .multiobject_env import MultiObjectVecEnv
    conditions = protocol['splits']['train'][:max_conditions]
    pairs = [[c['mass_kg'], c['friction']] for c in conditions]
    envs = []
    try:
        for obj in protocol['objects']:
            envs.append(MultiObjectVecEnv(object_spec=obj['object_spec'],
                trajectory_path=obj['trajectory_path'], n_envs=len(pairs),
                seed=seed, material_conditions=pairs, mode='train',
                config=AdaptiveConfig(**stage['config']), audit_prefix=True,
                output_dir=output/'environment'/obj['object_id']))
        return ComposedShapeVecEnv(envs, [o['object_id'] for o in protocol['objects']],
            conditions, training_seed=seed, method=method, split='train')
    except BaseException:
        if envs:
            envs[0].close()
        raise


def source_snapshot(output):
    files = set((ROOT/'ur7e_xhand_sim').rglob('*.py'))
    files.update((ROOT/'Genesis/genesis').rglob('*.py'))
    files.update(path for path in (ROOT/'ur7e_xhand_sim/sources').rglob('*') if path.is_file())
    result = {}
    for source in sorted(files):
        if not source.is_file():
            continue
        relative = source.relative_to(ROOT)
        destination = output/'source_snapshot'/relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
        result[str(relative)] = digest(destination)
    return result


def weight_digest(model):
    h = hashlib.sha256()
    for name, tensor in sorted(model.policy.state_dict().items()):
        h.update(name.encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


class MultiObjectTrainingCallback(AdaptiveTrainingCallback):
    def __init__(self, output, checkpoint_every, object_ids, lanes):
        super().__init__(output,checkpoint_every)
        self.shape_exposure={name:{'transitions':0,'episodes':0,'task_successes':0}
                             for name in object_ids}
        self.lanes=lanes

    def _on_step(self):
        for row in self.shape_exposure.values():row['transitions']+=self.lanes
        for done,info in zip(self.locals['dones'],self.locals['infos']):
            if done:
                metrics=info['episode_metrics'];row=self.shape_exposure[metrics['object_id']]
                row['episodes']+=1;row['task_successes']+=int(metrics['task_success'])
        return super()._on_step()

    def _on_rollout_end(self):
        super()._on_rollout_end()
        status=json.loads((self.output/'status.json').read_text())
        write_json(self.output/'status.json',{**status,'shape_exposure':self.shape_exposure})


def make_model(env, settings, seed, method, *, resume=None):
    from .normalized_sde_policy import NormalizedSDEMultiInputPolicy
    if not settings.get('shared_encoder',True):
        raise ValueError('This protocol fixes shared encoders to match the established baseline')
    if not settings.get('use_sde',True) or not settings.get('normalize_sde_latent',True):
        raise ValueError('This protocol requires normalized gSDE')
    scale = float(settings.get('reward_scale', 1.))
    use_touch = method == 'tactile'
    env = wrap_training_rewards(env, scale)
    with torch.device('cpu'):
        if resume:
            model = CPUTrainingPPO.load(resume, env=env, device='cpu')
            validate_resume_exploration(model, settings.get('use_sde', True),
                                       settings.get('normalize_sde_latent', True))
            validate_resume_extractors(model, False, not use_touch)
            validate_resume_optimizer(model, settings.get('adam_eps', 1e-5))
            resolve_training_reward_scale(scale, checkpoint_model=model, explicitly_requested=True)
            model.set_random_seed(seed)
        else:
            model = CPUTrainingPPO(NormalizedSDEMultiInputPolicy, env, seed=seed, device='cpu', verbose=1,
                n_steps=int(settings.get('n_steps',128)), batch_size=int(settings.get('batch_size',256)),
                n_epochs=int(settings.get('n_epochs',settings.get('epochs',5))),
                learning_rate=float(settings.get('learning_rate',3e-4)),
                gamma=float(settings.get('gamma',1.)), gae_lambda=float(settings.get('gae_lambda',.95)),
                use_sde=bool(settings.get('use_sde',True)),
                sde_sample_freq=int(settings.get('sde_sample_freq',30)),
                clip_range=.2, ent_coef=.001, vf_coef=.5, max_grad_norm=.5, target_kl=.03,
                policy_kwargs=training_policy_kwargs(use_tactile=use_touch,
                    action_std_init=float(settings.get('action_std_init',.12)),
                    adam_eps=float(settings.get('adam_eps',1e-5))))
            if settings.get('initial_gain') is not None:
                initialize_constant_gain(model, env, env.unwrapped.config, float(settings['initial_gain']))
    if model.n_steps*env.num_envs % model.batch_size:
        raise ValueError('Rollout size must divide into complete minibatches')
    model.training_reward_scale = scale
    return model, env


def validate_completed_run(output, protocol, stage, seed, method, experiment_sha256):
    output=Path(output)
    manifest=json.loads((output/'manifest.json').read_text())
    status=json.loads((output/'status.json').read_text())
    checkpoint=output/'final_model.zip'
    expected_lanes=len(protocol['objects'])*len(protocol['splits']['train'])
    expected=(manifest.get('experiment_sha256')==experiment_sha256 and
        manifest.get('stage',{}).get('name')==stage['name'] and
        manifest.get('seed')==seed and manifest.get('method')==method and
        manifest.get('diagnostic_max_conditions') is None and
        manifest.get('requested_timesteps')==stage['requested_timesteps'] and
        manifest.get('actual_model_config',{}).get('n_envs')==expected_lanes and
        status.get('status')=='completed' and checkpoint.is_file() and
        status.get('final_checkpoint_sha256')==digest(checkpoint) and
        status.get('requested_additional_timesteps')==stage['requested_timesteps'])
    if not expected:
        raise ValueError('Completed run provenance, identity, budget, lanes or checkpoint hash is invalid')
    rollout=expected_lanes*int(protocol['training']['n_steps'])
    expected_actual=int(np.ceil(stage['requested_timesteps']/rollout))*rollout
    if status.get('actual_additional_timesteps')!=expected_actual:
        raise ValueError('Completed run actual budget differs from complete-rollout schedule')
    return manifest,status


def validate_stage_resume(args, protocol):
    stages=protocol['training']['stages']
    position=next(i for i,stage in enumerate(stages) if stage['name']==args.stage)
    if position==0:
        if args.resume:raise ValueError('Stage1 starts from paired fresh initialization; resume is forbidden')
        return
    if not args.resume:
        raise ValueError('Later reward stages must resume the preceding completed full stage')
    if args.resume.name!='final_model.zip':
        raise ValueError('Later stages require the preceding final_model.zip')
    prior,_=validate_completed_run(args.resume.parent,protocol,stages[position-1],
                                  args.seed,args.method,digest(args.experiment_json))
    if prior.get('input_checkpoint') is not None and position==1:
        raise ValueError('Stage1 must have been trained from a fresh initialization')


def run(args, protocol):
    from .multiobject_protocol import validate_protocol
    validate_protocol(protocol,require_demonstrations=True)
    stage = next(s for s in protocol['training']['stages'] if s['name'] == args.stage)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    env = model = logger = callback = None
    initial_steps = 0
    requested = args.total_timesteps or int(stage['requested_timesteps'])
    try:
        validate_stage_resume(args,protocol)
        require_checkpoint_tactile_compatibility(args.resume)
        manifest = {'created_utc':datetime.now(timezone.utc).isoformat(),
            'experiment_path':str(args.experiment_json.resolve()),
            'experiment_sha256':digest(args.experiment_json), 'experiment':protocol,
            'source_sha256':source_snapshot(output), 'stage':stage, 'seed':args.seed,
            'tactile_model':current_tactile_model(),
            'method':args.method, 'requested_timesteps':requested,
            'diagnostic_max_conditions':args.max_conditions,
            'input_checkpoint':str(args.resume.resolve()) if args.resume else None,
            'input_checkpoint_sha256':digest(args.resume) if args.resume else None,
            'conditioning':'No object identity, material ground truth or object pose added to actor inputs',
            'scene_clock':'Each shape independently auto-resets; shared PPO batches every shape at each vector step'}
        write_json(output/'manifest.json', manifest)
        write_json(output/'status.json', {'status':'initializing'})
        configure_seed(args.seed,4)
        raw_env = build_environment(protocol, stage, output, seed=args.seed,
                                    method=args.method, max_conditions=args.max_conditions)
        env = VecCheckNan(raw_env, raise_exception=True, check_inf=True)
        model, env = make_model(env, protocol['training'], args.seed, args.method, resume=args.resume)
        initial_steps = model.num_timesteps
        manifest['initial_policy_sha256'] = weight_digest(model)
        manifest['actual_model_config'] = {'n_envs':env.num_envs,
            'n_steps':model.n_steps,'batch_size':model.batch_size,'n_epochs':model.n_epochs,
            'gamma':model.gamma,'gae_lambda':model.gae_lambda,
            'shared_features':model.policy.share_features_extractor,
            'ppo_train_device_context':'CPU only during optimizer/minibatch updates; Genesis stepping context unchanged',
            'policy_distribution':distribution_metadata(model), **optimizer_metadata(model),
            'initial_timesteps':initial_steps,'reward_scale':model.training_reward_scale}
        write_json(output/'manifest.json',manifest)
        model.save(output/'initial_model')
        logger = Logger(str(output), [HumanOutputFormat(sys.stdout),JsonMetricsWriter(output/'training_metrics.jsonl')])
        model.set_logger(logger)
        (output/'checkpoints').mkdir()
        callback = MultiObjectTrainingCallback(output,int(protocol['training'].get('checkpoint_every',10**12)),
            [obj['object_id'] for obj in protocol['objects']], raw_env.envs[0].num_envs)
        model.learn(total_timesteps=requested, callback=callback,
            reset_num_timesteps=not bool(args.resume),log_interval=1,progress_bar=False)
        model.save(output/'final_model')
        for obj in protocol['objects']:
            if digest(Path(obj['trajectory_path'])) != obj['trajectory_sha256']:
                raise RuntimeError('Demonstration mutated during training')
        write_json(output/'status.json', {'status':'completed','timesteps':model.num_timesteps,
            'requested_additional_timesteps':requested,'actual_additional_timesteps':model.num_timesteps-initial_steps,
            'completed_episodes':callback.episode_count,'task_successes':callback.task_successes,
            'strict_successes':callback.strict_successes,'shape_exposure':callback.shape_exposure,
            'optimizer_state_devices':sorted({str(value.device)
                for state in model.policy.optimizer.state.values()
                for value in state.values() if isinstance(value,torch.Tensor)}),
            'final_checkpoint_sha256':digest(output/'final_model.zip')})
    except BaseException as exc:
        if model is not None:
            model.save(output/'interrupted_model')
        write_json(output/'status.json',{'status':'failed','error':repr(exc),
            'traceback':traceback.format_exc(),'timesteps':getattr(model,'num_timesteps',0)})
        raise
    finally:
        if callback is not None: callback.handle.close()
        if logger is not None: logger.close()
        if env is not None: env.close()


def run_suite(args, protocol):
    """Sequential fresh processes release Genesis GPU memory between stages."""
    args.output.mkdir(parents=True,exist_ok=True)
    for seed in protocol['training']['seeds']:
        for method in protocol['training']['methods']:
            previous = None
            for stage in protocol['training']['stages']:
                out = args.output/f"seed{seed}_{method}_{stage['name']}"
                status_path = out/'status.json'
                if status_path.is_file() and json.loads(status_path.read_text()).get('status') == 'completed':
                    checkpoint = out/'final_model.zip'
                    validate_completed_run(out,protocol,stage,seed,method,digest(args.experiment_json))
                    previous = checkpoint
                    continue
                command = [sys.executable,'-m',__package__+'.multiobject_train',
                    '--experiment-json',str(args.experiment_json.resolve()),'--output',str(out.resolve()),
                    '--seed',str(seed),'--method',method,'--stage',stage['name']]
                if previous: command += ['--resume',str(previous.resolve())]
                write_json(args.output/'suite_status.json',{'status':'running','seed':seed,'method':method,'stage':stage['name']})
                with (args.output/f"seed{seed}_{method}_{stage['name']}.log").open('w') as log:
                    subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT,cwd=ROOT)
                previous = out/'final_model.zip'
    write_json(args.output/'suite_status.json',{'status':'completed'})


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment-json',type=Path,required=True)
    p.add_argument('--output',type=Path)
    p.add_argument('--seed',type=int,default=1)
    p.add_argument('--method',choices=['tactile','no_tactile'],default='tactile')
    p.add_argument('--stage',default='stage1')
    p.add_argument('--resume',type=Path)
    p.add_argument('--suite',action='store_true')
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--max-conditions',type=int,help='Diagnostic lane subset only; never used by --suite')
    p.add_argument('--total-timesteps',type=int,help='Diagnostic budget override, recorded in manifest')
    args=p.parse_args(argv)
    protocol=load_experiment(args.experiment_json,require_trajectories=not args.dry_run)
    if args.seed<0:p.error('seed must be nonnegative')
    if args.total_timesteps is not None and args.total_timesteps<1:p.error('total-timesteps must be positive')
    if args.stage not in [s['name'] for s in protocol['training']['stages']]:p.error('Unknown reward stage')
    if args.max_conditions is not None and args.max_conditions < 1: p.error('max-conditions must be positive')
    lanes=min(args.max_conditions or len(protocol['splits']['train']),len(protocol['splits']['train']))
    samples=lanes*len(protocol['objects'])*int(protocol['training'].get('n_steps',128))
    if samples % int(protocol['training'].get('batch_size',256)): p.error('Rollout is not divisible by batch size')
    if args.dry_run:
        print(json.dumps({'objects':[o['object_id'] for o in protocol['objects']],
            'lanes_per_shape':lanes,'total_lanes':lanes*len(protocol['objects']),
            'rollout_samples':samples,'rollout_observations_MiB':samples*(21600+948)*4/2**20,
            'gpu_memory':'Requires physical preflight; no GPU initialized by dry-run'}))
        return
    if args.output is None: p.error('--output is required unless --dry-run')
    if args.suite:
        if args.max_conditions or args.total_timesteps or args.resume: p.error('Diagnostic/resume flags cannot change suite')
        run_suite(args,protocol)
    else:
        run(args,protocol)


if __name__=='__main__':main()
