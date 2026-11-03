"""v1 PPO training: residual hand-shape control from close, one policy shared by every scene.

Methods differ only in actor inputs: tactile (touch + Astra level), no_tactile
(proprioception + Astra level) and tactile_no_level (touch, level zeroed).
``--smoke`` runs one tiny rollout and update as an integration check.
``--dry-run`` validates the experiment and base libraries without Genesis.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import traceback

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import HumanOutputFormat, KVWriter, Logger
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import VecCheckNan

from .adaptive_policy import policy_kwargs
from .core import ROOT
from .normalized_sde_policy import NormalizedSDEMultiInputPolicy
from .tactile_provenance import current_tactile_model, require_checkpoint_tactile_compatibility
from .v1_bases import TRAINING_LABELS, experiment_action_joints, load_experiment, load_library, sha256
from .v1_task import V1Config, SamplerConfig

METHODS = {'tactile': dict(use_tactile=True, use_level=True),
           'no_tactile': dict(use_tactile=False, use_level=True),
           'tactile_no_level': dict(use_tactile=True, use_level=False)}


def to_json(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False,
                      default=lambda v: v.tolist() if hasattr(v, 'tolist') else str(v))


def write_json(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(to_json(value)+'\n')
    temporary.replace(path)


def configure_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(4)
    set_random_seed(seed, using_cuda=torch.cuda.is_available())
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def source_snapshot(output):
    files = set((ROOT/'ur7e_xhand_sim').rglob('*.py'))
    files.update((ROOT/'Genesis/genesis').rglob('*.py'))
    files.update(path for path in (ROOT/'ur7e_xhand_sim/sources').rglob('*') if path.is_file())
    result = {}
    for source in sorted(files):
        relative = source.relative_to(ROOT)
        destination = output/'source_snapshot'/relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
        result[str(relative)] = sha256(destination)
    return result


def weight_digest(model):
    h = hashlib.sha256()
    for name, tensor in sorted(model.policy.state_dict().items()):
        h.update(name.encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


class CPUTrainingPPO(PPO):
    """Keep lazy optimizer scalars on CPU without changing Genesis stepping."""
    def train(self):
        with torch.device('cpu'):
            return super().train()


class JsonMetricsWriter(KVWriter):
    def __init__(self, path):
        self.handle = path.open('w', buffering=1)

    def write(self, key_values, key_excluded, step=0):
        values = {k: v for k, v in key_values.items() if 'json' not in key_excluded.get(k, ())}
        self.handle.write(to_json({'step': int(step), **values})+'\n')

    def close(self):
        self.handle.close()


class TrainingCallback(BaseCallback):
    """episodes.jsonl, periodic checkpoints and status.json with per-scene exposure."""
    def __init__(self, output, checkpoint_every, object_ids, lanes):
        super().__init__()
        self.output, self.checkpoint_every, self.lanes = output, checkpoint_every, lanes
        self.episode_count = self.task_successes = 0
        self.shape_exposure = {name: {'transitions': 0, 'episodes': 0, 'task_successes': 0} for name in object_ids}
        self.handle = (output/'episodes.jsonl').open('w', buffering=1)

    def _on_training_start(self):
        self.returns = np.zeros(self.training_env.num_envs)
        self.lengths = np.zeros(self.training_env.num_envs, dtype=np.int64)
        self.next_checkpoint = self.num_timesteps+self.checkpoint_every

    def _on_step(self):
        for row in self.shape_exposure.values():
            row['transitions'] += self.lanes
        self.returns += self.locals['rewards']
        self.lengths += 1
        for i, (done, info) in enumerate(zip(self.locals['dones'], self.locals['infos'])):
            if not done:
                continue
            metrics = info['episode_metrics']
            row = self.shape_exposure[metrics['object_id']]
            row['episodes'] += 1
            row['task_successes'] += int(metrics['task_success'])
            self.task_successes += int(metrics['task_success'])
            self.handle.write(to_json({'episode_index': self.episode_count, 'env_index': i,
                'return': float(self.returns[i]), 'policy_steps': int(self.lengths[i]),
                'training_timesteps': self.num_timesteps, 'success': bool(metrics['success']),
                'episode_metrics': metrics})+'\n')
            self.episode_count += 1
            self.returns[i] = self.lengths[i] = 0
        if self.num_timesteps >= self.next_checkpoint:
            self.model.save(self.output/'checkpoints'/f'ppo_{self.num_timesteps:09d}')
            self.next_checkpoint = self.num_timesteps+self.checkpoint_every
        return True

    def _on_rollout_end(self):
        self.logger.record('rollout/cumulative_task_success_rate', self.task_successes/max(self.episode_count, 1))
        write_json(self.output/'status.json', {'status': 'running', 'timesteps': self.num_timesteps,
            'completed_episodes': self.episode_count, 'task_successes': self.task_successes,
            'shape_exposure': self.shape_exposure, 'updated_utc': datetime.now(timezone.utc).isoformat()})


def stage_of(experiment, name):
    stage = next((s for s in experiment['stages'] if s['name'] == name), None)
    if stage is None:
        raise ValueError(f'Unknown stage {name}')
    return stage


def build_environment(experiment, stage, output, *, seed, method, lanes=None, fixed_gain=None,
                      split='train', labels=TRAINING_LABELS):
    from .v1_env import V1VecEnv, V1ComposedVecEnv
    config = V1Config(**experiment.get('config', {}))
    sampler = SamplerConfig(**{**experiment.get('sampler', {}), 'levels': tuple(stage['levels'])})
    libraries = [load_library(obj['base_library'], split=split, labels=labels) for obj in experiment['objects']]
    joints = experiment_action_joints(experiment, libraries)
    envs = []
    try:
        for i, (obj, (library, bases)) in enumerate(zip(experiment['objects'], libraries)):
            envs.append(V1VecEnv(library['object_spec'], bases, n_envs=lanes or experiment['lanes_per_object'],
                seed=seed*1000+i, output_dir=Path(output)/'environment'/obj['object_id'],
                object_id=obj['object_id'], hand_shape=library['hand_shape'], action_joints=joints,
                config=config, sampler=sampler,
                use_level=METHODS[method]['use_level'],
                joint_friction_Nm=experiment.get('joint_friction_Nm', [0., 0.]),
                tactile_noise_N=experiment.get('tactile_noise_N', 0.), fixed_gain=fixed_gain,
                augment=split == 'train'))
        return V1ComposedVecEnv(envs)
    except BaseException:
        if envs:
            envs[0].close()
        raise


def make_model(env, settings, seed, method, *, resume=None):
    use_touch = METHODS[method]['use_tactile']
    with torch.device('cpu'):
        if resume:
            model = CPUTrainingPPO.load(resume, env=env, device='cpu')
            if model.policy.features_extractor.use_tactile != use_touch:
                raise ValueError(f'{resume} was trained with a different tactile input than {method}')
            model.set_random_seed(seed)
        else:
            kwargs = policy_kwargs(use_tactile=use_touch, action_std_init=float(settings['action_std_init']))
            kwargs['optimizer_kwargs'] = {'eps': float(settings['adam_eps'])}
            model = CPUTrainingPPO(NormalizedSDEMultiInputPolicy, env, seed=seed, device='cpu', verbose=1,
                n_steps=int(settings['n_steps']), batch_size=int(settings['batch_size']),
                n_epochs=int(settings['n_epochs']), learning_rate=float(settings['learning_rate']),
                gamma=float(settings['gamma']), gae_lambda=float(settings['gae_lambda']), use_sde=True,
                sde_sample_freq=int(settings['sde_sample_freq']), clip_range=.2, ent_coef=.001, vf_coef=.5,
                max_grad_norm=.5, target_kl=.03, policy_kwargs=kwargs)
            # Start exactly on the base: zero residual mean, exploration unchanged.
            with torch.no_grad():
                model.policy.action_net.weight.zero_()
                model.policy.action_net.bias.zero_()
    if model.n_steps*env.num_envs % model.batch_size:
        raise ValueError('Rollout size must divide into complete minibatches')
    return model


def run(args, experiment):
    stage = stage_of(experiment, args.stage)
    settings = dict(experiment['training'])
    lanes = None
    requested = args.total_timesteps or int(stage['requested_timesteps'])
    if args.smoke:
        lanes = 2
        settings.update(n_steps=32, n_epochs=1, batch_size=32*lanes*len(experiment['objects']))
        requested = settings['batch_size']
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    env = model = logger = callback = None
    try:
        require_checkpoint_tactile_compatibility(args.resume)
        manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(),
            experiment_path=str(args.experiment_json.resolve()), experiment_sha256=sha256(args.experiment_json),
            experiment=experiment, stage=stage, seed=args.seed, method=args.method, inputs=METHODS[args.method],
            smoke=args.smoke, requested_timesteps=requested, training_settings=settings,
            tactile_model=current_tactile_model(), source_sha256=source_snapshot(output),
            input_checkpoint=str(args.resume.resolve()) if args.resume else None,
            input_checkpoint_sha256=sha256(args.resume) if args.resume else None,
            conditioning='No object identity, material ground truth, F_break or object pose in actor inputs')
        write_json(output/'manifest.json', manifest)
        write_json(output/'status.json', {'status': 'initializing'})
        configure_seed(args.seed)
        raw = build_environment(experiment, stage, output, seed=args.seed, method=args.method, lanes=lanes)
        env = VecCheckNan(raw, raise_exception=True, check_inf=True)
        model = make_model(env, settings, args.seed, args.method, resume=args.resume)
        initial = model.num_timesteps
        manifest.update(initial_policy_sha256=weight_digest(model), n_envs=env.num_envs,
            bases_per_object={e.object_id: [b.base_id for b in e.bases] for e in raw.envs},
            excluded_bases={e.object_id: e.excluded_bases for e in raw.envs})
        write_json(output/'manifest.json', manifest)
        model.save(output/'initial_model')
        logger = Logger(str(output), [HumanOutputFormat(sys.stdout), JsonMetricsWriter(output/'training_metrics.jsonl')])
        model.set_logger(logger)
        (output/'checkpoints').mkdir()
        callback = TrainingCallback(output, int(settings.get('checkpoint_every', 10**12)),
            [o['object_id'] for o in experiment['objects']], raw.envs[0].num_envs)
        model.learn(total_timesteps=requested, callback=callback, reset_num_timesteps=not bool(args.resume),
                    log_interval=1, progress_bar=False)
        model.save(output/'final_model')
        write_json(output/'status.json', {'status': 'completed', 'timesteps': model.num_timesteps,
            'requested_additional_timesteps': requested, 'actual_additional_timesteps': model.num_timesteps-initial,
            'completed_episodes': callback.episode_count, 'task_successes': callback.task_successes,
            'shape_exposure': callback.shape_exposure, 'final_checkpoint_sha256': sha256(output/'final_model.zip')})
    except BaseException as exc:
        if model is not None:
            model.save(output/'interrupted_model')
        write_json(output/'status.json', {'status': 'failed', 'error': repr(exc),
            'traceback': traceback.format_exc(), 'timesteps': getattr(model, 'num_timesteps', 0)})
        raise
    finally:
        if callback is not None:
            callback.handle.close()
        if logger is not None:
            logger.close()
        if env is not None:
            env.close()


def run_suite(args, experiment):
    """Fresh process per run releases Genesis GPU memory; stages with resume_previous chain."""
    args.output.mkdir(parents=True, exist_ok=True)
    for seed in experiment['training']['seeds']:
        for method in experiment['training']['methods']:
            previous = None
            for stage in experiment['stages']:
                out = args.output/f"seed{seed}_{method}_{stage['name']}"
                status = out/'status.json'
                if status.is_file() and json.loads(status.read_text()).get('status') == 'completed':
                    previous = out/'final_model.zip'
                    continue
                command = [sys.executable, '-m', __package__+'.v1_train', '--experiment-json',
                    str(args.experiment_json.resolve()), '--output', str(out.resolve()),
                    '--seed', str(seed), '--method', method, '--stage', stage['name']]
                if stage.get('resume_previous'):
                    if previous is None:
                        raise ValueError(f"{stage['name']} resumes a previous stage, but none completed")
                    command += ['--resume', str(previous.resolve())]
                with (args.output/f"seed{seed}_{method}_{stage['name']}.log").open('w') as log:
                    subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
                previous = out/'final_model.zip'


def dry_run(experiment):
    libraries = [load_library(obj['base_library']) for obj in experiment['objects']]
    report = {'action_joints': list(experiment_action_joints(experiment, libraries))}
    for obj, (library, bases) in zip(experiment['objects'], libraries):
        report[obj['object_id']] = dict(object_spec=library['object_spec'], hand_shape=library['hand_shape'],
            labels={label: sum(b.label == label for b in bases) for label in ('S', 'R', 'U')},
            splits={split: sum(b.split == split for b in bases) for split in ('train', 'test')},
            takeover_s=sorted({round(b.takeover_tick/300, 4) for b in bases}),
            episode_s=sorted({round((b.end_tick-b.takeover_tick)/300, 4) for b in bases}))
    lanes = experiment['lanes_per_object']*len(experiment['objects'])
    t = experiment['training']
    report['rollout'] = dict(total_lanes=lanes, samples=lanes*t['n_steps'],
                             divisible_by_batch=lanes*t['n_steps'] % t['batch_size'] == 0)
    print(json.dumps(report, indent=2))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment-json', type=Path, required=True)
    p.add_argument('--output', type=Path)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--method', choices=sorted(METHODS), default='tactile')
    p.add_argument('--stage')
    p.add_argument('--resume', type=Path)
    p.add_argument('--suite', action='store_true')
    p.add_argument('--smoke', action='store_true', help='2 lanes per object, one 32-step rollout and update')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--total-timesteps', type=int)
    args = p.parse_args(argv)
    experiment = load_experiment(args.experiment_json)
    if args.dry_run:
        return dry_run(experiment)
    if args.output is None:
        p.error('--output is required')
    if args.suite:
        if args.smoke or args.resume or args.total_timesteps or args.stage:
            p.error('--suite runs every seed, method and stage as frozen in the experiment')
        return run_suite(args, experiment)
    args.stage = args.stage or experiment['stages'][0]['name']
    run(args, experiment)


if __name__ == '__main__':
    main()
