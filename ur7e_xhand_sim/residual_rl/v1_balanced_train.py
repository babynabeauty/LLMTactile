"""Opt-in profile-balanced V1 trainer; native V1 processes remain unchanged."""
from __future__ import annotations

import argparse
from functools import partial
import json
from pathlib import Path

import numpy as np
import torch

from . import v1_train
from .profile_balanced_ppo import ProfileBalancedDictRolloutBuffer, ProfileBalancedPPO
from .frozen_scene_assets import frozen_scene_assets

PROTOCOL = dict(version='profile_equal_loss_v1', profile_fields=['mass_kg', 'friction', 'f_break_N'],
                normalization='current_rollout_inverse_profile_counts',
                terms=['policy', 'value', 'entropy', 'kl'], advantage_normalization='native_minibatch',
                missing_profiles='equal_present_profiles_and_log', actor_profile_id=False)
_NATIVE_MAKE_MODEL = v1_train.make_model


def profile_key(spec):
    force = spec['f_break_N']
    return (float(spec['mass_kg']), float(spec['friction']),
            None if force is None or np.isinf(force) else float(force))


def current_profile_ids(scenes, mapping):
    return np.asarray([mapping[profile_key(spec)] for scene in scenes for spec in scene.episode], dtype=np.int64)


class BalancedTrainingCallback(v1_train.TrainingCallback):
    """Remember profile before stepping: done lanes have already reset at callback time."""
    def _on_training_start(self):
        super()._on_training_start()
        self.profile_map = {profile_key(p): i for i, p in enumerate(self.scenes[0].sampler.material_profiles)}

    def _on_rollout_start(self):
        super()._on_rollout_start()
        self.transition_profiles = current_profile_ids(self.scenes, self.profile_map)

    def _on_step(self):
        buffer = self.model.rollout_buffer
        buffer.profile_ids[buffer.pos] = self.transition_profiles
        self.transition_profiles = current_profile_ids(self.scenes, self.profile_map)
        return super()._on_step()


def make_model(env, settings, seed, method, *, resume=None, output, profiles):
    if resume:
        v1_train.require_checkpoint_control_compatibility(resume)
        v1_train.require_checkpoint_conditioning(resume, env.get_attr('actor_conditioning')[0])
        with torch.device('cpu'):
            model = ProfileBalancedPPO.load(resume, env=env, device='cpu')
        if getattr(model, 'training_balance', None) != PROTOCOL or model.balance_profiles != profiles:
            raise ValueError('Resume must preserve the exact balanced protocol and profiles')
        for field in ('n_steps', 'batch_size', 'n_epochs', 'gamma', 'gae_lambda', 'sde_sample_freq'):
            if getattr(model, field) != settings[field]:
                raise ValueError(f'Resume changes training setting: {field}')
        if model.lr_schedule(1.) != settings['learning_rate']:
            raise ValueError('Resume changes learning rate')
        if model.policy.features_extractor.use_tactile != v1_train.METHODS[method]['use_tactile']:
            raise ValueError('Resume changes tactile input')
        if tuple(model.action_joints) != tuple(env.get_attr('action_joints')[0]):
            raise ValueError('Resume changes action joints')
        model.set_random_seed(seed)
    else:
        original_class = v1_train.CPUTrainingPPO
        try:
            v1_train.CPUTrainingPPO = partial(ProfileBalancedPPO,
                rollout_buffer_class=ProfileBalancedDictRolloutBuffer,
                rollout_buffer_kwargs=dict(n_profiles=len(profiles)))
            model = _NATIVE_MAKE_MODEL(env, settings, seed, method)
        finally:
            v1_train.CPUTrainingPPO = original_class
        model.training_balance, model.balance_profiles = PROTOCOL, profiles
    model.balance_log_path = str(Path(output)/'balanced_updates.jsonl')
    return model


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment-json', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--stage')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--method', choices=sorted(v1_train.METHODS), default='tactile')
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--total-timesteps', type=int)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args(argv)
    experiment = v1_train.load_experiment(args.experiment_json)
    if experiment.get('training_balance') != PROTOCOL:
        parser.error('Experiment must declare the exact profile_equal_loss_v1 protocol')
    profiles = experiment['sampler']['material_profiles']
    if not profiles or len({profile_key(p) for p in profiles}) != len(profiles):
        parser.error('Balanced training requires distinct explicit profiles')
    if args.dry_run:
        v1_train.dry_run(experiment)
        print(json.dumps(dict(training_balance=PROTOCOL, profiles=profiles), indent=2))
        return
    if args.output is None:
        parser.error('--output is required')
    args.stage = args.stage or experiment['stages'][0]['name']
    original_factory, original_callback = v1_train.make_model, v1_train.TrainingCallback
    try:
        v1_train.make_model = partial(make_model, output=args.output.resolve(), profiles=profiles)
        v1_train.TrainingCallback = BalancedTrainingCallback
        with frozen_scene_assets(experiment.get('frozen_scene_assets')):
            v1_train.run(args, experiment)
    finally:
        v1_train.make_model, v1_train.TrainingCallback = original_factory, original_callback


if __name__ == '__main__':
    main()
