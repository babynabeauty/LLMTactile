"""Profile-balanced PPO initialized from an existing compatible V1 policy.

Transfer preserves actor, critic and learned exploration weights, but starts a
new Adam optimizer and additional-training counter. Ordinary --resume preserves
the complete state within this new experiment. No teacher is used at inference.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from functools import partial
from pathlib import Path

import torch

from . import v1_train
from . import v1_balanced_train as balanced
from .frozen_scene_assets import frozen_scene_assets
from .normalized_sde_policy import NormalizedSDEMultiInputPolicy


def make_model(env, settings, seed, method, *, resume=None, output, profiles, initialization):
    source_path = Path(initialization['checkpoint'])
    if v1_train.sha256(source_path) != initialization['checkpoint_sha256']:
        raise ValueError('Initialization checkpoint hash changed')
    if initialization['kind'] != 'compatible_v1_weights_fresh_adam':
        raise ValueError('Unknown initialization protocol')
    model = balanced.make_model(env, settings, seed, method, resume=resume,
                                output=output, profiles=profiles)
    if resume:
        if getattr(model, 'initialization_protocol', None) != initialization:
            raise ValueError('Resume changes initialization provenance')
        return model
    v1_train.require_checkpoint_control_compatibility(source_path)
    v1_train.require_checkpoint_conditioning(source_path, env.get_attr('actor_conditioning')[0])
    v1_train.require_checkpoint_tactile_compatibility(source_path)
    with torch.device('cpu'):
        source = v1_train.CPUTrainingPPO.load(source_path, device='cpu')
    if type(source.policy) is not NormalizedSDEMultiInputPolicy:
        raise ValueError('Transfer requires native normalized V1 exploration')
    if source.v1_config != asdict(env.get_attr('config')[0]):
        raise ValueError('Transfer changes reward or task contract')
    if tuple(source.action_joints) != tuple(model.action_joints):
        raise ValueError('Transfer changes action joints')
    if source.policy.features_extractor.use_tactile != model.policy.features_extractor.use_tactile:
        raise ValueError('Transfer changes tactile input')
    if source.num_timesteps != initialization['source_training_timesteps']:
        raise ValueError('Source training counter does not match initialization record')
    model.policy.load_state_dict(source.policy.state_dict(), strict=True)
    assert v1_train.weight_digest(model) == v1_train.weight_digest(source)
    assert model.num_timesteps == 0 and not model.policy.optimizer.state
    model.initialization_protocol = initialization
    model.set_random_seed(seed)
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
    if experiment.get('training_balance') != balanced.PROTOCOL:
        parser.error('Requires profile_equal_loss_v1 protocol')
    initialization = experiment['initialization']
    if args.dry_run:
        v1_train.dry_run(experiment)
        print(v1_train.to_json(dict(initialization=initialization, training_balance=balanced.PROTOCOL), indent=2))
        return
    if args.output is None:
        parser.error('--output is required')
    args.stage = args.stage or experiment['stages'][0]['name']
    original_factory, original_callback = v1_train.make_model, v1_train.TrainingCallback
    try:
        v1_train.make_model = partial(make_model, output=args.output.resolve(),
            profiles=experiment['sampler']['material_profiles'], initialization=initialization)
        v1_train.TrainingCallback = balanced.BalancedTrainingCallback
        with frozen_scene_assets(experiment.get('frozen_scene_assets')):
            v1_train.run(args, experiment)
    finally:
        v1_train.make_model, v1_train.TrainingCallback = original_factory, original_callback


if __name__ == '__main__':
    main()
