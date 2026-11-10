"""V1.5: resume a V1 T1+T2 policy with per-finger correlated exploration.

The original trainer, environment, loss and callback remain unchanged. This
entry point selects its model factory only for this process and restores it on
exit. Old V1 training and evaluation processes continue using their own factory.
"""
from __future__ import annotations

import argparse
import contextlib
from copy import deepcopy
from functools import partial
import io
import json
import math
from numbers import Real
from pathlib import Path

import torch
from stable_baselines3.common.save_util import load_from_zip_file

from . import v1_train
from .finger_sde_policy import FingerSDEMultiInputPolicy
from .normalized_sde_policy import NormalizedSDEMultiInputPolicy


def exploration_protocol(correlation):
    return dict(version='finger_correlated_sde_v1', correlation=float(correlation),
                groups=[[1, 2], [4, 5], [6, 7], [8, 9], [10, 11]],
                independent_joints=[0, 3], preserve_marginal_std=True,
                preserve_action_dimensions=True)


def assert_state_equal(first, second):
    """Check nested policy/Adam state without ignoring tensor values or order."""
    if isinstance(first, torch.Tensor):
        if not isinstance(second, torch.Tensor) or not torch.equal(first.cpu(), second.cpu()):
            raise ValueError('Checkpoint tensor changed during V1.5 migration')
    elif isinstance(first, dict):
        if not isinstance(second, dict) or list(first) != list(second):
            raise ValueError('Checkpoint parameter/state ordering changed')
        for key in first:
            assert_state_equal(first[key], second[key])
    elif isinstance(first, (list, tuple)):
        if type(first) is not type(second) or len(first) != len(second):
            raise ValueError('Checkpoint sequence changed')
        for a, b in zip(first, second):
            assert_state_equal(a, b)
    elif first != second:
        raise ValueError(f'Checkpoint state changed: {first!r} != {second!r}')


def load_finger_model(checkpoint, *, env=None, seed=None, correlation=.8):
    """Keep every weight and Adam state; change only the exploration distribution."""
    v1_train.require_checkpoint_control_compatibility(checkpoint)
    v1_train.require_checkpoint_tactile_compatibility(checkpoint)
    if env is not None:
        v1_train.require_checkpoint_conditioning(checkpoint, env.get_attr('actor_conditioning')[0])
    data, parameters, _ = load_from_zip_file(checkpoint, device='cpu')
    if data['policy_class'] not in (NormalizedSDEMultiInputPolicy, FingerSDEMultiInputPolicy):
        raise ValueError('V1.5 requires a normalized gSDE V1 or V1.5 checkpoint')
    joints = tuple(data['action_joints'])
    if env is not None and tuple(env.get_attr('action_joints')[0]) != joints:
        raise ValueError('V1.5 action joint IDs must match the source checkpoint')
    kwargs = deepcopy(data['policy_kwargs'])
    if data['policy_class'] is FingerSDEMultiInputPolicy:
        if tuple(kwargs['action_joints']) != joints or kwargs['finger_noise_correlation'] != correlation:
            raise ValueError('Cannot silently change a V1.5 checkpoint exploration protocol')
    kwargs.update(action_joints=joints, finger_noise_correlation=correlation)
    with torch.device('cpu'):
        model = v1_train.CPUTrainingPPO.load(
            checkpoint, env=env, device='cpu',
            custom_objects={'policy_class': FingerSDEMultiInputPolicy, 'policy_kwargs': kwargs})
        if seed is not None:
            model.set_random_seed(seed)
    assert_state_equal(parameters['policy'], model.policy.state_dict())
    assert_state_equal(parameters['policy.optimizer'], model.policy.optimizer.state_dict())
    if model.num_timesteps != data['num_timesteps']:
        raise ValueError('V1.5 migration changed the training step counter')
    model.exploration_protocol = exploration_protocol(correlation)
    model.exploration_migration = dict(source_checkpoint=str(Path(checkpoint).resolve()),
                                     source_sha256=v1_train.sha256(checkpoint),
                                     initial_timesteps=model.num_timesteps,
                                     identical_weights=True, identical_optimizer=True)
    return model


def make_model(env, settings, seed, method, *, resume=None, correlation=.8):
    if resume is None:
        raise ValueError('V1.5 requires --resume from the shared V1 checkpoint')
    model = load_finger_model(resume, env=env, seed=seed, correlation=correlation)
    if model.policy.features_extractor.use_tactile != v1_train.METHODS[method]['use_tactile']:
        raise ValueError('V1.5 tactile input must match the source checkpoint')
    for field in ('n_steps', 'batch_size', 'n_epochs', 'gamma', 'gae_lambda', 'sde_sample_freq'):
        if getattr(model, field) != settings[field]:
            raise ValueError(f'V1.5 must preserve source training setting: {field}')
    if model.lr_schedule(1.) != settings['learning_rate']:
        raise ValueError('V1.5 must preserve the source learning rate')
    if model.n_steps * env.num_envs % model.batch_size:
        raise ValueError('Rollout size must divide into complete minibatches')
    return model


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment-json', type=Path, required=True)
    parser.add_argument('--resume', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--stage')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--method', choices=sorted(v1_train.METHODS), default='tactile')
    parser.add_argument('--total-timesteps', type=int)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    args.smoke = False
    experiment = v1_train.load_experiment(args.experiment_json)
    protocol = experiment.get('exploration_protocol', {})
    correlation = protocol.get('correlation')
    if isinstance(correlation, bool) or not isinstance(correlation, Real) or not math.isfinite(correlation) or not 0 <= correlation < 1:
        parser.error('exploration correlation must be in [0, 1)')
    if protocol != exploration_protocol(correlation):
        parser.error('experiment requires an explicit finger_correlated_sde_v1 exploration_protocol')
    if args.dry_run:
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            v1_train.dry_run(experiment)
        report = json.loads(buffer.getvalue())
        report['exploration_protocol'] = protocol
        report['resume_checkpoint'] = str(args.resume.resolve())
        print(json.dumps(report, indent=2))
        return
    if args.output is None:
        parser.error('--output is required')
    args.stage = args.stage or experiment['stages'][0]['name']
    original_factory = v1_train.make_model
    try:
        v1_train.make_model = partial(make_model, correlation=correlation)
        v1_train.run(args, experiment)
    finally:
        v1_train.make_model = original_factory


if __name__ == '__main__':
    main()
