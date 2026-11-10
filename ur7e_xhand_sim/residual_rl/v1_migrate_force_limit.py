"""Explicit Stage1 checkpoint migration to a numeric force-limit observation.

At the reference finite limit and unlimited, retain the old level-3 / level-4
actor mean, value, and gSDE distribution. No policy/action head is reinitialized.
Other numeric limits interpolate through the new scalar; this is not a claim
of pre-existing generalization. Source checkpoints are never overwritten.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from .v1_conditioning import (LEVEL, FORCE_LIMIT, FORCE_SCALE_N, CONDITION_START,
                              require_checkpoint_conditioning)
from .control_protocol import require_checkpoint_control_compatibility
from .tactile_provenance import require_checkpoint_tactile_compatibility


def input_column_mapping(old_dim, finite_limit_N, finite_level=3):
    if not np.isfinite(finite_limit_N) or finite_limit_N <= 0 or finite_level not in (1, 2, 3):
        raise ValueError('Need a positive finite reference limit and a finite legacy level 1..3')
    start = CONDITION_START
    if old_dim < start+4:
        raise ValueError('Unexpected legacy state dimension')
    row = list(range(start))+[start+finite_level-1, start+3]+list(range(start+4, old_dim))
    columns = row+[i+old_dim for i in row]+[2*old_dim]
    scale = np.ones(len(columns), np.float32)
    scale[start] = scale[len(row)+start] = FORCE_SCALE_N/finite_limit_N
    return columns, scale


def migrate_policy(policy, finite_limit_N, finite_level=3):
    """Transfer both input encoders and Adam moments in the new coordinates."""
    old_dim = policy.observation_space['state'].shape[-1]
    columns, scale = input_column_mapping(old_dim, finite_limit_N, finite_level)
    params = policy._get_constructor_parameters()
    spaces = dict(policy.observation_space.spaces)
    spaces['state'] = gym.spaces.Box(-np.inf, np.inf, (12, old_dim-2), dtype=np.float32)
    params['observation_space'] = gym.spaces.Dict(spaces)
    with torch.device('cpu'):
        target = type(policy)(**params)
        scale = torch.as_tensor(scale)
        values = copy.deepcopy(policy.state_dict())
        for name in values:
            if name.endswith('state_encoder.0.weight'):
                values[name] = values[name][:, columns]*scale
        target.load_state_dict(values, strict=True)
        optimizer = copy.deepcopy(policy.optimizer.state_dict())
        # SB3 has one parameter group, in policy.parameters() order.
        if len(optimizer['param_groups']) != 1:
            raise ValueError('Only the SB3 single optimizer group is supported')
        names = [name for name, _ in policy.named_parameters()]
        if names != [name for name, _ in target.named_parameters()]:
            raise ValueError('Migration changed parameter ordering')
        for name, identifier in zip(names, optimizer['param_groups'][0]['params']):
            if not name.endswith('state_encoder.0.weight'):
                continue
            state = optimizer['state'].get(identifier, {})
            # W_new=W_old*scale, so g_new=g_old/scale. Preserve Adam step
            # counters and transform first/second moments with those units.
            for key, power in [('exp_avg', 1), ('exp_avg_sq', 2), ('max_exp_avg_sq', 2)]:
                if key in state:
                    state[key] = state[key][:, columns]/scale**power
        target.optimizer.load_state_dict(optimizer)
        target.set_training_mode(False)
    return target


def verify_reference_equivalence(old, new, finite_limit_N, finite_level=3):
    """CPU checks on mixed 12-frame histories, including changes of condition."""
    rng = np.random.default_rng(20261108)
    size = old.observation_space['state'].shape[-1]
    state = rng.normal(0, .1, (8, 12, size)).astype(np.float32)
    finite = rng.integers(0, 2, (8, 12)).astype(bool)
    levels = np.where(finite, finite_level, 4)
    state[..., CONDITION_START:CONDITION_START+4] = np.eye(4)[levels-1]
    mapped = np.concatenate([state[..., :CONDITION_START],
        np.stack([finite*finite_limit_N/FORCE_SCALE_N, ~finite], axis=-1),
        state[..., CONDITION_START+4:]], axis=-1).astype(np.float32)
    tactile = rng.normal(0, .001, (8, 12, 5, 120, 3)).astype(np.float32)
    output = []
    for policy, s in [(old, state), (new, mapped)]:
        policy.set_training_mode(False)
        with torch.no_grad():
            obs, _ = policy.obs_to_tensor(dict(state=s, tactile=tactile))
            distribution = policy.get_distribution(obs).distribution
            output.append([distribution.mean.clone(), distribution.stddev.clone(), policy.predict_values(obs)])
    errors = {}
    for name, a, b in zip(['action_mean', 'action_std', 'value'], *output):
        torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5)
        errors[name] = float((a-b).abs().max())
    return errors


def main(argv=None):
    from .v1_train import CPUTrainingPPO, configure_seed
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--finite-limit-N', type=float, required=True)
    parser.add_argument('--finite-level', type=int, default=3)
    args = parser.parse_args(argv)
    source = args.checkpoint.resolve()
    require_checkpoint_control_compatibility(source)
    require_checkpoint_tactile_compatibility(source)
    require_checkpoint_conditioning(source, LEVEL)
    configure_seed(1)
    model = CPUTrainingPPO.load(source, device='cpu')
    if not model.use_level:
        raise ValueError('Source checkpoint does not use the task condition')
    policy = migrate_policy(model.policy, args.finite_limit_N, args.finite_level)
    errors = verify_reference_equivalence(model.policy, policy, args.finite_limit_N, args.finite_level)
    model.policy = policy
    model.observation_space = policy.observation_space
    model.actor_conditioning = FORCE_LIMIT
    model._last_obs = None
    args.output.mkdir(parents=True, exist_ok=False)
    model.save(args.output/'checkpoint')
    source_dir = source.parent.parent if source.parent.name == 'checkpoints' else source.parent
    manifest = json.loads((source_dir/'manifest.json').read_text())
    report = dict(source_checkpoint=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        actor_conditioning=FORCE_LIMIT, finite_reference_N=args.finite_limit_N,
        finite_reference_level=args.finite_level, reference_equivalence_max_abs_errors=errors,
        new_state_dimension=policy.observation_space['state'].shape[-1],
        transfer='Remap both actor/critic state input columns and corresponding Adam moments; '
                 'all other weights, action mean head, log_std, optimizer states and counters retained.')
    manifest['checkpoint_derivation'] = report
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2))
    (args.output/'migration.json').write_text(json.dumps(report, indent=2))
    require_checkpoint_conditioning(args.output/'checkpoint.zip', FORCE_LIMIT)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
