"""Read-only v3 reporting: material reachability, exploratory training and evaluation.

No physical environment or policy module is imported. Every discovered run is
reported, including resumes and failed/incomplete runs; no best checkpoint is
selected. Held-out outcomes require an explicit --include-heldout opt-in.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import zipfile

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'adaptive_rl_matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


class Reader:
    def __init__(self, root):
        self.root = root
        self.sources, self.warnings, self.pending = {}, [], []

    def name(self, path):
        return str(path.relative_to(self.root))

    def bytes(self, path):
        data = path.read_bytes()
        self.sources[self.name(path)] = {'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}
        return data

    def json(self, path, required=False):
        if not path.is_file():
            if required:
                self.pending.append(self.name(path))
            return None
        try:
            return json.loads(self.bytes(path))
        except (ValueError, UnicodeDecodeError) as exc:
            self.warnings.append(f'{self.name(path)}: invalid/incomplete JSON: {exc}')
            return None

    def jsonl(self, path):
        if not path.is_file():
            self.pending.append(self.name(path))
            return []
        rows = []
        for i, line in enumerate(self.bytes(path).splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError('Expected an object')
                rows.append(row)
            except (ValueError, UnicodeDecodeError) as exc:
                self.warnings.append(f'{self.name(path)}:{i}: incomplete/invalid row skipped: {exc}')
        return rows


def mean(values):
    values = np.asarray([v for v in values if v is not None], float)
    values = values[np.isfinite(values)]
    return float(values.mean()) if values.size else None


def extreme(values, op):
    values = [v for v in values if v is not None and np.isfinite(v)]
    return float(op(values)) if values else None


def fmt(value, places=3):
    return '待产出' if value is None else f'{value:.{places}f}'


def same_material_conditions(first, second):
    """Compare material multisets allowing recorded float32 conversion error."""
    if not first or not second or len(first) != len(second):
        return False
    left, right = np.asarray(sorted(first), float), np.asarray(sorted(second), float)
    return bool(left.shape == right.shape and np.allclose(left, right, rtol=0, atol=1e-7))


def episode_summary(episodes):
    task = [e for e in episodes if e.get('task_success') is not None]
    strict = [e for e in episodes if e.get('strict_success') is not None]
    return {'episodes': len(episodes), 'task_label_count': len(task), 'strict_label_count': len(strict),
            'task_successes': sum(bool(e['task_success']) for e in task),
            'strict_successes': sum(bool(e['strict_success']) for e in strict),
            'mean_return': mean([e.get('return') for e in episodes]),
            'mean_load_N': mean([e.get('mean_contact_force_N') for e in episodes]),
            'max_episode_mean_load_N': extreme([e.get('mean_contact_force_N') for e in episodes], max),
            'peak_label_count': sum(e.get('peak_contact_force_N') is not None for e in episodes),
            'mean_per_lane_peak_load_N': mean([e.get('peak_contact_force_N') for e in episodes]),
            'max_peak_all_lanes_N': extreme([e.get('peak_contact_force_N') for e in episodes], max),
            'mean_gain': mean([mean(e.get('mean_gains', [])) for e in episodes]),
            'mean_contact_fraction': mean([e.get('hold_contact_fraction') for e in episodes]),
            'min_contact_fraction': extreme([e.get('hold_contact_fraction') for e in episodes], min),
            'max_unsupported_hold_s': extreme([e.get('hold_max_unsupported_duration_s') for e in episodes], max),
            'min_hold_clearance_m': extreme([e.get('hold_min_clearance_m') for e in episodes], min)}


def selected_source_hashes(manifest):
    hashes = manifest.get('source_sha256', manifest.get('source_hashes', {}))
    names = {'adaptive_env.py', 'env.py', 'core.py', 'adaptive_policy.py',
             'adaptive_train.py', 'train.py', 'normalized_sde_policy.py', 'matched_initialization.py',
             'reward_scale.py'}
    return {Path(key).name: value for key, value in hashes.items()
            if Path(key).name in names and ('/' not in key or '/residual_rl/' in key)}


def read_feature_sharing(reader, directory, model, requested_separate=None):
    """Check saved tensor aliases without constructing a policy or environment."""
    explicit = model.get('share_features_extractor')
    expected = (not requested_separate) if explicit is None and requested_separate is not None else (
        True if explicit is None else bool(explicit))
    checks = []
    for name in ('initial_model.zip', 'final_model.zip'):
        path = directory / name
        if not path.is_file():
            continue
        try:
            import torch
            with zipfile.ZipFile(io.BytesIO(reader.bytes(path))) as archive:
                metadata = json.loads(archive.read('data'))
                saved_kwargs = metadata.get('policy_kwargs', {})
                declared = saved_kwargs.get('share_features_extractor', True)
                # weights_only prevents loading arbitrary policy/optimizer objects.
                state = torch.load(io.BytesIO(archive.read('policy.pth')),
                                   map_location='cpu', weights_only=True)
            prefix = 'pi_features_extractor.'
            keys = [key[len(prefix):] for key in state if key.startswith(prefix)]
            aliases = []
            for key in keys:
                actor, critic = state[prefix + key], state['vf_features_extractor.' + key]
                aliases.append(actor.untyped_storage().data_ptr() == critic.untyped_storage().data_ptr()
                    and actor.storage_offset() == critic.storage_offset() and actor.shape == critic.shape)
            inferred = True if aliases and all(aliases) else False if aliases and not any(aliases) else None
            passed = inferred is not None and inferred == expected and declared == expected
            checks.append({'checkpoint': name, 'declared_or_default_shared': declared,
                'matched_tensor_pairs': len(keys), 'aliased_tensor_pairs': sum(aliases),
                'inferred_shared_from_storage': inferred, 'matches_resolved_topology': passed})
            if not passed:
                reader.warnings.append(f'{reader.name(path)}: feature-sharing metadata/tensor aliases disagree')
        except (OSError, ValueError, KeyError, RuntimeError, zipfile.BadZipFile) as exc:
            reader.warnings.append(f'{reader.name(path)}: cannot verify feature sharing: {exc}')
    return {'share_features_extractor': expected,
        'metadata_basis': ('explicit_model_config' if explicit is not None else
            'requested_CLI_topology_not_yet_verified' if requested_separate is not None else 'legacy_SB3_default_true'),
        'saved_checkpoints': checks,
        'saved_topology_verified': bool(checks) and all(row['matches_resolved_topology'] for row in checks),
        'method': 'CPU weights_only policy state-dict alias audit; no model construction or prediction'}


def read_optimizer_epsilon(reader, directory, model, requested_eps=None):
    """Resolve actual epsilon from saved optimizer groups, never CLI defaults."""
    declared = model.get('optimizer_param_group_eps')
    requested = model.get('adam_eps_requested', requested_eps)
    checks = []
    for name in ('initial_model.zip', 'final_model.zip'):
        path = directory / name
        if not path.is_file():
            continue
        try:
            import torch
            with zipfile.ZipFile(io.BytesIO(reader.bytes(path))) as archive:
                member = ('policy.optimizer.pth' if 'policy.optimizer.pth' in archive.namelist()
                          else 'optimizer.pth')
                state = torch.load(io.BytesIO(archive.read(member)), map_location='cpu', weights_only=True)
            groups = state.get('param_groups', [])
            values = [float(group['eps']) if group.get('eps') is not None else None for group in groups]
            finite_positive = bool(values) and all(value is not None and np.isfinite(value) and value > 0
                                                  for value in values)
            metadata_match = declared is None or values == declared
            checks.append({'checkpoint': name, 'optimizer_archive_member': member,
                'param_group_eps': values, 'finite_positive_groups': finite_positive,
                'matches_explicit_metadata': metadata_match})
            if not finite_positive or not metadata_match:
                reader.warnings.append(f'{reader.name(path)}: optimizer eps invalid or differs from metadata')
        except (OSError, ValueError, KeyError, RuntimeError, zipfile.BadZipFile) as exc:
            reader.warnings.append(f'{reader.name(path)}: cannot verify optimizer epsilon: {exc}')
            checks.append({'checkpoint': name, 'param_group_eps': None,
                'finite_positive_groups': False, 'matches_explicit_metadata': False,
                'read_error': str(exc)})
    saved = [row['param_group_eps'] for row in checks if row['finite_positive_groups']]
    consistent = bool(saved) and all(values == saved[0] for values in saved)
    if saved and not consistent:
        reader.warnings.append(f'{reader.name(directory)}: initial/final optimizer eps differ')
    actual = saved[0] if consistent else declared if not checks else None
    verified = bool(checks) and consistent and all(row['finite_positive_groups']
                                                  and row['matches_explicit_metadata'] for row in checks)
    return {'optimizer_class': model.get('optimizer_class'),
        'class_metadata_note': 'Historical checkpoint optimizer state does not identify the exact optimizer class; no class is guessed from eps.',
        'adam_eps_requested': requested, 'optimizer_param_group_eps': actual,
        'metadata_param_group_eps': declared,
        'basis': ('saved_optimizer_param_groups' if verified else 'explicit_model_config_unverified'
                  if declared is not None else 'requested_or_unspecified_unverified'),
        'saved_optimizer_eps_verified': verified, 'saved_checkpoints': checks,
        'method': 'CPU weights_only optimizer state-dict param_groups; no optimizer construction or updates'}


def read_training_reward_scale(reader, directory, model, requested_scale=None, manifest_model=None,
                               requested_explicit=None):
    """Audit the saved scalar without constructing a policy or unpickling data.

    Missing checkpoint attributes mean legacy, unscaled training (1.0). That
    fallback applies only after reading an actual archive, never to a pending run.
    """
    manifest_model = manifest_model or {}
    declarations = {name: metadata['training_reward_scale'] for name, metadata in
        (('model_config', model), ('manifest_actual_model_config', manifest_model))
        if 'training_reward_scale' in metadata}
    declared = next(iter(declarations.values()), None)
    requested = model.get('reward_scale_requested', requested_scale)

    def valid(value):
        return (isinstance(value, (int, float)) and not isinstance(value, bool)
                and np.isfinite(value) and value > 0)

    declared_valid = all(valid(value) for value in declarations.values())
    declarations_agree = declared_valid and all(value == declared for value in declarations.values())
    if not declared_valid:
        reader.warnings.append(f'{reader.name(directory)}: invalid training reward scale metadata')
    elif not declarations_agree:
        reader.warnings.append(f'{reader.name(directory)}: model/manifest training reward scales differ')
    checks = []
    for name in ('initial_model.zip', 'final_model.zip'):
        path = directory / name
        if not path.is_file():
            continue
        try:
            with zipfile.ZipFile(io.BytesIO(reader.bytes(path))) as archive:
                data = json.loads(archive.read('data'))
            if not isinstance(data, dict):
                raise ValueError('Checkpoint data must be a JSON object')
            legacy = 'training_reward_scale' not in data
            value = data.get('training_reward_scale', 1.)
            finite_positive = valid(value)
            metadata_match = declarations_agree and (not declarations or value == declared)
            checks.append({'checkpoint': name, 'training_reward_scale': value,
                'legacy_missing_attribute_uses_one': legacy,
                'finite_positive': bool(finite_positive),
                'matches_explicit_metadata': bool(metadata_match)})
            if not finite_positive or not metadata_match:
                reader.warnings.append(f'{reader.name(path)}: training reward scale invalid or differs from metadata')
        except (OSError, ValueError, KeyError, RuntimeError, zipfile.BadZipFile) as exc:
            reader.warnings.append(f'{reader.name(path)}: cannot verify training reward scale: {exc}')
            checks.append({'checkpoint': name, 'training_reward_scale': None,
                'finite_positive': False, 'matches_explicit_metadata': False,
                'read_error': str(exc)})
    saved = [row['training_reward_scale'] for row in checks if row['finite_positive']]
    consistent = bool(saved) and all(value == saved[0] for value in saved)
    if saved and not consistent:
        reader.warnings.append(f'{reader.name(directory)}: initial/final training reward scales differ')
    actual = float(saved[0]) if consistent else declared if not checks and declarations_agree else None
    verified = bool(checks) and consistent and all(row['finite_positive']
        and row['matches_explicit_metadata'] for row in checks)
    both = {row['checkpoint'] for row in checks} == {'initial_model.zip', 'final_model.zip'}
    return {'training_reward_scale': actual, 'reward_scale_requested': requested,
        'reward_scale_request_explicit': requested_explicit,
        'metadata_training_reward_scale': declared,
        'metadata_scale_declarations': declarations,
        'basis': ('saved_checkpoint_JSON_with_legacy_default_one' if verified else
                  'explicit_model_config_unverified' if declared is not None else
                  'requested_or_unspecified_unverified'),
        'saved_reward_scale_verified': verified,
        'initial_final_reward_scale_verified': bool(verified and both),
        'saved_checkpoints': checks,
        'method': 'Read checkpoint ZIP data JSON only; no model construction, tensor load or unpickling',
        'logging_semantics': {
            'episode_returns': 'episodes.jsonl return/reward_components and rollout/ep_rew_mean remain raw, unscaled environment metrics',
            'optimization': 'PPO rollout rewards, returns and value_loss use training_reward_scale; smaller value_loss across scales is not policy improvement',
            'evaluation': 'Frozen evaluation uses the unchanged raw reward and physical success/load/contact metrics'}}


def read_learning_rate(reader, directory, model, manifest_model=None, requested=None):
    """Verify constant learning rates from checkpoint JSON and optimizer groups."""
    declarations = {name: metadata['learning_rate'] for name, metadata in
        (('model_config', model), ('manifest_actual_model_config', manifest_model or {}))
        if 'learning_rate' in metadata}
    valid = lambda value: isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(value) and value > 0
    checks = []
    for name in ('initial_model.zip', 'final_model.zip'):
        path = directory / name
        if not path.is_file():
            continue
        try:
            import torch
            with zipfile.ZipFile(io.BytesIO(reader.bytes(path))) as archive:
                saved = json.loads(archive.read('data')).get('learning_rate')
                member = 'policy.optimizer.pth' if 'policy.optimizer.pth' in archive.namelist() else 'optimizer.pth'
                state = torch.load(io.BytesIO(archive.read(member)), map_location='cpu', weights_only=True)
            groups = [group.get('lr') for group in state.get('param_groups', [])]
            passed = bool(valid(saved) and groups and all(valid(value) and value == saved for value in groups)
                and all(valid(value) and value == saved for value in declarations.values()))
            checks.append({'checkpoint': name, 'model_learning_rate': saved,
                'optimizer_group_learning_rates': groups, 'matches_metadata_and_groups': passed})
            if not passed:
                reader.warnings.append(f'{reader.name(path)}: constant learning rate missing/invalid or metadata/groups disagree')
        except (OSError, ValueError, KeyError, RuntimeError, AttributeError, zipfile.BadZipFile) as exc:
            checks.append({'checkpoint': name, 'model_learning_rate': None,
                'matches_metadata_and_groups': False, 'read_error': str(exc)})
            reader.warnings.append(f'{reader.name(path)}: cannot verify learning rate: {exc}')
    values = [check['model_learning_rate'] for check in checks if valid(check['model_learning_rate'])]
    consistent = bool(values) and all(value == values[0] for value in values)
    if values and not consistent:
        reader.warnings.append(f'{reader.name(directory)}: initial/final learning rates differ')
    metadata_values = list(declarations.values())
    metadata_agree = bool(metadata_values) and all(valid(value) and value == metadata_values[0] for value in metadata_values)
    actual = values[0] if consistent else metadata_values[0] if not checks and metadata_agree else None
    verified = (len(checks) == 2 and consistent
        and all(check['matches_metadata_and_groups'] for check in checks))
    return {'actual_learning_rate': actual, 'requested_learning_rate': requested,
        'metadata_declarations': declarations, 'initial_final_learning_rate_verified': bool(verified),
        'basis': 'checkpoint_JSON_and_optimizer_groups' if verified else 'metadata_or_partial_checkpoint_unverified',
        'saved_checkpoints': checks,
        'scope': 'Constant LR only; no guessed defaults or deserialization of callable schedules. Initial/final group rates must match model JSON and actual-model metadata.'}


def training_reward_config_audit(resolved, environment, requested, required_fields):
    """Environment reward coefficients are metadata, not PPO ZIP attributes."""
    available = {name: value for name, value in
        (('config.resolved.json', resolved), ('environment_protocol.config', environment)) if isinstance(value, dict) and value}
    values = list(available.values())
    complete = bool(required_fields) and bool(values) and all(set(value) == set(required_fields) for value in values)
    agree = bool(values) and all(value == values[0] for value in values)
    actual = values[0] if agree else None
    request_matches = isinstance(requested, dict) and actual is not None and all(actual.get(key) == value for key, value in requested.items())
    return {'actual_reward_config': actual,
        'failure_penalty': actual.get('failure_penalty') if actual else None,
        'metadata_sources': available, 'requested_config': requested,
        'complete_resolved_and_environment_config_verified': bool(len(values) == 2 and complete and agree and request_matches),
        'basis': list(available),
        'scope': 'Full resolved runtime config checked against environment protocol and requested fields; failure_penalty is not stored in the PPO checkpoint and is not inferred from a directory name or current defaults.'}


def training_objective_match_checks(touch, blind):
    a, b = touch['learning_rate_audit'], blind['learning_rate_audit']
    x, y = touch['training_reward_config'], blind['training_reward_config']
    return {'same_actual_learning_rate': a['actual_learning_rate'] is not None and a['actual_learning_rate'] == b['actual_learning_rate'],
        'initial_final_learning_rate_verified': a['initial_final_learning_rate_verified'] and b['initial_final_learning_rate_verified'],
        'same_complete_runtime_reward_config': x['actual_reward_config'] is not None and x['actual_reward_config'] == y['actual_reward_config'],
        'runtime_reward_config_metadata_verified': x['complete_resolved_and_environment_config_verified'] and y['complete_resolved_and_environment_config_verified']}


def read_training(reader, directory, manifest, frozen_config, noise_calibration=None, noise_audit=None):
    status = reader.json(directory / 'status.json') or {}
    resolved_config = reader.json(directory / 'config.resolved.json', True)
    config = resolved_config or manifest.get('config', {})
    model = reader.json(directory / 'model_config.json', True) or manifest.get('actual_model_config', {})
    args = manifest.get('arguments', {})
    resumed = model.get('resumed', bool(args.get('resume')))
    sharing = read_feature_sharing(reader, directory, model, args.get('separate_feature_extractors'))
    optimizer = read_optimizer_epsilon(reader, directory, model, args.get('adam_eps'))
    reward_scaling = read_training_reward_scale(reader, directory, model, args.get('reward_scale'),
        manifest.get('actual_model_config'), args.get('reward_scale_explicit'))
    protocol = reader.json(directory / 'environment/environment_protocol.json', True) or {}
    learning_rate = read_learning_rate(reader, directory, model, manifest.get('actual_model_config'), args.get('learning_rate'))
    reward_config = training_reward_config_audit(resolved_config, protocol.get('config'),
        manifest.get('requested_config', manifest.get('config')), frozen_config.keys())
    logs = reader.jsonl(directory / 'training_metrics.jsonl')
    records = reader.jsonl(directory / 'episodes.jsonl')
    episodes = [r.get('episode_metrics', r) for r in records]
    grouped = defaultdict(list)
    for record in records:
        if record.get('training_timesteps') is not None:
            grouped[int(record['training_timesteps'])].append(record.get('episode_metrics', record))
    curve = [dict(timesteps=step, **episode_summary(values)) for step, values in sorted(grouped.items())]
    use_sde = bool(model.get('use_sde', args.get('use_sde', False)))
    distribution = model.get('policy_distribution', {})
    normalized_sde = bool(model.get('normalize_sde_latent', distribution.get('normalize_sde_latent', False)))
    construction = {key: model.get(key) for key in ('policy_construction_device_context',
        'torch_default_dtype_at_policy_construction', 'policy_parameter_devices_at_run_start')}
    gamma = model.get('gamma', args.get('gamma'))
    initial_gain = model.get('initial_gain_requested', args.get('initial_gain'))
    requested_std = model.get('action_std_init_requested', args.get('action_std_init'))
    uses_tactile = model.get('uses_tactile', manifest.get('policy', {}).get('uses_tactile'))
    calibrated_sigma = None
    if use_sde and not normalized_sde and requested_std is not None:
        for item in (noise_calibration or {}).get('fresh_seed0_sde_initialization', []):
            if item.get('seed') == args.get('seed') and item.get('use_tactile') == uses_tactile:
                for target in item.get('targets', []):
                    if np.isclose(target.get('required_weight_std', np.nan), requested_std, rtol=1e-6, atol=1e-9):
                        calibrated_sigma = target.get('verified_action_sigma_rms')
    actual_initial_prefix_sigma = None
    for item in (noise_audit or {}).get('models', []):
        if item.get('checkpoint') == 'initial' and item.get('sha256') == model.get('initial_checkpoint_sha256'):
            actual_initial_prefix_sigma = item.get('prefix_statistics', {}).get('sigma')
    initial_unit_latent_sigma = None
    parameter_std = np.asarray(distribution.get('parameter_std', []), float)
    if normalized_sde and parameter_std.size and np.isfinite(parameter_std).all() and np.all(parameter_std == parameter_std.flat[0]):
        variance_floor = distribution.get('epsilon')
        if variance_floor is not None and variance_floor >= 0:
            initial_unit_latent_sigma = float(np.sqrt(parameter_std.flat[0] ** 2 + variance_floor))
    by_material = defaultdict(list)
    for e in episodes:
        if e.get('mass_kg') is not None and e.get('friction') is not None:
            by_material[(round(e['mass_kg'], 7), round(e['friction'], 7))].append(e)
    observation = manifest.get('observation_space', {}).get('spaces', {})
    state_shape = observation.get('state', {}).get('shape')
    return dict(name=reader.name(directory), evidence='exploratory_training_not_validation_or_test',
        execution_status=status.get('status', 'pending'), status_record=status,
        seed=args.get('seed'), uses_tactile=uses_tactile,
        variant=('normalized-gSDE' if normalized_sde else 'native-gSDE' if use_sde else 'iid')
            + (' / checkpoint-restore' if resumed else ' / default-init' if initial_gain is None else f' / init-gain={initial_gain:g}')
            + f" / gamma={gamma} / lambda={model.get('gae_lambda', 'unrecorded')} / rollout={model.get('n_steps', args.get('n_steps'))}"
            + (' / shared-encoder' if sharing['share_features_extractor'] else ' / separate-encoders')
            + f" / Adam-eps={optimizer['optimizer_param_group_eps'] if optimizer['optimizer_param_group_eps'] is not None else 'unverified'}"
            + f" / reward-scale={reward_scaling['training_reward_scale'] if reward_scaling['training_reward_scale'] is not None else 'unverified'}"
            + f" / P={reward_config['failure_penalty']} / LR={learning_rate['actual_learning_rate']}"
            + (' / CPU-init' if construction['policy_construction_device_context'] == 'cpu' else ' / init-device-unrecorded'),
        initialization={'initial_gain_requested': initial_gain,
            'initial_action_weight_zeroed': model.get('initial_action_weight_zeroed', False),
            'construction': construction,
            'method': model.get('initialization', {}).get('method',
                'initialization_pending' if not model else 'checkpoint_restore'
                if model.get('resumed', bool(args.get('resume'))) else 'native_shared_policy_construction')},
        initialization_audit=model.get('initialization', {}), feature_sharing=sharing,
        optimizer=optimizer, reward_scaling=reward_scaling,
        learning_rate_audit=learning_rate, training_reward_config=reward_config,
        parameter_count=model.get('parameter_count'),
        exploration={'use_sde': use_sde,
            'normalize_sde_latent': normalized_sde,
            'sde_normalization_eps': distribution.get('sde_normalization_eps') if normalized_sde else None,
            'sde_sample_freq': model.get('sde_sample_freq', args.get('sde_sample_freq')) if use_sde else None,
            'requested_std': requested_std, 'std_semantics': 'latent_noise_weight_std' if use_sde else 'action_std',
            'fresh_prefix_action_sigma_rms': calibrated_sigma,
            'actual_initial_prefix_sigma': actual_initial_prefix_sigma,
            'initial_uniform_weight_unit_latent_sigma': initial_unit_latent_sigma,
            'prefix_calibration_scope': 'Fresh CPU reconstruction is not the actual checkpoint: legacy Genesis default-device initialization differs. Actual checkpoint audit takes precedence. Normalized equal-weight formula assumes nonzero unit-length latent; learned weights may still vary with direction.'},
        resumed=resumed, input_checkpoint=args.get('resume'),
        input_checkpoint_sha256=manifest.get('input_checkpoint_sha256'),
        timesteps_at_run_start=model.get('timesteps_at_run_start', 0),
        timesteps=status.get('timesteps'), requested_additional_timesteps=args.get('total_timesteps'),
        actual_additional_timesteps=status.get('actual_additional_timesteps'),
        rollout_schedule={'n_steps': model.get('n_steps', args.get('n_steps')),
            'n_envs': manifest.get('n_envs'), 'batch_size': model.get('batch_size', args.get('batch_size')),
            'n_epochs': model.get('n_epochs', args.get('n_epochs')), 'gae_lambda': model.get('gae_lambda')},
        final_checkpoint_sha256=status.get('final_checkpoint_sha256'),
        initial_checkpoint_sha256=model.get('initial_checkpoint_sha256'),
        config=config, config_matches_frozen=config == frozen_config,
        conditions_kg_mu=manifest.get('conditions_kg_mu', []),
        state_shape=state_shape, state_history=protocol.get('state_history'),
        state_history_60hz_12frames=state_shape == [12, 79] and protocol.get('tactile_hz') == 60,
        actor_state_fields=protocol.get('actor_state_fields'), rates_hz=manifest.get('rates_hz'),
        source_hashes=selected_source_hashes(manifest), model_config=model,
        training_log_timing={
            'sb3_order': 'collect_rollouts -> dump_logs -> train',
            'train_metrics': 'train/* usually describes the previous PPO update, while logged timesteps identify the just-completed rollout. First dump may lack train/*; final update may have no subsequent dump.',
            'completion_source': 'Execution status comes from status.json, not from the last train/* record or the requested budget.'},
        aggregate=episode_summary(episodes), first_32=episode_summary(episodes[:32]), last_32=episode_summary(episodes[-32:]),
        per_material_training=[dict(mass_kg=m, friction=mu, **episode_summary(values))
            for (m, mu), values in sorted(by_material.items())],
        per_material_last_32=[dict(mass_kg=m, friction=mu, **episode_summary(values[-32:]))
            for (m, mu), values in sorted(by_material.items())],
        episode_batch_curve=curve, rollout_metrics=logs,
        resume_interpretation='This directory contains additional training records, not another independent seed.')


def read_reset(reader, directory, status, expected_keys):
    data = reader.json(directory / 'reset_consistency.json')
    missing = sorted(set(expected_keys) - set(data or {}))
    values = [] if data is None else list(data.values())
    finite = bool(values) and all(isinstance(v, (int, float)) and np.isfinite(v) and v >= 0 for v in values)
    passed = finite and not missing and max(values) <= 1e-5 and status.get('status') == 'completed'
    if finite and max(values) > 1e-5:
        reader.warnings.append(f'{reader.name(directory)}: physical reset repeatability failed')
    return {'recorded_max_differences': data, 'missing_required_channels': missing,
            'matched_comparison_validated': bool(passed),
            'all_recorded_differences_exactly_zero': bool(finite and not missing and all(v == 0 for v in values)),
            'status': 'passed' if passed else 'failed' if finite and max(values) > 1e-5 else 'pending'}


def read_results(reader, directory):
    raw = reader.json(directory / 'results.json', True)
    if raw is None:
        return []
    if not isinstance(raw, list):
        reader.warnings.append(f'{reader.name(directory)}/results.json is not a list')
        return []
    rows = []
    for item in raw:
        if not isinstance(item, dict) or 'task_successes' not in item:
            reader.warnings.append(f'{reader.name(directory)}: unsupported result row')
            continue
        episodes = item.get('episodes', [])
        summary = episode_summary(episodes)
        for name in ('task_successes', 'strict_successes'):
            if item.get(name) != summary[name]:
                reader.warnings.append(f'{reader.name(directory)}: {name} aggregate disagrees with episodes')
        rows.append(dict(controller=item.get('controller', item.get('name', 'unknown')),
                         gain=item.get('gain'), **summary, per_lane=episodes))
    return rows


def read_scan(reader, directory, manifest):
    status = reader.json(directory / 'status.json') or {}
    protocol = reader.json(directory / 'environment/environment_protocol.json') or {}
    rows = read_results(reader, directory)
    conditions = manifest.get('conditions_kg_mu', [])
    masses = sorted(set(round(pair[0], 8) for pair in conditions))
    frictions = sorted(set(round(pair[1], 8) for pair in conditions))
    task = [[None for _ in frictions] for _ in masses]
    strict = [[None for _ in frictions] for _ in masses]
    # Minimum among actually tested successful gains; no assumption of monotonicity.
    for row in rows:
        if row['controller'] == 'reset_check' or row['gain'] is None:
            continue
        for episode in row['per_lane']:
            m = min(range(len(masses)), key=lambda i: abs(masses[i] - episode['mass_kg']))
            f = min(range(len(frictions)), key=lambda i: abs(frictions[i] - episode['friction']))
            if abs(masses[m] - episode['mass_kg']) > 1e-6 or abs(frictions[f] - episode['friction']) > 1e-6:
                reader.warnings.append('Scan episode does not match declared material grid')
                continue
            for key, matrix in [('task_success', task), ('strict_success', strict)]:
                if episode.get(key):
                    matrix[m][f] = row['gain'] if matrix[m][f] is None else min(matrix[m][f], row['gain'])
    return dict(name=reader.name(directory), execution_status=status.get('status', 'pending'),
        evidence='development_material_action_reachability_only', conditions_kg_mu=conditions,
        masses_kg=masses, frictions=frictions, minimum_tested_task_success_gain=task,
        minimum_tested_strict_success_gain=strict, tested_gains=manifest.get('gains'),
        task_reachable_grid_cells=sum(v is not None for line in task for v in line),
        strict_reachable_grid_cells=sum(v is not None for line in strict for v in line),
        config=manifest.get('config', protocol.get('config')), rows=rows,
        reset=read_reset(reader, directory, status, ['clearance_m', 'contact_load_N', 'slip_speed_m_s', 'fault']),
        return_comparison_allowed=False,
        reward_warning='Scan ran older ungated height reward and failure_penalty=25; current training gates progress by physical contact and records each run\'s failure penalty separately. Never rank current policies against scan returns.')


def paired_success_load_comparison(tactile, baseline, conditions):
    """Pair unique declared materials before selecting joint task successes.

    All-condition success differences always accompany the conditional load
    comparison. Missing/duplicate/mismatched materials or labels are rejected,
    rather than silently paired by row order or treated as failures.
    """
    result = {'tactile_controller': tactile['controller'], 'baseline_controller': baseline['controller'],
        'comparison_valid': False, 'rejection_reason': None,
        'conditions': len(conditions),
        'task_success_count_delta': None, 'strict_success_count_delta': None,
        'task_success_fraction_delta': None, 'strict_success_fraction_delta': None,
        'common_task_success_lane_count': None, 'common_task_success_materials_kg_mu': [],
        'tactile_mean_load_N_on_common_task_success': None,
        'baseline_mean_load_N_on_common_task_success': None,
        'paired_mean_load_delta_N_on_common_task_success': None,
        'paired_mean_load_reduction_fraction_on_common_task_success': None,
        'difference_sign': 'tactile minus baseline; positive success difference and negative load difference favor tactile',
        'reduction_definition': '1 - mean(tactile load)/mean(baseline load) on the identical joint-task-success material subset; not mean of per-lane ratios',
        'scope': 'Conditional load comparison never replaces all-condition success, strict/contact constraints, reset audit or original acceptance gate'}
    try:
        declared = np.asarray(conditions, dtype=float)
        if (declared.ndim != 2 or declared.shape[1] != 2 or not len(declared)
                or not np.isfinite(declared).all() or (declared[:, 0] <= 0).any()
                or (declared[:, 1] < 0).any()):
            raise ValueError('Invalid or missing declared material conditions')

        def align(row):
            episodes = row['per_lane']
            materials = np.asarray([[episode['mass_kg'], episode['friction']] for episode in episodes], float)
            if materials.shape != declared.shape or not np.isfinite(materials).all():
                raise ValueError(f"{row['controller']}: material count/shape/nonfinite mismatch")
            matches = np.isclose(declared[:, None, :], materials[None, :, :], rtol=0, atol=1e-7).all(-1)
            if not ((matches.sum(0) == 1).all() and (matches.sum(1) == 1).all()):
                raise ValueError(f"{row['controller']}: missing, mismatched or ambiguous duplicate materials")
            indices = matches.argmax(1)
            aligned = [episodes[int(index)] for index in indices]
            for episode in aligned:
                if not all(isinstance(episode.get(key), (bool, np.bool_))
                           for key in ('task_success', 'strict_success')):
                    raise ValueError(f"{row['controller']}: missing/nonboolean success labels")
            return aligned, indices.tolist()

        left, left_indices = align(tactile)
        right, right_indices = align(baseline)
        task_left = np.array([episode['task_success'] for episode in left], bool)
        task_right = np.array([episode['task_success'] for episode in right], bool)
        strict_left = np.array([episode['strict_success'] for episode in left], bool)
        strict_right = np.array([episode['strict_success'] for episode in right], bool)
        common_indices = np.flatnonzero(task_left & task_right)
        left_load, right_load = [], []
        for index in common_indices:
            a, b = left[int(index)]['mean_contact_force_N'], right[int(index)]['mean_contact_force_N']
            if not all(isinstance(value, (int, float, np.number)) and not isinstance(value, (bool, np.bool_))
                       and np.isfinite(value) and value >= 0 for value in (a, b)):
                raise ValueError('Invalid load on a jointly successful material')
            left_load.append(float(a)); right_load.append(float(b))
        task_delta = int(task_left.sum() - task_right.sum())
        strict_delta = int(strict_left.sum() - strict_right.sum())
        result.update(comparison_valid=True,
            tactile_task_successes=int(task_left.sum()), baseline_task_successes=int(task_right.sum()),
            tactile_strict_successes=int(strict_left.sum()), baseline_strict_successes=int(strict_right.sum()),
            task_success_count_delta=task_delta, strict_success_count_delta=strict_delta,
            task_success_fraction_delta=task_delta / len(declared),
            strict_success_fraction_delta=strict_delta / len(declared),
            common_task_success_lane_count=len(common_indices),
            common_task_success_materials_kg_mu=declared[common_indices].tolist(),
            material_alignment={'tactile_source_indices': left_indices, 'baseline_source_indices': right_indices})
        if len(common_indices):
            a, b = float(np.mean(left_load)), float(np.mean(right_load))
            result.update(tactile_mean_load_N_on_common_task_success=a,
                baseline_mean_load_N_on_common_task_success=b,
                paired_mean_load_delta_N_on_common_task_success=float(np.mean(np.asarray(left_load) - right_load)),
                paired_mean_load_reduction_fraction_on_common_task_success=1 - a / b if b > 0 else None)
    except (KeyError, TypeError, ValueError) as exc:
        result['rejection_reason'] = str(exc)
    return result


def read_evaluation(reader, directory, manifest, frozen_config):
    status = reader.json(directory / 'status.json') or {}
    protocol = reader.json(directory / 'environment/environment_protocol.json') or {}
    rows = read_results(reader, directory)
    conditions = manifest.get('conditions_kg_mu', [])
    config = manifest.get('config', protocol.get('config', {}))
    fixed = next((row for row in rows if row['controller'] == 'fixed_2'), None)
    for row in rows:
        row['task_failures'] = [episode for episode in row['per_lane'] if episode.get('task_success') is False]
        row['load_mean_includes_task_failures'] = bool(row['task_failures'])
        row['strict_only_failures'] = [episode for episode in row['per_lane']
            if episode.get('task_success') is True and episode.get('strict_success') is False]
        row['load_change_vs_fixed_2_fraction'] = ((row['mean_load_N'] / fixed['mean_load_N'] - 1)
            if fixed and fixed['mean_load_N'] and row['mean_load_N'] is not None else None)
    by_controller = {row['controller']: row for row in rows}
    tactile, blind, frozen = (by_controller.get(name) for name in
                             ('tactile', 'no_tactile', 'tactile_frozen_after_probe'))
    held_gain = by_controller.get('tactile_gain_hold_after_probe')
    paired = [paired_success_load_comparison(tactile, by_controller[name], conditions)
        for name in ('fixed_2', 'no_tactile', 'tactile_gain_hold_after_probe')
        if tactile and name in by_controller]
    for comparison in paired:
        if not comparison['comparison_valid']:
            reader.warnings.append(f"{reader.name(directory)}: tactile vs {comparison['baseline_controller']} "
                f"paired comparison rejected: {comparison['rejection_reason']}")
    diagnostics = {
        'tactile_more_task_successes_than_blind': bool(tactile and blind and tactile['task_successes'] > blind['task_successes']),
        'frozen_tactile_after_probe_fewer_task_successes': bool(tactile and frozen
            and frozen['task_successes'] < tactile['task_successes']),
        'frozen_tactile_after_probe_fewer_strict_successes': bool(tactile and frozen
            and frozen['strict_successes'] < tactile['strict_successes']),
        'frozen_tactile_after_probe_no_worse_success_and_load': bool(tactile and frozen
            and frozen['task_successes'] >= tactile['task_successes']
            and frozen['strict_successes'] >= tactile['strict_successes']
            and frozen['mean_load_N'] is not None and tactile['mean_load_N'] is not None
            and frozen['mean_load_N'] <= tactile['mean_load_N']),
        'gain_hold_after_probe_no_worse_success': bool(tactile and held_gain
            and held_gain['task_successes'] >= tactile['task_successes']
            and held_gain['strict_successes'] >= tactile['strict_successes']),
        'tactile_load_reduction_vs_gain_hold_fraction': ((1 - tactile['mean_load_N'] / held_gain['mean_load_N'])
            if tactile and held_gain and tactile['mean_load_N'] is not None and held_gain['mean_load_N'] else None),
        'interpretation_scope': 'Single seed and one attempt per condition; early tactile usefulness and continued-feedback benefit are separate hypotheses.'}
    return dict(name=reader.name(directory), split=manifest.get('split', 'unspecified'),
        evidence='frozen_deterministic_evaluation', execution_status=status.get('status', 'pending'),
        seed=manifest.get('seed'), conditions_kg_mu=conditions,
        unique_material_conditions=len(set(tuple(round(v, 7) for v in pair) for pair in conditions)),
        models=manifest.get('models', {}), config=config, config_matches_frozen=config == frozen_config,
        actor_inputs=manifest.get('actor_inputs'), state_history=protocol.get('state_history'),
        tactile_feedback_diagnostics=diagnostics, paired_success_load_comparisons=paired,
        state_history_60hz_12frames=protocol.get('tactile_hz') == 60 and bool(protocol.get('state_history')),
        source_hashes=selected_source_hashes(manifest), rows=rows,
        reset=read_reset(reader, directory, status,
            ['clearance_m', 'contact_load_N', 'slip_speed_m_s', 'fault', 'state_before',
             'tactile_before', 'prefix_state', 'prefix_tactile', 'actions', 'applied_gains']),
        comparison_warning='One attempt per controller per material condition. Interventions are diagnostics, not independent retraining or standalone evidence of tactile benefit.')


def reward_scale_match_checks(touch, blind):
    """A matched numerical treatment needs actual, consistent saved scales."""
    return {
        'same_actual_training_reward_scale': touch['training_reward_scale'] is not None
            and touch['training_reward_scale'] == blind['training_reward_scale'],
        'saved_training_reward_scale_verified': touch['saved_reward_scale_verified']
            and blind['saved_reward_scale_verified'],
        'initial_final_training_reward_scale_verified': touch['initial_final_reward_scale_verified']
            and blind['initial_final_reward_scale_verified'],
    }


def training_check_applicability(touch, blind, checks):
    """Annotate raw comparisons without replacing their values or granting a pass."""
    both_resumed = bool(touch.get('resumed') and blind.get('resumed'))
    fresh_only = {'same_initial_action_noise_scale', 'same_policy_initialization'} if both_resumed else set()
    return {'comparison_scope': 'cross_ablation_continuations_from_separately_trained_parents'
            if both_resumed else 'fresh_training_comparison' if not touch.get('resumed') and not blind.get('resumed')
            else 'mixed_fresh_and_continued_training',
        'not_applicable_as_fresh_initialization_checks': sorted(fresh_only & checks.keys()),
        'raw_false_other_checks': [key for key, value in checks.items() if not value and key not in fresh_only],
        'interpretation': 'Raw checks and their all-match aggregate are unchanged. '
            'For resumed tactile versus no-tactile runs, learned parent weights/noise may differ as expected; '
            'fresh equal-weight noise and initialization requirements do not certify or reject a continuation. '
            'If stage-2 penalty branches are declared, compare P75/P200 within the same ablation and parent checkpoint; '
            'a budget-only continuation does not introduce a penalty contrast. '
            'Other false checks remain visible and may indicate missing evidence or real differences, not automatically a failure.'}


def training_comparisons(training):
    pairs = []
    for touch in training:
        if touch['uses_tactile'] is not True:
            continue
        for blind in training:
            if blind['uses_tactile'] is not False or touch['seed'] != blind['seed']:
                continue
            touch_noise, blind_noise = touch['exploration'], blind['exploration']
            same_kind = (touch_noise['use_sde'] == blind_noise['use_sde'] and
                         touch_noise['normalize_sde_latent'] == blind_noise['normalize_sde_latent'])
            if same_kind and touch_noise['use_sde']:
                if touch_noise['normalize_sde_latent']:
                    ta, ba = (touch_noise['initial_uniform_weight_unit_latent_sigma'],
                              blind_noise['initial_uniform_weight_unit_latent_sigma'])
                    scale_match = (ta is not None and ba is not None and bool(np.isclose(ta, ba, rtol=1e-5, atol=1e-6))
                        and touch_noise['sde_normalization_eps'] == blind_noise['sde_normalization_eps'])
                    scale_basis = 'Equal initial uniform weights with unit noise latent; excludes zero/near-zero latent and later learned weights.'
                else:
                    ta, ba = touch_noise['actual_initial_prefix_sigma'], blind_noise['actual_initial_prefix_sigma']
                    scale_match = (ta is not None and ba is not None and
                        all(ta.get(key) is not None and ba.get(key) is not None and
                            bool(np.isclose(ta[key], ba[key], rtol=1e-5, atol=1e-6)) for key in ('min', 'p50', 'p95', 'max', 'mean')))
                    scale_basis = 'Actual checkpoint prefix distribution audit; fresh CPU calibration cannot certify saved legacy models.'
            else:
                scale_match = same_kind and touch_noise['requested_std'] == blind_noise['requested_std']
                scale_basis = 'Same iid initial std, or incompatible exploration families.'
            touch_construction, blind_construction = touch['initialization']['construction'], blind['initialization']['construction']
            checks = {
                'same_state_history': touch['state_history_60hz_12frames'] and blind['state_history_60hz_12frames']
                    and touch['state_history'] == blind['state_history']
                    and touch['actor_state_fields'] == blind['actor_state_fields'],
                'same_conditions': touch['conditions_kg_mu'] == blind['conditions_kg_mu'],
                'same_config': touch['config'] == blind['config'],
                'same_rates': touch['rates_hz'] == blind['rates_hz'],
                'same_rollout_schedule': touch['rollout_schedule'] == blind['rollout_schedule'],
                'same_requested_additional_budget': touch['requested_additional_timesteps'] == blind['requested_additional_timesteps'],
                'same_starting_timestep': touch['timesteps_at_run_start'] == blind['timesteps_at_run_start'],
                'same_completed_additional_budget': touch['execution_status'] == blind['execution_status'] == 'completed'
                    and touch['timesteps'] is not None and blind['timesteps'] is not None
                    and touch['timesteps'] - touch['timesteps_at_run_start'] == blind['timesteps'] - blind['timesteps_at_run_start'],
                'same_policy_initialization': touch['initialization'] == blind['initialization'],
                'same_feature_extractor_topology': touch['feature_sharing']['share_features_extractor']
                    == blind['feature_sharing']['share_features_extractor'],
                'saved_feature_extractor_topology_verified': touch['feature_sharing']['saved_topology_verified']
                    and blind['feature_sharing']['saved_topology_verified'],
                'same_actual_optimizer_eps': touch['optimizer']['optimizer_param_group_eps'] is not None
                    and touch['optimizer']['optimizer_param_group_eps'] == blind['optimizer']['optimizer_param_group_eps'],
                'saved_optimizer_eps_verified': touch['optimizer']['saved_optimizer_eps_verified']
                    and blind['optimizer']['saved_optimizer_eps_verified'],
                **reward_scale_match_checks(touch['reward_scaling'], blind['reward_scaling']),
                **training_objective_match_checks(touch, blind),
                'explicit_matching_cpu_construction': touch_construction == blind_construction
                    and touch_construction['policy_construction_device_context'] == 'cpu'
                    and touch_construction['torch_default_dtype_at_policy_construction'] is not None
                    and touch_construction['policy_parameter_devices_at_run_start'] == ['cpu'],
                'same_exploration_family_and_resampling': same_kind and touch_noise['sde_sample_freq'] == blind_noise['sde_sample_freq'],
                'same_initial_action_noise_scale': bool(scale_match),
                'same_core_source_hashes': bool(touch['source_hashes']) and touch['source_hashes'] == blind['source_hashes'],
                'same_ppo_hyperparameters': all(touch['model_config'].get(key) is not None and
                    touch['model_config'].get(key) == blind['model_config'].get(key)
                    for key in ('n_steps', 'batch_size', 'n_epochs', 'learning_rate', 'gamma', 'gae_lambda', 'ent_coef')),
            }
            pairs.append({'tactile_run': touch['name'], 'no_tactile_run': blind['name'],
                          'protocol_match_checks': checks, 'all_protocol_checks_match': all(checks.values()),
                          'comparison_type': 'within_variant' if touch['variant'] == blind['variant'] else 'cross_variant_not_tactile_ablation',
                          'noise_scale_match_basis': scale_basis,
                          'applicability': training_check_applicability(touch, blind, checks),
                          'missing_metadata_is_not_a_mismatch_proof': True,
                          'not_a_performance_comparison': True})
    return pairs


def completed_independent_seeds(root, training):
    """A resumed checkpoint inherits its originating run; a new CLI seed is not a fresh model."""
    by_directory = {(root / run['name']).resolve(): run for run in training}
    seeds = set()
    for run in training:
        if run['uses_tactile'] is not True or run['execution_status'] != 'completed':
            continue
        origin, seen = run, set()
        while origin['resumed']:
            path = origin.get('input_checkpoint')
            parent = Path(path).parent.resolve() if path else None
            # Periodic checkpoints live one level below their training run.
            if parent is not None and parent.name == 'checkpoints':
                parent = parent.parent
            if parent in seen or parent not in by_directory:
                origin = None
                break
            seen.add(parent)
            origin = by_directory[parent]
        if origin is not None and origin['seed'] is not None:
            seeds.add(origin['seed'])
            run['independent_origin_run'] = origin['name']
    return sorted(seeds)


def continuation_audits(root, training):
    """Describe resume provenance using already-read metadata; never open parent checkpoints."""
    root = Path(root).resolve()
    by_directory = {(root / run['name']).resolve(): run for run in training}

    def parent_run(checkpoint):
        if not checkpoint:
            return None
        path = Path(checkpoint)
        # CLI paths may be absolute or relative to the project/report directory.
        candidates = [path] if path.is_absolute() else [base / path for base in (Path.cwd(), root, *root.parents)]
        matches = {}
        for candidate in candidates:
            directory = candidate.parent.resolve()
            if directory.name == 'checkpoints':
                directory = directory.parent
            if directory in by_directory:
                matches[str(directory)] = by_directory[directory]
        return next(iter(matches.values())) if len(matches) == 1 else None

    def stage(run):
        # Missing resume metadata must not become a fresh start at zero.
        start = run.get('model_config', {}).get('timesteps_at_run_start',
            run.get('timesteps_at_run_start') if not run.get('resumed') else None)
        requested, total = run.get('requested_additional_timesteps'), run.get('timesteps')
        actual = run.get('actual_additional_timesteps')
        if actual is None and start is not None and total is not None:
            actual = total - start
        objective = run.get('training_reward_config', {})
        return {'run': run['name'], 'resumed': bool(run.get('resumed')),
            'start_timesteps': start, 'requested_additional_timesteps': requested,
            'requested_cumulative_timesteps': start + requested if start is not None and requested is not None else None,
            'actual_additional_timesteps': actual, 'recorded_cumulative_timesteps': total,
            'execution_status': run.get('execution_status'),
            'failure_penalty': objective.get('failure_penalty'),
            'runtime_reward_config_verified': objective.get('complete_resolved_and_environment_config_verified', False)}

    audits = []
    for run in training:
        if not run.get('resumed'):
            continue
        checkpoint = run.get('input_checkpoint')
        parent = parent_run(checkpoint)
        chain, seen, current = [], set(), run
        complete = True
        while current is not None:
            if current['name'] in seen:
                complete = False
                break
            seen.add(current['name'])
            chain.append(stage(current))
            if not current.get('resumed'):
                break
            current = parent_run(current.get('input_checkpoint'))
            if current is None:
                complete = False
        chain.reverse()
        parent_final = bool(parent is not None and checkpoint and Path(checkpoint).name == 'final_model.zip')
        expected_sha = parent.get('final_checkpoint_sha256') if parent_final else None
        input_sha = run.get('input_checkpoint_sha256')
        current_stage = stage(run)
        audits.append({'run': run['name'], 'training_kind': 'checkpoint_continuation_not_fresh',
            'input_checkpoint': checkpoint, 'input_checkpoint_sha256': input_sha,
            'parent_run': parent['name'] if parent else None,
            'parent_reference_resolved': parent is not None,
            'parent_final_checkpoint_sha256': expected_sha,
            'parent_final_checkpoint_hash_matches': input_sha == expected_sha if input_sha and expected_sha else None,
            'parent_final_timestep_matches_start': current_stage['start_timesteps'] == parent.get('timesteps')
                if parent_final and current_stage['start_timesteps'] is not None and parent.get('timesteps') is not None else None,
            'lineage_complete': complete, 'stages': chain, 'current_stage': current_stage,
            'origin_run': chain[0]['run'] if complete else None,
            'stage_failure_penalties': [item['failure_penalty'] for item in chain],
            'saved_initial_model_meaning': 'Restored already-trained parent at this stage start; not the original untrained constant-gain baseline.',
            'scope': 'Checkpoint identity and stage budgets/objectives from already-read metadata only. '
                'This does not prove bitwise uninterrupted training or equal tactile/no-tactile parent weights. '
                'A continuation is not an additional independent training seed.'})
    return audits


def continuation_markdown(audits):
    if not audits:
        return []
    lines = ['', '**续训来源与分阶段预算**：下表使用已读取的父run及当前run元数据。'
        '“申请累计”是起点加申请新增步数，不表示已经完成。续训的`initial_model.zip`是已训练父模型的恢复状态，'
        '不能冒充原始未训练的恒定增益对照；完整效果仍须保留原始初始策略和固定增益基线。'
        '每个续训目录只统计本阶段episode；它不增加独立训练seed数。', '',
        '| 续训运行 | 父run | 本阶段起点 / 申请新增 / 申请累计 | 实际新增 / 已记录累计 | 各阶段实际P（配置核验） | 父final SHA / 起点核验 |',
        '| --- | --- | --- | --- | --- | --- |']
    for audit in audits:
        stage = audit['current_stage']
        penalties = ' → '.join(f"{item['failure_penalty'] if item['failure_penalty'] is not None else '未记录'}"
            f"（{'已核验' if item['runtime_reward_config_verified'] else '未核验'}）" for item in audit['stages'])
        if not audit['lineage_complete']:
            penalties = '上游未解析 → ' + penalties
        def value(key):
            return stage[key] if stage[key] is not None else '未记录'
        lines.append(f"| {audit['run']} | {audit['parent_run'] or '未解析'} | "
            f"{value('start_timesteps')} / {value('requested_additional_timesteps')} / {value('requested_cumulative_timesteps')} | "
            f"{value('actual_additional_timesteps')} / {value('recorded_cumulative_timesteps')} | {penalties} | "
            f"{audit['parent_final_checkpoint_hash_matches']} / {audit['parent_final_timestep_matches_start']} |")
    lines += ['', '父checkpoint路径与完整SHA、各阶段起止计数保存在JSON的`continuation_audits`。'
        'SHA核验只核对输入checkpoint与已记录父final身份；不等于训练不中断的逐位复现，也不要求触觉与无触觉父模型权重相同。'
        '缺失来源或实际元数据保留未解析/未核验，不根据目录名补齐。', '']
    return lines


def continuation_parent_contrasts(training, audits):
    """Identify same-ablation penalty branches by recorded parents, not equal fresh noise."""
    runs = {run['name']: run for run in training}
    grouped = defaultdict(list)
    for audit in audits:
        run = runs.get(audit['run'], {})
        if audit['parent_run'] is not None and run.get('uses_tactile') is not None:
            grouped[(run['uses_tactile'], audit['parent_run'])].append(audit)
    result = []
    for (uses_tactile, parent), branches in grouped.items():
        if len({b['current_stage']['failure_penalty'] for b in branches
                if b['current_stage']['failure_penalty'] is not None}) < 2:
            continue
        shas = [b['input_checkpoint_sha256'] for b in branches]
        checks = [b['parent_final_checkpoint_hash_matches'] for b in branches]
        status = 'matched_parent_identity' if all(check is True for check in checks) and len(set(shas)) == 1 else (
            'parent_identity_mismatch' if any(check is False for check in checks)
            or len({sha for sha in shas if sha is not None}) > 1 else 'unverified_parent_identity')
        result.append({'uses_tactile': uses_tactile, 'parent_run': parent, 'parent_identity_status': status,
            'branches': [{'run': b['run'], 'failure_penalty': b['current_stage']['failure_penalty'],
                'input_checkpoint_sha256': b['input_checkpoint_sha256'],
                'parent_final_checkpoint_hash_matches': b['parent_final_checkpoint_hash_matches']} for b in branches],
            'scope': 'Same-ablation stage-2 penalty contrast identified by parent metadata. '
                'This does not independently recheck restored tensors/optimizer or all remaining protocol settings; '
                'the recorded startup audit supplies that separate evidence.'})
    return result


def read_confirmation_acceptance(reader, evaluations, include_heldout=False):
    """Accept only an explicit, fingerprint-bound review of three heldout suites.

    The default branch returns before even inspecting the certificate path.
    Completed jobs or good-looking outcomes never create an acceptance decision.
    """
    pending = {'status': 'not_requested', 'effectiveness': 'pending', 'goal_acceptance': 'pending'}
    if not include_heldout:
        return pending
    certificate_path = reader.root / 'diagnostics/confirmation_acceptance_review.json'
    certificate = reader.json(certificate_path)
    if certificate is None:
        return dict(pending, status='missing_or_unreadable_certificate')

    def require(condition, message):
        if not condition:
            raise ValueError(message)

    def pinned_json(identity, expected_relative):
        path = Path(identity['path'])
        path = path.resolve() if path.is_absolute() else (reader.root / path).resolve()
        require(path == (reader.root / expected_relative).resolve(), 'Unexpected review evidence path: ' + str(path))
        value = reader.json(path, True)
        require(isinstance(value, dict), 'Missing/unreadable review evidence: ' + str(path))
        require(reader.sources[reader.name(path)]['sha256'] == identity['sha256'], 'Review evidence SHA mismatch: ' + str(path))
        return value

    try:
        require(certificate['schema'] == 'confirmation_acceptance_review_v1', 'Unsupported acceptance schema')
        require(certificate['status'] == 'passed_predeclared_benchmark_limited_scope' and
                certificate['effectiveness'] == 'positive_with_limits' and
                certificate['goal_acceptance'] == 'passed_predeclared_scope', 'Acceptance status is not an explicit limited-scope pass')
        require(certificate['seeds'] == [1, 2, 3], 'All three confirming seeds must be retained')
        require(isinstance(certificate['decision_utc'], str) and certificate['decision_utc'], 'Missing decision date')
        require(isinstance(certificate['limitations'], list) and certificate['limitations'] and
                all(isinstance(v, str) and v.strip() for v in certificate['limitations']), 'Explicit limitations are required')
        protocol = reader.json(reader.root / 'predeclared_protocol.json', True) or {}
        require(protocol.get('proposed_engineering_gate') == {
                    'heldout_task_success_fraction_min': .95,
                    'mean_load_reduction_vs_validation_selected_global_fixed_gain_min': .1,
                    'hold_contact_fraction_min': .99, 'max_unsupported_hold_s': .05,
                    'independent_training_seeds_min': 3},
                'Current declared engineering criteria differ from the reviewed five original gates')
        audit = pinned_json(certificate['audit'], 'diagnostics/confirmation_heldout_audit.json')
        independent = pinned_json(certificate['independent_review'], 'diagnostics/confirmation_heldout_independent_review.json')
        require(independent.get('status') == 'passed_integrity_audit' and
                independent.get('all_three_seeds_engineering_pass') is True, 'Independent review did not confirm the three-seed engineering pass')
        require(audit['status'] == 'complete_audit' and audit['errors'] == [] and audit['missing_or_pending'] == [] and
                audit['summary']['all_three_seed_engineering_pass'] is True and
                set(audit['seeds']) == {'1', '2', '3'}, 'Heldout integrity/engineering audit did not pass for every seed')
        suites, declared_rows = certificate['heldout_suites'], certificate['per_seed_results']
        require(len(suites) == len(declared_rows) == 3 and {s['seed'] for s in suites} == {1, 2, 3} and
                {s['seed'] for s in declared_rows} == {1, 2, 3}, 'Missing/duplicate seed in acceptance review')
        require(len({s['name'] for s in suites}) == 3, 'Duplicate accepted suite name')
        checked_rows = []
        for seed in (1, 2, 3):
            declared_suite = next(s for s in suites if s['seed'] == seed)
            candidates = [s for s in evaluations if s['name'] == declared_suite['name']]
            require(len(candidates) == 1, f'Seed{seed} accepted suite is absent/duplicated in this report')
            suite, audited = candidates[0], audit['seeds'][str(seed)]
            require(suite['split'] == 'heldout' and suite['execution_status'] == 'completed' and
                    audited['status'] == 'audited' and audited['split'] == 'heldout', f'Seed{seed} suite is not a completed audited heldout run')
            directory = (reader.root / suite['name']).resolve()
            require(directory.is_relative_to(reader.root) and directory == Path(audited['directory']).resolve(), 'Suite/audit directory mismatch')
            model = suite['models']['tactile']
            require(model['sha256'] == declared_suite['model_sha256'] == audited['models']['tactile']['sha256'] and
                    audited['models']['tactile']['data']['seed'] == seed and
                    audited['models']['tactile']['data']['num_timesteps'] == 786432, 'Accepted final model SHA/seed/budget differs')
            # Bind displayed metrics to the exact result files covered by the
            # physical-trace audit, so a stale certificate cannot bless edits.
            for relative in ('manifest.json', 'results.json', 'status.json', 'reset_consistency.json',
                             'environment/environment_protocol.json'):
                path = directory / relative
                recorded = reader.sources.get(reader.name(path))
                require(recorded is not None and recorded['sha256'] == audit['sources'].get(str(path), {}).get('sha256'),
                        'Reported suite source differs from audited source: ' + str(path))
            rows = {r['controller']: r for r in suite['rows']}
            live, fixed = rows['tactile'], rows['fixed_2']
            lanes = audited['all_lane_metrics']['tactile']
            require(len(lanes) == live['task_label_count'] == live['strict_label_count'] == fixed['task_label_count'] == 9,
                    'Every accepted seed must have exactly nine cases')
            force, reference = live['mean_load_N'], fixed['mean_load_N']
            require(np.isfinite([force, reference]).all() and force >= 0 and reference > 0, 'Invalid force comparison')
            coverage = min(l['hold_contact_fraction'] for l in lanes)
            gap = max(l['hold_max_unsupported_duration_s'] for l in lanes)
            computed = {'seed': seed, 'task_successes': live['task_successes'], 'strict_successes': live['strict_successes'],
                        'cases': 9, 'mean_load_N': force, 'fixed2_mean_load_N': reference,
                        'force_reduction_fraction': 1 - force / reference,
                        'min_contact_fraction': coverage, 'max_unsupported_hold_s': gap,
                        'engineering_pass': bool(live['task_successes'] == 9 and force <= .9 * reference + 1e-12 * max(1., reference)
                            and coverage >= .99 - 1e-10 and gap <= .05 + 1e-10)}
            require(computed['engineering_pass'] and audited['engineering_gate']['passes'] is True and
                    all(audited['engineering_gate']['components'].values()), f'Seed{seed} does not meet the original per-seed gate')
            require(live['task_successes'] == audited['summaries']['tactile']['task_successes'] and
                    live['strict_successes'] == audited['summaries']['tactile']['strict_successes'] and
                    abs(force - audited['summaries']['tactile']['mean_load_N_all_lanes']) < 1e-8 and
                    abs(reference - audited['summaries']['fixed_2']['mean_load_N_all_lanes']) < 1e-8,
                    'Displayed aggregate differs from independent physical audit')
            declared = next(r for r in declared_rows if r['seed'] == seed)
            for key, value in computed.items():
                if isinstance(value, float):
                    require(isinstance(declared.get(key), (int, float)) and np.isfinite(declared[key]) and
                            abs(declared[key] - value) <= 1e-8, f'Certificate seed{seed} {key} differs from audited result')
                else:
                    require(type(declared.get(key)) is type(value) and declared[key] == value,
                            f'Certificate seed{seed} {key} differs from audited result')
            checked_rows.append(dict(computed, suite=suite['name'], model_sha256=model['sha256']))
        return {'status': 'verified_limited_scope', 'effectiveness': certificate['effectiveness'],
                'goal_acceptance': certificate['goal_acceptance'], 'certificate_status': certificate['status'],
                'certificate_path': reader.name(certificate_path), 'decision_utc': certificate['decision_utc'],
                'audit': certificate['audit'], 'independent_review': certificate['independent_review'],
                'per_seed_results': checked_rows, 'limitations': certificate['limitations'],
                'accepted_suite_names': [s['name'] for s in suites],
                'scope': 'Only the three named final786432 heldout suites receive this decision. Historical failures remain failures; no pooled rescue or real-robot claim.'}
    except (KeyError, TypeError, ValueError, OSError) as exc:
        reader.warnings.append('Acceptance review rejected: ' + str(exc))
        return dict(pending, status='rejected', error=str(exc))


def collect(root, include_heldout=False):
    reader = Reader(root)
    protocol = reader.json(root / 'predeclared_protocol.json', True) or {}
    config = reader.json(root / 'config.json', True) or {}
    revisions = {path.name: reader.json(path) for path in sorted(root.glob('revision_*.json'))}
    noise_calibration = reader.json(root / 'sde_prefix_noise_calibration.json')
    noise_audit = reader.json(root / 'sde_distribution_audit_40064.json')
    coverage_audits = []
    for name in ('contact_coverage_audit', 'contact_coverage_fullmc'):
        coverage_directory = root / 'diagnostics' / name
        if not (coverage_directory / 'manifest.json').is_file():
            continue
        coverage_interpretation = reader.json(coverage_directory / 'interpretation.json')
        summaries = {}
        for name in ('fixed_2', 'tactile'):
            raw = reader.json(coverage_directory / name / 'summary.json', True) or {}
            summaries[name] = {key: raw.get(key) for key in ('all_sample_count',
                'max_omitted_hand_load_N', 'omitted_hand_active_lane_samples_over_001N',
                'omitted_active_links', 'non_hand_active_links', 'tracked_metric_max_abs_error_N',
                'reference_trace_sha256', 'reference_max_abs_differences', 'phase_summary',
                'peak_hand_load_per_lane_N', 'peak_tracked_load_per_lane_N')}
        expected = {'tracked_five_load_N', 'clearance_m', 'slip_speed_m_s', 'fault', 'actions', 'applied_gains'}
        repeated = all(isinstance(row.get('reference_max_abs_differences'), dict)
            and expected <= row['reference_max_abs_differences'].keys()
            and all(value == 0 for value in row['reference_max_abs_differences'].values())
            for row in summaries.values())
        coverage_audits.append({'interpretation': coverage_interpretation,
            'comparison': reader.json(coverage_directory / 'comparison.json', True),
            'manifest': reader.json(coverage_directory / 'manifest.json', True),
            'status': reader.json(coverage_directory / 'status.json', True),
            'controllers': summaries, 'reference_replay_all_recorded_differences_zero': repeated,
            'source': reader.name(coverage_directory)})
    coverage = coverage_audits[0] if coverage_audits else None
    fullmc_review = reader.json(root / 'diagnostics/fullmc_evaluation_independent_review.json')
    training, scans, evaluations, skipped = [], [], [], []
    for current, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if d not in {'source_snapshot', 'environment', 'checkpoints', '__pycache__'})
        directory = Path(current)
        if not include_heldout:
            for d in list(dirs):
                if 'heldout' in d.lower() or 'held_out' in d.lower():
                    skipped.append(str((directory / d).relative_to(root)))
                    dirs.remove(d)
        if directory == root or 'manifest.json' not in files:
            continue
        manifest = reader.json(directory / 'manifest.json') or {}
        if manifest.get('split') == 'heldout' and not include_heldout:
            skipped.append(reader.name(directory))
            dirs[:] = []
            continue
        if manifest.get('variant') == 'adaptive_common_gain':
            training.append(read_training(reader, directory, manifest, config, noise_calibration, noise_audit))
        elif 'gains' in manifest and 'conditions_kg_mu' in manifest and 'models' not in manifest:
            scans.append(read_scan(reader, directory, manifest))
        elif manifest.get('split') in {'development', 'validation', 'heldout'}:
            evaluations.append(read_evaluation(reader, directory, manifest, config))
        # Child directories contain controller traces, not more independent runs.
    completed_seeds = completed_independent_seeds(root, training)
    resumes = continuation_audits(root, training)
    for suite in evaluations:
        suite['conditions_match_training_run_names'] = [run['name'] for run in training
            if same_material_conditions(suite['conditions_kg_mu'], run['conditions_kg_mu'])]
    seen_feedback = [suite for suite in evaluations
        if suite['conditions_match_training_run_names'] and suite['reset']['matched_comparison_validated']
        and suite['tactile_feedback_diagnostics']['frozen_tactile_after_probe_fewer_task_successes']]
    seen_checkpoints = {suite['models'].get('tactile', {}).get('sha256') for suite in seen_feedback}
    seen_checkpoints.discard(None)
    feedback_summary = {
        'seen_material_suites_where_freezing_reduces_task_success': [suite['name'] for suite in seen_feedback],
        'validation_suites_where_freezing_is_no_worse_on_success_and_load': [suite['name'] for suite in evaluations
            if suite['split'] == 'validation' and suite['reset']['matched_comparison_validated']
            and suite['tactile_feedback_diagnostics']['frozen_tactile_after_probe_no_worse_success_and_load']],
        'same_checkpoint_validation_suites_where_freezing_is_no_worse': [suite['name'] for suite in evaluations
            if suite['split'] == 'validation' and suite['reset']['matched_comparison_validated']
            and suite['tactile_feedback_diagnostics']['frozen_tactile_after_probe_no_worse_success_and_load']
            and suite['models'].get('tactile', {}).get('sha256') in seen_checkpoints],
        'scope': 'Keep both observations: a seen-material intervention effect does not prove continued-feedback benefit generalizes to new materials.'}
    acceptance = read_confirmation_acceptance(reader, evaluations, include_heldout)
    return {'generated_utc': datetime.now(timezone.utc).isoformat(), 'root': str(root),
            'report_kind': 'all_discovered_runs_no_checkpoint_selection',
            'effectiveness_status': acceptance['effectiveness'], 'goal_acceptance_status': acceptance['goal_acceptance'],
            'confirmation_acceptance_review': acceptance,
            'acceptance_interpretation': acceptance.get('scope', 'Execution completion does not establish useful tactile adaptation. '
                'Frozen held-out evidence, a validation-selected baseline, load/contact constraints, '
                'matched no-tactile controls and independent seeds require explicit review.'),
            'include_heldout': include_heldout, 'heldout_suites_not_read': sorted(set(skipped)),
            'predeclared_protocol': protocol, 'frozen_config': config,
            'development_revisions': revisions, 'sde_prefix_noise_calibration': noise_calibration,
            'sde_distribution_audit_40064': noise_audit,
            'contact_coverage_audit': coverage,
            'contact_coverage_audits': coverage_audits,
            'fullmc_independent_review': fullmc_review,
            'independent_completed_tactile_training_seeds': sorted(completed_seeds),
            'continuation_audits': resumes,
            'continuation_parent_contrasts': continuation_parent_contrasts(training, resumes),
            'training': training, 'training_protocol_comparisons': training_comparisons(training),
            'material_scans': scans, 'evaluations': evaluations,
            'continued_feedback_evidence': feedback_summary,
            'pending_files': sorted(set(reader.pending)), 'warnings': reader.warnings,
            'source_files': reader.sources}


def plot_report(root, report):
    # Display all evaluation rows in stable discovery order; never select the best suite/model.
    rows = [(suite, row) for suite in report['evaluations'] for row in suite['rows']]
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 2, figsize=(17, max(10, 5 + len(rows) * .24)), constrained_layout=True)
    ax = axes[0, 0]
    scans = report['material_scans']
    if scans:
        # Separate panels would grow without bound; all grids are retained in JSON/Markdown.
        scan = scans[0]
        values = np.array([[np.nan if v is None else v for v in row]
                           for row in scan['minimum_tested_task_success_gain']])
        if values.size:
            im = ax.imshow(np.ma.masked_invalid(values), vmin=.6, vmax=3., cmap='viridis', aspect='auto')
            fig.colorbar(im, ax=ax, label='Minimum tested task-success gain')
            for i in range(len(values)):
                for j in range(len(values[i])):
                    task = scan['minimum_tested_task_success_gain'][i][j]
                    strict = scan['minimum_tested_strict_success_gain'][i][j]
                    ax.text(j, i, f'{fmt(task, 2) if task is not None else "NA"} / '
                            f'{fmt(strict, 2) if strict is not None else "NA"}', ha='center', va='center',
                            color='white' if task is not None and task < 1.8 else 'black', fontsize=10)
            ax.set_xticks(range(len(scan['frictions'])), [f'{v:g}' for v in scan['frictions']])
            ax.set_yticks(range(len(scan['masses_kg'])), [f'{v * 1000:g}' for v in scan['masses_kg']])
        ax.set(title=f"Development reachability: {scan['name']}\nCells = task / strict minimum TESTED gain; not a learned policy",
               xlabel='Contact friction coefficient', ylabel='Mass (g)')
    else:
        ax.text(.5, .5, 'Material scan pending', ha='center', transform=ax.transAxes)
    ax = axes[0, 1]
    for run in report['training']:
        curve = run['episode_batch_curve']
        if curve:
            ax.plot([x['timesteps'] for x in curve], [x['mean_return'] for x in curve], label=run['name'])
    ax.set(title='Exploratory raw training returns (different penalty objectives are NOT directly comparable)', xlabel='Cumulative training transitions', ylabel='Mean completed-episode raw return')
    ax.grid(alpha=.2)
    if ax.lines:
        ax.legend(fontsize=7)
    else:
        ax.text(.5, .5, 'Training episode records pending', ha='center', transform=ax.transAxes)
    success_ax, load_ax = axes[1]
    if rows:
        y = np.arange(len(rows))
        task = [r['task_successes'] / r['task_label_count'] if r['task_label_count'] else 0 for _, r in rows]
        strict = [r['strict_successes'] / r['strict_label_count'] if r['strict_label_count'] else 0 for _, r in rows]
        success_ax.barh(y - .17, task, height=.30, label='Task', color='#2b7d9f')
        success_ax.barh(y + .17, strict, height=.30, label='Strict', color='#e49f40')
        labels = [f"{s['split']} | {s['name']} / {r['controller']}" for s, r in rows]
        success_ax.set_yticks(y, labels, fontsize=7)
        success_ax.set_xlim(0, 1.15)
        for i, (_, r) in enumerate(rows):
            success_ax.text(max(task[i], strict[i]) + .01, i,
                            f"{r['task_successes']}/{r['task_label_count']} | {r['strict_successes']}/{r['strict_label_count']}", fontsize=6, va='center')
        success_ax.legend(ncol=2, fontsize=8)
        load_ax.barh(y, [r['mean_load_N'] or 0 for _, r in rows], color='#66788c', height=.55)
        load_ax.set_yticks(y, [])
        for i, (_, r) in enumerate(rows):
            load_ax.text((r['mean_load_N'] or 0) + .05, i, fmt(r['mean_load_N'], 2), fontsize=7, va='center')
        success_ax.invert_yaxis(); load_ax.invert_yaxis()
        load_ax.margins(x=.15)
    else:
        for ax in (success_ax, load_ax):
            ax.text(.5, .5, 'Frozen evaluation results pending', ha='center', transform=ax.transAxes)
    success_ax.set(title='Frozen evaluations: task and strict remain separate', xlabel='Observed fraction (one trial per condition)')
    load_ax.set(title='Same rows: whole-episode summed contact load', xlabel='Mean contact force magnitude sum (N)')
    scope = ('Decision applies only to the three named held-out final models; historical outcomes remain unchanged'
             if report.get('confirmation_acceptance_review', {}).get('status') == 'verified_limited_scope'
             else 'No acceptance without explicit verified review; held-out data require opt-in')
    fig.suptitle(f"Adaptive common-gain PPO | {report['effectiveness_status']} / {report['goal_acceptance_status']}\n{scope}", fontsize=12)
    fig.savefig(root / 'overview.png', dpi=160)
    plt.close(fig)


def markdown(report):
    p = report['predeclared_protocol']
    acceptance = report.get('confirmation_acceptance_review', {})
    accepted = acceptance.get('status') == 'verified_limited_scope'
    lines = ['# 材料自适应夹持：开发、训练与评估', '',
        f"生成时间：{report['generated_utc']}。**策略有效性：{report['effectiveness_status']}；goal验收：{report['goal_acceptance_status']}。**", '',
        '程序完成不等于策略有效。此报告列出所有发现的运行，包括续训和未完成运行；不挑选最佳模型、不从测试结果反选checkpoint。', '',
        '验收结论只适用于明确列出的最终模型及留出基准。下文历史实验保留其成功、失败和限制，不随本次验收追溯改判。', '',
        '本轮仅调整拇指/食指共同PD增益，名义手形轨迹固定。task是最终3秒木块最低点始终至少5 cm且无规定碰撞；'
        'strict额外保留每个60 Hz采样至少两个触觉指节接触载荷超过0.05 N。旧字段success仍表示strict。', '',
        '## 预先声明与验收状态', '',
        f"计划训练/验证/留出条件数：{p.get('training_conditions', '待产出')} / {p.get('validation_conditions', '待产出')} / {p.get('heldout_conditions', '待产出')}。",
        f"已完成的独立触觉训练seed：{report['independent_completed_tactile_training_seeds']}；续训不增加独立seed数。"
        'seed0仅用于方案探索，本次冻结确认仅seed1/2/3。', '',
        '| 预先声明的工程门槛 | 数值 | 状态 |', '| --- | ---: | --- |']
    translations = {'heldout_task_success_fraction_min': '留出task成功比例至少',
        'mean_load_reduction_vs_validation_selected_global_fixed_gain_min': '相对验证集选定全局固定增益的载荷降低至少',
        'hold_contact_fraction_min': '保持接触比例至少', 'max_unsupported_hold_s': '保持期最长无双指支持时间至多（s）',
        'independent_training_seeds_min': '独立训练seed至少'}
    for key, value in p.get('proposed_engineering_gate', {}).items():
        state = '通过，仅限下列三个seed的留出基准' if accepted else 'pending，需匹配证据的显式验收记录'
        lines.append(f"| {translations.get(key, key)} | {value} | {state} |")
    lines += ['', '工程门槛通过也不自动证明触觉有用；仍要比较未训练初始策略、验证集选定的全局固定增益、相同预算且本体历史一致的无触觉重训练，以及冻结模型的触觉置零/错配干预。', '',
              '冻结配置：`' + json.dumps(report['frozen_config'], ensure_ascii=False) + '`。', '']
    if not report['include_heldout']:
        lines += ['**本次未读取留出测试结果。** 留出条件仅引用预声明的数量；未读取heldout_conditions或任何留出轨迹/结果。', '']
    if accepted:
        lines += ['## 已审核的受限验收结论', '',
            '**三个独立训练seed分别通过预声明的仿真留出工程门槛；触觉策略具有正向证据，结论受下列范围限制。** '
            '每个seed均单独核验9/9 task、相对fixed2平均载荷至少降低10%、每个条件接触覆盖率至少0.99且最长中断不超过0.05 s。'
            'strict按原定义另外列出；不以汇总27/27补救单个seed，也不把工程通过等同于所有条件都需要持续触觉。', '',
            f"审核时间：{acceptance['decision_utc']}。审计文件与独立复核文件SHA、报告中的三个suite及其触觉模型SHA均已匹配。", '',
            '| seed | task | strict | 触觉平均载荷(N) | fixed2平均载荷(N) | 相对fixed2降低 | 最低接触覆盖率 | 最长无支持(s) | 工程门槛 |',
            '| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |']
        for row in acceptance['per_seed_results']:
            lines.append(f"| {row['seed']} | {row['task_successes']}/{row['cases']} | {row['strict_successes']}/{row['cases']} | "
                f"{fmt(row['mean_load_N'])} | {fmt(row['fixed2_mean_load_N'])} | {100 * row['force_reduction_fraction']:.2f}% | "
                f"{row['min_contact_fraction']:.6f} | {row['max_unsupported_hold_s']:.6f} | 通过 |")
        lines += ['', '**审核记录中的限制（原文保留）：**', '']
        lines += ['- ' + limitation for limitation in acceptance['limitations']]
        lines += ['', '本次接受的suite及触觉final模型：', '']
        lines += [f"- seed{row['seed']}：`{row['suite']}`；SHA256 `{row['model_sha256']}`。" for row in acceptance['per_seed_results']]
        lines += ['', f"来源：[验收记录]({Path(report['root']) / acceptance['certificate_path']})；"
                  f"[逐条件审计]({acceptance['audit']['path']})；[独立复核]({acceptance['independent_review']['path']})。", '']
    elif report['include_heldout']:
        lines += [f"**本次验收记录状态：{acceptance.get('status', 'pending')}。** 已纳入留出结果不自动构成验收通过；"
                  '只有证书、审计、独立复核及对应模型/结果一致时才更新结论。', '']
        if acceptance.get('error'):
            lines += ['验收记录拒绝原因：' + acceptance['error'], '']
    if report['heldout_suites_not_read']:
        lines += ['跳过目录：' + '、'.join(report['heldout_suites_not_read']) + '。', '']
    lines += ['## 开发材料扫描：可达性，不是PPO表现', '',
        '**扫描使用旧奖励：高度项当时未按接触门控，失败惩罚也与新训练不同。扫描return不能与新训练/验证return横向比较。**'
        '下表只回答固定增益能否完成物理任务；最低增益是离散测试值，不代表连续最优值或成功随增益单调。', '']
    for scan in report['material_scans']:
        lines += [f"### {scan['name']}", '', f"执行状态：{scan['execution_status']}；reset检查：{scan['reset']['status']}。",
            f"task可达格点 {scan['task_reachable_grid_cells']}/{len(scan['conditions_kg_mu'])}；strict可达格点 {scan['strict_reachable_grid_cells']}/{len(scan['conditions_kg_mu'])}。", '',
            '| 质量(g) | ' + ' | '.join(f'μ={mu:g}' for mu in scan['frictions']) + ' |',
            '| ---: | ' + ' | '.join('---:' for _ in scan['frictions']) + ' |']
        for i, m in enumerate(scan['masses_kg']):
            cells = [f"{fmt(a, 2)} / {fmt(b, 2)}" for a, b in zip(scan['minimum_tested_task_success_gain'][i], scan['minimum_tested_strict_success_gain'][i])]
            lines.append(f'| {m * 1000:g} | ' + ' | '.join(cells) + ' |')
        lines += ['', '单元格：最低测试成功增益 task / strict；没有成功值表示所测增益中未达到，不能推断绝对不可达。', '',
            '| 固定增益 | task | strict | 全episode平均接触载荷(N) | 每lane峰值均值(N) | 全部lane最大峰值(N) |',
            '| ---: | ---: | ---: | ---: | ---: | ---: |']
        for row in scan['rows']:
            lines.append(f"| {row['controller']} | {row['task_successes']}/{row['task_label_count']} | {row['strict_successes']}/{row['strict_label_count']} | {fmt(row['mean_load_N'])} | {fmt(row['mean_per_lane_peak_load_N'])} | {fmt(row['max_peak_all_lanes_N'])} |")
        lines.append('')
    lines += ['## 探索训练记录', '',
        '**累计训练成功率不作为有效性证据。** 训练episode重复相同lane材料，成功比例和回报仅描述带随机动作的训练，不是独立验证或泛化率。'
        '高增益初始策略本身可能已经成功；学习效果必须与原始未训练策略及匹配固定增益比较。续训目录单列，transition轴沿用累计计数；'
        '续训保存的初始模型已继承父模型训练，含义见续训来源表。', '',
        '**日志时间语义**：SB3先收集rollout，再 `dump_logs()`，最后 `train()`。'
        '`training_metrics.jsonl` 的步数对应刚收集的数据，但 `train/*` 通常来自上一次更新；第一条可能无训练指标，'
        '最后一次更新也可能没有下一次dump。完成状态只读取 `status.json`，不根据最后一条loss或申请预算推断完成。'
        '本图使用已完成episode回报，不把 `train/*` 当作同一行步数处的冻结性能。', '',
        '| 运行 | 变体 | 类型/seed | 状态 | 累计steps（起点） | 本目录episode | task | strict | 最近32条回报均值 |',
        '| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for run in report['training']:
        a = run['aggregate']
        kind = ('触觉' if run['uses_tactile'] else '无触觉') + ('续训' if run['resumed'] else '')
        lines.append(f"| {run['name']} | {run['variant']} | {kind}/{run['seed']} | {run['execution_status']} | {run['timesteps']}（{run['timesteps_at_run_start']}） | {a['episodes']} | {a['task_successes']}/{a['task_label_count']} | {a['strict_successes']}/{a['strict_label_count']} | {fmt(run['last_32']['mean_return'])} |")
    lines += ['', '默认初始化、gain=2.75初始化、原生gSDE和归一化gSDE是不同开发变体。'
              '归一化版本同时修改噪声特征归一化、权重std=0.12、显式CPU初始化和γ=1；后续变化不能归因于单一修改。'
              'γ=1对应有限episode未折扣回报；之前γ=0.995对较晚的成功/失败结果赋予较低权重。各run的终止惩罚按下表完整配置核验，不假设跨变体奖励目标相同。', '',
              'gSDE记录中的std是潜变量噪声权重标准差。旧版按重新构造的CPU初始模型校准约0.12，'
              '但Genesis默认设备影响了真实初始化，**不能将此校准视为实际checkpoint初始噪声已匹配的证明**。'
              '已读取的真实checkpoint分布审计单独保存在JSON。归一化版等权重且非零单位噪声特征下，初始边际std约0.120004；'
              '它不保证学习后权重不增长，也不保证裁剪和增益限速之后的探索相同。', '',
              '| 运行 | γ / GAE λ | 噪声特征 | 权重std | CPU构造 / 参数设备 | 单位噪声特征初始σ |',
              '| --- | --- | --- | ---: | --- | ---: |']
    for run in report['training']:
        model, noise = run['model_config'], run['exploration']
        construction = run['initialization']['construction']
        lines.append(f"| {run['name']} | {model.get('gamma')} / {model.get('gae_lambda', '未记录')} | "
            f"{'L2归一化' if noise['normalize_sde_latent'] else '原生latent' if noise['use_sde'] else 'iid'} | "
            f"{fmt(noise['requested_std'], 6)} | {construction['policy_construction_device_context'] or '未记录'} / "
            f"{construction['policy_parameter_devices_at_run_start'] or '未记录'} | {fmt(noise['initial_uniform_weight_unit_latent_sigma'], 6)} |")
    lines += ['',
              '| 运行 | 申请新增 / 实际新增steps | 每lane rollout步数 × lane数 | minibatch / epochs | GAE λ |',
              '| --- | --- | --- | --- | --- |']
    for run in report['training']:
        schedule = run['rollout_schedule']
        completed = run['actual_additional_timesteps']
        if completed is None and run['timesteps'] is not None:
            completed = run['timesteps'] - run['timesteps_at_run_start']
        lines.append(f"| {run['name']} | {run['requested_additional_timesteps']} / {completed if completed is not None else '进行中/未记录'} | "
            f"{schedule['n_steps']} × {schedule['n_envs']} | {schedule['batch_size']} / {schedule['n_epochs']} | "
            f"{schedule['gae_lambda'] if schedule['gae_lambda'] is not None else '未记录'} |")
    lines += continuation_markdown(report.get('continuation_audits', []))
    continuation = report['development_revisions'].get('revision_low_lr_continuation.json')
    if continuation:
        common = continuation.get('common', {})
        branches = continuation.get('branches', [])
        lines += ['', '**低学习率续训预声明**：从已训练的低LR父模型恢复，每条分支申请再训练'
            f"{common.get('additional_timesteps')}步，目标累计{common.get('total_timesteps')}步；"
            '这四组不是fresh run。阶段2分支为'
            + '、'.join(f"{branch.get('id')}（P={branch.get('failure_penalty')}）" for branch in branches)
            + '，实际分阶段P和来源核验以上表为准。P75/P200分支应在各自触觉消融组内使用同一父checkpoint；'
            '触觉与无触觉父模型经过独立的第一阶段训练，权重不相同。', '',
            '恢复保留父模型和Adam状态，但新建环境并重启seed0随机数，不保留旧rollout buffer，因此不等于不中断训练。'
            '只评估声明的最终累计预算模型，不挑中间checkpoint；所有冻结评估继续用原P75目标。'
            '若该方案进入独立seed确认，各seed必须从头重做阶段1和阶段2，不能从seed0父权重开始。'
            '预声明预算与启动状态不构成训练完成或有效性验收。', '']
    mc = report['development_revisions'].get('revision_full_episode_mc.json')
    if mc:
        changes = mc.get('changes', {})
        budget = changes.get('total_timesteps')
        lines += ['', '**全episode Monte Carlo候选的冻结配置**：'
            f"rollout={changes.get('n_steps')}步、γ={changes.get('gamma')}、GAE λ={changes.get('gae_lambda')}、"
            f"batch={changes.get('batch_size')}、每组预算={budget} transitions。"
            '这是优势估计、更新时机和batch size的共同修改；266432相对262144增加1.64%，不能作为完全等预算的跨变体或仅λ消融。'
            '同一新候选的触觉/无触觉两组应使用相同预算，实际执行和配对检查以上表为准；预声明配置不表示已完成训练或有效性验收。', '']
        for audit in mc.get('post_freeze_audits', []):
            lines += [f"后补审计（{audit.get('audited_utc')}）：`{audit.get('field')}` 已补充已见材料与新验证材料两边的反馈证据。"
                f"原配置冻结时间 {mc.get('created_utc')} 保留；本次证据修订不冒充训练前声明，不改候选配置。"
                '原文字、修订前哈希与来源哈希均保存在 revision JSON 的 `post_freeze_audits`。', '']
    separate = report['development_revisions'].get('revision_separate_encoder_fullmc.json')
    if separate:
        lines += ['', '**分离编码器候选**：`share_features_extractor=False` 为actor和critic提供独立参数。'
            '先构造同seed、尚未训练的共享模板，再严格复制完整policy state到两个独立分支，保留新空优化器，'
            '并恢复模板构造后的随机数状态。初始gain仍为2.75；不是复制已学会抓握的教师。'
            '参数总数增加一份critic编码器，**actor推理结构与容量没有扩大**。'
            '相同初始动作不足以证明初始化匹配，因为零action-head权重会遮蔽编码器差异；仍需张量、存储独立性、'
            'actor特征、价值预测、分布及保存加载检查。全局梯度裁剪和critic相关优势仍可能耦合两个分支。', '',
            '| 运行 | 共享编码器 | 已保存权重存储核验 | 唯一参数总数 | 初始化方法 / 模板→目标参数数 |',
            '| --- | --- | --- | ---: | --- |']
        for run in report['training']:
            topology, init = run['feature_sharing'], run['initialization_audit']
            lines.append(f"| {run['name']} | {topology['share_features_extractor']}（{topology['metadata_basis']}） | "
                f"{topology['saved_topology_verified']} | {run['parameter_count']} | {run['initialization']['method']} / "
                f"{init.get('template_unique_parameter_count', '—')}→{init.get('target_unique_parameter_count', '—')} |")
        lines += ['', '历史manifest未写拓扑时按SB3默认共享True解析，并检查initial/final checkpoint中actor/critic张量的存储别名；'
            '若无可核对的权重或与metadata冲突，不将其标为已验证。完整模板复制和RNG metadata保存在JSON的 `initialization_audit`。'
            '配置预声明不代表候选训练已完成；本报告始终保留未完成状态。', '']
    lines += ['', '**Adam epsilon按实际优化器状态核对**：请求值与实际param-group值分开记录。'
        '历史metadata缺失时直接以CPU weights_only读取保存checkpoint的 `policy.optimizer.pth/param_groups`，'
        '不猜测默认值；初始/最终记录须一致。无保存模型时只展示请求值，actual与verified保持未核验。'
        '旧优化器state不能唯一识别Adam/AdamW等类，因此未记录的类名保留缺失，不从eps反推。', '',
        '| 运行 | 优化器类metadata | 请求Adam eps | 保存的实际各group eps | 保存状态核验 |',
        '| --- | --- | --- | --- | --- |']
    for run in report['training']:
        opt = run['optimizer']
        lines.append(f"| {run['name']} | {opt['optimizer_class'] or '未记录'} | "
            f"{opt['adam_eps_requested'] if opt['adam_eps_requested'] is not None else '未指定'} | "
            f"{opt['optimizer_param_group_eps'] if opt['optimizer_param_group_eps'] is not None else '未核验'} | "
            f"{opt['saved_optimizer_eps_verified']}（{opt['basis']}） |")
    lines += ['', '**实际训练目标与学习率**：P来自完整`config.resolved.json`与环境protocol的交叉核验，'
        '并检查manifest请求配置；P不保存在PPO checkpoint中，不能从目录名或当前默认值推断。'
        '常数LR直接核验initial/final checkpoint JSON的`learning_rate`、优化器各group的`lr`及actual model metadata。'
        '缺失或冲突保持未核验，CLI请求值不能替代实际值。', '',
        '| 运行 | 实际failure_penalty / 完整配置已核验 | 实际LR / 初末模型已核验 | LR依据 |',
        '| --- | --- | --- | --- |']
    for run in report['training']:
        objective, lr = run['training_reward_config'], run['learning_rate_audit']
        lines.append(f"| {run['name']} | {objective['failure_penalty']} / "
            f"{objective['complete_resolved_and_environment_config_verified']} | "
            f"{lr['actual_learning_rate']} / {lr['initial_final_learning_rate_verified']} | {lr['basis']} |")
    lines += ['', '**不同失败惩罚下的训练raw return不可直接比较**，即使reward_scale相同。'
        '触觉/无触觉匹配检查同时要求完整运行奖励配置、实际LR及保存证据一致。'
        '原始训练回报不重写、不按另一惩罚重算；可靠性、载荷与接触指标仍按原定义报告。', '']
    ab = report['development_revisions'].get('revision_reliability_vs_step_size.json')
    if ab:
        lines += ['**可靠性权重与优化步长A/B**：登记状态为 `' + str(ab.get('status', 'unrecorded')) + '`。'
            'A `penalty200_scaled_fullmc_*`相对scaled参考仅将训练failure_penalty从75提高到200，LR仍3e-4；'
            'B `lr1e4_scaled_fullmc_*`仅将LR降至1e-4，P仍75。两组均fresh初始化，scale=.01、shared、eps=1e-5、预算266432。'
            'A与B彼此相差两个因素，不把它们的直接差异称作单项消融；分别与原scaled参考比较。'
            '新冻结评估统一使用原config.json的P75，因此eval return保持同一目标。'
            '该终止系数不改变物理、actor输入或成功/载荷定义；训练raw return跨A/B不可横比。', '']
    eps_candidate = report['development_revisions'].get('revision_separate_adam_eps.json')
    if eps_candidate:
        lines += ['', '**Adam epsilon条件候选**：`revision_separate_adam_eps.json` 登记状态为 `'
            + str(eps_candidate.get('status', 'unrecorded')) + '`。'
            '登记方案仅将分离编码器配置的Adam eps从1e-5改为1e-8，要求先保存并审核前一轮冻结评估；'
            '只有前一轮未达到现有验收条件时才决定是否启动。候选登记不代表已有该配置的训练或成功证据。'
            '实际run及其eps以上表和保存状态为准；CLI省略参数保留原生行为，resume保留保存的eps并拒绝显式冲突。', '']
    lines += ['', '**训练奖励缩放按保存模型核对**：只读checkpoint ZIP的JSON `training_reward_scale`。'
        '旧checkpoint缺该属性时按历史未缩放行为解析为1，并与显式metadata核对；'
        '初始、最终及触觉/无触觉配对的实际scale须一致。无模型的run仅显示请求或metadata，保持未核验。', '',
        '| 运行 | 请求reward scale（含CLI默认；是否显式） | 保存的实际scale | 已有checkpoint核验 | 初始与最终共同核验 |',
        '| --- | --- | --- | --- | --- |']
    for run in report['training']:
        scaling = run['reward_scaling']
        lines.append(f"| {run['name']} | "
            f"{scaling['reward_scale_requested'] if scaling['reward_scale_requested'] is not None else '未指定'}"
            f"（{scaling['reward_scale_request_explicit']}） | "
            f"{scaling['training_reward_scale'] if scaling['training_reward_scale'] is not None else '未核验'} | "
            f"{scaling['saved_reward_scale_verified']}（{scaling['basis']}） | "
            f"{scaling['initial_final_reward_scale_verified']} |")
    lines += ['', '`episodes.jsonl`的return、reward_components及`rollout/ep_rew_mean`保持原始环境奖励单位；'
        'PPO rollout buffer中的reward/return与`train/value_loss`使用缩放后的训练单位。'
        '**不能因value_loss数值下降认定策略变好**：跨scale的MSE量纲不同，且价值预测和优化轨迹也会改变。'
        '冻结评估继续使用原始奖励、task/strict成功、物理载荷与接触判据；结果表和有效性判定不依据value_loss。', '']
    scale_candidate = report['development_revisions'].get('revision_reward_scale_fullmc.json')
    if scale_candidate:
        lines += ['**训练reward缩放候选**：`revision_reward_scale_fullmc.json` 状态为 `'
            + str(scale_candidate.get('status', 'unrecorded')) + '`。'
            '比较参考为旧shared fullMC，仅将PPO训练reward乘0.01；seed0、266432 transitions、'
            'Adam eps=1e-5、共享编码器及初始化均保留旧shared配置。'
            '不能把它与eps8分离编码器的跨变体差异解释成单独reward scale效果。'
            '候选声明不等于已启动或通过验收，实际执行和保存scale以上表为准。', '']
    lines += ['',
              '| 运行 | 最初32条平均载荷/增益 | 最后32条平均载荷/增益 | 最后32条task/strict |', '| --- | ---: | ---: | ---: |']
    for run in report['training']:
        first, last = run['first_32'], run['last_32']
        lines.append(f"| {run['name']} | {fmt(first['mean_load_N'])} N / {fmt(first['mean_gain'])} | "
                     f"{fmt(last['mean_load_N'])} N / {fmt(last['mean_gain'])} | {last['task_successes']}/{last['episodes']} / {last['strict_successes']}/{last['episodes']} |")
    lines += ['', '各训练条件的计数、平均载荷和最长接触中断保存在JSON的 `per_material_training`，'
              '同材料最近至多32条保存在 `per_material_last_32`；降低平均力不能掩盖最难条件失败。', '',
              '**200 g、μ=0.2训练条件审计（探索回合，不是冻结验证）**', '',
              '| 运行 | 最近回合数 | task / strict成功 | 平均载荷(N) / 增益 | 最低保持高度(m) |',
              '| --- | ---: | --- | --- | ---: |']
    for run in report['training']:
        hard = next((item for item in run['per_material_last_32']
            if abs(item['mass_kg'] - .2) < 1e-5 and abs(item['friction'] - .2) < 1e-5), None)
        if hard:
            lines.append(f"| {run['name']} | {hard['episodes']} | {hard['task_successes']} / {hard['strict_successes']} | "
                f"{fmt(hard['mean_load_N'])} / {fmt(hard['mean_gain'])} | {fmt(hard['min_hold_clearance_m'], 5)} |")
    lines += ['', '### 本体历史与比较条件', '']
    if not report['training_protocol_comparisons']:
        lines += ['尚无可核对的同seed触觉/无触觉训练对。']
    for pair in report['training_protocol_comparisons']:
        lines += [f"- `{pair['tactile_run']}` / `{pair['no_tactile_run']}`（{pair['comparison_type']}）：协议检查全部一致={pair['all_protocol_checks_match']}；"
                  '逐项：`' + json.dumps(pair['protocol_match_checks'], ensure_ascii=False) + '`。']
        applicability = pair.get('applicability', {})
        if applicability.get('not_applicable_as_fresh_initialization_checks'):
            lines += ['  适用范围：触觉/无触觉各自恢复独立训练的阶段1父模型；'
                '`' + '、'.join(applicability['not_applicable_as_fresh_initialization_checks'])
                + '`保留原始值，但fresh初始化/等权噪声假设不适用，不能仅用其false判定续训不匹配。'
                '只有另行声明P75/P200分支时，才按同消融组父来源检查惩罚对照；仅加预算的续训没有惩罚对照。'
                '其余false项仍为：`' + json.dumps(applicability['raw_false_other_checks']) + '`。']
    contrasts = report.get('continuation_parent_contrasts', [])
    if contrasts:
        lines += ['', '**阶段2惩罚对照按同一触觉消融组核对父来源**（不是触觉与无触觉初始权重相等检查）：', '']
        for contrast in contrasts:
            lines.append('- ' + ('触觉' if contrast['uses_tactile'] else '无触觉')
                + f"，父run `{contrast['parent_run']}`，来源状态 `{contrast['parent_identity_status']}`："
                + '；'.join(f"`{b['run']}` P={b['failure_penalty']}" for b in contrast['branches']) + '。')
        lines += ['', '此处只复核已记录父checkpoint身份，恢复张量/优化器逐项相等证据另见预声明所引用的startup audit。'
            '父SHA冲突明确记为mismatch，缺证据记为unverified；不会用“不适用fresh检查”掩盖父来源冲突。', '']
    lines += ['', '这里只核验相同60 Hz、12帧本体历史以及材料、完整奖励配置、实际LR、源代码、噪声方式、CPU构造、编码器共享拓扑、初始化方法、实际Adam group eps、保存的训练reward scale、γ/GAE和预算；不是性能优劣结论。'
              '未完成运行的实际完成预算检查为false；旧metadata缺少设备或GAE记录也记为未核验，不等于已经证明配置不同。', '',
              '## 冻结模型与基线评估', '']
    feedback = report['continued_feedback_evidence']
    if feedback['seen_material_suites_where_freezing_reduces_task_success']:
        lines += ['**持续反馈证据分范围解释**：在与训练材料集合一致的 `'
            + '`、`'.join(feedback['seen_material_suites_where_freezing_reduces_task_success'])
            + '` 中，冻结探测后的触觉会降低task完成数，说明这些策略对后续输入变化敏感。'
            '是否优于探测后保持已选增益，仍需查看独立的gain_hold对照；冻结输入也可能形成与本体状态不一致的分布。', '']
        if feedback['same_checkpoint_validation_suites_where_freezing_is_no_worse']:
            lines += ['同时，新验证材料的 `'
                + '`、`'.join(feedback['same_checkpoint_validation_suites_where_freezing_is_no_worse'])
                + '` 使用同一冻结checkpoint，冻结触觉的成功数与平均载荷均不劣。**持续反馈的泛化收益仍未证明**；'
                '保留这两组结果，不能用其中一组覆盖另一组，也不能再将全部证据概括为仅早期触觉适应。', '']
    if not report['evaluations']:
        lines += ['尚无评估结果，不由训练曲线推断验证或测试表现。', '']
    for suite in report['evaluations']:
        lines += [f"### {suite['name']}（{suite['split']}）", '',
            f"执行状态：{suite['execution_status']}；reset验收：{suite['reset']['status']}；与冻结配置一致：{suite['config_matches_frozen']}。",
            f"本组评估实际failure_penalty={suite['config'].get('failure_penalty', '未记录')}，eval return依据本组配置，不继承训练run的惩罚系数。",
            f"{len(suite['conditions_kg_mu'])}个lane，{suite['unique_material_conditions']}种质量/摩擦条件，每方法每条件一次。",
            '模型checkpoint路径和SHA256均保留在JSON；置零/错配是同一冻结策略的诊断，不代替无触觉重训练。', '',
            '| 方法 | task | strict | 均值载荷(N) | 保持双指比例均值/最小 | 最长无双指支持(s) | 回报 |',
            '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
        for row in suite['rows']:
            lines.append(f"| {row['controller']} | {row['task_successes']}/{row['task_label_count']} | {row['strict_successes']}/{row['strict_label_count']} | {fmt(row['mean_load_N'])} | {fmt(row['mean_contact_fraction'], 4)} / {fmt(row['min_contact_fraction'], 4)} | {fmt(row['max_unsupported_hold_s'], 4)} | {fmt(row['mean_return'])} |")
        if not suite['rows']:
            lines += ['| 待产出 | — | — | — | — | — | — |']
        if suite['rows']:
            lines += ['', '**采样峰值单独报告**：先取每条lane的episode最大接触载荷，再分别求均值和全局最大值。'
                '这与上表的时间平均载荷不同；当前来自60 Hz记录，不能当作采样间隔内的真实瞬时最大力。', '',
                '| 方法 | 有峰值记录的lane数 | 每lane峰值的均值(N) | 所有lane中的最大峰值(N) |',
                '| --- | ---: | ---: | ---: |']
            for row in suite['rows']:
                lines.append(f"| {row['controller']} | {row['peak_label_count']} | {fmt(row['mean_per_lane_peak_load_N'])} | {fmt(row['max_peak_all_lanes_N'])} |")
        lines += ['', 'reset记录：`' + json.dumps(suite['reset']['recorded_max_differences'], ensure_ascii=False) + '`。', '']
        if not suite['reset']['matched_comparison_validated']:
            lines += ['**尚未通过完整reset复核；本组目前不作为已验证的配对性能证据。**', '']
        paired = suite['paired_success_load_comparisons']
        if paired:
            lines += ['**按材料配对的共同成功载荷比较**：下表同时保留全条件task/strict成功数差；'
                '载荷只在双方task均成功的同一材料子集计算，不能替代上表全条件结果或原验收门槛。'
                '材料按质量/摩擦匹配，不按记录顺序；缺失、不一致或重复导致配对不唯一时拒绝比较。'
                '无共同成功时载荷统计为None；对照均值为0时降低比例为None。', '',
                '| 对照 | 全条件Δtask / Δstrict（触觉−对照） | 共同task成功 / 全条件 | 共同条件载荷：触觉 / 对照(N) | 配对Δ载荷(N) | 载荷降低比例 |',
                '| --- | ---: | ---: | ---: | ---: | ---: |']
            for comparison in paired:
                name = comparison['baseline_controller']
                if not comparison['comparison_valid']:
                    lines.append(f"| {name} | 拒绝：{comparison['rejection_reason']} | — | — | — | — |")
                    continue
                left = comparison['tactile_mean_load_N_on_common_task_success']
                right = comparison['baseline_mean_load_N_on_common_task_success']
                delta = comparison['paired_mean_load_delta_N_on_common_task_success']
                reduction = comparison['paired_mean_load_reduction_fraction_on_common_task_success']
                load_text = f'{left:.3f} / {right:.3f}' if left is not None else '—'
                delta_text = f'{delta:+.3f}' if delta is not None else '—'
                reduction_text = f'{reduction:+.2%}' if reduction is not None else '—'
                lines.append(f"| {name} | {comparison['task_success_count_delta']:+d} / "
                    f"{comparison['strict_success_count_delta']:+d} | "
                    f"{comparison['common_task_success_lane_count']} / {comparison['conditions']} | "
                    f"{load_text} | {delta_text} | {reduction_text} |")
            lines += ['', '配对载荷差是共同条件内`mean(触觉−对照)`；降低比例是'
                '`1−共同条件触觉均值/共同条件对照均值`，不是逐lane比例的平均。'
                '共同task成功仍可能包含strict失败，因此须同时检查全条件strict差和接触约束。', '']
        touch = next((row for row in suite['rows'] if row['controller'] == 'tactile'), None)
        success_gate = p.get('proposed_engineering_gate', {}).get('heldout_task_success_fraction_min')
        if (touch and touch['task_label_count'] and success_gate is not None
                and touch['task_successes'] / touch['task_label_count'] < success_gate):
            lines += [f"**当前条件组的成功率不足**：触觉task为{touch['task_successes']}/{touch['task_label_count']}，"
                f"低于预声明成功比例{success_gate:.0%}（此处仅作开发/验证诊断，不替代最终测试）。"
                f"{fmt(touch['mean_load_N'])} N均值包含未抓住或掉落回合，不能解释为成功抓握下的省力；"
                'value_loss或训练奖励单位变化也不能改变这一结论。', '']
        comparisons = [row for row in suite['rows'] if row['controller'] in {'tactile', 'no_tactile'}
                       and row['load_change_vs_fixed_2_fraction'] is not None]
        if comparisons:
            lines += ['相对本组固定g=2的全episode平均载荷变化：' + '；'.join(
                f"`{row['controller']}` {row['load_change_vs_fixed_2_fraction']:+.2%}" for row in comparisons)
                + '。只有相同成功和接触要求下才能解释为节力；不能仅以更低均值忽略失败。', '']
            touch = next((row for row in comparisons if row['controller'] == 'tactile'), None)
            reduction_gate = p.get('proposed_engineering_gate', {}).get(
                'mean_load_reduction_vs_validation_selected_global_fixed_gain_min')
            if touch and reduction_gate is not None and 0 < -touch['load_change_vs_fixed_2_fraction'] < reduction_gate:
                lines += [f"本组触觉策略节力 {-touch['load_change_vs_fixed_2_fraction']:.2%}，"
                    f"低于预声明的 {reduction_gate:.0%} 节力门槛；当前{suite['split']}结果不能替代最终留出验收。", '']
        diagnostic = suite['tactile_feedback_diagnostics']
        if diagnostic['frozen_tactile_after_probe_fewer_task_successes']:
            controllers = {row['controller']: row for row in suite['rows']}
            normal, frozen = controllers['tactile'], controllers['tactile_frozen_after_probe']
            scope = '与训练相同的已见材料集合' if suite['conditions_match_training_run_names'] else f"本组{suite['split']}材料"
            lines += [f"**持续触觉的组内干预证据**：在{scope}中，冻结探测后的触觉使task "
                f"{normal['task_successes']}/{normal['task_label_count']} → {frozen['task_successes']}/{frozen['task_label_count']}，"
                f"strict {normal['strict_successes']}/{normal['strict_label_count']} → {frozen['strict_successes']}/{frozen['strict_label_count']}。"
                '冻结后较低的平均载荷包含失败回合，不能视为相同成功质量下的节力。'
                '这说明当前策略对新触觉输入敏感；冻结输入可能与真实本体状态不一致，不能单独证明持续反馈优于早期识别后保持增益。', '']
        if diagnostic['tactile_more_task_successes_than_blind']:
            lines += ['触觉重训练策略在本组完成次数多于无触觉重训练；需结合训练配置配对检查，'
                      '可作为当前单seed下利用触觉信息的初步证据，不能直接等同于持续高频反馈的收益。', '']
        if diagnostic['frozen_tactile_after_probe_no_worse_success_and_load']:
            lines += ['**尚未证明持续触觉更新受益**：探测后冻结触觉输入的诊断保持不少于原策略的task/strict成功，'
                      '且全episode平均载荷不更高。当前证据与主要利用早期触觉适应相容；'
                      '冻结诊断仍保留本体历史和策略更新，并非完整开环控制。', '']
        if diagnostic['gain_hold_after_probe_no_worse_success']:
            controllers = {row['controller']: row for row in suite['rows']}
            normal, held = controllers['tactile'], controllers['tactile_gain_hold_after_probe']
            change = diagnostic['tactile_load_reduction_vs_gain_hold_fraction']
            lines += [f"**探测后保持增益的对照边界**：gain_hold的task/strict为 "
                f"{held['task_successes']}/{held['task_label_count']}、{held['strict_successes']}/{held['strict_label_count']}，"
                f"不低于动态策略的 {normal['task_successes']}/{normal['task_label_count']}、"
                f"{normal['strict_successes']}/{normal['strict_label_count']}。平均载荷分别 "
                f"{fmt(held['mean_load_N'])} / {fmt(normal['mean_load_N'])} N；动态相对gain_hold的载荷降低 "
                f"{change:.2%}。" if change is not None else '探测后保持增益的成功数不低于动态策略。',
                '因此不能只用冻结触觉输入失败来宣称持续反馈必需；须同时保留gain_hold结果和接触/峰值约束。', '']
        failures = [(row, episode) for row in suite['rows'] for episode in row['task_failures'] + row['strict_only_failures']]
        if failures:
            lines += ['失败条件（每条件一次，保留task与strict差别）：', '',
                '| 方法 | 质量(g) / μ | task / strict | 最低保持高度(m) | 保持双指比例 |',
                '| --- | --- | --- | ---: | ---: |']
            for row, episode in failures:
                lines.append(f"| {row['controller']} | {episode['mass_kg'] * 1000:.2f} / {episode['friction']:.3f} | "
                    f"{episode.get('task_success')} / {episode.get('strict_success')} | {fmt(episode.get('hold_min_clearance_m'), 5)} | "
                    f"{fmt(episode.get('hold_contact_fraction'), 4)} |")
            lines.append('')
    coverage = report.get('contact_coverage_audit')
    if coverage:
        interpretation = coverage['interpretation']
        peak = interpretation.get('peak_load', {})
        phases = interpretation.get('phases', {})
        lines += ['## 接触覆盖与峰值专项审计', '',
            '本审计只针对归一化262k版本的固定g=2与触觉策略验证轨迹，不自动适用于其它材料或后续模型。'
            f"审计状态：{(coverage.get('status') or {}).get('status', 'pending')}；与原轨迹已记录通道逐点差全0："
            f"{coverage['reference_replay_all_recorded_differences_zero']}。", '',
            '按60 Hz采样核对全手和机械臂接触后，两个控制器未计入的手部接触载荷均为0，'
            '实际带载接触仅出现在拇指和食指对应指节。因此该验证组的平均节力不是把接触转移到了漏计link；'
            '不覆盖采样间隔内的接触、其它姿态或未测材料，也不等于实机法向力标定。', '',
            '| 控制器 | 全episode平均载荷(N) | 每lane采样峰值均值(N) | 全部lane最大采样峰值(N) |',
            '| --- | ---: | ---: | ---: |']
        whole = phases.get('entire_control_window', {})
        for name in ('fixed_2', 'tactile'):
            row = peak.get(name, {})
            lines.append(f"| {name} | {fmt(whole.get(name + '_mean_load_N'))} | {fmt(row.get('mean_peak_per_lane_N'))} | {fmt(row.get('max_peak_all_lanes_N'))} |")
        lines += ['', '**平均节力不等于峰值降低。** 此组平均载荷降低约5.64%，但最大采样峰值由约24.00 N升至26.69 N，'
            '每lane峰值均值也从约21.95 N升至24.16 N。16条lane中10条平均载荷降低、6条升高；主要下降发生在抬升阶段，'
            '预紧和试抬的平均载荷反而增加。', '',
            '| 阶段 | 固定g=2平均载荷(N) | 触觉策略平均载荷(N) |', '| --- | ---: | ---: |']
        for name in ('preload', 'test_lift', 'lift', 'hold'):
            row = phases.get(name, {})
            lines.append(f"| {name} | {fmt(row.get('fixed_2_mean_load_N'))} | {fmt(row.get('tactile_mean_load_N'))} |")
        lines += ['', f"来源：[覆盖审计解释]({Path(report['root']) / coverage['source'] / 'interpretation.json'})；"
            '两个controller的summary、原轨迹SHA256、比较记录和审计manifest均已纳入报告数据源哈希。', '']
    for audit in report.get('contact_coverage_audits', [])[1:]:
        comparison = (audit.get('comparison') or {}).get('mean_all_hand_load_N', {})
        lines += [f"### 后续模型独立覆盖审计：{audit['source']}", '',
            f"状态：{(audit.get('status') or {}).get('status', 'pending')}；与原评估轨迹核对通道逐点差全0："
            f"{audit['reference_replay_all_recorded_differences_zero']}。本记录针对其manifest绑定的checkpoint，未沿用旧模型的覆盖结论。", '',
            '| 控制器 | 全手平均载荷(N) | 最大漏计手部载荷(N) | 漏计活跃link | 每lane峰值均值(N) | 全局峰值(N) |',
            '| --- | ---: | ---: | --- | ---: | ---: |']
        for name, values in audit['controllers'].items():
            peaks = values.get('peak_hand_load_per_lane_N') or []
            lines.append(f"| {name} | {fmt(comparison.get(name))} | {fmt(values.get('max_omitted_hand_load_N'))} | "
                f"{values.get('omitted_active_links')} | {fmt(mean(peaks))} | {fmt(extreme(peaks, max))} |")
        if comparison.get('relative_reduction') is not None:
            lines += ['', f"全手平均载荷降低 {comparison['relative_reduction']:.2%}。"
                '覆盖结论限于这些60 Hz记录，不证明采样间隔内没有其它接触或最大瞬时力更低。', '']
    review = report.get('fullmc_independent_review')
    if review:
        lines += ['## 全episode MC结果的独立复核', '',
            'MC的16种验证材料中，触觉策略task/strict均16/16，保持双指比例均为1、最长中断为0，'
            '最低保持高度69.96 mm；相对固定g=2，每条验证lane的平均载荷均降低，总体降低23.28%。'
            '这是单seed、当前验证集合上的正向结果，不是完整范围可靠性的验收。', '',
            '探测后保持增益也16/16 strict，平均13.209 N；动态策略12.962 N仅再降低1.87%。'
            '开发32条件中，动态策略27 task / 26 strict，无触觉28/28，gain_hold29/27；'
            '所以不能把冻结触觉干预退化独立解释为持续更新优于早期探测后的固定控制。', '',
            '开发五条task失败均先升过5 cm后真实掉落：150 g/μ=.2，200 g/μ=.2、.4、.8，以及137.295 g/μ=.2018。'
            '181.423 g/μ=.6819另有strict失败：保持期无双指支持41.67 ms，任务区间最低54.86 mm，'
            '任务截止后31.9 s再跌破5 cm。详细时刻及原指标重算见独立复核JSON。', '',
            '验证集全局采样峰值24.00→26.03 N，开发最难lane出现60.25 N峰值；平均节力不等于峰值更安全。'
            '本验证集已在多轮开发中反复查看，不是未触碰的最终测试；该历史阶段尚无独立seed和留出验收证据。'
            '后续验收状态以本报告开头的显式审核记录为准，不改变这次历史实验的局限。', '',
            f"来源：[MC独立复核]({Path(report['root']) / 'diagnostics/fullmc_evaluation_independent_review.json'})。", '']
    lines += ['## 解释边界与验收范围', '',
        '- 载荷是物理接触力模长之和，全局部episode取均值；不是标定指腹法向夹持力，也不是taxel求和。',
        '- 每lane峰值均值和所有lane最大峰值均按60 Hz样本计算；降低时间平均载荷并不意味着峰值也降低。',
        '- 接触比例和最长中断来自60 Hz采样区间统计，不是连续时间的精确测量；task成功不能隐去strict失败。',
        '- 当前仅是固定几何和名义轨迹上的增益适应；没有检验手形探索或重抓。',
        '- 当前KinematicTaxel剪切含仿真相对速度代理，触觉未做实机标定；仿真效果不能直接宣称真实XHand可迁移。',
        '- 验证集可用于选择全局固定增益/冻结checkpoint，留出测试只能在选择冻结后评估。报告列出全部运行，不替你挑选最好结果。', '',
        (f"**当前结论：{report['effectiveness_status']}；{report['goal_acceptance_status']}。** "
         '仅限已审核的三个最终模型、固定几何/轨迹及指定仿真材料集合；全部历史结果和上述限制继续保留。'
         if accepted else '**有效性和goal验收继续pending。** 必须结合预声明门槛、足够独立seed、冻结评估、载荷与接触质量，以及匹配无触觉对照的显式审核记录作出结论。'), '',
        f"![材料扫描、训练与评估]({Path(report['root']) / 'overview.png'})", '',
        '## 数据来源与读取状态', '',
        '每个实际读取的JSON/JSONL文件均记录SHA256与字节数，见 `adaptive_report.json/source_files`。'
        '训练和评估manifest中的代码、条件及checkpoint哈希随记录保留；仅以CPU weights_only读取保存权重验证共享存储、读取优化器param_groups核对eps，'
        '不构造策略、不预测动作、不启动物理。', '']
    lines += [f'- 待产出：`{name}`' for name in report['pending_files']]
    lines += [f'- 读取告警：{warning}' for warning in report['warnings']]
    if not report['pending_files'] and not report['warnings']:
        lines.append('当前发现的运行未发现必需数据缺失或读取告警；读取完整性本身不构成有效性验收。')
    lines += ['', '刷新（默认不读取留出结果）：', '', '```bash',
              '.venv/bin/python -m ur7e_xhand_sim.residual_rl.adaptive_report --root ' + report['root'], '```', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--include-heldout', action='store_true', help='Explicitly include frozen held-out result suites; never selects a best model')
    args = parser.parse_args()
    root = args.root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    report = collect(root, args.include_heldout)
    plot_report(root, report)
    for path, content in [(root / 'adaptive_report.json', json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n'),
                          (root / 'ADAPTIVE_REPORT.zh-CN.md', markdown(report))]:
        tmp = path.with_suffix(path.suffix + '.tmp')
        tmp.write_text(content)
        tmp.replace(path)
    print(json.dumps({'report': str(root / 'ADAPTIVE_REPORT.zh-CN.md'),
        'effectiveness': report['effectiveness_status'], 'goal_acceptance': report['goal_acceptance_status'],
        'training_runs': len(report['training']), 'evaluation_suites': len(report['evaluations']),
        'heldout_included': report['include_heldout'], 'warnings': report['warnings']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
