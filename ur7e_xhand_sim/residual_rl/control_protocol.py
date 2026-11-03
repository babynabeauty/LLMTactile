"""Versioned V1 timing contract, with CPU-only migration and checkpoint checks.

Legacy 15 Hz experiments retain their simulated duration: rollout, minibatch,
exploration hold, checkpoint interval and stage budgets all receive four times
as many samples at 60 Hz. Discount and GAE factors retain their time horizons.
Frozen source files are never rewritten by this migration.
"""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import zipfile
import numpy as np

from .core import RateConfig

V1_RATES = RateConfig(physics_hz=300, tactile_hz=60, action_hz=60)
HISTORY_SAMPLES = 12
VERSION = 'v1_tactile60_control60_v2'
LEGACY_ACTION_HZ = 15
RATE_MULTIPLIER = V1_RATES.action_hz // LEGACY_ACTION_HZ


def current_control_protocol():
    return dict(schema='v1_control_protocol', version=VERSION, **asdict(V1_RATES),
        base_hz=60, base_target_sampling='recorded_command_zero_order_hold_on_integer_ticks',
        history_samples=HISTORY_SAMPLES, history_duration_s=HISTORY_SAMPLES/V1_RATES.tactile_hz,
        state_time_feature='elapsed_since_close_div10_clipped', state_time_scale_s=10.,
        takeover='first_close_on_60_hz_grid; reset_pose_with_zero_velocity',
        residual_update='one_inference_per_new_tactile_sample')


def elapsed_feature(elapsed_s):
    """Online-observable phase clock; no future plan duration enters the actor."""
    return np.clip(np.asarray(elapsed_s)/10., 0., 1.)


def migrate_experiment(experiment):
    """Upgrade unversioned (15 Hz) settings explicitly; reject unknown protocols."""
    result = deepcopy(experiment)
    existing = result.get('control_protocol')
    if existing is not None:
        if existing != current_control_protocol():
            raise ValueError('Experiment control protocol is incompatible with V1 60 Hz')
        return result
    training = result.get('training', {})
    for key in ('n_steps', 'batch_size', 'sde_sample_freq', 'checkpoint_every'):
        if key in training and training[key] > 0:
            training[key] *= RATE_MULTIPLIER
    for key in ('gamma', 'gae_lambda'):
        if key in training:
            training[key] **= 1/RATE_MULTIPLIER
    for stage in result.get('stages', []):
        stage['requested_timesteps'] *= RATE_MULTIPLIER
    config = result.get('config', {})
    # Only explicit legacy overrides need conversion; V1Config supplies new defaults.
    for key in ('residual_rate_rad', 'residual_change_weight'):
        if key in config:
            config[key] /= RATE_MULTIPLIER
    result['control_protocol'] = current_control_protocol()
    result['control_protocol_migration'] = dict(from_action_hz=LEGACY_ACTION_HZ,
        to_action_hz=V1_RATES.action_hz, sample_count_multiplier=RATE_MULTIPLIER,
        source_file_modified=False,
        policy='Preserve simulated rollout, exploration, training and discount time horizons; '
               'preserve residual slew speed and squared-speed penalty per second.')
    return result


def require_checkpoint_control_compatibility(checkpoint):
    """Reject old/unversioned SB3 checkpoints before allocating a simulator."""
    if checkpoint is None:
        return
    path = Path(checkpoint).expanduser().resolve()
    if not path.is_file() and path.suffix != '.zip':
        path = path.with_suffix('.zip')
    with zipfile.ZipFile(path) as archive:
        data = json.loads(archive.read('data'))
    if data.get('control_protocol') != current_control_protocol():
        raise ValueError('Checkpoint control protocol is missing or incompatible with V1 60 Hz; '
                         'legacy 15 Hz checkpoints require retraining, not resume/evaluation.')
    return data
