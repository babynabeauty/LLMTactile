"""Versioned actor task constraints, separate from the unchanged timing protocol."""
import json
import zipfile
from pathlib import Path

import numpy as np

LEVEL = 'astra_level_one_hot_v1'
FORCE_LIMIT = 'force_limit_N_v1'
FORCE_SCALE_N = 10.
# Four joint-state groups, gain, five phase flags, and elapsed time.
CONDITION_START = 18*4+1+5+1


def conditioning_size(mode):
    if mode == LEVEL:
        return 4
    if mode == FORCE_LIMIT:
        return 2
    raise ValueError(f'Unknown actor_conditioning: {mode}')


def condition_features(mode, astra_level, force_limit_N=None, enabled=True):
    """Finite limit / 10 N plus an unlimited flag; never feed inf into the actor."""
    conditioning_size(mode)
    if mode == LEVEL:
        levels = np.asarray(astra_level)
        if not np.isin(levels, [1, 2, 3, 4]).all():
            raise ValueError('astra_level must be in 1..4')
        values = np.eye(4, dtype=np.float32)[levels.astype(int)-1]
    else:
        if force_limit_N is None:
            raise ValueError('Numeric conditioning needs explicit force_limit_N; use +inf for unlimited')
        force = np.asarray(force_limit_N, dtype=np.float64)
        if np.any(np.isnan(force) | (force <= 0)):
            raise ValueError('force_limit_N must be positive finite N or +inf')
        unlimited = np.isposinf(force)
        values = np.stack([np.where(unlimited, 0., force/FORCE_SCALE_N), unlimited], axis=-1)
        if np.any(np.abs(values) > np.finfo(np.float32).max):
            raise ValueError('force_limit_N is outside the actor float32 range')
    return (values*bool(enabled)).astype(np.float32)


def conditioning_protocol(mode):
    return dict(version=mode, size=conditioning_size(mode),
        fields=(['astra_level_one_hot[4]'] if mode == LEVEL else
                ['force_limit_N_div_10[1]', 'force_unlimited[1]']),
        source=('Astra level' if mode == LEVEL else 'supplied task force limit; matches F_break in this experiment'))


def require_checkpoint_conditioning(checkpoint, mode):
    conditioning_size(mode)
    if checkpoint is None:
        return
    checkpoint = Path(checkpoint)
    if not checkpoint.is_file() and checkpoint.suffix != '.zip':
        checkpoint = checkpoint.with_suffix('.zip')
    with zipfile.ZipFile(checkpoint) as archive:
        saved = json.loads(archive.read('data')).get('actor_conditioning', LEVEL)
    if saved != mode:
        raise ValueError(f'Checkpoint actor_conditioning {saved} != {mode}; explicitly migrate its input layers first')
