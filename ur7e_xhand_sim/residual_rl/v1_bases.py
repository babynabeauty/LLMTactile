"""v1 base libraries: recorded command trajectories that the small model corrects.

A library holds the bases of ONE object, all sharing the demonstration's hand
shape. Labels follow PLAN_v1 section 5: S succeeds without residual, R fails
but is repairable after first contact, U is evaluation-only. CPU-only.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .core import Trajectory
from .control_protocol import V1_RATES, current_control_protocol, migrate_experiment
from .solver_profile import trajectory_solver_profile
from .v1_task import HAND_SHAPES, action_joints, takeover_tick

SCHEMA = 'v1_base_library'
EXPERIMENT_SCHEMA = 'v1_experiment'
LABELS = ('S', 'R', 'U')
TRAINING_LABELS = ('S', 'R')
REQUIRED_PHASES = ('close', 'test_lift', 'lift', 'hold')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def resolve(path, root):
    path = Path(path).expanduser()
    return (path if path.is_absolute() else Path(root)/path).resolve()


class Base:
    """One recorded base: commands, phases, gain schedule and object start pose."""
    def __init__(self, entry, root):
        self.base_id = str(entry['base_id'])
        self.label = entry['label']
        self.split = entry['split']
        if self.label not in LABELS or self.split not in ('train', 'test'):
            raise ValueError(f'{self.base_id}: label must be S/R/U and split train/test')
        if self.split == 'train' and self.label not in TRAINING_LABELS:
            raise ValueError(f'{self.base_id}: U bases are evaluation-only')
        self.path = resolve(entry['trajectory_path'], root)
        if sha256(self.path) != entry['trajectory_sha256']:
            raise ValueError(f'{self.base_id}: trajectory hash mismatch')
        self.trajectory = Trajectory(self.path)
        t = self.trajectory
        if t.targets.shape[1] != 18 or abs(t.times[0]) > 1e-8:
            raise ValueError(f'{self.base_id}: bases need 18 ordered joints starting at time zero')
        missing = [p for p in REQUIRED_PHASES if p not in t.phase_starts]
        if missing:
            raise ValueError(f'{self.base_id}: missing phases {missing}')
        starts = [t.phase_starts[p] for p in REQUIRED_PHASES]
        if np.any(np.diff(starts) <= 0):
            raise ValueError(f'{self.base_id}: close, test_lift, lift and hold must be ordered')
        self.hold_start_s = t.duration_s-3.
        if t.phase_starts['hold'] > self.hold_start_s+1e-8 or np.any(t.phases[t.times >= t.phase_starts['hold']] != 'hold'):
            raise ValueError(f'{self.base_id}: the final 3 s must be one uninterrupted hold')
        self.takeover_tick = takeover_tick(t.phase_starts)
        ticks = t.ticks(V1_RATES.physics_hz)
        if np.any(ticks % V1_RATES.action_stride) or np.any(np.diff(ticks) != V1_RATES.action_stride):
            raise ValueError(f'{self.base_id}: V1 requires one recorded command per 60 Hz tick')
        self.end_tick = int(ticks[-1])
        self.object_position = np.asarray(entry['object_initial_position_m'], dtype=float)
        self.object_quaternion = np.asarray(entry.get('object_initial_quaternion_wxyz', [1, 0, 0, 0]), dtype=float)
        if self.object_position.shape != (3,) or self.object_quaternion.shape != (4,):
            raise ValueError(f'{self.base_id}: object pose needs xyz and wxyz')
        self.object_quaternion /= np.linalg.norm(self.object_quaternion)
        self.solver_profile = trajectory_solver_profile(self.path)[0]
        self.metadata = {k: v for k, v in entry.items() if k not in ('trajectory_path',)}

    def summary(self):
        return dict(base_id=self.base_id, label=self.label, split=self.split,
            trajectory_path=str(self.path), trajectory_sha256=self.trajectory.sha256,
            takeover_time_s=self.takeover_tick/300, end_time_s=self.end_tick/300,
            object_initial_position_m=self.object_position.tolist(),
            object_initial_quaternion_wxyz=self.object_quaternion.tolist())


def load_library(path, *, split=None, labels=None):
    path = Path(path).expanduser().resolve()
    data = json.loads(path.read_text())
    if data.get('schema') != SCHEMA:
        raise ValueError(f'{path}: not a {SCHEMA}')
    if data.get('hand_shape') not in HAND_SHAPES:
        raise ValueError(f'{path}: v1 supports hand shapes {sorted(HAND_SHAPES)}')
    entries = [e for e in data['bases'] if (split is None or e['split'] == split)
               and (labels is None or e['label'] in labels)]
    if len({e['base_id'] for e in data['bases']}) != len(data['bases']):
        raise ValueError(f'{path}: duplicate base_id')
    bases = [Base(e, path.parent) for e in entries]
    if not bases:
        raise ValueError(f'{path}: no bases for split={split} labels={labels}')
    profiles = {json.dumps(b.solver_profile, sort_keys=True) for b in bases}
    if len(profiles) != 1:
        raise ValueError(f'{path}: bases use different solver profiles')
    return data, bases


def load_experiment(path):
    path = Path(path).expanduser().resolve()
    data = json.loads(path.read_text())
    if data.get('schema') != EXPERIMENT_SCHEMA:
        raise ValueError(f'{path}: not a {EXPERIMENT_SCHEMA}')
    ids = [o['object_id'] for o in data['objects']]
    if not ids or len(set(ids)) != len(ids) or not all(str(i).isidentifier() for i in ids):
        raise ValueError('object_id values must be distinct identifiers')
    for obj in data['objects']:
        obj['base_library'] = str(resolve(obj['base_library'], path.parent))
    names = [s['name'] for s in data['stages']]
    if len(set(names)) != len(names):
        raise ValueError('Duplicate stage names')
    return migrate_experiment(data)


def experiment_action_joints(experiment, libraries):
    """Policy action = joint union over the declared (or present) hand shapes."""
    shapes = experiment.get('hand_shapes') or [library['hand_shape'] for library, _ in libraries]
    missing = {library['hand_shape'] for library, _ in libraries}-set(shapes)
    if missing:
        raise ValueError(f'hand_shapes omits {sorted(missing)}')
    return action_joints(shapes)


TRAINING_DEFAULTS = dict(n_steps=512, batch_size=1024, n_epochs=5, learning_rate=3e-4, gamma=1., gae_lambda=.95**.25,
    use_sde=True, normalize_sde_latent=True, sde_sample_freq=120, action_std_init=.12, adam_eps=1e-5,
    seeds=[1, 2, 3], methods=['tactile', 'no_tactile', 'tactile_no_level'], checkpoint_every=262144)
STAGES = [dict(name='stage1', levels=[4], requested_timesteps=2621440),
          dict(name='stage2', levels=[1, 2, 3, 4], requested_timesteps=2621440, resume_previous=True)]


def write_experiment(path, libraries, *, lanes_per_object=4, hand_shapes=None, name='v1'):
    """One scene per library (object x hand shape); stage1 is T1 only, stage2 adds T2 and resumes stage1."""
    objects, shapes = [], set()
    for library in libraries:
        data = json.loads(Path(library).read_text())
        shapes.add(data['hand_shape'])
        objects.append(dict(object_id=f"{data['object_id']}_{data['hand_shape']}",
                            base_library=str(Path(library).resolve())))
    experiment = dict(schema=EXPERIMENT_SCHEMA, name=name, objects=objects, lanes_per_object=lanes_per_object,
        hand_shapes=sorted(hand_shapes or shapes), config={}, sampler={}, joint_friction_Nm=[0., 0.],
        tactile_noise_N=0., training=TRAINING_DEFAULTS, stages=STAGES,
        control_protocol=current_control_protocol())
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(experiment, indent=2))
    return Path(path)


def main(argv=None):
    p = argparse.ArgumentParser(description='Write a v1 experiment from base libraries (e.g. demo bases.json for smoke tests).')
    p.add_argument('--libraries', type=Path, nargs='+', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--lanes-per-object', type=int, default=4)
    args = p.parse_args(argv)
    print(write_experiment(args.output, args.libraries, lanes_per_object=args.lanes_per_object))


if __name__ == '__main__':
    main()
