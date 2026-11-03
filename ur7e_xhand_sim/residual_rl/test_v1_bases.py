import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from .v1_bases import SCHEMA, experiment_action_joints, load_experiment, load_library, sha256

PHASE_SECONDS = (('settle', .5), ('approach', 1.), ('align', .5), ('close', .5), ('preload', .25),
                 ('test_lift', .5), ('lift', 1.), ('hold', 3.))


def write_trajectory(path, phases=PHASE_SECONDS, dt=1/60):
    rows, t = [], 0.
    for phase, seconds in phases:
        for _ in range(int(round(seconds/dt))):
            rows.append({'before': {'sim_time_s': t}, 'after': {'sim_time_s': t+dt},
                         'sent_target_rad': [0.]*18, 'phase': phase, 'grip_gain': 1.})
            t += dt
    path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    return path


def write_library(folder, trajectory, label='S', split='train', **extra):
    library = dict(schema=SCHEMA, object_id='cuboid', object_spec={'shape': 'cuboid'}, hand_shape='pinch',
        bases=[dict(base_id='b0', trajectory_path=trajectory.name, trajectory_sha256=sha256(trajectory),
            object_initial_position_m=[-.6, -.1333, .2225], label=label, split=split, **extra)])
    path = folder/'library.json'
    path.write_text(json.dumps(library))
    return path


class BaseLibraryTest(unittest.TestCase):
    def setUp(self):
        self.folder = Path(tempfile.mkdtemp())

    def test_takeover_end_and_hold(self):
        trajectory = write_trajectory(self.folder/'control.jsonl')
        _, (base,) = load_library(write_library(self.folder, trajectory))
        self.assertEqual(base.takeover_tick, 600)            # close starts at 2.0 s
        self.assertEqual(base.end_tick, int(np.ceil(7.25*15))*20)
        self.assertAlmostEqual(base.hold_start_s, 4.25)
        np.testing.assert_allclose(base.object_quaternion, [1, 0, 0, 0])

    def test_requires_close_and_a_final_three_second_hold(self):
        no_close = [p for p in PHASE_SECONDS if p[0] != 'close']
        trajectory = write_trajectory(self.folder/'control.jsonl', no_close)
        with self.assertRaisesRegex(ValueError, 'missing phases'):
            load_library(write_library(self.folder, trajectory))
        short_hold = PHASE_SECONDS[:-1]+(('hold', 2.),)
        trajectory = write_trajectory(self.folder/'control.jsonl', short_hold)
        with self.assertRaisesRegex(ValueError, 'final 3 s'):
            load_library(write_library(self.folder, trajectory))

    def test_unrepairable_bases_never_train(self):
        trajectory = write_trajectory(self.folder/'control.jsonl')
        with self.assertRaisesRegex(ValueError, 'evaluation-only'):
            load_library(write_library(self.folder, trajectory, label='U'))
        _, bases = load_library(write_library(self.folder, trajectory, label='U', split='test'))
        self.assertEqual(bases[0].label, 'U')

    def test_hash_is_checked(self):
        trajectory = write_trajectory(self.folder/'control.jsonl')
        library = write_library(self.folder, trajectory)
        trajectory.write_text(trajectory.read_text()+'\n')
        with self.assertRaisesRegex(ValueError, 'hash'):
            load_library(library)

    def test_hand_shapes_set_the_shared_action(self):
        trajectory = write_trajectory(self.folder/'control.jsonl')
        pinch = load_library(write_library(self.folder, trajectory))
        tripod = (dict(pinch[0], hand_shape='tripod'), pinch[1])
        self.assertEqual(experiment_action_joints({}, [pinch]), (0, 1, 2, 3, 4, 5))
        self.assertEqual(experiment_action_joints({}, [pinch, tripod]), tuple(range(8)))
        self.assertEqual(experiment_action_joints({'hand_shapes': ['pinch', 'wrap']}, [pinch]), tuple(range(12)))
        with self.assertRaisesRegex(ValueError, 'omits'):
            experiment_action_joints({'hand_shapes': ['pinch']}, [pinch, tripod])
        library = json.loads((self.folder/'library.json').read_text())
        library['hand_shape'] = 'fist'
        (self.folder/'library.json').write_text(json.dumps(library))
        with self.assertRaisesRegex(ValueError, 'hand shapes'):
            load_library(self.folder/'library.json')

    def test_experiment_resolves_library_paths(self):
        trajectory = write_trajectory(self.folder/'control.jsonl')
        write_library(self.folder, trajectory)
        experiment = dict(schema='v1_experiment', objects=[dict(object_id='cuboid', base_library='library.json')],
                          stages=[dict(name='smoke', levels=[4], requested_timesteps=64)])
        (self.folder/'experiment.json').write_text(json.dumps(experiment))
        loaded = load_experiment(self.folder/'experiment.json')
        data, (base,) = load_library(loaded['objects'][0]['base_library'])
        self.assertEqual(data['object_id'], 'cuboid')
        np.testing.assert_allclose(base.object_position, [-.6, -.1333, .2225])


if __name__ == '__main__':
    unittest.main()
