"""CPU-only tests for external trajectory provenance and evaluation wiring.

No Genesis scene is initialized. The benchmark integration test substitutes a
minimal environment and inference-only policy so an accidental default-trajectory
fallback or training call fails before an expensive physics evaluation.
"""
from contextlib import redirect_stdout
import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from . import adaptive_benchmark as benchmark
from .core import DEMO, Trajectory, interval_overlap


def write_external_fixture(root):
    path = root / 'commands.jsonl'
    starts = [0., .025, .05, .075]
    phases = ['preload', 'test_lift', 'lift', 'hold']
    rows = [dict(before={'sim_time_s': start},
                 after={'sim_time_s': end},
                 sent_target_rad=[.001 * (i + 1)] * 18, phase=phase)
            for i, (start, end, phase) in enumerate(zip(
                starts, starts[1:] + [3.075], phases))]
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    metadata = {
        'trajectory_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'joint_order': benchmark.ARM_JOINTS + benchmark.HAND_JOINTS,
        'scene': {'target_initial_position_m': [-.6, -.1333, .225],
                  'robot_base_position_m': [0, 0, .2], 'table_top_z_m': .2,
                  'cube_size_m': [.045] * 3, 'distractors': []},
        'initial_state': {
            'home_joint_positions_rad': np.r_[
                np.deg2rad([0, -89.95, 89.96, -89.96, -89.95, -44.98]),
                np.zeros(12)].tolist(),
            'home_joint_velocities_rad_s': [0.] * 18},
        'source_pd': {'kp': [4500, 4500, 3500, 2000, 2000, 2000] + [3.] * 12,
                      'kv': [450, 450, 350, 200, 200, 200] + [.08] * 12},
    }
    (root / 'trajectory_metadata.json').write_text(json.dumps(metadata))
    return path, metadata


class TestExternalTrajectory(unittest.TestCase):
    def test_large_translation_selects_matching_cube_position(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            path,metadata=write_external_fixture(root)
            metadata['scene'].update(scenario_name='large',translation_m=[.15,.1,0.],
                target_initial_position_m=[-.45,-.0333,.225])
            meta_path=root/'trajectory_metadata.json'
            meta_path.write_text(json.dumps(metadata))
            _,info=benchmark.freeze_trajectory_input(path,root/'frozen')
            np.testing.assert_allclose(info['cube_initial_position_m'],[-.45,-.0333,.225],rtol=0,atol=1e-12)
            metadata['scene']['target_initial_position_m']=[-.6,-.1333,.225]
            meta_path.write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError,'scene differs'):
                benchmark.freeze_trajectory_input(path,root/'wrong_cube')

    def test_external_commands_are_frozen_and_source_edits_cannot_change_them(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, metadata = write_external_fixture(root)
            frozen, info = benchmark.freeze_trajectory_input(path, root / 'frozen')
            self.assertEqual(info['sha256'], metadata['trajectory_sha256'])
            self.assertEqual(frozen.read_bytes(), path.read_bytes())
            path.write_text('changed after evaluation input was frozen')
            trajectory = Trajectory(frozen)
            np.testing.assert_allclose(trajectory.target(.025), [.002] * 18)
            self.assertEqual(trajectory.sha256, metadata['trajectory_sha256'])

    def test_external_missing_or_incompatible_provenance_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, metadata = write_external_fixture(root)
            meta_path = root / 'trajectory_metadata.json'
            meta_path.unlink()
            with self.assertRaisesRegex(ValueError, 'requires trajectory_metadata'):
                benchmark.freeze_trajectory_input(path, root / 'missing')
            changes = [
                ('hash', lambda m: m.update(trajectory_sha256='0' * 64)),
                ('order', lambda m: m.update(joint_order=m['joint_order'][::-1])),
                ('scene', lambda m: m['scene'].update(target_initial_position_m=[-.57, -.1133, .225])),
                ('clutter', lambda m: m['scene'].update(distractors=['blue_box'])),
                ('home', lambda m: m['initial_state']['home_joint_positions_rad'].__setitem__(0, .1)),
                ('velocity', lambda m: m['initial_state']['home_joint_velocities_rad_s'].__setitem__(0, .1)),
                ('pd', lambda m: m['source_pd']['kp'].__setitem__(0, 9000)),
            ]
            for name, change in changes:
                with self.subTest(name=name):
                    invalid = copy.deepcopy(metadata)
                    change(invalid)
                    meta_path.write_text(json.dumps(invalid))
                    with self.assertRaises(ValueError):
                        benchmark.freeze_trajectory_input(path, root / name)
                    self.assertFalse((root / name).exists())

    def test_default_freeze_preserves_15hz_targets_and_original_hold_interval(self):
        # An independent source-row read guards the prior command-start convention.
        with (DEMO / 'control.jsonl').open() as stream:
            rows = [json.loads(line) for line in stream]
        times = np.asarray([row['before']['sim_time_s'] for row in rows])
        values = np.asarray([row['sent_target_rad'] for row in rows])
        with tempfile.TemporaryDirectory() as directory:
            frozen, _ = benchmark.freeze_trajectory_input(None, Path(directory) / 'frozen')
            trajectory = Trajectory(frozen)
            action_times = np.arange(480) / 15
            expected = np.stack([np.interp(action_times, times, values[:, i])
                                 for i in range(18)], axis=-1)
            np.testing.assert_array_equal(trajectory.target(action_times), expected)
            self.assertEqual(trajectory.phase_starts['hold'], 28.875)
            self.assertEqual(trajectory.duration_s, 31.875)
            sample_times = np.arange(1, 1917) / 60
            weights = interval_overlap(sample_times - 1 / 60, sample_times,
                                       trajectory.phase_starts['hold'], trajectory.duration_s)
            self.assertAlmostEqual(weights.sum(), 3., places=10)

    def test_benchmark_uses_external_copy_and_only_deterministic_inference(self):
        created = []
        policy_calls = []

        class InferenceOnlyPolicy:
            def predict(self, obs, deterministic):
                policy_calls.append(deterministic)
                return np.array([[2.]], dtype=np.float32), None

            def learn(self, *args, **kwargs):
                raise AssertionError('Frozen evaluation must not train')

        class FakeEnv:
            def __init__(self, **kwargs):
                created.append(self)
                self.kwargs = kwargs
                self.trajectory = Trajectory(kwargs['trajectory_path'])
                self.num_envs = 1
                self.start_tick = 0
                self.closed = False

            def reset(self):
                self.tick = 0
                return {'state': np.zeros((1, 12, 79), np.float32),
                        'tactile': np.zeros((1, 12, 5, 120, 3), np.float32)}

            def action_for_gain(self, value):
                return value

            def step(self, action):
                self.last_interval_metrics = {'clearance_m': np.full((4, 1), .07)}
                metrics = {'task_success': True, 'strict_success': True,
                           'mean_contact_force_N': 1., 'return': 1.}
                infos = [{'applied_gains': np.repeat(action[0], 2), 'episode_metrics': metrics}]
                return self.reset(), np.ones(1), np.array([True]), infos

            def close(self):
                self.closed = True

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            external, metadata = write_external_fixture(root)
            metadata['scene'].update(scenario_name='large',translation_m=[.15,.1,0.],
                target_initial_position_m=[-.45,-.0333,.225])
            (root/'trajectory_metadata.json').write_text(json.dumps(metadata))
            conditions = root / 'conditions.json'
            conditions.write_text('[[0.075, 0.3]]')
            config = root / 'config.json'
            config.write_text('{}')
            model = root / 'frozen_model.zip'
            model.write_bytes(b'inference test fixture')
            model_before = model.read_bytes()
            output = root / 'evaluation'
            argv = ['benchmark', '--output', str(output), '--conditions-json', str(conditions),
                    '--config-json', str(config), '--trajectory', str(external),
                    '--models', 'tactile=' + str(model), '--fixed-gains', '2', '3',
                    '--split', 'trajectory_transfer']
            with patch('sys.argv', argv), patch.object(benchmark, 'AdaptiveVecEnv', FakeEnv), \
                    patch.object(benchmark.PPO, 'load', return_value=InferenceOnlyPolicy()), \
                    redirect_stdout(io.StringIO()):
                benchmark.main()
            self.assertEqual(len(created), 1)
            env = created[0]
            self.assertEqual(env.kwargs['mode'], 'eval')
            np.testing.assert_allclose(env.kwargs['cube_initial_position'],[-.45,-.0333,.225],rtol=0,atol=1e-12)
            self.assertEqual(env.kwargs['trajectory_path'], output / 'trajectory_input' / 'control.jsonl')
            self.assertEqual(env.trajectory.sha256, metadata['trajectory_sha256'])
            self.assertTrue(env.closed)
            self.assertEqual(policy_calls, [True])
            self.assertEqual(model.read_bytes(), model_before)
            manifest = json.loads((output / 'manifest.json').read_text())
            self.assertIs(manifest['training'], False)
            self.assertEqual(manifest['requested_interventions'], [])
            self.assertIs(manifest['reactive'], False)
            results = json.loads((output / 'results.json').read_text())
            self.assertEqual([r['controller'] for r in results], ['fixed_2', 'fixed_3', 'tactile', 'reset_check'])
            self.assertEqual(json.loads((output / 'status.json').read_text())['status'], 'completed')

            # Explicit size transfer keeps the same archived commands, adjusts
            # the center to keep the bottom on the table, and executes only
            # requested checkpoints rather than silently adding baselines.
            size_output = root / 'size_evaluation'
            size_argv = ['benchmark', '--output', str(size_output),
                         '--conditions-json', str(conditions), '--config-json', str(config),
                         '--trajectory', str(external), '--models', 'tactile=' + str(model),
                         '--models-only', '--cube-size-m', '.03', '--split', 'size_transfer']
            with patch('sys.argv', size_argv), patch.object(benchmark, 'AdaptiveVecEnv', FakeEnv), \
                    patch.object(benchmark.PPO, 'load', return_value=InferenceOnlyPolicy()), \
                    redirect_stdout(io.StringIO()):
                benchmark.main()
            size_env = created[-1]
            self.assertEqual(size_env.kwargs['cube_size_m'], .03)
            np.testing.assert_allclose(size_env.kwargs['cube_initial_position'], [-.45,-.0333,.215])
            self.assertEqual(size_env.trajectory.sha256, env.trajectory.sha256)
            self.assertTrue(size_env.closed)
            size_results = json.loads((size_output / 'results.json').read_text())
            self.assertEqual([r['controller'] for r in size_results], ['tactile'])
            self.assertEqual(model.read_bytes(), model_before)


if __name__ == '__main__':
    unittest.main()
