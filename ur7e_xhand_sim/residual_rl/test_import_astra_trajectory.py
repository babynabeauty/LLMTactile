"""Pure importer checks: target/time identity and rejection of malformed logs."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from .core import ROOT, Trajectory
from .import_astra_trajectory import (
    ARCHIVED_ENV_SHA256, CONTROL_DT, HOME_DEGREES, JOINT_ORDER, PHASE_ORDER,
    compare_teacher, compare_teacher_by_phase, convert_rows, import_trajectory, sha256,
)


def source_rows():
    phases = list(PHASE_ORDER[:-1]) + ['hold'] * 120
    home = np.r_[np.deg2rad(HOME_DEGREES), np.zeros(12)]
    rows = []
    for number, phase in enumerate(phases, 1):
        target = home.copy()
        if number > 1:
            target[0] += .001 * number
            target[10] = .4
        rows.append({'step': number, 'phase': phase, 'target': target.tolist(),
                     'local_feedback': {},
                     'observation': {'episode': 1, 'episode_step': number,
                                     'sim_time_s': number * CONTROL_DT,
                                     'terminated': False, 'fault': None,
                                     'joint_positions_rad': dict.fromkeys(JOINT_ORDER, 123.),
                                     'joint_velocities_rad_s': dict.fromkeys(JOINT_ORDER, 456.),
                                     'wrist_position_m': [.001 * number, 0., .1],
                                     'tactile': {'never_copy_me': [789.]}}})
    return rows


def write_rows(path, rows):
    Path(path).write_text(''.join(json.dumps(row) + '\n' for row in rows))


class AstraImportContracts(unittest.TestCase):
    def test_target_preserved_and_after_time_shifted_to_interval_start(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'source.jsonl'
            source = source_rows()
            write_rows(path, source)
            result, phases, _ = convert_rows(path, len(source))
            self.assertEqual(result[0]['before']['sim_time_s'], 0.)
            self.assertEqual(result[0]['after']['sim_time_s'], .025)
            self.assertEqual(result[1]['before']['sim_time_s'], .025)
            self.assertEqual(result[-1]['after']['sim_time_s'], len(source) * CONTROL_DT)
            self.assertEqual(phases['hold']['end_s'] - phases['hold']['start_s'], 3.)
            for imported, original in zip(result, source):
                self.assertEqual(imported['sent_target_rad'], original['target'])
                self.assertEqual(set(imported), {'step', 'phase', 'before', 'after', 'sent_target_rad'})
                self.assertNotIn('tactile', json.dumps(imported))
                self.assertNotIn(123., imported['sent_target_rad'])
            converted = Path(directory) / 'control.jsonl'
            write_rows(converted, result)
            trajectory = Trajectory(converted)
            np.testing.assert_array_equal(trajectory.target(0), source[0]['target'])
            np.testing.assert_array_equal(trajectory.target(.025), source[1]['target'])
            np.testing.assert_allclose(trajectory.target(.0125),
                                       (np.array(source[0]['target']) + source[1]['target']) / 2)

    def test_reject_malformed_steps_times_values_and_joint_order(self):
        cases = []
        rows = source_rows(); rows[1]['step'] = 1; cases.append(rows)
        rows = source_rows(); rows[1]['observation']['sim_time_s'] += .025; cases.append(rows)
        rows = source_rows(); rows[1]['observation']['episode_step'] = 1; cases.append(rows)
        rows = source_rows(); rows[1]['observation']['episode'] = 2; cases.append(rows)
        rows = source_rows(); rows[1]['target'] = [0.] * 17; cases.append(rows)
        rows = source_rows(); rows[1]['target'][0] = float('nan'); cases.append(rows)
        rows = source_rows(); rows[1]['target'][0] = True; cases.append(rows)
        rows = source_rows(); rows[1]['phase'] = 'settle'; cases.append(rows)
        rows = source_rows(); rows[0]['target'][0] = .2; cases.append(rows)
        rows = source_rows(); rows[1]['observation']['fault'] = 'collision'; cases.append(rows)
        rows = source_rows(); rows[1]['observation']['joint_positions_rad'] = dict.fromkeys(reversed(JOINT_ORDER), 0.); cases.append(rows)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'control.jsonl'
            for number, rows in enumerate(cases):
                with self.subTest(case=number):
                    write_rows(path, rows)
                    with self.assertRaises(ValueError):
                        convert_rows(path, len(rows))
            write_rows(path, source_rows())
            with self.assertRaisesRegex(ValueError, 'row count'):
                convert_rows(path, len(source_rows()) + 1)

    def test_bundle_provenance_does_not_certify_new_replay_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'original'
            source.mkdir()
            rows = source_rows()
            write_rows(source / 'control.jsonl', rows)
            reference = {'arm_joint_order': JOINT_ORDER[:6], 'hand_joint_order': JOINT_ORDER[6:]}
            (source / 'demonstration_reference.json').write_text(json.dumps(reference))
            manifest = {'high_level_driver': 'vlm_policy_CodexAppServerPolicy',
                        'reference_sha256': sha256(source / 'demonstration_reference.json'),
                        'source_hashes': {'/historical/environment.py': ARCHIVED_ENV_SHA256}}
            (source / 'manifest.json').write_text(json.dumps(manifest))
            scenario = {'name': 'original', 'distractors': [], 'translation_m': [0, 0, 0],
                        'target_initial_position_m': [-.6, -.1333, .225]}
            (source / 'scenario_evaluation_only.json').write_text(json.dumps(scenario))
            (source / 'score.json').write_text(json.dumps({'success': True, 'fault': None, 'control_steps': len(rows)}))
            converted, _, _ = convert_rows(source / 'control.jsonl', len(rows))
            teacher = root / 'teacher.jsonl'
            write_rows(teacher, converted)
            source_before = {p.name: sha256(p) for p in source.iterdir()}
            # The small archived source snapshot is a provenance fixture, not a simulator import.
            suite = ROOT / 'outputs/s8v4_four_tests/demo_context_four_20261024_v1'
            output = root / 'bundle'
            metadata = import_trajectory(source, output, suite, teacher)
            self.assertEqual(metadata['status'], 'converted_not_replay_validated')
            self.assertIsNone(metadata['conversion']['new_replay_success'])
            self.assertFalse(metadata['conversion']['historical_tactile_copied'])
            self.assertFalse((output / 'reference.json').exists())
            self.assertEqual(metadata['trajectory_sha256'], sha256(output / 'control.jsonl'))
            self.assertEqual(metadata['source_pd']['kp'][6:], [3.] * 12)
            self.assertEqual(metadata['scene'], {
                'scenario_name': 'original', 'target_initial_position_m': [-.6, -.1333, .225],
                'translation_m': [0., 0., 0.], 'distractors': [],
                'robot_base_position_m': [0., 0., .2], 'table_top_z_m': .2,
                'cube_size_m': [.045] * 3})
            self.assertTrue(metadata['original_teacher_comparison']['raw_targets_exactly_equal'])
            self.assertEqual(metadata['original_teacher_comparison']['resampled_max_abs_difference_rad'], 0.)
            self.assertTrue(metadata['original_teacher_comparison']['interpretation'].startswith('Near-identical'))
            self.assertNotIn('phase_aligned_teacher_comparison', metadata)
            self.assertEqual(source_before, {p.name: sha256(p) for p in source.iterdir()})
            with self.assertRaises(FileExistsError):
                import_trajectory(source, output, suite, teacher)
            wrong = copy.deepcopy(scenario)
            wrong['name'] = 'small'
            (source / 'scenario_evaluation_only.json').write_text(json.dumps(wrong))
            with self.assertRaisesRegex(ValueError, 'original or large uncluttered'):
                import_trajectory(source, root / 'bad_bundle', suite, teacher)
            self.assertFalse((root / 'bad_bundle').exists())

    def test_large_scene_bundle_and_inferred_suite_preserve_scene_targets_and_times(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 's8v4_genesis' / 'fixture_large'
            source.mkdir(parents=True)
            rows = source_rows()
            write_rows(source / 'control.jsonl', rows)
            reference = {'arm_joint_order': JOINT_ORDER[:6], 'hand_joint_order': JOINT_ORDER[6:]}
            (source / 'demonstration_reference.json').write_text(json.dumps(reference))
            manifest = {'high_level_driver': 'vlm_policy_CodexAppServerPolicy',
                        'reference_sha256': sha256(source / 'demonstration_reference.json'),
                        'source_hashes': {'/historical/environment.py': ARCHIVED_ENV_SHA256}}
            (source / 'manifest.json').write_text(json.dumps(manifest))
            scenario = {'name': 'large', 'distractors': [], 'translation_m': [.15, .1, 0],
                        'target_initial_position_m': [-.6 + .15, -.1333 + .1, .225]}
            scenario_path = source / 'scenario_evaluation_only.json'
            scenario_path.write_text(json.dumps(scenario))
            (source / 'score.json').write_text(json.dumps({'success': True, 'fault': None, 'control_steps': len(rows)}))
            converted, intervals, _ = convert_rows(source / 'control.jsonl', len(rows))
            teacher = root / 'teacher.jsonl'
            teacher_rows = copy.deepcopy(converted)
            for original, row in zip(rows, teacher_rows):
                row['after']['wrist_position_m'] = original['observation']['wrist_position_m']
            write_rows(teacher, teacher_rows)
            snapshot = root / 's8v4_four_tests/fixture/sources/12_environment.py'
            snapshot.parent.mkdir(parents=True)
            snapshot.write_bytes((ROOT / 'outputs/s8v4_four_tests/demo_context_four_20261024_v1/sources/12_environment.py').read_bytes())
            output = root / 'large_bundle'
            metadata = import_trajectory(source, output, teacher_path=teacher)
            self.assertEqual(metadata['scene']['scenario_name'], 'large')
            self.assertEqual(metadata['scene']['target_initial_position_m'], scenario['target_initial_position_m'])
            self.assertEqual(metadata['scene']['translation_m'], [.15, .1, 0.])
            self.assertEqual(metadata['phase_intervals'], intervals)
            self.assertEqual(metadata['duration_s'], len(rows) * CONTROL_DT)
            self.assertEqual(metadata['archived_environment']['path'], str(snapshot))
            self.assertEqual(metadata['initial_state']['home_arm_degrees'], HOME_DEGREES)
            self.assertEqual(metadata['source_pd']['kp'][6:], [3.] * 12)
            self.assertIsNone(metadata['conversion']['new_replay_success'])
            self.assertFalse(metadata['conversion']['historical_tactile_copied'])
            aligned = metadata['phase_aligned_teacher_comparison']
            self.assertEqual(aligned['samples_per_phase'], 101)
            self.assertEqual(aligned['common_phases'], list(PHASE_ORDER))
            self.assertEqual(aligned['arm_max_deg'], 0.)
            self.assertEqual(aligned['hand_max_deg'], 0.)
            self.assertEqual(aligned['wrist_max_distance_cm'], 0.)
            self.assertEqual([json.loads(line) for line in (output / 'control.jsonl').read_text().splitlines()], converted)

            malformed_scenes = []
            wrong = copy.deepcopy(scenario); wrong['translation_m'][0] = .16; malformed_scenes.append(wrong)
            wrong = copy.deepcopy(scenario); wrong['target_initial_position_m'][0] += .001; malformed_scenes.append(wrong)
            wrong = copy.deepcopy(scenario); wrong['target_initial_position_m'][1] = float('nan'); malformed_scenes.append(wrong)
            wrong = copy.deepcopy(scenario); wrong['translation_m'][2] = False; malformed_scenes.append(wrong)
            wrong = copy.deepcopy(scenario); wrong['name'] = 'small'; malformed_scenes.append(wrong)
            wrong = copy.deepcopy(scenario); wrong['distractors'] = [{'shape': 'sphere'}]; malformed_scenes.append(wrong)
            for number, wrong in enumerate(malformed_scenes):
                with self.subTest(scene=number):
                    scenario_path.write_text(json.dumps(wrong))
                    rejected_output = root / f'rejected_scene_{number}'
                    with self.assertRaises(ValueError):
                        import_trajectory(source, rejected_output, teacher_path=teacher)
                    self.assertFalse(rejected_output.exists())

            scenario_path.write_text(json.dumps(scenario))
            rows[1]['observation']['sim_time_s'] += CONTROL_DT
            write_rows(source / 'control.jsonl', rows)
            with self.assertRaisesRegex(ValueError, '40 Hz step grid'):
                import_trajectory(source, root / 'rejected_time', teacher_path=teacher)
            self.assertFalse((root / 'rejected_time').exists())

    def test_phase_comparison_measures_matching_progress_not_wall_clock(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = source_rows()
            source_path = root / 'source.jsonl'
            write_rows(source_path, source)
            converted, _, _ = convert_rows(source_path, len(source))
            teacher_rows = []
            # A known arm offset and wrist translation; source has unchanged hand
            # commands. Duplicate each teacher sample so phase durations differ.
            for original, row in zip(source, converted):
                row = copy.deepcopy(row)
                row['sent_target_rad'][0] -= .1
                row['after']['wrist_position_m'] = (np.asarray(original['observation']['wrist_position_m']) - [.03, .04, 0.]).tolist()
                teacher_rows.extend([copy.deepcopy(row), copy.deepcopy(row)])
            for number, row in enumerate(teacher_rows, 1):
                row['step'] = number
                row['before']['sim_time_s'] = (number - 1) * CONTROL_DT
                row['after']['sim_time_s'] = number * CONTROL_DT
            teacher_path = root / 'teacher.jsonl'
            write_rows(teacher_path, teacher_rows)
            aligned = compare_teacher_by_phase(source_path, teacher_path)
            # First eight phases each have one source sample, so their offsets
            # are exact even though teacher sample counts and durations doubled.
            for phase in PHASE_ORDER[:-1]:
                phase_result = aligned['per_phase'][phase]
                self.assertAlmostEqual(phase_result['arm_max_deg'], np.rad2deg(.1))
                self.assertAlmostEqual(phase_result['wrist_max_distance_cm'], 5.)
                self.assertEqual(phase_result['hand_max_deg'], 0.)
            wall_clock = compare_teacher(converted, teacher_path)
            self.assertFalse(wall_clock['approximately_equal_at_1e_minus5_rad'])
            self.assertIn('phase-duration shifts', wall_clock['interpretation'])


if __name__ == '__main__':
    unittest.main()
