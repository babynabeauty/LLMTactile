"""Import audited original/large-scene Astra logs without replaying sensors.

This bounded importer supports the archived S8V4 four-case runner's original
and large-translation uncluttered scenes. It does not run Genesis, call Astra,
change source logs, or certify that
the converted trajectory succeeds at the residual environment's rates.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from ur7e_xhand_sim.build_assets import ARM_JOINTS, HAND_JOINTS
from .core import DEMO, Trajectory


CONTROL_DT = .025
JOINT_ORDER = ARM_JOINTS + HAND_JOINTS
PHASE_ORDER = ('settle', 'preshape', 'approach', 'align', 'close', 'preload',
               'test_lift', 'lift', 'hold')
# The constants below were read from this archived, hash-bound environment,
# not imported from today's potentially changed environment implementation.
ARCHIVED_ENV_SHA256 = '51da437d965535d533694d14fd4b2c227641bf8dfe2d9b51327da20823d96d95'
HOME_DEGREES = [0., -89.95, 89.96, -89.96, -89.95, -44.98]
SOURCE_PD = {'kp': [4500, 4500, 3500, 2000, 2000, 2000] + [3.] * 12,
             'kv': [450, 450, 350, 200, 200, 200] + [.08] * 12}
SCENE_TRANSLATIONS = {'original': [0., 0., 0.], 'large': [.15, .10, 0.]}
SOURCE_TARGET_POSITION = np.asarray([-.6, -.1333, .225])


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    with Path(path).open() as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f'Expected JSON object in {path}')
    return value


def finite_vector(value, length, name):
    if (not isinstance(value, list) or len(value) != length
            or any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in value)):
        raise ValueError(f'{name} must contain {length} numeric values')
    result = np.asarray(value, dtype=float)
    if not np.isfinite(result).all():
        raise ValueError(f'{name} contains nonfinite values')
    return result


def convert_rows(control_path, expected_steps):
    """Validate post-step observations and preserve the preceding sent target.

    Timestamps are calculated on the integer 40 Hz grid; the logged observation
    must agree with its interval end. Large tactile arrays are never copied.
    """
    rows, phases, feedback_rows = [], [], 0
    episode = None
    with Path(control_path).open() as stream:
        for number, line in enumerate(stream, 1):
            try:
                row = json.loads(line)
                if type(row['step']) is not int or row['step'] != number:
                    raise ValueError('Steps must be contiguous integers starting at 1')
                observation = row['observation']
                end = observation['sim_time_s']
                if (isinstance(end, bool) or not isinstance(end, (int, float))
                        or not np.isfinite(end) or abs(end - number * CONTROL_DT) > 1e-9):
                    raise ValueError('Post-step observation time differs from the 40 Hz step grid')
                if type(observation['episode_step']) is not int or observation['episode_step'] != number:
                    raise ValueError('Observation step differs from command step')
                if type(observation['episode']) is not int:
                    raise ValueError('Invalid source episode')
                if episode is None:
                    episode = observation['episode']
                if episode != observation['episode']:
                    raise ValueError('Multiple source episodes are not one trajectory')
                if observation.get('fault') is not None or observation.get('terminated') is not False:
                    raise ValueError('Faulted/terminated source trajectory')
                for key in ('joint_positions_rad', 'joint_velocities_rad_s'):
                    values = observation[key]
                    if not isinstance(values, dict) or list(values) != JOINT_ORDER:
                        raise ValueError(f'{key} joint order differs from the robot contract')
                    finite_vector(list(values.values()), 18, key)
                target = finite_vector(row['target'], 18, 'target')
                phase = row['phase']
                if phase not in PHASE_ORDER:
                    raise ValueError('Unknown trajectory phase')
                if not phases or phases[-1] != phase:
                    phases.append(phase)
                if row.get('local_feedback'):
                    feedback_rows += 1
                rows.append({'step': number, 'phase': phase,
                             'before': {'sim_time_s': (number - 1) * CONTROL_DT},
                             'after': {'sim_time_s': number * CONTROL_DT},
                             'sent_target_rad': target.tolist()})
            except (KeyError, TypeError, json.JSONDecodeError, ValueError) as error:
                raise ValueError(f'Invalid control row {number}: {error}') from error
    if type(expected_steps) is not int or expected_steps < 1 or len(rows) != expected_steps:
        raise ValueError('Source control row count differs from score.control_steps')
    if tuple(phases) != PHASE_ORDER:
        raise ValueError('Expected one complete, ordered settle-through-hold trajectory')
    if not np.allclose(rows[0]['sent_target_rad'],
                       np.r_[np.deg2rad(HOME_DEGREES), np.zeros(12)], atol=1e-6, rtol=0):
        raise ValueError('Initial target differs from the supported home posture')
    intervals = {}
    for phase in phases:
        group = [row for row in rows if row['phase'] == phase]
        intervals[phase] = {'start_s': group[0]['before']['sim_time_s'],
                            'end_s': group[-1]['after']['sim_time_s'],
                            'row_count': len(group)}
    if intervals['hold']['end_s'] - intervals['hold']['start_s'] < 3. - 1e-9:
        raise ValueError('Source final hold is shorter than three seconds')
    hand = np.asarray([row['sent_target_rad'][6:] for row in rows
                       if row['phase'] in ('test_lift', 'lift', 'hold')])
    return rows, intervals, {
        'rows_with_local_feedback_diagnostics': feedback_rows,
        'target_semantics': 'post-RuleSupportController commanded target; measured joint positions are not targets',
        'lift_phase_hand_target_max_change_rad': float(np.abs(hand - hand[0]).max()),
        'warning': 'Command constancy does not independently prove controller internals or historical tactile validity',
    }


def compare_teacher(rows, teacher_path):
    teacher_path = Path(teacher_path).resolve()
    teacher = Trajectory(teacher_path)
    times = np.asarray([row['before']['sim_time_s'] for row in rows])
    targets = np.asarray([row['sent_target_rad'] for row in rows])
    duration = rows[-1]['after']['sim_time_s']
    # Same action-start grid and same linear interpolation as the PPO replay.
    grid = np.arange(int(np.ceil(min(duration, teacher.duration_s) * 15))) / 15.
    candidate = np.stack([np.interp(grid, times, targets[:, j]) for j in range(18)], axis=-1)
    difference = candidate - teacher.target(grid)
    raw = targets - teacher.targets if targets.shape == teacher.targets.shape and np.array_equal(times, teacher.times) else None
    starts = {row['phase']: row['before']['sim_time_s'] for row in reversed(rows)}
    approximately_equal = bool(np.abs(difference).max() <= 1e-5)
    return {
        'teacher_path': str(teacher_path), 'teacher_sha256': teacher.sha256,
        'comparison_hz': 15, 'comparison_samples': len(grid),
        'interpolation': 'linear targets at interval start; endpoint target held after last row start',
        'duration_equal': duration == teacher.duration_s,
        'phase_starts_equal': starts == teacher.phase_starts,
        'raw_targets_exactly_equal': bool(raw is not None and np.array_equal(targets, teacher.targets)),
        'raw_max_abs_difference_rad': None if raw is None else float(np.abs(raw).max()),
        'resampled_max_abs_difference_rad': float(np.abs(difference).max()),
        'resampled_arm_max_abs_difference_rad': float(np.abs(difference[:, :6]).max()),
        'resampled_hand_max_abs_difference_rad': float(np.abs(difference[:, 6:]).max()),
        'resampled_rms_difference_rad': float(np.sqrt(np.mean(difference ** 2))),
        'resampled_max_abs_difference_per_joint_rad': np.abs(difference).max(axis=0).tolist(),
        'approximately_equal_at_1e_minus5_rad': approximately_equal,
        'interpretation': (
            'Near-identical trajectories support source-path replay verification, not evidence of substantial new-motion generalization'
            if approximately_equal else
            'Wall-clock alignment mixes motion differences with phase-duration shifts; these maxima are not pure pose or hand-shape differences. Use the phase-normalized comparison for matched skill progression.'),
    }


def compare_teacher_by_phase(control_path, teacher_path):
    """Recompute matched-progression diagnostics from source logs, not a simulator.

    Command targets and measured post-step wrist positions are separate series.
    Each phase's first through last sample is normalized to 0..1, then linearly
    interpolated at 101 equally spaced progress values. No wrist samples or
    historical sensor values become playback commands or policy observations.
    """
    def records(path, target_key, observation_key):
        result = {}
        with Path(path).open() as stream:
            for number, line in enumerate(stream, 1):
                try:
                    row = json.loads(line)
                    phase = row['phase']
                    if phase not in PHASE_ORDER:
                        raise ValueError('Unknown phase')
                    target = finite_vector(row[target_key], 18, target_key)
                    wrist = finite_vector(row[observation_key]['wrist_position_m'], 3,
                                          'wrist_position_m')
                    result.setdefault(phase, []).append(np.r_[target, wrist])
                except (KeyError, TypeError, ValueError) as error:
                    raise ValueError(f'Invalid phase comparison row {number} in {path}: {error}') from error
        if tuple(result) != PHASE_ORDER:
            raise ValueError('Phase comparison requires all matching skill phases')
        return result

    candidate = records(control_path, 'target', 'observation')
    teacher = records(teacher_path, 'sent_target_rad', 'after')
    progress = np.linspace(0., 1., 101)
    per_phase = {}
    for phase in PHASE_ORDER:
        def resample(values):
            values = np.asarray(values)
            old_progress = np.linspace(0., 1., len(values))
            return np.stack([np.interp(progress, old_progress, values[:, j])
                             for j in range(21)], axis=-1)
        difference = resample(candidate[phase]) - resample(teacher[phase])
        per_phase[phase] = {
            'teacher_rows': len(teacher[phase]), 'candidate_rows': len(candidate[phase]),
            'arm_max_deg': float(np.rad2deg(np.abs(difference[:, :6]).max())),
            'hand_max_deg': float(np.rad2deg(np.abs(difference[:, 6:18]).max())),
            'wrist_max_distance_cm': float(np.linalg.norm(difference[:, 18:], axis=1).max() * 100),
            'mean_wrist_difference_m': difference[:, 18:].mean(axis=0).tolist(),
        }
    return {
        'teacher_path': str(Path(teacher_path).resolve()), 'teacher_sha256': sha256(teacher_path),
        'source_control_sha256': sha256(control_path),
        'alignment': 'Each matching skill phase independently normalized to 101 equally spaced progression samples; wrist positions are measured, joint targets commanded.',
        'samples_per_phase': 101, 'common_phases': list(PHASE_ORDER),
        **{key: max(value[key] for value in per_phase.values())
           for key in ('arm_max_deg', 'hand_max_deg', 'wrist_max_distance_cm')},
        'per_phase': per_phase,
        'interpretation': 'Phase normalization separates timing changes from matched skill progression; measured wrist displacement describes the historical runs, not new replay success or new hand-shape learning.',
    }


def import_trajectory(source_run, output, source_suite=None, teacher_path=DEMO / 'control.jsonl'):
    source_run, output = Path(source_run).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite trajectory bundle: {output}')
    if source_run == output or source_run in output.parents:
        raise ValueError('Write the imported bundle outside the source run')
    names = ('control.jsonl', 'manifest.json', 'scenario_evaluation_only.json',
             'demonstration_reference.json', 'score.json')
    hashes = {name: sha256(source_run / name) for name in names}
    manifest = read_json(source_run / 'manifest.json')
    reference = read_json(source_run / 'demonstration_reference.json')
    scenario = read_json(source_run / 'scenario_evaluation_only.json')
    score = read_json(source_run / 'score.json')
    if manifest.get('high_level_driver') != 'vlm_policy_CodexAppServerPolicy':
        raise ValueError('Source is not the supported Astra S8V4 driver')
    if manifest.get('reference_sha256') != hashes['demonstration_reference.json']:
        raise ValueError('Copied demonstration reference differs from source manifest')
    if reference.get('arm_joint_order', []) + reference.get('hand_joint_order', []) != JOINT_ORDER:
        raise ValueError('Demonstration reference joint order differs from the robot contract')
    scenario_name = scenario.get('name')
    if not isinstance(scenario_name, str) or scenario_name not in SCENE_TRANSLATIONS:
        raise ValueError('Only the original or large uncluttered scene is supported by this importer')
    translation = finite_vector(scenario.get('translation_m'), 3, 'translation_m')
    target_position = finite_vector(scenario.get('target_initial_position_m'), 3,
                                    'target_initial_position_m')
    expected_translation = np.asarray(SCENE_TRANSLATIONS[scenario_name])
    # Match the actual archived arithmetic exactly, including its binary float
    # representation. Do not turn this into a general scene-translation loader.
    if (scenario.get('distractors') != []
            or not np.array_equal(translation, expected_translation)
            or not np.array_equal(target_position, SOURCE_TARGET_POSITION + expected_translation)):
        raise ValueError('Scene differs from the supported original or large uncluttered source contract')
    if score.get('success') is not True or score.get('fault') is not None:
        raise ValueError('Expected the completed successful source run; this is not replay validation')
    if source_suite is None:
        suffix = '_' + scenario_name
        if not source_run.name.endswith(suffix):
            raise ValueError('Cannot infer archived source suite; pass --source-suite')
        source_suite = source_run.parent.parent / 's8v4_four_tests' / source_run.name.removesuffix(suffix)
    snapshot = Path(source_suite).resolve() / 'sources/12_environment.py'
    declared_env_hashes = [value for key, value in manifest.get('source_hashes', {}).items()
                           if Path(key).name == 'environment.py']
    if declared_env_hashes != [ARCHIVED_ENV_SHA256] or sha256(snapshot) != ARCHIVED_ENV_SHA256:
        raise ValueError('Historical environment snapshot does not match the supported source contract')
    rows, phases, feedback = convert_rows(source_run / 'control.jsonl', score.get('control_steps'))
    comparison = compare_teacher(rows, teacher_path)
    metadata = {
        'schema': 'astra_command_trajectory_v1', 'status': 'converted_not_replay_validated',
        'trajectory_file': 'control.jsonl', 'source_run': str(source_run), 'source_files': hashes,
        'archived_environment': {'path': str(snapshot), 'sha256': ARCHIVED_ENV_SHA256},
        'joint_order': JOINT_ORDER, 'control_dt_s': CONTROL_DT, 'source_physics_dt_s': .005,
        'row_count': len(rows), 'duration_s': rows[-1]['after']['sim_time_s'],
        'phase_intervals': phases,
        'scene': {'scenario_name': scenario_name, 'target_initial_position_m': target_position.tolist(),
                  'translation_m': translation.tolist(), 'distractors': [],
                  'robot_base_position_m': [0., 0., .2], 'table_top_z_m': .2,
                  'cube_size_m': [.045] * 3},
        'initial_state': {'home_arm_degrees': HOME_DEGREES,
                          'home_joint_positions_rad': np.r_[np.deg2rad(HOME_DEGREES), np.zeros(12)].tolist(),
                          'home_joint_velocities_rad_s': [0.] * 18},
        'source_pd': SOURCE_PD,
        'source_material': {'cube_density_kg_m3': 500., 'cube_mass_kg': 500. * .045 ** 3,
                            'cube_friction': .8, 'robot_friction': .8, 'table_friction': .8},
        'source_recorded_score': score, 'source_controller_audit': feedback,
        'original_teacher_comparison': comparison,
        'conversion': {'sent_target_source': 'control.jsonl.target',
                       'interval_start': '(step - 1) * 0.025 seconds',
                       'interval_end': 'step * 0.025 seconds; verified against observation.sim_time_s',
                       'historical_tactile_copied': False, 'historical_physics_state_replayed': False,
                       'resampling_not_performed_by_importer': True,
                       'new_replay_success': None},
    }
    if scenario_name == 'large':
        metadata['phase_aligned_teacher_comparison'] = compare_teacher_by_phase(
            source_run / 'control.jsonl', teacher_path)
    # Recheck files after streaming them, before publishing a new bundle.
    if hashes != {name: sha256(source_run / name) for name in names}:
        raise ValueError('Source files changed while converting')
    output.mkdir(parents=True, exist_ok=False)
    control_path = output / 'control.jsonl'
    with control_path.open('x') as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False, separators=(',', ':')) + '\n')
    metadata['trajectory_sha256'] = sha256(control_path)
    with (output / 'trajectory_metadata.json').open('x') as stream:
        json.dump(metadata, stream, indent=2, allow_nan=False)
        stream.write('\n')
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--source-suite', type=Path)
    parser.add_argument('--teacher-trajectory', type=Path, default=DEMO / 'control.jsonl')
    args = parser.parse_args()
    result = import_trajectory(args.source_run, args.output, args.source_suite, args.teacher_trajectory)
    print(json.dumps({key: result[key] for key in ('schema', 'status', 'row_count', 'duration_s',
                     'trajectory_sha256', 'original_teacher_comparison')}, indent=2))


if __name__ == '__main__':
    main()
