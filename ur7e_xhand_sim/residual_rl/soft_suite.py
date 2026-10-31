"""Run the frozen elastic large-translation evaluation in fresh processes.

Only protocol-declared official episodes enter the summary: 18 for seed1_only,
or 72 for the legacy full_three_seeds scope. Existing
attempts, including failures/interrupted runs, are never automatically retried.
Preflight and repeat-reference directories are deliberately not discovered.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback
import zipfile

import numpy as np

from .core import ROOT, Trajectory, interval_overlap


DEFAULT_ROOT = ROOT/'outputs/soft_large_trajectory_20261026_v1'
EXPECTED_CONTROLLERS = ('fixed_2', 'fixed_3', 'tactile_seed1', 'no_tactile_seed1',
                        'tactile_seed2', 'no_tactile_seed2', 'tactile_seed3', 'no_tactile_seed3')
SEED1_CONTROLLERS = ('tactile_seed1', 'no_tactile_seed1')
PREFIX_FIELDS = frozenset(('state', 'tactile', 'vertices', 'q'))


def utc():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    """Atomically publish progress so an external viewer never reads half JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temp.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resolved(path):
    path = Path(path).expanduser()
    return (path if path.is_absolute() else ROOT/path).resolve()


def validate_prefix_tolerance(value, *, required=False):
    """Validate a predeclared absolute-only budget; legacy audits stay exact."""
    if value is None:
        if required:
            raise ValueError('Freeze protocol.prefix_tolerance_abs before official episodes')
        return {key: 0. for key in sorted(PREFIX_FIELDS)}
    if not isinstance(value, dict) or set(value) != PREFIX_FIELDS:
        raise ValueError('prefix_tolerance_abs must contain exactly state/tactile/vertices/q')
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v) or v < 0
           for v in value.values()):
        raise ValueError('Prefix absolute tolerances must be finite nonnegative numbers')
    return {key: float(value[key]) for key in sorted(PREFIX_FIELDS)}


def load_protocol(root):
    protocol_path = root/'protocol.json'
    protocol = read(protocol_path)
    if protocol.get('training') is not False or protocol.get('no_live_astra_or_rule_feedback') is not True:
        raise ValueError('Suite requires frozen evaluation without training or live feedback')
    scope = protocol.get('scope', 'full_three_seeds')
    if scope == 'seed1_only':
        if tuple(protocol['controllers']) != SEED1_CONTROLLERS:
            raise ValueError('seed1_only requires exactly tactile_seed1 then no_tactile_seed1; no extra controllers')
        expected_models = set(SEED1_CONTROLLERS)
        expected_model_episodes, expected_baseline_episodes = 18, 0
    elif scope == 'full_three_seeds':
        if (len(protocol['controllers']) != 8 or
                set(protocol['controllers']) != set(EXPECTED_CONTROLLERS)):
            raise ValueError('full_three_seeds requires fixed2/fixed3 and six seed1/2/3 policies')
        expected_models = set(EXPECTED_CONTROLLERS)-{'fixed_2', 'fixed_3'}
        expected_model_episodes, expected_baseline_episodes = 54, 18
    else:
        raise ValueError(f'Unsupported experiment scope: {scope}')
    if set(protocol['models']) != expected_models:
        raise ValueError(f'Protocol must freeze exactly the model paths and hashes for {scope}')
    conditions = np.asarray(protocol['conditions_kg_mu'], dtype=float)
    expected_conditions = {(m, f) for m in (.075, .125, .175) for f in (.3, .5, .7)}
    if conditions.shape != (9, 2) or not np.isfinite(conditions).all() or set(map(tuple, conditions)) != expected_conditions:
        raise ValueError('Expected the declared nine mass/friction conditions')
    if (protocol.get('official_model_episodes') != expected_model_episodes or
            protocol.get('official_fixed_baseline_episodes') != expected_baseline_episodes):
        raise ValueError(f'{scope} requires {expected_model_episodes} model and {expected_baseline_episodes} baseline episodes')
    if protocol.get('rates_hz') != {'physics': 300, 'touch': 60, 'action': 15}:
        raise ValueError('The frozen actors require physics/touch/action rates 300/60/15')
    prefix_tolerance = validate_prefix_tolerance(protocol.get('prefix_tolerance_abs'), required=True)
    physics = protocol.get('physics', {})
    substeps = physics.get('substeps', protocol.get('physics_substeps'))
    newton = physics.get('newton_tolerance', protocol.get('physics_newton_tolerance'))
    maxvolume = physics.get('maxvolume_m3', protocol.get('tet_maxvolume_m3', 1e-7))
    if type(substeps) is not int or substeps < 1:
        raise ValueError('Freeze an explicit positive integer physics_substeps')
    if newton is None or not np.isfinite(newton) or newton <= 0:
        raise ValueError('Freeze physics.newton_tolerance or physics_newton_tolerance before running')
    if not np.isfinite(maxvolume) or maxvolume <= 0:
        raise ValueError('Invalid frozen tet maxvolume')
    material = protocol['material']
    if (material['class'] != 'FEM.Elastic' or material['model'] != 'stable_neohookean'
            or material['E_Pa'] != 1e5 or material['nu'] != .45
            or not np.array_equal(material['size_m'], [.045]*3)
            or not np.array_equal(material['initial_position_m'], [-.45, -.0333, .225])):
        raise ValueError('Protocol material differs from the supported soft scene')
    sources = protocol.get('frozen_sources')
    if not isinstance(sources, dict) or not sources:
        raise ValueError('Populate protocol.frozen_sources before running official episodes')
    required_sources = {f'ur7e_xhand_sim/residual_rl/{name}.py' for name in
                        ('soft_suite', 'soft_benchmark', 'soft_scene', 'soft_taxel', 'soft_contact',
                         'core', 'adaptive_policy')}
    if not required_sources.issubset(sources):
        raise ValueError('Frozen sources omit required evaluator/physics/sensor/policy modules')
    for rel in sources:
        path = Path(rel)
        if path.is_absolute() or '..' in path.parts or not resolved(path).is_relative_to(ROOT.resolve()):
            raise ValueError(f'Frozen source must be repository relative: {rel}')
    result = {'protocol': protocol, 'scope': scope, 'protocol_sha256': sha(protocol_path),
              'trajectory': str(resolved(protocol['trajectory'])),
              'prefix_tolerance_abs': prefix_tolerance,
              'physics': {'substeps': substeps, 'newton_tolerance': float(newton),
                          'maxvolume_m3': float(maxvolume)}}
    verify_inputs(root, result)
    trajectory = Trajectory(result['trajectory'])
    if not np.allclose([trajectory.phase_starts['hold'], trajectory.duration_s],
                       protocol['hold_interval_s'], rtol=0, atol=1e-10):
        raise ValueError('Hold interval differs from the frozen trajectory')
    if abs(trajectory.duration_s-trajectory.phase_starts['hold']-3.) > 1e-10:
        raise ValueError('Expected a final 3-second hold')
    result['timing'] = {'start_tick': int(trajectory.phase_starts['preload']*15)*20,
                        'end_tick': int(np.ceil(trajectory.duration_s*15))*20,
                        'hold_start': trajectory.phase_starts['hold'], 'hold_end': trajectory.duration_s}
    return result


def verify_inputs(root, frozen):
    """Fail closed if any input changes before, between or after episodes."""
    protocol = frozen['protocol']
    if sha(root/'protocol.json') != frozen['protocol_sha256']:
        raise ValueError('protocol.json changed after suite preparation')
    if sha(frozen['trajectory']) != protocol['trajectory_sha256']:
        raise ValueError('Frozen trajectory hash mismatch')
    metadata = read(Path(frozen['trajectory']).with_name('trajectory_metadata.json'))
    if metadata.get('trajectory_sha256') != protocol['trajectory_sha256']:
        raise ValueError('Trajectory metadata hash does not match commands')
    if metadata['scene'].get('scenario_name') != 'large':
        raise ValueError('This suite requires the large-translation trajectory')
    for name, info in protocol['models'].items():
        if sha(resolved(info['path'])) != info['sha256']:
            raise ValueError(f'Frozen model hash mismatch: {name}')
    for rel, expected in protocol['frozen_sources'].items():
        if sha(resolved(rel)) != expected:
            raise ValueError(f'Frozen source hash mismatch: {rel}')


def make_jobs(root, frozen):
    protocol = frozen['protocol']
    jobs = []
    for i, (mass, friction) in enumerate(protocol['conditions_kg_mu']):
        condition = f'condition_{i:02d}_{round(mass*1000):03d}g_mu{friction:g}'
        for controller in protocol['controllers']:
            key = f'{condition}__{controller}'
            output = root/'official'/condition/controller
            cmd = [sys.executable, '-u', '-m', 'ur7e_xhand_sim.residual_rl.soft_benchmark',
                   '--output', str(output), '--trajectory', frozen['trajectory'],
                   '--controller', controller, '--mass-kg', repr(float(mass)),
                   '--friction', repr(float(friction)), '--substeps', str(frozen['physics']['substeps']),
                   '--newton-tolerance', repr(frozen['physics']['newton_tolerance']),
                   '--maxvolume', repr(frozen['physics']['maxvolume_m3'])]
            if controller.startswith('fixed_'):
                cmd += ['--fixed-gain', controller.removeprefix('fixed_')]
            else:
                cmd += ['--model', str(resolved(protocol['models'][controller]['path']))]
            jobs.append({'id': key, 'condition_index': i, 'condition': condition,
                         'controller': controller, 'mass_kg': float(mass), 'friction': float(friction),
                         'output': str(output), 'log': str(root/'official_logs'/f'{key}.log'),
                         'attempt': str(root/'official_attempts'/f'{key}.json'), 'command': cmd})
    return jobs


def process_token(pid):
    """Linux process start time distinguishes a live child from a reused PID."""
    try:
        # Fields after the final ')' begin at stat field 3; starttime is field 22.
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return fields[19] if fields[0] != 'Z' else None
    except (FileNotFoundError, ProcessLookupError, PermissionError, IndexError):
        return None


def attempted(job):
    return any(Path(job[key]).exists() for key in ('attempt', 'output', 'log'))


def existing_running(job):
    path = Path(job['attempt'])
    if not path.exists():
        return False
    attempt = read(path)
    token = attempt.get('process_start_token')
    return bool(token and attempt.get('returncode') is None
                and process_token(attempt.get('pid', -1)) == token)


def progress(root, jobs, active, state, error=None):
    entries = []
    for job in jobs:
        output = Path(job['output'])
        status = {'status': 'pending'}
        if (output/'status.json').exists():
            try:
                status = read(output/'status.json')
            except (ValueError, OSError):
                status = {'status': 'unreadable_episode_status'}
        elif attempted(job):
            status = {'status': 'attempted_without_episode_status'}
        entries.append({'id': job['id'], 'controller': job['controller'],
                        'condition_index': job['condition_index'], 'log': job['log'],
                        'active': job['id'] in active, 'episode_status': status})
    report_error = None
    try:
        if (root/'report_status.json').exists():
            report_error = read(root/'report_status.json').get('report_error')
    except (OSError, ValueError):
        report_error = 'Report status could not be read'
    write(root/'suite_status.json', {'status': state, 'updated_utc': utc(),
          'official_planned': len(jobs), 'attempted': sum(attempted(j) for j in jobs),
          'active': len(active), 'pending': sum(not attempted(j) for j in jobs),
          'completed_process_outputs': sum(x['episode_status'].get('status') == 'completed' for x in entries),
          'error': error, 'report_error': report_error, 'episodes': entries})


def numeric_summary(values):
    values = [float(v) for v in values if v is not None and np.isfinite(v)]
    return {'n': len(values), 'mean': float(np.mean(values)) if values else None,
            'min': min(values) if values else None, 'max': max(values) if values else None}


def assess_job(job, frozen):
    """Independently recompute official labels and aggregate trace diagnostics."""
    out = Path(job['output'])
    result = {k: job[k] for k in ('id', 'controller', 'condition_index', 'mass_kg', 'friction', 'output', 'log')}
    result.update(attempted=attempted(job), valid_completed=False)
    try:
        if not attempted(job):
            result.update(status='pending', failure_reason='not_attempted')
            return result
        if Path(job['attempt']).exists():
            attempt = read(job['attempt'])
            if attempt.get('returncode') is None and (attempt.get('status') == 'launching' or existing_running(job)):
                result.update(status='running', failure_reason=None)
                return result
            if attempt.get('returncode') != 0:
                result.update(status='execution_failed_or_interrupted',
                              failure_reason=f"Process returncode {attempt.get('returncode')}: {attempt.get('status')}")
                return result
        status = read(out/'status.json') if (out/'status.json').exists() else {}
        if status.get('status') != 'completed':
            result.update(status='execution_failed_or_interrupted',
                          failure_reason=status.get('error', status.get('status', 'no_episode_status')))
            return result
        score, manifest = read(out/'score.json'), read(out/'manifest.json')
        p, timing = frozen['protocol'], frozen['timing']
        if (manifest['controller'] != job['controller'] or manifest.get('training') is not False
                or manifest['trajectory_sha256'] != p['trajectory_sha256']
                or manifest['mass_kg'] != job['mass_kg'] or manifest['friction'] != job['friction']
                or manifest['substeps'] != frozen['physics']['substeps']
                or manifest['start_tick'] != timing['start_tick'] or manifest['end_tick'] != timing['end_tick']
                or not score.get('complete_planned_trajectory')):
            raise ValueError('Episode manifest/length differs from frozen job')
        if job['controller'] in p['models']:
            if manifest['model']['sha256'] != p['models'][job['controller']]['sha256'] or score.get('model_unchanged') is not True:
                raise ValueError('Episode model provenance differs')
        elif manifest['model'] is not None:
            raise ValueError('Fixed baseline unexpectedly loaded a model')
        source_manifest = manifest['source_sha256']
        for rel, expected in p['frozen_sources'].items():
            if Path(rel).name in source_manifest and source_manifest[Path(rel).name] != expected:
                raise ValueError(f'Episode used changed source: {rel}')
        environment = read(out/'environment'/'soft_scene_protocol.json')
        phy = environment['physics']
        if (phy['substeps'] != frozen['physics']['substeps'] or phy['dt_s'] != 1/300
                or phy['coupler_options']['newton_tolerance'] != frozen['physics']['newton_tolerance']
                or environment['mesh']['maxvolume_m3'] != frozen['physics']['maxvolume_m3']):
            raise ValueError('Actual physics configuration differs from frozen settings')
        material = environment['material']
        if (not material['mass_checks_passed'] or
                not np.isclose(material['mass_from_actual_tet_volume_kg'], job['mass_kg'], rtol=1e-5)
                or not np.isclose(material['ipc_pair_readback']['hand_soft']['friction'], job['friction'], atol=1e-12)
                or not np.isclose(material['ipc_pair_readback']['table_soft']['friction'], .8, atol=1e-12)):
            raise ValueError('Actual mass/contact pair friction does not match the job')
        with np.load(out/'trace.npz', allow_pickle=False) as trace:
            t = trace['time_s']
            expected_t = np.arange(5, timing['end_tick']+1, 5)/300
            if not np.array_equal(t, expected_t):
                raise ValueError('Missing/reordered tactile samples')
            actions = trace['actions']
            if not np.array_equal(actions[:, 0], np.arange(timing['start_tick'], timing['end_tick'], 20)/300):
                raise ValueError('Missing/reordered policy actions')
            if trace['state_before'].shape != (len(actions), 12, 79) or trace['tactile_before'].shape != (len(actions), 12, 5, 120, 3):
                raise ValueError('Unexpected actor observation trace shape')
            with np.load(out/'prefix_observation.npz', allow_pickle=False) as prefix:
                if (not np.array_equal(trace['state_before'][0], prefix['state'])
                        or not np.array_equal(trace['tactile_before'][0], prefix['tactile'])):
                    raise ValueError('First actor input differs from the recorded prefix observation')
            clearance, normal = trace['clearance_m'], trace['contact_normal_load_N']
            min_j, volume = trace['min_J'], trace['volume_ratio']
            if not all(np.isfinite(x).all() for x in (clearance, normal, min_j, volume, actions,
                                                       trace['state_before'], trace['tactile_before'])):
                raise ValueError('Nonfinite trace values')
            weights = interval_overlap(t-1/60, t, timing['hold_start'], timing['hold_end'])
            hold = weights > 1e-12
            fault = bool(trace['fault'].any())
            task = bool(abs(weights.sum()-3.) < 1e-7 and (clearance[hold] >= .05).all()
                        and not fault and (min_j > 0).all())
            strict = bool(task and ((normal[hold] > .05).sum(-1) >= 2).all())
            if task != score['task_success'] or strict != score['strict_success']:
                raise ValueError('Saved task labels disagree with independent trace computation')
            policy = t > timing['start_tick']/300
            mean_load = float(normal[policy].sum(-1).mean())
            if not np.isclose(mean_load, score['mean_normal_load_N'], rtol=1e-7, atol=1e-9):
                raise ValueError('Saved mean normal load disagrees with trace')
            contact_positive = normal > .05
            touch_positive = np.linalg.norm(trace['tactile'], axis=-1).max(-1) > 1e-5
            counts = contact_positive.sum(0)
            misses = (contact_positive & ~touch_positive).sum(0)
            if (not np.array_equal(counts, score['sensor_coverage']['independent_positive_samples_per_finger'])
                    or not np.array_equal(misses, score['sensor_coverage']['positive_contact_but_zero_taxel_samples_per_finger'])):
                raise ValueError('Saved sensor-coverage counts disagree with trace')
            result.update(valid_completed=True, status='completed', task_success=task, strict_success=strict,
                mean_normal_load_N=mean_load, hold_min_clearance_m=float(clearance[hold].min()),
                final_clearance_m=float(clearance[-1]), collision=fault,
                deformation={'min_J': float(min_j.min()), 'min_volume_ratio': float(volume.min()),
                             'max_volume_ratio': float(volume.max()), 'final_volume_ratio': float(volume[-1])},
                sensor_coverage={'positive_samples_per_finger': counts.tolist(),
                                 'missed_samples_per_finger': misses.tolist()},
                failure_reason=None if strict else 'hold_failed' if not task else 'contact_discontinuity',
                original_score=score, independently_recomputed_labels=True)
    except (OSError, ValueError, KeyError, IndexError, TypeError, AssertionError, zipfile.BadZipFile) as exc:
        result.update(status='invalid_artifact', failure_reason=str(exc))
    return result


def group_summary(rows):
    valid = [x for x in rows if x['valid_completed']]
    counts = np.sum([r['sensor_coverage']['positive_samples_per_finger'] for r in valid], axis=0) if valid else np.zeros(5, int)
    misses = np.sum([r['sensor_coverage']['missed_samples_per_finger'] for r in valid], axis=0) if valid else np.zeros(5, int)
    task = sum(r['task_success'] for r in valid)
    strict = sum(r['strict_success'] for r in valid)
    unfinished = {'pending', 'not_attempted', 'running', 'initializing', 'awaiting_audit'}
    invalid = [r for r in rows if r['attempted'] and not r['valid_completed'] and r.get('status') not in unfinished]
    return {'planned': len(rows), 'attempted': sum(r['attempted'] for r in rows),
            'valid_completed': len(valid), 'execution_or_integrity_failures': len(invalid),
            'pending': sum(r.get('status') in ('pending', 'not_attempted') for r in rows),
            'running': sum(r.get('status') in ('running', 'initializing') for r in rows),
            'task_success_count': task, 'strict_success_count': strict,
            'task_success_rate_valid_completed': task/len(valid) if valid else None,
            'strict_success_rate_valid_completed': strict/len(valid) if valid else None,
            'verified_task_success_fraction_of_planned': task/len(rows) if rows else None,
            'verified_strict_success_fraction_of_planned': strict/len(rows) if rows else None,
            'normal_load_all_valid_episodes_N': numeric_summary([r['mean_normal_load_N'] for r in valid]),
            'normal_load_successful_episodes_N': numeric_summary([r['mean_normal_load_N'] for r in valid if r['task_success']]),
            'deformation': {key: numeric_summary([r['deformation'][key] for r in valid]) for key in
                            ('min_J', 'min_volume_ratio', 'max_volume_ratio', 'final_volume_ratio')},
            'sensor_coverage': {'positive_samples_per_finger': counts.tolist(), 'missed_samples_per_finger': misses.tolist(),
                'agreement_fraction_per_finger': [float((c-m)/c) if c else None for c, m in zip(counts, misses)],
                'meaning': 'Virtual sensor presence vs independent normal IPC contact, not force calibration'},
            'failures': [r['id'] for r in invalid] + [r['id'] for r in valid if not r['strict_success']]}


def prefix_audit(root, jobs, tolerance_abs=None):
    """Report exact equality AND predeclared absolute numerical differences.

    The optional third argument preserves older two-argument callers. A local
    protocol supplies their budgets when available; without one, zero tolerance
    preserves the previous exact-only acceptance. No relative tolerance exists.
    """
    protocol = read(root/'protocol.json') if (root/'protocol.json').exists() else {}
    expected_controllers = set(protocol.get('controllers', {j['controller'] for j in jobs}))
    declared = tolerance_abs
    if declared is None:
        declared = protocol.get('prefix_tolerance_abs')
    tolerance_abs = validate_prefix_tolerance(declared)
    records = []
    for condition in sorted({j['condition_index'] for j in jobs}):
        peers = [j for j in jobs if j['condition_index'] == condition]
        available = [j for j in peers if (Path(j['output'])/'prefix_observation.npz').exists()]
        loaded, errors = {}, {}
        for job in available:
            try:
                with np.load(Path(job['output'])/'prefix_observation.npz', allow_pickle=False) as data:
                    loaded[job['id']] = {key: data[key] for key in data.files}
            except (OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
                errors[job['id']] = str(exc)
        readable = [j for j in available if j['id'] in loaded]
        reference = next((j for j in readable if j['controller'] == 'fixed_2'), readable[0] if readable else None)
        schedule_matches = (len(peers) == len(expected_controllers)
                            and {j['controller'] for j in peers} == expected_controllers)
        record = {'condition_index': condition, 'expected_controllers': len(expected_controllers),
                  'scheduled_controllers': len(peers), 'schedule_matches_protocol': schedule_matches,
                  'missing_controllers': sorted(expected_controllers-{j['controller'] for j in peers}),
                  'available_prefixes': len(available), 'reference': reference['id'] if reference else None,
                  'missing': [j['id'] for j in peers if j not in available],
                  'unreadable': errors, 'comparisons': []}
        if reference:
            base = loaded[reference['id']]
            required = PREFIX_FIELDS
            if set(base) != required:
                record['error'] = 'Unexpected reference prefix fields'
            for job in readable:
                data = loaded[job['id']]
                comp = {'id': job['id'], 'keys_match': set(data) == required, 'arrays': {}}
                for key in sorted(required):
                    if key not in data or key not in base:
                        comp['arrays'][key] = {'exact': False, 'within_tolerance': False,
                            'max_abs_difference': None, 'absolute_tolerance': tolerance_abs[key],
                            'error': 'missing field'}
                        continue
                    a, b = base[key], data[key]
                    shape = a.shape == b.shape
                    numeric = bool(np.issubdtype(a.dtype, np.number) and np.issubdtype(b.dtype, np.number)
                                   and not np.iscomplexobj(a) and not np.iscomplexobj(b))
                    finite = bool(numeric and a.size and b.size and np.isfinite(a).all() and np.isfinite(b).all())
                    same_dtype = a.dtype == b.dtype
                    with np.errstate(over='ignore', invalid='ignore'):
                        differences = np.abs(a.astype(np.longdouble)-b.astype(np.longdouble)) if shape and finite else None
                        maximum = float(differences.max()) if differences is not None else None
                    comparison_finite = bool(differences is not None and np.isfinite(differences).all())
                    comp['arrays'][key] = {'shape_matches': shape, 'dtype_matches': a.dtype == b.dtype,
                        'finite': finite, 'numeric_real_dtype': numeric,
                        'exact': bool(shape and finite and same_dtype and np.array_equal(a, b)),
                        'max_abs_difference': maximum if maximum is not None and np.isfinite(maximum) else None,
                        'difference_finite': comparison_finite,
                        'absolute_tolerance': tolerance_abs[key],
                        'within_tolerance': bool(shape and finite and same_dtype and comparison_finite
                                                 and np.all(differences <= tolerance_abs[key]))}
                comp['exact'] = bool(comp['keys_match'] and all(x['exact'] for x in comp['arrays'].values()))
                comp['within_tolerance'] = bool(comp['keys_match'] and
                    all(x['within_tolerance'] for x in comp['arrays'].values()))
                record['comparisons'].append(comp)
        record['all_available_exact'] = bool(record['comparisons'] and all(x['exact'] for x in record['comparisons']))
        record['complete_and_exact'] = bool(schedule_matches and len(readable) == len(peers) and record['all_available_exact'])
        record['all_available_within_tolerance'] = bool(record['comparisons'] and
            all(x['within_tolerance'] for x in record['comparisons']))
        record['complete_and_within_tolerance'] = bool(schedule_matches and len(readable) == len(peers) and
                                                      record['all_available_within_tolerance'])
        records.append(record)
    all_nine = len(records) == 9 and all(r['schedule_matches_protocol'] for r in records)
    audit = {'all_nine_conditions_complete_and_exact': bool(all_nine and all(r['complete_and_exact'] for r in records)),
             'all_nine_conditions_complete_and_within_tolerance': bool(all_nine and
                 all(r['complete_and_within_tolerance'] for r in records)),
             'all_nine_complete_and_within_tolerance': bool(all_nine and
                 all(r['complete_and_within_tolerance'] for r in records)),
             'absolute_tolerances': tolerance_abs, 'relative_tolerance': 0.,
             'tolerances_explicitly_declared': declared is not None,
             'method': 'Per-material exact shape/dtype/value equality remains explicit. Separately require matching '
                       'shape/dtype, finite real numeric values, and every absolute difference <= the declared '
                       'field budget. No relative tolerance, averaging, rounding, or mismatched-array acceptance.',
             'interpretation': 'Declared tolerances are engineering acceptance gates fixed before official runs; '
                               'within_tolerance does not mean the inputs are completely identical.',
             'conditions': records}
    write(root/'prefix_audit.json', audit)
    return audit


def summarize(root, frozen, jobs):
    rows = [assess_job(job, frozen) for job in jobs]
    audit = prefix_audit(root, jobs, frozen['protocol'].get('prefix_tolerance_abs'))
    controllers = {name: group_summary([r for r in rows if r['controller'] == name])
                   for name in frozen['protocol']['controllers']}
    groups = {'tactile_selected_seeds': group_summary([r for r in rows if r['controller'].startswith('tactile_')]),
              'no_tactile_selected_seeds': group_summary([r for r in rows if r['controller'].startswith('no_tactile_')])}
    seeds = {}
    actual_seeds = sorted({int(name.rsplit('seed', 1)[1]) for name in controllers if name.startswith('tactile_seed')})
    for seed in actual_seeds:
        tactile = {r['condition_index']: r for r in rows if r['controller'] == f'tactile_seed{seed}'}
        blind = {r['condition_index']: r for r in rows if r['controller'] == f'no_tactile_seed{seed}'}
        common = [i for i in tactile if tactile[i]['valid_completed'] and blind[i]['valid_completed']
                  and tactile[i]['task_success'] and blind[i]['task_success']]
        t_load = float(np.mean([tactile[i]['mean_normal_load_N'] for i in common])) if common else None
        b_load = float(np.mean([blind[i]['mean_normal_load_N'] for i in common])) if common else None
        seeds[str(seed)] = {'tactile': controllers[f'tactile_seed{seed}'], 'no_tactile': controllers[f'no_tactile_seed{seed}'],
            'jointly_successful_conditions': common, 'paired_tactile_mean_normal_load_N': t_load,
            'paired_no_tactile_mean_normal_load_N': b_load,
            'paired_tactile_load_reduction_fraction': 1-t_load/b_load if b_load and t_load is not None else None}
    integrity_error = None
    try:
        verify_inputs(root, frozen)
    except (OSError, ValueError, KeyError) as exc:
        integrity_error = str(exc)
    summary = {'updated_utc': utc(), 'protocol_sha256': frozen['protocol_sha256'],
        'scope': frozen['protocol'].get('scope', 'full_three_seeds'), 'evaluated_training_seeds': actual_seeds,
        'official_planned': len(jobs), 'official_valid_completed': sum(r['valid_completed'] for r in rows),
        'official_pending': sum(r.get('status') in ('pending', 'not_attempted') for r in rows),
        'official_running': sum(r.get('status') in ('running', 'initializing') for r in rows),
        'all_official_attempted': all(r['attempted'] for r in rows),
        'input_integrity_passed': integrity_error is None, 'input_integrity_error': integrity_error,
        'all_prefixes_exact': audit['all_nine_conditions_complete_and_exact'],
        'all_prefixes_within_tolerance': audit['all_nine_conditions_complete_and_within_tolerance'],
        'prefix_tolerance_abs': audit['absolute_tolerances'],
        'controllers': controllers, 'aggregates': groups, 'seeds': seeds, 'episodes': rows,
        'excluded_from_official_statistics': 'All preflight, smoke, repeat/reference and nonofficial directories.',
        'interpretation': ['Frozen rigid-trained policies; no soft retraining or outcome-based checkpoint selection.',
            'All-valid load averages include failed grasps; use paired common-success comparisons to discuss load economy.',
            'Normal primitive IPC load is distinct from virtual taxel values and from net/complete grip force.',
            'Strict support uses independent normal load; it is not identical to old rigid vector-magnitude support.',
            'Deformation validity is checked at 300 Hz outer physics steps, not each IPC substep.',
            'Prefix tolerances are predeclared engineering gates. Accepted bounded differences remain visible '
            'and do not establish completely identical actor inputs.',
            'Execution/integrity failures remain explicit, are not silently removed or automatically rerun.']}
    summary['completion_status'] = completion_status(summary)
    write(root/'experiment_summary.json', summary)
    return summary


def completion_status(summary):
    if summary.get('official_pending', 0) or summary.get('official_running', 0):
        return 'in_progress'
    if (summary['official_valid_completed'] != summary['official_planned']
            or not summary['input_integrity_passed'] or not summary['all_prefixes_within_tolerance']):
        return 'completed_with_failures_or_audit_findings'
    return 'completed_exact' if summary['all_prefixes_exact'] else 'completed_with_bounded_numeric_differences'


def render_results(root, summary):
    """Publish offline PNG/HTML atomically; presentation cannot fail physics."""
    error = None
    try:
        from . import soft_results
        report = soft_results.normalize(root)
        png = soft_results.chart(report)
        page = soft_results.make_html(report, png, root)
        # Build both outputs before replacing either existing result. The HTML
        # embeds its PNG, so readers never depend on cross-file atomicity.
        for name, content in (('results.png', png), ('results.html', page.encode('utf-8'))):
            target = root/name
            temporary = target.with_name(target.name+f'.{os.getpid()}.tmp')
            temporary.write_bytes(content)
            temporary.replace(target)
    except Exception:
        error = traceback.format_exc()
    summary['report_error'] = error
    write(root/'experiment_summary.json', summary)
    write(root/'report_status.json', {'updated_utc': utc(), 'status': 'render_failed' if error else 'rendered',
                                    'report_error': error, 'official_valid_completed': summary['official_valid_completed']})
    if error:
        print(json.dumps({'event': 'report_error', 'error': error}), flush=True)
    return error


def refresh_reports(root, frozen, jobs, active, *, final=False):
    """One report update for a batch of newly finished child processes."""
    try:
        summary = summarize(root, frozen, jobs)
    except Exception:
        error = traceback.format_exc()
        write(root/'report_status.json', {'updated_utc': utc(), 'status': 'summary_failed', 'report_error': error})
        print(json.dumps({'event': 'report_error', 'error': error}), flush=True)
        return None
    state = summary['completion_status'] if final else 'running'
    # Report rendering uses this fresh active set, never a prior poll's set.
    progress(root, jobs, active, state)
    render_results(root, summary)
    progress(root, jobs, active, state)
    return summary


def run_suite(root, frozen, jobs, workers, resume):
    manifest_path = root/'suite_manifest.json'
    if manifest_path.exists():
        if not resume:
            raise ValueError('Suite already exists; use --resume to continue only unattempted episodes')
        old = read(manifest_path)
        if old['protocol_sha256'] != frozen['protocol_sha256'] or old['jobs'] != jobs:
            raise ValueError('Resume input/schedule differs from the original suite')
    else:
        if any(attempted(j) for j in jobs):
            raise ValueError('Official attempts exist without the suite manifest; refusing to overwrite/mix them')
        snapshot = root/'source_snapshot'
        for rel in frozen['protocol']['frozen_sources']:
            destination = snapshot/rel
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(resolved(rel), destination)
        shutil.copy2(root/'protocol.json', root/'protocol.frozen.json')
        write(manifest_path, {'created_utc': utc(), 'protocol_sha256': frozen['protocol_sha256'],
                             'physics': frozen['physics'], 'initial_workers': workers, 'jobs': jobs,
                             'retry_policy': 'Never automatically retry an existing attempt, including interruptions.'})
    live = [j['id'] for j in jobs if existing_running(j)]
    if live:
        raise RuntimeError(f'Existing child processes are still running; resume after they finish: {live}')
    pending = [j for j in jobs if not attempted(j)]
    active = {}
    last_status = 0.
    try:
        while pending or active:
            while pending and len(active) < workers:
                verify_inputs(root, frozen)
                job = pending.pop(0)
                Path(job['log']).parent.mkdir(parents=True, exist_ok=True)
                attempt = {'id': job['id'], 'started_utc': utc(), 'command': job['command'],
                           'protocol_sha256': frozen['protocol_sha256'], 'returncode': None,
                           'status': 'launching'}
                write(job['attempt'], attempt)
                log = open(job['log'], 'xb')
                try:
                    process = subprocess.Popen(job['command'], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                               start_new_session=True)
                except BaseException:
                    log.close()
                    attempt.update(status='launch_failed', error=traceback.format_exc(), ended_utc=utc())
                    write(job['attempt'], attempt)
                    raise
                attempt.update(status='running', pid=process.pid, process_start_token=process_token(process.pid))
                write(job['attempt'], attempt)
                active[job['id']] = (job, process, log, attempt)
                print(json.dumps({'event': 'started', 'job': job['id'], 'pid': process.pid}), flush=True)
            finished_batch = []
            for key, (job, process, log, attempt) in list(active.items()):
                returncode = process.poll()
                if returncode is None:
                    continue
                log.close()
                attempt.update(status='process_completed' if returncode == 0 else 'process_failed',
                               returncode=returncode, ended_utc=utc())
                write(job['attempt'], attempt)
                del active[key]
                finished_batch.append(key)
                print(json.dumps({'event': 'finished', 'job': key, 'returncode': returncode}), flush=True)
            if finished_batch and (pending or active):
                # Multiple children finishing in this poll trigger one update.
                # The last batch is rendered by the final summary below.
                refresh_reports(root, frozen, jobs, active)
            if time.monotonic()-last_status >= 5:
                progress(root, jobs, active, 'running')
                last_status = time.monotonic()
            if pending or active:
                time.sleep(.5)
        summary = refresh_reports(root, frozen, jobs, {}, final=True)
        state = summary['completion_status'] if summary else 'completed_with_failures_or_audit_findings'
        progress(root, jobs, {}, state)
        print(json.dumps({'event': 'suite_finished', 'status': state,
                          'official_valid_completed': summary['official_valid_completed'] if summary else None,
                          'summary': str(root/'experiment_summary.json')}), flush=True)
        return 0 if state in ('completed_exact', 'completed_with_bounded_numeric_differences') else 2
    except BaseException as exc:
        for job, process, log, attempt in active.values():
            try:
                process.terminate()
            except ProcessLookupError:
                pass
        for job, process, log, attempt in active.values():
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            log.close()
            attempt.update(status='interrupted_preserved', returncode=process.returncode, ended_utc=utc())
            write(job['attempt'], attempt)
        progress(root, jobs, {}, 'interrupted_preserved', error=repr(exc))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--resume', action='store_true')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--validate-only', action='store_true', help='Verify inputs and print the schedule without running physics')
    mode.add_argument('--summarize-only', action='store_true', help='Recompute summaries for existing official attempts')
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    root = args.root.expanduser().resolve()
    frozen = load_protocol(root)
    jobs = make_jobs(root, frozen)
    if args.validate_only:
        print(json.dumps({'status': 'validated', 'official_episodes': len(jobs),
                          'physics': frozen['physics'], 'protocol_sha256': frozen['protocol_sha256'],
                          'first_command': jobs[0]['command']}, indent=2))
        return 0
    with (root/'.soft_suite.lock').open('a+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another suite process currently owns this experiment root')
        if args.summarize_only:
            summary = summarize(root, frozen, jobs)
            print(json.dumps({'official_valid_completed': summary['official_valid_completed'],
                              'summary': str(root/'experiment_summary.json')}))
            return 0
        def interrupted(signum, frame):
            raise KeyboardInterrupt(f'Suite interrupted by signal {signum}')
        previous_handler = signal.signal(signal.SIGTERM, interrupted)
        try:
            return run_suite(root, frozen, jobs, args.workers, args.resume)
        finally:
            signal.signal(signal.SIGTERM, previous_handler)


if __name__ == '__main__':
    raise SystemExit(main())
