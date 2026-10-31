"""Offline trace audit and incomplete-safe report for frozen rigid size transfer.

Reads protocol.json and size_NNNmm/evaluation/<seedN_kind> artifacts. No Genesis,
policy loading, inference, physics, training, or input-artifact edits occur here.
Force efficiency comparisons require paired task success at the same size/seed.
"""
from __future__ import annotations

import argparse
import base64
from collections import Counter
from datetime import datetime, timezone
import hashlib
import html
import io
import json
import os
from pathlib import Path
import re
from urllib.parse import quote

import numpy as np


STATES = {'pending': ('待测', '#e5e9ed'), 'running': ('运行中', '#c5d9f4'),
          'awaiting_audit': ('待产出完整文件', '#d9ccef'), 'invalid': ('审计 / 执行异常', '#f4bf8b'),
          'failed': ('任务未成功', '#eab8b5'), 'task_only': ('任务成功', '#f3dfa0'),
          'strict': ('任务及严格成功', '#a7d9c8')}
LOAD_KEYS = ('sensor_policy_mean_N', 'sensor_hold_mean_N',
             'robot_policy_mean_N', 'robot_hold_mean_N')
DESCRIPTIVE_KEYS = tuple(f'{body}_{metric}' for body in ('sensor', 'robot') for metric in (
    'policy_mean_N', 'hold_mean_N', 'contact_mean_N', 'hold_contact_mean_N',
    'contact_duration_s', 'hold_contact_duration_s'))


def read(path, default=None):
    return json.loads(Path(path).read_text()) if Path(path).is_file() else default


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def identity(controller):
    match = re.fullmatch(r'seed(\d+)_(tactile|no_tactile)', controller)
    if not match:
        raise ValueError(f'Unsupported controller name: {controller}')
    return int(match[1]), match[2]


def label(controller):
    seed, kind = identity(controller)
    return f"种子 {seed} · {'触觉' if kind == 'tactile' else '无触觉'}"


def _lane(value, width=None):
    value = np.asarray(value)
    shape = (len(value), 1) if width is None else (len(value), 1, width)
    if value.shape != shape:
        raise ValueError(f'Expected single-lane shape {shape}, got {value.shape}')
    return value[:, 0]


def trace_metrics(a, size_mm, hold_interval, start_time_s, end_time_s, table_top_m=.2):
    """Recompute labels and clearly separated duration-weighted load summaries."""
    t = np.asarray(a['sample_time_s'], float)
    if t.ndim != 1 or not len(t) or not np.isfinite(t).all():
        raise ValueError('Finite nonempty sample_time_s required')
    expected = start_time_s + np.arange(1, len(t) + 1) / 60
    if not np.allclose(t, expected, rtol=0, atol=1e-9):
        raise ValueError('Trace must use the complete contiguous 60 Hz policy grid')
    if abs(t[-1] - end_time_s) > 1e-7:
        raise ValueError('Trace does not reach the declared policy end time')
    clearance = _lane(a['clearance_m']).astype(float)
    load = _lane(a['contact_load_N'], 5).astype(float)
    robot = _lane(a['all_robot_contact_load_N']).astype(float)
    fault = _lane(a['fault']).astype(bool)
    friction = _lane(a['contact_friction_error']).astype(float)
    pos = _lane(a['cube_position_m'], 3).astype(float)
    quat = _lane(a['cube_quaternion_wxyz'], 4).astype(float)
    for key in a:
        v = np.asarray(a[key])
        if v.dtype.kind in 'fci' and not np.isfinite(v).all():
            raise ValueError(f'Nonfinite trace field: {key}')
    if any(len(v) != len(t) for v in (clearance, load, robot, fault, friction, pos, quat)):
        raise ValueError('Physical trace arrays have inconsistent sample counts')
    sensor = load.sum(-1)
    if (load < -1e-8).any() or (robot < -1e-8).any():
        raise ValueError('Contact magnitudes cannot be negative')
    if np.any(robot + 2e-5 + 2e-6 * np.maximum(robot, sensor) < sensor):
        raise ValueError('All-robot contact load is smaller than its five-link subset')
    if np.max(np.abs(friction)) > 1e-5:
        raise ValueError('Actual contact friction readback differs from requested coefficient')
    if np.max(np.abs(np.linalg.norm(quat, axis=-1) - 1)) > 1e-4:
        raise ValueError('Cube quaternions are not unit wxyz quaternions')
    w, x, y, z = quat.T
    extent = size_mm / 2000 * (np.abs(2*(x*z-w*y)) + np.abs(2*(y*z+w*x)) + np.abs(1-2*(x*x+y*y)))
    independently_computed_clearance = pos[:, 2] - extent - table_top_m
    clearance_error = float(np.max(np.abs(clearance - independently_computed_clearance)))
    if clearance_error > 2e-6:
        raise ValueError('Clearance does not match actual size, cube position and orientation')
    lo, hi = map(float, hold_interval)
    if not np.isclose(hi-lo, 3., rtol=0, atol=1e-7):
        raise ValueError('Frozen task requires the original three-second hold interval')
    weights = np.maximum(0, np.minimum(t, hi) - np.maximum(t-1/60, lo))
    hold = weights > 1e-12
    hold_duration = float(weights.sum())
    if abs(hold_duration - (hi-lo)) > 1e-7 or not hold.any():
        raise ValueError('Trace does not cover the entire declared hold interval')
    support = (load > .05).sum(-1) >= 2
    task = bool((clearance[hold] >= .05).all() and not fault.any())
    strict = bool(task and support[hold].all())
    duration = len(t) / 60
    result = {'task_success': task, 'strict_success': strict, 'policy_duration_s': duration,
              'hold_duration_s': hold_duration, 'hold_min_clearance_m': float(clearance[hold].min()),
              'final_clearance_m': float(clearance[-1]), 'max_clearance_m': float(clearance.max()),
              'collision': bool(fault.any()), 'hold_supported_fraction': float(np.dot(weights, support)/hold_duration),
              'sample_count': len(t), 'max_friction_error': float(np.max(np.abs(friction))),
              'clearance_reconstruction_max_error_m': clearance_error}
    for name, values in (('sensor', sensor), ('robot', robot)):
        contact = values > .05
        contact_duration = float(contact.sum()/60)
        contact_hold_duration = float(np.sum(weights*contact))
        result.update({f'{name}_policy_mean_N': float(values.mean()),
                       f'{name}_hold_mean_N': float(np.dot(weights, values)/hold_duration),
                       f'{name}_contact_mean_N': float(values[contact].mean()) if contact.any() else None,
                       f'{name}_hold_contact_mean_N': float(np.dot(weights*contact, values)/contact_hold_duration) if contact_hold_duration else None,
                       f'{name}_contact_duration_s': contact_duration,
                       f'{name}_hold_contact_duration_s': contact_hold_duration,
                       f'{name}_contact_fraction': contact_duration/duration,
                       f'{name}_peak_N': float(values.max()),
                       f'{name}_policy_integral_Ns': float(values.sum()/60)})
    result['other_robot_policy_mean_N'] = float((robot-sensor).mean())
    result['failure_reason'] = ('collision' if fault.any() else 'hold_below_5cm' if not task
                                else 'contact_discontinuity' if not strict else None)
    gains = np.asarray(a['applied_gains'], float)
    actions = np.asarray(a['actions'], float)
    if gains.shape != (len(t)//4, 1, 2) or actions.shape != (len(t)//4, 1, 1):
        raise ValueError('Expected one common gain command at 15 Hz')
    if (gains < .6-1e-6).any() or (gains > 3+1e-6).any():
        raise ValueError('Applied gain outside frozen [.6, 3] range')
    if not np.allclose(gains[..., 0], gains[..., 1], rtol=0, atol=1e-6):
        raise ValueError('Thumb and index must use a common gain')
    if np.max(np.abs(np.diff(np.r_[1., gains[:, 0, 0]]))) > .2+1e-6:
        raise ValueError('Gain exceeds frozen .2/action rate limit')
    result['gain_mean'] = float(gains.mean())
    return result


def environment_checks(env, size_mm, mass, mu):
    """Check solver readbacks numerically, never trust a verified=True flag."""
    side = size_mm/1000
    initial = np.array([-.6, -.1333, .2+side/2])
    def same(key, expected, atol=1e-7, rtol=0):
        try:
            actual, expected = np.asarray(env[key], float), np.asarray(expected, float)
            return bool(actual.shape == expected.shape and np.isfinite(actual).all()
                        and np.allclose(actual, expected, rtol=rtol, atol=atol))
        except (KeyError, TypeError, ValueError):
            return False
    checks = {
        'cube_requested_size': same('cube_size_m', side),
        'cube_collision_mesh_size_readback': same('cube_size_readback_m', [side]*3),
        'table_height': same('table_top_z_m', .2),
        'requested_mass': same('mass_kg_per_lane', [mass]),
        'solver_mass_readback': same('mass_readback_kg_per_lane', [mass], atol=1e-8, rtol=1e-5),
        'requested_friction': same('friction_per_lane', [mu]),
        'requested_density': same('density_kg_m3_per_lane', [mass/side**3], rtol=1e-5),
        'solver_density_readback': same('density_readback_kg_m3_per_lane', [mass/side**3], rtol=1e-5),
        'solver_inertia_readback': same('inertia_readback_kg_m2_per_lane',
                                       (mass*side**2/6*np.eye(3))[None], atol=1e-9, rtol=1e-4),
        'requested_initial_position': same('cube_initial_position_m', initial),
        'initial_position_readback': same('cube_initial_position_readback_m', initial[None]),
        'initial_bottom_readback': same('cube_initial_bottom_readback_m', [.2]),
        'prefix_audit_enabled': env.get('prefix_physics_audit', {}).get('enabled') is True,
    }
    if checks['cube_collision_mesh_size_readback'] and checks['solver_mass_readback']:
        derived = np.asarray(env['mass_readback_kg_per_lane']) / np.prod(env['cube_size_readback_m'])
        checks['density_equals_measured_mass_over_volume'] = same('density_readback_kg_m3_per_lane', derived, rtol=1e-7)
    return checks


def prefix_metrics(path, env, size_mm):
    """Pre-policy observations are diagnostic; collision does not relabel task."""
    with np.load(path, allow_pickle=False) as a:
        t = np.asarray(a['sample_time_s'], float)
        if t.ndim != 1 or not len(t) or not np.isfinite(t).all():
            raise ValueError('Prefix must contain finite physical samples')
        if not np.allclose(t, np.arange(1,len(t)+1)/60, rtol=0, atol=1e-9):
            raise ValueError('Prefix physical trace is not a complete 60 Hz sequence')
        if abs(t[-1]-env['start_time_s']) > 1e-7:
            raise ValueError('Prefix trace does not end at policy start')
        for key in a:
            value = np.asarray(a[key])
            if value.dtype.kind in 'fci' and not np.isfinite(value).all():
                raise ValueError('Nonfinite prefix field: '+key)
        pos = _lane(a['cube_position_m'],3).astype(float)
        quat = _lane(a['cube_quaternion_wxyz'],4).astype(float)
        clearance = _lane(a['clearance_m']).astype(float)
        fault = _lane(a['fault']).astype(bool)
        load = _lane(a['contact_load_N'],5).astype(float).sum(-1)
        robot = _lane(a['all_robot_contact_load_N']).astype(float)
        friction = _lane(a['contact_friction_error']).astype(float)
        if any(len(x)!=len(t) for x in (pos,quat,clearance,fault,load,robot,friction)):
            raise ValueError('Prefix array lengths differ')
        if np.max(np.abs(np.linalg.norm(quat,axis=-1)-1)) > 1e-4:
            raise ValueError('Prefix cube quaternion is not unit wxyz')
        w,x,y,z = quat.T
        extent = size_mm/2000*(np.abs(2*(x*z-w*y))+np.abs(2*(y*z+w*x))+np.abs(1-2*(x*x+y*y)))
        if np.max(np.abs(clearance-(pos[:,2]-extent-env['table_top_z_m']))) > 2e-6:
            raise ValueError('Prefix clearance reconstruction mismatch')
        if np.max(np.abs(friction)) > 1e-5:
            raise ValueError('Prefix actual contact friction mismatch')
        if np.any(load < -1e-8) or np.any(robot < -1e-8) or np.any(robot+2e-5+2e-6*np.maximum(robot,load)<load):
            raise ValueError('Prefix all-robot/five-link load inconsistency')
        declared = env['prefix_physics_audit']
        if declared.get('samples') != len(t) or declared.get('collision_per_lane') != [bool(fault.any())]:
            raise ValueError('Prefix collision/sample metadata differs from raw trace')
        displacement = pos-np.asarray(env['cube_initial_position_readback_m'])[0]
        return {'samples':len(t), 'duration_s':len(t)/60, 'collision':bool(fault.any()),
                'collision_sample_count':int(fault.sum()),
                'first_collision_time_s':float(t[np.flatnonzero(fault)[0]]) if fault.any() else None,
                'cube_start_position_m':pos[0].tolist(), 'cube_policy_start_position_m':pos[-1].tolist(),
                'cube_policy_start_displacement_m':displacement[-1].tolist(),
                'cube_max_displacement_m':float(np.linalg.norm(displacement,axis=-1).max()),
                'cube_max_xy_displacement_m':float(np.linalg.norm(displacement[:,:2],axis=-1).max()),
                'minimum_clearance_m':float(clearance.min()),
                'sensor_mean_N':float(load.mean()), 'robot_mean_N':float(robot.mean()),
                'sha256':digest(path), 'path':str(path)}


def descriptive(rows):
    """Unpaired all/success subsets are descriptive, never efficiency estimates."""
    valid = [r for r in rows if r['valid_completed']]
    def group(subset):
        return {'episodes':len(subset), 'task_successes':sum(r['metrics']['task_success'] for r in subset),
                'means':{key:{'value':float(np.mean(values)) if values else None, 'defined_episodes':len(values)}
                         for key in DESCRIPTIVE_KEYS
                         for values in [[r['metrics'][key] for r in subset if r['metrics'][key] is not None]]}}
    return {'all_valid_episodes':group(valid),
            'task_successful_episodes':group([r for r in valid if r['metrics']['task_success']]),
            'strict_successful_episodes':group([r for r in valid if r['metrics']['strict_success']])}


def counts(rows):
    valid = [r for r in rows if r['valid_completed']]
    return {'planned': len(rows), 'valid_completed': len(valid),
            'task': sum(r['metrics']['task_success'] for r in valid),
            'strict': sum(r['metrics']['strict_success'] for r in valid),
            **{key: sum(r['state'] == key for r in rows)
               for key in ('pending', 'running', 'awaiting_audit', 'invalid')}}


def paired(rows, criterion='task_success'):
    """No failed/absent loads enter paired efficiency estimates, even if zero."""
    indexed = {(r['size_mm'], r['seed'], r['kind']): r for r in rows}
    pairs = []
    for size, seed in sorted({(r['size_mm'], r['seed']) for r in rows}):
        t, b = (indexed.get((size, seed, k)) for k in ('tactile', 'no_tactile'))
        if not t or not b or not all(r['valid_completed'] and r['metrics'][criterion] for r in (t, b)):
            continue
        values = {key: {'tactile': t['metrics'][key], 'no_tactile': b['metrics'][key],
                        'difference_N': t['metrics'][key]-b['metrics'][key],
                        'relative_reduction': 1-t['metrics'][key]/b['metrics'][key]
                        if b['metrics'][key] > 0 else None} for key in LOAD_KEYS}
        pairs.append({'size_mm': size, 'seed': seed, 'loads': values})
    def aggregate(subset):
        return {'pairs': len(subset), 'conditions': [[x['size_mm'], x['seed']] for x in subset],
                'loads': {key: {'tactile': float(np.mean([x['loads'][key]['tactile'] for x in subset])) if subset else None,
                               'no_tactile': float(np.mean([x['loads'][key]['no_tactile'] for x in subset])) if subset else None,
                               'mean_paired_difference_N': float(np.mean([x['loads'][key]['difference_N'] for x in subset])) if subset else None}
                          for key in LOAD_KEYS}}
    return {'criterion': criterion, 'pairs': pairs, 'aggregate': aggregate(pairs),
            'by_seed': {str(seed): aggregate([x for x in pairs if x['seed']==seed]) for seed in sorted({r['seed'] for r in rows})},
            'by_size': {str(size): aggregate([x for x in pairs if x['size_mm']==size]) for size in sorted({r['size_mm'] for r in rows})}}


def audit(root, synthetic=False):
    root = Path(root).resolve()
    p = read(root/'protocol.json')
    if not isinstance(p, dict):
        raise ValueError('protocol.json required')
    sizes = p['sizes_mm']
    controllers = p.get('controllers') or list(p['models'])
    if not sizes or len(set(sizes)) != len(sizes) or any(not 0 < s < 1000 for s in sizes):
        raise ValueError('Protocol sizes must be positive, distinct millimeters')
    if len(set(controllers)) != len(controllers) or set(controllers) != set(p['models']):
        raise ValueError('Protocol controllers/models differ or are duplicated')
    for name in controllers:
        identity(name)
    if p.get('planned_episodes',len(sizes)*len(controllers)) != len(sizes)*len(controllers):
        raise ValueError('Planned episode count differs from controller/size cross product')
    mass = float(p['mass_kg'])
    mu = float(p['friction'])
    report = {'root': str(root), 'protocol': p, 'protocol_sha256': digest(root/'protocol.json'),
              'synthetic': bool(synthetic or p.get('synthetic')), 'updated_utc': datetime.now(timezone.utc).isoformat(),
              'sizes_mm': sizes, 'controllers': controllers, 'mass_kg': mass, 'friction': mu,
              'rows': [], 'warnings': [], 'size_integrity': {}, 'prefix_by_size': {},
              'environment_readbacks_by_size': {}}
    repository = Path(__file__).resolve().parents[2]
    checks = {}
    for c, model in p['models'].items():
        path = Path(model['path'])
        checks['checkpoint_file_'+c] = path.is_file() and digest(path) == model['sha256']
    for rel, expected in p.get('source_hashes', {}).items():
        path = repository/rel
        checks['source_'+rel] = path.is_file() and digest(path) == expected
        snapshot = root/'source_snapshot'/rel
        checks['source_snapshot_'+rel] = snapshot.is_file() and digest(snapshot) == expected
    for filename, expected in p.get('frozen_input_hashes', {}).items():
        path = Path(filename)
        checks['frozen_input_'+filename] = path.is_file() and digest(path) == expected
    trajectory = p.get('trajectory', {})
    if isinstance(trajectory, dict) and trajectory.get('path'):
        path = Path(trajectory['path'])
        if path.is_dir(): path = path/'control.jsonl'
        checks['trajectory_file'] = path.is_file() and digest(path) == trajectory.get('sha256')
    report['input_integrity_checks'] = checks
    report['input_integrity_passed'] = bool(checks) and all(checks.values())
    run_paths = {row['size_mm']: row['output'] for row in p.get('size_runs', [])}
    if run_paths and set(run_paths) != set(sizes):
        raise ValueError('Protocol size_runs must match sizes_mm exactly')
    for size in sizes:
        base = Path(run_paths.get(size, f'size_{int(size):03d}mm/evaluation'))
        base = base if base.is_absolute() else root/base
        status = read(base/'status.json', {})
        manifest = read(base/'manifest.json', {})
        env = read(base/'environment/environment_protocol.json', {})
        reference = None
        size_checks = {}; model_hashes = manifest.get('models', {})
        if manifest:
            size_checks['training_disabled'] = manifest.get('training') is False
            size_checks['single_material_matches'] = bool(np.asarray(manifest.get('conditions_kg_mu', [])).shape == (1, 2)
                and np.allclose(manifest['conditions_kg_mu'], [[mass, mu]], rtol=0, atol=1e-7))
            for c in controllers:
                size_checks['checkpoint_'+c] = model_hashes.get(c, {}).get('sha256') == p['models'][c]['sha256']
                trained = manifest.get('checkpoint_metadata', {}).get(c, {})
                size_checks['checkpoint_contract_'+c] = (trained.get('training_steps') == 786432
                    and trained.get('uses_tactile') is (identity(c)[1] == 'tactile'))
            for rel, expected in p.get('source_hashes', {}).items():
                if Path(rel).parent == Path('ur7e_xhand_sim/residual_rl'):
                    size_checks['source_'+rel] = manifest.get('source_hashes',{}).get(Path(rel).name) == expected
            expected_hash = p.get('trajectory_sha256') or (p.get('trajectory') or {}).get('sha256')
            if expected_hash:
                size_checks['trajectory_hash'] = manifest.get('trajectory', {}).get('sha256') == expected_hash
            if 'config' in p:
                size_checks['frozen_config_manifest'] = manifest.get('config') == p['config']
                size_checks['frozen_config_environment'] = env.get('config') == p['config']
            size_checks['manifest_scope'] = set(model_hashes) == set(controllers)
            if 'planned_episodes' in p:
                size_checks['models_only'] = (manifest.get('models_only') is True and
                                              not manifest.get('fixed_gains_requested'))
                size_checks['manifest_size'] = manifest.get('cube_size_m') == size/1000
        if env:
            size_checks.update(environment_checks(env, size, mass, mu))
            report['environment_readbacks_by_size'][str(size)] = {key:env.get(key) for key in (
                'cube_size_readback_m','mass_readback_kg_per_lane','density_readback_kg_m3_per_lane',
                'inertia_readback_kg_m2_per_lane','cube_initial_bottom_readback_m')}
            if trajectory.get('phase_starts') and trajectory.get('duration_s'):
                phases,duration=trajectory['phase_starts'],trajectory['duration_s']
                size_checks['original_trajectory_timing'] = bool(
                    np.allclose(env.get('hold_interval_s',[0,0]),[phases['hold'],duration],rtol=0,atol=1e-8)
                    and np.isclose(env.get('start_time_s',-1),int(phases['preload']*15)/15,rtol=0,atol=1e-8)
                    and np.isclose(env.get('end_physics_time_s',-1),np.ceil(duration*15)/15,rtol=0,atol=1e-8))
                size_checks['environment_trajectory_hash'] = env.get('demonstration_sha256') == trajectory['sha256']
            prefix_path = base/'environment/prefix_physics_trace.npz'
            try:
                prefix = prefix_metrics(prefix_path, env, size)
                report['prefix_by_size'][str(size)] = {'valid':True, **prefix}
                size_checks['prefix_physics_trace'] = True
                if prefix['collision']:
                    report['warnings'].append(f'{size} mm 的共同前缀发生机器人碰撞；单独报告，不修改原任务标签。')
            except (OSError, KeyError, ValueError, TypeError) as exc:
                report['prefix_by_size'][str(size)] = {'valid':False, 'error':str(exc), 'path':str(prefix_path)}
                size_checks['prefix_physics_trace'] = False
        report['size_integrity'][str(size)] = size_checks
        for c in controllers:
            seed, kind = identity(c); out = base/c
            row = {'size_mm': size, 'seed': seed, 'kind': kind, 'controller': c, 'output': str(out),
                   'state': 'pending', 'valid_completed': False, 'errors': [], 'metrics': None}
            report['rows'].append(row)
            if not (out/'evaluation.json').exists() or not (out/'trace.npz').exists():
                row['state'] = ('invalid' if status.get('status') == 'failed' else 'awaiting_audit' if out.exists()
                                else 'running' if status.get('status') in ('running', 'initializing') else 'pending')
                if row['state']=='invalid': row['errors'].append(status.get('error', 'Benchmark failed before this controller completed'))
                continue
            try:
                evaluation = read(out/'evaluation.json')
                if evaluation.get('controller') != c or len(evaluation.get('episodes', [])) != 1:
                    raise ValueError('Evaluation must identify the controller and exactly one lane')
                if not manifest or not env or not all(size_checks.values()) or not report['input_integrity_passed']:
                    failed = [k for k,v in size_checks.items() if not v]
                    raise ValueError('Missing or mismatched protocol/manifest/environment metadata: '+', '.join(failed))
                original = evaluation['episodes'][0]
                with np.load(out/'trace.npz', allow_pickle=False) as a:
                    metrics = trace_metrics(a, size, env['hold_interval_s'], env['start_time_s'],
                                            env['end_physics_time_s'], env.get('table_top_z_m', .2))
                    prefix = {key: np.asarray(a[key]).copy() for key in ('prefix_state', 'prefix_tactile')}
                    if any(not np.isfinite(v).all() for v in prefix.values()):
                        raise ValueError('Nonfinite policy prefix')
                    if reference is None:
                        reference = prefix
                    exact = {key: bool(prefix[key].shape == reference[key].shape and np.array_equal(prefix[key], reference[key])) for key in prefix}
                    row['prefix_exact'] = exact
                    if not all(exact.values()):
                        raise ValueError('Controllers do not start from identical prefix observation history')
                for key in ('task_success', 'strict_success'):
                    if type(original.get(key)) is not bool or original[key] != metrics[key]:
                        raise ValueError('Recomputed outcome differs: '+key)
                for key, value in (('mass_kg',mass),('friction',mu)):
                    if not np.isclose(original[key], value, rtol=1e-5, atol=1e-7):
                        raise ValueError('Episode material differs from protocol: '+key)
                for old, new in (('mean_contact_force_N','sensor_policy_mean_N'),
                                 ('hold_min_clearance_m','hold_min_clearance_m'), ('hold_duration_s','hold_duration_s')):
                    if not np.isclose(original[old], metrics[new], rtol=2e-6, atol=2e-7):
                        raise ValueError('Recomputed metric differs: '+old)
                row.update(valid_completed=True, metrics=metrics, original_episode=original,
                           trace_sha256=digest(out/'trace.npz'), evaluation_sha256=digest(out/'evaluation.json'),
                           state='strict' if metrics['strict_success'] else 'task_only' if metrics['task_success'] else 'failed')
            except (OSError, ValueError, KeyError, TypeError) as exc:
                row['state']='invalid'; row['errors'].append(str(exc))
    report['counts'] = counts(report['rows'])
    report['by_controller'] = {c: counts([r for r in report['rows'] if r['controller']==c]) for c in controllers}
    report['by_size'] = {str(s): counts([r for r in report['rows'] if r['size_mm']==s]) for s in sizes}
    report['descriptive_loads'] = descriptive(report['rows'])
    report['descriptive_loads_by_controller'] = {c:descriptive([r for r in report['rows'] if r['controller']==c]) for c in controllers}
    report['paired_task_success'] = paired(report['rows'])
    report['paired_strict_success'] = paired(report['rows'], 'strict_success')
    totals = report['counts']
    if not report['input_integrity_passed']:
        report['warnings'].append('冻结模型、轨迹或源码完整性尚未通过，详情见 input_integrity_checks。')
    report['status'] = ('已完成' if totals['valid_completed']==totals['planned'] and report['input_integrity_passed'] else
                        '含审计 / 执行异常' if totals['invalid'] else '运行中 / 部分结果' if totals['valid_completed'] else '尚无完整结果')
    return report


def chart(report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    font = Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams['font.family'] = font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({'axes.unicode_minus':False, 'font.size':10})
    cs, sizes = report['controllers'], report['sizes_mm']
    indexed = {(r['controller'],r['size_mm']):r for r in report['rows']};keys=list(STATES)
    fig=plt.figure(figsize=(14,10),facecolor='#f5f7fa')
    grid=fig.add_gridspec(2,2,left=.15,right=.95,bottom=.12,top=.80,height_ratios=[1.25,1],hspace=.65,wspace=.45)
    ax=fig.add_subplot(grid[0,:]); data=np.array([[keys.index(indexed[c,s]['state']) for s in sizes] for c in cs])
    ax.imshow(data,cmap=ListedColormap([STATES[k][1] for k in keys]),vmin=-.5,vmax=len(keys)-.5,aspect='auto')
    ax.set_xticks(range(len(sizes)),[f'{s:g} mm' for s in sizes]);ax.set_yticks(range(len(cs)),[label(c) for c in cs])
    for i,c in enumerate(cs):
        for j,s in enumerate(sizes):
            r=indexed[c,s];text=(f"{int(r['metrics']['task_success'])} / {int(r['metrics']['strict_success'])}" if r['valid_completed'] else STATES[r['state']][0])
            ax.text(j,i,text,ha='center',va='center',fontsize=9)
    ax.set_title('各尺寸与训练种子：任务 / 严格成功（1 或 0）',loc='left',pad=13,fontweight='bold')
    ax.legend(handles=[Patch(color=v[1],label=v[0]) for v in STATES.values()],bbox_to_anchor=(.5,-.15),loc='upper center',ncol=4,frameon=False,fontsize=8)
    seeds=sorted({r['seed'] for r in report['rows']})
    for k,key,title in [(0,'sensor_policy_mean_N','策略控制时段 · 五个传感指节接触载荷'),(1,'robot_hold_mean_N','保持时段 · 全机器人接触载荷')]:
        ax=fig.add_subplot(grid[1,k]); ax.set_facecolor('#f5f7fa');xs=np.arange(len(seeds)); pairs=report['paired_task_success']['by_seed']
        for sign,kind,color,name in [(-1,'tactile','#3478a5','触觉'),(1,'no_tactile','#bd7e3e','无触觉')]:
            ys=[pairs[str(s)]['loads'][key][kind] for s in seeds]
            ax.bar(xs+sign*.17,[np.nan if y is None else y for y in ys],.32,color=color,label=name)
        ax.set_xticks(xs,[f'种子 {s}\n共同成功 {pairs[str(s)]["pairs"]}/{len(sizes)}' for s in seeds]);ax.set_ylabel('N（接触力模长之和）')
        ax.set_title(title,fontsize=11,loc='left',pad=12);ax.legend(frameon=False);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
        ax.margins(y=.25)
        if not any(pairs[str(s)]['pairs'] for s in seeds):
            ax.text(.5,.5,'尚无共同成功配对\n不使用失败后的零载荷',transform=ax.transAxes,ha='center',va='center');ax.set_ylim(0,1)
    total=report['counts']; title=('SYNTHETIC · 合成报告测试\n' if report['synthetic'] else '')+'刚体尺寸迁移 · 固定规则示范 + 冻结 PPO'
    fig.suptitle(title,x=.04,y=.97,ha='left',fontsize=18,fontweight='bold')
    fig.text(.04,.88,f"{report['status']}  |  完成 {total['valid_completed']}/{total['planned']}  |  质量 {report['mass_kg']*1000:g} g  |  μ = {report['friction']:g}",fontsize=11)
    fig.text(.04,.04,'力比较仅包含同尺寸、同训练种子的共同任务成功回合；失败、缺失和审计异常不视为低力优势。\n每个尺寸每个模型一次确定性回放；训练种子不是新的独立物体样本。',fontsize=9,color='#445566')
    out=io.BytesIO();fig.savefig(out,format='png',dpi=150);plt.close(fig);return out.getvalue()


def _fmt(value, digits=3):
    return '—' if value is None else f'{value:.{digits}f}'


def make_html(report, png, output_dir):
    esc=lambda x:html.escape(str(x),quote=True)
    root=Path(report['root']);outdir=Path(output_dir)
    def link(path,name):
        path=Path(path)
        if not path.exists():return f'<span class="muted">{esc(name)}（未提供）</span>'
        href=quote(os.path.relpath(path.resolve(),outdir.resolve()),safe='/')
        return f'<a href="{esc(href)}">{esc(name)}</a>'
    rows=[]
    for r in report['rows']:
        m=r['metrics'] or {};state=r['state'];reason='；'.join(r['errors']) if r['errors'] else m.get('failure_reason') or '—'
        reason={'hold_below_5cm':'保持高度未达到 5 cm','collision':'机器人碰撞','contact_discontinuity':'任务成功，但两指节支撑不连续'}.get(reason,reason)
        o=Path(r['output']); artifacts=' · '.join([link(o/'evaluation.json','评分'),link(o/'trace.npz','轨迹'),link(o.parent/'environment/environment_protocol.json','环境'),link(o/'prefix_consistency.json','前缀')])
        row=[str(r['size_mm']),esc(label(r['controller'])),f'<span class="pill" style="background:{STATES[state][1]}">{STATES[state][0]}</span>',esc(reason),
             _fmt(m.get('hold_min_clearance_m')), *[_fmt(m.get(k)) for k in ('sensor_policy_mean_N','sensor_hold_mean_N','sensor_contact_mean_N','sensor_contact_duration_s','robot_policy_mean_N','robot_hold_mean_N','other_robot_policy_mean_N')],artifacts]
        rows.append('<tr>'+''.join(f'<td>{v}</td>' for v in row)+'</tr>')
    summaries=[]
    for c,s in report['by_controller'].items():
        summaries.append(f'<tr><td>{esc(label(c))}</td><td>{s["valid_completed"]}/{s["planned"]}</td><td>{s["task"]}</td><td>{s["strict"]}</td><td>{s["invalid"]}</td></tr>')
    pair_rows=[]
    for seed,s in report['paired_task_success']['by_seed'].items():
        cols=[seed,str(s['pairs']),esc(', '.join(str(v[0]) for v in s['conditions']) or '无')]
        for key in LOAD_KEYS:
            d=s['loads'][key];cols.append(f'{_fmt(d["tactile"])} / {_fmt(d["no_tactile"])}')
        pair_rows.append('<tr>'+''.join(f'<td>{v}</td>' for v in cols)+'</tr>')
    diagnostic_rows=[]
    for c,groups in report['descriptive_loads_by_controller'].items():
        for key,name in [('all_valid_episodes','全部有效回合'),('task_successful_episodes','该模型任务成功回合')]:
            g=groups[key];cols=[esc(label(c)),name,str(g['episodes'])]
            for metric in (*LOAD_KEYS,'sensor_contact_mean_N','robot_contact_mean_N'):
                item=g['means'][metric]
                cols.append(_fmt(item['value'])+f' (n={item["defined_episodes"]})')
            diagnostic_rows.append('<tr>'+''.join(f'<td>{v}</td>' for v in cols)+'</tr>')
    prefix_rows=[]
    for size in report['sizes_mm']:
        pre=report['prefix_by_size'].get(str(size),{});env=report['environment_readbacks_by_size'].get(str(size),{})
        def scalar(key):
            value=env.get(key)
            return value[0] if value is not None else None
        collision=('是' if pre['collision'] else '否') if pre.get('valid') else '—'
        moved=pre.get('cube_policy_start_displacement_m')
        cols=[str(size),_fmt(scalar('mass_readback_kg_per_lane'),6),_fmt(scalar('density_readback_kg_m3_per_lane'),2),
              _fmt(scalar('cube_initial_bottom_readback_m'),6),collision,
              str(pre.get('collision_sample_count','—')),_fmt(pre.get('first_collision_time_s')),
              _fmt(pre.get('cube_max_xy_displacement_m'),6),
              ' / '.join(_fmt(v,6) for v in moved) if moved is not None else '—',
              esc(pre.get('error','通过' if pre.get('valid') else '未完成')),
              link(pre.get('path',root/'missing-prefix'),'前缀物理轨迹')]
        prefix_rows.append('<tr>'+''.join(f'<td>{v}</td>' for v in cols)+'</tr>')
    model_rows=''.join(f'<tr><td>{esc(label(c))}</td><td>{link(Path(v["path"]),"冻结模型")}</td><td class="hash">{esc(v["sha256"])}</td></tr>' for c,v in report['protocol']['models'].items())
    watermark='<div class="synthetic">SYNTHETIC · 仅为合成数据渲染测试，不是实验结果</div>' if report['synthetic'] else ''
    counts_=report['counts']
    return f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>刚体尺寸迁移结果</title><style>
body{{margin:0;background:#f5f7fa;color:#183044;font:15px/1.65 system-ui,"Noto Sans CJK SC",sans-serif}}main{{max-width:1500px;margin:30px auto;padding:0 22px}}section{{background:white;border:1px solid #dce3e8;border-radius:12px;padding:22px;margin:20px 0}}h1{{font-size:28px}}h2{{font-size:20px}}.scroll{{overflow:auto}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{padding:9px 12px;border-bottom:1px solid #e2e8ed;text-align:left;white-space:nowrap}}th{{background:#eef3f6}}.pill{{padding:3px 8px;border-radius:5px}}.muted,.foot{{color:#607383;font-size:13px}}.hash{{font:11px monospace}}.synthetic{{padding:16px;background:#ffe5a1;font-weight:bold}}a{{color:#176ca0}}img{{width:100%;height:auto}}.note{{border-left:4px solid #bb8137;padding:12px;background:#fff5e5}}</style><main>{watermark}
<h1>刚体尺寸迁移 · 固定规则示范与冻结触觉 PPO</h1><p>{esc(report['status'])} · 有效完成 {counts_['valid_completed']}/{counts_['planned']} · 审计 / 执行异常 {counts_['invalid']} · 质量 {report['mass_kg']*1000:g} g · μ={report['friction']:g}</p>
<p>尺寸：{esc(' / '.join(map(str,report['sizes_mm'])))} mm。每个尺寸分别使用三个训练种子的触觉与无触觉模型；不重训，不更换基础示范，不按结果选择模型。</p>
<section><img alt="尺寸与种子结果矩阵、共同成功配对载荷" src="data:image/png;base64,{base64.b64encode(png).decode()}"></section>
<section><h2>成功数与完成情况</h2><p>成功数只计入已完成且通过独立审计的回合。未完成、运行中和审计异常不当作抓取失败。</p><div class="scroll"><table><tr><th>模型</th><th>有效 / 计划</th><th>任务成功数</th><th>严格成功数</th><th>审计 / 执行异常</th></tr>{''.join(summaries)}</table></div></section>
<section><h2>共同成功的配对载荷</h2><p class="note">仅比较同一尺寸、同一训练种子下两模型都任务成功的回合。掉落后的低载荷不能证明省力。没有共同成功时显示“—”，不补零。以下每格为触觉 / 无触觉（N）；三种子均分别报告，汇总采用配对等权均值。</p><div class="scroll"><table><tr><th>种子</th><th>共同成功对数</th><th>尺寸 mm</th><th>五指节 · 策略时段</th><th>五指节 · 保持时段</th><th>全机器人 · 策略时段</th><th>全机器人 · 保持时段</th></tr>{''.join(pair_rows)}</table></div><p>共同严格成功配对、每尺寸配对及逐对差值也保存在 {link(outdir/'size_audit.json','独立审计 JSON')}。配对筛选只回答成功抓持时的载荷差异；不能替代全部回合的成功率。</p></section>
<section><h2>载荷描述统计：全部回合与各自成功子集</h2><p>这是诊断性描述，不是效率比较：两模型各自成功子集可能包含不同尺寸；全部回合均值可能被掉落后的零载荷压低。每项显示均值与有定义的回合数 n；无接触时有接触均值无定义，不补零。保持阶段有接触均值、接触时长及严格成功子集也在审计 JSON 中。</p><div class="scroll"><table><tr><th>模型</th><th>子集</th><th>回合数</th><th>五指节 · 策略 N</th><th>五指节 · 保持 N</th><th>全机器人 · 策略 N</th><th>全机器人 · 保持 N</th><th>五指节 · 有接触 N</th><th>全机器人 · 有接触 N</th></tr>{''.join(diagnostic_rows)}</table></div></section>
<section><h2>真实参数读回与策略接管前的共同前缀</h2><p>每个尺寸单独建场景，在策略接管前完整重放同一规则示范前缀。表中质量和密度来自求解器质量与碰撞网格尺寸读回；三轴边长、完整惯量矩阵和初始位置逐值核验保存在审计 JSON。前缀碰撞与位移是尺寸迁移结果的一部分，按尺寸记录一次，六个模型共享该物理起点。<strong>前缀碰撞独立展示，不偷偷修改原任务成功定义，也不删除相应回合。</strong>位移以建场景时的初始位置为基准，单位 m；首次碰撞为原示范时间 s。</p><div class="scroll"><table><tr><th>边长 mm</th><th>实测质量 kg</th><th>导出密度 kg/m³</th><th>初始底面 m</th><th>前缀碰撞</th><th>碰撞采样数</th><th>首次碰撞 s</th><th>最大平面位移 m</th><th>接管时位移 x / y / z</th><th>前缀审计</th><th>原始数据</th></tr>{''.join(prefix_rows)}</table></div></section>
<section><h2>各回合完整记录</h2><p>失败回合的力均值仍原样展示供诊断，不计入效率优势。接触条件均值必须连同接触时长阅读。净空单位 m，力单位 N，时长单位 s。</p><div class="scroll"><table><tr><th>边长 mm</th><th>模型</th><th>状态</th><th>原因</th><th>保持最低净空</th><th>五指节策略均值</th><th>五指节保持均值</th><th>五指节有接触均值</th><th>五指节接触时长</th><th>全机器人策略均值</th><th>全机器人保持均值</th><th>其他部件策略均值</th><th>原始数据</th></tr>{''.join(rows)}</table></div></section>
<section><h2>指标定义与解释边界</h2><p><strong>任务成功：</strong>原示范最后三秒的所有 60 Hz 保持采样，真实方块底部距桌面至少 5 cm，且策略控制时段无机器人碰撞。严格成功额外要求每个保持采样至少两个传感指节的物理接触载荷大于 0.05 N。用方块实际尺寸、姿态与位置独立重算底部净空。</p><p><strong>原有五指节载荷：</strong>方块与五个传感指节间每个物理接触力模长的总和；这是求解器接触量，包含切向成分，不是法向力，不是合力模长，也不是虚拟 taxel 估计。<strong>全机器人载荷：</strong>方块与任意机器人部件的接触力模长之和，排除桌面、地面与自碰撞；用于识别尺寸改变后非传感部位提供的支撑。</p><p><strong>策略时段均值：</strong>预紧策略启动到示范结束，包括零接触和掉落后的所有 60 Hz 样本；与原 mean_contact_force_N 保持同一口径。它不是完整接近前缀的均值。<strong>保持时段均值：</strong>在原最后三秒内按采样区间与保持区间的重叠时长加权，仍包含无接触。<strong>有接触均值：</strong>只取对应载荷总和大于 0.05 N 的样本，同时记录有效时长；这是描述量，不能单独作为优越性判断。</p><p>改变尺寸时保持质量与接触摩擦，因此密度和转动惯量应相应变化。本轮只检验冻结刚体模型的尺寸迁移；质量、摩擦和尺寸不会作为真实物体标签输入策略。策略仅调整共同夹持 PD 增益，没有主动重规划手形或改写基础轨迹。每尺寸每模型一次确定性回放；三个种子是三个训练结果，不足以宣称普遍显著优势或实机可靠性。</p><p class="foot">当前审计核查：60 Hz 完整采样、有限数组、实际姿态净空、实际摩擦误差、全机器人载荷包含五指节、增益范围与速率、评分重算、模型/轨迹声明哈希、同尺寸的共同前缀精确一致。尚不把虚拟触觉当成硬件标定，也不从平均载荷推断滑移因果。</p></section>
<section><h2>冻结来源与审计</h2><p>{link(root/'protocol.json','实验协议')} · {link(outdir/'size_audit.json','完整独立审计')}</p><p>{esc('；'.join(report['warnings']) or '当前无额外审计提示。')}</p><div class="scroll"><table><tr><th>模型</th><th>来源</th><th>SHA-256</th></tr>{model_rows}</table></div><p class="foot">生成时间 {esc(report['updated_utc'])}。页面图片与样式内嵌；原始数据链接依赖本地目录结构。报告只读输入，未运行物理或模型。</p></section></main></html>'''


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--synthetic',action='store_true')
    parser.add_argument('--final',action='store_true',help='Exit nonzero after reporting incomplete runs or audit errors')
    args=parser.parse_args();root=args.root.expanduser().resolve();out=(args.output_dir or root).expanduser().resolve()
    report=audit(root,args.synthetic);out.mkdir(parents=True,exist_ok=True)
    png=chart(report);page=make_html(report,png,out)
    report_bytes=(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode()
    for name,payload in [('size_audit.json',report_bytes),('experiment_summary.json',report_bytes),
                         ('results.png',png),('results.html',page.encode())]:
        temp=out/f'.{name}.{os.getpid()}.tmp';temp.write_bytes(payload);temp.replace(out/name)
    print(json.dumps({'status':report['status'],'counts':report['counts'],'output':str(out)},ensure_ascii=False))
    if args.final and (report['counts']['valid_completed'] != report['counts']['planned'] or not report['input_integrity_passed']):
        raise SystemExit(1)


if __name__=='__main__':main()
