"""Offline soft-body sensor/actor audit; never initialize or step Genesis.

Example::
    python -m ur7e_xhand_sim.residual_rl.soft_audit --run RUN --output NEW_DIR

The output directory must not exist, preserving earlier failed evidence. A
passed audit establishes the recorded bridge's coverage and input contract for
this run, not force calibration, real-material validity, or PPO superiority.
``--self-test`` runs CPU-only tests of the audit's decision and geometry helpers.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import trimesh

from .core import Trajectory, interval_overlap
from .soft_taxel import SoftTaxelAdapter


ROOT = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_safe(value):
    """Keep invalid numeric evidence explicit without emitting nonstandard JSON."""
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def quantiles(value):
    x = np.asarray(value).reshape(-1)
    x = x[np.isfinite(x)]
    if not len(x):
        return None
    return dict(zip(('min', 'p05', 'median', 'p95', 'max'),
                    map(float, np.quantile(x, [0, .05, .5, .95, 1]))))


def intervals(mask, time):
    ids = np.flatnonzero(mask)
    if not len(ids):
        return []
    blocks = np.split(ids, np.flatnonzero(np.diff(ids) > 1) + 1)
    return [{'first_sample_s': float(time[b[0]]), 'last_sample_s': float(time[b[-1]]),
             'sample_count': int(len(b))} for b in blocks]


def pose_diagnostics(arrays, names):
    """IPC ABD transforms can contain affine strain: do not call all of it rotation."""
    if 'finger_ipc_transforms' not in arrays:
        return {'available': False,
                'uncertainty': 'Actual IPC ABD transforms were not saved; dynamic proxy alignment cannot be inferred.'}
    transform = arrays['finger_ipc_transforms']
    if transform.shape != (len(arrays['time_s']), 5, 4, 4):
        return {'available': True, 'valid_shape': False, 'finite': bool(np.isfinite(transform).all())}
    finite = bool(np.isfinite(transform).all())
    if not finite:
        return {'available': True, 'valid_shape': True, 'finite': False}
    linear = transform[..., :3, :3]
    rotation = arrays['finger_rotations']
    position_delta = np.linalg.norm(transform[..., :3, 3] - arrays['finger_positions'], axis=-1)
    matrix_delta = np.linalg.norm(linear - rotation, axis=(-2, -1))
    orthogonal_error = np.linalg.norm(np.swapaxes(linear, -1, -2) @ linear - np.eye(3), axis=(-2, -1))
    u, singular, vh = np.linalg.svd(linear)
    proper = u @ vh
    reflected = np.linalg.det(proper) < 0
    u[reflected, :, -1] *= -1
    proper = u @ vh
    relative = np.swapaxes(rotation, -1, -2) @ proper
    angle = np.rad2deg(np.arccos(np.clip((np.trace(relative, axis1=-2, axis2=-1) - 1) / 2, -1, 1)))
    per_finger = {}
    for f, name in enumerate(names):
        per_finger[name] = {
            'position_difference_mm': quantiles(position_delta[:, f] * 1000),
            'linear_matrix_difference_Frobenius': quantiles(matrix_delta[:, f]),
            'IPC_orthogonality_error_Frobenius': quantiles(orthogonal_error[:, f]),
            'IPC_determinant': quantiles(np.linalg.det(linear[:, f])),
            'IPC_singular_values': quantiles(singular[:, f]),
            'nearest_proper_rotation_difference_deg': quantiles(angle[:, f]),
            'position_difference_when_contact_positive_mm': quantiles(
                position_delta[arrays['contact_normal_load_N'][:, f] > .05, f] * 1000),
        }
    return {'available': True, 'valid_shape': True, 'finite': True, 'per_finger': per_finger,
            'interpretation': 'Position compares origins. Matrix error includes affine strain; the angle uses the nearest proper rotation from polar decomposition. No assumption that the IPC ABD matrix is exactly orthogonal.'}


def observation_contract(arrays, manifest, scene):
    required = ('state', 'q', 'qd', 'gain', 'actions', 'state_before', 'tactile_before')
    missing = [key for key in required if key not in arrays]
    if missing:
        return {'checked': False, 'passed': False, 'missing_fields': missing}
    trajectory_path = Path(manifest['trajectory'])
    model_path = Path(scene['model_path'])
    if not trajectory_path.exists() or not model_path.exists():
        return {'checked': False, 'passed': False,
                'uncertainty': 'Trajectory or authored robot joint-limit file is unavailable.'}
    trajectory = Trajectory(trajectory_path)
    xml = ET.parse(model_path).getroot()
    bounds = [[float(xml.find(f"joint[@name='{name}']/limit").get(key))
               for name in scene['joint_order']] for key in ('lower', 'upper')]
    low, high = np.asarray(bounds, np.float32)
    mid, half = (low + high) / 2, (high - low) / 2
    time = arrays['time_s']
    start, end = manifest['start_tick'], manifest['planned_end_tick']
    expected = []
    for i, now in enumerate(time):
        nominal = trajectory.target(now)
        phase = trajectory.phase(now)
        expected.append(np.r_[(arrays['q'][i] - mid) / half,
            np.clip(arrays['qd'][i] / 3, -10, 10), (nominal - arrays['q'][i]) / .2,
            (trajectory.target(now + 1 / 15) - nominal) / .1,
            arrays['gain'][i] - 1, arrays['gain'][i] - 1,
            [phase == x for x in ('preload', 'test_lift', 'lift', 'hold')],
            (round(now * 300) - start) / (end - start)].astype(np.float32))
    state_error = float(np.max(np.abs(np.asarray(expected) - arrays['state'])))
    differences, time_errors = [], []
    for k, action in enumerate(arrays['actions']):
        last = int(np.argmin(np.abs(time - action[0])))
        ids = np.arange(last - 11, last + 1)
        if ids[0] < 0:
            return {'checked': False, 'passed': False, 'uncertainty': 'Saved trace omits actor history before the first action.'}
        differences.append([float(np.max(np.abs(arrays['state_before'][k] - arrays['state'][ids]))),
                            float(np.max(np.abs(arrays['tactile_before'][k] - arrays['tactile'][ids])))])
        time_errors.append(abs(float(time[last] - action[0])))
    errors = np.max(differences, axis=0) if differences else None
    shapes = (tuple(arrays['state_before'].shape[1:]) == (12, 79)
              and tuple(arrays['tactile_before'].shape[1:]) == (12, 5, 120, 3))
    observed_actions = len(differences) > 0
    passed = bool(shapes and observed_actions and state_error <= 2e-6
                  and np.max(errors) <= 2e-6 and max(time_errors) <= 1e-8
                  and trajectory.sha256 == manifest['trajectory_sha256'])
    return {'checked': True, 'passed': passed, 'state_shape': list(arrays['state_before'].shape[1:]),
            'tactile_shape': list(arrays['tactile_before'].shape[1:]),
            'state_formula_max_error': state_error,
            'history_max_error_state_tactile': None if errors is None else errors.tolist(),
            'action_history_time_max_error_s': max(time_errors) if time_errors else None,
            'observed_actor_decisions': int(len(differences)),
            'trajectory_hash_matches': trajectory.sha256 == manifest['trajectory_sha256'],
            'object_or_IPC_truth_in_actor_input': False if passed else None,
            'observation_fields': '12 frames of normalized joint positions, clipped joint velocity, nominal tracking error, nominal future delta, two common gains, four phases, progress; plus link-frame 5x120x3 virtual touch.'}


def geometry_check(arrays, boundary, layouts, sample_budget):
    time = arrays['time_s']
    positive = arrays['contact_normal_load_N'] > .05
    signal = np.linalg.norm(arrays['tactile'], axis=-1).max(-1) > 1e-5
    engaged = np.flatnonzero(positive.any(-1) | signal.any(-1))
    start = int(engaged[0]) if len(engaged) else 0
    ids = set(np.linspace(start, len(time) - 1, sample_budget, dtype=int).tolist())
    ids.update([0, len(time) - 1])
    for f in range(5):
        for mask in (positive[:, f], signal[:, f], positive[:, f] & ~signal[:, f]):
            x = np.flatnonzero(mask)
            if len(x):
                ids.update([int(x[0]), int(x[-1])])
    raw_positions = np.asarray([layout['positions_link_m'] for layout in layouts])
    selected_fingers = np.unique(np.r_[0, 1, np.flatnonzero(positive.any(0) | signal.any(0))])
    rows, failures = [], []
    has_mask = 'sensor_contact_mask' in arrays
    for i in sorted(ids):
        adapter = SoftTaxelAdapter(layouts)
        try:
            if i:
                adapter.sample(arrays['vertices'][i-1], boundary, arrays['finger_positions'][i-1],
                    arrays['finger_rotations'][i-1],
                    contact_mask=arrays['sensor_contact_mask'][i-1] if has_mask else None)
            result = adapter.sample(arrays['vertices'][i], boundary, arrays['finger_positions'][i],
                arrays['finger_rotations'][i], dt=float(time[i]-time[i-1]) if i else 1/60,
                contact_mask=arrays['sensor_contact_mask'][i] if has_mask else None)
            mesh = trimesh.Trimesh(arrays['vertices'][i], boundary, process=False)
            probes = np.einsum('fij,fpj->fpi', arrays['finger_rotations'][i], raw_positions) + arrays['finger_positions'][i, :, None, :]
            selected = probes[selected_fingers]
            # Independent full nearest-surface reference, without adapter pruning.
            reference = -trimesh.proximity.signed_distance(mesh, selected.reshape(-1, 3)).reshape(-1, 120)
            queried = result['queried'][selected_fingers]
            distance_error = float(np.max(np.abs(result['signed_distance_m'][selected_fingers][queried] - reference[queried]))) if queried.any() else 0.
            minimum = [None] * 5
            for k, finger in enumerate(selected_fingers):
                minimum[int(finger)] = float(reference[k].min() * 1000)
            rows.append({'sample_index': int(i), 'time_s': float(time[i]),
                'minimum_probe_surface_signed_distance_mm': minimum,
                'IPC_normal_load_N': arrays['contact_normal_load_N'][i].tolist(),
                'saved_contact_mask': arrays['sensor_contact_mask'][i].tolist() if has_mask else None,
                'active_taxels_per_finger': result['active'].sum(1).tolist(),
                'max_difference_to_saved_force_N': float(np.max(np.abs(result['force_link_N'] - arrays['tactile'][i]))),
                'max_distance_difference_from_independent_trimesh_m': distance_error})
        except Exception as exc:
            failures.append({'sample_index': int(i), 'time_s': float(time[i]), 'error': repr(exc)})
    max_force = max((r['max_difference_to_saved_force_N'] for r in rows), default=None)
    max_distance = max((r['max_distance_difference_from_independent_trimesh_m'] for r in rows), default=None)
    return {'passed': bool(rows and not failures and max_force <= 2e-6 and max_distance <= 1e-7),
            'replayed_saved_contact_mask': has_mask, 'requested_regular_samples': sample_budget,
            'actual_sample_count': len(rows), 'max_force_replay_difference_N': max_force,
            'max_signed_distance_reference_difference_m': max_distance,
            'representative_sample_checks_only': True, 'samples': rows, 'failures': failures}


def derive_status(coverage, minimum_coverage, checks):
    reasons = []
    coverage_missing = False
    for name in ('thumb', 'index'):
        if coverage.get(name, {}).get('independent_contact_positive_samples', 0) == 0:
            reasons.append(f'{name}: no independent positive contact; coverage is untested')
    for name, values in coverage.items():
        recall = values['recall_given_positive_normal_contact']
        if recall is not None and recall < minimum_coverage:
            coverage_missing = True
            reasons.append(f'{name}: per-link coverage {recall:.6f} below declared {minimum_coverage:.6f}; identify whether contact lies on an instrumented surface or an uncovered shell')
    for name, passed in checks.items():
        if not passed:
            reasons.append(f'{name}: failed or insufficient evidence')
    if coverage_missing:
        status = 'requires_contact_region_investigation'
    else:
        status = 'blocked_bridge_preflight_audit' if reasons else 'passed_bridge_preflight_audit'
    return status, reasons


def run_audit(run, output, minimum_coverage=.99, geometry_samples=16):
    run, output = Path(run).resolve(), Path(output).resolve()
    if not 0 < minimum_coverage <= 1 or geometry_samples < 2:
        raise ValueError('coverage must be in (0,1], geometry-samples >=2')
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite existing audit evidence: {output}')
    trace_path = run/'trace.npz'
    partial = not trace_path.exists()
    if partial:
        trace_path = run/'partial_trace.npz'
    with np.load(trace_path) as archive:
        arrays = {key: archive[key] for key in archive.files}
    with np.load(run/'environment/soft_mesh.npz') as archive:
        mesh = {key: archive[key] for key in archive.files}
    manifest = read(run/'manifest.json')
    scene = read(run/'environment/soft_scene_protocol.json')
    score = read(run/'score.json') if (run/'score.json').exists() else None
    layout_path = ROOT/'ur7e_xhand_sim/generated/tactile_layout.json'
    layouts = read(layout_path)
    names = [layout['finger'] for layout in layouts]
    time = arrays['time_s']
    if not len(time) or np.any(np.diff(time) <= 0):
        raise ValueError('Saved sample time must be nonempty and strictly increasing')
    finite_by_field = {key: bool(np.isfinite(value).all()) for key, value in arrays.items()
                       if np.issubdtype(value.dtype, np.number)}
    magnitude = np.linalg.norm(arrays['tactile'], axis=-1)
    signal = magnitude.max(-1) > 1e-5
    positive = arrays['contact_normal_load_N'] > .05
    coverage = {}
    for f, name in enumerate(names):
        count = int(positive[:, f].sum())
        miss = positive[:, f] & ~signal[:, f]
        coverage[name] = {
            'independent_contact_positive_samples': count,
            'contact_positive_but_no_virtual_signal': int(miss.sum()),
            'recall_given_positive_normal_contact': None if count == 0 else float((positive[:, f] & signal[:, f]).sum()/count),
            'miss_intervals': intervals(miss, time),
            'signal_below_0_05_N_support_threshold_count': int((signal[:, f] & ~positive[:, f]).sum()),
            'signal_with_zero_normal_load_count': int((signal[:, f] & (arrays['contact_normal_load_N'][:, f] < 1e-8)).sum()),
            'signal_without_positive_contact_intervals': intervals(signal[:, f] & ~positive[:, f], time),
            'active_taxels_on_positive_contact': quantiles(arrays['taxel_active_count'][positive[:, f], f]),
            'max_virtual_per_taxel_N': float(magnitude[:, f].max()),
            'max_IPC_normal_load_N': float(arrays['contact_normal_load_N'][:, f].max()),
            'interpretation': 'IPC loads aggregate the complete sensor-bearing link, including merged fixed fingertip/back-shell parts. A contact outside the 120-probe footprint can legitimately have no sensor response. Missing per-link coverage is not sufficient evidence of a taxel algorithm defect.',
        }
    gate = {'available': 'sensor_contact_mask' in arrays}
    if gate['available']:
        mask = arrays['sensor_contact_mask']
        valid = mask.shape == positive.shape and mask.dtype == np.dtype(bool)
        gate.update(valid_shape_and_type=bool(valid))
        if valid:
            gate.update(signal_with_disabled_pair=int((signal & ~mask).sum()),
                        positive_contact_with_disabled_pair=int((positive & ~mask).sum()),
                        max_force_on_disabled_pair_N=float(magnitude[~mask].max()) if (~mask).any() else 0.,
                        max_penetration_on_disabled_pair_m=float(np.abs(arrays['taxel_penetration_m'][~mask]).max()) if (~mask).any() else 0.)
    gate['passed'] = bool(gate.get('valid_shape_and_type') and gate.get('signal_with_disabled_pair') == 0
                          and gate.get('positive_contact_with_disabled_pair') == 0
                          and gate.get('max_force_on_disabled_pair_N') == 0
                          and gate.get('max_penetration_on_disabled_pair_m') == 0)
    gate['interpretation'] = 'Contact-pair existence, not force magnitude, gates the virtual sensor. Below 0.05 N can still be legitimate contact; that is a task support threshold, not the pair gate.'
    actor = observation_contract(arrays, manifest, scene)
    geometry = geometry_check(arrays, mesh['boundary'], layouts, geometry_samples)
    poses = pose_diagnostics(arrays, names)
    hold_start, hold_end = manifest['hold_interval_s']
    weights = interval_overlap(time-1/60, time, hold_start, hold_end)
    hold = weights > 1e-12
    complete = bool(not partial and abs(float(time[-1])-manifest['planned_end_tick']/300) < 1e-8)
    full_hold = abs(float(weights.sum())-3.) < 1e-7
    supported = (arrays['contact_normal_load_N'] > .05).sum(-1) >= 2
    task = bool(full_hold and (arrays['clearance_m'][hold] >= .05).all()
                and not arrays['fault'].any() and arrays['min_J'].min() > 0)
    strict = bool(task and supported[hold].all())
    score_matches = score is not None and task == score['task_success'] and strict == score['strict_success']
    tri = arrays['vertices'][:, mesh['boundary']]
    area = np.linalg.norm(np.cross(tri[:, :, 1]-tri[:, :, 0], tri[:, :, 2]-tri[:, :, 0]), axis=-1).sum(-1)/2
    rest = mesh['rest'][mesh['boundary']]
    rest_area = np.linalg.norm(np.cross(rest[:, 1]-rest[:, 0], rest[:, 2]-rest[:, 0]), axis=-1).sum()/2
    sample_error = float(np.max(np.abs(np.diff(time)-1/60))) if len(time)>1 else None
    action_error = float(np.max(np.abs(np.diff(arrays['actions'][:, 0])-1/15))) if len(arrays.get('actions', []))>1 else None
    checks = {'all_saved_numeric_outputs_finite': all(finite_by_field.values()),
              'actor_observation_contract': actor['passed'], 'representative_geometry_reference': geometry['passed'],
              'contact_presence_gate': gate['passed'], 'complete_planned_trajectory': complete,
              '60Hz_sampling': sample_error is not None and sample_error < 1e-8,
              '15Hz_actions': action_error is not None and action_error < 1e-8,
              'saved_task_score_reproduced': bool(score_matches)}
    status, reasons = derive_status(coverage, minimum_coverage, checks)
    report = {
        'status': status, 'blocking_reasons': reasons, 'interpretation': 'A block denotes insufficient or inconsistent sensor-transfer evidence; it does not by itself diagnose an implementation bug. Real contact on an uninstrumented merged shell must not be made tactile merely to pass a coverage check.',
        'source_run': str(run), 'partial_trace': partial,
        'input_sha256': {str(path.relative_to(run)): sha(path) for path in [trace_path,run/'manifest.json',run/'environment/soft_mesh.npz',run/'environment/soft_scene_protocol.json']},
        'audit_code_sha256': sha(__file__), 'layout_sha256': sha(layout_path),
        'recorded_source_sha256': manifest['source_sha256'],
        'recompute_adapter_sha256': sha(Path(__file__).with_name('soft_taxel.py')),
        'thresholds': {'minimum_contact_coverage': minimum_coverage, 'positive_IPC_normal_load_N': .05,
            'zero_IPC_normal_load_N': 1e-8, 'positive_virtual_taxel_norm_N': 1e-5,
            'probe_radius_m': .0015, 'IPC_contact_d_hat_m': scene['physics']['coupler_options']['contact_d_hat']},
        'valid_time_interval_s': [float(time[0]), float(time[-1])],
        'independent_reference': 'IPC last-assembled normal contact gradient / negative substep_dt squared; never virtual taxel force.',
        'reference_limits': 'Reference normal-force sign/units were validated in separate static contact tests; dynamic pointwise accuracy is not guaranteed. Virtual taxel values are not calibrated physical force. Friction-gradient validity depends on solver tolerance.',
        'checks': checks, 'finite_by_field': finite_by_field,
        'coverage': coverage, 'contact_gate': gate, 'actor_contract': actor,
        'contact_region_attribution': {'available_in_trace': False,
            'uncertainty': 'Trace normal loads are per link, not resolved by authored fixed component or instrumented surface patch. Locate missing-signal contacts against all merged fixed child parts before changing geometry or sensor parameters.'},
        'sampling_interval_max_error_s': sample_error, 'action_interval_max_error_s': action_error,
        'geometry_reference': geometry, 'Genesis_vs_IPC_pose': poses,
        'task_recomputed': {'task_success': task, 'strict_success': strict,
            'hold_duration_s': float(weights.sum()), 'hold_interval_s': [hold_start,hold_end],
            'min_hold_clearance_m': float(arrays['clearance_m'][hold].min()) if hold.any() else None,
            'minimum_recorded_interval_J': float(arrays['min_J'].min()),
            'success_is_separate_from_sensor_validity': True},
        'deformation': {'volume_ratio': quantiles(arrays['volume_ratio']), 'global_surface_area_ratio': quantiles(area/rest_area),
            'local_contact_area_available': False, 'interpretation': 'Active taxels indicate sampled footprint only. Global surface area is not local contact area; no area/force calibration is established.'},
        'domain_shift_limits': ['FEM deformation and IPC coupling replace the original rigid dynamics.',
            'Same actor dimensions and normalization do not establish matched input distributions.',
            'Virtual force is a geometric spring/damper estimate, not calibrated FEM traction.',
            'Material-point backward differences differ from rigid point velocities.',
            'Geometry recomputation checks selected recorded frames, not every frame or every collision pair.'],
    }
    report = json_safe(report)
    output.mkdir(parents=True, exist_ok=False)
    (output/'audit.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    _plot(output/'audit.png', arrays, manifest, report, area/rest_area, names)
    return report


def _plot(path, arrays, manifest, report, area_ratio, names):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    time = arrays['time_s']
    norm = np.linalg.norm(arrays['tactile'], axis=-1)
    fig, axes = plt.subplots(3, 2, figsize=(13, 10), constrained_layout=True)
    rows = report['geometry_reference']['samples']
    for f, color in [(0, 'tab:orange'), (1, 'tab:blue')]:
        axes[0,0].plot(time, arrays['contact_normal_load_N'][:,f], label=names[f], color=color)
        axes[0,1].plot(time, arrays['taxel_active_count'][:,f], label=names[f], color=color)
        axes[1,0].plot(time, norm[:,f].sum(-1), label=names[f], color=color)
        points = [r for r in rows if r['minimum_probe_surface_signed_distance_mm'][f] is not None]
        axes[1,1].plot([r['time_s'] for r in points], [r['minimum_probe_surface_signed_distance_mm'][f] for r in points], 'o-', markersize=3, label=names[f], color=color)
    axes[0,0].set(title='Independent IPC normal contact', ylabel='Primitive normal load (N)')
    axes[0,1].set(title='Virtual sensor footprint', ylabel='Active taxels (of 120)')
    axes[1,0].set(title='Virtual estimate: not calibrated contact force', ylabel='Sum of taxel force norms (nominal N)')
    axes[1,1].axhline(1.5, color='black', ls='--', label='1.5 mm probe radius')
    axes[1,1].set(title='Nearest sensor point to FEM boundary', ylabel='Signed distance (mm)', ylim=(-.5,4))
    if rows:
        active_times=[r['time_s'] for r in rows if max(r['IPC_normal_load_N'])>.05]
        if active_times:
            axes[1,1].set_xlim(max(float(time[0]),min(active_times)-1),float(time[-1])+.2)
    axes[2,0].plot(time, arrays['volume_ratio'], label='Volume / rest volume')
    axes[2,0].plot(time, area_ratio, label='Surface area / rest surface area')
    axes[2,0].set(title='Global deformation, not local contact area', ylabel='Ratio')
    axes[2,1].plot(time, arrays['clearance_m']*100, label='FEM bottom clearance')
    axes[2,1].axhline(5, color='black', ls='--', label='Task threshold')
    axes[2,1].set(title='Task success is separate from sensor validity', ylabel='Clearance (cm)')
    for ax in axes.ravel():
        ax.axvspan(*manifest['hold_interval_s'], alpha=.1, color='green')
        ax.set_xlabel('Simulation time (s)');ax.legend(loc='best');ax.grid(alpha=.2)
    fig.suptitle(f"Soft-body bridge audit: {manifest['controller']}, {manifest['mass_kg']*1000:.4f} g, mu={manifest['friction']}\n{report['status']}", fontsize=13)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def self_test():
    """CPU-only tests kept here because this audit is a single-file utility."""
    import unittest
    class AuditTests(unittest.TestCase):
        def test_intervals_keep_all_disjoint_misses(self):
            result = intervals([False,True,True,False,True], np.arange(5)/60)
            self.assertEqual([r['sample_count'] for r in result], [2,1])
            self.assertAlmostEqual(result[1]['first_sample_s'],4/60)
        def test_missing_contact_is_unknown_not_perfect_coverage(self):
            coverage={n:{'independent_contact_positive_samples':0,'recall_given_positive_normal_contact':None} for n in ['thumb','index']}
            status,reasons=derive_status(coverage,.99,{'finite':True})
            self.assertEqual(status,'blocked_bridge_preflight_audit');self.assertEqual(len(reasons),2)
        def test_pass_requires_coverage_and_independent_checks(self):
            coverage={n:{'independent_contact_positive_samples':100,'recall_given_positive_normal_contact':1.} for n in ['thumb','index']}
            self.assertEqual(derive_status(coverage,.99,{'actor':True})[0],'passed_bridge_preflight_audit')
            self.assertEqual(derive_status(coverage,.99,{'actor':False})[0],'blocked_bridge_preflight_audit')
            coverage['thumb']['recall_given_positive_normal_contact']=.98
            self.assertEqual(derive_status(coverage,.99,{'actor':True})[0],'requires_contact_region_investigation')
        def test_actual_IPC_pose_difference_and_affine_distortion(self):
            data={'time_s':np.arange(3), 'finger_positions':np.zeros((3,5,3)),
                  'finger_rotations':np.broadcast_to(np.eye(3),(3,5,3,3)).copy(),
                  'contact_normal_load_N':np.ones((3,5)),
                  'finger_ipc_transforms':np.broadcast_to(np.eye(4),(3,5,4,4)).copy()}
            data['finger_ipc_transforms'][...,:3,3]=[.003,0,0]
            data['finger_ipc_transforms'][...,0,0]=1.01
            result=pose_diagnostics(data,['thumb','index','middle','ring','pinky'])
            self.assertAlmostEqual(result['per_finger']['thumb']['position_difference_mm']['max'],3.)
            self.assertAlmostEqual(result['per_finger']['thumb']['IPC_orthogonality_error_Frobenius']['max'],.0201)
            self.assertAlmostEqual(result['per_finger']['thumb']['nearest_proper_rotation_difference_deg']['max'],0.)
        def test_missing_IPC_pose_is_explicit(self):
            self.assertFalse(pose_diagnostics({},[])['available'])
        def test_nonfinite_evidence_does_not_emit_invalid_JSON(self):
            value=json_safe({'finite':False,'error':float('nan'),'x':[float('inf'),1.]})
            self.assertEqual(value,{'finite':False,'error':None,'x':[None,1.]})
            json.dumps(value,allow_nan=False)
        def test_masked_replay_keeps_pair_gate_without_force_input(self):
            body=trimesh.creation.box(extents=[.02]*3)
            layouts=np.tile([.001,.002,.0105],(5,120,1))
            sensor=SoftTaxelAdapter(layouts).sample(body.vertices,body.faces,np.zeros((5,3)),
                np.tile(np.eye(3),(5,1,1)),contact_mask=np.zeros(5,dtype=bool))
            self.assertFalse(sensor['active'].any());self.assertTrue((sensor['raw_penetration_m']>0).all())
        def test_geometry_replay_consumes_saved_per_frame_pair_masks(self):
            body=trimesh.creation.box(extents=[.02]*3)
            layouts=[{'positions_link_m':np.tile([.001,.002,.0105],(120,1)).tolist()} for _ in range(5)]
            data={'time_s':np.array([1/60,2/60]),'vertices':np.stack([body.vertices,body.vertices+[.0001,0,0]]),
                  'finger_positions':np.zeros((2,5,3)),
                  'finger_rotations':np.broadcast_to(np.eye(3),(2,5,3,3)),
                  'sensor_contact_mask':np.array([[False]*5,[True,False,False,False,False]]),
                  'contact_normal_load_N':np.array([[0.]*5,[1.,0.,0.,0.,0.]])}
            adapter=SoftTaxelAdapter(layouts);forces=[]
            for i in range(2):
                forces.append(adapter.sample(data['vertices'][i],body.faces,data['finger_positions'][i],
                    data['finger_rotations'][i],contact_mask=data['sensor_contact_mask'][i])['force_link_N'])
            data['tactile']=np.asarray(forces)
            result=geometry_check(data,body.faces,layouts,2)
            self.assertTrue(result['passed']);self.assertTrue(result['replayed_saved_contact_mask'])
            self.assertEqual(result['max_force_replay_difference_N'],0.)
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(AuditTests))
    return result.wasSuccessful()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--minimum-coverage',type=float,default=.99)
    parser.add_argument('--geometry-samples',type=int,default=16)
    parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    if args.self_test:
        raise SystemExit(0 if self_test() else 1)
    if args.run is None or args.output is None:
        parser.error('--run and --output are required unless using --self-test')
    report=run_audit(args.run,args.output,args.minimum_coverage,args.geometry_samples)
    print(json.dumps({'status':report['status'],'blocking_reasons':report['blocking_reasons'],
        'output':str(args.output.resolve())},indent=2))


if __name__=='__main__':
    main()
