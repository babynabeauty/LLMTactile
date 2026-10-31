"""Read-only, trace-audited comparison of multi-object frozen-policy evaluations.

Success counts always include failures. Force-efficiency comparisons only use
successful holds, with same-seed/same-material joint-success pairs made explicit.
The optional selection command applies only the preregistered validation rule;
test reports never select checkpoints and no training process is launched.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import html
import json
import math
from pathlib import Path

import numpy as np

from .multiobject_protocol import METHODS, SEEDS, validate_protocol, protocol_shapes


HOLD_FIELDS = (
    'hold_mean_all_robot_normal_force_N', 'hold_mean_all_robot_contact_force_N',
    'hold_mean_sensor_contact_force_N', 'hold_peak_all_robot_normal_force_N',
    'hold_peak_all_robot_contact_force_N', 'hold_peak_sensor_contact_force_N')
POLICY_FIELDS = ('mean_all_robot_normal_force_N', 'mean_all_robot_contact_force_N',
                 'mean_contact_force_N', 'peak_all_robot_normal_force_N',
                 'peak_all_robot_contact_force_N', 'peak_contact_force_N')


def fixed_control_map(gains):
    """Use exactly the benchmark's canonical names, with no integer-only list."""
    result={}
    for gain in gains:
        if isinstance(gain,bool) or not isinstance(gain,(int,float)) or not math.isfinite(gain) or gain<=0:
            raise ValueError('Fixed gains must be finite positive numbers')
        name=f'fixed_{gain:g}'.replace('.','p')
        if name in result:
            raise ValueError('Fixed gains must have distinct canonical controller names')
        result[name]=float(gain)
    return result


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _finite(values, label):
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f'Nonfinite {label}')
    return values


def trace_statistics(trace, hold_interval_s):
    """Independently integrate actual 60Hz contacts, including boundary overlap."""
    times = _finite(trace['sample_time_s'], 'sample times')
    if times.ndim != 1 or len(times) < 2 or not np.allclose(np.diff(times), 1/60, rtol=0, atol=1e-7):
        raise ValueError('Expected strictly increasing uniform60Hz sample_time_s')
    start, end = map(float, hold_interval_s)
    if not np.isfinite([start,end]).all() or not np.isclose(end-start, 3., atol=1e-8):
        raise ValueError('The original hold interval must last3s')
    dt = np.full(len(times), 1/60.)
    weights = np.maximum(0., np.minimum(times,end)-np.maximum(times-dt,start))
    if not np.isclose(weights.sum(), 3., atol=1e-7):
        raise ValueError('Physical trace does not cover the complete final3s')
    included = weights > 1e-10
    # Frozen online peak accumulators use tick/300 and `if hold_dt`.
    # Trace timestamps may instead be constructed by adding action offsets.
    # Reconstruct that clock for auditing recorded peaks, including positive
    # floating-point boundary overlap. Keep the task window tolerance unchanged.
    online_times = np.rint(times * 300.) / 300.
    online_weights = np.maximum(0., np.minimum(online_times, end)
                                - np.maximum(online_times - dt, start))
    peak_included = online_weights > 0.
    clearance = _finite(trace['clearance_m'], 'clearance')
    if clearance.ndim == 1:
        clearance = clearance[:,None]
    shape = clearance.shape
    if shape[0] != len(times):
        raise ValueError('Trace time and lane dimensions differ')
    faults = _finite(trace['fault'], 'fault flags')
    if faults.shape != shape:
        raise ValueError('Fault trace shape differs from clearance')
    if not np.isin(faults,[0.,1.]).all():
        raise ValueError('Fault flags must be boolean/0/1')
    finger = _finite(trace['contact_load_N'], 'sensor contact force')
    if finger.shape != shape+(5,):
        raise ValueError('Expected five physical sensor-bearing link loads per lane')
    total = _finite(trace['all_robot_contact_load_N'], 'all-robot force')
    normal = _finite(trace['all_robot_normal_load_N'], 'all-robot normal force')
    friction_error = _finite(trace['contact_friction_error'], 'friction error')
    if total.shape != shape or normal.shape != shape or friction_error.shape != shape:
        raise ValueError('All-robot contact trace shape differs from clearance')
    if np.any(finger < -1e-7) or np.any(normal < -1e-7) or np.any(total < -1e-7):
        raise ValueError('Physical contact magnitudes must be nonnegative')
    if np.any(normal > total+2e-5) or np.any(finger.sum(-1) > total+2e-5):
        raise ValueError('Normal/sensor force cannot exceed all-robot contact magnitudes')
    if np.max(friction_error) > 1e-5:
        raise ValueError('Actual contact friction differs from requested material')
    supported = (finger > .05).sum(-1) >= 2
    hold_min = clearance[included].min(0)
    fault = faults.astype(bool).any(0)
    task = (hold_min >= .05) & ~fault
    strict = task & supported[included].all(0)
    values = {}
    for label, force in [('all_robot_normal',normal),('all_robot_contact',total),('sensor_contact',finger.sum(-1))]:
        values[f'hold_mean_{label}_force_N'] = (force*weights[:,None]).sum(0)/3.
        values[f'hold_peak_{label}_force_N'] = force[peak_included].max(0)
        key = 'contact' if label == 'sensor_contact' else label
        values[f'mean_{key}_force_N'] = force.mean(0)
        values[f'peak_{key}_force_N'] = force.max(0)
    result = []
    for lane in range(shape[1]):
        row = {key:float(value[lane]) for key,value in values.items()}
        row.update(task_success=bool(task[lane]), strict_success=bool(strict[lane]),
            hold_min_clearance_m=float(hold_min[lane]), hold_duration_s=float(weights.sum()),
            hold_contact_ok=bool(supported[included,lane].all()),
            hold_contact_fraction=float((supported[:,lane]*weights).sum()/3.),
            force_measurement_duration_s=float(dt.sum()),
            hold_force_measurement_duration_s=float(weights.sum()),
            max_clearance_m=float(clearance[:,lane].max()),
            contact_friction_error_max=float(friction_error[:,lane].max()),
            collision=bool(fault[lane]))
        result.append(row)
    return result


def audit_recorded_episode(recorded, calculated):
    errors = []
    for key in ('task_success','strict_success'):
        if type(recorded.get(key)) is not bool or recorded[key] != calculated[key]:
            errors.append(f'{key}: recorded={recorded.get(key)!r}, trace={calculated[key]}')
    for key in HOLD_FIELDS+POLICY_FIELDS+('hold_min_clearance_m','hold_duration_s'):
        if key not in recorded:
            errors.append(f'Missing required episode metric {key}')
            continue
        value = recorded[key]
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
            errors.append(f'Nonfinite or nonnumeric {key}')
        elif not math.isclose(float(value), calculated[key], rel_tol=2e-5, abs_tol=2e-5):
            errors.append(f'{key}: recorded={value}, trace={calculated[key]}')
    return errors


def primitive_properties(spec):
    """Independent analytic volume, axis-aligned bounds and inertia per kg."""
    shape=spec['shape']
    side,radius,height = spec['side_m'],spec['radius_m'],spec['height_m']
    if shape=='cube':
        volume=side**3; bounds=[side]*3; inertia=np.eye(3)*side**2/6
    elif shape=='cuboid':
        x,y,z=spec['size_m']
        volume=x*y*z;bounds=[x,y,z]
        inertia=np.diag([y*y+z*z,x*x+z*z,x*x+y*y])/12
    elif shape=='hex_prism':
        across=spec['across_flats_m'];circumradius=across/math.sqrt(3)
        volume=3*math.sqrt(3)/2*circumradius**2*height
        bounds=[across,2*circumradius,height]
        inertia=np.diag([5*circumradius**2/24+height**2/12]*2+[5*circumradius**2/12])
    elif shape=='sphere':
        volume=4*np.pi*radius**3/3; bounds=[2*radius]*3; inertia=np.eye(3)*2*radius**2/5
    elif shape=='cylinder':
        volume=np.pi*radius**2*height; bounds=[2*radius,2*radius,height]
        inertia=np.diag([(3*radius**2+height**2)/12]*2+[radius**2/2])
    else:
        raise ValueError(f'Unsupported primitive shape {shape!r}')
    return volume,np.asarray(bounds),inertia


def audit_environment(environment, episodes, protocol):
    """Check solver readbacks against independently evaluated primitive formulas."""
    errors = []
    if not episodes:
        return ['No physical lanes to audit']
    shape = episodes[0]['object_shape']
    declared = next((o for o in protocol['objects'] if o['object_id']==shape),None)
    if declared is None:
        return ['Unknown physical object shape']
    volume,bounds,inertia=primitive_properties(declared['object_spec'])
    mass=np.array([e['mass_kg'] for e in episodes])
    expected={
        'mass_readback_kg_per_lane':mass,
        'density_readback_kg_m3_per_lane':mass/volume,
        'inertia_readback_kg_m2_per_lane':mass[:,None,None]*inertia,
        'object_bounds_readback_m':np.array(bounds),
        'object_initial_bottom_readback_m':np.full(len(episodes),.2),
        'friction_per_lane':np.array([e['friction'] for e in episodes]),
    }
    for field,wanted in expected.items():
        try:
            actual=_finite(environment[field],field)
            if actual.shape != wanted.shape or not np.allclose(actual,wanted,rtol=2e-5,atol=1e-8):
                errors.append(f'{field} differs from protocol/episode physical values')
        except (KeyError,TypeError,ValueError) as exc:
            errors.append(f'{field}: {exc}')
    if environment.get('object_spec',{}).get('shape') != shape:
        errors.append('Physical environment shape differs from episode shape')
    if shape=='hex_prism':
        try:
            physical_spec=environment['object_spec'];asset=physical_spec['mesh_asset']
            mesh_path=Path(asset['path'])
            if sha256(mesh_path)!=asset['sha256']:
                raise ValueError('mesh file hash differs from the recorded collider')
            angles=np.deg2rad(30+60*np.arange(6))
            radius=declared['object_spec']['across_flats_m']/math.sqrt(3)
            height=declared['object_spec']['height_m']
            xy=radius*np.column_stack([np.cos(angles),np.sin(angles)])
            wanted=np.concatenate([np.column_stack([xy,np.full(6,z)]) for z in (-height/2,height/2)])
            vertices=np.asarray([[float(v) for v in line.split()[1:]]
                for line in mesh_path.read_text().splitlines() if line.startswith('v ')])
            if vertices.shape!=(12,3) or not np.allclose(vertices,wanted,rtol=0,atol=1e-10):
                raise ValueError('mesh vertices differ from declared hexagonal dimensions/orientation')
            if not np.allclose(physical_spec['vertices_local_m'],wanted,rtol=0,atol=1e-10):
                raise ValueError('recorded hexagonal support vertices differ from the collider')
        except (KeyError,OSError,TypeError,ValueError) as exc:
            errors.append(f'Hexagonal collider audit: {exc}')
    if declared.get('trajectory_sha256') and environment.get('demonstration_sha256') != declared['trajectory_sha256']:
        errors.append('Physical prefix uses a different demonstration hash')
    return errors


def _mean(rows, field):
    return float(np.mean([row[field] for row in rows])) if rows else None


def benchmark_provenance(path, root, evaluation):
    """Resolve the benchmark manifest that actually supplied this controller."""
    for directory in Path(path).parents:
        if directory == Path(root).parent:
            break
        manifest_path = directory/'manifest.json'
        if not manifest_path.is_file():
            continue
        manifest=json.loads(manifest_path.read_text())
        if 'models' not in manifest or 'split' not in manifest:
            continue
        if manifest['split']!=evaluation['split']:
            raise ValueError('Benchmark manifest split differs from evaluated records')
        controller=evaluation['controller']
        if evaluation['method'] in METHODS:
            model=manifest['models'].get(controller)
            if model is None:
                raise ValueError('Controller is absent from benchmark model provenance')
            for field in ('training_seed','method','checkpoint_stage'):
                if model.get(field)!=evaluation.get(field):
                    raise ValueError(f'Benchmark checkpoint {field} differs from evaluated record')
            if not isinstance(model.get('sha256'),str) or len(model['sha256'])!=64:
                raise ValueError('Benchmark has no valid evaluated checkpoint hash')
        else:
            model=None
            if evaluation['method'] not in fixed_control_map(manifest.get('fixed_gains',[])):
                raise ValueError('Fixed gain absent from benchmark provenance')
        return dict(benchmark_manifest_path=str(manifest_path.resolve()),
            benchmark_manifest_sha256=sha256(manifest_path),
            evaluated_checkpoint_sha256=model['sha256'] if model else None,
            evaluated_checkpoint_path=model['path'] if model else None)
    raise ValueError('Cannot locate the benchmark manifest for checkpoint provenance')


def summarize_group(rows):
    successful = [row for row in rows if row['task_success']]
    return dict(episodes=len(rows), task_successes=len(successful),
        strict_successes=sum(row['strict_success'] for row in rows),
        task_success_rate=len(successful)/len(rows) if rows else None,
        successful_hold_forces={field:_mean(successful,field) for field in HOLD_FIELDS},
        all_episode_hold_forces_descriptive_only={field:_mean(rows,field) for field in HOLD_FIELDS},
        all_episode_policy_forces_descriptive_only={field:_mean(rows,field) for field in POLICY_FIELDS},
        force_interpretation='Failed episodes remain in success denominators. Their zero force after a drop must not be interpreted as force efficiency.')


def paired_comparison(tactile, comparator, comparator_name):
    def key(row):
        return (row['object_shape'],row['condition_id'],row['training_seed'])
    left = {key(row):row for row in tactile}
    right = {key(row):row for row in comparator}
    # A fixed deterministic baseline can be shared with each model seed.
    if comparator_name.startswith('fixed_'):
        reusable = {(r['object_shape'],r['condition_id']):r for r in comparator
                    if r['training_seed'] in (None,0)}
        for target in left:
            if target not in right and target[:2] in reusable:
                right[target] = reusable[target[:2]]
    common = sorted(left.keys() & right.keys())
    successful = [key_ for key_ in common if left[key_]['task_success'] and right[key_]['task_success']]
    force = {}
    for field in HOLD_FIELDS:
        a = [left[key_][field] for key_ in successful]
        b = [right[key_][field] for key_ in successful]
        am, bm = (float(np.mean(a)),float(np.mean(b))) if a else (None,None)
        force[field] = dict(tactile_mean=am, comparator_mean=bm,
            tactile_minus_comparator_mean=float(np.mean(np.array(a)-np.array(b))) if a else None,
            relative_change_percent=100*(am-bm)/bm if bm is not None and bm>0 else None)
    return dict(comparator=comparator_name, paired_cases=len(common),
        tactile_successes=sum(left[k]['task_success'] for k in common),
        comparator_successes=sum(right[k]['task_success'] for k in common),
        jointly_successful_cases=len(successful),
        tactile_only_success=sum(left[k]['task_success'] and not right[k]['task_success'] for k in common),
        comparator_only_success=sum(right[k]['task_success'] and not left[k]['task_success'] for k in common),
        unmatched_tactile_cases=len(left.keys()-right.keys()),
        jointly_successful_force_comparison=force,
        jointly_successful_case_keys=[list(k) for k in successful],
        interpretation='Force differences condition on joint success and do not replace success-rate comparison. Fixed baseline reuse is pairing, not additional independent baseline trials.')


def collect(root, protocol, split='test', checkpoint_stage=None):
    root = Path(root)
    validate_protocol(protocol)
    shapes=protocol_shapes(protocol)
    fixed_controls=fixed_control_map(protocol['evaluation']['fixed_gains'])
    all_methods=(*METHODS,*fixed_controls)
    expected_conditions = {r['condition_id']:r for r in protocol['splits'][split]}
    rows, files, errors, seen = [], [], [], set()
    search_root = root/'eval'/split
    for path in sorted(search_root.rglob('evaluation.json')):
        value = json.loads(path.read_text())
        if (checkpoint_stage is not None and value.get('method') in METHODS
                and value.get('checkpoint_stage') != checkpoint_stage):
            continue
        episodes = value['episodes']
        interval = value.get('hold_interval_s')
        if interval is None:
            errors.append(f'{path}: missing hold_interval_s; cannot audit original success')
            continue
        trace_path = path.parent/'trace.npz'
        try:
            provenance=benchmark_provenance(path,root,value)
            actual_trace_sha = sha256(trace_path)
            if value.get('trace_sha256') != actual_trace_sha:
                raise ValueError('Saved trace SHA256 is missing or mismatched')
            prefix = json.loads((path.parent/'prefix_consistency.json').read_text())
            if not prefix or any(not isinstance(v,(int,float)) or not math.isfinite(v) or v!=0 for v in prefix.values()):
                raise ValueError('Compared controllers have different prefix observations')
            environment_path = Path(value['environment_protocol_path'])
            if value.get('environment_protocol_sha256') != sha256(environment_path):
                raise ValueError('Physical environment protocol SHA256 is missing or mismatched')
            environment = json.loads(environment_path.read_text())
            physical_errors = audit_environment(environment,episodes,protocol)
            if physical_errors:
                raise ValueError('; '.join(physical_errors))
            with np.load(trace_path, allow_pickle=False) as trace:
                calculated = trace_statistics(trace, interval)
            if len(episodes) != len(calculated):
                raise ValueError('Episode and trace lane counts differ')
        except (OSError,KeyError,ValueError) as exc:
            errors.append(f'{path}: {exc}')
            continue
        files.append(dict(path=str(path.resolve()),sha256=sha256(path),
                          trace_path=str(trace_path.resolve()),trace_sha256=actual_trace_sha,
                          environment_protocol_path=str(environment_path.resolve()),
                          environment_protocol_sha256=sha256(environment_path),**provenance))
        for lane, (episode, metric) in enumerate(zip(episodes,calculated)):
            if (checkpoint_stage is not None and episode.get('method') in METHODS
                    and episode.get('checkpoint_stage') != checkpoint_stage):
                continue
            local = audit_recorded_episode(episode, metric)
            for field in ('controller','method','training_seed','checkpoint_stage'):
                if episode.get(field)!=value.get(field):
                    local.append(f'Episode {field} differs from benchmark controller record')
            identity = {key:episode.get(key) for key in ('condition_id','object_shape','split','training_seed','method')}
            condition = expected_conditions.get(identity['condition_id'])
            if identity['split'] != split or identity['object_shape'] not in shapes or condition is None:
                local.append('Unknown shape, split or material condition identity')
            if condition is not None:
                for name in ('mass_kg','friction'):
                    if not np.isclose(episode.get(name,np.nan), condition[name], atol=1e-6, rtol=0):
                        local.append(f'{name} differs from protocol condition')
            method = identity['method']
            if method not in all_methods:
                local.append(f'Unexpected method {method!r}')
            if method in METHODS and identity['training_seed'] not in SEEDS:
                local.append('Policy evaluation has an unknown training seed')
            key = tuple(identity.values())
            if key in seen:
                local.append('Duplicate episode identity; refusing to double-count')
            seen.add(key)
            if local:
                errors.extend(f'{path} lane{lane}: {error}' for error in local)
                continue
            rows.append({**episode,**metric,**provenance,'source_evaluation':str(path.resolve()),'source_lane':lane})
    groups = defaultdict(list)
    for row in rows:
        groups[(row['object_shape'],row['method'])].append(row)
    overall = {method:summarize_group([r for r in rows if r['method']==method])
               for method in all_methods}
    by_seed = {str(seed):{method:summarize_group([r for r in rows if r['method']==method and r['training_seed']==seed])
                         for method in METHODS} for seed in SEEDS}
    comparisons = []
    for shape in (*shapes,'all'):
        selected = [r for r in rows if shape=='all' or r['object_shape']==shape]
        tactile = [r for r in selected if r['method']=='tactile']
        for comparator in ('no_tactile',*fixed_controls):
            result = paired_comparison(tactile,[r for r in selected if r['method']==comparator],comparator)
            comparisons.append(dict(object_shape=shape,**result))
    expected = {(shape,condition,seed,method) for shape in shapes for condition in expected_conditions
                for seed in SEEDS for method in METHODS}
    observed = {(r['object_shape'],r['condition_id'],r['training_seed'],r['method']) for r in rows if r['method'] in METHODS}
    fixed_expected = {(shape,condition,method) for shape in shapes for condition in expected_conditions
                      for method in fixed_controls}
    fixed_observed = {(r['object_shape'],r['condition_id'],r['method']) for r in rows if r['method'].startswith('fixed_')}
    missing = sorted(expected-observed)
    fixed_missing = sorted(fixed_expected-fixed_observed)
    return dict(schema_version=1,experiment_id=protocol['experiment_id'],split=split,
        checkpoint_stage=checkpoint_stage,
        status='audited_complete' if not errors and not missing and not fixed_missing else 'incomplete_or_audit_failed',
        audit_errors=errors,missing_policy_cases=[list(k) for k in missing],
        missing_fixed_cases=[list(k) for k in fixed_missing],
        expected_policy_episodes=len(expected),expected_fixed_unique_episodes=len(fixed_expected),
        fixed_controls=fixed_controls,
        demonstration_provenance=[dict(object_id=obj['object_id'],
            demonstration_source=(obj.get('demonstration_evidence') or {}).get('demonstration_source'),
            actual_astra_api=(obj.get('demonstration_evidence') or {}).get('actual_astra_api'),
            privileged_teacher=(obj.get('demonstration_evidence') or {}).get('privileged_teacher'),
            trajectory_sha256=obj.get('trajectory_sha256'),
            source_tactile_model=(obj.get('demonstration_evidence') or {}).get('source_tactile_model'),
            verification_tactile_model=(obj.get('demonstration_evidence') or {}).get('tactile_model'))
            for obj in protocol['objects']],
        audited_episode_count=len(rows),source_files=files,
        overall=overall,by_seed=by_seed,
        by_shape={shape:{method:summarize_group(groups[(shape,method)])
                        for method in all_methods} for shape in shapes},
        paired_comparisons=comparisons,episodes=rows,
        interpretation='Known shapes, unseen material pairs. Three trained seeds; deterministic material lanes are paired cases. No inferential significance or unseen-shape claim is implied.')


def select_validation_checkpoint(stage1, stage2, protocol, checkpoint_paths):
    """Apply the preregistered rule to one seed/modality, never to test data."""
    expected = {(shape,row['condition_id']) for shape in protocol_shapes(protocol) for row in protocol['splits']['validation']}
    identity = set()
    mappings = []
    for stage,rows in [('stage1',stage1),('stage2',stage2)]:
        mapping = {}
        for row in rows:
            if row.get('split') != 'validation' or row.get('checkpoint_stage') != stage:
                raise ValueError('Selection requires correctly identified validation stages only')
            identity.add((row['training_seed'],row['method']))
            key = (row['object_shape'],row['condition_id'])
            if key in mapping:
                raise ValueError('Duplicate validation case')
            if not isinstance(row['task_success'],bool):
                raise ValueError('Validation success label must be a boolean')
            if not math.isfinite(row['hold_mean_all_robot_contact_force_N']):
                raise ValueError('Nonfinite validation force')
            mapping[key] = row
        if set(mapping) != expected:
            raise ValueError(f'Both checkpoints require all{len(expected)} preregistered validation cases')
        mappings.append(mapping)
    if len(identity) != 1:
        raise ValueError('Compare one seed and one modality at a time')
    seed,method = next(iter(identity))
    if seed not in SEEDS or method not in METHODS:
        raise ValueError('Unknown policy seed/modality')
    a,b = mappings
    successes = [sum(row['task_success'] for row in mapping.values()) for mapping in mappings]
    common = sorted(k for k in expected if a[k]['task_success'] and b[k]['task_success'])
    forces = [float(np.mean([mapping[k]['hold_mean_all_robot_contact_force_N'] for k in common]))
              if common else None for mapping in mappings]
    if successes[0] != successes[1]:
        index = int(successes[1] > successes[0]); reason = 'higher_validation_task_success_count'
    elif common and forces[0] != forces[1]:
        index = int(forces[1] < forces[0]); reason = 'lower_force_on_jointly_successful_validation_cases'
    else:
        index = 1; reason = 'stage2_default_when_tied_or_no_joint_success'
    stage = ('stage1','stage2')[index]
    paths = {key:Path(value).expanduser().resolve() for key,value in checkpoint_paths.items()}
    if set(paths) != {'stage1','stage2'}:
        raise ValueError('Both checkpoint paths are required for a bound selection')
    hashes = {key:sha256(path) for key,path in paths.items()}
    for name,rows in [('stage1',stage1),('stage2',stage2)]:
        if any(row.get('evaluated_checkpoint_sha256')!=hashes[name] for row in rows):
            raise ValueError(f'{name} checkpoint differs from the model actually used by validation benchmark')
    return dict(training_seed=seed,method=method,selected_stage=stage,reason=reason,
        validation_case_count=len(expected),task_successes=dict(zip(('stage1','stage2'),successes)),
        jointly_successful_cases=len(common),joint_success_force_N=dict(zip(('stage1','stage2'),forces)),
        checkpoint_paths={key:str(path) for key,path in paths.items()},checkpoint_sha256=hashes,
        selected_model=dict(path=str(paths[stage]),sha256=hashes[stage],checkpoint_stage=stage),
        validation_sources=sorted({row['source_evaluation'] for row in stage1+stage2}),
        rule=protocol['training']['checkpoint_selection'])


def build_selection(root, protocol, checkpoint_mapping):
    """Return benchmark-compatible selected_models with audited validation evidence.

    checkpoint_mapping maps seed1_tactile etc to {stage1:path, stage2:path}.
    A seed1 pilot may provide only its two modalities; the final suite supplies6.
    """
    reports = {stage:collect(root,protocol,'validation',stage) for stage in ('stage1','stage2')}
    for stage,report in reports.items():
        if report['audit_errors']:
            raise ValueError(f'{stage} validation failed trace audit: {report["audit_errors"]}')
    decisions,models = {},{}
    for name,paths in checkpoint_mapping.items():
        expected = {(f'seed{seed}_{method}',seed,method) for seed in SEEDS for method in METHODS}
        identity = [(seed,method) for label,seed,method in expected if label==name]
        if len(identity)!=1:
            raise ValueError(f'Unknown checkpoint name {name}')
        seed,method=identity[0]
        subsets={stage:[r for r in report['episodes'] if r['training_seed']==seed and r['method']==method]
                 for stage,report in reports.items()}
        decision=select_validation_checkpoint(subsets['stage1'],subsets['stage2'],protocol,paths)
        decisions[name]=decision;models[name]=decision['selected_model']
    if not models:
        raise ValueError('No checkpoints requested for validation selection')
    return dict(schema_version=1,experiment_id=protocol['experiment_id'],
        selection_split='validation',test_outcomes_read=False,selected_models=models,
        decisions=decisions,validation_audit_sources={stage:report['source_files'] for stage,report in reports.items()},
        rule=protocol['training']['checkpoint_selection'])


def render_html(report):
    def value(x):
        return '—' if x is None else f'{x:.3f}'
    labels = {'cube':'方块','cuboid':'长方体','hex_prism':'正六棱柱','cylinder':'圆柱','sphere':'球体','tactile':'触觉 PPO','no_tactile':'无触觉 PPO'}
    labels.update({name:f'固定增益 {gain:g}' for name,gain in report.get('fixed_controls',{}).items()})
    body = []
    for shape, groups in report['by_shape'].items():
        for method, group in groups.items():
            force = group['successful_hold_forces']
            body.append('<tr>'+''.join(f'<td>{html.escape(str(cell))}</td>' for cell in (
                labels[shape],labels.get(method,method),f'{group["task_successes"]}/{group["episodes"]}',
                value(force['hold_mean_all_robot_contact_force_N']),
                value(force['hold_mean_all_robot_normal_force_N']),
                value(force['hold_mean_sensor_contact_force_N']),
                value(force['hold_peak_all_robot_contact_force_N'])))+'</tr>')
    pairs = [];fixed_pairs=[]
    for pair in report['paired_comparisons']:
        f = pair['jointly_successful_force_comparison']['hold_mean_all_robot_contact_force_N']
        cells=(labels.get(pair['object_shape'],'全部'),pair['paired_cases'],pair['jointly_successful_cases'],
               value(f['tactile_mean']),value(f['comparator_mean']),value(f['relative_change_percent']))
        if pair['comparator']=='no_tactile':
            pairs.append('<tr>'+''.join(f'<td>{html.escape(str(cell))}</td>' for cell in cells)+'</tr>')
        else:
            cells=(cells[0],labels.get(pair['comparator'],pair['comparator']),*cells[1:])
            fixed_pairs.append('<tr>'+''.join(f'<td>{html.escape(str(cell))}</td>' for cell in cells)+'</tr>')
    errors = ''.join(f'<li>{html.escape(e)}</li>' for e in report['audit_errors']) or '<li>已读取记录未发现审计错误；仍须查看是否缺少实验。</li>'
    demonstrations = report.get('demonstration_provenance', [])
    frozen_replays = [row for row in demonstrations
                      if row.get('demonstration_source') == 'direct_teacher_frozen_replay']
    source_note = ''
    if frozen_replays:
        source_note = ('<p class="note">示范来源：仿真器内采集的直接教师轨迹，再按原命令、增益和时间重放并重新采集 v3 触觉；'
            '不是 Astra API 推理。每个绑定示范另经独立场景验证。命令相同不表示物理状态逐帧完全相同。'
            'v3 信号尚未实物标定，切向通道是相对速度代理；拇指覆盖缺口仍保留，原始零响应未填补。</p>')
    return f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>多物体触觉与低力抓取实验</title>
<style>body{{max-width:1200px;margin:36px auto;padding:0 24px;font:16px/1.7 system-ui;color:#172238;background:#f6f8fc}}table{{border-collapse:collapse;width:100%;background:white}}th,td{{padding:8px;border-bottom:1px solid #dce2ec;text-align:left}}th{{background:#e7edf6}}.note{{background:#fff1cf;padding:16px;border-radius:8px}}h1{{font-size:28px}}h2{{font-size:21px}}</style>
<h1>多物体触觉与低力抓取实验</h1><p>数据集：{html.escape(report['split'])} · 状态：{html.escape(report['status'])} · 已审计 {report['audited_episode_count']} 条回放。</p>
{source_note}
<p class="note">成功标准沿用最终 3 秒最低表面离桌 ≥5 cm，且无机器人碰撞。平均力只在成功抓取中比较；失败后掉落产生的零力不表示省力。这里检验已知形状的质量／摩擦泛化，不代表未知形状泛化。</p>
<h2>全部成功率与成功案例的保持阶段力</h2><table><thead><tr><th>形状</th><th>方法</th><th>成功／总数</th><th>平均全接触合力 N</th><th>平均法向力 N</th><th>平均五指旧指标 N</th><th>合力峰值均值 N</th></tr></thead><tbody>{''.join(body)}</tbody></table>
<h2>有／无触觉都成功时的配对比较</h2><p>同形状、材料条件和训练 seed 配对。主指标为整个手—物接触的力矢量模之和。负百分比表示触觉模型施力更小；该子集不能替代上表的全部成功率。</p><table><thead><tr><th>形状</th><th>配对数</th><th>共同成功数</th><th>触觉全接触合力 N</th><th>无触觉全接触合力 N</th><th>相对变化 %</th></tr></thead><tbody>{''.join(pairs)}</tbody></table>
<details><summary>查看与全部固定增益的配对比较</summary><p>每个固定增益只在同形状、同材料条件运行一次，并与三个训练 seed 分别配对；这些复用记录不是额外独立基准实验。</p><table><thead><tr><th>形状</th><th>固定基准</th><th>配对数</th><th>共同成功数</th><th>触觉全接触合力 N</th><th>基准全接触合力 N</th><th>相对变化 %</th></tr></thead><tbody>{''.join(fixed_pairs)}</tbody></table></details>
<h2>完整性与审计</h2><p>缺少策略案例 {len(report['missing_policy_cases'])} 条；缺少固定增益基准 {len(report['missing_fixed_cases'])} 条。</p><ul>{errors}</ul><p><a href="report.json">完整结果、逐 seed 统计、固定增益配对和原始记录索引</a></p></html>'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True,type=Path)
    parser.add_argument('--protocol',required=True,type=Path)
    parser.add_argument('--split',choices=['validation','test'],required=True)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--checkpoint-stage',choices=['stage1','stage2','selected'])
    parser.add_argument('--selection-checkpoints',type=Path,
                        help='JSON mapping seedN_method to {stage1:path,stage2:path}; writes selection.json using validation only')
    parser.add_argument('--allow-incomplete',action='store_true')
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    if args.selection_checkpoints:
        if args.split!='validation' or args.checkpoint_stage is not None:
            parser.error('Selection requires --split validation and no --checkpoint-stage')
        result=build_selection(args.root,protocol,json.loads(args.selection_checkpoints.read_text()))
        result['protocol_path']=str(args.protocol.resolve());result['protocol_sha256']=sha256(args.protocol)
        destination=args.output or args.root/'selection.json'
        destination.parent.mkdir(parents=True,exist_ok=True)
        with destination.open('x') as stream:
            json.dump(result,stream,indent=2,ensure_ascii=False,allow_nan=False);stream.write('\n')
        print(json.dumps({'selection':str(destination.resolve()),'models':list(result['selected_models'])}))
        return
    result = collect(args.root,protocol,args.split,args.checkpoint_stage)
    result['protocol_path'] = str(args.protocol.resolve())
    result['protocol_sha256'] = sha256(args.protocol)
    destination = args.output or args.root/'reports'/args.split
    destination.mkdir(parents=True,exist_ok=True)
    (destination/'report.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    (destination/'index.html').write_text(render_html(result))
    print(json.dumps({key:result[key] for key in ('status','split','audited_episode_count')},ensure_ascii=False))
    if result['status'] != 'audited_complete' and not args.allow_incomplete:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
