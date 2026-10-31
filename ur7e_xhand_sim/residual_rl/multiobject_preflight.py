"""Training-material feasibility replay of explicitly verified demonstrations.

No policy is trained or selected and no validation/test outcome is read. Each
shape uses one 12-lane scene and the identical material-specific prefix for
fixed gain 1 and fixed gain 3. Gain 3 retains the ordinary action slew limit.
Run explicitly with --experiment-json and a new --output directory.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import traceback

import numpy as np

from .multiobject_protocol import validate_protocol
from .multiobject_report import (audit_environment, audit_recorded_episode,
    sha256, summarize_group, trace_statistics)


GAINS = (1., 3.)
PHYSICS_SEED = 7000


def durable_json(path, value):
    """Atomic, fsynced status/report replacement, including on interruption."""
    path = Path(path)
    temporary = path.with_name(path.name+'.tmp')
    with temporary.open('w') as stream:
        json.dump(value,stream,indent=2,allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    descriptor = os.open(path.parent,os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def finite_tree(value, label):
    if isinstance(value,dict):
        for key,item in value.items():
            finite_tree(item,f'{label}.{key}')
    elif isinstance(value,(list,tuple)):
        for i,item in enumerate(value):
            finite_tree(item,f'{label}[{i}]')
    elif isinstance(value,(np.ndarray,np.number,int,float,bool)):
        array = np.asarray(value)
        if array.dtype.kind not in 'biufc' or not np.isfinite(array).all():
            raise FloatingPointError(f'Nonfinite or nonnumeric physical data: {label}')


def load_frozen_experiment(path):
    path = Path(path).expanduser().resolve()
    protocol = json.loads(path.read_text())
    if protocol.get('status') != 'frozen_ready_for_training':
        raise ValueError('Preflight requires all verified, frozen demonstrations')
    for obj in protocol['objects']:
        trajectory = Path(obj.get('trajectory_path') or '').expanduser()
        if not trajectory.is_absolute():
            trajectory = path.parent/trajectory
        obj['trajectory_path'] = str(trajectory.resolve())
        evidence = obj.get('demonstration_evidence') or {}
        direct = (evidence.get('demonstration_source') in
                  ('direct_teacher_search', 'direct_teacher_frozen_replay')
                  and evidence.get('privileged_teacher') is True)
        if evidence.get('actual_astra_api') is not True and not direct:
            raise ValueError('Every frozen demonstration needs actual Astra actions or audited direct-teacher provenance')
    # This validates the complete frozen hash/evidence chain, including the
    # explicitly declared direct replay source; the driver label alone is not proof.
    validate_protocol(protocol,require_demonstrations=True)
    return protocol


class FiniteReplayGuard:
    """Fail on invalid telemetry immediately, not on an ordinary failed grasp."""
    def __init__(self, env):
        self.env = env

    def __getattr__(self, name):
        return getattr(self.env,name)

    def reset(self):
        observation = self.env.reset()
        finite_tree(observation,'reset_observation')
        return observation

    def step(self, action):
        finite_tree(action,'action')
        result = self.env.step(action)
        finite_tree(result,'step')
        finite_tree(self.env.last_interval_metrics,'physical_interval')
        if np.max(self.env.last_interval_metrics['contact_friction_error']) > 1e-5:
            raise ValueError('Actual contact friction differs from requested training material')
        return result


def audit_prefix(environment_path, conditions, obj, protocol):
    physical = json.loads(environment_path.read_text())
    expected = [dict(object_shape=obj['object_spec']['shape'],
                     mass_kg=row['mass_kg'],friction=row['friction']) for row in conditions]
    errors = audit_environment(physical,expected,protocol)
    if errors:
        raise ValueError('Invalid physical readbacks: '+'; '.join(errors))
    if physical.get('mass_and_inertia_readback_verified') is not True:
        raise ValueError('Missing verified solver mass/inertia readback')
    audit = physical.get('prefix_physics_audit',{})
    trace_path = environment_path.parent/audit.get('path','prefix_physics_trace.npz')
    with np.load(trace_path,allow_pickle=False) as trace:
        for key in trace.files:
            finite_tree(trace[key],f'prefix.{key}')
        if trace['clearance_m'].shape[1] != len(conditions):
            raise ValueError('Prefix trace lane count differs from training materials')
        if np.max(trace['contact_friction_error']) > 1e-5:
            raise ValueError('Invalid friction during physical prefix replay')
        collision = trace['fault'].astype(bool).any(axis=0).tolist()
    return dict(environment_protocol_sha256=sha256(environment_path),
                prefix_trace_sha256=sha256(trace_path),
                collision_per_lane=collision,physical_readbacks_verified=True,
                collision_interpretation='Prefix collisions are reported separately; original residual task labels are unchanged.')


def audit_result(result, result_dir, protocol):
    finite_tree(result,'evaluation')
    with np.load(result_dir/'trace.npz',allow_pickle=False) as trace:
        for key in trace.files:
            finite_tree(trace[key],f'trace.{key}')
        calculated = trace_statistics(trace,result['hold_interval_s'])
    if len(calculated) != len(result['episodes']):
        raise ValueError('Episode and physical trace lane counts differ')
    errors = []
    for lane,(recorded,truth) in enumerate(zip(result['episodes'],calculated)):
        errors += [f'lane {lane}: {error}' for error in audit_recorded_episode(recorded,truth)]
    physical = json.loads(Path(result['environment_protocol_path']).read_text())
    errors += audit_environment(physical,result['episodes'],protocol)
    if errors:
        raise ValueError('Invalid replay audit: '+'; '.join(errors))
    prefix = json.loads((result_dir/'prefix_consistency.json').read_text())
    finite_tree(prefix,'prefix_consistency')
    if not prefix or any(value != 0 for value in prefix.values()):
        raise ValueError('Gain controllers have unequal physical observation prefixes')
    audit = dict(passed=True,trace_recomputed=calculated,
                 trace_sha256=sha256(result_dir/'trace.npz'),prefix_equal=True)
    durable_json(result_dir/'preflight_audit.json',audit)
    return audit


def summarize(results, prefix_audits):
    groups,by_case = {},{}
    for result in results:
        key = f"{result['object_id']}/{result['controller']}"
        groups[key] = summarize_group(result['episodes'])
        for episode in result['episodes']:
            case = (episode['object_id'],episode['condition_id'])
            row = by_case.setdefault(case,dict(object_id=case[0],condition_id=case[1],
                mass_kg=episode['mass_kg'],friction=episode['friction'],controllers={}))
            row['controllers'][result['controller']] = dict(episode)
    cases = []
    for case,row in sorted(by_case.items()):
        maximum = row['controllers'].get('fixed_3')
        row['fixed_gain_3_task_success'] = maximum['task_success'] if maximum else None
        row['fixed_gain_3_strict_success'] = maximum['strict_success'] if maximum else None
        cases.append(row)
    failed = sum(not e['task_success'] for result in results for e in result['episodes'])
    return dict(training=False,split='train_preflight',fixed_gains=list(GAINS),
        completed_batches=len(results),episodes=sum(r['num_episodes'] for r in results),
        grasp_failures=failed,groups=groups,cases=cases,prefix_audits=prefix_audits,
        interpretation='Fixed-gain feasibility checks on training conditions only. A failed gain-3 replay does not prove that all adaptive policies bounded by 3 are infeasible. Failed grabs remain in success denominators; post-drop low force is not force efficiency.',
        gain_3_semantics='Requested gain 3 from takeover; applied gain starts at the recorded source prefix gain and follows the same configured slew limit as training.',
        success_criterion='Unchanged task and strict criteria over the complete final 3 seconds; earlier commands and original hold phase labels are preserved.')


def run(experiment_path, output):
    experiment_path = Path(experiment_path).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True,exist_ok=False)
    results,prefix_audits,env = [],{},None
    status = {'started_utc':datetime.now(timezone.utc).isoformat(),
              'training':False,'split':'train_preflight','completed_batches':0}
    def update(state,**fields):
        status.update(status=state,**fields)
        durable_json(output/'status.json',status)
    try:
        update('validating')
        protocol = load_frozen_experiment(experiment_path)
        # These imports may load simulation libraries, but no scene exists until
        # this explicit CLI execution. Importing this module is CPU-only.
        from .adaptive_env import AdaptiveConfig
        from .multiobject_env import MultiObjectVecEnv
        from .multiobject_benchmark import evaluate_controller
        from .multiobject_train import source_snapshot
        import torch
        torch.set_num_threads(4)
        config = AdaptiveConfig(**protocol['training']['stages'][-1]['config'])
        if config.gain_max != 3. or not config.gain_min <= 1.:
            raise ValueError('Preflight requires the frozen action maximum 3 and nominal gain 1')
        conditions = protocol['splits']['train']
        durable_json(output/'manifest.json',dict(
            experiment_path=str(experiment_path),experiment_sha256=sha256(experiment_path),
            experiment=protocol,source_sha256=source_snapshot(output),
            training=False,split='train_preflight',conditions=conditions,
            fixed_gains=list(GAINS),physics_seed=PHYSICS_SEED,config=asdict(config),
            lanes_per_shape=12,scene_schedule='one shape at a time; same-scene reset for gain 1 and gain 3',
            validation_test_outcomes_read=False,checkpoint_selection=False))
        for obj in protocol['objects']:
            object_id = obj['object_id']
            update('initializing',object_id=object_id,controller=None)
            env = MultiObjectVecEnv(object_spec=obj['object_spec'],trajectory_path=obj['trajectory_path'],
                n_envs=12,seed=PHYSICS_SEED,mode='eval',config=config,audit_prefix=True,
                output_dir=output/'environment'/object_id,
                material_conditions=[[r['mass_kg'],r['friction']] for r in conditions])
            prefix_audits[object_id] = audit_prefix(env.output_dir/'environment_protocol.json',conditions,obj,protocol)
            durable_json(output/'prefix_audits.json',prefix_audits)
            guarded,reference = FiniteReplayGuard(env),None
            for gain in GAINS:
                name = f'fixed_{gain:g}'
                update('running',controller=name)
                destination = output/name/f'shape_{object_id}'
                result,prefix = evaluate_controller(guarded,name=name,model=None,fixed_gain=gain,
                    output=destination,conditions=conditions,object_id=object_id,
                    split='train_preflight',training_seed=0,method=name,
                    prefix_reference=reference,save_tactile=True)
                audit_result(result,destination,protocol)
                if reference is None:
                    reference = prefix
                results.append(result)
                durable_json(output/'results.json',results)
                durable_json(output/'summary.json',summarize(results,prefix_audits))
                update('running',completed_batches=len(results),last_task_successes=result['task_successes'],
                       last_strict_successes=result['strict_successes'])
                print(json.dumps(dict(object_id=object_id,controller=name,
                    task_successes=result['task_successes'],strict_successes=result['strict_successes'],
                    episodes=result['num_episodes'])),flush=True)
            env.close()
            env = None
        summary = summarize(results,prefix_audits)
        update('completed_with_grasp_failures' if summary['grasp_failures'] else 'completed',
               episodes=summary['episodes'],grasp_failures=summary['grasp_failures'],engineering_valid=True,
               completed_utc=datetime.now(timezone.utc).isoformat())
        return summary
    except BaseException as exc:
        update('engineering_error',error=repr(exc),traceback=traceback.format_exc(),engineering_valid=False)
        raise
    finally:
        if env is not None:
            env.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment-json',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args(argv)
    run(args.experiment_json,args.output)


if __name__ == '__main__':
    main()
