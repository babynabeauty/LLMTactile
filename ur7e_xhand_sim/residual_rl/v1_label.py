"""Label BasePolicyV1 runs S / R / U and write base libraries and experiments (PLAN_v1 section 5).

1 record: execution error, incomplete plan, or hand-object contact before close -> unusable
2 zero-residual replay at 50 g / friction 0.8 / unbreakable: success -> S; a replay that contradicts the
  collection (drop or collision against a collected success, success against a collected failure) is excluded
3 object center in the palm frame at close start vs the demonstration: > 2 cm or yaw > 10 deg -> U
4 repair test: a constant residual added after first contact; any of ~29 candidates succeeds -> R, else U
Train libraries keep S and R; test libraries keep S, R and U. One library per object x hand shape.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from .v1_bases import SCHEMA, Base, write_experiment
from .v1_task import FINGER_JOINTS, HAND_SHAPES, YAW_PERIOD_DEG, SUCCESS_RULE, SamplerConfig, V1Config, hand_shape_joints, wrap_yaw
from .control_protocol import current_control_protocol

GRID = (-.15, -.075, 0., .075, .15)
CONTACT_N = .05
COLLECTION = dict(mass_kg=.05, friction=.8, level=4, f_break_N=float('inf'), astra_level=4,
                  f_min_N=.05*9.81/1.6, window_ratio=float('inf'), narrow_window=False)


def candidates(hand_shape):
    """Thumb flexion x opposing-finger flexion grid, plus thumb rotation and index side swing."""
    fingers = HAND_SHAPES[hand_shape]
    thumb = FINGER_JOINTS['thumb'][1:]
    other = [j for f in fingers if f != 'thumb' for j in FINGER_JOINTS[f][-2:]]
    grid = [{**{j: a for j in thumb}, **{j: b for j in other}} for a in GRID for b in GRID]
    return grid+[{joint: value} for joint in (0, 3) for value in (-.1, .1)]


def candidate_names():
    """Labels for candidates(), in the same order for every hand shape."""
    return [f'thumb {a:+.3f}, fingers {b:+.3f}' for a in GRID for b in GRID]+[
        f'joint {joint} {value:+.1f}' for joint in (0, 3) for value in (-.1, .1)]


def quaternion_matrix(q):
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)], [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                     [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])


def record_check(run, demo):
    """Step 1 and the step-3 geometry, from the bridge logs (privileged evaluation stream)."""
    if 'error' in run or not run.get('complete'):
        return dict(usable=False, reason=run.get('error', 'incomplete plan'))
    bridge = Path(run['bridge_dir'])
    rows = [json.loads(line) for line in (bridge/'control.jsonl').open()]
    evals = [json.loads(line) for line in (bridge/'evaluation.jsonl').open()]
    k = next(i for i, r in enumerate(rows) if r['phase'] == 'close')
    early = max((c['normal_force_N'] for e in evals[:k] for c in e['solver_contacts']
                 if {c['entity_a'], c['entity_b']} == {'cube', 'ur7e_xhand'}), default=0.)
    if early > CONTACT_N:
        return dict(usable=False, reason=f'contact before close ({early:.3f} N)')
    o = rows[k]['observation']
    # Commanded palm frame at the actual wrist position, the frame v1_demo exports object_center_in_palm_m in.
    rotation = quaternion_matrix(demo['palm_quaternion_wxyz'])
    center = rotation.T@(np.asarray(evals[k]['object_position_m'])-np.asarray(o['wrist_position_m']))
    offset = center-np.asarray(demo['object_center_in_palm_m'])
    yaw_error = abs(wrap_yaw(run['placement']['yaw_deg'], YAW_PERIOD_DEG.get(run['object_spec']['shape'])))
    return dict(usable=True, deviation=dict(palm_frame_m=offset.tolist(), world_m=(rotation@offset).tolist(),
                                            distance_m=float(np.linalg.norm(offset)), yaw_deg=float(yaw_error)))


def run_specs(env, specs, act):
    env.set_episode_specs(specs)
    obs, results = env.reset(), []
    while len(results) < len(specs):
        obs, _, _, infos = env.step(act(obs))
        results += [i['episode_metrics'] for i in infos
                    if 'episode_metrics' in i and not i['episode_metrics']['discard']]
    return {r['spec_index']: r for r in results}


def label_group(runs, demo, output, lanes, seed, *, finger_armature=None):
    from .v1_env import V1VecEnv
    checks = {r['run_id']: record_check(r, demo) for r in runs}
    usable = [r for r in runs if checks[r['run_id']]['usable']]
    labels = {r['run_id']: dict(checks[r['run_id']]) for r in runs}
    if not usable:
        return labels
    shape = runs[0]['hand_shape']
    bases = [Base(dict(base_id=r['run_id'], trajectory_path=r['trajectory_path'],
        trajectory_sha256=r['trajectory_sha256'], object_initial_position_m=r['object_initial_position_m'],
        object_initial_quaternion_wxyz=r['object_initial_quaternion_wxyz'], label='U', split='test'), output)
        for r in usable]
    joints = hand_shape_joints(shape)
    env = V1VecEnv(runs[0]['object_spec'], bases, n_envs=lanes, seed=seed, output_dir=output/'label_env',
                   object_id=runs[0]['object_id'], hand_shape=shape, action_joints=joints,
                   config=V1Config(), sampler=SamplerConfig(levels=(4,)), finger_armature=finger_armature)
    try:
        for b in env.excluded_bases:
            labels[b['base_id']].update(usable=False, reason='contact before close in replay')
        replayable = [b for b in env.bases]
        zero = lambda obs: np.zeros((env.num_envs, len(joints)), np.float32)
        replay = run_specs(env, [dict(COLLECTION, base_id=b.base_id) for b in replayable], zero)
        (output/'zero_residual_episodes.json').write_text(json.dumps(replay, indent=2))
        pending, collected = [], {r['run_id']: bool((r.get('score') or {}).get('success')) for r in runs}
        for i, b in enumerate(replayable):
            row, result = labels[b.base_id], replay[i]
            row['zero_residual'] = result
            # Instability, missing support and breakage are stricter v1 criteria, not replay errors.
            if collected[b.base_id] != result['success'] and (
                    not collected[b.base_id] or result['termination_reason'] in ('not_lifted', 'dropped', 'collision')):
                row.update(usable=False, reason=f"replay differs from collection ({result['termination_reason']})")
            elif result['success']:
                row['label'] = 'S'
            elif row['deviation']['distance_m'] > .02 or row['deviation']['yaw_deg'] > 10.:
                row.update(label='U', reason='deviation beyond pre-screen')
            else:
                pending.append(b)
        options = candidates(shape)
        vectors = np.array([[c.get(j, 0.) for j in joints] for c in options])/env.config.residual_bound_rad
        specs = [dict(COLLECTION, base_id=b.base_id, candidate=k) for b in pending for k in range(len(options))]

        def repair(obs):
            action = np.zeros((env.num_envs, len(joints)), np.float32)
            for lane in range(env.num_envs):
                spec = env.episode[lane]
                if 'candidate' in spec and env.peak_region[lane] > CONTACT_N:
                    action[lane] = vectors[spec['candidate']]
            return action
        outcome = run_specs(env, specs, repair) if specs else {}
        (output/'repair_episodes.json').write_text(json.dumps(outcome, indent=2))
        for index, spec in enumerate(specs):
            row = labels[spec['base_id']]
            if outcome[index]['success'] and row.get('label') != 'R':
                row.update(label='R', reason='candidate residual succeeded',
                           repair_residual_rad={str(j): v for j, v in options[spec['candidate']].items()})
        for b in pending:
            if labels[b.base_id].get('label') != 'R':
                labels[b.base_id].update(label='U', reason='no candidate residual succeeded')
    finally:
        env.close()
    return labels


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--collection', type=Path, required=True, help='v1_collect output directory')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--lanes', type=int, default=16)
    p.add_argument('--lanes-per-object', type=int, default=4, help='training lanes per scene')
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--split', choices=('all', 'train', 'test'), default='all',
                   help='Only label the selected placement split')
    p.add_argument('--finger-armature', type=float, default=None,
                   help='Override all twelve finger DOFs in kg m²; leave arm DOFs unchanged')
    args = p.parse_args(argv)
    if args.finger_armature is not None and (not np.isfinite(args.finger_armature) or args.finger_armature < 0):
        p.error('--finger-armature must be finite and nonnegative')
    runs = [json.loads(line) for line in (args.collection/'runs.jsonl').open()]
    if args.split != 'all':
        runs = [r for r in runs if r['split'] == args.split]
    if not runs:
        p.error('No collection runs for the selected split')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output/'control_protocol.json').write_text(json.dumps(current_control_protocol(), indent=2))
    (output/'label_protocol.json').write_text(json.dumps(dict(
        collection=str(args.collection.resolve()), split=args.split, seed=args.seed,
        finger_armature_kg_m2=args.finger_armature, success_rule=SUCCESS_RULE,
        mass_kg=COLLECTION['mass_kg'], friction=COLLECTION['friction'], level=4,
        unbreakable=True), indent=2))
    groups = {}
    for r in runs:
        groups.setdefault((r['object_id'], r['hand_shape']), []).append(r)
    report, train_libraries, test_libraries = {}, [], []
    for (object_id, shape), group in sorted(groups.items()):
        demo = json.loads(Path(group[0]['demo']).read_text())
        print(f'Labeling {object_id}_{shape}: {len(group)} runs', flush=True)
        labels = label_group(group, demo, output/f'{object_id}_{shape}', args.lanes, args.seed,
                             finger_armature=args.finger_armature)
        report[f'{object_id}_{shape}'] = labels
        for split, libraries in (('train', train_libraries), ('test', test_libraries)):
            keep = ('S', 'R') if split == 'train' else ('S', 'R', 'U')
            bases = [dict(base_id=r['run_id'], trajectory_path=r['trajectory_path'],
                trajectory_sha256=r['trajectory_sha256'], object_initial_position_m=r['object_initial_position_m'],
                object_initial_quaternion_wxyz=r['object_initial_quaternion_wxyz'], split=split,
                source='BasePolicyV1', unseen_object=bool(r['placement'].get('unseen_object')),
                edge=bool(r['placement'].get('edge')),
                **{k: v for k, v in labels[r['run_id']].items() if k in ('label', 'deviation', 'repair_residual_rad')})
                for r in group if r['split'] == split and labels[r['run_id']].get('label') in keep]
            if bases:
                path = output/f'{object_id}_{shape}_{split}.json'
                path.write_text(json.dumps(dict(schema=SCHEMA, object_id=object_id, object_spec=group[0]['object_spec'],
                    hand_shape=shape, source=str(args.collection.resolve()), bases=bases,
                    finger_armature_kg_m2=args.finger_armature, success_rule=SUCCESS_RULE), indent=2))
                libraries.append(path)
    (output/'labels.json').write_text(json.dumps(report, indent=2))
    counts = {g: {k: sum(v.get('label') == k for v in labels.values()) for k in ('S', 'R', 'U')} |
              {'unusable': sum(not v['usable'] for v in labels.values()),
               'replay_mismatch': sum(v.get('reason', '').startswith('replay differs') for v in labels.values())}
              for g, labels in report.items()}
    (output/'label_counts.json').write_text(json.dumps(counts, indent=2))
    shapes = sorted({json.loads(p.read_text())['hand_shape'] for p in train_libraries})
    if train_libraries:
        write_experiment(output/'experiment_train.json', train_libraries,
                         lanes_per_object=args.lanes_per_object, name='v1_train')
    if test_libraries:
        write_experiment(output/'experiment_test.json', test_libraries,
                         lanes_per_object=args.lanes_per_object, hand_shapes=shapes or None, name='v1_test')
    print(json.dumps(counts, indent=2))


if __name__ == '__main__':
    main()
