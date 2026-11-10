"""v1 evaluation on a frozen condition grid: base replay, fixed-gain baselines or a trained policy.

Every controller sees exactly the same episodes (base x material x feasible
level, one F_break each). --base-only replays the base (Astra only, zero
residual); a fixed gain replaces the recorded gain after takeover, residual zero.
--residual-grid runs every constant residual of the S/R/U repair test, from first
contact or from takeover, to bound what the residual action can achieve.
"""
import argparse
import collections
import json
from pathlib import Path

import numpy as np

from .v1_conditioning import LEVEL, conditioning_protocol, require_checkpoint_conditioning
from .v1_bases import load_experiment, sha256
from .control_protocol import current_control_protocol, require_checkpoint_control_compatibility
from .tactile_provenance import require_checkpoint_tactile_compatibility
from .v1_label import CONTACT_N, candidate_names, candidates
from .v1_task import SamplerConfig, episode_specs, minimum_grip_force

# Held-out grid; 0.3 kg at friction 0.3 needs more grip than the hand can give and is left out.
DEFAULT_MATERIALS = [[m, u] for m in (.12, .2, .3) for u in (.3, .45, .7)
                     if minimum_grip_force(m, u) <= SamplerConfig().max_grip_force_N]
FAILURES = ('break', 'dropped', 'collision', 'not_lifted', 'no_opposing_support', 'unstable')


def run_episodes(env, act):
    """Step until every scene finished its specs; filler episodes are dropped."""
    target = sum(len(e.specs) for e in env.envs)
    obs, results = env.reset(), []
    while len(results) < target:
        obs, _, _, infos = env.step(act(obs))
        results += [i['episode_metrics'] for i in infos
                    if 'episode_metrics' in i and not i['episode_metrics']['discard']]
    return results


def summarize(rows):
    def stats(selected):
        if not selected:
            return dict(n=0)
        reasons = [r['termination_reason'] for r in selected]
        return dict(n=len(selected), success=float(np.mean([r['success'] for r in selected])),
            **{f'{f}_rate': float(np.mean([x == f for x in reasons])) for f in FAILURES},
            mean_slip_mm=float(np.mean([r['slip_distance_m'] for r in selected])*1000),
            mean_peak_region_N=float(np.mean([r['peak_region_normal_N'] for r in selected])))
    out = dict(all=stats(rows), T1_level4=stats([r for r in rows if r['level'] == 4]),
               T2_levels1to3=stats([r for r in rows if r['level'] < 4]),
               T2_narrow=stats([r for r in rows if r['level'] < 4 and r['narrow_window']]))
    for key in ('object_id', 'hand_shape', 'base_label', 'level', 'edge_placement', 'unseen_object'):
        out['by_'+key] = {str(v): stats([r for r in rows if r[key] == v]) for v in sorted({r[key] for r in rows})}
    return out


def grid_vectors(child):
    """The repair test's constant residuals as normalized actions in one scene's action space."""
    joints = np.asarray(child.action_joints).tolist()
    return np.array([[c.get(j, 0.) for j in joints] for c in candidates(child.hand_shape)])/child.config.residual_bound_rad


def grid_action(env, vectors, k, after_contact):
    """Candidate k in every lane; with after_contact a lane starts once the hand touches the object."""
    parts = []
    for child, v in zip(env.envs, vectors):
        on = child.peak_region > CONTACT_N if after_contact else np.ones(child.num_envs, dtype=bool)
        parts.append(np.where(on[:, None], v[k], 0.))
    return np.concatenate(parts).astype(np.float32)


def print_grid(controllers):
    """Each candidate's outcomes, then per hand shape, level and material the episodes some candidate solved."""
    names = candidate_names()
    print('%-30s success  drop  break  unstable' % 'candidate')
    for name, rows in zip(names, controllers.values()):
        count = lambda reason: sum(r['termination_reason'] == reason for r in rows)
        print('%-30s %3d/%-3d %4d %6d %9d' % (name, sum(r['success'] for r in rows), len(rows),
              count('dropped'), count('break'), count('unstable')))
    total, solved, wins = collections.defaultdict(set), collections.defaultdict(set), collections.defaultdict(collections.Counter)
    for k, rows in enumerate(controllers.values()):
        for r in rows:
            group, episode = (r['hand_shape'], r['level'], r['mass_kg'], r['friction']), (r['base_id'], r['f_break_N'])
            total[group].add(episode)
            if r['success']:
                solved[group].add(episode)
                wins[group][k] += 1
    print('\nshape   level    kg    mu  solved  best candidate (episodes)')
    for group in sorted(total):
        best = wins[group].most_common(1)
        print('%-7s %5d %5.2f %5.2f  %2d/%-3d  %s' % (*group, len(solved[group]), len(total[group]),
              '%s (%d)' % (names[best[0][0]], best[0][1]) if best else '-'))


def assign_specs(env, materials, levels, seed):
    for i, child in enumerate(env.envs):
        specs = episode_specs(range(len(child.bases)), materials, levels, seed*1000+i, child.sampler)
        for s in specs:
            s['base_id'] = child.bases[s.pop('base_index')].base_id
        child.set_episode_specs(specs)


def main(argv=None):
    from .v1_train import METHODS, CPUTrainingPPO, build_environment, to_json
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment-json', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--split', choices=['train', 'test'], default='test')
    p.add_argument('--labels', nargs='+', default=['S', 'R', 'U'])
    p.add_argument('--materials', type=json.loads, default=DEFAULT_MATERIALS,
                   help='JSON list of [mass_kg, friction] pairs')
    p.add_argument('--levels', type=int, nargs='+', default=[1, 2, 3, 4])
    p.add_argument('--lanes', type=int, default=12)
    p.add_argument('--seed', type=int, default=1)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--base-only', action='store_true')
    mode.add_argument('--fixed-gains', type=float, nargs='+')
    mode.add_argument('--checkpoint', type=Path)
    mode.add_argument('--residual-grid', choices=['contact', 'takeover'],
                      help="the repair test's constant residuals, from first contact or from takeover")
    p.add_argument('--method', choices=sorted(METHODS), default='tactile')
    args = p.parse_args(argv)
    experiment = load_experiment(args.experiment_json)
    require_checkpoint_control_compatibility(args.checkpoint)
    require_checkpoint_conditioning(args.checkpoint, experiment.get("actor_conditioning", LEVEL))
    require_checkpoint_tactile_compatibility(args.checkpoint)
    args.output.mkdir(parents=True, exist_ok=False)
    env = build_environment(experiment, {'levels': args.levels}, args.output, seed=args.seed, method=args.method,
        lanes=args.lanes, fixed_gain=args.fixed_gains[0] if args.fixed_gains else None,
        split=args.split, labels=tuple(args.labels))
    summary = dict(actor_conditioning=conditioning_protocol(experiment.get("actor_conditioning", LEVEL)),
                   experiment_sha256=sha256(args.experiment_json), split=args.split, labels=args.labels,
                   control_protocol=current_control_protocol(),
                   materials=args.materials, levels=args.levels, seed=args.seed)
    try:
        controllers = {}
        zero = lambda obs: np.zeros((env.num_envs, *env.action_space.shape), np.float32)
        if args.base_only:
            assign_specs(env, args.materials, args.levels, args.seed)
            controllers['base_only'] = run_episodes(env, zero)
        elif args.fixed_gains:
            for gain in args.fixed_gains:
                for child in env.envs:
                    child.fixed_gain = gain
                assign_specs(env, args.materials, args.levels, args.seed)
                controllers[f'fixed_gain_{gain:g}'] = run_episodes(env, zero)
        elif args.residual_grid:
            vectors = [grid_vectors(child) for child in env.envs]
            after_contact = args.residual_grid == 'contact'
            for k in range(len(vectors[0])):
                assign_specs(env, args.materials, args.levels, args.seed)
                controllers[f'grid_{k:02d}'] = run_episodes(env, lambda obs, k=k: grid_action(env, vectors, k, after_contact))
            summary.update(residual_grid=args.residual_grid, candidates=candidate_names())
        else:
            model = CPUTrainingPPO.load(args.checkpoint, device='cpu')
            summary.update(checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=sha256(args.checkpoint),
                           method=args.method)
            assign_specs(env, args.materials, args.levels, args.seed)
            controllers[args.method] = run_episodes(env, lambda obs: model.predict(obs, deterministic=True)[0])
        for name, rows in controllers.items():
            with (args.output/f'episodes_{name}.jsonl').open('w') as stream:
                for row in rows:
                    stream.write(to_json(row)+'\n')
            summary[name] = summarize(rows)
        (args.output/'summary.json').write_text(to_json(summary, indent=2))
        if args.residual_grid:
            print_grid(controllers)
        else:
            print(json.dumps({k: summary[k]['all'] for k in controllers}, indent=2))
    finally:
        env.close()


if __name__ == '__main__':
    main()
