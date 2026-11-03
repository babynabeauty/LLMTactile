"""v1 evaluation on a frozen condition grid: base replay, fixed-gain baselines or a trained policy.

Every controller sees exactly the same episodes (base x material x feasible
level, one F_break each). --base-only replays the base (Astra only, zero
residual); a fixed gain replaces the recorded gain after takeover, residual zero.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from .v1_bases import load_experiment, sha256
from .v1_task import episode_specs

DEFAULT_MATERIALS = [[m, u] for m in (.08, .12, .18) for u in (.3, .45, .7)]
FAILURES = ('break', 'slip', 'collision', 'no_opposing_support', 'hold_failed')


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
    for key in ('object_id', 'hand_shape', 'base_label', 'level'):
        out['by_'+key] = {str(v): stats([r for r in rows if r[key] == v]) for v in sorted({r[key] for r in rows})}
    return out


def assign_specs(env, materials, levels, seed):
    for i, child in enumerate(env.envs):
        specs = episode_specs(range(len(child.bases)), materials, levels, seed*1000+i)
        for s in specs:
            s['base_id'] = child.bases[s.pop('base_index')].base_id
        child.set_episode_specs(specs)


def main(argv=None):
    from .v1_train import METHODS, CPUTrainingPPO, build_environment
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
    p.add_argument('--method', choices=sorted(METHODS), default='tactile')
    args = p.parse_args(argv)
    experiment = load_experiment(args.experiment_json)
    args.output.mkdir(parents=True, exist_ok=False)
    env = build_environment(experiment, {'levels': args.levels}, args.output, seed=args.seed, method=args.method,
        lanes=args.lanes, fixed_gain=args.fixed_gains[0] if args.fixed_gains else None,
        split=args.split, labels=tuple(args.labels))
    summary = dict(experiment_sha256=sha256(args.experiment_json), split=args.split, labels=args.labels,
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
        else:
            model = CPUTrainingPPO.load(args.checkpoint, device='cpu')
            summary.update(checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=sha256(args.checkpoint),
                           method=args.method)
            assign_specs(env, args.materials, args.levels, args.seed)
            controllers[args.method] = run_episodes(env, lambda obs: model.predict(obs, deterministic=True)[0])
        for name, rows in controllers.items():
            with (args.output/f'episodes_{name}.jsonl').open('w') as stream:
                for row in rows:
                    stream.write(json.dumps(row, allow_nan=False)+'\n')
            summary[name] = summarize(rows)
        (args.output/'summary.json').write_text(json.dumps(summary, indent=2))
        print(json.dumps({k: summary[k]['all'] for k in controllers}, indent=2))
    finally:
        env.close()


if __name__ == '__main__':
    main()
