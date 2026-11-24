"""Minimal swap check (5090): does the allocation matter in simulation at all?

From one object's snapshot, every env gets a weak region under the index finger or under the middle
finger (threshold drawn in --weak-range N), every other region is strong (--strong-range). Methods
(same grasp, same force controller, same lift/hold; only the per-finger targets differ):
  equal    index and middle forced equal (risk-blind)
  tactile  minimal forces with the slip margin, no damage constraint (TacDexGrasp-style, risk-blind)
  oracle   knows the true thresholds (upper reference)
Default object 065-a_cups: its thumb sits between index and middle, so both sides leave room to move
load. On a grasp where the thumb faces one finger (e.g. 065-d) that finger carries most of the weight
share and the window on its side nearly vanishes (CPU check with the search contacts, 0.15 kg, mu-hat
0.5, safety 2: 065-a equal 2.03/2.03 N, tactile 2.45/1.52 N, oracle can hold index <= 1.05 N or middle
<= 0.75 N; 065-d oracle needs the weak index threshold >= 1.6 N while equal uses 1.68 N).
The object is made heavier (--mass) and the allocator's friction estimate conservative (--mu-hat) so
the forces needed to hold it with a slip margin reach the weak threshold: that is the narrow window in
which the allocation decides between holding and breaking.

  python risk_mvp/swap_check.py --object 065-a_cups --snapshots outputs/risk_mvp/snapshots --out outputs/risk_mvp/swap_check
Pictures:
  swap_forces.png   hold-phase index/middle force vs weak side, per method (the crossing plot)
  swap_outcomes.png break / drop / success rate per method and weak side
  swap_traces.png   force traces of one env per method x side, with the weak threshold
Send back these three PNGs and result.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

METHODS = ('equal', 'tactile', 'oracle')
SIDES = ('index', 'middle')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--object', default='065-a_cups')
    ap.add_argument('--snapshots', type=Path, default=ROOT/'outputs/risk_mvp/snapshots')
    ap.add_argument('--out', type=Path, default=ROOT/'outputs/risk_mvp/swap_check')
    ap.add_argument('--per-cell', type=int, default=16)
    ap.add_argument('--mass', type=float, default=.15, help='kg (e.g. a filled cup)')
    ap.add_argument('--mu', type=float, default=.6, help='true contact friction')
    ap.add_argument('--mu-hat', type=float, default=.5, help="allocator's friction estimate")
    ap.add_argument('--safety', type=float, default=2.)
    ap.add_argument('--weak-range', type=float, nargs=2, default=[1.2, 1.6])
    ap.add_argument('--strong-range', type=float, nargs=2, default=[4., 8.])
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--backend', default='gpu')
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    from risk_mvp import objects as O
    from risk_mvp.episode import FINGERS, TrialConfig, run_trial
    from risk_mvp.regions import object_mesh, partition
    from risk_mvp.scene import GraspScene
    snap = json.loads((a.snapshots/f'{a.object}.json').read_text())
    part = partition(object_mesh(O.region_spec(a.object)), snap['k_regions'], seed=snap['partition_seed'],
                     separate=snap['partition_separate'])
    region_of = {f: snap['contacts'][f]['region'] for f in FINGERS}
    rng = np.random.default_rng(a.seed)
    cells = [(m, s) for m in METHODS for s in SIDES]
    modes, sides, thr = [], [], []
    for m, s in cells:
        for _ in range(a.per_cell):
            t = rng.uniform(*a.strong_range, part.k)
            t[region_of[s]] = rng.uniform(*a.weak_range)
            modes.append(m)
            sides.append(s)
            thr.append(t)
    thr = np.array(thr)
    E = len(modes)
    sc = GraspScene(a.object, E, backend=a.backend, mass_kg=a.mass)
    sc.set_mass(a.mass)
    sc.set_friction(a.mu)
    cfg = TrialConfig(mu_hat=a.mu_hat, safety=a.safety)
    res = run_trial(sc, snap, part, modes, thr, cfg, mass_kg=a.mass)
    hold = np.array(res.log['phase']) == 'hold'
    fn_hold = res.log['fn'][hold].mean(0)                     # (E, 3)
    out = dict(config={k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()},
               regions=region_of, cells={})
    for m, s in cells:
        idx = [e for e in range(E) if modes[e] == m and sides[e] == s]
        out['cells'][f'{m}/{s}'] = dict(
            n=len(idx), broke=float(res.broke[idx].mean()), dropped=float(res.dropped[idx].mean()),
            success=float(res.success[idx].mean()), feasible=float(res.log['feasible'][idx].mean()),
            hold_N={f: float(fn_hold[idx, i].mean()) for i, f in enumerate(FINGERS)},
            weak_threshold_N=float(np.mean([thr[e, region_of[s]] for e in idx])),
            broken_regions=[res.events[e]['region'] for e in idx if res.events[e]])
    (a.out/'result.json').write_text(json.dumps(out, indent=1)+'\n')
    print(json.dumps(out['cells'], indent=1))

    # ---- pictures ----
    import os
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/risk_mvp_mpl')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, len(METHODS), figsize=(4.2*len(METHODS), 3.8), sharey=True)
    for ax, m in zip(axs, METHODS):
        for i, f, col in ((1, 'index', 'C1'), (2, 'middle', 'C2')):
            ys = [out['cells'][f'{m}/{s}']['hold_N'][f] for s in SIDES]
            ax.plot([0, 1], ys, 'o-', color=col, label=f'{f} force')
        wt = [out['cells'][f'{m}/{s}']['weak_threshold_N'] for s in SIDES]
        ax.plot([0, 1], wt, 'k:', label='weak threshold (mean)')
        ax.set_xticks([0, 1], ['weak under index', 'weak under middle'])
        br = [out['cells'][f'{m}/{s}']['broke'] for s in SIDES]
        ax.set_title(f'{m}\nbroke {br[0]:.0%} / {br[1]:.0%}')
        ax.set_ylabel('hold normal force N')
    axs[0].legend(fontsize=8)
    fig.suptitle(f'{a.object}: does the load move away from the weak side? (lines should cross for oracle)')
    fig.tight_layout()
    fig.savefig(a.out/'swap_forces.png', dpi=110)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 3.6))
    labels = [f'{m}\n{s}' for m, s in cells]
    b = np.array([out['cells'][f'{m}/{s}']['broke'] for m, s in cells])
    d = np.array([out['cells'][f'{m}/{s}']['dropped'] for m, s in cells])
    ok = np.array([out['cells'][f'{m}/{s}']['success'] for m, s in cells])
    x = np.arange(len(cells))
    ax.bar(x, ok, color='tab:green', label='held')
    ax.bar(x, d, bottom=ok, color='tab:orange', label='dropped')
    ax.bar(x, b, bottom=ok+d, color='tab:red', label='broke')
    ax.set_xticks(x, labels, fontsize=8)
    ax.set_ylabel('fraction of envs')
    ax.legend(fontsize=8, loc='upper right')
    ax.set_title(f'{a.object}: outcome per method and weak side ({a.per_cell} envs each)')
    fig.tight_layout()
    fig.savefig(a.out/'swap_outcomes.png', dpi=110)
    plt.close(fig)

    t = np.arange(len(res.log['phase']))/60
    starts = [(t[i], p) for i, p in enumerate(res.log['phase']) if i == 0 or res.log['phase'][i-1] != p]
    fig, axs = plt.subplots(len(SIDES), len(METHODS), figsize=(4.2*len(METHODS), 5.4), sharex=True, sharey=True)
    for j, m in enumerate(METHODS):
        for i, s in enumerate(SIDES):
            ax = axs[i, j]
            e = next(e for e in range(E) if modes[e] == m and sides[e] == s)
            for k, f in enumerate(FINGERS):
                ax.plot(t, res.log['fn'][:, e, k], f'C{k}', label=f)
                ax.plot(t, res.log['target'][:, e, k], f'C{k}--', lw=.7)
            ax.axhline(thr[e, region_of[s]], color='k', ls=':', label='weak threshold')
            for ts, p in starts:
                ax.axvline(ts, color='gray', lw=.5)
            ev = res.events[e]
            if ev:
                ax.axvline(ev['step']/60, color='r', lw=1.5)
            ax.set_title(f'{m}, weak under {s}' + (' - BROKE' if ev else ''), fontsize=9)
    axs[0, 0].legend(fontsize=7)
    for ax in axs[-1]:
        ax.set_xlabel('s (preload | lift | hold)')
    for ax in axs[:, 0]:
        ax.set_ylabel('normal force N (dashed: target)')
    fig.tight_layout()
    fig.savefig(a.out/'swap_traces.png', dpi=110)
    plt.close(fig)
    print('WROTE', a.out)


if __name__ == '__main__':
    main()
