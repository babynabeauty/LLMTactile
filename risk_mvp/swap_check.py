"""Minimal swap check (5090): does the allocation matter in simulation at all?

From one object's snapshot, every env gets a weak region under the index finger or under the middle
finger (threshold drawn in --weak-range N), every other region is strong (--strong-range). Methods
(same grasp, same force controller, same lift/hold; only the per-finger targets differ):
  equal    index and middle forced equal (risk-blind)
  tactile  minimal forces with the slip margin, no damage constraint (TacDexGrasp-style, risk-blind)
  oracle   knows the true thresholds (upper reference)
The weak threshold is drawn per side between the oracle floor (lowest threshold the oracle allocation can
respect on this snapshot) and the equal split, so equal should break and oracle should not; a side whose
floor is not below the equal split has no room and gets thresholds around the equal split (has_room false).
A fixed absolute range (first run: 1.2-1.6 N) sat above 065-a's equal split (0.98 N): nothing broke and the
oracle constraint never became active.
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
  swap_traces.png   force traces of one env per method x side, with the weak threshold; red shading =
                    ticks where that env's allocation was infeasible
  swap_inhand.png   object rotation / slip relative to the wrist per method x side
Send back these four PNGs and result.json.
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
    ap.add_argument('--weak-range', type=float, nargs=2, default=None,
                    help='fixed weak threshold range N for both sides; default: per side from the snapshot window')
    ap.add_argument('--window-frac', type=float, nargs=2, default=[.3, .8],
                    help='auto range: floor + frac x (equal - floor), floor = lowest threshold the oracle can respect')
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
    if not all(snap.get('touching_after_settle', {}).values()):
        raise SystemExit(f"{a.object}: not all three fingers touch in the snapshot {snap.get('touching_after_settle')}; "
                         'rerun snapshot.py or pick an object whose snapshot touches with all three')
    part = partition(object_mesh(O.region_spec(a.object)), snap['k_regions'], seed=snap['partition_seed'],
                     separate=snap['partition_separate'])
    region_of = {f: snap['contacts'][f]['region'] for f in FINGERS}
    # the swap window the episode allocator sees: equal-split force and, per side, the lowest weak
    # threshold the oracle can still respect (snapshot contacts, normals and pose; this run's mass/mu/safety)
    from risk_mvp.episode import safe_normal
    from risk_mvp.scene import quat_to_mat
    from risk_mvp.snapshot import swap_window
    cfg = TrialConfig(mu_hat=a.mu_hat, safety=a.safety)
    pts = {f: np.array(snap['contacts'][f]['point_obj']) for f in FINGERS}
    nrms = {f: safe_normal(np.array(snap['contacts'][f]['normal_obj'], float), pts[f]) for f in FINGERS}
    eqf, wi, wm = swap_window(pts, nrms, quat_to_mat(snap['obj_quat']), mass=a.mass, mu=a.mu_hat, safety=a.safety,
                              margin=cfg.margin_N, mass_margin=cfg.mass_margin)
    window = dict(equal_N=float(eqf), floor_N=dict(index=float(wi), middle=float(wm)), weak_range_N={}, has_room={})
    for s, w in (('index', wi), ('middle', wm)):
        if a.weak_range is not None:
            lo, hi = a.weak_range
        elif eqf - w > .1:
            lo, hi = (w + fr*(eqf - w) for fr in a.window_frac)
        else:                        # geometry leaves no room on this side: thresholds around the equal split
            lo, hi = .9*eqf, 1.1*eqf
        window['weak_range_N'][s] = [float(lo), float(hi)]
        window['has_room'][s] = bool(eqf - w > .1)
    print('swap window', json.dumps(window), flush=True)
    rng = np.random.default_rng(a.seed)
    cells = [(m, s) for m in METHODS for s in SIDES]
    modes, sides, thr = [], [], []
    for m, s in cells:
        for _ in range(a.per_cell):
            t = rng.uniform(*a.strong_range, part.k)
            t[region_of[s]] = rng.uniform(*window['weak_range_N'][s])
            modes.append(m)
            sides.append(s)
            thr.append(t)
    thr = np.array(thr)
    E = len(modes)
    sc = GraspScene(a.object, E, backend=a.backend, mass_kg=a.mass)
    sc.set_mass(a.mass)
    sc.set_friction(a.mu)
    res = run_trial(sc, snap, part, modes, thr, cfg, mass_kg=a.mass)
    phase_arr = np.array(res.log['phase'])
    hold = phase_arr == 'hold'
    fn_hold = res.log['fn'][hold].mean(0)                     # (E, 3)
    out = dict(config={k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()},
               regions=region_of, window=window, cells={})
    for m, s in cells:
        idx = [e for e in range(E) if modes[e] == m and sides[e] == s]
        out['cells'][f'{m}/{s}'] = dict(
            n=len(idx), broke=float(res.broke[idx].mean()), dropped=float(res.dropped[idx].mean()),
            success=float(res.success[idx].mean()), feasible=float(res.log['feasible'][idx].mean()),
            hold_N={f: float(fn_hold[idx, i].mean()) for i, f in enumerate(FINGERS)},
            weak_threshold_N=float(np.mean([thr[e, region_of[s]] for e in idx])),
            broken_regions=[res.events[e]['region'] for e in idx if res.events[e]],
            break_time_s=[round(res.events[e]['step']/60, 3) for e in idx if res.events[e]],
            live_region_mismatch=float(res.log['live_region_mismatch'][:, idx].mean()),
            settle_peak_N=float(res.log['settle_peak_N'][idx].max()),
            max_rot_deg=float(res.log['obj_rot_deg'][:, idx].max()),
            end_rot_deg=float(res.log['obj_rot_deg'][-1, idx].mean()),
            max_slip_mm=float(res.log['obj_slip_mm'][:, idx].max()),
            alloc_ok_frac={ph: float(res.log['alloc_ok'][phase_arr == ph][:, idx].mean())
                           for ph in ('preload', 'lift', 'hold')},
            max_eq_slack=float(res.log['eq_slack'][:, idx].max()),
            max_risk_slack=float(res.log['risk_slack'][:, idx].max()))
    (a.out/'result.json').write_text(json.dumps(out, indent=1)+'\n')
    print(json.dumps(out['cells'], indent=1))
    print(f"{'cell':16s} broke dropped  index  middle  max_rot_deg max_slip_mm  settle_peak_N  alloc_ok(hold)")
    for k, v in out['cells'].items():
        print(f"{k:16s} {v['broke']:5.2f} {v['dropped']:7.2f}  {v['hold_N']['index']:5.2f}  {v['hold_N']['middle']:6.2f}"
              f"  {v['max_rot_deg']:11.1f} {v['max_slip_mm']:11.1f}  {v['settle_peak_N']:13.2f}  {v['alloc_ok_frac']['hold']:14.2f}")

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
        ax.plot([0, 1], [window['floor_N'][s] for s in SIDES], 'kv', ms=5, mfc='none',
                label='oracle floor (weak side)')
        ax.axhline(window['equal_N'], color='gray', lw=.6, ls='--', label='planned equal split')
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
            bad = ~res.log['alloc_ok'][:, e]
            if bad.any():
                ax.fill_between(t, 0, 1, where=bad, color='tab:red', alpha=.12, lw=0,
                                transform=ax.get_xaxis_transform(), label='allocator infeasible')
            ev = res.events[e]
            if ev:
                ax.axvline(ev['step']/60, color='r', lw=1.5)
            ax.set_title(f'{m}, weak under {s}' + (' - BROKE' if ev else ''), fontsize=9)
    axs[0, 0].legend(fontsize=7)
    for ax in axs[-1]:
        ax.set_xlabel('s (settle | preload | lift | hold)')
    for ax in axs[:, 0]:
        ax.set_ylabel('normal force N (dashed: target)')
    fig.tight_layout()
    fig.savefig(a.out/'swap_traces.png', dpi=110)
    plt.close(fig)

    # in-hand motion: object rotation / slip relative to the wrist since the preload started,
    # and the fraction of envs whose allocation was infeasible
    rows = (('obj_rot_deg', 'in-hand rotation deg'), ('obj_slip_mm', 'in-hand slip mm'),
            ('alloc_bad', 'infeasible allocation\n(fraction of envs)'))
    fig, axs = plt.subplots(len(rows), len(SIDES), figsize=(5.2*len(SIDES), 7.2), sharex=True, sharey='row')
    for j, s in enumerate(SIDES):
        for i, (key, lab) in enumerate(rows):
            ax = axs[i, j]
            for k, m in enumerate(METHODS):
                idx = [e for e in range(E) if modes[e] == m and sides[e] == s]
                if key == 'alloc_bad':
                    ax.plot(t, 1-res.log['alloc_ok'][:, idx].mean(1), f'C{k}', label=m)
                    ax.set_ylim(-.05, 1.05)
                    continue
                y = res.log[key][:, idx]
                ax.plot(t, np.median(y, 1), f'C{k}', label=m)
                ax.fill_between(t, y.min(1), y.max(1), color=f'C{k}', alpha=.15, lw=0)
            for ts, p in starts:
                ax.axvline(ts, color='gray', lw=.5)
            if j == 0:
                ax.set_ylabel(lab)
            if i == 0:
                ax.set_title(f'weak under {s}', fontsize=9)
    axs[0, 0].legend(fontsize=7, title='median, band = min..max', title_fontsize=7)
    for ax in axs[-1]:
        ax.set_xlabel('s (settle | preload | lift | hold)')
    fig.suptitle(f'{a.object}: does the object turn or slide in the hand?')
    fig.tight_layout()
    fig.savefig(a.out/'swap_inhand.png', dpi=110)
    plt.close(fig)
    print('WROTE', a.out)


if __name__ == '__main__':
    main()
