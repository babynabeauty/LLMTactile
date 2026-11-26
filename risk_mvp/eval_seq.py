"""Closed-loop evaluation (5090): consecutive grasps of the same instance, the belief updated in between.

New instances (seed differs from collect.py's) are drawn as in collect.py (appearance-linked: visible /
hidden / decoy / plain weak spot, mass from what the instance's thresholds allow, true friction above the
hard-cone bound). Every instance is grasped --grasps times by every method, each in its own env, always at
the method's own forces (no exploration scale):
  equal, tactile   risk-blind
  prior            risk allocator, handwritten log-normal belief from Astra's labels, never updated
  model            risk allocator, trained risk model (--model): Astra's labels + geometry + settle-phase
                   touch + the archive of THIS method's earlier grasps of the instance
  model_novlm      the same with a model trained without Astra's answers (--model-novlm), if given
  oracle           knows the thresholds
Before the grasps a short probe (settle phase only) gives every env its settle-phase touch features; they
are the same for all methods (same start forces), so the model's cards are known before the first grasp.

  python risk_mvp/eval_seq.py --model outputs/risk_mvp/risk_collect2/risk_collect.pt \\
      [--model-novlm outputs/risk_mvp/risk_novlm/risk_collect.pt] --out outputs/risk_mvp/eval_seq
Writes episodes.jsonl, summary.json and eval_seq.png (success / broke / dropped vs grasp index per method).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FINGERS = ('thumb', 'index', 'middle')
STABLE_OBJECTS = ['065-a_cups', '065-d_cups', '015_peach', '018_plum', '056_tennis_ball', '061_foam_brick']


def load_model(path):
    import torch
    from risk_mvp.risk_model import RiskConfig, RiskModel
    ck = torch.load(path, map_location='cpu')
    cfg = RiskConfig(**ck['cfg'])
    m = RiskModel(cfg)
    m.load_state_dict(ck['state'])
    return m.eval(), cfg, bool(ck.get('no_vlm', False))


def model_hazard(model, cfg, no_vlm, recs, oid, G, settle, archives, sigma):
    """(E, K, 101) hazards for instance records recs (one per env) with their archives (lists of label
    rows from that env's earlier grasps; the 10 most recent are used)."""
    import torch
    from risk_mvp import collect_data as CD
    from risk_mvp.synthetic import collate
    from risk_mvp.train_risk import run_model
    samples = []
    for rec, (tact, hist, fr), arc in zip(recs, settle, archives):
        obj, vlm_part, extra = CD.instance_obj(rec, oid, cfg, sigma, no_vlm)
        cards = np.concatenate([vlm_part, G, tact, extra], 1).astype(np.float32)
        rows = [dict(region=r, broke=False, lower=1., upper=np.inf, max_force=0., hist=hist[r])
                for r in sorted(set(fr.values()))]                           # carries the history; no label
        samples.append(dict(obj=obj, cards=cards, rows=rows,
                            archive=[CD.archive_row(G, obj['label'], obj['conf'], row) for row in arc[-10:]]))
    with torch.no_grad():
        H = run_model(model, collate(samples, cfg)).numpy()
    k = G.shape[0]
    return H[:, :k]


def run_object(oid, a, labels, manifest, models, rng, out):
    from risk_mvp import collect as C
    from risk_mvp import collect_data as CD
    from risk_mvp.appearance import canonical_partition
    from risk_mvp.episode import TrialConfig, run_trial
    from risk_mvp.scene import GraspScene
    snap = json.loads((a.snapshots/f'{oid}.json').read_text())
    part, pinfo = canonical_partition(oid, a.snapshots)
    if manifest[oid]['partition']['fingerprint'] != pinfo['fingerprint']:
        raise RuntimeError('appearance manifest and snapshot partitions differ')
    m_hold = C.holdable_mass(snap, a.mu_hat, a.safety)
    instances, _ = C.plan_object(oid, part, labels, a.instances, 0, rng, a, snap=snap, m_hold=m_hold)
    methods = list(a.methods)
    for m in ('model', 'model_novlm'):
        if m in methods and models.get(m) is None:
            methods.remove(m)
    envs = [(i, m) for i in range(len(instances)) for m in methods]
    E = len(envs)
    sc = GraspScene(oid, E, backend=a.backend)
    inst = [instances[i] for i, _ in envs]
    thr = np.array([r['thresholds_N'] for r in inst])
    mass = np.array([r['mass_kg'] for r in inst])
    sc.set_mass(mass)
    sc.set_friction(np.array([r['mu'] for r in inst]))
    fr = {f: snap['contacts'][f]['region'] for f in FINGERS}
    G = CD.region_features(manifest[oid])
    cfg = TrialConfig(mu_hat=a.mu_hat, safety=a.safety)
    # probe: settle phase only (same start forces for every method)
    probe = run_trial(sc, snap, part, ['equal']*E, thr, TrialConfig(mu_hat=a.mu_hat, safety=a.safety, preload_ticks=1,
                                                                      lift_ticks=1, hold_ticks=1), mass_kg=mass)
    settle = []
    for e in range(E):
        tact, hist = CD.settle_features(probe.log['fn'][::2, e], probe.log['ft'][::2, e], probe.log['region_F'][::2, e],
                                        fr, part.k)
        settle.append((tact, hist, fr))
    archives = [[] for _ in range(E)]
    prior_H = np.stack([C_prior(r, part.k, a.sigma) for r in inst])
    rows_out = []
    t0 = time.time()
    for g in range(a.grasps):
        hazard = prior_H.copy()
        for name in ('model', 'model_novlm'):
            idx = [e for e, (_, m) in enumerate(envs) if m == name]
            if idx:
                model, mcfg, no_vlm = models[name]
                hazard[idx] = model_hazard(model, mcfg, no_vlm, [inst[e] for e in idx], oid, G,
                                           [settle[e] for e in idx], [archives[e] for e in idx], a.sigma)
        modes = ['risk' if m in ('prior', 'model', 'model_novlm') else m for _, m in envs]
        res = run_trial(sc, snap, part, modes, thr, cfg, mass_kg=mass, hazard=hazard)
        for e, (i, m) in enumerate(envs):
            tact, hist, _ = settle[e]
            archives[e] += CD.archive_rows_of(res.archive[e], hist)
            rows_out.append(dict(object=oid, instance=i, method=m, grasp=g, condition=inst[e]['condition'],
                                 broke=bool(res.broke[e]), dropped=bool(res.dropped[e]), success=bool(res.success[e]),
                                 event=res.events[e], max_rot_deg=float(res.log['obj_rot_deg'][:, e].max()),
                                 alloc_ok_frac=float(res.log['alloc_ok'][:, e].mean()),
                                 hold_N={f: float(res.log['fn'][np.array(res.log['phase']) == 'hold', e, j].mean())
                                         for j, f in enumerate(FINGERS)}))
        print(f'  {oid}: grasp {g+1}/{a.grasps}, {time.time()-t0:.0f} s, ' + ', '.join(
            f"{m} ok {np.mean([res.success[e] for e, (_, mm) in enumerate(envs) if mm == m]):.2f}" for m in methods),
              flush=True)
    with open(out/'episodes.jsonl', 'a') as f:
        for r in rows_out:
            f.write(json.dumps(r)+'\n')
    return methods


def C_prior(rec, k, sigma):
    from risk_mvp import prior as P
    return P.hazard_from_label(dict(object=rec['vlm_object'], regions=rec['vlm']), k, sigma=sigma)


def summarise(out: Path, max_rot_deg):
    rows = [json.loads(l) for l in open(out/'episodes.jsonl')]
    methods = sorted({r['method'] for r in rows}, key=lambda m: ['equal', 'tactile', 'prior', 'model_novlm', 'model',
                                                                 'oracle'].index(m))
    G = max(r['grasp'] for r in rows)+1
    S = dict(n=len(rows), methods={}, max_rot_deg=max_rot_deg)
    for m in methods:
        rm = [r for r in rows if r['method'] == m]
        d = dict(overall={k: float(np.mean([r[k] for r in rm])) for k in ('success', 'broke', 'dropped')},
                 first_grasp={k: float(np.mean([r[k] for r in rm if r['grasp'] == 0])) for k in ('success', 'broke', 'dropped')},
                 by_grasp={k: [float(np.mean([r[k] for r in rm if r['grasp'] == g])) for g in range(G)]
                           for k in ('success', 'broke', 'dropped')},
                 by_condition={c: {k: float(np.mean([r[k] for r in rm if r['condition'] == c]))
                                   for k in ('success', 'broke', 'dropped')}
                               for c in ('plain', 'visible', 'hidden', 'decoy') if any(r['condition'] == c for r in rm)},
                 stable_only={k: float(np.mean([r[k] for r in rm if r['max_rot_deg'] <= max_rot_deg] or [np.nan]))
                              for k in ('success', 'broke', 'dropped')})
        S['methods'][m] = d
    (out/'summary.json').write_text(json.dumps(S, indent=1)+'\n')
    return S, methods, G


def plot(S, methods, G, path):
    import os
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/risk_mvp_mpl')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 4, figsize=(19, 4))
    x = np.arange(1, G+1)
    for ax, k in zip(axs[:3], ('success', 'broke', 'dropped')):
        for j, m in enumerate(methods):
            ax.plot(x, S['methods'][m]['by_grasp'][k], 'o-', ms=3, color=f'C{j}', label=m)
        ax.set_xlabel('grasp of the same instance')
        ax.set_ylabel(f'{k} rate')
        ax.set_title(f'{k} vs grasp index', fontsize=9)
    axs[0].legend(fontsize=7)
    ax = axs[3]
    conds = ['plain', 'visible', 'hidden', 'decoy']
    w = .8/len(methods)
    for j, m in enumerate(methods):
        ax.bar(np.arange(4)+j*w, [S['methods'][m]['by_condition'].get(c, {}).get('success', np.nan) for c in conds], w,
               color=f'C{j}', label=m)
    ax.set_xticks(np.arange(4)+.4-w/2, conds)
    ax.set_ylabel('success rate (all grasps)')
    ax.set_title('success per condition', fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--snapshots', type=Path, default=ROOT/'outputs/risk_mvp/snapshots')
    ap.add_argument('--appearance', type=Path, default=ROOT/'outputs/risk_mvp/appearance')
    ap.add_argument('--vlm-labels', type=Path, default=ROOT/'outputs/risk_mvp/vlm_cues/labels.json')
    ap.add_argument('--model', type=Path, default=None)
    ap.add_argument('--model-novlm', type=Path, default=None)
    ap.add_argument('--out', type=Path, default=ROOT/'outputs/risk_mvp/eval_seq')
    ap.add_argument('--objects', nargs='*', default=STABLE_OBJECTS)
    ap.add_argument('--methods', nargs='*', default=['equal', 'tactile', 'prior', 'model_novlm', 'model', 'oracle'])
    ap.add_argument('--instances', type=int, default=12, help='instances per object')
    ap.add_argument('--grasps', type=int, default=10)
    # instance generation as collect.py
    ap.add_argument('--mass-frac', type=float, nargs=2, default=[.5, .9])
    ap.add_argument('--mass-range', type=float, nargs=2, default=None)
    ap.add_argument('--p-touched', type=float, default=.8)
    ap.add_argument('--mu-range', type=float, nargs=2, default=[.5, .9])
    ap.add_argument('--mu-hat', type=float, default=.5)
    ap.add_argument('--safety', type=float, default=2.)
    ap.add_argument('--sigma', type=float, default=.5)
    ap.add_argument('--scale-range', type=float, nargs=2, default=[1., 1.])
    ap.add_argument('--max-rot-deg', type=float, default=20., help='also report grasps that stayed under this')
    ap.add_argument('--seed', type=int, default=100)
    ap.add_argument('--backend', default='gpu')
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out/'episodes.jsonl').write_text('')
    labels = json.loads(a.vlm_labels.read_text())
    manifest = json.loads((a.appearance/'manifest.json').read_text())
    models = dict(model=load_model(a.model) if a.model else None,
                  model_novlm=load_model(a.model_novlm) if a.model_novlm else None)
    rng = np.random.default_rng(a.seed)
    skipped = {}
    for oid in a.objects:
        try:
            run_object(oid, a, labels, manifest, models, rng, a.out)
        except Exception as exc:                      # noqa: BLE001
            import traceback
            skipped[oid] = f'{type(exc).__name__}: {exc}'
            print(f'{oid} skipped: {skipped[oid]}\n{traceback.format_exc()[-1500:]}', flush=True)
    S, methods, G = summarise(a.out, a.max_rot_deg)
    S['skipped'] = skipped
    S['config'] = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()}
    (a.out/'summary.json').write_text(json.dumps(S, indent=1)+'\n')
    plot(S, methods, G, a.out/'eval_seq.png')
    print(f"{'method':12s} {'success':>8s} {'broke':>7s} {'dropped':>8s}   first grasp: success broke   "
          f"last grasp: success broke")
    for m in methods:
        d = S['methods'][m]
        print(f"{m:12s} {d['overall']['success']:8.2f} {d['overall']['broke']:7.2f} {d['overall']['dropped']:8.2f}   "
              f"{d['first_grasp']['success']:20.2f} {d['first_grasp']['broke']:5.2f}   "
              f"{d['by_grasp']['success'][-1]:19.2f} {d['by_grasp']['broke'][-1]:5.2f}")
    print('success per condition:')
    for m in methods:
        print(f'  {m:12s} ' + '  '.join(f"{c} {v['success']:.2f}" for c, v in S['methods'][m]['by_condition'].items()))
    print('WROTE', a.out)


if __name__ == '__main__':
    main()
