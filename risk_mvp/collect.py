"""Collect grasp episodes for the risk model (5090): appearance-linked instances, censored outcomes.

Per object (its snapshot grasp, its canonical partition):
  instances  appearance.sample_instance (strength from the category; visible / hidden / decoy / plain
             weak spot), object mass log-uniform in --mass-range, true contact friction uniform in
             --mu-range (the allocator's hard-cone friction --mu-hat stays a valid lower bound)
  grasps     --grasps per instance, each with a data policy and a force scale:
               equal / tactile   risk-blind allocations
               vlm               risk allocator with the handwritten log-normal prior from Astra's labels
                                 of the variant this instance shows (prior.py)
               oracle            knows the thresholds (near-threshold survivals)
             scale: 1 with probability 1/2, else log-uniform in --scale-range (squeeze harder, so
             thresholds are crossed and the archive holds breaks as well as survivals)
  episode    episode.run_trial: settle, 1 s preload, lift 5 cm, hold 2 s; DamageJudge archive rows per
             touched region: survived above L, or broke in (L, U]

Only variants that have Astra labels are used (a mark is put only on a region whose cue variant was
asked; the plain variant must be there). Physics uses the plain mesh: the variants differ only in texture.

  python risk_mvp/collect.py --snapshots outputs/risk_mvp/snapshots --appearance outputs/risk_mvp/appearance \\
      --vlm-labels outputs/risk_mvp/vlm_cues/labels.json --out outputs/risk_mvp/collect --instances 30
Writes per object <id>.jsonl (one line per episode), <id>_traces.npz (region force, finger normal and
tangential force, targets; every 2nd tick), instances.json, summary.json and collect_summary.png.
Send back summary.json and collect_summary.png.
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

POLICIES = dict(equal=.2, tactile=.3, vlm=.3, oracle=.2)
FINGERS = ('thumb', 'index', 'middle')


def plan_object(oid, part, labels, n_inst, grasps, rng, a):
    """Instances and their grasps for one object. Returns (instances, envs) or raises if labels are missing."""
    from risk_mvp import appearance as A
    from risk_mvp import prior as P
    if P.label_for(labels, oid, 'plain') is None:
        raise KeyError(f'no Astra labels for {oid}/plain')
    labelled = [r for r in A.mark_regions(part) if P.label_for(labels, oid, f'cue_r{r}') is not None]
    if not labelled:
        raise KeyError(f'no Astra labels for any marked variant of {oid}')
    cond_names = list(A.DEFAULT_MIX)
    cond_p = np.array([A.DEFAULT_MIX[c] for c in cond_names])
    instances, envs = [], []
    for i in range(n_inst):
        cond = cond_names[rng.choice(len(cond_names), p=cond_p/cond_p.sum())]
        kw = {}
        if cond == 'visible':
            kw['weak_region'] = int(rng.choice(labelled))
        elif cond == 'decoy':
            kw['mark_region'] = int(rng.choice(labelled))
        inst = A.sample_instance(oid, part, rng, condition=cond, **kw)
        lab = P.label_for(labels, oid, inst.variant)
        rec = dict(inst.to_json(), instance=i, mass_kg=float(np.exp(rng.uniform(*np.log(a.mass_range)))),
                   mu=float(rng.uniform(*a.mu_range)),
                   vlm={r: dict(label=v['label'], confidence=v['confidence'], reason=v.get('reason', ''))
                        for r, v in lab['regions'].items()},
                   vlm_object=lab.get('object', {}))
        rec['_hazard'] = P.hazard_from_label(lab, part.k, sigma=a.sigma)
        instances.append(rec)
        pol_names = list(POLICIES)
        pol_p = np.array([POLICIES[p] for p in pol_names])
        for g in range(grasps):
            pol = pol_names[rng.choice(len(pol_names), p=pol_p/pol_p.sum())]
            scale = 1. if rng.random() < .5 else float(np.exp(rng.uniform(*np.log(a.scale_range))))
            envs.append(dict(instance=i, grasp=g, policy=pol, scale=scale))
    return instances, envs


def run_object(oid, a, labels, manifest, rng, out: Path):
    from risk_mvp.appearance import canonical_partition
    from risk_mvp.episode import TrialConfig, run_trial
    from risk_mvp.scene import GraspScene
    snap_path = a.snapshots/f'{oid}.json'
    snap = json.loads(snap_path.read_text())
    if not all(snap.get('touching_after_settle', {}).values()):
        raise RuntimeError(f'snapshot does not touch with all three fingers: {snap.get("touching_after_settle")}')
    part, pinfo = canonical_partition(oid, a.snapshots)
    if manifest.get(oid, {}).get('partition', {}).get('fingerprint') != pinfo['fingerprint']:
        raise RuntimeError('appearance manifest uses another partition than the snapshot: rerun appearance.py '
                           'and vlm_query.py after snapshot.py')
    instances, envs = plan_object(oid, part, labels, a.instances, a.grasps, rng, a)
    cfg = TrialConfig(mu_hat=a.mu_hat, safety=a.safety)
    E = min(a.batch_envs, len(envs))
    sc = GraspScene(oid, E, backend=a.backend)
    region_of = {f: snap['contacts'][f]['region'] for f in FINGERS}
    jsonl = open(out/f'{oid}.jsonl', 'w')
    traces = dict(region_F=[], fn=[], ft=[], target=[], env_index=[])
    t0 = time.time()
    for b0 in range(0, len(envs), E):
        batch = envs[b0:b0+E]
        n = len(batch)
        pad = batch+[batch[-1]]*(E-n)                     # a short last batch reruns its last env
        inst = [instances[e['instance']] for e in pad]
        thr = np.array([i['thresholds_N'] for i in inst])
        modes = ['risk' if e['policy'] == 'vlm' else e['policy'] for e in pad]
        hazard = np.stack([i['_hazard'] for i in inst])
        mass = np.array([i['mass_kg'] for i in inst])
        sc.set_mass(mass)
        sc.set_friction(np.array([i['mu'] for i in inst]))
        res = run_trial(sc, snap, part, modes, thr, cfg, mass_kg=mass, hazard=hazard,
                        scale=np.array([e['scale'] for e in pad]))
        phase = np.array(res.log['phase'])
        hold = phase == 'hold'
        for j in range(n):
            e, ins = batch[j], inst[j]
            ev = res.events[j]
            row = dict(object=oid, instance=e['instance'], grasp=e['grasp'], policy=e['policy'], scale=e['scale'],
                       condition=ins['condition'], variant=ins['variant'], weak_region=ins['weak_region'],
                       mark_region=ins['mark_region'], mass_kg=ins['mass_kg'], mu=ins['mu'],
                       thresholds_N=ins['thresholds_N'], finger_region=region_of,
                       broke=bool(res.broke[j]), dropped=bool(res.dropped[j]), success=bool(res.success[j]),
                       event=ev, archive=res.archive[j],
                       hold_N={f: float(res.log['fn'][hold, j, i].mean()) for i, f in enumerate(FINGERS)},
                       max_region_N=res.log['region_F'][:, j].max(0).tolist(),
                       settle_peak_N=float(res.log['settle_peak_N'][j]),
                       max_rot_deg=float(res.log['obj_rot_deg'][:, j].max()),
                       max_slip_mm=float(res.log['obj_slip_mm'][:, j].max()),
                       lifted_mm=float(1000*res.log['lifted_m'][j]),
                       alloc_ok_frac=float(res.log['alloc_ok'][:, j].mean()),
                       trace_index=len(traces['env_index']))
            jsonl.write(json.dumps(row)+'\n')
            traces['region_F'].append(res.log['region_F'][::2, j].astype(np.float16))
            traces['fn'].append(res.log['fn'][::2, j].astype(np.float16))
            traces['ft'].append(res.log['ft'][::2, j].astype(np.float16))
            traces['target'].append(res.log['target'][::2, j].astype(np.float16))
            traces['env_index'].append(b0+j)
        jsonl.flush()
        print(f'  {oid}: {b0+n}/{len(envs)} episodes, {time.time()-t0:.0f} s, this batch broke '
              f'{res.broke[:n].mean():.0%} dropped {res.dropped[:n].mean():.0%}', flush=True)
    jsonl.close()
    np.savez_compressed(out/f'{oid}_traces.npz', phase=np.array(res.log['phase'])[::2],
                        **{k: np.stack(v) for k, v in traces.items()})
    for i in instances:
        i.pop('_hazard')
    return instances


def summarise(out: Path, objects):
    rows = [json.loads(l) for oid in objects if (out/f'{oid}.jsonl').exists() for l in open(out/f'{oid}.jsonl')]
    S = dict(n_episodes=len(rows), objects={})
    for oid in objects:
        r = [x for x in rows if x['object'] == oid]
        if not r:
            continue
        d = dict(n=len(r), broke=np.mean([x['broke'] for x in r]), dropped=np.mean([x['dropped'] for x in r]),
                 by_policy={}, by_condition={},
                 archive_rows=sum(len(x['archive']) for x in r),
                 archive_broke=sum(a['outcome'] == 'broke' for x in r for a in x['archive']),
                 median_max_rot_deg=float(np.median([x['max_rot_deg'] for x in r])))
        for key, field in (('by_policy', 'policy'), ('by_condition', 'condition')):
            for v in sorted({x[field] for x in r}):
                rr = [x for x in r if x[field] == v]
                d[key][v] = dict(n=len(rr), broke=float(np.mean([x['broke'] for x in rr])),
                                 dropped=float(np.mean([x['dropped'] for x in rr])))
        d['broke'], d['dropped'] = float(d['broke']), float(d['dropped'])
        S['objects'][oid] = d
    (out/'summary.json').write_text(json.dumps(S, indent=1)+'\n')
    return rows, S


def plot(rows, S, path: Path):
    import os
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/risk_mvp_mpl')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    objs = list(S['objects'])
    pols = list(POLICIES)
    fig, axs = plt.subplots(1, 2, figsize=(14, 4.6), gridspec_kw=dict(width_ratios=[1.6, 1]))
    ax = axs[0]
    w = .8/len(pols)
    for k, p in enumerate(pols):
        x = np.arange(len(objs))+k*w
        b = [S['objects'][o]['by_policy'].get(p, {}).get('broke', np.nan) for o in objs]
        d = [S['objects'][o]['by_policy'].get(p, {}).get('dropped', np.nan) for o in objs]
        ax.bar(x, b, w, color=f'C{k}', label=f'{p} broke')
        ax.bar(x, d, w, bottom=np.nan_to_num(b), color=f'C{k}', alpha=.35, hatch='//', label=f'{p} dropped')
    ax.set_xticks(np.arange(len(objs))+.4-w/2, [o.split('_', 1)[1] if '_' in o else o for o in objs], fontsize=8)
    ax.set_ylabel('fraction of episodes')
    ax.set_title('outcome per object and data policy (solid: broke, hatched: dropped)', fontsize=9)
    ax.legend(fontsize=6, ncol=2)
    # censoring sanity: per touched region, max sustained force vs its threshold; broke iff above
    ax = axs[1]
    for r in rows:
        thr = np.array(r['thresholds_N'])
        for a_ in r['archive']:
            t = thr[a_['region']]
            if a_['outcome'] == 'broke':
                ax.plot(t, a_['upper_N'], 'v', color='tab:red', ms=3, alpha=.5)
            else:
                ax.plot(t, a_['lower_N'], '^', color='tab:green', ms=3, alpha=.3)
    lim = ax.get_xlim()[1]
    ax.plot([0, lim], [0, lim], 'k:', lw=.8)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('true region threshold N')
    ax.set_ylabel('force N (red: break force, green: survived max)')
    ax.set_title('archive rows: red above the diagonal, green below', fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--snapshots', type=Path, default=ROOT/'outputs/risk_mvp/snapshots')
    ap.add_argument('--appearance', type=Path, default=ROOT/'outputs/risk_mvp/appearance')
    ap.add_argument('--vlm-labels', type=Path, default=ROOT/'outputs/risk_mvp/vlm_cues/labels.json')
    ap.add_argument('--out', type=Path, default=ROOT/'outputs/risk_mvp/collect')
    ap.add_argument('--objects', nargs='*', default=None)
    ap.add_argument('--instances', type=int, default=30, help='instances per object')
    ap.add_argument('--grasps', type=int, default=10, help='grasps per instance')
    ap.add_argument('--batch-envs', type=int, default=120)
    ap.add_argument('--mass-range', type=float, nargs=2, default=[.05, .25], help='kg, log-uniform')
    ap.add_argument('--mu-range', type=float, nargs=2, default=[.5, .9], help='true friction, uniform')
    ap.add_argument('--mu-hat', type=float, default=.5, help='allocator hard-cone friction (lower bound)')
    ap.add_argument('--safety', type=float, default=2.)
    ap.add_argument('--sigma', type=float, default=.5, help='spread of the handwritten VLM prior')
    ap.add_argument('--scale-range', type=float, nargs=2, default=[1., 2.5])
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--backend', default='gpu')
    a = ap.parse_args()
    if a.mu_range[0] < a.mu_hat:
        raise SystemExit('--mu-range must start at or above --mu-hat (the hard cone must be a friction lower bound)')
    a.out.mkdir(parents=True, exist_ok=True)
    from risk_mvp import objects as O
    labels = json.loads(a.vlm_labels.read_text())
    manifest = json.loads((a.appearance/'manifest.json').read_text())
    rng = np.random.default_rng(a.seed)
    objects = a.objects or list(O.CATALOG)
    all_inst, done, skipped = {}, [], {}
    for oid in objects:
        try:
            all_inst[oid] = run_object(oid, a, labels, manifest, rng, a.out)
            done.append(oid)
        except Exception as exc:                      # noqa: BLE001
            skipped[oid] = f'{type(exc).__name__}: {exc}'
            print(f'{oid} skipped: {skipped[oid]}', flush=True)
        (a.out/'instances.json').write_text(json.dumps(all_inst, indent=1)+'\n')
    rows, S = summarise(a.out, done)
    S['skipped'] = skipped
    S['config'] = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()}
    (a.out/'summary.json').write_text(json.dumps(S, indent=1)+'\n')
    if rows:
        plot(rows, S, a.out/'collect_summary.png')
    print(f"{'object':18s} n     broke  dropped  archive rows (broke)")
    for oid, d in S['objects'].items():
        print(f"{oid:18s} {d['n']:<5d} {d['broke']:5.2f}  {d['dropped']:6.2f}   {d['archive_rows']} ({d['archive_broke']})")
    for oid, why in skipped.items():
        print(f'{oid:18s} skipped: {why}')
    print('WROTE', a.out)


if __name__ == '__main__':
    main()
