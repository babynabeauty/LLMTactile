"""One "already touching" grasp state per object (5090), the start of every risk_mvp episode.

For each object the palm offset comes from the pregrasp search (best successful env, or for an object
without one, the env that reached a three-finger touch with the least object push). The grasp is closed
in a one-env scene with index/middle at --touch (default 0.5 N) and the thumb at twice that and the state at the end of the
close is saved: joint positions/targets/gains, object pose, contact points and normals in the object frame.
The object regions are then partitioned with the index and middle contacts forced into different regions
(the swap experiment puts the weak spot under one of them).

  python risk_mvp/snapshot.py --search outputs/risk_mvp/pregrasp_v2 --out outputs/risk_mvp/snapshots
Writes <id>.json per object, <id>_snapshot.png (two views + the contacts on the regions) and overview.png.
Send back overview.png and summary.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FINGERS = ('thumb', 'index', 'middle')
K_REGIONS = 5


def swap_window(points, normals=None, R=None, mass=.15, mu=.5, safety=2., margin=.1, mass_margin=.1):
    """How well a three-finger contact set suits the swap experiment, from the allocator itself.
    normals: contact normals in the object frame (default: horizontal radial, as for the search contacts;
    pass the snapshot normals to get the numbers the episode allocator will actually see). R: object
    orientation (object -> world) for the gravity direction, default identity.
    Returns (equal_force, w_index, w_middle): the index/middle force of the risk-blind equal split, and for
    each side the lowest weak-region threshold the oracle allocation can still respect. A good grasp has
    both w_* well below equal_force (room to move load away from either side)."""
    from risk_mvp.allocator import Allocator, Contacts, gravity_wrench
    pos = np.array([points[f] for f in FINGERS], float)
    if normals is None:
        nrm = -pos.copy()
        nrm[:, 2] = 0
    else:
        nrm = np.array([normals[f] for f in FINGERS], float)
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-9)
    C = Contacts(pos, nrm, np.array([0, 1, 2]), list(FINGERS))
    w = gravity_wrench(mass*(1+mass_margin), np.eye(3) if R is None else np.asarray(R, float))
    eq = Allocator(C, 3, mode='equal').solve(w, mu=mu, fmax=8., margin=margin, safety=safety)
    if not eq['feasible']:
        return np.inf, np.inf, np.inf
    out = []
    for reg in (1, 2):
        lo, hi = 0., 8.
        for _ in range(12):          # bisection on the weak threshold
            mid = (lo+hi)/2
            lim = np.full(3, 8.)
            lim[reg] = mid
            ok = Allocator(C, 3, mode='oracle').solve(w, mu=mu, fmax=8., margin=margin, safety=safety, limit=lim)['feasible']
            lo, hi = (lo, mid) if ok else (mid, hi)
        out.append(hi)
    return eq['finger_normal']['index'], out[0], out[1]


def candidate_offsets(search_json: dict):
    """Successful poses ranked by the swap window (worst side's room to move load, relative to the
    equal split), then the three-finger touches. Picking by the thumb sitting between the fingers in y
    alone chose poses where thumb and index both pushed sideways and the allocator needed 3x the force."""
    def score(e):
        p = e['contact_obj_m']
        if not all(f in p for f in FINGERS):
            return np.inf, None
        eqf, wi, wm = swap_window(p)
        return max(wi, wm)/max(eqf, 1e-6), (eqf, wi, wm)
    succ = [(score(e), e) for e in search_json['envs'] if e['success']]
    succ.sort(key=lambda x: (x[0][0], x[1]['drift_mm']))
    out = [(e['offset_m'], 'successful pose, equal split %.2f N, oracle keeps weak index <= %.2f N, weak middle <= %.2f N'
            % sc[1]) for sc, e in succ if sc[1] is not None]
    touched = sorted([e for e in search_json['envs'] if e.get('all_touch_s') is not None and not e['pre_touch']
                      and not e['success']], key=lambda e: e['close_shift_mm'])
    out += [(e['offset_m'], 'three-finger touch without a successful lift') for e in touched]
    if not out:
        raise RuntimeError('no env of the search reached a three-finger touch')
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--search', type=Path, default=ROOT/'outputs/risk_mvp/pregrasp_v2')
    ap.add_argument('--out', type=Path, default=ROOT/'outputs/risk_mvp/snapshots')
    ap.add_argument('--objects', nargs='*', default=None)
    ap.add_argument('--backend', default='gpu')
    ap.add_argument('--max-tries', type=int, default=6)
    ap.add_argument('--touch', type=float, default=.5, help='index/middle force at the snapshot (thumb twice), N')
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    from risk_mvp import objects as O, viz
    from risk_mvp.pregrasp_search import TEMPLATE, CAMERAS, run_episode
    from risk_mvp.regions import object_mesh, partition
    from risk_mvp.scene import GraspScene, cpu
    spec = json.loads(TEMPLATE.read_text())
    summary, tiles = {}, []
    for oid in a.objects or list(O.CATALOG):
        try:
            sj = json.loads((a.search/f'{oid}.json').read_text())
            t_touch = a.touch
            # try candidate poses until one ends the close with all three fingers touching
            for k, (offset, why) in enumerate(candidate_offsets(sj)[:a.max_tries]):
                sc = GraspScene(oid, 1, backend=a.backend, cameras=CAMERAS)
                st = run_episode(sc, spec, np.array([offset]), targets=dict(thumb=2*t_touch, index=t_touch, middle=t_touch),
                                 stop_after_close=True)
                hist = st['close_hist']
                touching = {f: bool(all(h[f]['has'][0] for h in hist)) for f in FINGERS}
                print(f'  {oid} candidate {k}: offset {np.round(np.array(offset)*1000, 1).tolist()} mm, touching {touching}',
                      flush=True)
                if all(touching.values()):
                    why += f' (candidate {k})'
                    break
            c = hist[-1]
            p_o, R_o = sc.object_pose()
            imgs = sc.render()
            from risk_mvp.episode import safe_normal
            pts, nrms = {}, {}
            for f in FINGERS:
                ok = [h for h in hist if h[f]['has'][0]]
                if not ok:
                    pts[f], nrms[f] = st['contacts'][f]['point_obj'][0], st['contacts'][f]['normal_obj'][0]
                    continue
                pts[f] = np.mean([R_o[0].T@(h[f]['point'][0]-p_o[0]) for h in ok], 0)
                nrms[f] = np.mean([R_o[0].T@h[f]['normal'][0] for h in ok], 0)
                nrms[f] = safe_normal(nrms[f], pts[f])
            hull = object_mesh(O.region_spec(oid))
            part = partition(hull, K_REGIONS, seed=0, separate=[pts['index'], pts['middle']])
            regions = {f: int(part.region_of(pts[f])) for f in FINGERS}
            # the window the episode allocator sees (snapshot normals and pose; defaults of swap_check)
            eqf, wi, wm = swap_window(pts, {f: safe_normal(nrms[f], pts[f]) for f in FINGERS}, R_o[0])
            window = dict(equal_N=round(float(eqf), 3), oracle_floor_index_N=round(float(wi), 3),
                          oracle_floor_middle_N=round(float(wm), 3))
            snap = dict(object=oid, offset_m=list(offset), offset_source=why, k_regions=K_REGIONS,
                        partition_seed=part.seed, partition_separate=[pts['index'].tolist(), pts['middle'].tolist()],
                        q=sc.q()[0].tolist(), target=sc.target[0].tolist(), kp=sc.kp_env[0].tolist(),
                        obj_pos_file=cpu(sc.obj.get_pos()).reshape(3).tolist(),
                        obj_quat=cpu(sc.obj.get_quat()).reshape(4).tolist(), obj_center=p_o[0].tolist(),
                        contacts={f: dict(point_obj=pts[f].tolist(), normal_obj=safe_normal(nrms[f], pts[f]).tolist(),
                                          fn=float(st['contacts'][f]['fn'][0]), region=regions[f]) for f in FINGERS},
                        touching_after_settle=touching, all_touch_s=float(st['all_touch_tick'][0]/60),
                        swap_window=window)
            (a.out/f'{oid}.json').write_text(json.dumps(snap, indent=1)+'\n')
            # picture: two views of the snapshot + contacts on the regions
            rows = [dict(success=True, contact_obj_m={f: pts[f].tolist() for f in FINGERS})]
            viz.contacts_on_regions(part, hull, rows, a.out/f'{oid}_regions.png',
                                    f'{oid}: thumb region {regions["thumb"]}, index region {regions["index"]}, '
                                    f'middle region {regions["middle"]}')
            from PIL import Image
            reg = np.asarray(Image.open(a.out/f'{oid}_regions.png').convert('RGB'))
            viz.keyframes([imgs[0], imgs[1], reg], ['snapshot view 0', 'snapshot view 1', 'regions'],
                          a.out/f'{oid}_snapshot.png', cols=3, size=(400, 300))
            tiles.append((imgs[0], f"{oid} I{regions['index']} M{regions['middle']} T{regions['thumb']}"
                                   f"{'' if all(touching.values()) else ' NOT ALL TOUCHING'}"))
            summary[oid] = dict(offset_m=offset, source=why, regions=regions, touching=touching, swap_window=window,
                                touch_N={f: round(float(c[f]['fn'][0]), 3) for f in FINGERS})
        except Exception as exc:
            import traceback
            summary[oid] = dict(error=f'{type(exc).__name__}: {exc}', trace=traceback.format_exc()[-2500:])
            print(summary[oid]['trace'], flush=True)
        print(oid, json.dumps({k: v for k, v in summary[oid].items() if k != 'trace'}), flush=True)
        (a.out/'summary.json').write_text(json.dumps(summary, indent=1)+'\n')
    if tiles:
        viz.keyframes([t[0] for t in tiles], [t[1] for t in tiles], a.out/'overview.png', cols=4)
    print('WROTE', a.out)


if __name__ == '__main__':
    main()
