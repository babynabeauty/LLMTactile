"""One "already touching" grasp state per object (5090), the start of every risk_mvp episode.

For each object the palm offset comes from the pregrasp search (best successful env, or for an object
without one, the env that reached a three-finger touch with the least object push). The grasp is closed
in a one-env scene with all three targets at the light touch force (0.2 N) and the state at the end of the
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


def pick_offset(search_json: dict):
    if search_json.get('best'):
        return search_json['best']['offset_m'], 'best successful pose'
    touched = [e for e in search_json['envs'] if e.get('all_touch_s') is not None and not e['pre_touch']]
    if not touched:
        raise RuntimeError('no env of the search reached a three-finger touch')
    e = min(touched, key=lambda e: (e['close_shift_mm'], abs(np.array(e['offset_m'])).sum()))
    return e['offset_m'], 'three-finger touch with least push (no successful lift in the search)'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--search', type=Path, default=ROOT/'outputs/risk_mvp/pregrasp_v2')
    ap.add_argument('--out', type=Path, default=ROOT/'outputs/risk_mvp/snapshots')
    ap.add_argument('--objects', nargs='*', default=None)
    ap.add_argument('--backend', default='gpu')
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    from risk_mvp import objects as O, viz
    from risk_mvp.pregrasp_search import TEMPLATE, TOUCH_N, CAMERAS, run_episode
    from risk_mvp.regions import object_mesh, partition
    from risk_mvp.scene import GraspScene
    spec = json.loads(TEMPLATE.read_text())
    summary, tiles = {}, []
    for oid in a.objects or list(O.CATALOG):
        try:
            sj = json.loads((a.search/f'{oid}.json').read_text())
            offset, why = pick_offset(sj)
            sc = GraspScene(oid, 1, backend=a.backend, cameras=CAMERAS)
            st = run_episode(sc, spec, np.array([offset]), targets={f: TOUCH_N for f in FINGERS},
                             stop_after_close=True)
            for _ in range(20):          # settle at the touch force
                sc.step(sc.target)
            c = sc.finger_contacts()
            touching = {f: bool(c[f]['has'][0]) for f in FINGERS}
            imgs = sc.render()
            pts = {f: st['contacts'][f]['point_obj'][0] for f in FINGERS}
            hull = object_mesh(O.region_spec(oid))
            part = partition(hull, K_REGIONS, seed=0, separate=[pts['index'], pts['middle']])
            regions = {f: int(part.region_of(pts[f])) for f in FINGERS}
            snap = dict(object=oid, offset_m=list(offset), offset_source=why, k_regions=K_REGIONS,
                        partition_seed=part.seed, partition_separate=[pts['index'].tolist(), pts['middle'].tolist()],
                        q=st['q'][0].tolist(), target=st['target'][0].tolist(), kp=st['kp'][0].tolist(),
                        obj_pos_file=st['obj_pos_file'][0].tolist(), obj_quat=st['obj_quat'][0].tolist(),
                        obj_center=st['obj_center'][0].tolist(),
                        contacts={f: dict(point_obj=pts[f].tolist(), normal_obj=st['contacts'][f]['normal_obj'][0].tolist(),
                                          fn=float(st['contacts'][f]['fn'][0]), region=regions[f]) for f in FINGERS},
                        touching_after_settle=touching, all_touch_s=float(st['all_touch_tick'][0]/60))
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
            summary[oid] = dict(offset_m=offset, source=why, regions=regions, touching=touching,
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
