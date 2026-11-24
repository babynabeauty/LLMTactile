"""Tripod pregrasp search per YCB object (5090): which palm pose gives a stable three-finger lift.

Template: configs/v1/demos/cuboid_tripod.json (palm orientation, open/closed hand, palm offset from the
object centre, grip gain 3). For each object the palm offset is shifted by the object's half-width
difference along x (the thumb-fingers axis of the template) plus a grid dx x dy x dz, one grid point per
env. Episode (60 Hz control, 1/300 s physics):
  settle 10 -> approach (joint-space to pre-pose 12 cm above, hand opening) 120 -> descend 60
  -> sag correction 20 -> force-controlled close 120 (thumb 2 N, index 1 N, middle 1 N)
  -> lift 5 cm 60 -> hold 120.
Score per env: touched the object before closing, all three fingers in contact during the hold,
lift height, object drift relative to the wrist during the hold, index/middle contact points in the
object frame (they must fall in different regions for the swap experiment), table/palm contact.

  python risk_mvp/pregrasp_search.py --out outputs/risk_mvp/pregrasp            # all catalogue objects
  python risk_mvp/pregrasp_search.py --objects 012_strawberry --out ...
Send back outputs/risk_mvp/pregrasp/summary.json (best pose per object; per-env table in <id>.json).
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TEMPLATE = ROOT/'configs/v1/demos/cuboid_tripod.json'
TEMPLATE_HALF_X = .045/2
FINGERS = ('thumb', 'index', 'middle')
TARGET_N = dict(thumb=2., index=1., middle=1.)


def grid(dx=(-.015, -.0075, 0., .0075, .015), dy=(-.01, 0., .01), dz=(-.015, 0., .015)):
    return np.array(list(itertools.product(dx, dy, dz)))


def run_object(oid, out_dir, backend='gpu'):
    from risk_mvp.scene import GraspScene, CONTROL_SUBSTEPS
    from risk_mvp.force_ctrl import BatchForceController, FINGER_JOINTS
    spec = json.loads(TEMPLATE.read_text())
    offsets = grid()
    E = len(offsets)
    t0 = time.time()
    sc = GraspScene(oid, E, backend=backend)
    build_s = time.time()-t0
    half_x = sc.info['extent_m'][0]/2
    palm_offset = np.asarray(spec['palm_offset_m'])+np.array([half_x-TEMPLATE_HALF_X, 0, 0])
    quat = np.tile(np.asarray(spec['palm_quaternion_wxyz'], float), (E, 1))
    open_h, closed_h = np.asarray(spec['open_hand_rad']), np.asarray(spec['closed_hand_rad'])
    sc.reset()
    sc.set_gains()
    center0, _ = sc.object_pose()
    align = center0+palm_offset+offsets
    pre = align+np.array([0, 0, .12])

    def move_arm(q_from, q_to, ticks, hand_from=None, hand_to=None):
        for i in range(1, ticks+1):
            u = i/ticks
            s = u*u*(3-2*u)
            tgt = sc.target.copy()
            tgt[:, :6] = q_from+s*(q_to-q_from)
            if hand_from is not None:
                tgt[:, 6:] = hand_from+s*(hand_to-hand_from)
            sc.step(tgt)

    for _ in range(10):
        sc.step(sc.target)
    q_pre, err_pre = sc.ik(pre, quat)
    move_arm(sc.q()[:, :6], q_pre, 120, np.zeros((E, 12)), np.tile(open_h, (E, 1)))
    q_al, err_al = sc.ik(align, quat)
    move_arm(q_pre, q_al, 60)
    for _ in range(10):
        sc.step(sc.target)
    # sag correction: re-aim so the measured wrist lands on the planned align position
    w, _ = sc.wrist_pose()
    sag = align-w
    q_al2, _ = sc.ik(align+sag, quat)
    move_arm(q_al, q_al2, 20)
    c = sc.finger_contacts()
    pre_touch = np.any([c[f]['has'] for f in FINGERS], 0) | (c['other_robot_obj_N'] > .05)

    # force-controlled close
    sc.set_gains(finger_gain=float(spec.get('grip_gain', 1.)))
    ctrl = {f: BatchForceController(E) for f in FINGERS}
    hand_names = sc.names[6:]
    closed_cols = {f: [hand_names.index(j) for j in FINGER_JOINTS[f]] for f in FINGERS}

    def hand_tick(log):
        c = sc.finger_contacts()
        J = sc.finger_jacobians()
        q = sc.q()
        tgt = sc.target.copy()
        for f in FINGERS:
            cols = sc.finger_cols[f]
            qt, _ = ctrl[f].step(np.full(E, TARGET_N[f]), q[:, cols], sc.kp_env[:, cols], sc.effort[cols],
                                 sc.lower[cols], sc.upper[cols], J[f][0], J[f][1], c[f]['point'], c[f]['normal'],
                                 c[f]['fn'], c[f]['has'], np.tile(closed_h[closed_cols[f]], (E, 1)))
            tgt[:, cols] = qt
        if log is not None:
            log.append(c)
        return tgt

    # joints the controller does not drive (thumb rotation, index abduction, ring, pinky) go to the
    # demo's closed values, as in the template close
    driven = {6+c for f in FINGERS for c in closed_cols[f]}
    others = [k for k in range(6, 18) if k not in driven]
    sc.target[:, others] = closed_h[[k-6 for k in others]]
    for _ in range(120):
        sc.step(hand_tick(None))
    obj_before, _ = sc.object_pose()
    lifted = align+sag+np.array([0, 0, .05])
    q_lift, _ = sc.ik(lifted, quat)
    q_now = sc.target[:, :6].copy()
    for i in range(1, 61):
        tgt = hand_tick(None)
        u = i/60
        tgt[:, :6] = q_now+(u*u*(3-2*u))*(q_lift-q_now)
        sc.step(tgt)
    hold, rel = [], []
    for _ in range(120):
        sc.step(hand_tick(hold))
        p, R = sc.object_pose()
        wp, wq = sc.wrist_pose()
        from risk_mvp.scene import quat_to_mat
        Rw = quat_to_mat(wq)
        rel.append((np.einsum('eji,ej->ei', Rw, p-wp), np.einsum('eji,ejk->eik', Rw, R)))
    p_end, R_end = sc.object_pose()
    lift_h = p_end[:, 2]-obj_before[:, 2]
    drift = np.linalg.norm(rel[-1][0]-rel[0][0], axis=1)
    rot = np.degrees(np.arccos(np.clip((np.einsum('eij,eij->e', rel[-1][1], rel[0][1])-1)/2, -1, 1)))
    rows = []
    for e in range(E):
        fn = {f: float(np.mean([h[f]['fn'][e] for h in hold])) for f in FINGERS}
        touch = {f: float(np.mean([h[f]['has'][e] for h in hold])) for f in FINGERS}
        pts = {}
        for f in ('index', 'middle', 'thumb'):
            ws = [(h[f]['point'][e], h[f]['fn'][e]) for h in hold if h[f]['has'][e]]
            if ws:
                pw = np.mean([x for x, _ in ws], 0)
                pts[f] = (R_end[e].T@(pw-p_end[e])).round(4).tolist()
        ok = bool(not pre_touch[e] and all(v > .95 for v in touch.values()) and lift_h[e] > .035
                  and drift[e] < .003 and rot[e] < 10.)
        rows.append(dict(env=e, offset_m=offsets[e].tolist(), success=ok, pre_touch=bool(pre_touch[e]),
                         lift_m=float(lift_h[e]), drift_mm=float(drift[e]*1000), rot_deg=float(rot[e]),
                         hold_normal_N=fn, hold_contact_frac=touch, contact_obj_m=pts,
                         table_robot_N=float(np.mean([h['table_robot_N'][e] for h in hold])),
                         palm_obj_N=float(np.mean([h['other_robot_obj_N'][e] for h in hold])),
                         ik_err_pre=float(np.linalg.norm(err_pre[e, :3])), ik_err_align=float(np.linalg.norm(err_al[e, :3])),
                         sag_mm=(sag[e]*1000).round(2).tolist()))
    succ = [r for r in rows if r['success']]
    best = min(succ, key=lambda r: (r['drift_mm']+r['rot_deg']/5, abs(np.array(r['offset_m'])).sum())) if succ else None
    res = dict(object=oid, n_envs=E, build_s=build_s, wall_s=time.time()-t0, palm_offset_base_m=palm_offset.tolist(),
               palm_quaternion_wxyz=spec['palm_quaternion_wxyz'], open_hand_rad=open_h.tolist(),
               closed_hand_rad=closed_h.tolist(), grip_gain=spec.get('grip_gain', 1.), n_success=len(succ),
               best=best, envs=rows)
    (out_dir/f'{oid}.json').write_text(json.dumps(res, indent=1)+'\n')
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, default=ROOT/'outputs/risk_mvp/pregrasp')
    ap.add_argument('--objects', nargs='*', default=None)
    ap.add_argument('--backend', default='gpu')
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    from risk_mvp import objects as O
    O.download()
    ids = a.objects or list(O.CATALOG)
    summary = {}
    for oid in ids:
        # one object per process keeps Genesis scenes independent; run sequentially in-process here
        try:
            r = run_object(oid, a.out, a.backend)
            summary[oid] = dict(n_success=r['n_success'], of=r['n_envs'], best=r['best'], wall_s=r['wall_s'],
                                failure_counts=dict(
                                    pre_touch=sum(e['pre_touch'] for e in r['envs']),
                                    lost_finger=sum(min(e['hold_contact_frac'].values()) <= .95 for e in r['envs']),
                                    low_lift=sum(e['lift_m'] <= .035 for e in r['envs']),
                                    drift=sum(e['drift_mm'] >= 3 or e['rot_deg'] >= 10 for e in r['envs'])))
        except Exception as exc:
            import traceback
            summary[oid] = dict(error=f'{type(exc).__name__}: {exc}', trace=traceback.format_exc()[-2000:])
        print(oid, json.dumps({k: v for k, v in summary[oid].items() if k != 'trace'}), flush=True)
        (a.out/'summary.json').write_text(json.dumps(summary, indent=1)+'\n')
    print('WROTE', a.out/'summary.json')


if __name__ == '__main__':
    main()
