"""Tripod pregrasp search per YCB object (5090): which palm pose gives a stable three-finger lift.

Template: configs/v1/demos/cuboid_tripod.json (palm orientation, open/closed hand, palm offset from the
object centre, grip gain 3). For each object the palm offset is shifted by the object's half-width
difference along x (the thumb-fingers axis of the template) plus a grid dx x dy x dz, one grid point per
env. Episode (60 Hz control, 1/300 s physics):
  settle 10 -> approach (joint space to a pre-pose 12 cm above, hand opening) 120 -> descend 60
  -> sag correction 20 -> force-controlled close 120 (thumb 2 N, index 1 N, middle 1 N)
  -> lift 5 cm 60 -> hold 120.
Success: no contact before closing, all three fingers in contact through the hold, lifted > 3.5 cm,
object drift relative to the wrist < 3 mm and < 10 deg during the hold.

Then the best pose is replayed in a one-env scene with two cameras: keyframes, a video and the
per-finger force trace. Pictures written per object:
  <id>_grid.png       outcome of every grid point (why it failed)
  <id>_contacts.png   regions of the object and where thumb/index/middle touched (successful envs)
  <id>_best_frames.png  keyframes of the best pose (2 views x 6 phases)
  <id>_best_forces.png  per-finger normal force over the replay
  <id>_best.mp4       video of the replay (front view)
  overview.png        the "held" keyframe of every object
  python risk_mvp/pregrasp_search.py --objects 061_foam_brick --out outputs/risk_mvp/pregrasp
  python risk_mvp/pregrasp_search.py --out outputs/risk_mvp/pregrasp
Send back summary.json, overview.png and the per-object PNGs you want to discuss.
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
CAMERAS = [dict(offset=(.20, -.30, .16)), dict(offset=(-.05, -.34, .06), lookat_offset=(-.02, 0, .03))]
PHASE_TICKS = dict(settle=10, approach=120, descend=60, correct=30, close=120, lift=60, hold=120)


def grid(dx=(-.015, -.0075, 0., .0075, .015), dy=(-.01, 0., .01), dz=(-.015, 0., .015)):
    return np.array(list(itertools.product(dx, dy, dz)))


def run_episode(sc, spec, offsets, on_phase_end=None):
    """Run the pregrasp episode on every env of `sc` (offsets (E,3) added to the palm offset).
    Returns per-env results and the per-tick force log."""
    from risk_mvp.force_ctrl import BatchForceController, FINGER_JOINTS
    from risk_mvp.scene import quat_to_mat
    E = sc.E
    half_x = sc.info['extent_m'][0]/2
    palm_offset = np.asarray(spec['palm_offset_m'])+np.array([half_x-TEMPLATE_HALF_X, 0, 0])
    quat = np.tile(np.asarray(spec['palm_quaternion_wxyz'], float), (E, 1))
    open_h, closed_h = np.asarray(spec['open_hand_rad']), np.asarray(spec['closed_hand_rad'])
    sc.reset()
    sc.set_gains()
    center0, _ = sc.object_pose()
    align = center0+palm_offset+offsets
    pre = align+np.array([0, 0, .12])
    log = dict(t=[], phase=[], **{f: [] for f in FINGERS})
    tick = [0]

    def record(c, phase):
        log['t'].append(tick[0]/60)
        log['phase'].append(phase)
        for f in FINGERS:
            log[f].append(c[f]['fn'].copy())
        tick[0] += 1

    def step(tgt, phase, c=None):
        sc.step(tgt)
        record(c if c is not None else sc.finger_contacts(), phase)

    def move_arm(q_from, q_to, ticks, phase, hand_from=None, hand_to=None, hand_fn=None):
        for i in range(1, ticks+1):
            s = (i/ticks)**2*(3-2*i/ticks)
            c = None
            if hand_fn is not None:
                tgt, c = hand_fn()
            else:
                tgt = sc.target.copy()
                if hand_from is not None:
                    tgt[:, 6:] = hand_from+s*(hand_to-hand_from)
            tgt[:, :6] = q_from+s*(q_to-q_from)
            step(tgt, phase, c)
        if on_phase_end:
            on_phase_end(phase)

    for _ in range(PHASE_TICKS['settle']):
        step(sc.target, 'settle')
    q_pre, err_pre = sc.ik(pre, quat)
    move_arm(sc.q()[:, :6], q_pre, PHASE_TICKS['approach'], 'approach', np.zeros((E, 12)), np.tile(open_h, (E, 1)))
    q_al, err_al = sc.ik(align, quat)
    move_arm(q_pre, q_al, PHASE_TICKS['descend'], 'descend')
    for _ in range(10):
        step(sc.target, 'correct')
    w, _ = sc.wrist_pose()
    sag = align-w                      # re-aim so the measured wrist lands on the planned pose
    q_al2, _ = sc.ik(align+sag, quat)
    move_arm(q_al, q_al2, PHASE_TICKS['correct']-10, 'correct')
    c = sc.finger_contacts()
    pre_touch = np.any([c[f]['has'] for f in FINGERS], 0) | (c['other_robot_obj_N'] > .05)

    sc.set_gains(finger_gain=float(spec.get('grip_gain', 1.)))
    ctrl = {f: BatchForceController(E) for f in FINGERS}
    hand_names = sc.names[6:]
    closed_cols = {f: [hand_names.index(j) for j in FINGER_JOINTS[f]] for f in FINGERS}
    driven = {6+k for f in FINGERS for k in closed_cols[f]}
    others = [k for k in range(6, 18) if k not in driven]
    sc.target[:, others] = closed_h[[k-6 for k in others]]   # thumb rotation, abduction, ring, pinky

    def hand_tick():
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
        return tgt, c

    for _ in range(PHASE_TICKS['close']):
        tgt, c = hand_tick()
        step(tgt, 'close', c)
    if on_phase_end:
        on_phase_end('close')
    obj_before, _ = sc.object_pose()
    q_lift, _ = sc.ik(align+sag+np.array([0, 0, .05]), quat)
    move_arm(sc.target[:, :6].copy(), q_lift, PHASE_TICKS['lift'], 'lift', hand_fn=hand_tick)
    hold, rel = [], []
    for _ in range(PHASE_TICKS['hold']):
        tgt, c = hand_tick()
        step(tgt, 'hold', c)
        hold.append(c)
        p, R = sc.object_pose()
        wp, wq = sc.wrist_pose()
        Rw = quat_to_mat(wq)
        rel.append((np.einsum('eji,ej->ei', Rw, p-wp), np.einsum('eji,ejk->eik', Rw, R)))
    if on_phase_end:
        on_phase_end('hold')
    p_end, R_end = sc.object_pose()
    lift_h = p_end[:, 2]-obj_before[:, 2]
    drift = np.linalg.norm(rel[-1][0]-rel[0][0], axis=1)
    rot = np.degrees(np.arccos(np.clip((np.einsum('eij,eij->e', rel[-1][1], rel[0][1])-1)/2, -1, 1)))
    rows = []
    for e in range(E):
        fn = {f: float(np.mean([h[f]['fn'][e] for h in hold])) for f in FINGERS}
        touch = {f: float(np.mean([h[f]['has'][e] for h in hold])) for f in FINGERS}
        pts = {}
        for f in FINGERS:
            ws = [h[f]['point'][e] for h in hold if h[f]['has'][e]]
            if ws:
                pts[f] = (R_end[e].T@(np.mean(ws, 0)-p_end[e])).round(4).tolist()
        ok = bool(not pre_touch[e] and all(v > .95 for v in touch.values()) and lift_h[e] > .035
                  and drift[e] < .003 and rot[e] < 10.)
        rows.append(dict(env=e, offset_m=offsets[e].tolist(), success=ok, pre_touch=bool(pre_touch[e]),
                         lift_m=float(lift_h[e]), drift_mm=float(drift[e]*1000), rot_deg=float(rot[e]),
                         hold_normal_N=fn, hold_contact_frac=touch, contact_obj_m=pts,
                         table_robot_N=float(np.mean([h['table_robot_N'][e] for h in hold])),
                         palm_obj_N=float(np.mean([h['other_robot_obj_N'][e] for h in hold])),
                         ik_err_pre=float(np.linalg.norm(err_pre[e, :3])), ik_err_align=float(np.linalg.norm(err_al[e, :3])),
                         sag_mm=(sag[e]*1000).round(2).tolist()))
    meta = dict(palm_offset_base_m=palm_offset.tolist(), palm_quaternion_wxyz=spec['palm_quaternion_wxyz'],
                open_hand_rad=open_h.tolist(), closed_hand_rad=closed_h.tolist(), grip_gain=spec.get('grip_gain', 1.))
    return rows, log, meta


def run_object(oid, out_dir, backend='gpu', k_regions=5):
    from risk_mvp.scene import GraspScene
    from risk_mvp.regions import object_mesh, partition
    from risk_mvp import objects as O, viz
    spec = json.loads(TEMPLATE.read_text())
    offsets = grid()
    t0 = time.time()
    sc = GraspScene(oid, len(offsets), backend=backend)
    rows, _, meta = run_episode(sc, spec, offsets)
    succ = [r for r in rows if r['success']]
    best = min(succ, key=lambda r: (r['drift_mm']+r['rot_deg']/5, abs(np.array(r['offset_m'])).sum())) if succ else None
    res = dict(object=oid, n_envs=len(offsets), search_wall_s=time.time()-t0, **meta, n_success=len(succ),
               best=best, envs=rows)
    # pictures: grid outcome and contacts on the regions
    hull = object_mesh(O.region_spec(oid))
    part = partition(hull, k_regions, seed=0)
    viz.grid_heatmap(rows, out_dir/f'{oid}_grid.png', f'{oid}: {len(succ)}/{len(rows)} poses lifted and held')
    viz.contacts_on_regions(part, hull, rows, out_dir/f'{oid}_contacts.png',
                            f'{oid}: contact points of the successful poses on the {k_regions} regions')
    # replay the best pose (or the least-bad one) with cameras
    pick = best or max(rows, key=lambda r: (not r['pre_touch'], min(r['hold_contact_frac'].values()), r['lift_m']))
    rs = GraspScene(oid, 1, backend=backend, cameras=CAMERAS)
    frames, labels = [], []
    rs.cams[0].start_recording(save_to_filename=str(out_dir/f'{oid}_best.mp4'), fps=30)

    def snap(phase):
        for v, img in enumerate(rs.render()):
            frames.append(img)
            labels.append(f'{phase} (view {v})')
    snap('start')
    _, log, _ = run_episode(rs, spec, np.array([pick['offset_m']]), on_phase_end=snap)
    rs.cams[0].stop_recording()
    order = list(range(0, len(frames), 2))+list(range(1, len(frames), 2))   # row 1: view 0, row 2: view 1
    viz.keyframes([frames[i] for i in order], [labels[i] for i in order], out_dir/f'{oid}_best_frames.png',
                  cols=len(frames)//2)
    starts = [(log['t'][i], p) for i, p in enumerate(log['phase']) if i == 0 or log['phase'][i-1] != p]
    viz.force_trace(log['t'], {f: np.array(log[f])[:, 0] for f in FINGERS}, starts, out_dir/f'{oid}_best_forces.png',
                    f'{oid}: replay of {"best" if best else "least-bad"} pose offset {np.round(np.array(pick["offset_m"])*1000, 1).tolist()} mm',
                    targets=TARGET_N)
    res['replayed'] = dict(offset_m=pick['offset_m'], was_success=bool(best))
    res['held_frame'] = frames[-2]
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, default=ROOT/'outputs/risk_mvp/pregrasp')
    ap.add_argument('--objects', nargs='*', default=None)
    ap.add_argument('--backend', default='gpu')
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    from risk_mvp import objects as O, viz
    O.download()
    ids = a.objects or list(O.CATALOG)
    summary_path = a.out/'summary.json'
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    held = []
    for oid in ids:
        try:
            r = run_object(oid, a.out, a.backend)
            held.append((r.pop('held_frame'), f"{oid}: {r['n_success']}/{r['n_envs']}"))
            (a.out/f'{oid}.json').write_text(json.dumps(r, indent=1)+'\n')
            summary[oid] = dict(n_success=r['n_success'], of=r['n_envs'], best=r['best'], wall_s=r['search_wall_s'],
                                failure_counts=dict(
                                    pre_touch=sum(e['pre_touch'] for e in r['envs']),
                                    lost_finger=sum(min(e['hold_contact_frac'].values()) <= .95 for e in r['envs']),
                                    low_lift=sum(e['lift_m'] <= .035 for e in r['envs']),
                                    drift=sum(e['drift_mm'] >= 3 or e['rot_deg'] >= 10 for e in r['envs'])))
        except Exception as exc:
            import traceback
            summary[oid] = dict(error=f'{type(exc).__name__}: {exc}', trace=traceback.format_exc()[-3000:])
            print(summary[oid]['trace'], flush=True)
        print(oid, json.dumps({k: v for k, v in summary[oid].items() if k != 'trace'}), flush=True)
        summary_path.write_text(json.dumps(summary, indent=1)+'\n')
    if held:
        viz.keyframes([h[0] for h in held], [h[1] for h in held], a.out/'overview.png', cols=4)
    print('WROTE', summary_path, a.out/'overview.png')


if __name__ == '__main__':
    main()
