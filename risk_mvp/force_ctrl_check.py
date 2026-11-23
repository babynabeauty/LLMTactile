"""Step 2 self-check on the 5090: per-finger force control on the tripod cuboid grasp.

Runs the demo's settle/approach/align/close (v1_demo.SpecDemo, finger armature 0.003), then
hands the three flexing fingers to FingerForceController at 60 Hz:
  A  object on the table   thumb 3 N, index 1 N, middle 2 N  for 3 s
  B  step                  thumb 4 N, index 2.5 N, middle 1.5 N  for 3 s
  C  lifted 5 cm           thumb 4 N, index 2 N, middle 2 N  for 3 s
(targets satisfy thumb = index + middle, which statics requires for opposing faces).
Pass: in the last 1 s of each segment |mean - target| < 10 % of target and std < 5 % of target.
Before the grasp it also checks the Jacobian rows against a finite difference.

  python risk_mvp/force_ctrl_check.py --out outputs/risk_mvp/force_ctrl
Send back outputs/risk_mvp/force_ctrl/result.json (and the plot if it was written).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'ur7e_xhand_sim'))
sys.path.insert(0, str(ROOT))

from risk_mvp.force_ctrl import FINGER_JOINTS, FINGER_LINKS, FingerForceController, finger_contact  # noqa: E402

SEGMENTS = [('table', dict(thumb=3., index=1., middle=2.), 0.),
            ('step', dict(thumb=4., index=2.5, middle=1.5), 0.),
            ('lifted', dict(thumb=4., index=2., middle=2.), .05)]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, default=ROOT/'outputs/risk_mvp/force_ctrl')
    ap.add_argument('--seconds', type=float, default=3.)
    ap.add_argument('--mass-kg', type=float, default=.05)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    from environment import Sandbox
    from v1_demo import SpecDemo

    spec = json.loads((ROOT/'configs/v1/demos/cuboid_tripod.json').read_text())
    sim = Sandbox(output_dir=a.out/'runtime', render=False, cube_mass_kg=a.mass_kg, cube_friction=.8,
                  robot_friction=.8, object_spec=spec['object_spec'], physics_dt_s=1/300,
                  control_substeps=5, noslip_iterations=5, deterministic=True)
    run_dir = a.out/'demo'
    if run_dir.exists():
        import shutil
        shutil.rmtree(run_dir)
    demo = SpecDemo(sim, run_dir, spec, finger_armature=.003)
    name_to_i = {n: i for i, n in enumerate(sim.names)}
    report = dict(segments={}, jacobian_check={})

    # --- Jacobian row convention check (hand open, no contact) ---
    for f, link_name in FINGER_LINKS.items():
        link = sim.robot.get_link(link_name)
        cols = [sim.indices[name_to_i[j]] for j in FINGER_JOINTS[f]]
        q0 = sim.array(sim.robot.get_dofs_position()).reshape(-1).copy()
        J = sim.array(sim.robot.get_jacobian(link)).reshape(-1, 6, sim.robot.n_dofs)[0][:, cols]
        p0 = sim.array(link.get_pos()).reshape(3).copy()
        dq = np.array([.01, .01])
        q1 = q0.copy()
        q1[cols] += dq
        sim.robot.set_dofs_position(q1, zero_velocity=True)
        p1 = sim.array(link.get_pos()).reshape(3)
        sim.robot.set_dofs_position(q0, zero_velocity=True)
        report['jacobian_check'][f] = dict(fd=(p1-p0).tolist(), J_lin_dq=(J[:3]@dq).tolist(),
                                           rel_err=float(np.linalg.norm(p1-p0-J[:3]@dq)/max(np.linalg.norm(p1-p0), 1e-12)))
    print('JACOBIAN', json.dumps(report['jacobian_check']), flush=True)

    # --- demo up to the end of close (same motions as SpecDemo.grasp, no image capture) ---
    c = spec
    quat = np.asarray(c['palm_quaternion_wxyz'], float)
    open_hand, closed = np.asarray(c['open_hand_rad'], float), np.asarray(c['closed_hand_rad'], float)
    obj0 = sim.array(sim.cube.get_pos()).reshape(3)
    for _ in range(10):
        demo.servo(demo.home_pos, demo.home_quat, sim.home[6:], 'settle')
    align = obj0+np.asarray(c['palm_offset_m'], float)
    pre = align+np.array([0, 0, .12])
    high = pre.copy()
    high[2] = max(pre[2], demo.home_pos[2])
    start_hand = sim.target[6:].copy()
    demo.move('approach', high, quat, start_hand+.4*(open_hand-start_hand), 3.)
    demo.move('approach', pre, quat, start_hand+.75*(open_hand-start_hand), 1.)
    demo.move('align', align, quat, open_hand, 2.)
    demo.set_gain(float(c.get('grip_gain', 1.)))
    demo.linear('close', align, quat, closed)

    # --- force control ---
    ctrl = FingerForceController()
    hand_names = sim.names[6:]
    kp_all = sim.array(sim.robot.get_dofs_kp(sim.indices)).reshape(-1)
    pos = align.copy()
    trace = []
    for seg, targets, lift in SEGMENTS:
        if lift:
            demo.linear('lift', align+np.array([0, 0, lift]), quat, sim.target[6:].copy())
            pos = align+np.array([0, 0, lift])
        rows = []
        for tick in range(int(a.seconds*60)):
            e = sim.evaluate()
            center = np.asarray(e['object_position_m'])
            hand_target = sim.target[6:].copy()
            q_all = sim.array(sim.robot.get_dofs_position(sim.indices)).reshape(-1)
            meas = {}
            for f, link_name in FINGER_LINKS.items():
                link = sim.robot.get_link(link_name)
                idx = [name_to_i[j] for j in FINGER_JOINTS[f]]
                cols = [sim.indices[i] for i in idx]
                J = sim.array(sim.robot.get_jacobian(link)).reshape(-1, 6, sim.robot.n_dofs)[0][:, cols]
                fn, p, n = finger_contact(e['solver_contacts'], link_name, object_center=center)
                out = ctrl.step(f, targets[f], q_all[idx], kp_all[idx], sim.effort[idx], sim.lower[idx], sim.upper[idx],
                                J_origin=J, link_pos=sim.array(link.get_pos()).reshape(3), contact_point=p,
                                normal_in=n, measured_N=fn, closed_q=closed[[hand_names.index(j) for j in FINGER_JOINTS[f]]])
                for j, v in zip(FINGER_JOINTS[f], out['q_target']):
                    hand_target[hand_names.index(j)] = v
                meas[f] = fn
            demo.servo(pos, quat, hand_target, f'force_{seg}', ticks=1)
            rows.append(meas)
            trace.append(dict(segment=seg, t=tick/60, **{f'{k}_N': v for k, v in meas.items()},
                              **{f'{k}_target_N': targets[k] for k in targets}))
        last = rows[-60:]
        seg_rep = {}
        for f, tgt in targets.items():
            v = np.array([r[f] for r in last])
            seg_rep[f] = dict(target=tgt, mean=float(v.mean()), std=float(v.std()), rel_err=float(abs(v.mean()-tgt)/tgt),
                              peak=float(np.max([r[f] for r in rows])),
                              ok=bool(abs(v.mean()-tgt) < .1*tgt and v.std() < .05*tgt))
        e = sim.evaluate()
        seg_rep['object_bottom_clearance_m'] = e['bottom_clearance_m']
        report['segments'][seg] = seg_rep
        print('SEGMENT', seg, json.dumps(seg_rep), flush=True)
    report['pass'] = all(v['ok'] for s in report['segments'].values() for k, v in s.items() if isinstance(v, dict))
    (a.out/'result.json').write_text(json.dumps(report, indent=1)+'\n')
    (a.out/'trace.json').write_text(json.dumps(trace)+'\n')
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 4))
        t = np.arange(len(trace))/60
        for f, col in zip(('thumb', 'index', 'middle'), ('C0', 'C1', 'C2')):
            ax.plot(t, [r[f'{f}_N'] for r in trace], col, label=f)
            ax.plot(t, [r[f'{f}_target_N'] for r in trace], col+'--', lw=1)
        ax.set(xlabel='s', ylabel='normal force N', title='force control (dashed: target)')
        ax.legend()
        fig.savefig(a.out/'force_ctrl.png', dpi=120)
    except Exception as exc:  # plotting is optional
        print('plot skipped:', exc)
    print('PASS' if report['pass'] else 'FAIL', flush=True)


if __name__ == '__main__':
    main()
