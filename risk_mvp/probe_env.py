"""Round 0 probe on the 5090: what the risk MVP can rely on before any design is fixed.

Part A (--demo): run the existing tripod demonstration (v1_demo.SpecDemo, finger
armature 0.003) and summarize the per-contact solver data it already logs every
60 Hz tick: how many object/finger contact points there are, the per-finger normal
force, and where on the object the contacts land (object frame). This answers
whether per-contact position + force vectors are available for region damage.

Part B (--throughput): an empty-hand scene (robot + one box per env) stepped at
1/300 s for each n_envs, timing scene.step() and get_contacts(is_padded=True).

Nothing in the existing code is changed. Usage (repo root, venv active):
  python risk_mvp/probe_env.py --out outputs/risk_mvp/probe --demo --throughput 1 64 256 1024
Send back outputs/risk_mvp/probe/summary.json (small).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
SIM = ROOT/'ur7e_xhand_sim'
sys.path.insert(0, str(SIM))

FINGER_KEYS = {'thumb': 'thumb', 'index': 'index', 'middle': 'mid', 'ring': 'ring', 'pinky': 'pinky'}


def finger_of(link):
    for f, k in FINGER_KEYS.items():
        if k in link:
            return f
    return 'palm' if 'hand' in link else 'other'


def part_a(out: Path, armature: float):
    from environment import Sandbox
    from v1_demo import SpecDemo
    spec = json.loads((ROOT/'configs/v1/demos/cuboid_tripod.json').read_text())
    sim = Sandbox(output_dir=out/'runtime_demo', render=False, cube_mass_kg=.05, cube_friction=.8,
                  robot_friction=.8, object_spec=spec['object_spec'], physics_dt_s=1/300,
                  control_substeps=5, noslip_iterations=5, deterministic=True)
    t0 = time.perf_counter()
    score = SpecDemo(sim, out/'demo', spec, finger_armature=armature).run()
    wall = time.perf_counter()-t0
    phases = {}
    for line in (out/'demo/evaluation.jsonl').read_text().splitlines():
        e = json.loads(line)
        ph = phases.setdefault(e['phase'], dict(ticks=0, points=[], per_finger=[], local=[], tangential_ratio=[]))
        ph['ticks'] += 1
        R = Rotation.from_quat(np.asarray(e['object_quaternion_wxyz'])[[1, 2, 3, 0]])
        center = np.asarray(e['object_position_m'])
        fingers = {}
        n = 0
        for c in e['solver_contacts']:
            if {c['entity_a'], c['entity_b']} != {'cube', 'ur7e_xhand'}:
                continue
            n += 1
            link = c['link_a'] if c['entity_a'] == 'ur7e_xhand' else c['link_b']
            f = finger_of(link)
            fingers[f] = fingers.get(f, 0.)+c['normal_force_N']
            force = np.asarray(c['force_on_a_world_N'])
            fn = c['normal_force_N']
            ph['tangential_ratio'].append(float(np.sqrt(max(force@force-fn*fn, 0.))/max(fn, 1e-9)))
            ph['local'].append((f, R.inv().apply(np.asarray(c['position_world_m'])-center).tolist(), fn))
        ph['points'].append(n)
        ph['per_finger'].append(fingers)
    summary = dict(score=score, wall_s=wall, phases={})
    for name, ph in phases.items():
        fingers = sorted({k for d in ph['per_finger'] for k in d})
        loc = ph['local']
        summary['phases'][name] = dict(
            ticks=ph['ticks'],
            contact_points_per_tick=dict(min=int(min(ph['points'])), median=float(np.median(ph['points'])),
                                         max=int(max(ph['points']))),
            normal_N_per_finger={f: dict(mean=float(np.mean([d.get(f, 0.) for d in ph['per_finger']])),
                                         max=float(np.max([d.get(f, 0.) for d in ph['per_finger']])))
                                 for f in fingers},
            contact_local_m_per_finger={f: dict(
                mean=np.mean([p for g, p, _ in loc if g == f], 0).round(4).tolist(),
                std=np.std([p for g, p, _ in loc if g == f], 0).round(4).tolist(),
                n=sum(g == f for g, _, _ in loc)) for f in fingers},
            tangential_over_normal_median=float(np.median(ph['tangential_ratio'])) if ph['tangential_ratio'] else None)
    return summary


def part_b(counts, backend):
    import genesis as gs
    from build_assets import build
    model, _ = build()
    gs.init(backend=getattr(gs, backend), precision='32', logging_level='warning')
    results = {}
    for n in counts:
        scene = gs.Scene(sim_options=gs.options.SimOptions(dt=1/300),
                         rigid_options=gs.options.RigidOptions(enable_self_collision=True, enable_joint_limit=True,
                                                               noslip_iterations=5),
                         show_viewer=False)
        robot = scene.add_entity(gs.morphs.URDF(file=str(model), fixed=True, pos=(0, 0, .2),
                                                merge_fixed_links=True, links_to_keep=['right_hand_link', 'tool0']))
        scene.add_entity(gs.morphs.Plane())
        scene.add_entity(gs.morphs.Box(size=(.045, .06, .045), pos=(-.5, .0, .03)))
        t0 = time.perf_counter()
        scene.build(n_envs=n)
        build_s = time.perf_counter()-t0
        for _ in range(20):
            scene.step()
        t0 = time.perf_counter()
        for _ in range(300):
            scene.step()
        step_s = (time.perf_counter()-t0)/300
        t0 = time.perf_counter()
        for _ in range(50):
            c = robot.get_contacts(is_padded=True)
            _ = {k: v.cpu() for k, v in c.items()}
        contacts_s = (time.perf_counter()-t0)/50
        results[str(n)] = dict(build_s=build_s, step_s=step_s, get_contacts_s=contacts_s,
                               env_steps_per_s=n/step_s)
        print('THROUGHPUT', n, results[str(n)], flush=True)
        scene.destroy() if hasattr(scene, 'destroy') else None
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, default=ROOT/'outputs/risk_mvp/probe')
    ap.add_argument('--demo', action='store_true')
    ap.add_argument('--armature', type=float, default=.003)
    ap.add_argument('--throughput', type=int, nargs='*', default=[])
    ap.add_argument('--backend', default='gpu')
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    summary = {}
    if a.demo:
        summary['demo'] = part_a(a.out, a.armature)
        print(json.dumps(summary['demo'], indent=1), flush=True)
    if a.throughput:
        # Sandbox already called gs.init in this process; run throughput separately if --demo was used.
        if a.demo:
            print('Run --throughput in a separate invocation (gs.init is per process).', flush=True)
        else:
            summary['throughput'] = part_b(a.throughput, a.backend)
    path = a.out/('summary.json' if a.demo else 'throughput.json')
    path.write_text(json.dumps(summary, indent=1)+'\n')
    print('WROTE', path)


if __name__ == '__main__':
    main()
