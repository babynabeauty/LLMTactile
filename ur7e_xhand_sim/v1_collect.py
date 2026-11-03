"""Collect v1 bases with BasePolicyV1: one Astra rollout per (demonstration, placement).

Each job starts its own bridge server (one port per worker), runs the Astra
agent in BasePolicyV1 mode, checks that the recorded hand targets follow the
demonstration (open -> closed during close, closed afterwards) and converts the
commands to the residual base format. Labels come later from v1_label. Jobs
already in runs.jsonl are skipped; --retry-errors reruns failed infrastructure.
"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import sys
import threading
import time
import urllib.request

import numpy as np

from platform_asset import TABLETOP_XY_M
from residual_rl.multiobject_geometry import ObjectSpec
from residual_rl.v1_task import scaled_object

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DT = 1/60
SOLVER = {'schema': 'genesis_rigid_solver_profile_v1', 'noslip_iterations': 5, 'noslip_tolerance': 1e-6}
LIFT_PHASES = ('test_lift', 'lift', 'hold')
TABLETOP_MARGIN_M = .02


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def call(port, name, payload=None):
    request = urllib.request.Request(f'http://127.0.0.1:{port}/{name}', json.dumps(payload or {}).encode(),
                                     {'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.load(response)


def plan_jobs(config):
    """Train and test placements within +-xy_range_m; edge test placements between that and +-edge_range_m.

    Each demonstration also covers its object scaled by every factor in 'scales' (train and test, no
    edge placements). Scaled and unseen objects reuse a demonstration; an unseen object is run with every
    hand shape demonstrated on its 'near' object. 'skip' lists object_id_hand_shape scenes to leave out
    (e.g. sizes that fail the v1_demo --scale check). Every object must stay TABLETOP_MARGIN_M inside the tabletop.
    """
    rng = np.random.default_rng(config.get('seed', 1))
    radius, yaw = config.get('xy_range_m', .10), config.get('yaw_range_deg', 10.)
    edge_radius = config.get('edge_range_m', .15)
    skip = set(config.get('skip', []))
    jobs = []

    def add(demo_path, demo, object_id, object_spec, split, count, unseen=False, edge=False):
        if f"{object_id}_{demo['hand_shape']}" in skip:
            return
        # Half diagonal of the footprint box bounds the footprint radius at any yaw.
        reach = float(np.hypot(*ObjectSpec.from_value(object_spec).bounds_size_m[:2]))/2+TABLETOP_MARGIN_M
        for i in range(count):
            xy = rng.uniform(-radius, radius, 2)
            while edge and np.abs(xy).max() <= radius:
                xy = rng.uniform(-edge_radius, edge_radius, 2)
            placement = {'offset_xy_m': xy.round(4).tolist(), 'yaw_deg': round(float(rng.uniform(-yaw, yaw)), 2)}
            if unseen:
                placement['unseen_object'] = True
            if edge:
                placement['edge'] = True
            run_id = f"{object_id}_{demo['hand_shape']}_{'edge' if edge else split}_{i:03d}"
            center = np.asarray(demo['object_initial_position_m'][:2])+placement['offset_xy_m']
            if any(not lo+reach <= c <= hi-reach for c, (lo, hi) in zip(center, TABLETOP_XY_M)):
                raise ValueError(f'{run_id}: object at {center.round(3).tolist()} m is closer than '
                                 f'{TABLETOP_MARGIN_M} m to the tabletop edge')
            jobs.append(dict(run_id=run_id, demo=str(Path(demo_path).resolve()), object_id=object_id,
                object_spec=object_spec, hand_shape=demo['hand_shape'], split=split, placement=placement))
    demos = [(path, json.loads(Path(path).read_text())) for path in config['demos']]
    for path, demo in demos:
        add(path, demo, demo['object_id'], demo['object_spec'], 'train', config['train_placements'])
        add(path, demo, demo['object_id'], demo['object_spec'], 'test', config['test_placements'])
        add(path, demo, demo['object_id'], demo['object_spec'], 'test', config.get('edge_placements', 0), edge=True)
        for scale in config.get('scales', []):
            object_id, spec = scaled_object(demo['object_id'], demo['object_spec'], scale)
            add(path, demo, object_id, spec, 'train', config['train_placements'])
            add(path, demo, object_id, spec, 'test', config['test_placements'])
    for row in config.get('unseen', []):
        near = [(path, demo) for path, demo in demos if demo['object_id'] == row['near']]
        if not near:
            raise ValueError(f"Unseen object {row['object_id']}: no demonstration of '{row['near']}'")
        for path, demo in near:
            add(path, demo, row['object_id'], row['object_spec'], 'test', config['unseen_placements'], unseen=True)
    return jobs


def convert(bridge, run_dir):
    """S8V4 log -> residual base trajectory, after checking the hand followed the demonstration."""
    manifest = json.loads((bridge/'manifest.json').read_text())
    if manifest.get('online_control'):
        raise ValueError('Online combined commands are not base-only collection trajectories')
    rows = [json.loads(line) for line in (bridge/'control.jsonl').open()]
    phases = [r['phase'] for r in rows]
    if [r['step'] for r in rows] != list(range(1, len(rows)+1)):
        raise ValueError('Recorded control steps are not contiguous')
    if any(p not in phases for p in ('close',)+LIFT_PHASES) or phases[-180:] != ['hold']*180:
        return dict(complete=False)
    hand = np.asarray([r['target'] for r in rows])[:, 6:]
    opened, closed = np.asarray(manifest['open_hand']), np.asarray(manifest['closed_hand'])
    first = phases.index('close')
    start = hand[first-1]
    span = closed-start
    fraction = (hand[first:]-start)@span/max(span@span, 1e-12)
    closing = [i-first for i, p in enumerate(phases) if p == 'close']
    lifting = [i for i, p in enumerate(phases) if p in LIFT_PHASES]
    if (not np.allclose(start, opened, atol=1e-6) or not np.allclose(hand[lifting], closed, atol=1e-6)
            or not np.allclose(hand[first:][closing], start+fraction[closing, None]*span, atol=1e-6)
            or fraction.min() < -1e-6 or fraction.max() > 1+1e-6):
        raise ValueError('Hand targets deviate from the demonstration (rule tightening or executor change)')
    out = run_dir/'base'
    out.mkdir(exist_ok=True)
    gain = float(manifest['grip_gain'])
    (out/'control.jsonl').write_text(''.join(json.dumps({'before': {'sim_time_s': (r['step']-1)*DT},
        'after': {'sim_time_s': r['step']*DT}, 'sent_target_rad': r['target'], 'phase': r['phase'],
        'grip_gain': gain})+'\n' for r in rows))
    digest = sha(out/'control.jsonl')
    (out/'trajectory_metadata.json').write_text(json.dumps({'trajectory_sha256': digest, 'solver_profile': SOLVER}))
    return dict(complete=True, trajectory_path=str(out/'control.jsonl'), trajectory_sha256=digest,
                object_initial_position_m=manifest['placement']['object_initial_position_m'],
                object_initial_quaternion_wxyz=manifest['placement']['object_initial_quaternion_wxyz'])


def run_job(job, port, output):
    run_dir = output/'runs'/job['run_id']
    run_dir.mkdir(parents=True, exist_ok=True)
    label = f"v1_{job['run_id']}_{time.strftime('%Y%m%d%H%M%S')}"
    result, started = dict(job, label=label, port=port), time.monotonic()
    log = (run_dir/'server.log').open('w')
    server = subprocess.Popen([sys.executable, '-u', 'ur7e_xhand_sim/server.py', '--port', str(port),
        '--output', str(run_dir/'runtime'), '--object-spec-json', json.dumps(job['object_spec']),
        '--cube-mass-g', '50'], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
        start_new_session=True)
    try:
        deadline = time.monotonic()+300
        while True:
            if server.poll() is not None:
                raise RuntimeError('Bridge server exited before readiness')
            try:
                if call(port, 'status')['ok']:
                    break
            except Exception:
                pass
            if time.monotonic() > deadline:
                raise TimeoutError('Bridge server readiness timeout')
            time.sleep(1)
        with (run_dir/'agent.log').open('w') as agent_log:
            agent = subprocess.run([sys.executable, '-u', 'ur7e_xhand_sim/run_base_policy_v1.py', '--label', label,
                '--url', f'http://127.0.0.1:{port}', '--demo', job['demo'],
                '--placement-json', json.dumps(job['placement'])],
                cwd=ROOT, stdout=agent_log, stderr=subprocess.STDOUT, timeout=1800)
        result['agent_exit_code'] = agent.returncode
        bridge = ROOT/'outputs/base_policy_v1'/label
        result['bridge_dir'] = str(bridge)
        agent_dir = ROOT/'outputs/base_policy_v1_agent'/label
        score = agent_dir/'score.json'
        result['score'] = json.loads(score.read_text()) if score.is_file() else None
        result['model_calls'] = len(list(agent_dir.glob('model_*.json')))
        result.update(convert(bridge, run_dir))
    except Exception as exc:
        result['error'] = repr(exc)
    finally:
        try:
            call(port, 'close')
            server.wait(timeout=30)
        except Exception:
            server.terminate()
            server.wait(timeout=30)
        log.close()
    result['wall_s'] = time.monotonic()-started
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--workers', type=int, default=1)
    p.add_argument('--base-port', type=int, default=8770)
    p.add_argument('--limit', type=int, help='run only the first N pending jobs (pilot)')
    p.add_argument('--retry-errors', action='store_true')
    args = p.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    jobs = plan_jobs(json.loads(args.config.read_text()))
    planned = output/'jobs.json'
    if planned.is_file() and json.loads(planned.read_text()) != jobs:
        raise ValueError('Config changed since jobs were planned; use a new output directory')
    planned.write_text(json.dumps(jobs, indent=2))
    runs = output/'runs.jsonl'
    previous = [json.loads(line) for line in runs.open()] if runs.is_file() else []
    keep = [r for r in previous if not (args.retry_errors and 'error' in r)]
    if len(keep) != len(previous):
        runs.write_text(''.join(json.dumps(r)+'\n' for r in keep))
    done = {r['run_id'] for r in keep}
    pending = [j for j in jobs if j['run_id'] not in done][:args.limit]
    lock = threading.Lock()

    def worker(index):
        while True:
            with lock:
                if not pending:
                    return
                job = pending.pop(0)
            result = run_job(job, args.base_port+index, output)
            with lock:
                with runs.open('a') as stream:
                    stream.write(json.dumps(result)+'\n')
                print(json.dumps({k: result.get(k) for k in ('run_id', 'complete', 'error', 'wall_s', 'model_calls')}),
                      flush=True)
    with ThreadPoolExecutor(args.workers) as pool:
        list(pool.map(worker, range(args.workers)))
    rows = [json.loads(line) for line in runs.open()]
    print(json.dumps({'jobs': len(jobs), 'recorded': len(rows), 'complete': sum(bool(r.get('complete')) for r in rows),
        'errors': sum('error' in r for r in rows),
        'mean_wall_s': float(np.mean([r['wall_s'] for r in rows])) if rows else None}))


if __name__ == '__main__':
    main()
