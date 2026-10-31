"""Six rigid cube sizes, six frozen policies; no training or soft-body scene."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys

from .core import DEMO, ROOT, Trajectory

SIZES_MM = [30, 35, 40, 50, 55, 60]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    tmp.replace(path)


def now():
    return datetime.now(timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.expanduser().resolve()
    if out.exists():
        raise FileExistsError('Preserve prior runs; choose a new output directory')
    checkpoints = ROOT/'outputs/residual_rl_20261025_v3/confirmation_normalized_p200_seeds123'
    models = {}
    for seed in (1, 2, 3):
        for modality in ('tactile', 'no_tactile'):
            model = checkpoints/f'confirm_normalized_p200_seed{seed}_{modality}_stage3_786432/final_model.zip'
            if not model.is_file():
                raise FileNotFoundError(model)
            models[f'seed{seed}_{modality}'] = {'path': str(model), 'sha256': digest(model), 'training_seed': seed, 'modality': modality}
    trajectory = Trajectory(DEMO/'control.jsonl')
    config_path = ROOT/'outputs/residual_rl_20261025_v3/config.json'
    config = json.loads(config_path.read_text())
    out.mkdir(parents=True)
    (out/'logs').mkdir()
    snapshot = out/'source_snapshot'
    snapshot.mkdir()
    sources = list(Path(__file__).parent.glob('*.py')) + [
        ROOT/'Genesis/genesis/engine/sensors/kinematic_tactile.py',
        ROOT/'ur7e_xhand_sim/build_assets.py',
    ]
    sources += [p for p in (ROOT/'ur7e_xhand_sim/sources').rglob('*') if p.is_file()]
    source_hashes = {}
    for source in sources:
        relative = source.relative_to(ROOT)
        dest = snapshot/relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
        source_hashes[str(relative)] = digest(source)
        assert digest(dest) == source_hashes[str(relative)]
    write(out/'conditions.json', [[.125, .5]])
    write(out/'config.json', config)
    frozen_inputs = {str(p): digest(p) for p in (out/'conditions.json', out/'config.json', config_path)}
    size_runs = [{'size_mm': size, 'output': f'size_{size:03d}mm/evaluation'} for size in SIZES_MM]
    protocol = {
        'created_utc': now(), 'scope': '36 rigid size-transfer episodes; three frozen training seeds, tactile/no-tactile per size',
        'sizes_mm': SIZES_MM, 'reference_size_mm': 45, 'mass_kg': .125, 'friction': .5,
        'material': 'Rigid', 'training': False, 'soft_body_enabled': False,
        'density_rule': 'mass / edge^3; actual mass, inertia and geometry read back before physical replay',
        'position_rule': 'Original teacher XY [-.6,-.1333]; initial bottom at table z=.2, center z=.2+edge/2',
        'trajectory': {'path': str(trajectory.path), 'sha256': trajectory.sha256, 'duration_s': trajectory.duration_s, 'phase_starts': trajectory.phase_starts},
        'models': models, 'controllers': list(models), 'model_training_steps': 786432,
        'size_runs': size_runs, 'planned_episodes': 36, 'rates_hz': {'physics': 300, 'tactile': 60, 'action': 15},
        'common_physics_seed': 7000, 'config': config,
        'source_hashes': source_hashes, 'frozen_input_hashes': frozen_inputs,
        'task': 'Deformed geometry is disabled. Rigid cube lowest surface >=5cm above table throughout original final 3s hold; no robot self/table collision in residual stage.',
        'strict': 'Task plus >=2 tactile-bearing links each independent solver contact load >.05N throughout hold',
        'force_reporting': 'Original five sensor-bearing-link contact force magnitude sum, plus all robot-cube contact magnitude sum. Report all episodes, successful episodes, hold and contact-conditioned means; efficacy comparison uses only same-size same-seed jointly successful pairs.',
        'prefix': 'Each size gets a fresh rigid scene and complete teacher prefix. Full SceneCheckpoint reset between six policies; prefix observation equality checked. Prefix collisions reported separately and never silently discarded.',
        'failure_policy': 'Keep physical grasp failures. Stop on execution/integrity failures, no retries, no checkpoint selection.',
    }
    write(out/'protocol.json', protocol)
    protocol_hash = digest(out/'protocol.json')
    state = {'status': 'running', 'created_utc': now(), 'completed_sizes': [], 'active_size_mm': None, 'planned_episodes': 36, 'completed_episodes': 0, 'child_pid': None}
    write(out/'status.json', state)
    process = None

    def verify_inputs():
        assert digest(out/'protocol.json') == protocol_hash, 'Protocol changed'
        assert digest(trajectory.path) == trajectory.sha256, 'Trajectory changed'
        for item in models.values():
            assert digest(item['path']) == item['sha256'], 'Checkpoint changed'
        for path, expected in frozen_inputs.items():
            assert digest(path) == expected, f'Input changed: {path}'
        for relative, expected in source_hashes.items():
            assert digest(ROOT/relative) == expected, f'Source changed: {relative}'
            assert digest(snapshot/relative) == expected, f'Snapshot changed: {relative}'

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'Signal {signum}')

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        for job in size_runs:
            verify_inputs()
            size = job['size_mm']
            directory = out/job['output']
            command = [sys.executable, '-m', 'ur7e_xhand_sim.residual_rl.adaptive_benchmark',
                '--output', str(directory), '--conditions-json', str(out/'conditions.json'),
                '--config-json', str(out/'config.json'), '--split', 'size_transfer', '--seed', '7000',
                '--cube-size-m', str(size/1000), '--audit-prefix', '--models-only', '--models']
            command += [f'{name}={item["path"]}' for name, item in models.items()]
            write(out/f'size_{size:03d}mm.command.json', command)
            with (out/'logs'/f'size_{size:03d}mm.log').open('w') as log:
                process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                state.update(active_size_mm=size, child_pid=process.pid, updated_utc=now())
                write(out/'status.json', state)
                print(json.dumps({'event': 'size_started', 'size_mm': size, 'pid': process.pid}), flush=True)
                code = process.wait()
                process = None
            if code:
                raise RuntimeError(f'Size {size} benchmark exited {code}; see its log')
            verify_inputs()
            result = json.loads((directory/'results.json').read_text())
            assert [r['controller'] for r in result] == list(models), 'Unexpected controller scope'
            assert all(r['conditions'] == 1 for r in result)
            assert json.loads((directory/'status.json').read_text())['status'] == 'completed'
            metadata = json.loads((directory/'manifest.json').read_text())['checkpoint_metadata']
            for name, model in models.items():
                assert metadata[name]['training_steps'] == 786432
                assert metadata[name]['uses_tactile'] == (model['modality'] == 'tactile')
            state['completed_sizes'].append(size)
            state.update(completed_episodes=6*len(state['completed_sizes']), active_size_mm=None, child_pid=None, updated_utc=now())
            write(out/'status.json', state)
            audit_code = subprocess.call([sys.executable, '-m', 'ur7e_xhand_sim.residual_rl.size_results', '--root', str(out)], cwd=ROOT)
            if audit_code:
                raise RuntimeError(f'Partial results audit exited {audit_code}')
            audit = json.loads((out/'size_audit.json').read_text())
            if audit['counts']['invalid'] or not audit['input_integrity_passed']:
                raise RuntimeError(f'Size {size} failed independent audit; see size_audit.json')
            print(json.dumps({'event': 'size_completed', 'size_mm': size, 'completed_episodes': state['completed_episodes']}), flush=True)
        verify_inputs()
        code = subprocess.call([sys.executable, '-m', 'ur7e_xhand_sim.residual_rl.size_results', '--root', str(out), '--final'], cwd=ROOT)
        if code:
            raise RuntimeError(f'Independent results audit exited {code}')
        state.update(status='completed', completed_utc=now())
        write(out/'status.json', state)
    except BaseException as exc:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        state.update(status='failed', error=repr(exc), active_size_mm=None, child_pid=None, updated_utc=now())
        write(out/'status.json', state)
        raise


if __name__ == '__main__':
    main()
