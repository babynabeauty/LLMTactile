"""Durable sequential runner for the frozen multi-object experiment.

Runs training, both validation checkpoints, preregistered selection, and exactly
one final selected-model test with fixed controls. Stops at the first error.
It does not use scheduler services and does not detach itself from its caller.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback

from .multiobject_protocol import validate_protocol
from .multiobject_report import build_selection, collect, sha256


ROOT=Path(__file__).resolve().parents[2]
MODULE='ur7e_xhand_sim.residual_rl.'
STEPS=('train','validation_stage1','audit_validation_stage1','validation_stage2',
       'audit_validation_stage2','selection','test','report')


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp')
    with temporary.open('w') as stream:
        json.dump(value,stream,indent=2,ensure_ascii=False,allow_nan=False)
        stream.write('\n');stream.flush();os.fsync(stream.fileno())
    temporary.replace(path)


def source_paths(root=ROOT):
    paths=set((root/'ur7e_xhand_sim').rglob('*.py'))
    paths.update((root/'Genesis/genesis').rglob('*.py'))
    paths.update(p for p in (root/'ur7e_xhand_sim/sources').rglob('*') if p.is_file())
    paths.update(p for p in (root/'ur7e_xhand_sim/generated/rigid_shapes').rglob('*') if p.is_file())
    requirement=root/'ur7e_xhand_sim/residual_rl/requirements.txt'
    if requirement.is_file():paths.add(requirement)
    return sorted(p for p in paths if p.is_file() and '__pycache__' not in p.parts)


def source_inventory(root=ROOT):
    return {str(path.relative_to(root)):sha256(path) for path in source_paths(root)}


def frozen_integrity(output, manifest, *, root=ROOT):
    protocol_path=output/'inputs/protocol.json'
    if sha256(protocol_path)!=manifest['protocol_sha256']:
        raise ValueError('Frozen experiment document changed')
    actual=source_inventory(root)
    if actual!=manifest['source_sha256']:
        changed=sorted(set(actual)^set(manifest['source_sha256']) | {
            key for key in actual.keys() & manifest['source_sha256'].keys()
            if actual[key]!=manifest['source_sha256'][key]})
        raise ValueError(f'Frozen source tree changed: {changed[:12]}')
    protocol=json.loads(protocol_path.read_text())
    validate_protocol(protocol,require_demonstrations=True)
    return protocol


def initialize_campaign(source_protocol, output, *, resume=False, root=ROOT):
    source_protocol=Path(source_protocol).expanduser().resolve()
    output=Path(output).expanduser().resolve()
    protocol=json.loads(source_protocol.read_text())
    validate_protocol(protocol,require_demonstrations=True)
    if output.exists():
        if not resume:
            raise FileExistsError('Campaign output exists; use --resume only for this exact campaign')
        manifest=json.loads((output/'manifest.json').read_text())
        if manifest['protocol_sha256']!=sha256(source_protocol):
            raise ValueError('Resume protocol differs from frozen campaign')
        frozen_integrity(output,manifest,root=root)
        return manifest
    if resume:
        raise FileNotFoundError('Cannot resume a campaign that does not exist')
    output.mkdir(parents=True,exist_ok=False)
    (output/'inputs').mkdir();(output/'logs').mkdir();(output/'steps').mkdir()
    shutil.copy2(source_protocol,output/'inputs/protocol.json')
    inventory=source_inventory(root)
    for relative in inventory:
        destination=output/'source_snapshot'/relative
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(root/relative,destination)
        if sha256(destination)!=inventory[relative]:
            raise ValueError(f'Source changed during snapshot: {relative}')
    manifest=dict(schema_version=1,experiment_id=protocol['experiment_id'],
        created_utc=utc_now(),source_protocol=str(source_protocol),
        protocol_path=str(output/'inputs/protocol.json'),protocol_sha256=sha256(source_protocol),
        source_root=str(root.resolve()),source_sha256=inventory,
        python_executable=sys.executable,steps=list(STEPS),
        policy='Stop on failure. Never select from test. Every command runs against frozen protocol and verified current source; no existing stage output is overwritten.',
        demonstration_sha256={o['object_id']:o['trajectory_sha256'] for o in protocol['objects']})
    atomic_json(output/'manifest.json',manifest)
    frozen_integrity(output,manifest,root=root)
    atomic_json(output/'status.json',dict(status='ready',created_utc=utc_now(),completed_steps=[]))
    return manifest


def step_command(step, output, *, python=sys.executable):
    output=Path(output)
    common=['--experiment-json',str(output/'inputs/protocol.json')]
    if step=='train':
        return [python,'-m',MODULE+'multiobject_train',*common,'--output',str(output/'train'),'--suite']
    if step.startswith('validation_stage'):
        stage=step.removeprefix('validation_')
        command=[python,'-m',MODULE+'multiobject_benchmark',*common,
            '--output',str(output/'eval/validation'/stage),'--split','validation',
            '--training-root',str(output/'train'),'--checkpoint-stage',stage]
        if stage=='stage2':command+=['--fixed-gains']
        return command
    if step=='test':
        return [python,'-m',MODULE+'multiobject_benchmark',*common,
            '--output',str(output/'eval/test'),'--split','test',
            '--selection-json',str(output/'selection.json')]
    if step=='report':
        return [python,'-m',MODULE+'multiobject_report','--root',str(output),
            '--protocol',str(output/'inputs/protocol.json'),'--split','test',
            '--output',str(output/'reports/test')]
    return None


def artifact_hashes(output, step, protocol):
    """Return the completed step's critical artifacts, refusing partial success."""
    output=Path(output);paths=[]
    if step=='train':
        status=output/'train/suite_status.json'
        if json.loads(status.read_text()).get('status')!='completed':
            raise ValueError('Training suite has not completed')
        paths.append(status)
        for seed in protocol['training']['seeds']:
            for method in protocol['training']['methods']:
                for stage in protocol['training']['stages']:
                    folder=output/'train'/f"seed{seed}_{method}_{stage['name']}"
                    state=json.loads((folder/'status.json').read_text())
                    info=json.loads((folder/'manifest.json').read_text())
                    checkpoint=folder/'final_model.zip'
                    if (state.get('status')!='completed' or state.get('final_checkpoint_sha256')!=sha256(checkpoint)
                            or info.get('experiment_sha256')!=sha256(output/'inputs/protocol.json')
                            or info.get('seed')!=seed or info.get('method')!=method
                            or info.get('stage',{}).get('name')!=stage['name']
                            or info.get('diagnostic_max_conditions') is not None
                            or info.get('requested_timesteps')!=stage['requested_timesteps']
                            or state.get('actual_additional_timesteps',0)<stage['requested_timesteps']):
                        raise ValueError(f'Training checkpoint identity/budget/hash failed: {folder}')
                    paths.extend([folder/'status.json',folder/'manifest.json',checkpoint])
    elif step.startswith('validation_stage') or step=='test':
        directory=output/'eval/test' if step=='test' else output/'eval/validation'/step.removeprefix('validation_')
        if json.loads((directory/'status.json').read_text()).get('status')!='completed':
            raise ValueError(f'Evaluation did not complete: {step}')
        info=json.loads((directory/'manifest.json').read_text())
        expected_split='test' if step=='test' else 'validation'
        if info.get('experiment_sha256')!=sha256(output/'inputs/protocol.json') or info.get('split')!=expected_split:
            raise ValueError('Completed evaluation belongs to a different experiment/split')
        paths.extend([directory/'status.json',directory/'manifest.json',directory/'results.json'])
        paths.extend(directory.rglob('evaluation.json'));paths.extend(directory.rglob('trace.npz'))
        paths.extend(directory.rglob('environment_protocol.json'));paths.extend(directory.rglob('prefix_consistency.json'))
    elif step.startswith('audit_validation_'):
        paths=[output/'audits'/f'{step}.json']
        if json.loads(paths[0].read_text())['status']!='audited_complete':
            raise ValueError('Validation audit is incomplete or failed')
    elif step=='selection':
        paths=[output/'selection.json']
        value=json.loads(paths[0].read_text())
        expected={f'seed{seed}_{method}' for seed in protocol['training']['seeds'] for method in protocol['training']['methods']}
        if (set(value['selected_models'])!=expected or value.get('selection_split')!='validation'
                or value.get('test_outcomes_read') is not False
                or value.get('protocol_sha256')!=sha256(output/'inputs/protocol.json')):
            raise ValueError('Selection is incomplete or not bound to frozen validation protocol')
        for item in value['selected_models'].values():
            if sha256(item['path'])!=item['sha256']:raise ValueError('Selected checkpoint hash changed')
    elif step=='report':
        paths=[output/'reports/test/report.json',output/'reports/test/index.html']
        record=json.loads(paths[0].read_text())
        if record['status']!='audited_complete' or record.get('protocol_sha256')!=sha256(output/'inputs/protocol.json'):
            raise ValueError('Final test audit is not complete')
    else:
        raise ValueError(f'Unknown campaign step {step}')
    if not paths:raise ValueError(f'No output artifacts for {step}')
    return {str(p.relative_to(output)):sha256(p) for p in sorted(set(paths))}


def completed_step_is_valid(output, step, manifest, protocol):
    path=output/'steps'/f'{step}.json'
    if not path.is_file():return False
    state=json.loads(path.read_text())
    if state.get('status')!='completed':return False
    if state.get('protocol_sha256')!=manifest['protocol_sha256']:
        raise ValueError('Completed stage belongs to a different frozen protocol')
    if state.get('artifact_sha256')!=artifact_hashes(output,step,protocol):
        raise ValueError(f'Completed step artifacts changed: {step}')
    return True


def archive_incomplete_outputs(output, step, *, retry_incomplete, protocol):
    """Preserve failed attempts; never overwrite or delete their evidence."""
    if step=='train':
        directory=output/'train'
        if not directory.exists():return
        incomplete=[]
        for seed in protocol['training']['seeds']:
            for method in protocol['training']['methods']:
                for stage in protocol['training']['stages']:
                    path=directory/f"seed{seed}_{method}_{stage['name']}"
                    if not path.exists():continue
                    status_path=path/'status.json'
                    status=json.loads(status_path.read_text()) if status_path.is_file() else {}
                    if status.get('status')!='completed':
                        incomplete.append(path)
                        log=directory/(path.name+'.log')
                        if log.exists():incomplete.append(log)
        targets=incomplete
    elif step.startswith('validation_stage'):
        targets=[output/'eval/validation'/step.removeprefix('validation_')]
    elif step=='test':targets=[output/'eval/test']
    elif step=='selection':targets=[output/'selection.json']
    elif step=='report':targets=[output/'reports/test']
    else:targets=[output/'audits'/f'{step}.json']
    targets=[path for path in targets if path.exists()]
    if not targets:return
    if not retry_incomplete:
        raise FileExistsError(f'Incomplete {step} outputs exist; inspect them, then use --retry-incomplete to archive and retry')
    destination=output/'incomplete_attempts'/f'{step}_{time.time_ns()}'
    destination.mkdir(parents=True,exist_ok=False)
    for path in targets:
        relative=path.relative_to(output)
        archived=destination/relative;archived.parent.mkdir(parents=True,exist_ok=True)
        shutil.move(str(path),str(archived))


def run_process(command, log_path, *, on_poll, root=ROOT):
    with Path(log_path).open('x') as log:
        process=subprocess.Popen(command,cwd=root,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            while process.poll() is None:
                on_poll(process.pid)
                try:process.wait(timeout=15)
                except subprocess.TimeoutExpired:pass
            if process.returncode:
                raise subprocess.CalledProcessError(process.returncode,command)
        except BaseException:
            if process.poll() is None:
                os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid,signal.SIGKILL);process.wait()
            raise


def local_step(step, output, protocol):
    if step.startswith('audit_validation_'):
        stage=step.removeprefix('audit_validation_')
        report=collect(output,protocol,'validation',stage)
        atomic_json(output/'audits'/f'{step}.json',report)
        if report['status']!='audited_complete':
            raise RuntimeError(f'{stage} validation audit failed/incomplete; see saved audit')
        return
    if step=='selection':
        mapping={f'seed{seed}_{method}':{stage['name']:str(output/'train'/f"seed{seed}_{method}_{stage['name']}"/'final_model.zip')
                  for stage in protocol['training']['stages']}
                 for seed in protocol['training']['seeds'] for method in protocol['training']['methods']}
        selected=build_selection(output,protocol,mapping)
        selected['protocol_path']=str(output/'inputs/protocol.json')
        selected['protocol_sha256']=sha256(output/'inputs/protocol.json')
        if (output/'selection.json').exists():raise FileExistsError('Selection already exists')
        atomic_json(output/'selection.json',selected)
        return
    raise ValueError(f'No local implementation for {step}')


def execute_campaign(output, manifest, *, retry_incomplete=False, root=ROOT,
                     process_runner=run_process, local_runner=local_step):
    output=Path(output);completed=[];step=None;started=utc_now()
    try:
        protocol=frozen_integrity(output,manifest,root=root)
        for step in STEPS:
            frozen_integrity(output,manifest,root=root)
            if completed_step_is_valid(output,step,manifest,protocol):
                completed.append(step);continue
            # Recover a child that finished before its parent recorded the
            # completion marker; do not accidentally repeat a completed test.
            try:
                recovered=artifact_hashes(output,step,protocol)
            except (OSError,KeyError,ValueError):
                recovered=None
            if recovered is not None:
                atomic_json(output/'steps'/f'{step}.json',dict(status='completed',step=step,
                    completed_utc=utc_now(),recovered_completed_artifacts=True,
                    protocol_sha256=manifest['protocol_sha256'],artifact_sha256=recovered))
                completed.append(step);continue
            archive_incomplete_outputs(output,step,retry_incomplete=retry_incomplete,protocol=protocol)
            attempt=time.time_ns();log=output/'logs'/f'{STEPS.index(step)+1:02d}_{step}_{attempt}.log'
            state=dict(status='running',step=step,started_utc=utc_now(),log_path=str(log),
                       protocol_sha256=manifest['protocol_sha256'])
            atomic_json(output/'steps'/f'{step}.json',state)
            def heartbeat(pid=None):
                atomic_json(output/'status.json',dict(status='running',step=step,child_pid=pid,
                    started_utc=started,updated_utc=utc_now(),completed_steps=completed,
                    log_path=str(log),protocol_sha256=manifest['protocol_sha256']))
            heartbeat()
            command=step_command(step,output)
            if command:
                state['command']=command;atomic_json(output/'steps'/f'{step}.json',state)
                process_runner(command,log,on_poll=heartbeat,root=root)
            else:
                with log.open('x') as handle:
                    try:
                        local_runner(step,output,protocol)
                        handle.write('Local read-only audit/selection completed.\n')
                    except BaseException:
                        traceback.print_exc(file=handle);raise
            frozen_integrity(output,manifest,root=root)
            state.update(status='completed',completed_utc=utc_now(),
                         artifact_sha256=artifact_hashes(output,step,protocol))
            atomic_json(output/'steps'/f'{step}.json',state)
            completed.append(step)
        atomic_json(output/'status.json',dict(status='completed',completed_utc=utc_now(),
            completed_steps=completed,report_path=str(output/'reports/test/index.html'),
            protocol_sha256=manifest['protocol_sha256']))
    except BaseException as exc:
        failure=dict(status='failed',step=step,failed_utc=utc_now(),completed_steps=completed,
            error=repr(exc),traceback=traceback.format_exc(),protocol_sha256=manifest['protocol_sha256'])
        atomic_json(output/'status.json',failure)
        if step is not None:
            path=output/'steps'/f'{step}.json'
            previous=json.loads(path.read_text()) if path.is_file() else {}
            atomic_json(path,{**previous,**failure})
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment-json',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True,help='New dedicated campaign directory; existing demonstration files stay outside it')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--retry-incomplete',action='store_true',help='With resume, archive unfinished stage outputs before rerunning; never overwrite them')
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args()
    if args.retry_incomplete and not args.resume:parser.error('--retry-incomplete requires --resume')
    if args.dry_run:
        protocol=json.loads(args.experiment_json.read_text());validate_protocol(protocol)
        print(json.dumps({'output':str(args.output.resolve()),'steps':[
            {'name':s,'command':step_command(s,args.output.resolve())} for s in STEPS],
            'gpu_started':False,'requires_verified_demonstrations_before_execution':True},indent=2))
        return
    # Convert process-manager termination into the same durable failure path.
    def terminated(signum,frame):raise KeyboardInterrupt(f'Campaign terminated by signal {signum}')
    signal.signal(signal.SIGTERM,terminated)
    output=args.output.expanduser().resolve()
    manifest=initialize_campaign(args.experiment_json,output,resume=args.resume)
    with (output/'.campaign.lock').open('a+') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise SystemExit('This campaign already has an active runner')
        execute_campaign(output,manifest,retry_incomplete=args.retry_incomplete)


if __name__=='__main__':main()
