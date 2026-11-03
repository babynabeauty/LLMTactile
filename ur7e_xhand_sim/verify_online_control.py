"""Verify continuous online control with recorded Astra actions and an injected wait.

This is a scheduler/controller integration check, not a fresh Astra rollout and
not a data-label assignment. The bridge owns all physics and model inference.
"""
from argparse import ArgumentParser
from copy import deepcopy
import json
from pathlib import Path
import time
import urllib.error
import urllib.request

import numpy as np


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False))


def load_rows(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def latest_actor_count(run_dir):
    """Read flushed local evidence; do not query privileged simulation state."""
    path = Path(run_dir)/'control.jsonl'
    if not path.exists():
        return None
    # The last line can be in flight while the server appends. Ignore incomplete JSON.
    with path.open('rb') as stream:
        stream.seek(max(0, path.stat().st_size-65536))
        lines = stream.read().splitlines()
    for line in reversed(lines):
        try:
            row = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if 'actor_inferences' in row:
            return dict(step=row['step'], count=row['actor_inferences'], active=row.get('actor_active', False))
    return None


def run(args):
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    calls, actions, delay, current = [], [], None, None
    outcome = dict(schema='online_control_verification_v1', planner='recorded_actions_with_injected_delay',
        fresh_astra_call=False, reference_run=str(args.reference_run.resolve()), demo=str(args.demo.resolve()),
        label=args.label, delay_requested_s=args.delay_seconds, calls=calls, actions=actions, completed=False)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def call(command, payload=None):
        started = time.monotonic()
        event = dict(command=command, started_monotonic_s=started)
        calls.append(event)
        request = urllib.request.Request(args.url.rstrip('/')+'/'+command,
            json.dumps(payload or {}).encode(), {'Content-Type': 'application/json'})
        try:
            with opener.open(request, timeout=args.request_timeout) as response:
                reply = json.load(response)
            if not reply.get('ok'):
                raise RuntimeError(reply.get('error', 'Bridge rejected request'))
            event['ok'] = True
            return reply['result']
        except urllib.error.HTTPError as exc:
            event['ok'] = False
            raise RuntimeError(f'{command} HTTP {exc.code}: {exc.read().decode(errors="replace")}') from exc
        except Exception:
            event['ok'] = False
            raise
        finally:
            event['wall_s'] = time.monotonic()-started
            write(out/'calls.json', calls)

    try:
        initial_status = call('status')
        if initial_status.get('online_control') is not True:
            raise ValueError('Verification requires server.py --online')
        description = call('describe')
        outcome.update(server_initial_status=initial_status, server_description=description)
        manifest = json.loads((args.reference_run/'bridge/manifest.json').read_text())
        models = sorted((args.reference_run/'agent').glob('model_*.json'))
        if not models:
            raise ValueError('reference-run/agent has no recorded model_*.json actions')
        recorded = [json.loads(path.read_text())['action'] for path in models]
        if not any(a.get('skill') == 'close' for a in recorded) or not any(a.get('skill') == 'test_lift' for a in recorded):
            raise ValueError('Reference must contain close followed by test_lift')
        current = call('trial_start', dict(label=args.label, demo=str(args.demo.resolve()), placement=manifest['placement']))
        outcome['run_dir'] = current['run_dir']
        selected = None
        for source, action in zip(models, recorded):
            action = deepcopy(action)
            action['observation_id'] = current['observation_id']
            if action['skill'] == 'approach':
                candidates = (current.get('proposals') or {}).get('candidates', [])
                if not candidates:
                    raise ValueError('Online locate returned no candidate offsets')
                requested = np.asarray(action['offset_xyz_m'])
                selected = min(candidates, key=lambda c: float(np.linalg.norm(np.asarray(c['offset_xyz_m'])-requested)))['offset_xyz_m']
                action['offset_xyz_m'] = selected
            elif action['skill'] in ('align', 'close') and selected is not None:
                action['offset_xyz_m'] = selected
            if action['skill'] == 'test_lift' and delay is None:
                before = call('status')
                actor_before = latest_actor_count(current['run_dir'])
                start = time.monotonic()
                time.sleep(args.delay_seconds)
                after = call('status')
                waited = time.monotonic()-start
                actor_after = latest_actor_count(current['run_dir'])
                ticks = after['episode_step']-before['episode_step']
                idle = after['online_runtime']['idle_ticks']-before['online_runtime']['idle_ticks']
                actor_delta = (actor_after['count']-actor_before['count']
                               if actor_before is not None and actor_after is not None else None)
                delay = dict(before=before, after=after, wall_s=waited, tick_delta=ticks,
                    idle_tick_delta=idle, measured_simulation_tick_hz=ticks/waited,
                    actor_before=actor_before, actor_after=actor_after, actor_inference_delta=actor_delta,
                    loop_advanced=ticks > 0 and idle > 0, actor_advanced=actor_delta is not None and actor_delta > 0,
                    note='Measured rate includes status latency; configured 60 Hz is not a wall-clock guarantee')
                outcome['injected_planner_delay'] = delay
                write(out/'delay.json', delay)
                if not delay['loop_advanced'] or not delay['actor_advanced']:
                    raise RuntimeError('Control/actor did not advance during the injected planner wait')
            reply = call('trial_act', action)
            current = reply['policy_observation']
            actions.append(dict(source=str(source.resolve()), action=action, receipt=reply['receipt'],
                                observation_id_after=current['observation_id']))
            write(out/'actions.json', actions)
            print(json.dumps(dict(skill=action['skill'], receipt=reply['receipt'], phase=current['phase'])), flush=True)
            if current.get('fault'):
                raise RuntimeError('Online executor fault: '+current['fault'])
            if action['status'] != 'continue':
                break
        else:
            action = dict(observation_id=current['observation_id'], status='done', explanation='Recorded verification sequence complete',
                          skill='none', offset_xyz_m=[0., 0., 0.], object_roi_xyxy=[0, 0, 0, 0])
            reply = call('trial_act', action)
            current = reply['policy_observation']
        if delay is None:
            raise RuntimeError('Reference finished before the injected post-close delay')
        score = call('trial_score')
        outcome['legacy_collection_score'] = score
        outcome['strict_outcome'] = dict(success=score['strict_success'], metrics=score['v1_metrics'],
            label=None, label_status='Online integration check; no S/R/U label assigned')
        controls = load_rows(Path(current['run_dir'])/'control.jsonl')
        actors = [row for row in controls if row.get('actor_active')]
        actor_every_tick = bool(actors and all(right['actor_inferences']-left['actor_inferences'] == 1
            for left, right in zip(actors, actors[1:])))
        outcome['actor_log_audit'] = dict(active_ticks=len(actors), updates_every_active_tick=actor_every_tick,
            final_inferences=max((row.get('actor_inferences', 0) for row in controls), default=0))
        outcome['server_final_status'] = call('status')
        outcome['completed'] = True
        outcome['integration_passed'] = bool(delay['loop_advanced'] and delay['actor_advanced']
            and actor_every_tick)
        write(out/'validation.json', outcome)
        print(json.dumps(dict(completed=True, integration_passed=outcome['integration_passed'],
            legacy_success=score['success'], strict_success=outcome['strict_outcome']['success'],
            validation=str(out/'validation.json'))), flush=True)
        return outcome
    except BaseException as exc:
        outcome['error'] = dict(type=type(exc).__name__, message=str(exc))
        write(out/'error.json', outcome['error'])
        write(out/'validation.json', outcome)
        raise
    finally:
        if args.close_server:
            try:
                outcome['close_result'] = call('close')
            except Exception as exc:
                outcome['close_error'] = str(exc)
        write(out/'validation.json', outcome)


def main():
    p = ArgumentParser(description=__doc__)
    p.add_argument('--url', default='http://127.0.0.1:8767')
    p.add_argument('--demo', type=Path, required=True)
    p.add_argument('--reference-run', type=Path, required=True)
    p.add_argument('--label', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--delay-seconds', type=float, default=3.)
    p.add_argument('--request-timeout', type=float, default=300.)
    p.add_argument('--keep-server', dest='close_server', action='store_false', default=True)
    args = p.parse_args()
    if not np.isfinite(args.delay_seconds) or not 0 < args.delay_seconds <= 60:
        p.error('--delay-seconds must be in (0, 60]')
    run(args)


if __name__ == '__main__':
    main()
