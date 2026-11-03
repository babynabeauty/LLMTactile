"""Local simulation bridge; --online runs continuous 60 Hz control while Astra thinks."""
import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer, ThreadingHTTPServer
from pathlib import Path
import threading


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--port', type=int, default=8767)
    p.add_argument('--output')
    p.add_argument('--object-spec-json', type=json.loads, required=True)
    p.add_argument('--cube-mass-g', type=float, default=50.)
    p.add_argument('--cube-friction', type=float, default=.8)
    p.add_argument('--robot-friction', type=float, default=.8)
    p.add_argument('--table-contact-force-limit-n', type=float, default=5.)
    p.add_argument('--online', action='store_true', help='Keep simulation/control running during Astra inference')
    p.add_argument('--residual-checkpoint', type=Path,
                   help='60 Hz PPO checkpoint; --online without it uses zero residual for integration tests')
    p.add_argument('--astra-level', type=int, choices=(1, 2, 3, 4), default=4)
    p.add_argument('--no-realtime', action='store_true',
                   help='Online test mode: run as fast as possible; simulated control is still 60 Hz')
    args = p.parse_args()
    if (args.residual_checkpoint or args.no_realtime) and not args.online:
        p.error('--residual-checkpoint and --no-realtime require --online')
    from environment import Sandbox
    # Binding before expensive scene construction reports port conflicts promptly.
    http_class = ThreadingHTTPServer if args.online else HTTPServer
    http = http_class(('127.0.0.1', args.port), BaseHTTPRequestHandler)
    try:
        sim = Sandbox(output_dir=args.output, object_spec=args.object_spec_json, cube_mass_kg=args.cube_mass_g/1000,
            cube_friction=args.cube_friction, robot_friction=args.robot_friction,
            table_contact_force_limit_N=args.table_contact_force_limit_n, physics_dt_s=1/300, control_substeps=5,
            noslip_iterations=5, deterministic=True)
    except BaseException:
        http.server_close()
        raise
    trial, closed, scheduler = None, False, None
    cache_lock, cache = threading.Lock(), {}

    def dispatch(command, payload):
        """Called exclusively by the main scene-owner thread in both modes."""
        nonlocal trial, closed
        if command == 'trial_start':
            if trial is not None:
                raise ValueError('One trial per bridge server')
            from base_policy_v1 import BasePolicyV1
            trial = BasePolicyV1(sim, payload['label'], payload['demo'], payload.get('placement', {}))
            if args.online:
                try:
                    trial.configure_online(residual_checkpoint=args.residual_checkpoint,
                                           zero_residual=args.residual_checkpoint is None,
                                           astra_level=args.astra_level)
                except Exception as exc:
                    trial.failed = 'Online controller setup failed: '+str(exc)
                    raise
            return trial.observe()
        if command in ('trial_observe', 'trial_act', 'trial_score'):
            if trial is None:
                raise ValueError('Start a trial first')
            if command == 'trial_observe':
                return trial.capture()
            if command == 'trial_act':
                return trial.execute_iter(payload) if args.online else trial.execute(payload)
            if not trial.finished and not trial.failed:
                raise ValueError('Finish policy control before reading evaluation')
            return trial.score()
        if command == 'describe':
            return sim.describe()
        if command == 'observe':
            return sim.observe(payload.get('full', False))
        if command == 'status':
            return dict(sim.status(), online_control=False)
        if command == 'evaluate':
            return sim.evaluate()
        if command == 'close':
            closed = True
            return {'closed': True}
        raise ValueError('Unknown command')

    def should_tick():
        return trial is not None and not trial.finished and not trial.failed

    def publish():
        # Do not render/read sensors here: snapshots must not add a second GPU
        # readback to each control tick. Image bytes update at decision captures.
        snapshot = dict(sim.status(), online_control=True, paused=not should_tick(),
                        phase=trial.phase if trial else None, trial_fault=trial.failed if trial else None)
        with cache_lock:
            cache['status'] = snapshot
            cache['images'] = dict(sim.image_bytes)

    def control_fault(exc):
        if trial is not None:
            trial.failed = f'Online control fault: {type(exc).__name__}: {exc}'

    if args.online:
        from online_runtime import ControlScheduler
        scheduler = ControlScheduler(dispatch, lambda: trial.idle_tick(), should_tick,
            control_hz=1/sim.control_dt_s, publish=publish, on_fault=control_fault,
            timing_path=sim.output_dir/'online_timing.jsonl', realtime=not args.no_realtime)
        # No HTTP worker calls describe(), status(), or reads sim.image_bytes.
        cache['describe'] = dict(sim.describe(), online_control=True,
            online_residual_mode='checkpoint' if args.residual_checkpoint else 'zero_residual',
            residual_checkpoint=str(args.residual_checkpoint.resolve()) if args.residual_checkpoint else None)
        publish()

    class Handler(BaseHTTPRequestHandler):
        def respond(self, body, mime='application/json', code=200):
            self.send_response(code)
            self.send_header('Content-Type', mime)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass  # A disconnected network client cannot stop the owner loop.

        def cached_status(self):
            with cache_lock:
                result = dict(cache['status'])
            result['online_runtime'] = scheduler.snapshot()
            return result

        def do_GET(self):
            path = self.path.split('?')[0]
            if args.online:
                if path == '/status':
                    self.respond(json.dumps({'ok': True, 'result': self.cached_status()}, allow_nan=False).encode())
                    return
                with cache_lock:
                    data = cache['images'].get(path[1:-4]) if path.endswith('.jpg') else None
            else:
                data = sim.image_bytes.get(path[1:-4]) if path.endswith('.jpg') else None
            if data is not None:
                self.respond(data, 'image/jpeg')
            else:
                self.respond(b'Not found', 'text/plain', 404)

        def do_POST(self):
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 <= size <= 1000000:
                    raise ValueError('Invalid request size')
                payload = json.loads(self.rfile.read(size) or b'{}')
                if not isinstance(payload, dict):
                    raise ValueError('Request must be a JSON object')
                command = self.path.strip('/')
                if args.online and command == 'status':
                    result = self.cached_status()
                elif args.online and command == 'describe':
                    with cache_lock:
                        result = cache['describe']
                elif args.online:
                    # The HTTP thread waits; physical control remains on main.
                    result = scheduler.submit(command, payload).result()
                else:
                    result = dispatch(command, payload)
                self.respond(json.dumps({'ok': True, 'result': result}, allow_nan=False).encode())
            except (ValueError, TypeError, KeyError) as exc:
                self.respond(json.dumps({'ok': False, 'error': str(exc)}).encode(), code=400)
            except Exception as exc:
                self.respond(json.dumps({'ok': False, 'error': str(exc)}).encode(), code=500)

        def log_message(self, *args):
            pass

    http.RequestHandlerClass = Handler
    worker = None
    print(f'UR7E_XHAND_READY http://127.0.0.1:{args.port}', flush=True)
    try:
        if args.online:
            worker = threading.Thread(target=http.serve_forever, kwargs={'poll_interval': .1},
                                      name='bridge-http', daemon=True)
            worker.start()
            scheduler.run()
        else:
            while not closed:
                http.handle_request()
    finally:
        if scheduler:
            scheduler.close()
            (sim.output_dir/'online_status.json').write_text(json.dumps(scheduler.snapshot(), indent=2, allow_nan=False))
        if worker:
            http.shutdown()
            worker.join(timeout=2.)
        http.server_close()
        if trial:
            for name in ('record', 'eval_record'):
                record = getattr(trial, name, None)
                if record and not record.closed:
                    record.close()
        sim.scene.destroy()


if __name__ == '__main__':
    main()
