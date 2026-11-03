"""Local-only simulation bridge for one BasePolicyV1 trial (300 Hz physics, 60 Hz control, noslip 5)."""
import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer

from environment import Sandbox


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--port', type=int, default=8767)
    p.add_argument('--output')
    p.add_argument('--object-spec-json', type=json.loads, required=True)
    p.add_argument('--cube-mass-g', type=float, default=50.)
    p.add_argument('--cube-friction', type=float, default=.8)
    p.add_argument('--robot-friction', type=float, default=.8)
    p.add_argument('--table-contact-force-limit-n', type=float, default=5.)
    args = p.parse_args()
    http = HTTPServer(('127.0.0.1', args.port), BaseHTTPRequestHandler)
    sim = Sandbox(output_dir=args.output, object_spec=args.object_spec_json, cube_mass_kg=args.cube_mass_g/1000,
        cube_friction=args.cube_friction, robot_friction=args.robot_friction,
        table_contact_force_limit_N=args.table_contact_force_limit_n, physics_dt_s=1/300, control_substeps=5,
        noslip_iterations=5, deterministic=True)
    trial, closed = None, False

    class Handler(BaseHTTPRequestHandler):
        def respond(self, body, mime='application/json', code=200):
            self.send_response(code)
            self.send_header('Content-Type', mime)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = self.path.split('?')[0]
            if path.endswith('.jpg') and path[1:-4] in sim.image_bytes:
                self.respond(sim.image_bytes[path[1:-4]], 'image/jpeg')
            else:
                self.respond(b'Not found', 'text/plain', 404)

        def do_POST(self):
            nonlocal trial, closed
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 <= size <= 1000000:
                    raise ValueError('Invalid request size')
                payload = json.loads(self.rfile.read(size) or b'{}')
                command = self.path.strip('/')
                if command == 'trial_start':
                    if trial is not None:
                        raise ValueError('One trial per bridge server')
                    from base_policy_v1 import BasePolicyV1
                    trial = BasePolicyV1(sim, payload['label'], payload['demo'], payload.get('placement', {}))
                    result = trial.observe()
                elif command in ('trial_observe', 'trial_act', 'trial_score'):
                    if trial is None:
                        raise ValueError('Start a trial first')
                    if command == 'trial_observe':
                        result = trial.capture()
                    elif command == 'trial_act':
                        result = trial.execute(payload)
                    else:
                        if not trial.finished and not trial.failed:
                            raise ValueError('Finish policy control before reading evaluation')
                        result = trial.score()
                elif command == 'describe':
                    result = sim.describe()
                elif command == 'observe':
                    result = sim.observe(payload.get('full', False))
                elif command == 'status':
                    result = sim.status()
                elif command == 'evaluate':
                    result = sim.evaluate()
                elif command == 'close':
                    closed, result = True, {'closed': True}
                else:
                    raise ValueError('Unknown command')
                self.respond(json.dumps({'ok': True, 'result': result}, allow_nan=False).encode())
            except (ValueError, TypeError) as exc:
                self.respond(json.dumps({'ok': False, 'error': str(exc)}).encode(), code=400)
            except Exception as exc:
                self.respond(json.dumps({'ok': False, 'error': str(exc)}).encode(), code=500)

        def log_message(self, *args):
            pass

    http.RequestHandlerClass = Handler
    print(f'UR7E_XHAND_READY http://127.0.0.1:{args.port}', flush=True)
    try:
        while not closed:
            http.handle_request()
    finally:
        http.server_close()
        sim.scene.destroy()


if __name__ == '__main__':
    main()
