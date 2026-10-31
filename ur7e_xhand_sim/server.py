"""Local-only interactive simulation service."""
import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from environment import Sandbox, HERE


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--port',type=int,default=8767)
    parser.add_argument('--output')
    parser.add_argument('--online-ppo-config',help='Opt-in frozen tactile PPO config for S8V4; enables 300/60Hz noslip5')
    parser.add_argument('--matched-online-physics',action='store_true',help='300Hz physics/60Hz sampling, noslip5, deterministic; independent of PPO')
    parser.add_argument('--ppo-replay-validation',action='store_true',help='Explicit diagnostic mode: 300Hz/60Hz, noslip5, headless; enables replay endpoints')
    parser.add_argument('--scenario',choices=['original','small','large','clutter','small_clutter'],default='original')
    parser.add_argument('--object-shape',choices=['cube','cuboid','hex_prism','cylinder','sphere'],default=None,
        help='Rigid multi-object study shape; explicit selection places its bottom on the table. Omit for the legacy cube scene.')
    mass_group=parser.add_mutually_exclusive_group()
    mass_group.add_argument('--cube-density',type=float,help='Target rigid object density in kg/m^3; default 500')
    mass_group.add_argument('--cube-mass-g',type=float,help='Target rigid object mass in grams; uses the selected shape volume')
    parser.add_argument('--cube-friction',type=float,default=.8)
    parser.add_argument('--robot-friction',type=float,default=.8)
    parser.add_argument('--silicone',action='store_true',help='Add experimental FEM silicone block; soft tactile not connected')
    parser.add_argument('--silicone-substeps',type=int,default=None,help='Physics substeps: IPC 2, legacy 20 unless specified')
    parser.add_argument('--silicone-solver',choices=['legacy','ipc'],default='ipc')
    parser.add_argument('--silicone-maxvolume',type=float,default=1e-7)
    parser.add_argument('--table-contact-force-limit-n',type=float,default=5.0,
        help='Online skill robot/table normal load sum limit in N; 0 stops any positive load (default5)')
    args=parser.parse_args()
    http=HTTPServer(('127.0.0.1',args.port),BaseHTTPRequestHandler)
    sim=Sandbox(output_dir=args.output,scenario=args.scenario,cube_density=args.cube_density,
        cube_mass_kg=None if args.cube_mass_g is None else args.cube_mass_g/1000,
        cube_friction=args.cube_friction,robot_friction=args.robot_friction,silicone=args.silicone,
        table_contact_force_limit_N=args.table_contact_force_limit_n,
        silicone_substeps=args.silicone_substeps,silicone_solver=args.silicone_solver,
        silicone_maxvolume=args.silicone_maxvolume,object_spec=args.object_shape,
        **({'render':not args.ppo_replay_validation,'physics_dt_s':1/300,'control_substeps':5,'noslip_iterations':5,'deterministic':True} if args.ppo_replay_validation or args.online_ppo_config or args.matched_online_physics else {}))
    if args.online_ppo_config:
        from online_ppo import OnlinePPO
        sim.online_ppo=OnlinePPO(sim,args.online_ppo_config)
    ppo_replay=None
    trial=None
    teacher=None
    soft_trial=None
    closed=False
    class Handler(BaseHTTPRequestHandler):
        def respond(self,body,mime='application/json',code=200):
            self.send_response(code)
            self.send_header('Content-Type',mime)
            self.send_header('Content-Length',str(len(body)))
            self.send_header('Cache-Control','no-store')
            self.end_headers()
            self.wfile.write(body)
        def do_GET(self):
            path=self.path.split('?')[0]
            if path=='/':self.respond((HERE/'index.html').read_bytes(),'text/html; charset=utf-8')
            elif path.endswith('.jpg') and path[1:-4] in sim.image_bytes:
                self.respond(sim.image_bytes[path[1:-4]],'image/jpeg')
            else:self.respond(b'Not found','text/plain',404)
        def do_POST(self):
            nonlocal closed,trial,teacher,soft_trial,ppo_replay
            try:
                size=int(self.headers.get('Content-Length','0'))
                if not 0<=size<=1000000:raise ValueError('Invalid request size')
                payload=json.loads(self.rfile.read(size) or b'{}')
                if not isinstance(payload,dict):raise ValueError('Payload must be an object')
                command=self.path.strip('/')
                if command.startswith('ppo_replay_'):
                    if not args.ppo_replay_validation:raise ValueError('Replay validation mode not enabled')
                    if command=='ppo_replay_start':
                        if ppo_replay is not None:raise ValueError('One replay per validation server')
                        from ppo_bridge_replay import PPOBridgeReplay
                        ppo_replay=PPOBridgeReplay(sim,payload);result=ppo_replay.observe()
                    elif command=='ppo_replay_advance' and ppo_replay is not None:
                        result=ppo_replay.advance(payload)
                    else:raise ValueError('Unknown replay command or replay not started')
                    self.respond(json.dumps({'ok':True,'result':result},allow_nan=False).encode());return
                if ppo_replay is not None and command not in ('describe','observe','status','evaluate','close'):
                    raise ValueError('PPO validation replay owns motion')
                if command in ['step','move','hand','reset','rule_demo_start','s8v4_start'] and soft_trial is not None and not soft_trial.finished:
                    raise ValueError('Silicone agent trial owns control until finished')
                if command=='silicone_start':
                    if soft_trial is not None and not soft_trial.finished:raise ValueError('Silicone trial active')
                    if teacher is not None and not teacher.finished:raise ValueError('Rule demo active')
                    if trial is not None and not trial.finished and not trial.failed:raise ValueError('S8V4 trial active')
                    from silicone_squeeze import SiliconeSqueeze
                    soft_trial=SiliconeSqueeze(sim,payload.get('label'))
                    result=soft_trial.observe()
                elif command=='silicone_act':
                    if soft_trial is None:raise ValueError('Start silicone trial first')
                    result=soft_trial.execute(payload)
                elif command=='silicone_observe':
                    if soft_trial is None:raise ValueError('Start silicone trial first')
                    result=soft_trial.observe()
                elif command=='rule_demo_start':
                    if teacher is not None and not teacher.finished:raise ValueError('Rule demonstration active')
                    if trial is not None and not trial.finished and not trial.failed:raise ValueError('S8V4 trial active')
                    from rule_demo import RuleDemo
                    if trial is not None:
                        trial.record.close();trial.eval_record.close();trial=None
                    teacher=RuleDemo(sim,payload.get('label'),payload.get('options'))
                    result=teacher.observe()
                elif command=='rule_demo_advance':
                    if teacher is None:raise ValueError('Start rule demonstration first')
                    result=teacher.advance()
                    if teacher.finished:result['score']=teacher.score
                elif command=='s8v4_start':
                    if hasattr(sim,'online_ppo') and trial is not None:
                        raise ValueError('Online PPO requires a fresh server for each trial')
                    if teacher is not None and not teacher.finished:raise ValueError('Rule demonstration active')
                    if trial is not None and not trial.finished and not trial.failed:
                        raise ValueError('An S8V4 trial is already active')
                    from s8v4 import S8V4
                    if trial is not None:
                        trial.record.close();trial.eval_record.close()
                    if not sim.legacy_object_placement:
                        if not payload.get('reference'):
                            raise ValueError('Explicit object shape requires a demonstration reference for S8V4')
                        from shape_s8v4 import ShapeS8V4
                        trial=ShapeS8V4(sim,payload.get('label'),payload['reference'])
                    elif payload.get('reference'):
                        from verified_s8v4 import VerifiedS8V4
                        trial=VerifiedS8V4(sim,payload.get('label'),payload['reference'])
                    else:
                        trial=S8V4(sim,payload.get('label'),payload.get('driver','vlm_policy_CodexAppServerPolicy'))
                    if hasattr(sim,'online_ppo'):
                        manifest_path=trial.out/'manifest.json'
                        manifest=json.loads(manifest_path.read_text())
                        manifest['online_ppo']=sim.online_ppo.meta
                        manifest['online_ppo_log']=str(sim.output_dir/'online_ppo.jsonl')
                        manifest_path.write_text(json.dumps(manifest,indent=2))
                    result=trial.observe()
                elif command=='s8v4_observe':
                    if trial is None:raise ValueError('Start an S8V4 trial first')
                    result=trial.capture()
                elif command=='s8v4_act':
                    if trial is None:raise ValueError('Start an S8V4 trial first')
                    result=trial.execute(payload)
                elif command=='s8v4_score':
                    if trial is None:raise ValueError('Start an S8V4 trial first')
                    if not trial.finished and not trial.failed:raise ValueError('Finish policy control before reading evaluation')
                    result=trial.score()
                elif command in ['step','move','hand','reset'] and teacher is not None and not teacher.finished:
                    raise ValueError('Rule demonstration owns control until it finishes')
                elif command in ['step','move','hand','reset'] and trial is not None and not trial.finished and not trial.failed:
                    raise ValueError('S8V4 controls this trial; manual motion is disabled until it finishes')
                elif command=='describe':
                    result=sim.describe()
                    result['online_ppo']=sim.online_ppo.meta if hasattr(sim,'online_ppo') else {'enabled':False}
                elif command=='observe':result=sim.observe(payload.get('full',False))
                elif command=='status':result=sim.status()
                elif command=='reset':result=sim.reset(payload.get('seed',42))
                elif command in ['step','move','hand']:result=getattr(sim,command)(payload)
                elif command=='evaluate':result=sim.evaluate()
                elif command=='tactile_debug':
                    from tactile_debug import snapshot
                    result=snapshot(sim)
                elif command=='close':closed=True;result={'closed':True}
                else:raise ValueError('Unknown command')
                self.respond(json.dumps({'ok':True,'result':result},allow_nan=False).encode())
            except (ValueError,TypeError) as exc:
                self.respond(json.dumps({'ok':False,'error':str(exc)}).encode(),code=400)
            except Exception as exc:
                self.respond(json.dumps({'ok':False,'error':str(exc)}).encode(),code=500)
        def log_message(self,*args):pass
    http.RequestHandlerClass=Handler
    print(f'UR7E_XHAND_READY http://127.0.0.1:{args.port}',flush=True)
    try:
        while not closed:http.handle_request()
    finally:
        http.server_close()
        sim.scene.destroy()


if __name__=='__main__':main()
