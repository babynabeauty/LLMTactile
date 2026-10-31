"""Interactive Panda + Wuji physics sandbox. No rollout/dataset collection."""
import argparse
import io
import json
import os
from pathlib import Path
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
os.environ.setdefault('OMP_NUM_THREADS', '8')
sys.path.insert(0, str(ROOT / 'astra_bridge'))
from protocol import validate_step
from build_model import build_model


class Sandbox:
    def __init__(self):
        import genesis as gs
        import numpy as np
        from PIL import Image
        self.gs, self.np, self.Image = gs, np, Image
        gs.init(backend=gs.gpu, precision='32', logging_level='warning')
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=0.005, substeps=1),
            rigid_options=gs.options.RigidOptions(enable_self_collision=True, enable_joint_limit=True,
                                                  enable_neutral_collision=True),
            vis_options=gs.options.VisOptions(background_color=(0.10,0.13,0.18), ambient_light=(.45,.45,.45)),
            show_viewer=False,
        )
        self.scene.add_entity(gs.morphs.Plane(), surface=gs.surfaces.Default(color=(.22,.25,.30,1)))
        self.scene.add_entity(gs.morphs.Box(size=(1.1,.8,.04), pos=(.38,0,.18), fixed=True),
                              surface=gs.surfaces.Default(color=(.40,.30,.20,1)), material=gs.materials.Rigid(friction=.8))
        for x in [-.07,.83]:
            for y in [-.30,.30]:
                self.scene.add_entity(gs.morphs.Box(size=(.035,.035,.16),pos=(x,y,.08),fixed=True),
                                      surface=gs.surfaces.Default(color=(.18,.21,.25,1)))
        self.layouts = json.loads((ROOT/'dexterous-hands/src/assets/sensors/wuji/low/probes_hand.json').read_text())
        self.robot = self.scene.add_entity(gs.morphs.MJCF(file=str(build_model()), pos=(0,0,.2)),
                            material=gs.materials.Rigid(friction=.8))
        self.cube = self.scene.add_entity(gs.morphs.Box(size=(.045,.045,.045),pos=(.55,-.04,.225)),
                            material=gs.materials.Rigid(rho=500,friction=.8),
                            surface=gs.surfaces.Default(color=(.95,.36,.10,1)))
        self.sensors = {}
        for entry in self.layouts:
            probes = entry['probes']
            positions = [[[p['pos'][c] for c in range(3)] for p in row] for row in probes]
            radii = [[p['radius'] for p in row] for row in probes]
            link = self.robot.get_link(entry['link_name'])
            self.sensors[entry['link_name']] = self.scene.add_sensor(gs.sensors.KinematicTaxel(
                entity_idx=self.robot.idx, link_idx_local=link.idx_local,
                probe_local_pos=positions, probe_radius=radii, history_length=5,
                normal_stiffness=500., normal_damping=1., normal_exponent=1.2,
                shear_scalar=2., twist_scalar=2., draw_debug=False,
            ))
        self.camera = self.scene.add_camera(res=(1100,800),pos=(1.5,-1.6,1.25),lookat=(.35,0,.45),fov=45,GUI=False)
        self.close_camera = self.scene.add_camera(res=(640,640),pos=(.88,-.62,.68),lookat=(.55,-.08,.48),fov=40,GUI=False)
        self.scene.build(n_envs=1)
        self.arm_names = [f'joint{i}' for i in range(1,8)]
        self.hand_names = [j.name for j in self.robot.joints if j.name.startswith('r_') and j.n_dofs]
        self.names = self.arm_names + self.hand_names
        self.indices = [self.robot.get_joint(n).dofs_idx_local[0] for n in self.names]
        assert len(self.names)==27 and self.robot.n_dofs==27
        self.arm_indices = self.indices[:7]
        self.wrist = self.robot.get_link('r_wrist')
        lo,hi = self.robot.get_dofs_limit(self.indices)
        self.lower,self.upper=self.array(lo).reshape(-1),self.array(hi).reshape(-1)
        self.offset=(self.lower+self.upper)/2
        self.scale=(self.upper-self.lower)/2
        self.robot.set_dofs_kp([4500,4500,3500,3500,2000,2000,2000],self.arm_indices)
        self.robot.set_dofs_kv([450,450,350,350,200,200,200],self.arm_indices)
        # Hand kp/kv and torque limits are imported unchanged from the vendor MJCF.
        home=np.array([.45882,.36950,-.26819,-2.06344,-1.35166,1.50485,-2.29429]+[0.0]*20)
        self.robot.set_dofs_position(home,self.indices,zero_velocity=True)
        # Local +Y is the palm normal, local -Z points along the fingers.
        # Map them to world -Z and -Y: palm down, fingers parallel to the table.
        q,error=self.robot.inverse_kinematics(self.wrist,pos=(.55,0,.50),quat=(2**-.5,-2**-.5,0,0),
                    dofs_idx_local=self.arm_indices,return_error=True,seed=42,max_solver_iters=100)
        initial_error = self.array(error).reshape(-1)
        if np.linalg.norm(initial_error[:3]) > .005 or np.linalg.norm(initial_error[3:]) > .03:
            raise RuntimeError(f'Initial IK did not converge: {error}')
        self.home=self.array(q).reshape(-1)[self.indices]
        self.home[7:]=0
        self.closed=False
        self.episode=-1
        self.image_bytes={}
        self.reset()

    def array(self,x):
        return x.detach().cpu().numpy() if hasattr(x,'detach') else self.np.asarray(x)

    def values(self,x):
        return self.array(x).reshape(-1).astype(float).tolist()

    def render(self):
        runtime=HERE/'runtime';runtime.mkdir(exist_ok=True)
        for name, camera in [('scene',self.camera),('hand',self.close_camera)]:
            rgb=camera.render(rgb=True)[0]
            rgb=self.array(rgb)
            if rgb.ndim==4:rgb=rgb[0]
            img=self.Image.fromarray(rgb.astype(self.np.uint8))
            buffer=io.BytesIO();img.save(buffer,format='JPEG',quality=90)
            self.image_bytes[name]=buffer.getvalue()
            # Only the latest screenshot is kept. No episode frames or dataset.
            img.save(runtime/f'{name}.png')

    def reset(self,seed=42):
        if type(seed) is not int:raise ValueError('seed must be an integer')
        self.scene.reset()
        self.robot.set_dofs_position(self.home,self.indices,zero_velocity=True)
        self.target=self.home.copy()
        self.robot.control_dofs_position(self.target,self.indices)
        self.actions=((self.target-self.offset)/self.scale).tolist()
        self.episode+=1;self.steps=0;self.fault=None
        self.render()
        return self.observe()

    def describe(self):
        return {'robot':'Panda 7-DOF + Wuji 20-DOF','joint_order':self.names,'arm_joints':self.arm_names,
            'hand_joints':self.hand_names,'control_dt_s':.025,'physics_dt_s':.005,
            'action_formula':'target_rad = joint_midpoint + normalized_action * half_joint_range',
            'joints':{n:{'index':i,'lower_rad':float(self.lower[i]),'upper_rad':float(self.upper[i]),
                         'offset_rad':float(self.offset[i]),'scale_rad':float(self.scale[i])} for i,n in enumerate(self.names)},
            'tactile_taxels':sum(s.n_probes for s in self.sensors.values()),
            'tactile_frame':'attachment link local; spring-damper estimate, includes possible self-contact',
            'tactile_layout':self.layouts,'mount':'one MJCF chain, r_wrist fixed beneath Panda link7; no free-floating wrist',
            'data_collection':False,'table_top_z_m':.20,'reset':'fixed initial scene; seed accepted for interface consistency; no randomization'}

    def status(self):
        return {'episode':self.episode,'episode_step':self.steps,'sim_time_s':round(self.steps*.025,6),
                'paused':True,'terminated':bool(self.fault),'fault':self.fault,'data_collection':False}

    def observe(self,full=False):
        tactile={}
        for name,sensor in self.sensors.items():
            r=sensor.read()
            f=self.array(r.force).reshape(-1,sensor.n_probes,3)[0]
            t=self.array(r.torque).reshape(-1,sensor.n_probes,3)[0]
            if not self.np.isfinite(f).all() or not self.np.isfinite(t).all():raise RuntimeError('Nonfinite tactile data')
            norms=self.np.linalg.norm(f,axis=-1)
            tactile[name]={'taxels':sensor.n_probes,'active_taxels':int((norms>1e-5).sum()),'peak_force_N':float(norms.max()),
                           'force_sum_N':f.sum(0).tolist(),'torque_sum_Nm':t.sum(0).tolist()}
            if full:tactile[name]['force_torque']=self.np.concatenate([f,t],axis=-1).tolist()
        result={**self.status(),'joint_positions_rad':dict(zip(self.names,self.values(self.robot.get_dofs_position(self.indices)))),
                'joint_velocities_rad_s':dict(zip(self.names,self.values(self.robot.get_dofs_velocity(self.indices)))),
                'last_actions':dict(zip(self.names,self.actions)),
                'wrist_position_m':self.values(self.wrist.get_pos()),'wrist_quaternion_wxyz':self.values(self.wrist.get_quat()),
                'rgb_path':str(HERE/'runtime/scene.png'),'hand_rgb_path':str(HERE/'runtime/hand.png'),'tactile':tactile}
        json.dumps(result,allow_nan=False)
        return result

    def step(self,payload):
        if self.fault:raise ValueError('Simulation fault: reset first')
        for key,expected in [('expected_episode',self.episode),('expected_step',self.steps)]:
            if key in payload and (type(payload[key]) is not int or payload[key]!=expected):raise ValueError('Stale state')
        actions,count=validate_step(payload,self.names,self.actions)
        self.actions=actions
        self.target=self.offset+self.np.array(actions)*self.scale
        self.robot.control_dofs_position(self.target,self.indices)
        for _ in range(count):
            for __ in range(5):self.scene.step()
            self.steps+=1
            q=self.array(self.robot.get_dofs_position(self.indices))
            if not self.np.isfinite(q).all():
                self.fault='Nonfinite joint state';break
        self.render()
        return self.observe()

    def move(self,payload):
        delta=self.np.array(payload.get('delta',[0,0,0]),dtype=float)
        if delta.shape!=(3,) or not self.np.isfinite(delta).all() or self.np.linalg.norm(delta)>.05:
            raise ValueError('delta must be a finite xyz vector, length <= 0.05 m')
        count=payload.get('steps',10)
        if type(count) is not int or not 1<=count<=10:raise ValueError('steps must be 1..10')
        position=self.array(self.wrist.get_pos()).reshape(3)+delta
        if not(-.1<=position[0]<=.85 and -.4<=position[1]<=.4 and .26<=position[2]<=.95):
            raise ValueError('Requested wrist target outside the interactive workspace')
        q,error=self.robot.inverse_kinematics(self.wrist,pos=position,quat=self.wrist.get_quat(),
                    dofs_idx_local=self.arm_indices,return_error=True,max_solver_iters=60,seed=42)
        error=self.array(error).reshape(-1)
        if self.np.linalg.norm(error[:3])>.003 or self.np.linalg.norm(error[3:])>.03:raise ValueError('IK could not reach requested pose')
        desired=self.array(q).reshape(-1)[self.indices]
        normalized=((desired-self.offset)/self.scale).clip(-1,1)
        return self.step({'actions':{n:float(normalized[i]) for i,n in enumerate(self.arm_names)},'steps':count})

    def hand(self,payload):
        closure=payload.get('closure',0)
        if isinstance(closure,bool) or not isinstance(closure,(int,float)) or not self.np.isfinite(closure) or not 0<=closure<=1:
            raise ValueError('closure must be in [0,1]')
        desired=self.target.copy()
        for i,n in enumerate(self.names[7:],7):
            desired[i]=closure*(.35 if n=='r_thumb_cmc_abd' else 0.0 if n.endswith('abd') else 1.25 if n.endswith('pip') else .85)
        a=(desired-self.offset)/self.scale
        return self.step({'actions':{n:float(a[i]) for i,n in enumerate(self.names) if i>=7},'steps':payload.get('steps',10)})


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--port',type=int,default=8766);args=parser.parse_args()
    http=HTTPServer(('127.0.0.1',args.port),BaseHTTPRequestHandler)
    sim=Sandbox()
    class Handler(BaseHTTPRequestHandler):
        def respond(self,body,mime='application/json',code=200):
            self.send_response(code);self.send_header('Content-Type',mime);self.send_header('Content-Length',str(len(body)))
            self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(body)
        def do_GET(self):
            path=self.path.split('?')[0]
            if path=='/':self.respond((HERE/'index.html').read_bytes(),'text/html; charset=utf-8')
            elif path in ['/scene.jpg','/hand.jpg']:self.respond(sim.image_bytes[path[1:-4]],'image/jpeg')
            else:self.respond(b'Not found','text/plain',404)
        def do_POST(self):
            try:
                size=int(self.headers.get('Content-Length','0'))
                if not 0<=size<=1000000:raise ValueError('Invalid request size')
                p=json.loads(self.rfile.read(size) or b'{}')
                if not isinstance(p,dict):raise ValueError('Payload must be an object')
                command=self.path.strip('/')
                if command=='describe':result=sim.describe()
                elif command=='reset':result=sim.reset(p.get('seed',42))
                elif command=='observe':result=sim.observe(p.get('full',False))
                elif command=='status':result=sim.status()
                elif command in ('v8_start','v8_skill','v8_perceive','v8_evaluate'):
                    from v8_adapter import V8Adapter
                    if not hasattr(sim,'v8'):sim.v8=V8Adapter(sim)
                    result=getattr(sim.v8,command.removeprefix('v8_'))(p)
                elif command=='step':result=sim.step(p)
                elif command=='move':result=sim.move(p)
                elif command=='hand':result=sim.hand(p)
                elif command=='close':sim.closed=True;result={'closed':True}
                else:raise ValueError('Unknown command')
                self.respond(json.dumps({'ok':True,'result':result},allow_nan=False).encode())
            except ValueError as e:self.respond(json.dumps({'ok':False,'error':str(e)}).encode(),code=400)
            except Exception as e:
                import traceback;traceback.print_exc()
                self.respond(json.dumps({'ok':False,'error':str(e)}).encode(),code=500)
        def log_message(self,*args):pass
    http.RequestHandlerClass=Handler
    print(f'ARM_HAND_READY http://127.0.0.1:{args.port}',flush=True)
    try:
        while not sim.closed:http.handle_request()
    finally:http.server_close()


if __name__=='__main__':main()
