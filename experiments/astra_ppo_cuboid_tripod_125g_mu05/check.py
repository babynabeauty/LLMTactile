"""Read-only local preflight. Never starts Astra or sends an inference request."""
import argparse,hashlib,json,socket
from pathlib import Path
from success_context import SuccessContext
parser=argparse.ArgumentParser()
parser.add_argument('--mode',choices=['ppo','no-ppo'],default='ppo')
args=parser.parse_args()
r=Path(__file__).resolve().parent;c=json.loads((r/'config.json').read_text())
for name in (['checkpoint'] if args.mode=='ppo' else [])+['trajectory','demonstration_reference']:
 assert hashlib.sha256(Path(c[name]).read_bytes()).hexdigest()==c[name+'_sha256'],name
ctx=SuccessContext(c['demonstration_reference']);content=ctx.content()
assert len(ctx.frames)==7 and sum(x['type']=='input_image' for x in content)==21
assert ctx.reference['object_shape']=='cuboid'
assert ctx.reference['contact_fingers']==['thumb','index','middle']
assert ctx.reference['execution_mode']=='recorded_tripod_closure_v1'
assert ctx.frames[-1]['time_s']==ctx.reference['validation']['control_steps']/60
assert Path(ctx.reference['source_run']).parent.name=='direct_teacher_cuboid_tripod_clearance_20261031_v2'
assert (r/'demo/control.jsonl').resolve()==Path(c['trajectory']).resolve()
if args.mode=='ppo':
 from stable_baselines3 import PPO
 model=PPO.load(c['checkpoint'],device='cpu');assert model.policy.features_extractor.use_tactile and model.seed==1
with socket.socket() as s:s.bind(('127.0.0.1',8775))
print(f'PASS: raised tripod cuboid demo / 7 frames / 21 images / 60Hz / mode={args.mode} / port8775 available; no Astra request.')
