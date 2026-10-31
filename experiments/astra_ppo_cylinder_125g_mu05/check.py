"""Read-only local preflight. Never starts Astra or sends an inference request."""
import hashlib,json,socket
from pathlib import Path
from success_context import SuccessContext
from stable_baselines3 import PPO
r=Path(__file__).resolve().parent;c=json.loads((r/'config.json').read_text())
for name in ['checkpoint','trajectory','demonstration_reference']:
 assert hashlib.sha256(Path(c[name]).read_bytes()).hexdigest()==c[name+'_sha256'],name
ctx=SuccessContext(c['demonstration_reference']);content=ctx.content()
assert len(ctx.frames)==7 and sum(x['type']=='input_image' for x in content)==21
assert ctx.reference['object_shape']=='cylinder'
assert ctx.frames[-1]['time_s']==1808/60
model=PPO.load(c['checkpoint'],device='cpu');assert model.policy.features_extractor.use_tactile and model.seed==1
with socket.socket() as s:s.bind(('127.0.0.1',8775))
print('PASS: cylinder demonstration / 7 frames / 21 images / correct 60Hz timestamps / frozen seed1 tactile PPO / port8775 available; no Astra request.')
