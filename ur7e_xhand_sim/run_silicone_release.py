"""Agent-authorized release of the existing finite FEM state after a protective stop."""
import json,os,time,urllib.request,base64
from pathlib import Path
import numpy as np
from run_s8v4 import Policy,Config,SkillAct,build_observation
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/silicone_agent/pink_firm_release_20261024_recovery'
OUT.mkdir(parents=True,exist_ok=False)
RUN=ROOT/'outputs/silicone_squeeze/pink_firm_release_20261024_v1'
for n in ('HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy'):os.environ.pop(n,None)
os.environ['NO_PROXY']=os.environ['no_proxy']='*'
INSTRUCTIONS='''You control only release/recovery in the existing SIMULATION, not hardware.
The previous independent Astra trial located and squeezed the pink FEM block. A
protection stopped closure when one element's volume ratio reached 0.112 (positive,
not inverted). Mesh positions are finite. The robot is currently stopped while
pinching. Task: open, retreat, allow recovery; never squeeze harder or lift a held object.
Use skill release first, then retreat, then hold, then done/none. All offsets/ROI zero.
Release slowly opens only index to its pregrasp target while holding arm targets.
Retreat is a bounded 12 cm upward motion after opening; hold waits 3 simulated seconds.
The host uses the existing bridge step/move APIs; no object reset or teleportation.
Current front,left,wrist images and FEM diagnostics are provided. FEM metrics are not
hardware tactile. This is not full S8V4. Report honestly whether the soft block recovers.
Use abort/none if finite-state or collision checks fail. Return SkillAct JSON only.
'''
(OUT/'instructions.txt').write_text(INSTRUCTIONS)
def call(n,p={}):
 req=urllib.request.Request('http://127.0.0.1:8767/'+n,json.dumps(p).encode(),{'Content-Type':'application/json'})
 with urllib.request.urlopen(req,timeout=300) as r:d=json.load(r)
 if not d['ok']:raise RuntimeError(d)
 return d['result']
phase='stopped';index=0
record=(RUN/'recovery.jsonl').open('w',buffering=1)
frames=RUN/'frames';frames.mkdir(exist_ok=True)
def snapshot_frames(step):
 for n in ('scene_front','scene_left','wrist'):
  with urllib.request.urlopen('http://127.0.0.1:8767/'+n+'.jpg') as r:data=r.read()
  (frames/f'{step:06d}_{n}.jpg').write_bytes(data)
def log_state(action,ob):
 e=call('evaluate');diag=call('silicone_observe')['experimental_FEM_diagnostics']
 diag['phase']='recovery_'+phase
 record.write(json.dumps({'action':action,'observation':ob,'evaluation':e,'deformation':diag})+'\n')
 if not e['silicone']['finite']:raise RuntimeError('Nonfinite FEM during release')
 if any(c['entity_a']==c['entity_b']=='ur7e_xhand' or 'table' in (c['entity_a'],c['entity_b']) for c in e['solver_contacts']):raise RuntimeError('Robot self/table contact')
 snapshot_frames(ob['episode_step'])
def observe():
 o=call('observe');d=call('silicone_observe')['experimental_FEM_diagnostics']
 folder=OUT/f'view_{index:03d}';folder.mkdir(exist_ok=True)
 for n in ('scene_front','scene_left','wrist'):
  with urllib.request.urlopen('http://127.0.0.1:8767/'+n+'.jpg') as r:(folder/(n+'.jpg')).write_bytes(r.read())
 return {'observation_id':f"{o['episode']}:{o['episode_step']}:{index}",'phase':phase,'observation':o,
  'image_dir':str(folder),'run_dir':str(RUN),'fault':None,'experimental_FEM_diagnostics':d,
  'allowed_skills':{'stopped':['release'],'released':['retreat'],'retreated':['hold'],'recovered':[]}[phase]}
policy=None
try:
 policy=Policy(Config(),OUT,task='Release pinched silicone after protective stop',model='gpt-6-astra',effort='medium',timeout=180,plan_steps=1,
  action_model=SkillAct,instruction_text=INSTRUCTIONS,codex_bin='/home/sai/zsq/bench2dex-work/codex/node_modules/.bin/codex',codex_home=str(Path.home()/'.codex'))
 print('AGENT_READY',policy.thread_id,flush=True)
 for index in range(6):
  current=observe();obs=build_observation(current);a,meta=policy.decide(obs)
  (OUT/f'model_{index:03d}.json').write_text(json.dumps({'action':a.model_dump(),'metadata':meta},indent=2))
  print('ACTION',a.model_dump_json(),flush=True)
  if a.status!='continue':
   if a.status=='done' and phase!='recovered':raise ValueError('Premature done')
   break
  if a.skill not in current['allowed_skills']:raise ValueError('Invalid recovery skill')
  if a.skill=='release':
   desc=call('describe');name=desc['hand_joints'][4];initial=current['observation']['last_actions'][name]
   j=desc['joints'][name];goal=(0.-j['offset_rad'])/j['scale_rad']
   for k in range(1,21):
    payload={'actions':{name:initial+(goal-initial)*k/20},'steps':2}
    o=call('step',payload);log_state(payload,o)
   phase='released'
  elif a.skill=='retreat':
   for k in range(6):
    payload={'delta':[0,0,.02],'steps':10};o=call('move',payload);log_state(payload,o)
   phase='retreated'
  elif a.skill=='hold':
   for k in range(12):
    o=call('step',{'steps':10});log_state({'steps':10},o)
   phase='recovered'
  receipt={'skill':a.skill,'phase':phase,'FEM':call('silicone_observe')['experimental_FEM_diagnostics']}
  policy.feedback(obs,a,receipt);print('RESULT',json.dumps(receipt),flush=True)
 final=observe();(OUT/'final.json').write_text(json.dumps(final,indent=2));(RUN/'recovery_final.json').write_text(json.dumps(final,indent=2))
 print('COMPLETE',phase,flush=True)
finally:
 record.close()
 if policy:policy.close()
