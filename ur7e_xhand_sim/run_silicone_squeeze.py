"""Independent Astra API driver for the FEM squeeze/release bridge."""
import argparse,json,os,urllib.request
from pathlib import Path
from run_s8v4 import Policy,Config,SkillAct,build_observation

INSTRUCTIONS='''You control the SIMULATED UR7e/XHand through bounded bridge skills.
Task: firmly squeeze the PINK silicone soft cube, hold briefly, then OPEN the fingers,
retreat upward and observe elastic recovery. The target is the pink cube near the cyan
sphere, NOT the orange wooden cube or purple cube. No hardware exists in this trial.
Return SkillAct JSON only. Follow allowed_skills in the observation. Typical sequence:
locate -> approach -> align -> close -> hold -> release -> retreat -> done/none.
For locate, give a tight ROI [x1,y1,x2,y2] of the PINK cube in CURRENT scene_front.
All offset_xyz_m must be [0,0,0]. Non-locate ROI must be [0,0,0,0].
Current images are front, left and wrist. Examine them after every skill.
Approach and align use live RGB-D registration plus a prior rigid demonstration's
palm offset/preshape. This is an experimental soft-material trial, not full S8V4.
close gradually increases the index target to 0.85 rad with the existing PD/effort
limits. There are NO volume-ratio, strain, or extent threshold stops. Mesh diagnostics
are recorded, not used to stop closure. No object attachment, teleportation or force invention.
Experimental_FEM_diagnostics are privileged simulation mesh strain, not tactile.
XHand tactile DOES NOT measure FEM contacts, so zero tactile does not mean no
soft contact. No lifting skill is available; squeeze and release on the table.
If closure missed, do not claim a successful squeeze. Still release and retreat
when possible. Even if compression is small, complete release and retreat.
Release first unloads the index, then opens ALL hand joints to the neutral open
targets while holding the palm still and waits before retreat. Check the cube is
actually free, since the old index-only release could carry it during retreat.
Only done after retreated. Abort if the bridge reports a numerical/collision fault.
Describe whether images and mesh strain support actual compression and recovery.
'''

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--label',required=True)
    parser.add_argument('--resume',action='store_true',help='Observe and continue an active bridge trial without resetting physics')
    parser.add_argument('--bridge-timeout',type=float,default=3600,help='Wall-clock seconds allowed for a synchronous physics skill')
    args=parser.parse_args()
    if not args.label or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in args.label):parser.error('Invalid label')
    for name in ('HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy'):os.environ.pop(name,None)
    os.environ['NO_PROXY']=os.environ['no_proxy']='*'
    out=Path(__file__).resolve().parents[1]/'outputs/silicone_agent'/args.label;out.mkdir(parents=True,exist_ok=False)
    (out/'instructions.txt').write_text(INSTRUCTIONS)
    def call(n,p={}):
        req=urllib.request.Request('http://127.0.0.1:8767/'+n,json.dumps(p).encode(),{'Content-Type':'application/json'})
        with urllib.request.urlopen(req,timeout=args.bridge_timeout) as r:d=json.load(r)
        if not d['ok']:raise RuntimeError(d)
        return d['result']
    policy=None;current=None
    try:
        policy=Policy(Config(),out,task='Firmly squeeze pink silicone block and release',model='gpt-6-astra',effort='medium',timeout=180,plan_steps=1,
            action_model=SkillAct,instruction_text=INSTRUCTIONS,codex_bin='/home/sai/zsq/bench2dex-work/codex/node_modules/.bin/codex',codex_home=str(Path.home()/'.codex'))
        print('AGENT_READY',policy.thread_id,flush=True)
        current=call('silicone_observe') if args.resume else call('silicone_start',{'label':args.label})
        (out/'bridge_run.json').write_text(json.dumps({'thread_id':policy.thread_id,'run_dir':current['run_dir']},indent=2))
        for i in range(12):
            obs=build_observation(current);print('MODEL_START',i,current['phase'],flush=True)
            a,meta=policy.decide(obs)
            (out/f'model_{i:03d}.json').write_text(json.dumps({'action':a.model_dump(),'metadata':meta},indent=2))
            print('ACTION',a.model_dump_json(),flush=True)
            reply=call('silicone_act',a.model_dump());policy.feedback(obs,a,reply['receipt']);current=reply['policy_observation']
            print('RESULT',json.dumps({'receipt':reply['receipt'],'phase':current['phase'],'FEM':current['experimental_FEM_diagnostics']}),flush=True)
            if reply['finished']:break
        else:raise RuntimeError('Agent action budget exhausted')
        (out/'final.json').write_text(json.dumps(current,indent=2));print('COMPLETE',current['run_dir'],flush=True)
    except BaseException as exc:
        (out/'error.json').write_text(json.dumps({'type':type(exc).__name__,'error':str(exc)},indent=2))
        if current is not None:
            try:
                current=call('silicone_observe')
                call('silicone_act',{'observation_id':current['observation_id'],'status':'abort','skill':'none','explanation':'Driver exception: '+str(exc)[:500],'offset_xyz_m':[0,0,0],'object_roi_xyxy':[0,0,0,0]})
            except Exception:pass
        raise
    finally:
        if policy:policy.close()
if __name__=='__main__':main()
