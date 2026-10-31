"""Frozen four-case comparison: independent Astra decisions via the bridge."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

ROOT=Path(__file__).resolve().parents[1]

def write(path,value):path.write_text(json.dumps(value,indent=2))
def call(name):
    req=urllib.request.Request('http://127.0.0.1:8767/'+name,b'{}',{'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=5) as r:return json.load(r)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--label',required=True);parser.add_argument('--reference',type=Path,required=True)
    args=parser.parse_args();out=ROOT/'outputs/s8v4_four_tests'/args.label;out.mkdir(parents=True,exist_ok=False)
    cases=[('original',[0,0,0]),('small',[.03,.02,0]),('large',[.15,.10,0]),('clutter',[0,0,0])]
    sources=list((ROOT/'ur7e_xhand_sim').glob('*.py'))+[Path('/home/sai/zsq/vlm-policy/real_variants/prior.py'),Path('/home/sai/zsq/vlm-policy/tactile_policy/runtime.py'),Path('/home/sai/zsq/vlm-policy/codex_policy.py')]
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    source_dir=out/'sources';source_dir.mkdir()
    for i,p in enumerate(sources):(source_dir/f'{i:02d}_{p.name}').write_bytes(p.read_bytes())
    write(out/'protocol.json',{'cases':cases,'reference':str(args.reference.resolve()),
        'reference_sha256':hashlib.sha256(args.reference.read_bytes()).hexdigest(),'source_hashes':hashes,
        'model':'gpt-6-astra','effort':'medium','live_views':['scene_front','scene_left','wrist'],
        'demonstration':'same 7 keyframes/21 images and recorded close/preload in every case',
        'repeats_per_case':1,'max_model_turns':12,'no_parameter_tuning_between_cases':True,
        'pass':'target cube bottom >=5 cm above table for final 3 s hold and no execution fault',
        'scope':'UR/XHand S8V4 port; GenDex CMap-CVAE bounded residual scoring, not native Allegro optimizer',
        'scenario_truth':'initialization and independent scoring only; never sent to Astra'})
    completed=[]
    for case,delta in cases:
        for p,h in hashes.items():
            if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h:raise RuntimeError('Frozen source changed: '+p)
        label=args.label+'_'+case;write(out/'progress.json',{'running':case,'completed':completed})
        print('CASE_START',case,flush=True)
        try:call('status')
        except Exception:pass
        else:raise RuntimeError('Port 8767 already serves a scene; do not replace an unknown active run')
        server_log=(out/f'{case}_server.log').open('w')
        server=subprocess.Popen([sys.executable,'-u','ur7e_xhand_sim/server.py','--scenario',case,'--output',str(out/f'{case}_runtime')],cwd=ROOT,stdout=server_log,stderr=subprocess.STDOUT,start_new_session=True)
        result={'case':case,'label':label,'translation_m':delta,'server_pid':server.pid}
        try:
            deadline=time.monotonic()+180
            while True:
                if server.poll() is not None:raise RuntimeError('Server exited before readiness')
                try:
                    if call('status')['ok']:break
                except Exception:pass
                if time.monotonic()>deadline:raise TimeoutError('Server readiness timeout')
                time.sleep(1)
            with (out/f'{case}_agent.log').open('w') as log:
                client=subprocess.run([sys.executable,'-u','ur7e_xhand_sim/run_s8v4.py','--label',label,'--reference',str(args.reference.resolve())],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=1800)
            result['runner_exit_code']=client.returncode
            score=ROOT/'outputs/s8v4_genesis'/label/'score.json'
            if score.exists():result['score']=json.loads(score.read_text())
            else:result['error']='No completed physical score; inspect agent log'
            result['model_calls']=len(list((ROOT/'outputs/s8v4_agent'/label).glob('model_*.json')))
        except Exception as exc:result['error']=str(exc)
        finally:
            completed.append(result);write(out/'results.json',completed)
            print('CASE_RESULT',json.dumps(result),flush=True)
            if case!='clutter' or 'error' in result:
                if server.poll() is None:
                    try:call('close');server.wait(timeout=30)
                    except Exception:server.terminate();server.wait(timeout=30)
                server_log.close()
    write(out/'progress.json',{'running':None,'completed':completed,'batch_complete':True})
    print('BATCH_COMPLETE',str(out),flush=True)

if __name__=='__main__':main()
