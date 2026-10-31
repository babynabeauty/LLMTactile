"""Dispatch the deterministic teacher through the simulation bridge."""
import argparse,json,urllib.request

def main():
    p=argparse.ArgumentParser();p.add_argument('--label',required=True);p.add_argument('--options',default='{}');p.add_argument('--url',default='http://127.0.0.1:8767');args=p.parse_args()
    def call(name,payload):
        request=urllib.request.Request(args.url+'/'+name,json.dumps(payload).encode(),{'Content-Type':'application/json'})
        with urllib.request.urlopen(request,timeout=180) as r:value=json.load(r)
        if not value['ok']:raise RuntimeError(value.get('error'))
        return value['result']
    state=call('rule_demo_start',{'label':args.label,'options':json.loads(args.options)});phase=None
    while not state['finished']:
        if state['phase']!=phase:
            phase=state['phase'];print(json.dumps({k:state[k] for k in ('phase','steps','run_dir')}),flush=True)
        state=call('rule_demo_advance',{})
    print(json.dumps(state,indent=2),flush=True)
    if not state.get('score',{}).get('success',False):
        raise SystemExit(1)

if __name__=='__main__':main()
