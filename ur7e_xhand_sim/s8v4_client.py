"""Inspect/control the same S8V4 skill adapter from an interactive VLM session."""
import argparse,json,urllib.request
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('command',choices=['start','observe','act','score']);p.add_argument('--json',default='{}');p.add_argument('--url',default='http://127.0.0.1:8767');args=p.parse_args()
request=urllib.request.Request(args.url+'/s8v4_'+args.command,json.dumps(json.loads(args.json)).encode(),{'Content-Type':'application/json'})
with urllib.request.urlopen(request,timeout=300) as response:data=json.load(response)
Path('/tmp/s8v4_current.json').write_text(json.dumps(data,indent=2))
if not data['ok']:raise RuntimeError(data['error'])
r=data['result'];o=r.get('policy_observation',r)
if 'observation' in o:
 raw=o['observation'];o={k:v for k,v in o.items() if k!='observation'}
 o['hand_position']=raw['wrist_position_m'];o['sim_time_s']=raw['sim_time_s']
 if 'receipt' in r:o['receipt']=r['receipt']
if o.get('proposals'):o['proposals']={k:o['proposals'][k] for k in ['candidates','score_meaning']}
print(json.dumps(o,indent=2))
