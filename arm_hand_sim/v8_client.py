"""Call one v8 skill and print compact live feedback."""
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'astra_bridge'))
from client import call

if __name__=='__main__':
    command=sys.argv[1]
    payload=json.loads(sys.argv[2]) if len(sys.argv)>2 else {}
    result=call('v8_'+command,payload,'http://127.0.0.1:8766')
    if 'observation' in result:
        o=result['observation']
        result['observation']={k:v for k,v in o.items() if k not in ('joint_velocities_rad_s','last_actions','tactile')}
        result['active_touch']={k:{'active':v['active_taxels'],'peak_N':round(v['peak_force_N'],4)} for k,v in o['tactile'].items() if v['active_taxels']}
    print(json.dumps(result,ensure_ascii=False))
