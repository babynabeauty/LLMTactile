"""Local client for the Panda + Wuji sandbox (port 8766)."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'astra_bridge'))
from client import call


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['describe','status','observe','reset','step','move','hand','close'])
    p.add_argument('--json',default='{}');p.add_argument('--url',default='http://127.0.0.1:8766')
    p.add_argument('--compact',action='store_true')
    a=p.parse_args()
    try:
        result=call(a.command,json.load(sys.stdin) if a.json=='-' else json.loads(a.json),a.url)
        if a.compact and 'tactile' in result:
            result={k:v for k,v in result.items() if k not in ['joint_velocities_rad_s','last_actions','tactile']}
        print(json.dumps(result,ensure_ascii=False,allow_nan=False))
    except (ValueError,RuntimeError,OSError) as e:
        print(str(e),file=sys.stderr);sys.exit(1)


if __name__=='__main__':main()
