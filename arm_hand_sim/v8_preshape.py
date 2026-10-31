"""Geometric Wuji preshape from visible RGB-D bounds, not a learned/demo prior."""
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent

def solve():
    meta=json.loads((HERE/'runtime/v8_run.json').read_text())
    bounds=np.array(meta['perception']['surface_bounds_m'])
    center=bounds.mean(axis=0)
    root=ET.parse(ROOT/'dexterous-hands/src/assets/hands/wuji/mjcf/right.xml').getroot()
    body=root.find('worldbody/body')
    joints=list(body.iter('joint'))
    names=[j.get('name') for j in joints]
    limits=np.array([np.fromstring(j.get('range'),sep=' ') for j in joints])
    limits[names.index('r_thumb_ip')]=[-.35,1.30]
    limits[names.index('r_thumb_mcp')]=[-.40,1.30]
    for i,name in enumerate(names):
        if any(f in name for f in ('index','middle')):
            if name.endswith('mcp_flex'):limits[i]=[0,1.25]
            elif name.endswith('pip'):limits[i]=[.05,1.5]
            elif name.endswith('dip'):limits[i]=[0,1.3]
    layout=json.loads((ROOT/'dexterous-hands/src/assets/sensors/wuji/low/probes_hand.json').read_text())
    pads={}
    for entry in layout:
        if 'tip_sensor' in entry['link_name']:
            probes=[p for row in entry['probes'] for p in row]
            pads[entry['link_name']]=(np.mean([p['pos'] for p in probes],axis=0),np.mean([p['normal'] for p in probes],axis=0))
    palm_R=Rotation.from_quat([-2**-.5,0,0,2**-.5]).as_matrix()
    selected=[i for i,n in enumerate(names) if any(f in n for f in ('thumb','index','middle'))]
    def fk(q,wrist):
        out={}
        def walk(node,T,is_root=False):
            T=T.copy()
            if not is_root:
                local=np.eye(4)
                quat=np.fromstring(node.get('quat','1 0 0 0'),sep=' ')
                local[:3,:3]=Rotation.from_quat(quat[[1,2,3,0]]).as_matrix()
                local[:3,3]=np.fromstring(node.get('pos','0 0 0'),sep=' ')
                j=node.find('joint')
                if j is not None:
                    axis=np.fromstring(j.get('axis','1 0 0'),sep=' ')
                    local[:3,:3]=local[:3,:3]@Rotation.from_rotvec(axis*q[names.index(j.get('name'))]).as_matrix()
                T=T@local
            if node.get('name') in pads:
                pos,normal=pads[node.get('name')]
                out[node.get('name')]=(T[:3,:3]@pos+T[:3,3],T[:3,:3]@normal)
            for child in node.findall('body'):walk(child,T)
        T=np.eye(4);T[:3,:3]=palm_R;T[:3,3]=wrist
        walk(body,T,True)
        return out
    # Opposed thumb and index/middle pads on the two Y faces.
    targets={
      'r_thumb_tip_sensor_frame':(center+np.array([.009,.018,0]),np.array([-.4,-.9,0])),
      'r_index_finger_tip_sensor_frame':(center+np.array([.010,-.018,0]),np.array([0,1,0])),
      'r_middle_finger_tip_sensor_frame':(center+np.array([-.010,-.018,0]),np.array([0,1,0]))}
    wrist0=center+[-.025,.11,.075]
    def residual(x):
        q=np.zeros(20);q[selected]=x[3:]
        found=fk(q,x[:3]);r=[]
        for name,(pos,normal) in targets.items():
            a,b=found[name]
            r.extend((a-pos)*100)
            r.extend((b/np.linalg.norm(b)-normal)*.3)
        r.extend((x[:3]-wrist0)*2)
        r.extend(x[3:]*.005)
        return r
    low=np.r_[center+[-.09,.04,.055],limits[selected,0]+.03]
    high=np.r_[center+[.05,.16,.13],limits[selected,1]-.03]
    seedq=np.array([.4,-.5,.7,.7]+[.7,0,1.0,.7]*2)
    sol=least_squares(residual,np.r_[wrist0,seedq],bounds=(low,high),max_nfev=350)
    q=np.zeros(20);q[selected]=sol.x[3:]
    opened=q.copy()
    for i,name in enumerate(names):
        if 'thumb' in name:
            if name.endswith(('mcp','ip')):opened[i]-=.45
        elif i in selected:
            if 'mcp_flex' in name:opened[i]-=.4
            if name.endswith(('pip','dip')):opened[i]-=.3
        else:
            # Keep unused fingers clear of the table.
            if 'mcp_flex' in name:opened[i]=q[i]=-.15
    opened=np.clip(opened,limits[:,0]+.02,limits[:,1]-.02)
    data={'source':'RGB-D observed box + public Wuji FK; no original LEAP trajectory reused',
          'wrist_position_m':sol.x[:3].tolist(),'quaternion_wxyz':[2**-.5,-2**-.5,0,0],
          'closed_hand_rad':dict(zip(names,q.tolist())),'open_hand_rad':dict(zip(names,opened.tolist())),
          'pad_fit':{n:{'position':a.tolist(),'normal':b.tolist(),'target':targets[n][0].tolist() if n in targets else None} for n,(a,b) in fk(q,sol.x[:3]).items()},
          'cost':float(sol.cost)}
    (Path(meta['run_dir'])/'preshape.json').write_text(json.dumps(data,indent=2))
    print(json.dumps(data,indent=2))

if __name__=='__main__':solve()
