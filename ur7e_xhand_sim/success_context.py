"""Verified same-embodiment simulation demonstration for model context."""
import base64
import hashlib
import json
from pathlib import Path

VIEWS=('scene_front','scene_left','wrist')

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load_reference(path):
    path=Path(path).resolve();ref=json.loads(path.read_text());root=path.parent
    if ref.get('schema')!='ur7e_xhand_genesis_success_reference_v1' or ref.get('review_status')!='simulation_verified' or not ref['validation']['success']:
        raise ValueError('A verified successful simulation reference is required')
    for name,value in ref['files'].items():
        if digest(root/name)!=value:raise ValueError('Demonstration file hash mismatch: '+name)
    return ref,root

class SuccessContext:
    def __init__(self,path):
        self.reference,self.root=load_reference(path);self.sha256=digest(path)
        selected={'open','aligned','light_contact','closed','test_lifted','lifted','held'}
        stages=json.loads((self.root/'stages.json').read_text())
        self.frames=[]
        for s in stages:
            if s['name'] not in selected:continue
            o=s['observation'];tactile={}
            for f,v in o['tactile'].items():
                tactile[f]={k:v[k] for k in ('active_taxels','peak_force_N','force_sum_sensor_N')}
                tactile[f]['nonzero_sensor_xyz']=[{'taxel':i+1,'xyz':xyz} for i,xyz in enumerate(v['force_sensor_N']) if any(abs(x)>1e-8 for x in xyz)]
            self.frames.append({'step':s['step'],'time_s':s.get('sample_time_s',s['step']*.025),'stage':s['name'],
                'actual_joint_positions_rad':o['joint_positions_rad'],
                'sent_joint_target_rad':s['sent_joint_target_rad'],
                'palm_pose_world':s['palm_pose_world'],'tactile':tactile,
                'images':[{'name':v,'path':str(self.root/Path(s['image_dir']).name/(v+'.jpg'))} for v in VIEWS]})
        if len(self.frames)!=7:raise ValueError('Missing demonstration keyframes')
    def metadata(self):
        return {'kind':'genesis_ur7e_xhand_success_context_v1','sha256':self.sha256,
                'source':str(self.root),'frames':[f['step'] for f in self.frames],
                'images':len(self.frames)*3,'trajectory_steps':self.reference['validation']['control_steps'],
                'sampling':'7 chronological keyframes, full recorded trajectory retained by skill executor',
                'simulation_only':True}
    def content(self,api_mode='responses',image_detail='high'):
        result=[{'type':'input_text','text':'BEGIN ONE HISTORICAL SUCCESSFUL UR7e/XHAND SIMULATION GRASP. '
            'These are ordered keyframes from the user-selected rule demonstration, with TRAJECTORY_GAPs. '
            'Use the demonstrated palm orientation, participating fingers and closure pattern; register translation from LIVE RGB-D. '
            'Historical images are NOT the current scene. Actual joints differ from commanded targets. '
            'Tactile is uncalibrated simulation nominal N, not real load. No live object truth is supplied. '
            'The full recorded close/preload hand commands are available to the skill executor. '
            'Choose skills and a GenDex offset or the explicitly supplied demonstrated baseline; inspect current feedback.'}]
        for f in self.frames:
            result.append({'type':'input_text','text':'HISTORICAL KEYFRAME / TRAJECTORY_GAP\n'+json.dumps({k:v for k,v in f.items() if k!='images'})})
            for im in f['images']:
                result.append({'type':'input_text','text':f"HISTORICAL step {f['step']} {f['stage']} camera {im['name']}"})
                result.append({'type':'input_image','image_url':'data:image/jpeg;base64,'+base64.b64encode(Path(im['path']).read_bytes()).decode(),'detail':image_detail})
        result.append({'type':'input_text','text':'END HISTORICAL EXAMPLE. Only the following LIVE images describe the current scene.'})
        return result
