import json,hashlib,shutil
from pathlib import Path
from scipy.spatial.transform import Rotation
root=Path(__file__).resolve().parents[2];out=Path(__file__).resolve().parent;demo=out/'demo'
src=root/'outputs/direct_teacher_replay_tactile_v3_20261028_v1/cylinder/attempt01';c=root/'outputs/shared_multishape_ppo_tactile_v3_20261028_v1/campaign'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
base=json.loads((root/'outputs/rule_demos/teacher_pinch_v2_20261024/reference.json').read_text());stages=json.loads((src/'stages.json').read_text());by={s['name']:s for s in stages}
mapping={'preshape_end':'open','align_end':'aligned','close_end':'light_contact','preload_end':'closed','test_lift_end':'test_lifted','lift_end':'lifted','hold_end':'held'}
rows=[]
import numpy as np
raw=np.load(src/'raw_tactile.npz')
finger_order=json.loads((src/'manifest.json').read_text())['sensor_finger_order']
for s in stages:
 if s['name'] not in mapping:continue
 s=dict(s);s['name']=mapping[s['name']];o=s['observation'];T=__import__('numpy').eye(4);q=o['wrist_quaternion_wxyz'];T[:3,:3]=Rotation.from_quat([q[1],q[2],q[3],q[0]]).as_matrix();T[:3,3]=o['wrist_position_m'];s['palm_pose_world']=T.tolist()
 for i,finger in enumerate(finger_order):o['tactile'][finger]['force_sensor_N']=raw['tactile'][s['step']-1,i].tolist()
 folder=demo/f"frame_{s['step']:06d}";folder.mkdir(exist_ok=True);s['image_dir']=str(folder)
 for view in ['scene_front','scene_left','wrist']:shutil.copy2(src/'frames'/f"{s['step']:06d}_{view}.jpg",folder/f'{view}.jpg')
 rows.append(s)
(demo/'stages.json').write_text(json.dumps(rows))
path=demo/'control.jsonl'
if not path.exists():path.symlink_to(src/'control.jsonl')
score=json.loads((src/'score.json').read_text());assert score['task_success']
base.update(run_dir=str(demo),validation=score,object_shape='cylinder',source_run=str(src),control_dt_s=1/60,
 open_hand_target_rad=by['preshape_end']['sent_joint_target_rad'][6:],light_hand_target_rad=by['close_end']['sent_joint_target_rad'][6:],closed_hand_target_rad=by['preload_end']['sent_joint_target_rad'][6:],
 grasp_quaternion_wxyz=by['align_end']['observation']['wrist_quaternion_wxyz'],
 desired_align_palm_world_m=by['align_end']['observation']['wrist_position_m'],
 align_palm_pose_world=next(s['palm_pose_world'] for s in rows if s['name']=='aligned'),
 registration_note='Uses prior calibrated nominal visible-surface anchor at the same original workspace location as the cylinder demonstration; alignment and hand targets are from actual cylinder replay. Approximate cross-shape registration anchor; live localization must verify alignment.')
base.pop('align_tool0_pose_world',None);base['files']={str(p.relative_to(demo)):sha(p) for p in demo.rglob('*') if p.is_file() and p.name!='reference.json'}
(demo/'reference.json').write_text(json.dumps(base,indent=2))
p=json.loads((c/'inputs/protocol.json').read_text());selected=json.loads((c/'selection.json').read_text())['selected_models']['seed1_tactile'];trajectory=Path(next(o for o in p['objects'] if o['object_id']=='cylinder')['trajectory_path'])
config={'checkpoint':selected['path'],'checkpoint_sha256':selected['sha256'],'trajectory':str(trajectory),'trajectory_sha256':sha(trajectory),'model_selection':'seed1 tactile previously used in HTTP parity check, not selected using new trial outcomes','object':'cylinder','mass_kg':.125,'friction':.5,'demonstration_reference':str(demo/'reference.json'),'demonstration_reference_sha256':sha(demo/'reference.json'),'online_adapter':'causal held-next-target, phase-local reference clock; experimental transfer'}
(out/'config.json').write_text(json.dumps(config,indent=2));print('Prepared',out)
