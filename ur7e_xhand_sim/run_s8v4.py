"""Actual vlm-policy Codex inference loop for the Genesis S8V4 skill port."""
import argparse,base64,json,os,time,urllib.request,urllib.error
from pathlib import Path
from PIL import Image
from scipy.spatial.transform import Rotation
from s8v4 import VLM_ROOT,SkillAct,ARM_JOINTS,HAND_JOINTS
from vlm_policy_local.codex_policy import CodexAppServerPolicy
from vlm_policy_local.core import Config

POLICY_VIEWS=('scene_front','scene_left','wrist')

INSTRUCTIONS='''You control a simulated UR7e + XHAND1 using the vlm-policy S8V4 adaptation.
Task: grasp the orange wooden-block proxy in front of the hand, lift it at least
5 cm clear of the table and hold for 3 simulated seconds. Physics pauses during
model decisions. No hardware is connected. Return only SkillAct JSON.
Use the current observation_id and sequence locate -> approach -> align -> close
-> test_lift -> lift -> hold -> done. These are genuine simulated physical motions,
not object attachment or teleportation. GenDexGrasp ranks small wrist offsets;
S8V4 RuleSupportController runs on simulated time during motion after contact.
The three current images are scene_front (fixed front overview), scene_left
(fixed left external view), and wrist (wrist-mounted camera following hand
translation and rotation). No hand close-ups are provided. Use all three views
to assess occlusion and contact.
For locate, give a tight bounding box around the orange block in scene_front ONLY,
using the width/height in stack.image_sizes_px, in [x1,y1,x2,y2] order. It is small on the tabletop;
exclude the hand. Local code reconstructs only visible RGB-D points within it.
For approach select an exact supplied GenDex candidate offset. For align and close
preserve that selected offset. Other offsets and non-locate boxes must be zero.
Observe images and feedback after each skill. Contact estimates and joint
tracking deflection are imperfect support signals, not success. After test_lift
or lift visually check whether the block moved with the hand. If the fingers
missed or the cube stayed on the table, abort instead of inventing success.
The skill executor currently cannot re-register/reposition after close. Do not
repeat skills invalid in the current phase. Return abort with skill=none if stuck.
Only return done, skill=none after a completed hold and visual evidence the block
is still held. Independent evaluation occurs after control, not in your input.
Tactile values are uncalibrated simulated nominal N, never hardware SDK raw units.
The initial hand targets derive from a candidate XHand example; this run validates
them in a new scene, not an already proven grasp. No trained tactile GRU is used.
'''
class Policy(CodexAppServerPolicy):
    def _inputs(self,observation):
        items,count=super()._inputs(observation)
        (self.run_dir/f'api_input_{self.calls:03d}.json').write_text(json.dumps(items,ensure_ascii=False,indent=2))
        return items,count
    def context(self,observation,interactions=None):
        return {'observation_id':observation['observation_id'],'state':observation['state'],
            'stack':observation['stack'],'recent_interactions':self.history[-1:] if interactions is None else interactions}

def build_observation(current):
    o=current['observation'];q=o['wrist_quaternion_wxyz']
    images=[];sizes={}
    for name in POLICY_VIEWS:
        path=Path(current['image_dir'])/(name+'.jpg')
        with Image.open(path) as im:sizes[name]=list(im.size)
        images.append({'name':name,'mime_type':'image/jpeg','data_base64':base64.b64encode(path.read_bytes()).decode()})
    # The skill policy never commands a TCP delta. Name the observed frame explicitly.
    state={'tcp_pose':o['wrist_position_m']+Rotation.from_quat([q[1],q[2],q[3],q[0]]).as_rotvec().tolist(),
        'pose_frame':'right_hand_link in simulation world; NOT hardware TCP',
        'arm_q':[o['joint_positions_rad'][n] for n in ARM_JOINTS],
        'hand_q':[o['joint_positions_rad'][n] for n in HAND_JOINTS],'tactile':o['tactile']}
    return {'observation_id':current['observation_id'],'backend':'genesis_simulation_only','state':state,'images':images,
        'stack':{**{k:v for k,v in current.items() if k not in ['observation','image_dir','run_dir','observation_id']},
            'image_sizes_px':sizes,'roi_camera':'scene_front'}}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--label',required=True);parser.add_argument('--url',default='http://127.0.0.1:8767');parser.add_argument('--max-turns',type=int,default=12)
    parser.add_argument('--connection',choices=['direct','inherit','project-proxy'],default='direct')
    parser.add_argument('--check-connection',action='store_true',help='Initialize the independent policy API without touching the simulation')
    parser.add_argument('--reference',type=Path,help='Verified successful Genesis demonstration reference')
    parser.add_argument('--object-shape',choices=['cube','cylinder','sphere','cuboid','hex_prism'],default=None,
        help='Explicit multi-object Astra demonstration; bridge must use matching --object-shape')
    parser.add_argument('--previous-attempt',type=Path,action='append',default=[],
        help='Previous actual Astra run directory; supply its actions and executor failure as retry context')
    args=parser.parse_args()
    def call(name,payload={}):
        req=urllib.request.Request(args.url+'/'+name,json.dumps(payload).encode(),{'Content-Type':'application/json'})
        try:
            with urllib.request.urlopen(req,timeout=300) as r:response=json.load(r)
        except urllib.error.HTTPError as exc:
            detail=exc.read().decode('utf-8',errors='replace')
            raise RuntimeError(f'Bridge {name} HTTP {exc.code}: {detail}') from exc
        if not response['ok']:raise RuntimeError(response.get('error'))
        return response['result']
    if not args.label or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in args.label):
        parser.error('label must be alphanumeric, underscore or hyphen')
    if args.connection=='direct':
        # Match the original project's B2D_CODEX_DIRECT=1 launch path.
        for name in ('HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy'):
            os.environ.pop(name,None)
        os.environ['NO_PROXY']=os.environ['no_proxy']='*'
    else:
        if args.connection=='project-proxy':
            local=VLM_ROOT/'configs/bench2dex.local.json'
            proxy=json.loads(local.read_text()).get('codex_proxy')
            if not proxy:parser.error('No codex_proxy configured')
            for name in ('HTTP_PROXY','HTTPS_PROXY','ALL_PROXY'):os.environ[name]=proxy
        bypass=os.environ.get('NO_PROXY','')
        os.environ['NO_PROXY']=os.environ['no_proxy']=bypass+',127.0.0.1,localhost'
    out=Path(__file__).resolve().parents[1]/'outputs/s8v4_agent'/args.label
    out.mkdir(parents=True,exist_ok=False)
    (out/'runner.json').write_text(json.dumps({'connection':args.connection,'bridge':args.url,
        'driver':'vlm_policy_CodexAppServerPolicy','model':'gpt-6-astra','check_only':args.check_connection,'image_views':POLICY_VIEWS},indent=2))
    policy=None;current=None
    instruction_text=INSTRUCTIONS
    example=None
    if args.reference:
        from success_context import SuccessContext
        example=SuccessContext(args.reference)
        instruction_text=INSTRUCTIONS.replace(
            'The initial hand targets derive from a candidate XHand example; this run validates\nthem in a new scene, not an already proven grasp. No trained tactile GRU is used.',
            'One verified successful UR7e/XHand SIMULATION demonstration is supplied below. '
            'The skill executor uses its palm orientation, RGB-D anchor-relative translation and full recorded close/preload commands. '
            'The nominal demonstrated zero offset is supplied alongside the GenDex-ranked candidates; choose from these using live images and demonstration evidence. '
            'The task target is the ORANGE CUBE. Ignore other colored cubes and spheres; do not touch them. '
            'The initial object location may differ from the historical example. Re-localize from LIVE images. '
            'Upward skills require BOTH thumb and index tactile above threshold for 5 consecutive samples. Joint deflection alone cannot authorize lift. '
            'RuleSupportController runs after closure; no trained tactile GRU is part of S8V4. '
            'This is the complete available UR/XHand S8V4 component chain, not the original Panda/Allegro embodiment.')
        (out/'demonstration_context.json').write_text(json.dumps({'metadata':example.metadata(),'frames':example.frames},indent=2))
    if args.object_shape:
        if not args.reference:parser.error('Multi-object demonstrations require the historical skill reference')
        description=call('describe')
        if description.get('object_spec',{}).get('shape')!=args.object_shape:
            raise ValueError('Requested Astra shape does not match the simulation bridge')
        (out/'object_scene.json').write_text(json.dumps(description,indent=2))
        instruction_text=instruction_text.replace('orange wooden-block proxy',f'orange rigid {args.object_shape}')
        instruction_text=instruction_text.replace('ORANGE CUBE',f'ORANGE {args.object_shape.upper()}')
        instruction_text=instruction_text.replace(
            'and full recorded close/preload commands',
            'with fresh bounded tactile close/preload commands')
        instruction_text=instruction_text.replace('select an exact supplied GenDex candidate offset',
            'select an exact supplied candidate offset (ranked or explicitly unranked)')
        instruction_text += (
            f'\nTHIS RUN: the current orange target is a {args.object_shape}, not necessarily the historical cube. '
            'The historical example supplies a nominal pinch pose only; it does not prove a grasp on this shape. '
            'The bridge now performs fresh bounded tactile closure for this object, not replay of historical close commands. '
            'It increases index closure until both thumb and index have sustained tactile contact, then adds bounded preload. '
            'Inspect LIVE RGB images and tactile feedback. Different shapes may need different alignments. '
            'The supplied candidates include explicitly labeled unranked small X/Z offsets as well as GenDex candidates. '
            'Choose the best candidate based on current geometry; do not assume zero offset is always best. '
            'All skills must be selected by you. Upward motion and holding require persistent support; abort on failed contact. '
            'Do not claim success until the orange object is lifted and held for the complete hold. '
            'Do not include material values in explanations; use observable geometry and contact feedback.\n')
        reference_data=json.loads(args.reference.read_text())
        reference_preload=(reference_data['closed_hand_target_rad'][4]
                           -reference_data['light_hand_target_rad'][4])
        instruction_text += (
            '\nThe shape-study close skill exposes preload_increment_rad. For close, explicitly '
            'choose one of 0.05, 0.10, 0.15, 0.20 radians using the demonstration and live evidence; '
            'for every other action set it to 0. This is a bounded index-joint preload AFTER fresh '
            'persistent contact, not a force measurement. Zero uses the default 0.05. '
            f'The actual historical successful example added {reference_preload:.3f} radians '
            'between light contact and the completed preload. Earlier shape-study attempts '
            'with no explicit selection used only 0.05. Account for this difference when '
            'deciding whether to change alignment or preload. Joint and effort limits remain enforced.\n')
        instruction_text += (
            '\nThe shape-study locate skill also exposes preshape_id. Choose an exact supplied '
            'preshape candidate from the current observation when locating; on every other action '
            'set preshape_id to keep. These are bounded, unverified hand-shape candidates. Their '
            'hand-only geometric scores are NOT measured grasp success or object ground truth. '
            'The chosen open hand, scoring hand and registration compensation are applied before '
            'GenDex ranking. Reference preserves the historical hand; alternative candidates '
            'can adjust thumb opposition/index spread or coordinate the index proximal/distal '
            'joints during preload to preserve fingertip orientation, as described in each candidate. '
            'Use the object geometry, prior attempts '
            'and live feedback to choose. More preload is not a substitute for opposing contacts '
            'on a curved object. No physical success is assumed for any candidate.\n')
        instruction_text += (
            '\nThe shape-study close skill also exposes grip_gain. On close choose 1, 2, or 3; '
            'on every other action set it to 0. This selects the common thumb/index PD gain '
            'multiplier, which remains active during subsequent lift and hold. It is not a '
            'force reading and does not change effort limits. Earlier attempts without this '
            'field used gain 1. Increasing stiffness may help maintain contact under load, '
            'but cannot guarantee a stable grasp or repair poor contact geometry. Choose '
            'using the demonstrated outcomes and current feedback.\n')
    if args.previous_attempt:
        retry_context=[]
        for previous in args.previous_attempt:
            previous=previous.resolve()
            actions=[json.loads(p.read_text())['action'] for p in sorted(previous.glob('model_[0-9][0-9][0-9].json'))]
            score=json.loads((previous/'score.json').read_text())
            source=Path(json.loads((previous/'bridge_run.json').read_text())['run_dir'])
            source_manifest=json.loads((source/'manifest.json').read_text())
            after=json.loads((source/'interactions.json').read_text())[-1]['after']
            retry_context.append({'previous_actions':actions,'executor_fault':score.get('fault'),
                'last_skill_phase':score.get('phase'),
                'object_shape':source_manifest.get('object_spec',{}).get('shape','cube'),
                'executor_closure':source_manifest.get('closure'),
                'executor_preshape':source_manifest.get('preshape_selection') or {
                    'selected_id':'reference','provenance':'older bridge exposed only the reference hand'},
                'executor_grip_gain':source_manifest.get('grip_gain_selection') or {
                    'effective_gain':1.,'provenance':'older bridge used the original PD gains'},
                'observable_feedback':{key:after[key] for key in ('support','closure_state','motion_recovery_state','preload_selection','preshape_selection','grip_gain_selection') if key in after}})
        # Only feedback already exposed by the skill executor is passed back.
        # Keep evaluator geometry/material truth out of the action policy.
        (out/'previous_attempt_context.json').write_text(json.dumps(retry_context,indent=2))
        instruction_text += ('\nPrevious actual attempts ended as follows; each record identifies its object shape. '
            'A result on another shape is comparative evidence, not proof of a grasp on the current target. '
            'Use this feedback together with the NEW live images to choose your next attempt; '
            'the scene is reset. The bridge permits bounded tabletop preload settling and '
            'contact recovery, but still requires confirmed thumb/index tactile before lift. '
            'Account for the recorded executor version: an early abort before completing preload '
            'under v1 is not evidence of failure under the current settling/recovery procedure.\n'
            +json.dumps(retry_context)+'\n')
    if args.reference and example.reference.get('execution_mode')=='recorded_tripod_closure_v1':
        instruction_text += ('\nTHREE-FINGER RECORDED DEMONSTRATION OVERRIDE: Use thumb, index and middle. '
            'Only preshape reference/keep is available. close replays the recorded three-finger close/preload hand targets; '
            'choose preload_increment_rad=0 (no extra preload override), and grip_gain=2 as in the source. '
            'All three fingers must meet tactile support gates. Astra still chooses locate/approach/align/close/lift skills from live views. '
            'Middle gain remains nominal. See the mode declaration below for whether a learned gain controller is loaded.\n')
    if args.object_shape and description.get('online_ppo',{}).get('enabled'):
        if example.reference.get('object_shape')==args.object_shape:
            instruction_text += ('\nThe supplied example is the previously recorded successful '
                +args.object_shape+' grasp, with refreshed tactile samples, not a cube-only example. '
                'Its nominal registration anchor is approximate; use LIVE views to assess alignment.\n')
        instruction_text += ('\nLOCAL PPO ENABLED: a frozen tactile PPO adjusts only common thumb/index PD gain '
            'at15Hz from preload onward. Your grip_gain choice applies before PPO takeover only. '
            'Original bounded closure, support gates and support target corrections remain active. '
            'You still choose all high-level skills; PPO does not authorize lift or establish success. '
            'This is experimental online transfer from demonstration replay; check actual image/contact feedback.\n')
        (out/'online_ppo.json').write_text(json.dumps(description['online_ppo'],indent=2))
    if args.object_shape and not description.get('online_ppo',{}).get('enabled'):
        instruction_text += ('\nLOCAL PPO DISABLED: no learned residual controller is loaded. '
            'Use the ordinary Astra skill executor and original tactile support rule. '
            'Your initial grip_gain remains in effect; there are no PPO gain updates. '
            'The same supplied object demonstration and live tactile gates still apply.\n')
        (out/'online_ppo.json').write_text(json.dumps({'enabled':False},indent=2))
    (out/'instructions.txt').write_text(instruction_text)
    try:
        policy=Policy(Config(),out,task=f'Lift the {args.object_shape or "block"} and hold it',model='gpt-6-astra',effort='medium',timeout=180,plan_steps=1,
            action_model=SkillAct,instruction_text=instruction_text,codex_bin='/home/sai/zsq/bench2dex-work/codex/node_modules/.bin/codex',codex_home=str(Path.home()/'.codex'))
        if example:
            policy.demonstration=example;policy.demonstration_content=example.content(image_detail=policy.image_detail)
            if args.object_shape:
                for item in policy.demonstration_content:
                    if item.get('type')=='input_text':
                        item['text']=item['text'].replace(
                            'The full recorded close/preload hand commands are available to the skill executor.',
                            'For this shape study the historical closure is context only; the bridge executes fresh tactile closure and your bounded preload selection.')
        print('AGENT_READY',policy.thread_id,flush=True)
        if args.check_connection:return
        current=call('s8v4_start',{'label':args.label,'driver':'vlm_policy_CodexAppServerPolicy',
            **({'reference':str(args.reference.resolve())} if args.reference else {})})
        (out/'bridge_run.json').write_text(json.dumps({'run_dir':current['run_dir'],'thread_id':policy.thread_id},indent=2))
        for turn in range(args.max_turns):
            obs=build_observation(current);print('MODEL_START',turn,current['phase'],flush=True)
            action,meta=policy.decide(obs)
            (out/f'model_{turn:03d}.json').write_text(json.dumps({'action':action.model_dump(),'metadata':meta},indent=2))
            print('ACTION',action.model_dump_json(),flush=True)
            reply=call('s8v4_act',action.model_dump());policy.feedback(obs,action,reply['receipt']);current=reply['policy_observation']
            print('RESULT',json.dumps(reply['receipt']),current['phase'],flush=True)
            if action.status!='continue' or current['fault']:break
        else:
            reply=call('s8v4_act',{'observation_id':current['observation_id'],'status':'abort','explanation':'Model-call budget reached','skill':'none','offset_xyz_m':[0.,0.,0.],'object_roi_xyxy':[0,0,0,0]})
        score=call('s8v4_score');(out/'score.json').write_text(json.dumps(score,indent=2))
        print('SCORE',json.dumps(score),flush=True)
    except BaseException as exc:
        (out/'runner_error.json').write_text(json.dumps({'error_type':type(exc).__name__,'error':str(exc)},indent=2))
        # End the paused trial on model startup/network failure; never retry a motion.
        try:
            if current is None:raise RuntimeError('No simulation trial was started')
            current=call('s8v4_observe')
            call('s8v4_act',{'observation_id':current['observation_id'],'status':'abort','explanation':'Runner failed before next action: '+str(exc)[:500],'skill':'none','offset_xyz_m':[0.,0.,0.],'object_roi_xyxy':[0,0,0,0]})
        except Exception:pass
        raise
    finally:
        if policy:policy.close()
if __name__=='__main__':main()
