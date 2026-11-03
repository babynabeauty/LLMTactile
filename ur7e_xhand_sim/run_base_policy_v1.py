"""Astra agent for one BasePolicyV1 rollout against a running bridge server (server.py)."""
from pathlib import Path
import argparse
import base64
import json
import os
import urllib.error
import urllib.request

from PIL import Image
from scipy.spatial.transform import Rotation

from base_policy_v1 import SkillAct, ARM_JOINTS, HAND_JOINTS, instructions
from vlm_policy_local.codex_policy import CodexAppServerPolicy
from vlm_policy_local.core import Config
from success_context import SuccessContext

ROOT = Path(__file__).resolve().parents[1]
POLICY_VIEWS = ('scene_front', 'scene_left', 'wrist')


class Policy(CodexAppServerPolicy):
    def _inputs(self, observation):
        items, count = super()._inputs(observation)
        (self.run_dir/f'api_input_{self.calls:03d}.json').write_text(json.dumps(items, ensure_ascii=False, indent=2))
        return items, count

    def context(self, observation, interactions=None):
        return {'observation_id': observation['observation_id'], 'state': observation['state'],
                'stack': observation['stack'], 'recent_interactions': self.history[-1:] if interactions is None else interactions}


def build_observation(current):
    o, q = current['observation'], current['observation']['wrist_quaternion_wxyz']
    images, sizes = [], {}
    for name in POLICY_VIEWS:
        path = Path(current['image_dir'])/(name+'.jpg')
        with Image.open(path) as image:
            sizes[name] = list(image.size)
        images.append({'name': name, 'mime_type': 'image/jpeg', 'data_base64': base64.b64encode(path.read_bytes()).decode()})
    state = {'tcp_pose': o['wrist_position_m']+Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_rotvec().tolist(),
             'pose_frame': 'right_hand_link in simulation world; NOT hardware TCP',
             'arm_q': [o['joint_positions_rad'][n] for n in ARM_JOINTS],
             'hand_q': [o['joint_positions_rad'][n] for n in HAND_JOINTS], 'tactile': o['tactile']}
    return {'observation_id': current['observation_id'], 'backend': 'genesis_simulation_only', 'state': state,
            'images': images, 'stack': {**{k: v for k, v in current.items()
                                           if k not in ['observation', 'image_dir', 'run_dir', 'observation_id']},
                                        'image_sizes_px': sizes, 'roi_camera': 'scene_front'}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--label', required=True)
    p.add_argument('--url', default='http://127.0.0.1:8767')
    p.add_argument('--demo', type=Path, required=True, help='v1 demo.json; its reference.json is the ICL example')
    p.add_argument('--placement-json', type=json.loads, default={})
    p.add_argument('--max-turns', type=int, default=12)
    args = p.parse_args()

    def call(name, payload=None):
        request = urllib.request.Request(args.url+'/'+name, json.dumps(payload or {}).encode(),
                                         {'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                reply = json.load(response)
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f'Bridge {name} HTTP {exc.code}: {exc.read().decode(errors="replace")}') from exc
        if not reply['ok']:
            raise RuntimeError(reply.get('error'))
        return reply['result']
    for name in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy'):
        os.environ.pop(name, None)
    os.environ['NO_PROXY'] = os.environ['no_proxy'] = '*'
    out = ROOT/'outputs/base_policy_v1_agent'/args.label
    out.mkdir(parents=True, exist_ok=False)
    demo = json.loads(args.demo.read_text())
    example = SuccessContext(args.demo.parent/demo['reference'])
    text = instructions(demo)
    (out/'instructions.txt').write_text(text)
    (out/'demonstration_context.json').write_text(json.dumps({'metadata': example.metadata(), 'frames': example.frames}, indent=2))
    policy = current = None
    abort = lambda obs, why: {'observation_id': obs['observation_id'], 'status': 'abort', 'explanation': why,
                              'skill': 'none', 'offset_xyz_m': [0., 0., 0.], 'object_roi_xyxy': [0, 0, 0, 0]}
    try:
        policy = Policy(Config(), out, task=f"Lift the {demo['object_spec']['shape']} and hold it", model='gpt-6-astra',
            effort='medium', timeout=180, plan_steps=1, action_model=SkillAct, instruction_text=text,
            codex_bin='/home/sai/zsq/bench2dex-work/codex/node_modules/.bin/codex', codex_home=str(Path.home()/'.codex'))
        policy.demonstration = example
        policy.demonstration_content = example.content(image_detail=policy.image_detail)
        current = call('trial_start', {'label': args.label, 'demo': str(args.demo.resolve()), 'placement': args.placement_json})
        (out/'bridge_run.json').write_text(json.dumps({'run_dir': current['run_dir'], 'thread_id': policy.thread_id}, indent=2))
        for turn in range(args.max_turns):
            obs = build_observation(current)
            action, meta = policy.decide(obs)
            (out/f'model_{turn:03d}.json').write_text(json.dumps({'action': action.model_dump(), 'metadata': meta}, indent=2))
            reply = call('trial_act', action.model_dump())
            policy.feedback(obs, action, reply['receipt'])
            current = reply['policy_observation']
            print('ACTION', action.model_dump_json(), 'RESULT', json.dumps(reply['receipt']), flush=True)
            if action.status != 'continue' or current['fault']:
                break
        else:
            call('trial_act', abort(current, 'Model-call budget reached'))
        score = call('trial_score')
        (out/'score.json').write_text(json.dumps(score, indent=2))
        print('SCORE', json.dumps(score), flush=True)
    except BaseException as exc:
        (out/'runner_error.json').write_text(json.dumps({'error_type': type(exc).__name__, 'error': str(exc)}, indent=2))
        try:
            if current is not None:
                call('trial_act', abort(call('trial_observe'), 'Runner failed: '+str(exc)[:500]))
        except Exception:
            pass
        raise
    finally:
        if policy:
            policy.close()


if __name__ == '__main__':
    main()
