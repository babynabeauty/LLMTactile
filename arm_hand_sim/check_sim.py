"""Exercise the live sandbox; keep only an aggregate validation report."""
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'astra_bridge'))
from client import call

URL = 'http://127.0.0.1:8766'


def api(command, payload=None):
    return call(command, payload, URL)


def main():
    report = {}
    d = api('describe')
    assert len(d['joint_order']) == len(set(d['joint_order'])) == 27
    assert [d['joints'][n]['index'] for n in d['joint_order']] == list(range(27))
    report['joint_count'] = 27
    o = api('reset')
    assert sum(s['active_taxels'] for s in o['tactile'].values()) == 0
    before = api('status');time.sleep(.4)
    assert api('status') == before
    for payload in [{'actions': {'missing_joint': 0}}, {'actions': {'joint1': 1.01}}, {'steps': 11}]:
        try:
            api('step', payload)
        except RuntimeError:
            pass
        else:
            raise AssertionError('Invalid command was accepted')
        assert api('status') == before
    report['paused_time_and_rejected_commands'] = 'pass'
    for _ in range(20):
        o = api('step', {'steps': 10})
        assert all(math.isfinite(v) for v in o['joint_positions_rad'].values())
        assert not o['terminated']
    assert o['episode_step'] == 200 and o['sim_time_s'] == 5
    report['stable_control_steps'] = 200
    z = o['wrist_position_m'][2]
    o = api('move', {'delta': [0, 0, .02], 'steps': 10})
    assert o['wrist_position_m'][2] > z + .005
    report['arm_wrist_motion_m'] = o['wrist_position_m'][2] - z
    q = o['joint_positions_rad']
    o = api('hand', {'closure': .55, 'steps': 10})
    motion = max(abs(o['joint_positions_rad'][n] - q[n]) for n in d['hand_joints'])
    assert motion > .1
    report['hand_max_joint_motion_rad'] = motion
    api('reset')
    # Horizontal palm starts farther above the object than the former fingertips-down pose.
    for _ in range(10):
        o = api('move', {'delta': [0, 0, -.02], 'steps': 10})
    o = api('observe', {'full': True})
    active = sum(s['active_taxels'] for s in o['tactile'].values())
    assert active > 0
    assert sum(len(s['force_torque']) for s in o['tactile'].values()) == 140
    for s in o['tactile'].values():
        assert len(s['force_torque']) == s['taxels']
        assert all(len(row) == 6 and all(math.isfinite(v) for v in row) for row in s['force_torque'])
    report['contact_active_taxels'] = active
    report['full_tactile_shape'] = [140, 6]
    from urllib.request import urlopen
    for image in ['scene', 'hand']:
        with urlopen(f'{URL}/{image}.jpg') as response:
            data = response.read()
        assert data[:2] == b'\xff\xd8' and data[-2:] == b'\xff\xd9'
    report['two_camera_jpegs'] = 'pass'
    api('reset')
    report['result'] = 'pass'
    report['scope'] = 'Integration and physical contact checks; not grasp success evaluation or hardware calibration.'
    (ROOT / 'arm_hand_sim/validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
