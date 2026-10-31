"""CPU compiled-kernel sign check, isolated from contact detection and actuation.

Two free bodies only provide genuine Genesis DynState fields. The production
spring/damper function is queried at a specified virtual depth and normal. This
is not a grasp, sensor calibration, or solver-contact force comparison.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault('QD_OFFLINE_CACHE_FILE_PATH', '/tmp/tactile_validation_cpu_qdcache')
os.environ.setdefault('GS_CACHE_FILE_PATH', '/tmp/tactile_validation_cpu_gscache')
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
os.environ.setdefault('OMP_NUM_THREADS', '4')

import genesis as gs
import numpy as np
import quadrants as qd
import torch
from genesis.engine.sensors.kinematic_tactile import _func_kinematic_spring_damper
import genesis.utils.array_class as array_class


@qd.kernel
def query(dyn_state: array_class.DynState, sensor_idx: int, other_idx: int,
          penetration: float, output: qd.types.ndarray()):
    force, torque = _func_kinematic_spring_damper(
        0, sensor_idx, penetration, other_idx, qd.Vector([0., 0., 1.]),
        qd.Vector([0., 0., 0.]), qd.Vector([0., 0., 0.]),
        qd.Vector([1., 0., 0., 0.]), 500., 1., 1.2, 2., 1., dyn_state)
    for j in qd.static(range(3)):
        output[j] = force[j]
        output[j + 3] = torque[j]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    gs.init(backend=gs.cpu, precision='32', logging_level='warning')
    scene = gs.Scene(show_viewer=False,
                     sim_options=gs.options.SimOptions(gravity=(0, 0, 0)))
    sensor = scene.add_entity(gs.morphs.Box(size=(.1, .1, .1), pos=(0, 0, 0)))
    other = scene.add_entity(gs.morphs.Box(size=(.1, .1, .1), pos=(1, 0, 0)))
    scene.build()
    cases = [
        ('static', .001, [0, 0, 0, 0, 0, 0]),
        ('other_plus_x', .001, [.02, 0, 0, 0, 0, 0]),
        ('other_minus_x', .001, [-.02, 0, 0, 0, 0, 0]),
        ('other_approach_normal', .001, [0, 0, .02, 0, 0, 0]),
        ('other_withdraw_normal', .001, [0, 0, -.02, 0, 0, 0]),
        ('other_plus_angular_z', .001, [0, 0, 0, 0, 0, .02]),
        ('no_contact_with_motion', 0., [.02, 0, .02, 0, 0, .02]),
    ]
    rows = []
    out = torch.zeros(6, dtype=gs.tc_float, device=gs.device)
    for name, depth, velocity in cases:
        sensor.set_dofs_velocity([0, 0, 0, 0, 0, 0])
        other.set_dofs_velocity(velocity)
        query(scene.rigid_solver.dyn_state, sensor.base_link.idx, other.base_link.idx,
              depth, out)
        values = out.cpu().numpy().copy()
        rows.append({'case': name, 'virtual_penetration_m': depth,
                     'other_relative_linear_velocity_m_s': velocity[:3],
                     'other_relative_angular_velocity_rad_s': velocity[3:],
                     'force_sensor_local_N': values[:3].tolist(),
                     'torque_sensor_local_Nm': values[3:].tolist(),
                     'translational_pair_power_W_if_force_on_sensor':
                         float(values[:3] @ -np.array(velocity[:3])),
                     'rotational_pair_power_W_if_torque_on_sensor':
                         float(values[3:] @ -np.array(velocity[3:])),
                     })
    path = Path(__file__).resolve().parents[2] / 'Genesis/genesis/engine/sensors/kinematic_tactile.py'
    result = {'schema': 'compiled_tactile_force_cpu_audit_v1', 'backend': 'cpu',
              'source': str(path), 'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
              'scope': __doc__, 'rows': rows,
              'checks': {
                  'compiled_static_positive_normal': rows[0]['force_sensor_local_N'][2] > 0,
                  'compiled_shear_dissipative_if_force_on_sensor':
                      all(rows[i]['translational_pair_power_W_if_force_on_sensor'] <= 0 for i in (1, 2)),
                  'compiled_torsion_dissipative_if_torque_on_sensor':
                      rows[5]['rotational_pair_power_W_if_torque_on_sensor'] <= 0,
                  'compiled_no_contact_zero': rows[6]['force_sensor_local_N'] == [0, 0, 0]
                      and rows[6]['torque_sensor_local_Nm'] == [0, 0, 0],
              }}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result['checks'], indent=2))
    gs.destroy()


if __name__ == '__main__':
    main()
