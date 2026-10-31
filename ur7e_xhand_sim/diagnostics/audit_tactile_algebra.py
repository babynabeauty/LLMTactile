"""CPU-only audit of the *current source* spring/damper algebra.

This executes the function body extracted from kinematic_tactile.py with small
NumPy vector adapters. It does not execute compiled Quadrants kernels or replace
the controlled physical-contact tests. In particular, failure of a physical
invariant is reported, rather than hidden behind an assertion expecting the bug.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'Genesis/genesis/engine/sensors/kinematic_tactile.py'


class Vector(np.ndarray):
    def __new__(cls, values):
        return np.asarray(values, dtype=float).view(cls)

    def cross(self, other):
        return Vector(np.cross(self, other))


def inverse_rotate(vector, quaternion):
    w, x, y, z = quaternion
    qv = np.array([x, y, z])
    return Vector(vector - 2 * w * np.cross(qv, vector)
                  + 2 * np.cross(qv, np.cross(qv, vector)))


def source_function():
    source = SOURCE.read_text()
    tree = ast.parse(source)
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == '_func_kinematic_spring_damper')
    function.decorator_list = []
    function.returns = None
    for arg in function.args.args:
        arg.annotation = None
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    namespace = {
        'qd': SimpleNamespace(Vector=SimpleNamespace(zero=lambda dtype, n: Vector(np.zeros(n))),
                              pow=np.power),
        'gs': SimpleNamespace(qd_float=float),
        'gu': SimpleNamespace(qd_inv_transform_by_quat=inverse_rotate),
    }
    exec(compile(module, str(SOURCE), 'exec'), namespace)
    return namespace[function.name], ast.get_source_segment(source, function), tree


def evaluate(function, penetration, other_velocity=(0, 0, 0), sensor_velocity=(0, 0, 0),
             normal=(0, 0, 1), quaternion=(1, 0, 0, 0), stiffness=500, damping=1,
             exponent=1.2, shear=2, other_angular=(0, 0, 0), twist=0):
    velocity = np.zeros((2, 1, 3)).view(Vector)
    velocity[0, 0] = sensor_velocity
    velocity[1, 0] = other_velocity
    angular = np.zeros_like(velocity).view(Vector)
    angular[1, 0] = other_angular
    state = SimpleNamespace(links=SimpleNamespace(cd_vel=velocity, cd_ang=angular,
                                                  root_COM=np.zeros_like(velocity)))
    force, torque = function(0, 0, penetration, 1, Vector(normal), Vector([0, 0, 0]),
                            Vector([0, 0, 0]), Vector(quaternion), stiffness, damping,
                            exponent, shear, twist, state)
    return np.asarray(force), np.asarray(torque)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    function, segment, tree = source_function()
    depths = np.array([0, .00001, .00005, .0001, .0005, .001, .002])
    static = np.array([evaluate(function, depth)[0] for depth in depths])
    no_contact = evaluate(function, 0, other_velocity=(.03, -.04, -.05),
                          other_angular=(1, 2, 3), twist=1)
    positive = evaluate(function, .001, other_velocity=(.02, 0, 0))[0]
    negative = evaluate(function, .001, other_velocity=(-.02, 0, 0))[0]
    approach = evaluate(function, .001, other_velocity=(0, 0, .02))[0]
    withdraw = evaluate(function, .001, other_velocity=(0, 0, -.02))[0]
    rotation = np.array([np.sqrt(.5), 0, 0, np.sqrt(.5)])
    rotated = evaluate(function, .001, other_velocity=(0, .02, 0), quaternion=rotation)[0]
    shear_power = float(positive @ np.array([-.02, 0, 0]))
    torsion = evaluate(function, .001, other_angular=(0, 0, .02), twist=1)[1]
    torsion_power = float(torsion @ np.array([0, 0, -.02]))
    checks = {
        'no_contact_outputs_zero': bool(all(np.all(v == 0) for v in no_contact)),
        'static_force_matches_500_depth_power_1p2': bool(np.allclose(static[:, 2], 500 * depths**1.2)),
        'static_force_strictly_increases_with_penetration': bool(np.all(np.diff(static[:, 2]) > 0)),
        'static_force_has_no_tangent': bool(np.all(static[:, :2] == 0)),
        'normal_damping_increases_for_approach': bool(approach[2] > static[-2, 2] > withdraw[2]),
        'shear_is_odd_in_relative_velocity': bool(np.allclose(positive[:2], -negative[:2])),
        'frame_rotation_covariance': bool(np.allclose(rotated, positive)),
        'shear_dissipative_if_force_is_on_sensor': shear_power <= 0,
        'torsion_dissipative_if_torque_is_on_sensor': torsion_power <= 0,
    }
    pref = {}
    for name in ('_kernel_build_sensor_contact_idx', '_kernel_build_sensor_geom_idx'):
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
        text = ast.unparse(node)
        pref[name] = dict(logical_to_physical_contact_mapping_present=
                         'collider_state.contact_sort_idx[i_c_logical, i_b]' in text)
    old = ROOT / 'outputs/tactile_contact_audit_20261024'
    prior = json.loads((old / 'source_after.json').read_text())
    current_hash = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    report = {
        'schema': 'source_algebra_cpu_audit_v1',
        'scope': 'Extracted current Python function algebra; no compiled kernel, no physical simulation, no source changes.',
        'source': str(SOURCE), 'source_sha256': current_hash,
        'function_sha256': hashlib.sha256(segment.encode()).hexdigest(),
        'current_source_matches_oct24_fixed_file': current_hash == prior['sha256'],
        'oct24_validation': str(old / 'validation.json'),
        'contact_prefilter_mapping_check': pref,
        'checks': checks,
        'static_penetration_m': depths.tolist(), 'static_force_sensor_local_N': static.tolist(),
        'tangent_other_relative_velocity_m_s': [.02, 0, 0],
        'force_sensor_local_N': positive.tolist(),
        'force_other_motion_reversed_sensor_local_N': negative.tolist(),
        'relative_pair_damping_power_W_if_sensor_force': shear_power,
        'relative_pair_torsion_power_W_if_sensor_torque': torsion_power,
        'interpretation': [
            'Static normal direction is defined by the other geometry outward normal; upstream static tests identify it as force on the sensor.',
            ('Current shear gives positive pair power; confirm the sign issue with a compiled controlled-motion test.'
             if shear_power > 0 else 'Current shear gives nonpositive pair power, consistent with dissipative force on the sensor.'),
            ('Twist has the analogous sign issue, but production twist_scalar is zero.'
             if torsion_power > 0 else 'Current twist gives nonpositive pair power; production twist_scalar remains zero.'),
            'Shear depends only on relative tangential velocity once penetration > 0: no static friction stress/history, no friction coefficient or Coulomb cap.',
            'Taxel force magnitude is an uncalibrated geometry/velocity estimate and is not a solver contact-force distribution.',
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'output': str(args.output), 'checks': checks, 'prefilter': pref,
                      'current_source_matches_oct24_fixed_file': report['current_source_matches_oct24_fixed_file']}, indent=2))


if __name__ == '__main__':
    main()
