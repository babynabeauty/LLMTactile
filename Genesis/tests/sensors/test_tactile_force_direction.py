"""Physical sign/frame invariants for the compiled kinematic taxel force model.

Contact-existence/SDF tests live in test_tactile.py. Here two free bodies provide
genuine DynState fields, but no dynamics are stepped: prescribed virtual contact
depth and outward counterpart normal isolate force/torque direction from the
collision solver. Returned force/torque are interpreted as acting on the sensor,
consistent with the existing plane/sphere static force-direction tests.
"""
import numpy as np
import pytest
import quadrants as qd
import torch

import genesis as gs
import genesis.utils.array_class as array_class
from genesis.engine.sensors.kinematic_tactile import _func_kinematic_spring_damper

from ..utils.assertions import assert_allclose


@qd.kernel
def _query_force_model(
    state: array_class.DynState,
    sensor_idx: int,
    other_idx: int,
    depth: float,
    normal: qd.types.ndarray(),
    output: qd.types.ndarray(),
):
    for batch in range(output.shape[0]):
        force, torque = _func_kinematic_spring_damper(
            batch,
            sensor_idx,
            depth,
            other_idx,
            qd.Vector([normal[0], normal[1], normal[2]]),
            state.links.pos[sensor_idx, batch],
            qd.Vector([0.0, 0.0, 0.0]),
            state.links.quat[sensor_idx, batch],
            500.0,
            1.0,
            1.2,
            2.0,
            1.0,
            state,
        )
        for component in qd.static(range(3)):
            output[batch, component] = force[component]
            output[batch, component + 3] = torque[component]


def _rotation_matrix(quaternion):
    w, x, y, z = quaternion
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 2])
def test_kinematic_taxel_force_direction_and_frame(tol, n_envs):
    scene = gs.Scene(
        show_viewer=False,
        sim_options=gs.options.SimOptions(gravity=(0.0, 0.0, 0.0)),
    )
    # Coincident COMs make the sampled point stationary under either body's spin.
    # No scene.step is called; overlap cannot cause any solver-generated motion.
    sensor = scene.add_entity(gs.morphs.Box(size=(0.1, 0.1, 0.1)))
    other = scene.add_entity(gs.morphs.Box(size=(0.1, 0.1, 0.1)))
    scene.build(n_envs=n_envs)
    output = torch.empty((max(n_envs, 1), 6), dtype=gs.tc_float, device=gs.device)

    def sample(depth, relative_linear=(0, 0, 0), relative_angular=(0, 0, 0),
               quaternion=(1, 0, 0, 0), move_sensor=False):
        rotation = _rotation_matrix(quaternion)
        sensor.set_quat(quaternion)
        linear = rotation @ np.asarray(relative_linear)
        angular = rotation @ np.asarray(relative_angular)
        sensor_velocity = np.r_[-linear, np.zeros(3)] if move_sensor else np.zeros(6)
        other_velocity = np.r_[np.zeros(3) if move_sensor else linear, angular]
        sensor.set_dofs_velocity(sensor_velocity)
        other.set_dofs_velocity(other_velocity)
        normal = torch.as_tensor(rotation[:, 2].copy(), dtype=gs.tc_float, device=gs.device)
        _query_force_model(scene.rigid_solver.dyn_state, sensor.base_link.idx,
                           other.base_link.idx, depth, normal, output)
        return output.cpu().numpy().copy()

    depths = (0.00005, 0.0001, 0.0005, 0.001)
    static = np.array([sample(depth) for depth in depths])
    assert (static[:, :, 2] > 0).all(), "Normal force must push into the sensor, along the counterpart outward normal."
    assert (np.diff(static[:, :, 2], axis=0) > 0).all(), "Increasing static indentation must increase normal load."
    assert_allclose(static[:, :, :2], 0.0, tol=tol)
    assert_allclose(static[:, :, 3:], 0.0, tol=tol)

    baseline = sample(0.001)
    approach = sample(0.001, relative_linear=(0, 0, 0.02))
    withdraw = sample(0.001, relative_linear=(0, 0, -0.02))
    assert (approach[:, 2] > baseline[:, 2]).all(), "Approach should increase compressive force."
    assert (withdraw[:, 2] < baseline[:, 2]).all(), "Withdrawal should reduce compressive force."
    # Subtract the elastic component: only damping must dissipate relative motion.
    assert (((approach[:, :3] - baseline[:, :3]) @ np.array([0, 0, -0.02])) < 0).all()
    assert (((withdraw[:, :3] - baseline[:, :3]) @ np.array([0, 0, 0.02])) < 0).all()

    tangent = np.array([0.02, -0.03, 0.0])
    positive = sample(0.001, relative_linear=tangent)
    negative = sample(0.001, relative_linear=-tangent)
    assert (positive[:, :3] @ -tangent < 0).all(), "Shear damping must dissipate sensor/other relative motion."
    assert (negative[:, :3] @ tangent < 0).all()
    assert_allclose(positive[:, :2], -negative[:, :2], tol=tol)
    assert_allclose(positive[:, 2], baseline[:, 2], tol=tol)
    assert_allclose(sample(0.001, relative_linear=tangent, move_sensor=True), positive, tol=tol)

    angular = np.array([0.0, 0.0, 0.02])
    spin_positive = sample(0.001, relative_angular=angular)
    spin_negative = sample(0.001, relative_angular=-angular)
    assert (spin_positive[:, 3:] @ -angular < 0).all(), "Twist damping must dissipate relative spin."
    assert (spin_negative[:, 3:] @ angular < 0).all()
    assert_allclose(spin_positive[:, 3:], -spin_negative[:, 3:], tol=tol)
    assert_allclose(spin_positive[:, :3], baseline[:, :3], tol=tol)

    # Rotating the entire experiment must leave sensor-local readings unchanged.
    combined_linear = np.array([0.02, -0.03, 0.01])
    reference = sample(0.001, relative_linear=combined_linear, relative_angular=angular)
    for quaternion in ((np.sqrt(0.5), 0, 0, np.sqrt(0.5)), (0.5, 0.5, 0.5, 0.5)):
        rotated = sample(0.001, relative_linear=combined_linear,
                         relative_angular=angular, quaternion=quaternion)
        assert_allclose(rotated, reference, tol=tol)

    for depth in (0.0, -0.0001):
        air = sample(depth, relative_linear=combined_linear,
                     relative_angular=angular, quaternion=(0.5, 0.5, 0.5, 0.5))
        assert_allclose(air, 0.0, tol=tol, err_msg="No penetration must suppress force and torque despite motion.")
