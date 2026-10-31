"""Millimetre-scale BOX tactile distance regressions.

Small boxes have a sampled SDF whose padded extent can be smaller than a taxel
radius. A probe beyond that grid can still overlap the real BOX and must respond
when the rigid solver reports a contact with the sensor's link.
"""
import numpy as np
import pytest
import quadrants as qd
import torch

import genesis as gs
from genesis.engine.sensors.kinematic_tactile import _func_tactile_box_sdf_and_normal
from genesis.utils.misc import tensor_to_array

from ..utils.assertions import assert_allclose


@qd.kernel
def _query_box_geometry(points: qd.types.ndarray(), half_sizes: qd.types.ndarray(),
                        output: qd.types.ndarray()):
    for point_idx in range(points.shape[0]):
        distance, normal = _func_tactile_box_sdf_and_normal(
            qd.Vector([points[point_idx, 0], points[point_idx, 1], points[point_idx, 2]]),
            qd.Vector([half_sizes[0], half_sizes[1], half_sizes[2]]))
        output[point_idx, 0] = distance
        for axis in qd.static(range(3)):
            output[point_idx, axis + 1] = normal[axis]


@pytest.mark.required
def test_tactile_box_signed_distance_and_surface_normals(tol):
    half_sizes = torch.as_tensor([0.004, 0.004, 0.003], dtype=gs.tc_float, device=gs.device)
    points = np.array([
        [0, 0, 0],                         # interior center: nearest normal sign is nonunique
        [0.0035, 0, 0],                    # interior near +x
        [0, -0.0032, 0],                   # interior near -y
        [0, 0, -0.003],                    # exact face
        [0, 0, -0.0044],                   # face exterior, beyond old SDF padding
        [0.0046, 0.0048, 0],               # edge exterior: 0.6/0.8/1.0 mm triangle
        [0.0043, 0.0044, 0.0042],          # corner exterior: 0.3/0.4/1.2/1.3 mm
        [-0.0043, -0.0044, -0.0042],       # negative corner
        [0.004, 0.004, 0],                 # exact edge: normal is nonunique
        [0.004, 0.004, 0.003],             # exact corner: normal is nonunique
    ])

    def query(values):
        array = torch.as_tensor(values.copy(), dtype=gs.tc_float, device=gs.device)
        result = torch.empty((len(values), 4), dtype=gs.tc_float, device=gs.device)
        _query_box_geometry(array, half_sizes, result)
        return result.cpu().numpy()

    result = query(points)
    expected_distance = np.array([-0.003, -0.0005, -0.0008, 0, 0.0014,
                                  0.001, 0.0013, 0.0013, 0, 0])
    assert_allclose(result[:, 0], expected_distance, tol=tol)
    assert np.isfinite(result).all()
    assert_allclose(np.linalg.norm(result[:, 1:], axis=1), 1, tol=tol)
    expected_normal = np.array([
        [1, 0, 0], [0, -1, 0], [0, 0, -1], [0, 0, -1], [.6, .8, 0],
        [3/13, 4/13, 12/13], [-3/13, -4/13, -12/13],
    ])
    assert_allclose(result[1:8, 1:], expected_normal, tol=tol)
    assert_allclose(np.abs(result[0, 1:]), [0, 0, 1], tol=tol)
    assert (result[8, 1:3] >= 0).all() and abs(result[8, 3]) < tol
    assert (result[9, 1:] >= 0).all()

    # Independent derivative check at differentiable samples, including inside
    # and exactly on a face. Avoid nondifferentiable edge/corner/center ties.
    step = 1e-7
    numerical = np.empty((7, 3))
    for axis in range(3):
        perturbation = np.zeros_like(points[1:8])
        perturbation[:, axis] = step
        numerical[:, axis] = (query(points[1:8] + perturbation)[:, 0]
                              - query(points[1:8] - perturbation)[:, 0]) / (2 * step)
    assert_allclose(result[1:8, 1:], numerical, tol=max(tol * 100, 1e-6))


def _rotation_matrix(quaternion):
    w, x, y, z = quaternion
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 2])
@pytest.mark.parametrize("quaternion", [(1, 0, 0, 0), (0.8, 0.2, -0.4, 0.4)])
def test_small_box_contact_probes_beyond_sampled_sdf_grid(tol, n_envs, quaternion):
    radius = 0.0015
    stiffness = 500.0
    exponent = 1.2
    initial_overlap = 0.00001
    half_height = 0.003
    gaps = np.array([0.0010, 0.0012, 0.0014, 0.0016])
    probe_positions = np.array([[0, 0, half_height - initial_overlap - gap] for gap in gaps])
    rotation = _rotation_matrix(quaternion)
    origin = np.array([0.03, -0.02, 0.04])
    counterpart_center = origin + rotation @ np.array([0, 0, 2 * half_height - initial_overlap])
    scene = gs.Scene(
        show_viewer=False,
        sim_options=gs.options.SimOptions(dt=1 / 300, gravity=(0, 0, 0)),
    )
    pad = scene.add_entity(gs.morphs.Box(size=(0.02, 0.02, 0.006), pos=origin, quat=quaternion))
    box = scene.add_entity(gs.morphs.Box(size=(0.008, 0.008, 0.006),
                                       pos=counterpart_center, quat=quaternion, fixed=True))
    common = dict(entity_idx=pad.idx, probe_local_pos=probe_positions.tolist(),
                  probe_radius=radius, draw_debug=False)
    depth_sensor = scene.add_sensor(gs.sensors.ContactDepthProbe(**common))
    force_sensor = scene.add_sensor(gs.sensors.KinematicTaxel(
        **common, normal_stiffness=stiffness, normal_exponent=exponent,
        normal_damping=0, shear_scalar=0, twist_scalar=0))
    scene.build(n_envs=n_envs)
    scene.step()

    contacts = pad.get_contacts()
    valid = tensor_to_array(contacts.get('valid_mask', np.ones_like(tensor_to_array(contacts['link_a'])))).astype(bool)
    pair = ((tensor_to_array(contacts['link_a']) == box.base_link.idx)
            | (tensor_to_array(contacts['link_b']) == box.base_link.idx)) & valid
    assert pair.any(), 'Regression requires a real pad/BOX contact, not an un-gated proximity query.'

    pad_positions = tensor_to_array(pad.get_pos()).reshape(-1, 3)
    pad_quaternions = tensor_to_array(pad.get_quat()).reshape(-1, 4)
    box_positions = tensor_to_array(box.get_pos()).reshape(-1, 3)
    box_quaternions = tensor_to_array(box.get_quat()).reshape(-1, 4)
    expected_depth, expected_force = [], []
    for batch in range(max(n_envs, 1)):
        pad_rotation = _rotation_matrix(pad_quaternions[batch])
        box_rotation = _rotation_matrix(box_quaternions[min(batch, len(box_quaternions) - 1)])
        box_position = box_positions[min(batch, len(box_positions) - 1)]
        probes_world = probe_positions @ pad_rotation.T + pad_positions[batch]
        probes_box = (probes_world - box_position) @ box_rotation
        # All these points lie below the flat central face; distance is therefore
        # independently obtained from the measured BOX face plane.
        distances = -half_height - probes_box[:, 2]
        assert (distances[:3] > 0.0009).all(), 'The probes must remain outside the tiny sampled-grid padding.'
        assert (distances[:3] < radius).all()
        assert distances[3] > radius
        depth = np.maximum(0, radius - distances)
        normal_in_pad = np.array([0, 0, -1]) @ box_rotation.T @ pad_rotation
        expected_depth.append(depth)
        expected_force.append(stiffness * depth[:, None] ** exponent * normal_in_pad)

    observed_depth = tensor_to_array(depth_sensor.read_ground_truth()).reshape(-1, len(gaps))
    observed_force = tensor_to_array(force_sensor.read_ground_truth().force).reshape(-1, len(gaps), 3)
    assert (observed_depth[:, :3] > 0).all(), 'Probe/BOX overlap outside the sampled SDF grid must not disappear.'
    assert_allclose(observed_depth, np.array(expected_depth), tol=tol)
    assert_allclose(observed_force, np.array(expected_force), tol=tol)
    assert_allclose(observed_depth[:, 3], 0, tol=tol)
    assert_allclose(observed_force[:, 3], 0, tol=tol)
