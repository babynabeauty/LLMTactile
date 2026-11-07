"""Geometric insertion checks in original asset frames (meters, wxyz).

These checks do not certify a grasp, a collision-free trajectory, stable contact,
or a completed robot episode. They can also be used before building Genesis.
Measurements come from UniVTAC TestTube{,Slot}.usd and ContactWorld's released
manifeel_usb_{plug,socket}.obj. In both movable meshes the origin is the tip,
not the center of mass. The USB's open-surface plastic insert is on negative X;
it must face away from the socket's negative-X tongue (180 degree yaw).
"""
from copy import deepcopy

import numpy as np
from scipy.spatial.transform import Rotation


_SPECS = {
    'tube_insert': {
        'mouth_position_local_m': [-.00781781, 0., .07650077],
        'mouth_quaternion_local_wxyz': [.9659258262890683, 0., -.25881904510252074, 0.],
        'object_goal_quaternion_in_mouth_wxyz': [1., 0., 0., 0.],
        'object_tip_local_m': [0., 0., 0.],
        'target_depth_m': .040,
        'success_depth_range_m': [.035, .045],
        'max_axis_angle_deg': 1.,
        'hole_radius_m': .0105,
        'hole_polygon_sides': 32,
        'shaft_radius_m': .010,
        'rounded_tip_length_m': .010,
        'axial_rotation_matters': False,
        'definition': '35–45 mm engagement, shaft fully inside the measured 21 mm bore; axial spin is free',
    },
    'usb_insert': {
        'mouth_position_local_m': [0., 0., .018],
        'mouth_quaternion_local_wxyz': [1., 0., 0., 0.],
        'object_goal_quaternion_in_mouth_wxyz': [0., 0., 0., 1.],
        'object_tip_local_m': [0., 0., 0.],
        'target_depth_m': .017,
        'success_depth_range_m': [.016, .018],
        'max_orientation_angle_deg': 1.,
        'max_lateral_components_m': [.00025, .0005],
        'socket_aperture_half_size_m': [.00459, .0117],
        'plug_metal_half_size_m': [.003, .011],
        'plug_empty_cavity_xy_bounds_m': [[-.0004, -.0104], [.0024, .0104]],
        'socket_tongue_bounds_m': [[-.00207, -.00882, 0.], [-.00009, .00882, .0162]],
        'axial_rotation_matters': True,
        'definition': '16–18 mm engagement, 180 degree keyed yaw, metal inside aperture and tongue inside empty cavity',
    },
}

_ALIASES = {'insert_tube': 'tube_insert', 'tube_insertion': 'tube_insert',
            'insert_usb': 'usb_insert', 'usb_insertion': 'usb_insert'}


def _task(task):
    task = _ALIASES.get(task, task)
    if task not in _SPECS:
        raise ValueError('Insertion task must be tube_insert or usb_insert')
    return task


def insertion_geometry_spec(task):
    """Return a JSON-safe copy of the task's geometry and chosen goal window."""
    task = _task(task)
    return dict(task=task, units='meters', quaternion_order='wxyz',
                success_scope='instantaneous geometric state; not complete robot-task success',
                **deepcopy(_SPECS[task]))


def _position(value, name):
    value = np.asarray(value, dtype=float)
    if value.shape != (3,) or not np.isfinite(value).all():
        raise ValueError(f'{name} must contain three finite coordinates')
    return value


def _rotation(value, name):
    value = np.asarray(value, dtype=float)
    if value.shape != (4,) or not np.isfinite(value).all() or np.linalg.norm(value) < 1e-12:
        raise ValueError(f'{name} must contain a finite nonzero wxyz quaternion')
    return Rotation.from_quat(value[[1, 2, 3, 0]])


def _mouth_frame(spec, socket_position, socket_quaternion_wxyz):
    position = _position(socket_position, 'socket_position')
    rotation = _rotation(socket_quaternion_wxyz, 'socket_quaternion_wxyz')
    mouth_position = position+rotation.apply(spec['mouth_position_local_m'])
    mouth_rotation = rotation*_rotation(spec['mouth_quaternion_local_wxyz'], 'mouth quaternion')
    return mouth_position, mouth_rotation


def target_object_pose(task, socket_position, socket_quaternion_wxyz):
    """Return the nominal object tip-origin pose, without moving any entity."""
    spec = _SPECS[_task(task)]
    mouth_position, mouth_rotation = _mouth_frame(spec, socket_position, socket_quaternion_wxyz)
    goal_rotation = mouth_rotation*_rotation(spec['object_goal_quaternion_in_mouth_wxyz'], 'goal quaternion')
    return dict(position_m=(mouth_position+mouth_rotation.apply([0., 0., -spec['target_depth_m']])).tolist(),
                quaternion_wxyz=goal_rotation.as_quat()[[3, 0, 1, 2]].tolist())


def _corners(lower, upper):
    return np.array(np.meshgrid(*zip(lower, upper), indexing='ij')).reshape(3, -1).T


def evaluate_insertion_pose(task, object_position, object_quaternion_wxyz,
                            socket_position, socket_quaternion_wxyz):
    """Evaluate tip depth, alignment and conservative geometric clearances.

    All input poses use original source mesh origins. Depth is positive into the
    socket. No pass condition relies on world Z or the fixture bounding-box center.
    Tube twist is unimportant; USB uses the full rotation error, including yaw.
    The finite depth window intentionally rejects a tip teleported through a base.
    """
    task = _task(task)
    spec = _SPECS[task]
    obj_position = _position(object_position, 'object_position')
    obj_rotation = _rotation(object_quaternion_wxyz, 'object_quaternion_wxyz')
    mouth_position, mouth_rotation = _mouth_frame(spec, socket_position, socket_quaternion_wxyz)
    tip = obj_position+obj_rotation.apply(spec['object_tip_local_m'])
    relative_tip = mouth_rotation.inv().apply(tip-mouth_position)
    relative_rotation = mouth_rotation.inv()*obj_rotation
    axis = relative_rotation.apply([0., 0., 1.])
    depth = -float(relative_tip[2])
    axis_error = float(np.degrees(np.arccos(np.clip(axis[2], -1., 1.))))
    goal_rotation = _rotation(spec['object_goal_quaternion_in_mouth_wxyz'], 'goal quaternion')
    full_error = float(np.degrees((goal_rotation.inv()*relative_rotation).magnitude()))
    dmin, dmax = spec['success_depth_range_m']
    # Tiny epsilon is numerical only, not permission for base penetration.
    depth_ok = dmin-1e-9 <= depth <= dmax+1e-9
    if task == 'tube_insert':
        # The inscribed bore radius is conservative for its 32-sided mesh.
        bore_radius = spec['hole_radius_m']*np.cos(np.pi/spec['hole_polygon_sides'])
        axis_z = max(float(axis[2]), 1e-9)
        shaft_start = spec['rounded_tip_length_m']
        shaft_end = max(shaft_start, max(depth, 0.)/axis_z)
        shaft_centers = relative_tip+np.array([shaft_start, shaft_end])[:, None]*axis
        max_radial = float(np.linalg.norm(shaft_centers[:, :2], axis=1).max())
        # Tilt creates an elliptical cross-section in a plane across the bore.
        clearance = float(bore_radius-spec['shaft_radius_m']/axis_z-max_radial)
        orientation_error = axis_error
        orientation_ok = axis_error <= spec['max_axis_angle_deg']
        alignment_ok = clearance >= -1e-9
        detail = dict(shaft_max_centerline_radial_error_m=max_radial,
                      conservative_radial_clearance_m=clearance)
    else:
        # Compare all engaged metal-shell bounding corners against the aperture.
        half = np.asarray(spec['plug_metal_half_size_m'])
        z_end = max(depth, 0.)/max(float(axis[2]), 1e-9)
        shell = _corners([-half[0], -half[1], 0.], [half[0], half[1], z_end])
        shell_mouth = relative_rotation.apply(shell)+relative_tip
        outer_clearance = float((np.asarray(spec['socket_aperture_half_size_m'])-
                                 np.abs(shell_mouth[:, :2])).min())
        floor_clearance = float(shell_mouth[:, 2].min()+spec['mouth_position_local_m'][2])
        # Socket tongue bounds use the socket frame; its lower unengaged part is
        # clipped at the tip's center plane. Roll/pitch is limited to one degree.
        lo, hi = np.asarray(spec['socket_tongue_bounds_m'])
        lo = lo.copy()
        lo[2] = min(hi[2], max(lo[2], spec['mouth_position_local_m'][2]-depth))
        tongue_mouth = _corners(lo, hi)-np.asarray(spec['mouth_position_local_m'])
        tongue_object = relative_rotation.inv().apply(tongue_mouth-relative_tip)
        cavity_lo, cavity_hi = np.asarray(spec['plug_empty_cavity_xy_bounds_m'])
        inner_clearance = float(min((tongue_object[:, :2]-cavity_lo).min(),
                                    (cavity_hi-tongue_object[:, :2]).min()))
        clearance = min(outer_clearance, inner_clearance, floor_clearance)
        orientation_error = full_error
        orientation_ok = full_error <= spec['max_orientation_angle_deg']
        alignment_ok = bool(np.all(np.abs(relative_tip[:2]) <=
                                     np.asarray(spec['max_lateral_components_m'])+1e-9)
                            and clearance >= -1e-9)
        detail = dict(metal_to_aperture_clearance_m=outer_clearance,
                      tongue_to_empty_cavity_clearance_m=inner_clearance,
                      metal_tip_to_socket_floor_clearance_m=floor_clearance)
    return dict(task=task, tip_world_m=tip.tolist(), mouth_world_m=mouth_position.tolist(),
                tip_in_mouth_m=relative_tip.tolist(),
                insertion_direction_world=mouth_rotation.apply([0., 0., -1.]).tolist(),
                lateral_error_m=float(np.linalg.norm(relative_tip[:2])),
                lateral_components_m=relative_tip[:2].tolist(), insertion_depth_m=depth,
                axis_error_deg=axis_error, orientation_error_deg=orientation_error,
                full_orientation_error_deg=full_error, clearance_margin_m=clearance,
                depth_ok=bool(depth_ok), alignment_ok=bool(alignment_ok),
                orientation_ok=bool(orientation_ok),
                geometric_success=bool(depth_ok and alignment_ok and orientation_ok),
                success_scope='instantaneous geometric state; not complete robot-task success',
                goal=target_object_pose(task, socket_position, socket_quaternion_wxyz),
                thresholds=insertion_geometry_spec(task), **detail)
