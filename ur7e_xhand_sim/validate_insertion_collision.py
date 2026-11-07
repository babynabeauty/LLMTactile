"""Read-only, CPU validation of a built insertion scene's actual collision SDF.

The validator samples the *collision* surfaces already loaded by Genesis, not
visual OBJ files or an analytic replacement. It never sets entity poses, steps
the scene, or launches another simulator. A pass concerns these finite samples;
it is not a continuous collision proof or a successful robot episode.
"""
import numpy as np
from scipy.spatial.transform import Rotation

try:
    from .insertion_geometry import insertion_geometry_spec, target_object_pose
except ImportError:  # environment.py is also used as a directly executed script.
    from insertion_geometry import insertion_geometry_spec, target_object_pose


def _array(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _rotation(quaternion):
    q = _array(quaternion).astype(float).reshape(4)
    return Rotation.from_quat(q[[1, 2, 3, 0]])


def _pose(entity):
    p, q = _array(entity.get_pos()), _array(entity.get_quat())
    if p.size != 3 or q.size != 4:
        raise ValueError('Collision validation expects a single environment')
    return p.astype(float).reshape(3), _rotation(q)


def _geoms(entity):
    # Imported insertion assets have one link, with possibly multiple collisions.
    links = list(getattr(entity, 'links', [entity.base_link]))
    if len(links) != 1:
        raise ValueError('Insertion collision validation requires single-link assets')
    geoms = list(entity.base_link.geoms)
    if not geoms:
        raise ValueError('Insertion asset has no built collision geoms')
    return geoms


def _surface_samples(geom, spacing):
    """Deterministic samples: vertices, subdivided edges, and triangle interiors.

    Interior density is at least two samples per spacing squared of surface area.
    Edge subdivision prevents small or skinny triangles losing their boundaries.
    This is finite coverage, with no random state or simulation state changes.
    """
    vertices = _array(geom.init_verts).astype(float)
    faces = _array(geom.init_faces).astype(int).reshape(-1, 3)
    triangles = vertices[faces]
    samples = [vertices, triangles.mean(axis=1)]
    edges = np.unique(np.sort(faces[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1), axis=0)
    for a, b in vertices[edges]:
        count = int(np.ceil(np.linalg.norm(b-a)/spacing))
        if count > 1:
            samples.append(a+(b-a)*np.arange(1, count)[:, None]/count)
    areas = np.linalg.norm(np.cross(triangles[:, 1]-triangles[:, 0],
                                    triangles[:, 2]-triangles[:, 0]), axis=1)/2
    for triangle, area in zip(triangles, areas):
        count = max(1, int(np.ceil(2*area/spacing**2)))
        # Stratified radial coordinate with a low-discrepancy angular coordinate.
        i = np.arange(count, dtype=float)
        u = np.sqrt((i+.5)/count)
        v = np.mod((i+.5)*0.6180339887498949, 1.)
        samples.append((1-u[:, None])*triangle[0]
                       +(u*(1-v))[:, None]*triangle[1]
                       +(u*v)[:, None]*triangle[2])
    local = np.concatenate(samples)
    # Geom frame -> original single-link/source-asset frame. Inertia/COM is not
    # the asset origin and must not be subtracted from these vertices.
    return _rotation(geom.init_quat).apply(local)+_array(geom.init_pos).reshape(3)


class _SDF:
    def __init__(self, geom):
        if getattr(geom, '_is_preprocessed', True) is False:
            raise ValueError('Build the scene before validating; SDF is not preprocessed')
        self.geom = geom
        self.values = _array(geom.sdf_val)
        self.resolution = _array(geom.sdf_res).astype(int).reshape(3)
        self.cell = np.broadcast_to(_array(geom.sdf_cell_size).astype(float), (3,)).copy()
        self.transform = _array(geom.T_mesh_to_sdf).astype(float)
        self.position = _array(geom.init_pos).astype(float).reshape(3)
        self.rotation = _rotation(geom.init_quat)
        self.maximum = float(geom.sdf_max)
        if (self.values.shape != tuple(self.resolution) or np.any(self.resolution < 2)
                or np.any(self.cell <= 0) or not np.isfinite(self.transform).all()):
            raise ValueError('Invalid collision SDF shape, transform, or cell size')

    def sample(self, points_in_link):
        mesh_points = self.rotation.inv().apply(points_in_link-self.position)
        grid = mesh_points@self.transform[:3, :3].T+self.transform[:3, 3]
        inside = np.all((grid >= 0.) & (grid <= self.resolution-1), axis=1)
        # Match Genesis utils/sdf.py's outside-grid proxy; its value is not an
        # exact mesh distance. The report distinguishes how many samples use it.
        outside_offset = np.maximum(np.maximum(-grid, grid-(self.resolution-1)), 0.)
        distance = np.linalg.norm(outside_offset*self.cell, axis=1)+self.maximum
        p = grid[inside]
        base = np.minimum(np.floor(p).astype(int), self.resolution-2)
        fraction = p-base
        interpolated = np.zeros(len(p), dtype=float)
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    offset = np.array([dx, dy, dz])
                    weight = np.prod(np.where(offset, fraction, 1-fraction), axis=1)
                    index = base+offset
                    interpolated += weight*self.values[index[:, 0], index[:, 1], index[:, 2]]
        distance[inside] = interpolated
        return distance, inside

    def metadata(self):
        return dict(geom_index=int(self.geom.idx), is_convex=bool(self.geom.is_convex),
                    vertex_count=len(self.geom.init_verts), face_count=len(self.geom.init_faces),
                    sdf_resolution=self.resolution.tolist(), sdf_cell_size_m=self.cell.tolist(),
                    T_mesh_to_sdf=self.transform.tolist(),
                    geom_position_in_link_m=self.position.tolist(),
                    geom_quaternion_in_link_wxyz=_array(self.geom.init_quat).reshape(4).tolist())


def _sample_union(sdfs, points):
    distance = np.full(len(points), np.inf)
    in_any_grid = np.zeros(len(points), dtype=bool)
    for sdf in sdfs:
        current, inside = sdf.sample(points)
        distance = np.minimum(distance, current)
        in_any_grid |= inside
    return distance, in_any_grid


def validate_collision_geometry(sim):
    """Return JSON-safe SDF evidence and ``pass`` for tube_insert/usb_insert.

    Usage after Sandbox construction::
        report = validate_collision_geometry(sim)

    All distances are meters; SDF values below zero mean socket penetration.
    Correct paths run from 5 mm above the mouth through target_object_pose's
    nominal depth. USB additionally tests zero keyed yaw at the same tip poses.
    No collision with the hand, table or initial holder is assessed here.
    """
    spec = insertion_geometry_spec(sim.task)
    task = spec['task']
    socket = sim.task_fixtures['socket']
    socket_position, socket_rotation = _pose(socket)
    socket_quaternion = socket_rotation.as_quat()[[3, 0, 1, 2]]
    goal = target_object_pose(task, socket_position, socket_quaternion)
    goal_position = np.asarray(goal['position_m'])
    goal_rotation = _rotation(goal['quaternion_wxyz'])
    mouth_rotation = socket_rotation*_rotation(spec['mouth_quaternion_local_wxyz'])
    outward_axis = mouth_rotation.apply([0., 0., 1.])
    target_depth = spec['target_depth_m']
    mouth_position = goal_position+target_depth*outward_axis
    sdfs = [_SDF(geom) for geom in _geoms(socket)]
    movable_geoms = _geoms(sim.cube)
    spacing = min(.0005, min(float(sdf.cell.min()) for sdf in sdfs))
    points = np.concatenate([_surface_samples(geom, spacing) for geom in movable_geoms])
    # A numerical tolerance only, not a voxel-sized penetration allowance.
    tolerance = 1e-6
    depth_step = min(.001, 2*min(float(sdf.cell.min()) for sdf in sdfs))
    depths = np.r_[np.linspace(-.005, 0., int(np.ceil(.005/depth_step))+1),
                   np.linspace(0., target_depth, int(np.ceil(target_depth/depth_step))+1)[1:]]

    def sweep(orientation):
        rotated = orientation.apply(points)
        rows = []
        for depth in depths:
            position = mouth_position-depth*outward_axis
            local = socket_rotation.inv().apply(rotated+position-socket_position)
            distance, inside = _sample_union(sdfs, local)
            worst = int(np.argmin(distance))
            rows.append(dict(depth_m=float(depth), min_distance_m=float(distance[worst]),
                             penetrating_sample_count=int(np.count_nonzero(distance < -tolerance)),
                             inside_grid_sample_count=int(inside.sum()),
                             outside_all_grids_sample_count=int((~inside).sum()),
                             minimum_sample_in_socket_grid=bool(inside[worst]),
                             minimum_sample_object_local_m=points[worst].tolist(),
                             clear=bool(np.isfinite(distance).all() and np.all(distance >= -tolerance))))
        return rows

    correct = sweep(goal_rotation)
    # Probe an open corridor down the aperture as an independent sealed-hole
    # check. The USB probe avoids the legitimate negative-X internal tongue.
    corridor_offset = ([spec['socket_aperture_half_size_m'][0]/2, 0., 0.]
                       if task == 'usb_insert' else [0., 0., 0.])
    engaged_depths = depths[depths >= 0.]
    corridor = (mouth_position+mouth_rotation.apply(corridor_offset)
                -engaged_depths[:, None]*outward_axis)
    corridor_distance, corridor_inside = _sample_union(
        sdfs, socket_rotation.inv().apply(corridor-socket_position))
    nonconvex = any(not sdf.geom.is_convex for sdf in sdfs)
    corridor_open = bool(corridor_inside.all() and np.isfinite(corridor_distance).all()
                         and np.all(corridor_distance > tolerance))
    correct_clear = all(row['clear'] for row in correct)
    wrong = sweep(mouth_rotation) if task == 'usb_insert' else None
    wrong_blocked = (any(row['min_distance_m'] < -tolerance for row in wrong
                         if row['depth_m'] > 0.) if wrong else None)
    final_clear = correct[-1]['clear'] and correct[-1]['inside_grid_sample_count'] > 0
    passed = nonconvex and corridor_open and correct_clear and final_clear and (wrong_blocked is not False)
    return dict(task=task, units='meters', scope='finite collision-surface samples against built socket SDF',
                **{'pass': bool(passed)},
                nonconvex_socket=bool(nonconvex), aperture_corridor_open=corridor_open,
                correct_path_clear=bool(correct_clear), target_pose_clear=bool(final_clear),
                wrong_usb_yaw_blocked=wrong_blocked,
                socket_pose=dict(position_m=socket_position.tolist(), quaternion_wxyz=socket_quaternion.tolist()),
                target_object_pose=goal, socket_geoms=[sdf.metadata() for sdf in sdfs],
                movable_geoms=[dict(geom_index=int(g.idx), is_convex=bool(g.is_convex),
                                    vertex_count=len(g.init_verts), face_count=len(g.init_faces))
                               for g in movable_geoms],
                sampling=dict(surface_sample_count=len(points), surface_spacing_m=spacing,
                              maximum_depth_step_m=depth_step, penetration_tolerance_m=tolerance,
                              method='vertices, subdivided edges, deterministic area-stratified triangle samples',
                              outside_grid_values='Genesis distance-to-grid proxy plus sdf_max'),
                aperture_corridor=dict(offset_in_mouth_m=corridor_offset,
                                       min_distance_m=float(corridor_distance.min()),
                                       samples=[dict(depth_m=float(d), distance_m=float(s), in_grid=bool(i))
                                                for d, s, i in zip(engaged_depths, corridor_distance, corridor_inside)]),
                correct_path=correct, wrong_usb_yaw_path=wrong,
                limitations=['Finite samples do not prove continuous collision freedom.',
                             'No scene steps, object moves, grasp, force or robot-task success are tested.'])
