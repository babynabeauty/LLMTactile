"""Optional source-bounded fingertip collision hulls for geometry diagnostics.

Genesis' generic decimation/decomposition can move collision vertices outside a
finger's authored surface. That is consequential when taxels are millimetres
wide. The current thumb coverage gap was traced to a real fixed back-shell,
not a millimetre of preprocessing expansion; this option is NOT its fix.
Resolve the ordinary robot description, then replace just the five sensor
link collision descriptions *before* either solver creates geometry. No global
Genesis patch, asset rewrite, sensor-radius change, or inertia recomputation is
performed. A source convex hull fills concavities, but introduces no vertex
outside the source convex envelope; this is explicitly not a faithful mesh of
holes or inner cavities.
"""
from dataclasses import replace
from pathlib import Path
import hashlib
import json
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
import trimesh


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _origin_transform(origin):
    T = np.eye(4)
    if origin is not None:
        T[:3, :3] = Rotation.from_euler('xyz', np.fromstring(origin.get('rpy', '0 0 0'), sep=' ')).as_matrix()
        T[:3, 3] = np.fromstring(origin.get('xyz', '0 0 0'), sep=' ')
    return T


def source_components(urdf_path, link_name):
    """Source meshes of a sensor link and its welded descendants, in link frame.

    The distal links include real fixed back-shell and tip sublinks. Keeping them
    is essential: their surfaces and inertias must not vanish in this correction.
    """
    root = ET.parse(urdf_path).getroot()
    links = {x.get('name'): x for x in root.findall('link')}
    children = {}
    for joint in root.findall('joint'):
        if joint.get('type') == 'fixed':
            children.setdefault(joint.find('parent').get('link'), []).append(joint)
    result = []
    stack = [(link_name, np.eye(4))]
    while stack:
        name, T_link = stack.pop()
        node = links[name]
        for collision in node.findall('collision'):
            mesh = collision.find('geometry/mesh')
            if mesh is None:
                raise ValueError(f'{name}: expected an authored collision mesh')
            path = Path(mesh.attrib['filename'])
            if not path.is_absolute():
                path = Path(urdf_path).parent/path
            source = trimesh.load_mesh(path, process=True)
            source.apply_scale(np.fromstring(mesh.get('scale', '1 1 1'), sep=' '))
            source.apply_transform(T_link @ _origin_transform(collision.find('origin')))
            hull = source.convex_hull
            if not hull.is_watertight or not hull.is_winding_consistent or not hull.is_convex or hull.volume <= 0:
                raise ValueError(f'{name}: source hull is not a closed outward convex solid')
            np.testing.assert_allclose(hull.bounds, source.bounds, rtol=0, atol=1e-12)
            error = float(cKDTree(source.vertices).query(hull.vertices)[0].max())
            if error > 1e-12:
                raise ValueError(f'{name}: source hull moved a vertex by {error} m')
            result.append((source, hull, path, name))
        for joint in children.get(name, []):
            stack.append((joint.find('child').get('link'), T_link @ _origin_transform(joint.find('origin'))))
    if not result:
        raise ValueError(f'{link_name}: no collision sources')
    return result


def _inertia_snapshot(link):
    return {key: np.asarray(getattr(link, key)).copy() for key in
            ('mass', 'inertial_pos', 'inertial_quat', 'inertia')}


def add_robot_with_exact_sensor_hulls(scene, morph, material, layouts, output_dir):
    """Add a normal rigid entity with five exact source hulls; return entity/report.

    The description API is internal to this vendored Genesis revision. It is the
    same path used by Scene._from_description, and preserves scene registration.
    It fails if a sensor link itself is missing or aligned, rather than guessing
    its coordinate transform. Fixed descendants remain separate source meshes.
    """
    import genesis as gs
    from genesis.engine.entities.rigid_entity.description import RigidEntityDescription

    if scene.is_built or morph.recompute_inertia:
        raise ValueError('Exact sensor hulls require an unbuilt scene and authored inertia')
    solver = scene.rigid_solver
    surface = gs.surfaces.Default(vis_mode='visual', double_sided=False)
    desc = RigidEntityDescription.resolve([morph], material, surface, solver._options,
                                        solver._enable_mujoco_compatibility)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    sensor_names = {x['link_name'] for x in layouts}
    untouched = {link.name: tuple(id(g) for g in link.geoms)
                 for link in desc.links if link.name not in sensor_names}
    for layout in layouts:
        matches = [link for link in desc.links if link.name == layout['link_name']]
        if len(matches) != 1:
            raise ValueError(f"Sensor link missing or merged: {layout['link_name']}")
        link = matches[0]
        if link.is_aligned or not link.geoms:
            raise ValueError(f'{link.name}: expected unaligned authored link with collision geometry')
        components = source_components(morph.file, link.name)
        inertial_before = _inertia_snapshot(link)
        old_geoms = link.geoms
        new_geoms, component_records = [], []
        for source, hull, source_path, source_link in components:
            templates = [g for g in old_geoms if Path(g.mesh.metadata['mesh_path']) == source_path]
            if len(templates) != 1:
                raise ValueError(f'{source_link}: expected one resolved geom for each authored mesh')
            template = templates[0]
            gs_mesh = gs.Mesh.from_trimesh(hull, surface=gs.surfaces.Collision(),
                convexify=False, decimate=False,
                metadata={'mesh_path': str(source_path), 'soft_exact_source_hull': True,
                          'source_sha256': _sha(source_path)})
            new_geoms.append(replace(template, type=gs.GEOM_TYPE.MESH, data=None, mesh=gs_mesh,
                                     pos=np.zeros(3), quat=np.array([1., 0., 0., 0.])))
            component_records.append({'source_link': source_link, 'source_path': str(source_path),
                'source_sha256': _sha(source_path), 'source_bounds_m': source.bounds.tolist(),
                'hull_bounds_m': hull.bounds.tolist(), 'hull_vertices': len(hull.vertices),
                'hull_faces': len(hull.faces), 'watertight': True, 'winding_consistent': True,
                'hull_vertex_source_error_m': 0., 'source_volume_m3': float(source.volume),
                'hull_volume_m3': float(hull.volume)})
        if len(new_geoms) != len(old_geoms):
            raise ValueError(f'{link.name}: replacement would add/remove a source collision component')
        link.geoms = new_geoms
        for key, before in inertial_before.items():
            np.testing.assert_array_equal(getattr(link, key), before)
        # Concatenate components without taking their union hull: that would fill
        # the spaces between separate source shells, introducing another surface.
        source = trimesh.util.concatenate([x[0] for x in components])
        hull = trimesh.util.concatenate([x[1] for x in components])
        probes = np.asarray(layout['positions_link_m'])
        source_dist = trimesh.proximity.closest_point(source, probes)[1]
        hull_dist = trimesh.proximity.closest_point(hull, probes)[1]
        np.savez_compressed(output_dir/f"{layout['finger']}_source_hull.npz",
                            vertices=hull.vertices, faces=hull.faces)
        records.append({'finger': layout['finger'], 'link_name': link.name,
            'source_bounds_m': source.bounds.tolist(), 'hull_bounds_m': hull.bounds.tolist(),
            'hull_vertices': len(hull.vertices), 'hull_faces': len(hull.faces),
            'components': component_records, 'watertight': True, 'winding_consistent': True,
            'probe_to_source_surface_mm_quantiles': (np.quantile(source_dist,[0,.25,.5,.75,1])*1000).tolist(),
            'probe_to_hull_surface_mm_quantiles': (np.quantile(hull_dist,[0,.25,.5,.75,1])*1000).tolist(),
            'old_collision_geom_count': len(old_geoms), 'collision_geom_count': len(new_geoms),
            'resolved_mass_kg': float(link.mass),
            'resolved_inertia_snapshot': {k: v.tolist() for k, v in inertial_before.items()},
            'authored_inertia_resolution_preserved': True,
            'fixed_descendant_geometry_and_inertia_preserved': True})
    assert untouched == {link.name: tuple(id(g) for g in link.geoms)
                         for link in desc.links if link.name not in sensor_names}
    entity = scene.sim._add_entity(desc=desc)
    report = {'method': 'exact convex hull of authored collision STL vertices; no decimation/decomposition',
        'scope': 'five sensor-bearing links plus their fixed source components; soft scene only',
        'exact_source_hulls_enabled': True, 'validated_tactile_coverage_fix': False,
        'source_assets_modified': False, 'global_genesis_modified': False,
        'taxel_layout_or_radius_modified': False, 'other_collision_geoms_unchanged': True,
        'visual_geoms_unchanged': True, 'authored_inertia_preserved': True,
        'genesis_resolved_inertia_preserved': True,
        'ipc_abd_inertia_preserved': False,
        'ipc_inertia_note': 'IPC derives ABD mass/inertia from collision volume times RHO_ROBOT; changing the hull alters that proxy inertia. This diagnostic is not dynamically identical to the baseline.',
        'limitation': 'Convex hull fills source concavities; no expansion outside the source convex envelope.',
        'links': records}
    (output_dir/'source_hull_report.json').write_text(json.dumps(report, indent=2)+'\n')
    return entity, report


def verify_ipc_sensor_hulls(scene, robot, layouts, report, output_dir):
    """Fail if built IPC geometry differs from the intended source hull surface."""
    output_dir = Path(output_dir)
    for layout, record in zip(layouts, report['links']):
        link = robot.get_link(layout['link_name'])
        slots = scene.sim.coupler._abd_slots_by_link[link]
        if len(slots) != 1:
            raise ValueError('Expected exactly one environment for source-hull verification')
        geometry = slots[0].geometry()
        vertices = np.asarray(geometry.positions().view()).reshape(-1, 3).copy()
        faces = np.asarray(geometry.triangles().topo().view()).reshape(-1, 3).copy()
        ref = np.load(output_dir/f"{layout['finger']}_source_hull.npz")
        hull = trimesh.Trimesh(ref['vertices'], ref['faces'], process=False)
        actual = trimesh.Trimesh(vertices, faces, process=False)
        errors = [float(cKDTree(hull.vertices).query(vertices)[0].max()),
                  float(cKDTree(vertices).query(hull.vertices)[0].max())]
        # Genesis imports float32 vertices, then IPC promotes them to float64.
        if max(errors) > 1e-8 or not actual.is_watertight or not actual.is_winding_consistent:
            raise ValueError(f"{layout['finger']}: built IPC surface differs from source hull: {errors}")
        if len(faces) != len(hull.faces) or len(link.geoms) != record['collision_geom_count']:
            raise ValueError(f"{layout['finger']}: topology changed after hull replacement")
        np.testing.assert_allclose(actual.bounds, hull.bounds, rtol=0, atol=1e-8)
        np.savez_compressed(output_dir/f"{layout['finger']}_ipc.npz", vertices=vertices, faces=faces,
            ipc_transform=np.asarray(geometry.transforms().view()).reshape(-1,4,4)[0])
        record['ipc_readback'] = {'verified': True, 'vertices': len(vertices), 'faces': len(faces),
            'bidirectional_vertex_error_m': errors, 'bounds_m': actual.bounds.tolist(),
            'watertight': bool(actual.is_watertight), 'winding_consistent': bool(actual.is_winding_consistent)}
    report['ipc_readback_all_verified'] = True
    (output_dir/'source_hull_report.json').write_text(json.dumps(report, indent=2)+'\n')
