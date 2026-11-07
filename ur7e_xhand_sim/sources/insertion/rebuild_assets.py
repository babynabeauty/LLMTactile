"""Offline asset conversion; never imported by the simulator.

Usage: python rebuild_assets.py --univtac-root PATH --contactworld-release PATH
Optional --tube-npz-dir uses an existing lossless USD transform export instead
of requiring pxr in the conversion environment. Runtime needs only the files
already packaged beside this script.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET

import numpy as np
import trimesh


HERE = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def commit(root):
    return subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()


def write_mesh(mesh, path):
    mesh.export(path, file_type='obj', include_color=False, include_texture=False)


def write_tube_visual(raw, path):
    """Retain USD face-varying corner normals and unwelded visual vertices."""
    vertices = raw['vertices'][raw['faces'].reshape(-1)]
    normals = raw['normals']
    assert vertices.shape == normals.shape
    lines = ['# Original USD face-varying normals; visual-only, no welding.']
    lines += ['v ' + ' '.join(format(float(v), '.17g') for v in row) for row in vertices]
    lines += ['vn ' + ' '.join(format(float(v), '.17g') for v in row) for row in normals]
    lines += [f'f {i}//{i} {i+1}//{i+1} {i+2}//{i+2}' for i in range(1, len(vertices)+1, 3)]
    path.write_text('\n'.join(lines)+'\n')


def mesh_properties(mesh):
    return {'bounds_m': mesh.bounds.tolist(), 'bounds_size_m': mesh.extents.tolist(),
            'volume_m3': float(mesh.volume), 'center_of_mass_m': mesh.center_mass.tolist(),
            'inertia_per_kg': (mesh.moment_inertia / mesh.volume).tolist()}


def collision_bounds_from_urdf(path):
    """Measure all authored collision parts in the single link's local frame."""
    root = ET.parse(path).getroot()
    assert len(root.findall('link')) == 1
    vertices = []
    for collision in root.findall('link/collision'):
        geometry = collision.find('geometry')
        if geometry.find('mesh') is not None:
            node = geometry.find('mesh')
            mesh = trimesh.load_mesh(path.parent / node.get('filename'), process=False)
            mesh.apply_scale(np.fromstring(node.get('scale', '1 1 1'), sep=' '))
        else:
            mesh = trimesh.creation.box(np.fromstring(geometry.find('box').get('size'), sep=' '))
        origin = collision.find('origin')
        if origin is not None:
            transform = trimesh.transformations.euler_matrix(*np.fromstring(origin.get('rpy', '0 0 0'), sep=' '))
            transform[:3, 3] = np.fromstring(origin.get('xyz', '0 0 0'), sep=' ')
            mesh.apply_transform(transform)
        vertices.append(mesh.vertices)
    vertices = np.concatenate(vertices)
    bounds = np.stack([vertices.min(axis=0), vertices.max(axis=0)])
    return {'collision_bounds_m': bounds.tolist(),
            'collision_bounds_size_m': np.diff(bounds, axis=0)[0].tolist()}


def urdf(name, visual, color, collisions, mass, properties):
    robot = ET.Element('robot', name=name)
    link = ET.SubElement(robot, 'link', name=name)
    inertial = ET.SubElement(link, 'inertial')
    ET.SubElement(inertial, 'origin', xyz=' '.join(map(str, properties['center_of_mass_m'])))
    ET.SubElement(inertial, 'mass', value=str(mass))
    inertia = np.asarray(properties['inertia_per_kg']) * mass
    ET.SubElement(inertial, 'inertia', **{key: str(inertia[i, j]) for key, i, j in
                  [('ixx', 0, 0), ('ixy', 0, 1), ('ixz', 0, 2), ('iyy', 1, 1), ('iyz', 1, 2), ('izz', 2, 2)]})
    vis = ET.SubElement(link, 'visual')
    ET.SubElement(ET.SubElement(vis, 'geometry'), 'mesh', filename=visual)
    ET.SubElement(ET.SubElement(vis, 'material', name=name+'_original_color'), 'color', rgba=' '.join(map(str, color)))
    for collision in collisions:
        node = ET.SubElement(link, 'collision')
        geometry = ET.SubElement(node, 'geometry')
        if 'mesh' in collision:
            ET.SubElement(geometry, 'mesh', filename=collision['mesh'])
        else:
            ET.SubElement(node, 'origin', xyz=' '.join(map(str, collision['center'])))
            ET.SubElement(geometry, 'box', size=' '.join(map(str, collision['size'])))
    ET.indent(robot)
    ET.ElementTree(robot).write(HERE / (name+'.urdf'), encoding='unicode')


def tube_mesh(root, name, npz_dir):
    if npz_dir:
        data = np.load(npz_dir / (name+'.npz'))
        raw = {key: data[key] for key in ('vertices', 'faces', 'normals')}
        return trimesh.Trimesh(data['vertices'], data['faces'], process=True), data['color'].tolist(), raw
    from pxr import Usd, UsdGeom, UsdShade
    stage = Usd.Stage.Open(str(root / 'assets/objects' / (name+'.usd')))
    meshes = [p for p in stage.Traverse() if p.IsA(UsdGeom.Mesh)]
    assert len(meshes) == 1
    prim = meshes[0]
    mesh = UsdGeom.Mesh(prim)
    vertices = np.array(mesh.GetPointsAttr().Get(), dtype=float)
    mat = np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
    vertices = (np.column_stack([vertices, np.ones(len(vertices))]) @ mat)[:, :3]
    assert np.all(np.array(mesh.GetFaceVertexCountsAttr().Get()) == 3)
    faces = np.array(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)
    assert mesh.GetNormalsInterpolation() == 'faceVarying'
    normals = np.asarray(mesh.GetNormalsAttr().Get(), dtype=float) @ np.linalg.inv(mat[:3, :3]).T
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    material = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]
    shader = next(p for p in material.GetPrim().GetChildren() if p.IsA(UsdShade.Shader))
    color = list(shader.GetAttribute('inputs:base_color_factor').Get())
    return trimesh.Trimesh(vertices, faces, process=True), color, {'vertices': vertices, 'faces': faces, 'normals': normals}


def build(args):
    manifest = {'schema': 'ur7e_xhand_insertion_assets_v1', 'units': 'm', 'scale': 1.0,
                'sources': {}, 'tasks': {}, 'packaged_sha256': {}}
    tube_records = {}
    for name, role in [('TestTube', 'tube'), ('TestTubeBase', 'holder'), ('TestTubeSlot', 'socket')]:
        mesh, color, raw = tube_mesh(args.univtac_root, name, args.tube_npz_dir)
        assert mesh.is_watertight and mesh.is_winding_consistent
        filename = name+'.obj'
        write_mesh(mesh, HERE / filename)
        visual_filename = name+'_visual.obj'
        write_tube_visual(raw, HERE / visual_filename)
        props = mesh_properties(mesh)
        convex = role == 'tube'
        record = {'name': role, 'role': role, 'urdf_path': name+'.urdf',
                  'visual_mesh_path': visual_filename, 'collision_mesh_path': filename,
                  **props, 'mass_kg': .03 if convex else 1., 'color_rgba': color+[1.],
                  'morph_options': {'align': False, 'convexify': convex, 'decimate': False, 'watertighten': None},
                  'material_options': {} if convex else {'sdf_cell_size': .0005, 'sdf_max_res': 256},
                  'collision_description': 'Original transformed mesh, unchanged closed surface; '+
                    ('convex tube.' if convex else 'non-convex SDF retains original hole; no hull or wrap.')}
        urdf(name, visual_filename, record['color_rgba'], [{'mesh': filename}], record['mass_kg'], props)
        tube_records[role] = record
    manifest['tasks']['tube_insert'] = {
        'source_key': 'univtac', 'movable': tube_records['tube'],
        'fixed': [tube_records['holder'], tube_records['socket']],
        'target': {'opening_center_local_m': [-.00781781, 0, .07650077],
                   'axis_local': [-.5, 0., .8660254037844386],
                   'object_quat_wxyz': [.9659258262890683, 0., -.25881904510252074, 0.],
                   'tip_local_m': [0., 0., 0.], 'hole_diameter_m': .021,
                   'reference_insertion_depth_m': .04,
                   'opening_definition': 'Exact sloped mesh mouth center; source policy uses approximate [-.008,0,.077].'},
        'initial_holder_tip_local_m': [0., 0., .002],
    }
    source_mesh = args.contactworld_release / 'industreal/mesh/TVB_mesh'
    usb_records = {}
    for role, color in [('plug', [.3, .3, 1., 1.]), ('socket', [1., .3, .3, 1.])]:
        original = source_mesh / ('manifeel_usb_'+role+'.obj')
        # Only remove external material references; all source vertices, normals,
        # face topology and coordinates are kept verbatim for the visible model.
        visual = 'USB_'+role+'_visual.obj'
        text = original.read_text()
        (HERE / visual).write_text('\n'.join(line for line in text.splitlines()
            if not line.startswith(('mtllib ', 'usemtl ')))+'\n')
        mesh = trimesh.load_mesh(original, process=False)
        mesh.merge_vertices(merge_tex=True, merge_norm=True)
        if role == 'socket':
            assert mesh.is_watertight and mesh.is_winding_consistent
            collision = 'USB_socket_collision.obj'
            write_mesh(mesh, HERE / collision)
            props = mesh_properties(mesh)
            collisions = [{'mesh': collision}]
            morph = {'align': False, 'convexify': False, 'decimate': False, 'watertighten': None}
            material = {'sdf_cell_size': .00025, 'sdf_max_res': 320}
            description = 'Original watertight socket surface with seam vertices welded. Non-convex SDF preserves rectangular hole and tongue.'
        else:
            # The source plug has open surfaces and degenerate duplicates. Use
            # explicit closed collision solids so its insertion cavity survives.
            # Exclude source connector internals that extend through the handle;
            # original handle is the largest connected component after welding.
            body = max(mesh.split(only_watertight=False), key=lambda part: len(part.faces)).convex_hull
            body.vertices[:, 2] = np.maximum(body.vertices[:, 2], .022)
            body = body.convex_hull
            collision = 'USB_plug_body_collision.obj'
            write_mesh(body, HERE / collision)
            collisions = [{'mesh': collision}]
            parts = [body]
            # Metal shell: 6 x 22 mm outer, 4.8 x 20.8 mm inner, 22 mm long.
            # Its original latch cutouts and flexible tabs are intentionally not
            # modeled by these rigid boxes; the insertion-facing cavity is kept.
            boxes = [([-.0027, 0., .011], [.0006, .022, .022]),
                     ([.0027, 0., .011], [.0006, .022, .022]),
                     ([0., -.0107, .011], [.0048, .0006, .022]),
                     ([0., .0107, .011], [.0048, .0006, .022]),
                     ([-.0014, .0004, .011], [.002, .020, .022])]
            for center, size in boxes:
                collisions.append({'center': center, 'size': size})
                part = trimesh.creation.box(extents=size)
                part.apply_translation(center)
                parts.append(part)
            props = mesh_properties(trimesh.util.concatenate(parts))
            # Preserve the visual extent for scene placement and reporting.
            props.update(bounds_m=mesh.bounds.tolist(), bounds_size_m=mesh.extents.tolist())
            morph = {'align': False, 'convexify': True, 'decimate': False, 'watertighten': None}
            material = {'sdf_cell_size': .00025, 'sdf_max_res': 256}
            description = ('Closed proxy: original handle convex hull cropped at z=22mm, four 0.6mm metal shell walls, '
                           'and internal plastic box. Source open surfaces, latch slots, handle cavity and flexible tabs '
                           'are simplified. External visual is unchanged; connector cavity and keyed side are retained. '
                           'Mass/inertia use the closed proxy solids, with an assumed total 50g mass.')
        record = {'name': 'usb' if role == 'plug' else 'socket', 'role': role,
                  'urdf_path': 'USB_'+role+'.urdf', 'visual_mesh_path': visual,
                  'collision_mesh_path': collision, **props, 'mass_kg': .05 if role == 'plug' else 1.,
                  'color_rgba': color, 'morph_options': morph, 'material_options': material,
                  'collision_description': description}
        if role == 'plug':
            record['collision_primitives'] = collisions
        urdf('USB_'+role, visual, color, collisions, record['mass_kg'], props)
        usb_records[role] = record
    manifest['tasks']['usb_insert'] = {
        'source_key': 'contactworld', 'movable': usb_records['plug'], 'fixed': [usb_records['socket']],
        'target': {'opening_center_local_m': [0., 0., .018], 'axis_local': [0., 0., 1.],
                   'object_quat_wxyz': [0., 0., 0., 1.], 'tip_local_m': [0., 0., 0.],
                   'opening_size_xy_m': [.00918, .0234], 'reference_insertion_depth_m': .017,
                   'keying': '180 degree local-Z rotation places plug plastic on +X, clear of socket tongue on -X.'},
    }
    tube_files = [args.univtac_root/'assets/objects'/(n+'.usd') for n in ['TestTube', 'TestTubeBase', 'TestTubeSlot']]
    tube_files += [args.univtac_root/'envs/insert_tube.py', args.univtac_root/'LICENSE']
    shutil.copyfile(args.univtac_root/'LICENSE', HERE/'LICENSE-UniVTAC.txt')
    manifest['sources']['univtac'] = {
        'project_url': 'https://github.com/univtac/UniVTAC', 'commit': commit(args.univtac_root),
        'sha256': {str(p.relative_to(args.univtac_root)): digest(p) for p in tube_files},
        'license': 'Repository root LICENSE is Apache-2.0; copied as LICENSE-UniVTAC.txt. README says MIT, an upstream inconsistency.',
        'conversion': 'USD mesh vertices transformed through complete local-to-world Xform at default time; metersPerUnit=1; no rescaling. Visual OBJ retains original face-varying corner normals and unwelded vertices; collision OBJ is separately welded. Original base_color_factor retained.',
        'mass_note': '30g tube mass is a local simulation assumption; source density=10 kg/m3 is not reused.'}
    usb_files = [source_mesh/('manifeel_usb_'+r+'.obj') for r in ['plug','socket']]
    usb_files += [args.contactworld_release/'industreal/urdf'/('USB_'+r+'.urdf') for r in ['plug','socket']]
    manifest['sources']['contactworld'] = {
        'project_url': 'https://github.com/PokuangZhou/ContactWorld',
        'asset_release_url': 'https://huggingface.co/datasets/Pokuang/ContactWorld',
        'sha256': {str(p.relative_to(args.contactworld_release)): digest(p) for p in usb_files},
        'license': 'No root LICENSE found in the inspected ContactWorld checkout; released asset license not established. No license grant is inferred here.',
        'conversion': 'Meter-scale original OBJ geometry, unchanged visible vertices/faces; external MTL references removed in favor of original URDF solid colors. Socket collision only welds identical seam vertices.',
        'mass_note': '50g plug mass is a local simulation assumption; source has no explicit inertial.'}
    for task in manifest['tasks'].values():
        for asset in [task['movable'], *task['fixed']]:
            asset.update(collision_bounds_from_urdf(HERE / asset['urdf_path']))
    for path in sorted(HERE.iterdir()):
        if path.suffix in ('.obj', '.urdf', '.txt'):
            manifest['packaged_sha256'][path.name] = digest(path)
    (HERE/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps({task: {'size_m': d['movable']['bounds_size_m'], 'mass_kg': d['movable']['mass_kg']}
                      for task, d in manifest['tasks'].items()}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--univtac-root', type=Path, required=True)
    parser.add_argument('--contactworld-release', type=Path, required=True)
    parser.add_argument('--tube-npz-dir', type=Path)
    build(parser.parse_args())
