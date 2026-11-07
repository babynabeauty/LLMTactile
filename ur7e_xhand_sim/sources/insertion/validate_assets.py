"""Check packaged meshes, portable URDF paths, and insertion clearances on CPU.

This checks geometry only. It is not a successful grasp/insert rollout.
"""
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from insertion_assets import INSERTION_TASKS, load_insertion_assets


def load_mesh(path):
    mesh = trimesh.load_mesh(path, process=False)
    mesh.merge_vertices(merge_tex=True, merge_norm=True)
    return mesh


def parts(asset):
    result = []
    root = ET.parse(asset['urdf_path']).getroot()
    assert len(root.findall('link')) == 1
    for collision in root.findall('link/collision'):
        geometry = collision.find('geometry')
        if geometry.find('mesh') is not None:
            node = geometry.find('mesh')
            path = Path(node.get('filename'))
            if not path.is_absolute():
                path = Path(asset['urdf_path']).parent / path
            mesh = load_mesh(path)
            mesh.apply_scale(np.fromstring(node.get('scale', '1 1 1'), sep=' '))
        else:
            mesh = trimesh.creation.box(np.fromstring(geometry.find('box').get('size'), sep=' '))
        origin = collision.find('origin')
        if origin is not None:
            transform = np.eye(4)
            transform[:3, :3] = Rotation.from_euler('xyz', np.fromstring(origin.get('rpy', '0 0 0'), sep=' ')).as_matrix()
            transform[:3, 3] = np.fromstring(origin.get('xyz', '0 0 0'), sep=' ')
            mesh.apply_transform(transform)
        assert mesh.is_watertight and mesh.is_winding_consistent
        result.append(mesh)
    return result


def validate():
    report = {'kind': 'static geometry verification, not a physics rollout', 'tasks': {}}
    for task in INSERTION_TASKS:
        data = load_insertion_assets(task)
        for asset in [data['movable'], *data['fixed']]:
            visual = load_mesh(asset['visual_mesh_path'])
            np.testing.assert_allclose(visual.bounds, asset['bounds_m'], atol=1e-8)
            assert np.all(np.linalg.eigvalsh(asset['inertia_per_kg']) > 0)
            root = ET.parse(asset['urdf_path']).getroot()
            for node in root.findall('.//mesh'):
                path = Path(node.get('filename'))
                assert not path.is_absolute(), 'URDF meshes must be portable relative paths'
                assert (Path(asset['urdf_path']).parent/path).is_file()
            assert float(root.find('link/inertial/mass').get('value')) == asset['mass_kg']
            collision_parts = parts(asset)
            for part in collision_parts:
                assert part.volume > 0
            vertices = np.concatenate([part.vertices for part in collision_parts])
            collision_bounds = np.stack([vertices.min(axis=0), vertices.max(axis=0)])
            np.testing.assert_allclose(collision_bounds, asset['collision_bounds_m'], atol=1e-9, rtol=0)
            np.testing.assert_allclose(np.diff(collision_bounds, axis=0)[0], asset['collision_bounds_size_m'], atol=1e-9, rtol=0)
        socket = load_mesh(next(a for a in data['fixed'] if a['name'] == 'socket')['collision_mesh_path'])
        target = data['target']
        tip = np.array(target['opening_center_local_m']) - np.array(target['axis_local']) * target['reference_insertion_depth_m']
        quat = np.array(target['object_quat_wxyz'])
        rotation = Rotation.from_quat(quat[[1, 2, 3, 0]]).as_matrix()
        samples = [np.vstack([part.vertices, trimesh.sample.sample_surface(part, 4000, seed=42)[0]])
                   for part in parts(data['movable'])]
        penetrations = [int(socket.contains(points @ rotation.T + tip).sum()) for points in samples]
        assert not any(penetrations), (task, penetrations)
        result = {'reference_depth_m': target['reference_insertion_depth_m'],
                  'visual_bounds_size_m': data['movable']['bounds_size_m'],
                  'collision_bounds_size_m': data['movable']['collision_bounds_size_m'],
                  'reference_tip_local_m': tip.tolist(), 'samples_per_collision_part': [len(p) for p in samples],
                  'penetrating_samples_per_part': penetrations}
        if task == 'usb_insert':
            wrong = int(socket.contains(np.concatenate(samples) + tip).sum())
            assert wrong > 0, 'Keying negative control must collide with the original socket tongue'
            probes = np.array([[0., 0., .01], [-.001, 0., .01], [.0037, 0., .01], [.01, 0., .01], [0., 0., -.005]])
            expected = [False, True, False, True, True]  # channel, tongue, channel, wall, floor
            assert socket.contains(probes).tolist() == expected
            result['wrong_orientation_penetrating_samples'] = wrong
            result['socket_channel_tongue_wall_floor_check'] = 'passed'
        report['tasks'][task] = result
    return report


if __name__ == '__main__':
    print(json.dumps(validate(), indent=2))
