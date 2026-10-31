"""Build the standard 850 mm UR7e + vendor XHAND1 right hand, offline after import."""
from pathlib import Path
import copy
import hashlib
import json
import shutil
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
HARDWARE = Path('/home/sai/zsq/Xhand1交付资料-带触觉/Xhand1交付资料-带触觉')
ASSEMBLY = Path('/home/sai/zsq/TactileTTT/ur5_xhand')
ARM_JOINTS = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
              'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
# Hardware communication IDs 0..11, independent of simulator traversal order.
HAND_JOINTS = ['right_hand_thumb_bend_joint', 'right_hand_thumb_rota_joint1',
               'right_hand_thumb_rota_joint2', 'right_hand_index_bend_joint',
               'right_hand_index_joint1', 'right_hand_index_joint2',
               'right_hand_mid_joint1', 'right_hand_mid_joint2',
               'right_hand_ring_joint1', 'right_hand_ring_joint2',
               'right_hand_pinky_joint1', 'right_hand_pinky_joint2']
FINGERS = {'thumb': 'right_hand_thumb_rota_link2', 'index': 'right_hand_index_rota_link2',
           'middle': 'right_hand_mid_link2', 'ring': 'right_hand_ring_link2',
           'pinky': 'right_hand_pinky_link2'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def import_sources():
    """Copy only referenced assets; never edit delivery files or the existing simulation."""
    src = HERE / 'sources'
    src.mkdir(parents=True, exist_ok=True)
    records = []
    def save(source, target):
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and sha(source) != sha(target):
            raise ValueError(f'Existing imported asset differs: {target}')
        shutil.copy2(source, target)
        records.append({'source': str(source), 'path': str(target.relative_to(HERE)), 'sha256': sha(target)})
    save(ASSEMBLY/'ur5_xhand.urdf', src/'ur5_xhand.urdf')
    model = ET.parse(src/'ur5_xhand.urdf').getroot()
    for link in model.findall('link'):
        if link.get('name').startswith('right_hand') or link.get('name') == 'platform_base_link':
            continue
        for mesh in link.findall('.//mesh'):
            rel = mesh.get('filename')
            save(ASSEMBLY/rel, src/'assembly'/rel)
    archive = HARDWARE/'urdf/Xhand-urdf.zip'
    with zipfile.ZipFile(archive) as z:
        prefix = 'Xhand-urdf/xhand1_right(1)/'
        for name in z.namelist():
            if not name.startswith(prefix) or not name.endswith(('.STL', '.urdf')):
                continue
            rel = Path(name.removeprefix(prefix))
            if '..' in rel.parts:
                raise ValueError('Unsafe archive entry')
            target = src/'xhand'/rel
            target.parent.mkdir(parents=True, exist_ok=True)
            data = z.read(name)
            if target.exists() and target.read_bytes() != data:
                raise ValueError(f'Existing vendor asset differs: {target}')
            target.write_bytes(data)
            records.append({'source': str(archive)+'!'+name, 'path': str(target.relative_to(HERE)), 'sha256': sha(target)})
    tactile = HARDWARE/'触觉传感器'
    for original, new in [('points.t16 (2).json','t16_sensor.json'),
                          ('points.t30 (2).json','t30_sensor.json'),
                          ('points_t16_transformed (2).json','t16_link.json'),
                          ('points_t30_right_hand_transformed.json','t30_right_link.json')]:
        save(tactile/original, src/'tactile'/new)
    official=json.loads((src/'ur7e_official/source.json').read_text())
    for rel in official['files']:
        target=src/'ur7e_official'/Path(rel).name
        records.append({'source':official['repository']+'/blob/'+official['commit']+'/'+rel,
                        'path':str(target.relative_to(HERE)),'sha256':sha(target)})
    (src/'asset_manifest.json').write_text(json.dumps(records, ensure_ascii=False, indent=2)+'\n')


def build():
    src = HERE/'sources'
    if not (src/'asset_manifest.json').exists():
        import_sources()
    for record in json.loads((src/'asset_manifest.json').read_text()):
        if sha(HERE/record['path']) != record['sha256']:
            raise ValueError(f"Imported asset hash mismatch: {record['path']}")
    root = ET.parse(src/'ur5_xhand.urdf').getroot()
    root.set('name', 'ur7e_850_xhand1_right')
    removed = {'world','platform_base_link','Link_1'}
    # The source includes a 0.9 m lab pedestal. Use the scene table as the base instead.
    for node in list(root):
        if node.tag == 'link' and (node.get('name') in removed or node.get('name').startswith('right_hand')):
            root.remove(node)
        elif node.tag == 'joint' and (node.find('parent').get('link') in removed or
                                     node.find('child').get('link') in removed or
                                     node.find('parent').get('link').startswith('right_hand')):
            root.remove(node)
        elif node.tag not in ('link','joint','material'):
            root.remove(node)
    for mesh in root.findall('.//mesh'):
        mesh.set('filename', str((src/'assembly'/mesh.get('filename')).resolve()))
    # The supplied adapter OBJ is already a single convex hull, enclosing empty space
    # occupied by the thumb. Decompose the original detailed STL instead of disabling collision.
    out = HERE/'generated'
    out.mkdir(exist_ok=True)
    adapter_source = src/'assembly/Flange_meshes/coupling_D405_m.stl'
    adapter_dir = out/('adapter_components_'+sha(adapter_source)[:12])
    if not (adapter_dir/'manifest.json').exists():
        import trimesh
        mesh=trimesh.load(adapter_source,force='mesh')
        # The detailed asset consists of three disconnected manufactured components.
        # Keep their separation, using one convex hull per component; do not bridge the gap.
        hulls=[part.convex_hull for part in mesh.split(only_watertight=True)]
        assert len(hulls)==3, 'Unexpected adapter source topology'
        adapter_dir.mkdir(exist_ok=True)
        for i,hull in enumerate(hulls):
            hull.export(adapter_dir/f'part_{i:03}.obj')
        (adapter_dir/'manifest.json').write_text(json.dumps({'source_sha256':sha(adapter_source),
            'method':'convex hull per disconnected STL component','hulls':len(hulls),
            'limitation':'Small holes/concavities within each component are still convex approximations'},indent=2)+'\n')
    adapter=root.find("link[@name='Flange_base_link']")
    for collision in list(adapter.findall('collision')):
        adapter.remove(collision)
    for part in sorted(adapter_dir.glob('part_*.obj')):
        collision=ET.SubElement(adapter,'collision')
        ET.SubElement(collision,'origin',xyz='0 0 0',rpy='0 0 0')
        geometry=ET.SubElement(collision,'geometry')
        ET.SubElement(geometry,'mesh',filename=str(part.resolve()))
    vendor = ET.parse(src/'xhand/urdf/xhand_right.urdf').getroot()
    for node in vendor:
        if node.tag not in ('link','joint','material'):
            continue
        node = copy.deepcopy(node)
        for mesh in node.findall('.//mesh'):
            mesh.set('filename', str((src/'xhand/meshes'/Path(mesh.get('filename')).name).resolve()))
        root.append(node)
    # Verify the arm's full six fixed transforms against the pinned official UR7e definition.
    kin = yaml.safe_load((src/'ur7e_official/default_kinematics.yaml').read_text())['kinematics']
    for joint, key in zip(ARM_JOINTS, ['shoulder','upper_arm','forearm','wrist_1','wrist_2','wrist_3']):
        origin = root.find(f"joint[@name='{joint}']/origin")
        np.testing.assert_allclose([float(v) for v in origin.get('xyz').split()],
                                   [kin[key][a] for a in ['x','y','z']], atol=1e-10)
        np.testing.assert_allclose([float(v) for v in origin.get('rpy').split()],
                                   [kin[key][a] for a in ['roll','pitch','yaw']], atol=1e-10)
    moving = {j.get('name') for j in root.findall('joint') if j.get('type') != 'fixed'}
    assert moving == set(ARM_JOINTS + HAND_JOINTS)
    links = {l.get('name') for l in root.findall('link')}
    children = {j.find('child').get('link') for j in root.findall('joint')}
    assert links - children == {'base_link'}
    for mesh in root.findall('.//mesh'):
        assert Path(mesh.get('filename')).is_file(), mesh.get('filename')
    ET.indent(root)
    path = out/'ur7e_xhand.urdf'
    ET.ElementTree(root).write(path, encoding='unicode')
    layouts = []
    for finger, link in FINGERS.items():
        thumb = finger == 'thumb'
        local_file = src/'tactile'/('t30_right_link.json' if thumb else 't16_link.json')
        raw_file = src/'tactile'/('t30_sensor.json' if thumb else 't16_sensor.json')
        def points(p):
            rows = sorted(json.loads(p.read_text())['measurement_points'],key=lambda r:r['point'])
            assert [r['point'] for r in rows] == list(range(1,121))
            return np.array([[r[a] for a in ['x','y','z']] for r in rows])
        raw, local = points(raw_file), points(local_file)
        rotation = np.eye(3) if thumb else np.array([[0,0,1],[1,0,0],[0,1,0]])
        translation = np.array([0,16.42,9.41] if thumb else [5.35,0,15.86])
        np.testing.assert_allclose(raw @ rotation.T + translation, local, atol=1e-8)
        layouts.append({'finger':finger,'link_name':link,'taxels':120,'point_ids':list(range(1,121)),
                        'positions_link_m':(local/1000).tolist(),
                        'sensor_to_link_rotation':rotation.tolist(),
                        'source':str(local_file.relative_to(HERE))})
    (out/'tactile_layout.json').write_text(json.dumps(layouts,indent=2)+'\n')
    return path, layouts


if __name__ == '__main__':
    print(build()[0])
