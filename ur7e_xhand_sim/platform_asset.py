"""Imported lab platform; retain the policy's existing tabletop coordinate frame."""
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent
TABLE_TOP_Z = .2
FLOOR_Z = -.7
PLATFORM_POSITION = (-.1, 0., FLOOR_Z)
# Tabletop extent in world XY: the 1.2 x 0.8 m CAD body box, centered at (-.12, -.27).
TABLETOP_XY_M = ((-.72, .48), (-.67, .13))


def build_platform():
    source = HERE/'sources/platform'
    manifest = json.loads((source/'manifest.json').read_text())
    for name, digest in manifest['sha256'].items():
        if hashlib.sha256((source/name).read_bytes()).hexdigest() != digest:
            raise ValueError(f'Platform source hash mismatch: {name}')
    root = ET.Element('robot', name='imported_lab_platform')
    link = ET.SubElement(root, 'link', name='platform_base_link')
    visual = ET.SubElement(link, 'visual')
    # Source CAD uses Y down. Snap its rounded -1.57 rotations to exact quarters.
    ET.SubElement(visual, 'origin', xyz='-0.02 0.03 0', rpy='-1.5707963267948966 0 -1.5707963267948966')
    ET.SubElement(ET.SubElement(visual, 'geometry'), 'mesh', filename=str(source/'base_link.STL'))
    ET.SubElement(ET.SubElement(visual, 'material', name='platform_white'), 'color', rgba='1 1 1 1')
    # Exact union of CAD solids: a 0.8 m body and four 0.1 m feet.
    boxes = [((-.02,-.27,.5),(1.2,.8,.8))]
    boxes += [((x,y,.05),(.1,.1,.1)) for x in (-.57,.53) for y in (-.62,.08)]
    for center, size in boxes:
        collision = ET.SubElement(link, 'collision')
        ET.SubElement(collision, 'origin', xyz=' '.join(map(str,center)))
        ET.SubElement(ET.SubElement(collision,'geometry'),'box',size=' '.join(map(str,size)))
    out = HERE/'generated/platform.urdf'
    out.parent.mkdir(exist_ok=True)
    ET.indent(root)
    ET.ElementTree(root).write(out,encoding='unicode')
    return out, {**manifest, 'world_position_m':list(PLATFORM_POSITION),
        'table_top_z_m':TABLE_TOP_Z, 'floor_z_m':FLOOR_Z,
        'collision':'exact five-box union of source CAD body and feet',
        'rotation_adjustment':'rounded -1.57 rad snapped to -pi/2 for a level tabletop',
        'bounds_world_m':[[-.72,-.67,-.7],[.48,.13,.2]]}
