"""Compose licensed Panda arm and upstream Wuji hand into one MJCF chain."""
from pathlib import Path
import copy
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def build_model():
    panda = HERE / 'vendor/franka_emika_panda/panda.xml'
    wuji = ROOT / 'dexterous-hands/src/assets/hands/wuji/mjcf/right.xml'
    arm, hand = ET.parse(panda).getroot(), ET.parse(wuji).getroot()
    arm.set('model', 'panda_wuji_27dof')
    for root, source in [(arm, panda), (hand, wuji)]:
        meshdir = source.parent / root.find('compiler').get('meshdir', '')
        for mesh in root.findall('asset/mesh'):
            if 'name' not in mesh.attrib:
                mesh.set('name', Path(mesh.get('file')).stem)
            mesh.set('file', str((meshdir / mesh.get('file')).resolve()))
        root.find('compiler').set('meshdir', '')
    # Removing the gripper also removes its joints: exactly 7 arm + 20 hand DOFs.
    link7 = arm.find(".//body[@name='link7']")
    link7.remove(link7.find("body[@name='hand']"))
    for tag in ['tendon', 'equality', 'keyframe']:
        element = arm.find(tag)
        if element is not None:
            arm.remove(element)
    actuators = arm.find('actuator')
    for actuator in list(actuators):
        if actuator.get('joint') not in [f'joint{i}' for i in range(1, 8)]:
            actuators.remove(actuator)
    # Explicit hand defaults prevent inheritance of Panda's z-axis joints.
    defaults = ET.SubElement(arm.find('default'), 'default', {'class': 'wuji'})
    ET.SubElement(defaults, 'joint', {'armature': '0.0002', 'damping': '0', 'axis': '1 0 0'})
    wrist = copy.deepcopy(hand.find("worldbody/body[@name='r_wrist']"))
    wrist.set('childclass', 'wuji')
    wrist.set('pos', '0 0 0.107')
    wrist.set('quat', '0 0 1 0')  # hand -Z fingers extend along flange +Z
    link7.append(wrist)
    # A visible fixed adapter, with explicit mass/inertia, between flange and wrist.
    adapter = ET.SubElement(link7, 'body', {'name':'wuji_adapter', 'pos':'0 0 0.065'})
    ET.SubElement(adapter,'inertial',{'pos':'0 0 0','mass':'0.08','diaginertia':'0.00003 0.00003 0.00003'})
    ET.SubElement(adapter,'geom',{'type':'cylinder','size':'0.025 0.035','rgba':'0.15 0.25 0.35 1'})
    for element in hand.find('asset'):
        arm.find('asset').append(copy.deepcopy(element))
    for element in hand.find('actuator'):
        actuators.append(copy.deepcopy(element))
    contact = arm.find('contact')
    for element in hand.find('contact'):
        contact.append(copy.deepcopy(element))
    ET.SubElement(contact,'exclude',{'body1':'link7','body2':'r_wrist'})
    ET.SubElement(contact,'exclude',{'body1':'wuji_adapter','body2':'r_wrist'})
    output = HERE / 'panda_wuji.xml'
    ET.indent(arm, space='  ')
    ET.ElementTree(arm).write(output, encoding='unicode')
    return output


if __name__ == '__main__':
    print(build_model())
