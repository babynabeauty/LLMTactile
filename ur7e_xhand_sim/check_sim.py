"""Bounded asset, control and five-finger contact acceptance; no policy training."""
import argparse
import json
import math
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import numpy as np
import trimesh
import yaml
from scipy.spatial.transform import Rotation

from environment import Sandbox
from build_assets import HERE, ARM_JOINTS, HAND_JOINTS, build, sha


def geometry_audit(model, layouts):
    root=ET.parse(model).getroot()
    class DegreeLoader(yaml.SafeLoader):
        pass
    DegreeLoader.add_constructor('!degrees', lambda loader,node: math.radians(float(loader.construct_scalar(node))))
    limits=yaml.load((HERE/'sources/ur7e_official/joint_limits.yaml').read_text(),Loader=DegreeLoader)['joint_limits']
    for name in ARM_JOINTS:
        limit=root.find(f"joint[@name='{name}']/limit")
        for local,official in [('lower','min_position'),('upper','max_position'),('effort','max_effort'),('velocity','max_velocity')]:
            np.testing.assert_allclose(float(limit.get(local)),limits[name][official])
    physical=yaml.safe_load((HERE/'sources/ur7e_official/physical_parameters.yaml').read_text())['inertia_parameters']
    for key in ['shoulder','upper_arm','forearm','wrist_1','wrist_2','wrist_3']:
        name='upper_arm_link' if key=='upper_arm' else key+'_link'
        node=root.find(f"link[@name='{name}']/inertial")
        np.testing.assert_allclose(float(node.find('mass').get('value')),physical[key+'_mass'])
        for k,v in physical['tensor'][key].items():
            np.testing.assert_allclose(float(node.find('inertia').get(k)),v)
    audit={}
    for layout in layouts:
        link=layout['link_name'];parts=[]
        nodes=[(root.find(f"link[@name='{link}']"),np.eye(4))]
        for j in root.findall('joint'):
            if j.get('type')=='fixed' and j.find('parent').get('link')==link:
                transform=np.eye(4)
                transform[:3,3]=[float(v) for v in j.find('origin').get('xyz').split()]
                transform[:3,:3]=Rotation.from_euler('xyz',[float(v) for v in j.find('origin').get('rpy').split()]).as_matrix()
                nodes.append((root.find(f"link[@name='{j.find('child').get('link')}']"),transform))
        for node,transform in nodes:
            for collision in node.findall('collision'):
                m=trimesh.load(collision.find('geometry/mesh').get('filename'),force='mesh')
                m.apply_transform(transform)
                parts.append(m)
        mesh=trimesh.util.concatenate(parts)
        points=np.array(layout['positions_link_m'])
        nearest,distance,tri=trimesh.proximity.closest_point(mesh,points)
        audit[layout['finger']]={'normal_method':'nearest triangle of original vendor mesh, test fixture only'}
        audit[layout['finger']]['distance_to_vendor_surface_mm']=(1000*np.quantile(distance,[0,.5,1])).tolist()
        assert distance.max()<.002, (layout['finger'],distance.max())
        # Press at the exposed front rather than the edge of the sensor.
        axis=2 if layout['finger']=='thumb' else 0
        candidates=np.flatnonzero(points[:,axis]>points[:,axis].max()-.001)
        idx=int(candidates[np.argmin(np.linalg.norm(points[candidates]-np.median(points,axis=0),axis=1))])
        normal=mesh.face_normals[tri[idx]].copy()
        if normal[axis]<0:normal=-normal
        audit[layout['finger']].update(test_point_id=idx+1,test_point_link_m=points[idx].tolist(),
                                      test_normal_link=normal.tolist())
    return audit


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    out=Path(args.output).resolve()
    out.mkdir(parents=True,exist_ok=False)
    model,layouts=build()
    report={'model_sha256':sha(model),
            'code_sha256':{p.name:sha(p) for p in HERE.glob('*.py')},
            'source_manifest_sha256':sha(HERE/'sources/asset_manifest.json'),
            'geometry':geometry_audit(model,layouts),
            'scope':'Model and sensor-interface acceptance; not force calibration, grasp success, or sim-to-real.'}
    sim=Sandbox(out,render=True,test_fixture=True)
    try:
        description=sim.describe()
        (out/'description.json').write_text(json.dumps(description,indent=2)+'\n')
        assert description['joint_order']==ARM_JOINTS+HAND_JOINTS
        assert sim.robot.n_dofs==18
        initial=sim.observe(full=True)
        assert not {'cube_position_m','solver_contacts','reward','success'}.intersection(initial)
        assert sum(s['active_taxels'] for s in initial['tactile'].values())==0
        state=sim.status();time.sleep(.1);assert sim.status()==state
        for payload in [{'actions':{'missing':0}},{'actions':{'shoulder_pan_joint':1.1}},
                        {'steps':0},{'steps':True},{'expected_episode':-20}]:
            try:sim.step(payload)
            except ValueError:pass
            else:raise AssertionError('Invalid input accepted')
            assert sim.status()==state
        sim.render_enabled=False
        start=time.perf_counter()
        for _ in range(20):
            obs=sim.step({'steps':10})
            assert np.isfinite(list(obs['joint_positions_rad'].values())).all()
            assert not obs['terminated']
        report['200_step_wall_s']=time.perf_counter()-start
        report['settled_wrist_drift_m']=float(np.linalg.norm(np.array(obs['wrist_position_m'])-initial['wrist_position_m']))
        report['settled_contacts']=sim.evaluate()['solver_contacts']
        assert report['settled_wrist_drift_m']<.015
        # A collision-free initial posture is necessary before contact tests.
        assert not report['settled_contacts'], report['settled_contacts']
        z=obs['wrist_position_m'][2]
        obs=sim.move({'delta':[0,0,.02],'steps':10})
        report['wrist_up_m']=obs['wrist_position_m'][2]-z
        assert report['wrist_up_m']>.005
        before=np.array(list(obs['joint_positions_rad'].values()))
        obs=sim.hand({'closure':.35,'steps':10})
        report['hand_joint_motion_rad']=float(np.max(abs(np.array(list(obs['joint_positions_rad'].values()))[6:]-before[6:])))
        assert report['hand_joint_motion_rad']>.05
        report['finger_press']={}
        for layout in layouts:
            sim.reset()
            for _ in range(10):sim.step({'steps':10})
            name=layout['finger'];a=report['geometry'][name]
            link=sim.robot.get_link(layout['link_name'])
            quat=sim.array(link.get_quat()).reshape(4)
            rotation=Rotation.from_quat(quat[[1,2,3,0]]).as_matrix()
            point=sim.array(link.get_pos()).reshape(3)+rotation@np.array(a['test_point_link_m'])
            normal=rotation@np.array(a['test_normal_link'])
            samples=[]
            for offset in [.010,.006,.005,.0045,.010]:
                sim.fixture.set_pos(point+normal*offset)
                sim.step({'steps':1})
                tactile=sim.observe(full=True)['tactile'][name]
                contacts=sim.robot.get_contacts(with_entity=sim.fixture)
                valid=sim.array(contacts['valid_mask']).reshape(-1)
                a_links=sim.array(contacts['link_a']).reshape(-1)
                b_links=sim.array(contacts['link_b']).reshape(-1)
                on_finger=valid & ((a_links==link.idx)|(b_links==link.idx))
                full=np.array(tactile['force_sensor_N'])
                assert full.shape==(120,3) and np.isfinite(full).all()
                back=full@np.array(layout['sensor_to_link_rotation']).T
                np.testing.assert_allclose(back,tactile['force_link_N'],atol=1e-6)
                samples.append({'offset_m':offset,'solver_contacts':int(valid.sum()),
                                'target_finger_contacts':int(on_finger.sum()),
                                'physical_contact_details':sim.evaluate()['solver_contacts'],
                                'active_taxels':tactile['active_taxels'],'peak_force_N':tactile['peak_force_N'],
                                'force_sensor_N':tactile['force_sensor_N']})
            report['finger_press'][name]=samples
            print('FINGER',name,[(s['offset_m'],s['solver_contacts'],s['active_taxels'],s['peak_force_N']) for s in samples],flush=True)
            assert any(s['target_finger_contacts']>0 and s['active_taxels']>0 for s in samples),name
            assert samples[0]['active_taxels']==samples[-1]['active_taxels']==0,name
        sim.reset()
        sim.render_enabled=True
        sim.render()
        for image in ['scene','hand']:
            assert sim.image_bytes[image].startswith(b'\xff\xd8')
        report['result']='pass'
        report['checks']=['18 DOF and SDK order','official UR7e arm kinematics, masses and inertias',
                          '600 source point IDs and mm-to-m conversion','invalid requests do not advance',
                          '200 stable control steps','wrist and finger motion','five-finger press and release',
                          'sensor/link force-frame conversion','two camera renders']
    except Exception as exc:
        report['result']='fail'
        report['error']=repr(exc)
        raise
    finally:
        (out/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
        sim.scene.destroy()
    print(json.dumps({k:v for k,v in report.items() if k not in ['geometry','finger_press']},indent=2))


if __name__=='__main__':main()
