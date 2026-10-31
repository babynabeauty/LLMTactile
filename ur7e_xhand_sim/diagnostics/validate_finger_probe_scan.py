"""Controlled diagnostic contacts on isolated, fixed XHand finger assemblies.

These are imposed initial poses followed by free physics steps, NOT grasp
demonstrations, force calibration, or evidence of hardware sensor accuracy.
No production assets, tactile layout, or Sandbox defaults are changed.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import time
import traceback
import xml.etree.ElementTree as ET

os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
os.environ.setdefault('OMP_NUM_THREADS', '8')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/tactile-thumb-matplotlib')

import numpy as np
from scipy.spatial.transform import Rotation

from audit_thumb_mesh_alignment import Model, ROOT, sha


GAPS_MM = [5., 2., 1., .5, 0., -.05, -.1, -.25, -.5, -1.]


def write(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')


def array(value):
    if hasattr(value, 'detach'):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def isolated_urdf(source, sensor_link, output):
    model = Model(source)
    names = [sensor_link]
    for name in names:
        names.extend(child for child, joint in model.parents.items()
                     if joint.get('type') == 'fixed' and joint.find('parent').get('link') == name)
    result = ET.Element('robot', name='diagnostic_'+sensor_link)
    for material in model.xml.findall('material'):
        result.append(copy.deepcopy(material))
    for name in names:
        result.append(copy.deepcopy(model.links[name]))
    for name in names[1:]:
        result.append(copy.deepcopy(model.parents[name]))
    ET.indent(result)
    ET.ElementTree(result).write(output, encoding='unicode')
    return names


def orientation(normal):
    normal = np.asarray(normal, float)
    normal /= np.linalg.norm(normal)
    seed = np.eye(3)[np.argmin(abs(normal))]
    first = np.cross(seed, normal)
    first /= np.linalg.norm(first)
    second = np.cross(normal, first)
    rotation = np.column_stack((first, second, normal))
    xyzw = Rotation.from_matrix(rotation).as_quat()
    return rotation, xyzw[[3, 0, 1, 2]]


def analytic_sdf(points, center, quat, shape):
    rotation = Rotation.from_quat(np.asarray(quat)[[1, 2, 3, 0]]).as_matrix()
    local = (points-center) @ rotation
    if shape == 'sphere':
        return np.linalg.norm(local, axis=1)-.0225
    signed_box = abs(local)-np.array([.004, .004, .003])
    return np.linalg.norm(np.maximum(signed_box, 0.), axis=1)+np.minimum(signed_box.max(axis=1), 0.)


def engine_grid_sdf_cpu(points, center, quat, geom):
    """Independent CPU reproduction of the engine's grid/proxy SDF path."""
    rotation=Rotation.from_quat(np.asarray(quat)[[1,2,3,0]]).as_matrix()
    local=(points-center)@rotation
    matrix=array(geom.T_mesh_to_sdf)
    grid=local@matrix[:3,:3].T+matrix[:3,3]
    resolution=array(geom.sdf_res)
    outside=((grid>=resolution-1)|(grid<=0)).any(axis=1)
    result=np.linalg.norm((grid-(resolution-1)/2)*array(geom.sdf_cell_size),axis=1)+float(geom.sdf_max)
    active=np.flatnonzero(~outside)
    if len(active):
        values=array(geom.sdf_val)
        base=np.minimum(np.floor(grid[active]).astype(int),resolution-2)
        result[active]=0.
        for x in (0,1):
            for y in (0,1):
                for z in (0,1):
                    index=base+[x,y,z]
                    weight=np.prod(1-abs(grid[active]-index),axis=1)
                    result[active]+=weight*values[index[:,0],index[:,1],index[:,2]]
    return result,outside


def export_geometry(scene, path):
    rows = []
    for geom in scene.rigid_solver.geoms:
        rows.append(dict(geom_index=geom.idx, link_index=geom.link.idx, link_name=geom.link.name,
            entity_index=geom.link.entity.idx, is_convex=bool(geom.is_convex),
            init_vertices_geom_frame_m=array(geom.init_verts).tolist(),
            init_faces=array(geom.init_faces).tolist(),
            geom_position_world_m=array(geom.get_pos(relative=False)).reshape(-1, 3)[0].tolist(),
            geom_quaternion_world_wxyz=array(geom.get_quat(relative=False)).reshape(-1, 4)[0].tolist(),
            vertices_world_m=array(geom.get_verts()).reshape(-1, 3).tolist()))
    write(path, dict(scope='Actual built engine collision geoms, exported before scanning', geoms=rows))


def local_geometry(entity, sensor_link):
    return [dict(init_vertices_geom_frame_m=array(geom.init_verts).tolist(),
        init_faces=array(geom.init_faces).tolist(),
        geom_position_link_m=array(geom.init_pos).tolist(),
        geom_quaternion_link_wxyz=array(geom.init_quat).tolist())
        for geom in entity.get_link(sensor_link).geoms]


def full_robot_mesh_reference(gs, output):
    """Load the exact original URDF/default mesh pipeline; no full scene build."""
    source=ROOT/'outputs/direct_teacher_multishape_noslip5_20261028_v2/sphere/attempt01/source_snapshot/ur7e_xhand_sim/generated/ur7e_xhand.urdf'
    scene=gs.Scene(show_viewer=False)
    robot=scene.add_entity(gs.morphs.URDF(file=str(source),fixed=True,pos=(0,0,.2),
        merge_fixed_links=True,links_to_keep=['right_hand_link','tool0']),material=gs.materials.Rigid(friction=.8))
    result={finger:local_geometry(robot,link) for finger,link in
            [('thumb','right_hand_thumb_rota_link2'),('index','right_hand_index_rota_link2')]}
    write(output/'full_robot_default_collision_geometry.json',dict(source_urdf_sha256=sha(source),
        full_scene_built=False,scope='Full articulated robot loaded through original default mesh processing, before GPU scene build.',fingers=result))
    scene.destroy()
    return result


def contacts_for(robot):
    data = {key:array(value) for key,value in robot.get_contacts().items()}
    count = data['link_a'].size
    mask = data.get('valid_mask', np.ones(count, bool)).reshape(-1)
    rows = []
    for i in np.flatnonzero(mask):
        item = {}
        for key, value in data.items():
            if key == 'valid_mask':
                continue
            if value.size == count:
                item[key] = value.reshape(-1)[i].item()
            else:
                item[key] = value.reshape(count, -1)[i].tolist()
        rows.append(item)
    return rows


def run_one(gs, layout, shape, targets, args, full_reference):
    output = args.output/f"{layout['finger']}_{shape}"
    output.mkdir(parents=True, exist_ok=False)
    source = ROOT/'outputs/direct_teacher_multishape_noslip5_20261028_v2/sphere/attempt01/source_snapshot/ur7e_xhand_sim/generated/ur7e_xhand.urdf'
    urdf = output/'isolated_finger.urdf'
    included = isolated_urdf(source, layout['link_name'], urdf)
    scene = gs.Scene(sim_options=gs.options.SimOptions(dt=1/300, substeps=1, gravity=(0, 0, 0)),
        rigid_options=gs.options.RigidOptions(noslip_iterations=5, noslip_tolerance=1e-6,
            enable_self_collision=True, enable_joint_limit=True, enable_neutral_collision=True), show_viewer=False)
    # A fully fixed isolated URDF can be classified as an object, while the
    # original articulated hand is classified as a robot. Match the latter's
    # default per-collision hull policy explicitly to avoid changing geometry.
    robot = scene.add_entity(gs.morphs.URDF(file=str(urdf), fixed=True, merge_fixed_links=True,
        convexify=True, decimate=True, decimate_face_num=500,
        decompose_object_error_threshold=float('inf'), decompose_robot_error_threshold=float('inf')),
                            material=gs.materials.Rigid(friction=.8))
    if shape == 'sphere':
        volume = 4*np.pi*.0225**3/3
        morph = gs.morphs.Sphere(radius=.0225, pos=(1, 1, 1))
    else:
        volume = .008*.008*.006
        morph = gs.morphs.Box(size=(.008, .008, .006), pos=(1, 1, 1))
    obj = scene.add_entity(morph, material=gs.materials.Rigid(rho=.05/volume, friction=.8))
    link = robot.get_link(layout['link_name'])
    sensor = scene.add_sensor(gs.sensors.KinematicTaxel(entity_idx=robot.idx,
        link_idx_local=link.idx_local, probe_local_pos=layout['positions_link_m'],
        probe_radius=.0015, history_length=5, normal_stiffness=500., normal_damping=1.,
        normal_exponent=1.2, shear_scalar=2., twist_scalar=0., draw_debug=False))
    scene.build(n_envs=1)
    export_geometry(scene, output/'engine_geometry.json')
    isolated=local_geometry(robot,layout['link_name'])
    reference=full_reference[layout['finger']]
    if len(reference)!=len(isolated):
        raise ValueError('Isolated collision geometry count differs from complete robot')
    for expected,actual in zip(reference,isolated):
        for key in expected:
            np.testing.assert_allclose(actual[key],expected[key],rtol=0,atol=1e-12,
                                       err_msg=f'Isolated collision geometry differs: {key}')
    np.testing.assert_allclose(array(link.get_pos()).reshape(-1,3)[0],0.,atol=1e-8)
    np.testing.assert_allclose(array(link.get_quat()).reshape(-1,4)[0],[1,0,0,0],atol=1e-8)
    mass = float(array(obj.get_mass()).reshape(-1)[0])
    object_geom=obj.geoms[0]
    np.savez_compressed(output/'object_engine_sdf_grid.npz',
        T_mesh_to_sdf=array(object_geom.T_mesh_to_sdf),sdf_values=array(object_geom.sdf_val),
        sdf_resolution=array(object_geom.sdf_res),sdf_cell_size=array(object_geom.sdf_cell_size),
        sdf_max=np.asarray(object_geom.sdf_max))
    if not np.isclose(mass, .05, rtol=1e-5):
        raise ValueError('Object engine mass differs from 50 g')
    write(output/'scene.json', dict(included_fixed_links=included, urdf_sha256=sha(urdf),
        source_frozen_urdf_sha256=sha(source), layout=layout, shape=shape, object_mass_kg=mass,
        object_density_kg_m3=.05/volume, friction=.8, solver_options=scene.rigid_solver._options.model_dump(mode='json'),
        dt_s=1/300, gravity=[0,0,0], diagnostic_initial_pose_writes=True,
        mesh_processing='Match original robot: decimate 500 faces, convexify each collision, decomposition threshold infinity.',
        actual_collision_geometry_matches_full_robot=True,
        production_layout_unchanged=True, sensor_semantics='Uncalibrated kinematic spring/damper proxy'))
    points = np.asarray(layout['positions_link_m'])
    records = []
    with (output/'records.jsonl').open('w', buffering=1) as stream:
        for target_name, target in targets.items():
            normal = np.asarray(target['outward_normal_link'])
            normal /= np.linalg.norm(normal)
            surface = np.asarray(target['primary_hull_surface_link_m'])
            rotation, quat = orientation(normal)
            for gap_mm in GAPS_MM:
                scene.reset()
                support = .0225 if shape == 'sphere' else .003
                center = surface+normal*(support+gap_mm/1000)
                obj.set_quat(quat, zero_velocity=True)
                obj.set_pos(center, zero_velocity=True)
                before = dict(position_m=array(obj.get_pos()).reshape(-1,3)[0].tolist(),
                    quaternion_wxyz=array(obj.get_quat()).reshape(-1,4)[0].tolist(),
                    velocity_m_s=array(obj.get_vel()).reshape(-1,3)[0].tolist(),
                    angular_velocity_rad_s=array(obj.get_ang()).reshape(-1,3)[0].tolist())
                if np.linalg.norm(before['velocity_m_s'])+np.linalg.norm(before['angular_velocity_rad_s'])>1e-7:
                    raise ValueError('Initial velocity was not zero')
                for step in range(1, 6):
                    scene.step()
                    if step not in (1, 5):
                        continue
                    pos = array(obj.get_pos()).reshape(-1,3)[0]
                    actual_quat = array(obj.get_quat()).reshape(-1,4)[0]
                    force = array(sensor.read().force).reshape(-1,120,3)[0]
                    contacts = contacts_for(robot)
                    sdf = analytic_sdf(points, pos, actual_quat, shape)
                    geom_pos=array(object_geom.get_pos(relative=False)).reshape(-1,3)[0]
                    geom_quat=array(object_geom.get_quat(relative=False)).reshape(-1,4)[0]
                    grid_sdf,outside_grid=engine_grid_sdf_cpu(points,geom_pos,geom_quat,object_geom)
                    overlapping = .0015-sdf > 0
                    expected = overlapping if contacts else np.zeros(120, bool)
                    grid_expected=(.0015-grid_sdf>0) if contacts else np.zeros(120,bool)
                    active = np.linalg.norm(force, axis=1)>1e-8
                    finite = bool(np.isfinite(np.r_[pos,actual_quat,force.ravel(),sdf]).all())
                    row = dict(finger=layout['finger'], shape=shape, target=target_name,
                        nominal_gap_mm=gap_mm, steps_after_initial_placement=step,
                        initial_pose=before, after_position_m=pos.tolist(), after_quaternion_wxyz=actual_quat.tolist(),
                        after_velocity_m_s=array(obj.get_vel()).reshape(-1,3)[0].tolist(),
                        after_object_geom_position_m=geom_pos.tolist(), after_object_geom_quaternion_wxyz=geom_quat.tolist(),
                        after_angular_velocity_rad_s=array(obj.get_ang()).reshape(-1,3)[0].tolist(),
                        contacts=contacts, contact_count=len(contacts),
                        total_contact_load_N=float(sum(np.linalg.norm(contact['force_a']) for contact in contacts)),
                        raw_taxel_force_sensor_nominal_N=force.tolist(),
                        taxel_norm_sum_nominal_N=float(np.linalg.norm(force,axis=1).sum()),
                        active_taxel_ids_1based=(np.flatnonzero(active)+1).tolist(),
                        analytic_probe_sdf_m=sdf.tolist(), analytic_probe_overlap_m=np.maximum(.0015-sdf,0.).tolist(),
                        engine_grid_sdf_cpu_m=grid_sdf.tolist(), engine_grid_outside_probe_ids_1based=(np.flatnonzero(outside_grid)+1).tolist(),
                        expected_grid_active_taxel_ids_1based=(np.flatnonzero(grid_expected)+1).tolist(),
                        active_mask_matches_engine_grid=bool(np.array_equal(active,grid_expected)) if shape=='plate' else None,
                        expected_active_taxel_ids_1based=(np.flatnonzero(expected)+1).tolist(),
                        active_mask_matches_analytic=bool(np.array_equal(active, expected)),
                        no_contact_zero_response=bool(contacts or not active.any()), all_finite=finite)
                    records.append(row)
                    stream.write(json.dumps(row, allow_nan=False)+'\n')
                    if not finite:
                        raise ValueError('Nonfinite sensor or physics result')
        summary = dict(samples=len(records), all_finite=all(r['all_finite'] for r in records),
            no_contact_zero_response=all(r['no_contact_zero_response'] for r in records),
            active_mask_match_count=sum(r['active_mask_matches_analytic'] for r in records),
            engine_grid_active_mask_match_count=sum(bool(r['active_mask_matches_engine_grid']) for r in records) if shape=='plate' else None,
            mismatches=[{k:r[k] for k in ('target','nominal_gap_mm','steps_after_initial_placement',
                'active_taxel_ids_1based','expected_active_taxel_ids_1based','contact_count')}
                for r in records if not r['active_mask_matches_analytic']],
            by_target={target:dict(
                physical_contact_samples=sum(r['contact_count']>0 for r in records if r['target']==target),
                responding_samples=sum(bool(r['active_taxel_ids_1based']) for r in records if r['target']==target),
                max_active_taxels=max(len(r['active_taxel_ids_1based']) for r in records if r['target']==target),
                max_contact_load_N=max(r['total_contact_load_N'] for r in records if r['target']==target))
                for target in targets})
    write(output/'summary.json',summary)
    scene.destroy()
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/tactile_thumb_validation_20261028_v1/controlled_scan')
    parser.add_argument('--finger',choices=['thumb','index','all'],default='all')
    parser.add_argument('--shape',choices=['sphere','plate','all'],default='all')
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    import genesis as gs
    sensor_file=ROOT/'Genesis/genesis/engine/sensors/kinematic_tactile.py'
    manifest=dict(schema='controlled_finger_probe_scan_v1', diagnostic_only=True,
        canonical_training_data=False, force_calibration=False, started_unix_s=time.time(),
        scope='Imposed initial gap/zero velocity, then 1 and 5 free 300 Hz physics steps; isolated fixed finger, all fixed descendants retained.',
        gaps_mm=GAPS_MM, readbacks_at_steps=[1,5], script_sha256=sha(__file__),
        sensor_implementation_sha256=sha(sensor_file))
    write(args.output/'manifest.json',manifest)
    layouts=json.loads((ROOT/'ur7e_xhand_sim/generated/tactile_layout.json').read_text())[:2]
    targets=json.loads((ROOT/'outputs/tactile_thumb_validation_20261028_v1/geometry/targets.json').read_text())
    results={}
    gs.init(backend=gs.gpu, precision='32', seed=42, logging_level='warning')
    try:
        full_reference=full_robot_mesh_reference(gs,args.output)
        for layout in layouts:
            if args.finger not in ('all',layout['finger']):
                continue
            for shape in ('sphere','plate'):
                if args.shape not in ('all',shape):
                    continue
                key=layout['finger']+'_'+shape
                print('BUILD',key,flush=True)
                results[key]=run_one(gs,layout,shape,targets[layout['finger']],args,full_reference)
                write(args.output/'summary.json',results)
                print('DONE',key,json.dumps(results[key]),flush=True)
    except Exception:
        (args.output/'error.txt').write_text(traceback.format_exc())
        raise
    finally:
        manifest.update(finished_unix_s=time.time(), sensor_source_unchanged_during_run=sha(sensor_file)==manifest['sensor_implementation_sha256'])
        write(args.output/'manifest.json',manifest)
        gs.destroy()


if __name__=='__main__':
    main()
