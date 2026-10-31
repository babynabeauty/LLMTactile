"""Isolated physical positive controls for all five original XHand sensor pads.

Reuse complete archived IPC finger collision assemblies, including fixed shells
and tips. A small FEM cube falls onto each upward-facing pad. Independent IPC
contact presence gates the unchanged manufacturer-layout virtual-taxel adapter;
force magnitudes never construct its signal. No full robot or controller runs.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys

os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
os.environ.setdefault('OMP_NUM_THREADS', '4')
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
import trimesh
import genesis as gs

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'ur7e_xhand_sim'))
from fem_diagnostics import use_physics_surface, mesh_quality
from ipc_support import weld_ipc_triangle_soups
from .soft_contact import IPCContactAudit
from .soft_taxel import SoftTaxelAdapter


def array(value):
    return value.detach().cpu().numpy() if hasattr(value, 'detach') else np.asarray(value)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def nearest_error(a, b):
    return float(max(cKDTree(a).query(b)[0].max(), cKDTree(b).query(a)[0].max()))


def run(args):
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    source = args.source.resolve()
    layout_path = ROOT/'ur7e_xhand_sim/generated/tactile_layout.json'
    layouts = json.loads(layout_path.read_text())
    fingers = [x['finger'] for x in layouts]
    assert fingers == ['thumb', 'index', 'middle', 'ring', 'pinky']
    records, meshes, transforms = [], [], []
    for i, layout in enumerate(layouts):
        path = source/f'{layout["finger"]}.npz'
        with np.load(path) as archived:
            verts, faces = archived['vertices'].copy(), archived['faces'].copy()
        mesh = trimesh.Trimesh(verts, faces, process=False)
        assert mesh.is_watertight and mesh.is_winding_consistent and mesh.volume > 0
        # Central uncovered pad point, selected against the full original mesh.
        # Agent's independent ray check established this point has an open
        # outward approach, without deleting back-shell or tip geometry.
        probe = np.asarray(layout['positions_link_m'])[61]
        closest, distance, tri = trimesh.proximity.closest_point(mesh, probe[None])
        normal = mesh.face_normals[int(tri[0])]
        rotation = Rotation.align_vectors([[0., 0., 1.]], [normal])[0].as_matrix()
        anchor = np.array([i*.15, 0., .05])
        translation = anchor - rotation @ closest[0]
        transforms.append((translation, rotation))
        world_mesh = mesh.copy()
        world_mesh.vertices = verts @ rotation.T + translation
        # Verify clear space over the finite cube footprint before release.
        xy = np.linspace(-.00399, .00399, 9)
        origins = np.array([anchor + [x, y, .1] for x in xy for y in xy])
        hits, ray_ids, _ = world_mesh.ray.intersects_location(origins, np.tile([0, 0, -1.], (len(origins), 1)))
        ceiling = float(hits[:, 2].max())
        bottom = ceiling + .003
        file = out/f'{layout["finger"]}_original_collision.obj'
        file.write_text(trimesh.exchange.obj.export_obj(mesh, include_normals=False, include_color=False, digits=16))
        records.append(dict(finger=layout['finger'], source=str(path), source_sha256=sha(path),
            source_vertices=len(verts), source_faces=len(faces), source_components=len(mesh.split()),
            selected_point_id=62, manufacturer_probe_link_m=probe.tolist(),
            selected_surface_point_link_m=closest[0].tolist(), outward_normal_link=normal.tolist(),
            probe_surface_distance_m=float(distance[0]), link_to_world_position_m=translation.tolist(),
            link_to_world_rotation=rotation.tolist(), fixture_obj=str(file),
            fixture_obj_sha256=sha(file), cube_center_m=[float(anchor[0]), 0., bottom+.004],
            initial_bottom_clearance_m=.003, sampled_pad_surface_max_z_m=ceiling))
        meshes.append(mesh)
    report = dict(passed=False, purpose='physical sensor-pad positive and no-contact negative controls',
        source_layout_path=str(layout_path), source_layout_sha256=sha(layout_path),
        physics_dt_s=1/300, substeps=2, tactile_hz=60, newton_tolerance=1e-5,
        probe_radius_m=.0015, normal_stiffness=500., normal_damping=1., normal_exponent=1.2,
        shear_scalar=2., sensor_geometry_unchanged=True,
        material=dict(E_Pa=1e5, nu=.45, rho_kg_m3=1100, model='stable_neohookean', friction_mu=.8),
        cube_size_m=[.008]*3, duration_s=args.duration, fingers=records,
        source_hashes={name: sha(Path(__file__).parent/name) for name in
                       ('calibrate_soft_pad_contact.py', 'soft_contact.py', 'soft_taxel.py')},
        limitations=['Isolated fixed-finger physical positive control, not a grasp or real sensor calibration.',
                    'Virtual taxel amplitudes remain geometric estimates, not IPC force measurements.',
                    'The cube and each finger are in transformed local test fixtures, not the full arm pose.',
                    'Normal loads sample the last IPC substep and are not interval averages.'])
    (out/'protocol.json').write_text(json.dumps(report, indent=2))
    gs.init(backend=gs.gpu, precision='32', seed=7010, logging_level='warning', use_deterministic_algorithms=True)
    try:
        scene = gs.Scene(sim_options=gs.options.SimOptions(dt=1/300, substeps=2), show_viewer=False,
            coupler_options=gs.options.IPCCouplerOptions(contact_d_hat=.0005,
                newton_tolerance=1e-5, newton_translation_tolerance=1e-5,
                linear_system_tolerance=1e-6, n_linesearch_iterations=32,
                enable_rigid_rigid_contact=False, enable_rigid_ground_contact=False),
            fem_options=gs.options.FEMOptions(use_implicit_solver=True, damping_alpha=5., damping_beta=5e-4),
            rigid_options=gs.options.RigidOptions(batch_links_info=False, batch_dofs_info=False))
        fixtures, blocks = [], []
        for record, (position, rotation) in zip(records, transforms):
            xyzw = Rotation.from_matrix(rotation).as_quat()
            fixture = scene.add_entity(gs.morphs.Mesh(file=record['fixture_obj'], pos=tuple(position),
                quat=tuple(xyzw[[3, 0, 1, 2]]), fixed=True, align=False, convexify=False,
                decimate=False, watertighten=None), material=gs.materials.Rigid(coup_friction=.8))
            block = scene.add_entity(gs.morphs.Box(size=(.008,)*3, pos=record['cube_center_m'],
                maxvolume=1e-8, nobisect=False), material=gs.materials.FEM.Elastic(
                    E=1e5, nu=.45, rho=1100, model='stable_neohookean', friction_mu=.8))
            use_physics_surface(block)
            fixtures.append(fixture)
            blocks.append(block)
        repairs = weld_ipc_triangle_soups(scene.sim.coupler)
        scene.build(n_envs=1)
        finger_links = {name: fixture.base_link for name, fixture in zip(fingers, fixtures)}
        audits = [IPCContactAudit(scene, block, robot=fixture, finger_links=finger_links)
                  for block, fixture in zip(blocks, fixtures)]
        adapters = [SoftTaxelAdapter(layouts) for _ in blocks]
        positions = np.array([array(f.base_link.get_pos()).reshape(3) for f in fixtures])
        rotations = np.array([Rotation.from_quat(array(f.base_link.get_quat()).reshape(4)[[1, 2, 3, 0]]).as_matrix()
                              for f in fixtures])
        for i, (record, mesh, fixture, block, audit) in enumerate(zip(records, meshes, fixtures, blocks, audits)):
            expected_world = mesh.vertices @ transforms[i][1].T + transforms[i][0]
            slots = scene.sim.coupler._abd_slots_by_link[fixture.base_link]
            ipc_world, ipc_local, genesis_local = [], [], []
            for slot in slots:
                geom = slot.geometry()
                v = np.asarray(geom.positions().view()).reshape(-1, 3)
                T = np.asarray(geom.transforms().view()).reshape(-1, 4, 4)[0]
                ipc_local.append(v)
                ipc_world.append(v @ T[:3, :3].T + T[:3, 3])
            for geom in fixture.base_link.geoms:
                q = np.asarray(geom.init_quat)[[1, 2, 3, 0]]
                genesis_local.append(np.asarray(geom.init_verts) @ Rotation.from_quat(q).as_matrix().T + geom.init_pos)
            err = dict(ipc_local_m=nearest_error(np.vstack(ipc_local), mesh.vertices),
                ipc_world_m=nearest_error(np.vstack(ipc_world), expected_world),
                genesis_local_m=nearest_error(np.vstack(genesis_local), mesh.vertices),
                link_position_m=float(np.max(np.abs(positions[i]-transforms[i][0]))),
                link_rotation=float(np.max(np.abs(rotations[i]-transforms[i][1]))))
            record['coordinate_errors'] = err
            assert max(err.values()) < 2e-7, err
            record['ipc_friction_readback'] = audit.friction_readback()
            quality = mesh_quality(array(block.init_positions).reshape(-1, 3), array(block.get_el2v()).astype(int))
            record['mesh_quality'] = quality
            record['actual_mass_kg'] = quality['volume_m3']*float(block.material.rho)
            assert abs(record['actual_mass_kg']/(1100*.008**3)-1) < 2e-5
        report['ipc_mesh_import'] = repairs
        report['actual_ipc_substep_dt_s'] = audits[0].dt
        (out/'protocol.json').write_text(json.dumps(report, indent=2))
        rows = []
        arrays = {key: [] for key in ('time_s', 'normal_load_N', 'contact_mask', 'active_taxels',
                                     'taxel_force_link_N', 'raw_penetration_m', 'vertices_world_m')}
        # At t=0 use the same geometric adapter with explicitly false presence:
        # no solver contact has yet been assembled. Dynamic falling samples
        # below independently validate negative controls after actual steps.
        for tick in range(1, round(args.duration*300)+1):
            scene.step()
            if tick % 5:
                continue
            sample_rows, loads, masks, active, forces, penetrations, vertices = [], [], [], [], [], [], []
            for i, (record, block, audit, adapter) in enumerate(zip(records, blocks, audits, adapters)):
                truth = audit.read()
                mask = np.array([truth['fingers'][name]['normal_primitive_count'] > 0 for name in fingers], dtype=bool)
                state = audit.soft_vertex_state()
                sensed = adapter.sample(state['position_m'], np.asarray(block._surface_tri_np, dtype=np.int64),
                    positions, rotations, 1/60, contact_mask=mask)
                load = np.array([truth['fingers'][name]['normal_primitive_load_N'] for name in fingers])
                active_i = sensed['active'].sum(1)
                loads.append(load); masks.append(mask); active.append(active_i)
                forces.append(sensed['force_link_N']); penetrations.append(sensed['raw_penetration_m'])
                vertices.append(state['position_m'])
                sample_rows.append(dict(finger=record['finger'], normal_primitive_load_N=float(load[i]),
                    normal_force_on_soft_N=truth['fingers'][record['finger']]['normal_force_on_soft_N'],
                    pair_present=bool(mask[i]), active_taxels=int(active_i[i]),
                    virtual_force_norm_sum_N=float(np.linalg.norm(sensed['force_link_N'][i], axis=-1).sum()),
                    raw_geometric_active_taxels=int((sensed['raw_penetration_m'][i]>0).sum()),
                    force_balance_error_N=truth['max_pair_force_balance_error_N'],
                    cube_min_z_m=float(state['position_m'][:, 2].min()),
                    unrelated_finger_contacts=int(np.delete(mask, i).sum()),
                    unrelated_finger_taxels=int(np.delete(active_i, i).sum())))
            rows.append(dict(time_s=tick/300, fingers=sample_rows))
            for name, value in zip(arrays, (tick/300, loads, masks, active, forces, penetrations, vertices)):
                arrays[name].append(value)
            if tick % 30 == 0:
                print(json.dumps(rows[-1]), flush=True)
        for i, record in enumerate(records):
            own = [row['fingers'][i] for row in rows]
            neg = [row for row in own if not row['pair_present']]
            pos = [row for row in own if row['pair_present']]
            checks = dict(negative_samples_exist=bool(neg), positive_samples_exist=bool(pos),
                negative_signal_zero=all(row['active_taxels']==0 and row['virtual_force_norm_sum_N']==0 for row in neg),
                positive_load_exists=any(row['normal_primitive_load_N']>1e-6 for row in pos),
                positive_taxel_signal_exists=any(row['active_taxels']>0 and row['virtual_force_norm_sum_N']>0 for row in pos),
                positive_all_samples_have_taxel=all(row['active_taxels']>0 for row in pos),
                unrelated_channels_zero=all(row['unrelated_finger_contacts']==0 and row['unrelated_finger_taxels']==0 for row in own),
                pair_force_balance=all(row['force_balance_error_N']<1e-8 for row in own))
            record.update(checks=checks, passed=all(checks.values()), negative_samples=len(neg),
                positive_samples=len(pos), max_active_taxels=max(row['active_taxels'] for row in own),
                max_normal_load_N=max(row['normal_primitive_load_N'] for row in own),
                first_positive_time_s=next((row['time_s'] for row in rows if row['fingers'][i]['pair_present']), None))
        report['passed'] = all(record['passed'] for record in records)
        (out/'trace.json').write_text(json.dumps(rows, indent=2))
        # Node counts can differ across otherwise equal TetGen meshes; keep
        # separate numeric arrays instead of pickled ragged object arrays.
        packed = {k: np.asarray(v) for k, v in arrays.items() if k != 'vertices_world_m'}
        for i, finger in enumerate(fingers):
            packed[f'{finger}_vertices_world_m'] = np.stack([frame[i] for frame in arrays['vertices_world_m']])
            packed[f'{finger}_surface_faces'] = np.asarray(blocks[i]._surface_tri_np, dtype=np.int64)
        np.savez_compressed(out/'trace.npz', **packed)
        (out/'report.json').write_text(json.dumps(report, indent=2))
        print('REPORT', json.dumps(dict(passed=report['passed'], fingers=[{k: r[k] for k in
            ('finger', 'passed', 'negative_samples', 'positive_samples', 'max_active_taxels', 'max_normal_load_N')}
            for r in records])), flush=True)
        return report['passed']
    finally:
        if gs._initialized:
            gs.destroy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT/'outputs/soft_contact_validation_20261026_v1/sensor_geometry')
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/soft_contact_validation_20261026_v1/pad_contact')
    parser.add_argument('--duration', type=float, default=.4)
    args = parser.parse_args()
    if args.duration < .1 or args.duration > 1.:
        parser.error('duration must be bounded to [0.1, 1.0] seconds')
    if not run(args):
        raise SystemExit('pad physical calibration failed; inspect preserved report and traces')


if __name__ == '__main__':
    main()
