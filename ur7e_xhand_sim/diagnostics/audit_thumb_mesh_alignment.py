"""Read-only CPU audit of tactile points and fixed-child collision surfaces.

No Genesis initialization, asset build, scene mutation, or robot command occurs.
The per-collision convex hull is a documented approximation: Genesis can also
decimate/decompose it. Source contact positions identify the likely component,
but exact engine geometry must be checked independently in a built scene.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation
import trimesh


ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def transform(origin):
    value = np.eye(4)
    if origin is not None:
        value[:3, 3] = np.fromstring(origin.get('xyz', '0 0 0'), sep=' ')
        value[:3, :3] = Rotation.from_euler(
            'xyz', np.fromstring(origin.get('rpy', '0 0 0'), sep=' ')).as_matrix()
    return value


class Model:
    def __init__(self, urdf):
        self.path = Path(urdf)
        self.xml = ET.parse(urdf).getroot()
        self.parents = {joint.find('child').get('link'): joint for joint in self.xml.findall('joint')}
        self.links = {link.get('name'): link for link in self.xml.findall('link')}

    def relative(self, ancestor, child, q=None):
        if child == ancestor:
            return np.eye(4)
        joint = self.parents[child]
        local = transform(joint.find('origin'))
        if joint.get('type') != 'fixed':
            axis = np.fromstring(joint.find('axis').get('xyz'), sep=' ')
            angle = (q or {}).get(joint.get('name'), 0.)
            local[:3, :3] = local[:3, :3] @ Rotation.from_rotvec(axis * angle).as_matrix()
        return self.relative(ancestor, joint.find('parent').get('link'), q) @ local

    def fixed_components(self, base):
        names = [base]
        for name in names:
            names.extend(child for child, joint in self.parents.items()
                         if joint.get('type') == 'fixed' and joint.find('parent').get('link') == name)
        result = []
        for name in names:
            for index, collision in enumerate(self.links[name].findall('collision')):
                mesh_xml = collision.find('geometry/mesh')
                if mesh_xml is None:
                    raise ValueError('Audit expects mesh collision geometry')
                path = Path(mesh_xml.get('filename'))
                if not path.is_absolute():
                    path = self.path.parent / path
                mesh = trimesh.load(path, force='mesh')
                mesh.apply_scale(np.fromstring(mesh_xml.get('scale', '1 1 1'), sep=' '))
                mesh.apply_transform(self.relative(base, name) @ transform(collision.find('origin')))
                result.append(dict(name=name, collision_index=index, path=str(path),
                                   sha256=sha(path), raw=mesh, hull=mesh.convex_hull))
        return result


def stats(value):
    value = np.asarray(value)
    return dict(min=float(value.min()), median=float(np.median(value)), max=float(value.max()))


def nearest(mesh, points):
    surface, distance, face = mesh.nearest.on_surface(points)
    return surface, distance, mesh.face_normals[face]


def audit_layout(layout, components):
    points = np.asarray(layout['positions_link_m'])
    main = components[0]
    per_component = []
    for component in components:
        item = {key: component[key] for key in ('name', 'collision_index', 'path', 'sha256')}
        for kind in ('raw', 'hull'):
            mesh = component[kind]
            surface, distance, normals = nearest(mesh, points)
            inside = mesh.contains(points)
            # Positive means outward of this closed surface, negative inside.
            signed = distance * np.where(inside, -1., 1.)
            item[kind] = dict(vertices=len(mesh.vertices), faces=len(mesh.faces),
                watertight=bool(mesh.is_watertight), inside_count=int(inside.sum()),
                signed_distance_mm=stats(signed * 1000),
                per_taxel_signed_distance_m=signed.tolist(),
                closest_surface_points_m=surface.tolist(), outward_normals=normals.tolist())
        per_component.append(item)

    points = np.asarray(points)
    axis = np.array([0., 0., 1.]) if layout['finger'] == 'thumb' else np.array([1., 0., 0.])
    long_axis, side_axis = (1, 0) if layout['finger'] == 'thumb' else (2, 1)
    surface, distance, normals = nearest(main['hull'], points)
    facing = np.flatnonzero(normals @ axis > .7)
    central = facing[np.argmin((points[facing, long_axis]-np.median(points[facing, long_axis]))**2
                              + points[facing, side_axis]**2)]
    edge = facing[np.argmax(abs(points[facing, side_axis]))]
    tip = int(np.argmax(points[:, long_axis]))
    targets = {}
    for label, index in [('central', central), ('edge', edge), ('tip', tip)]:
        targets[label] = dict(taxel_id=int(index + 1), sensor_link=layout['link_name'],
            probe_center_link_m=points[index].tolist(),
            primary_hull_surface_link_m=surface[index].tolist(), outward_normal_link=normals[index].tolist(),
            caveat='Primary collision hull target; check fixed-child obstruction and actual engine geometry.')
    return dict(finger=layout['finger'], sensor_link=layout['link_name'],
                taxel_count=len(points), merged_fixed_collision_components=per_component, scan_targets=targets)


def source_contact_audit(source, layout, components):
    source = Path(source)
    controls = [json.loads(line) for line in (source/'control.jsonl').read_text().splitlines()]
    evaluations = [json.loads(line) for line in (source/'evaluation.jsonl').read_text().splitlines()]
    frozen = source/'source_snapshot/ur7e_xhand_sim/generated/ur7e_xhand.urdf'
    if sha(frozen) != sha(ROOT/'ur7e_xhand_sim/generated/ur7e_xhand.urdf'):
        raise ValueError('Source frozen URDF differs from geometry being audited')
    source_layout = next(row for row in json.loads((source/'object_scene.json').read_text())['tactile_layout']
                         if row['finger'] == layout['finger'])
    np.testing.assert_array_equal(source_layout['positions_link_m'], layout['positions_link_m'])
    model = Model(frozen)
    points = np.asarray(layout['positions_link_m'])
    records = []
    for control, evaluation in zip(controls, evaluations):
        if evaluation['phase'] != 'hold':
            continue
        observation = control['after']
        wrist = np.eye(4)
        wrist[:3, 3] = observation['wrist_position_m']
        quat = np.asarray(observation['wrist_quaternion_wxyz'])
        wrist[:3, :3] = Rotation.from_quat(quat[[1, 2, 3, 0]]).as_matrix()
        pose = wrist @ model.relative('right_hand_link', layout['link_name'], observation['joint_positions_rad'])
        contacts = [contact for contact in evaluation['solver_contacts']
                    if layout['link_name'] in (contact['link_a'], contact['link_b'])]
        for contact in contacts:
            position = (np.asarray(contact['position_world_m'])-pose[:3, 3]) @ pose[:3, :3]
            normal_world = np.asarray(contact['normal_world'])
            # Orient toward the object center, avoiding solver normal convention assumptions.
            object_center = (np.asarray(evaluation['object_position_m'])-pose[:3, 3]) @ pose[:3, :3]
            normal = normal_world @ pose[:3, :3]
            if np.dot(normal, object_center-position) < 0:
                normal = -normal
            per_component = []
            for component in components:
                item = dict(name=component['name'])
                for kind in ('raw', 'hull'):
                    closest, distance, normals = nearest(component[kind], np.array([position, object_center]))
                    item[kind] = dict(contact_point_distance_mm=float(distance[0]*1000),
                        closest_to_contact_link_m=closest[0].tolist(),
                        sphere_surface_gap_mm=float(distance[1]*1000-22.5)
                        if evaluation['object_spec']['shape'] == 'sphere' else None)
                per_component.append(item)
            to_probe = points-position
            normal_distance = to_probe @ normal
            tangent_distance = np.linalg.norm(to_probe-normal_distance[:, None]*normal, axis=1)
            nearest_taxel = int(np.argmin(np.linalg.norm(to_probe, axis=1)))
            item = dict(step=evaluation['step'], sample_time_s=evaluation['sample_time_s'],
                contact_link_m=position.tolist(), contact_outward_normal_link=normal.tolist(),
                object_center_link_m=object_center.tolist(), normal_force_N=contact['normal_force_N'],
                nearest_taxel_to_contact_id=nearest_taxel+1,
                nearest_taxel_to_contact_distance_mm=float(np.linalg.norm(to_probe[nearest_taxel])*1000),
                nearest_taxel_normal_offset_mm=float(normal_distance[nearest_taxel]*1000),
                nearest_taxel_tangential_offset_mm=float(tangent_distance[nearest_taxel]*1000),
                components=per_component,
                closest_raw_component=min(per_component, key=lambda r:r['raw']['contact_point_distance_mm'])['name'],
                closest_hull_component=min(per_component, key=lambda r:r['hull']['contact_point_distance_mm'])['name'])
            if evaluation['object_spec']['shape'] == 'sphere':
                gaps = np.linalg.norm(points-object_center, axis=1)-.0225
                item.update(nearest_probe_sphere_gap_mm=float(gaps.min()*1000),
                            nearest_probe_to_sphere_id=int(gaps.argmin()+1),
                            analytic_overlapping_probe_count=int(np.sum(gaps < .0015)))
            records.append(item)
    counts = {component['name']:sum(row['closest_hull_component']==component['name'] for row in records)
              for component in components}
    summary = dict(contact_records=len(records), closest_hull_component_counts=counts)
    if records:
        for key in ('nearest_taxel_to_contact_distance_mm', 'nearest_taxel_normal_offset_mm',
                    'nearest_taxel_tangential_offset_mm', 'nearest_probe_sphere_gap_mm'):
            if key in records[0]:
                summary[key] = stats([row[key] for row in records])
        summary['component_contact_distance_mm'] = {
            component['name']:{kind:stats([row['components'][i][kind]['contact_point_distance_mm'] for row in records])
                               for kind in ('raw', 'hull')}
            for i, component in enumerate(components)}
    return dict(source=str(source), source_sha256={name:sha(source/name)
                for name in ('control.jsonl', 'evaluation.jsonl', 'object_scene.json')},
                frozen_urdf_sha256=sha(frozen), frozen_urdf_and_layout_match=True,
                summary=summary, contact_records=records)


def plot_result(output, layout, components, audit):
    import os
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/tactile-thumb-matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    points = np.asarray(layout['positions_link_m'])*1000
    axes_pairs = [(1, 2), (0, 1)]
    fig, axs = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)
    colors = ['#86a3bb', '#a0b3a5', '#f7a35c']
    sample = audit['contact_records'][len(audit['contact_records'])//2]
    for ax, (x, y) in zip(axs, axes_pairs):
        for component, color in zip(components, colors):
            mesh = component['hull']
            vertices = mesh.vertices*1000
            polygons = vertices[mesh.faces][:, :, [x, y]]
            ax.add_collection(PolyCollection(polygons, facecolors=color, edgecolors='none', alpha=.1,
                                            label=component['name'].replace('right_hand_', '')))
        ax.scatter(points[:, x], points[:, y], s=12, color='#1864ab', label='120 tactile point centers', zorder=3)
        positions = np.asarray([row['contact_link_m'] for row in audit['contact_records']])*1000
        ax.scatter(positions[:, x], positions[:, y], s=8, c='#c92a2a', label='Solver contact (3 s hold)', zorder=4)
        center = np.asarray(sample['object_center_link_m'])*1000
        ax.add_patch(plt.Circle(center[[x, y]], 22.5, fill=False, color='#c92a2a', linestyle='--',
                               label='Sphere projected silhouette'))
        point = points[sample['nearest_probe_to_sphere_id']-1]
        ax.scatter(point[x], point[y], s=65, facecolors='none', edgecolors='black', zorder=5,
                   label=f"Nearest probe #{sample['nearest_probe_to_sphere_id']}")
        ax.set(xlabel=f'{"XYZ"[x]} in sensor-link frame (mm)', ylabel=f'{"XYZ"[y]} in sensor-link frame (mm)',
               title='Side projection' if (x, y)==(1,2) else 'Top projection', aspect='equal')
        ax.autoscale_view()
        ax.grid(alpha=.2)
    axs[0].legend(loc='upper left', fontsize=7)
    fig.suptitle('Thumb: sphere contact is nearest to the fixed tip component\nCPU hull approximation; projected circle is not a cross-section', fontsize=12)
    fig.savefig(output/'thumb_sphere_collision_coverage.png', dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    for component, color in zip(components, colors):
        mesh = component['hull']
        polygons = (mesh.vertices[mesh.faces]*1000)[:, :, [1, 2]]
        ax.add_collection(PolyCollection(polygons, facecolors=color, edgecolors='none', alpha=.15,
                                        label=component['name'].replace('right_hand_', '')))
    center = np.asarray(sample['object_center_link_m'])*1000
    probe = points[sample['nearest_probe_to_sphere_id']-1]
    contact = np.asarray(sample['contact_link_m'])*1000
    # A sphere slice through the selected taxel avoids using its larger silhouette
    # to visually overstate probe penetration. Meshes remain projections.
    slice_radius = np.sqrt(22.5**2-(center[0]-probe[0])**2)
    ax.add_patch(plt.Circle(center[[1, 2]], slice_radius, fill=False, color='#c92a2a',
                           linewidth=2, label='Sphere slice through selected probe X'))
    nearby = points[:, 1] > 46
    ax.scatter(points[nearby, 1], points[nearby, 2], color='#1864ab', s=25, zorder=4,
               label='Probe centers')
    ax.add_patch(plt.Circle(probe[[1, 2]], 1.5, color='#1864ab', alpha=.15, label='1.5 mm virtual probe'))
    ax.scatter(contact[1], contact[2], color='#c92a2a', s=50, zorder=5, label='Physical contact point')
    ax.annotate(f"Probe #{sample['nearest_probe_to_sphere_id']}\nSphere gap {sample['nearest_probe_sphere_gap_mm']:.2f} mm",
                xy=probe[[1, 2]], xytext=(43.4, 6.8), fontsize=10,
                arrowprops=dict(arrowstyle='->', color='#1864ab'))
    ax.set(xlim=(43, 58), ylim=(-6, 9), aspect='equal', xlabel='Y in sensor-link frame (mm)',
           ylabel='Z in sensor-link frame (mm)', title='Thumb tip detail: physical contact beyond dense tactile coverage')
    ax.legend(loc='lower right', fontsize=7)
    ax.grid(alpha=.2)
    fig.savefig(output/'thumb_sphere_tip_detail.png', dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/tactile_thumb_validation_20261028_v1/geometry')
    parser.add_argument('--dataset', type=Path, default=ROOT/'outputs/direct_teacher_multishape_noslip5_20261028_v2')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    urdf = ROOT/'ur7e_xhand_sim/generated/ur7e_xhand.urdf'
    layout_path = ROOT/'ur7e_xhand_sim/generated/tactile_layout.json'
    model = Model(urdf)
    layouts = json.loads(layout_path.read_text())[:2]
    components = {layout['finger']: model.fixed_components(layout['link_name']) for layout in layouts}
    report = dict(schema='tactile_mesh_cpu_audit_v1', read_only=True, simulator_initialized=False,
        scope='Raw URDF meshes and per-collision convex hull approximation, including fixed children. No decimation or engine SDF reconstruction.',
        signed_distance_convention='Positive outside, negative inside each individual closed component; not a union SDF.',
        probe_radius_m=.0015, urdf_sha256=sha(urdf), tactile_layout_sha256=sha(layout_path),
        layouts={layout['finger']:audit_layout(layout, components[layout['finger']]) for layout in layouts},
        source_contacts={})
    for shape in ('sphere', 'cylinder', 'cuboid', 'hex_prism'):
        source = args.dataset/shape/'attempt01'
        if not (source/'evaluation.jsonl').exists():
            continue
        report['source_contacts'][shape] = {}
        for layout in layouts:
            report['source_contacts'][shape][layout['finger']] = source_contact_audit(
                source, layout, components[layout['finger']])
    sphere = report['source_contacts'].get('sphere', {}).get('thumb', {})
    if sphere.get('contact_records'):
        sample = sphere['contact_records'][len(sphere['contact_records'])//2]
        tip = next(component for component in components['thumb'] if component['name'].endswith('rota_tip'))
        surface, distance, normals = nearest(tip['hull'], [sample['object_center_link_m']])
        report['layouts']['thumb']['scan_targets']['source_contact_tip'] = dict(
            sensor_link=layouts[0]['link_name'], collision_component=tip['name'],
            primary_hull_surface_link_m=surface[0].tolist(),
            outward_normal_link=normals[0].tolist(),
            source_solver_outward_normal_link=sample['contact_outward_normal_link'],
            source_solver_contact_link_m=sample['contact_link_m'],
            source_sample_step=sample['step'],
            caveat='Closest tip-hull point to the sphere center, near a hull edge. Facet normal is not unique at the edge; source solver normal is also provided.')
    (args.output/'mesh_alignment.json').write_text(json.dumps(report, indent=2)+'\n')
    targets = {finger:value['scan_targets'] for finger, value in report['layouts'].items()}
    (args.output/'scan_targets.json').write_text(json.dumps(targets, indent=2)+'\n')
    (args.output/'targets.json').write_text(json.dumps(targets, indent=2)+'\n')
    if 'sphere' in report['source_contacts']:
        plot_result(args.output, layouts[0], components['thumb'], report['source_contacts']['sphere']['thumb'])
    print(json.dumps({shape:{finger:value['summary'] for finger,value in row.items()}
                      for shape,row in report['source_contacts'].items()}, indent=2))


if __name__ == '__main__':
    main()
