"""CPU recheck of source contacts against exported actual engine mesh geometry."""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation
import trimesh

from audit_thumb_mesh_alignment import ROOT, Model, audit_layout, source_contact_audit, sha


def main():
    base=ROOT/'outputs/tactile_thumb_validation_20261028_v1'
    source=base/'controlled_scan/full_robot_default_collision_geometry.json'
    exported=json.loads(source.read_text())
    model=Model(ROOT/'ur7e_xhand_sim/generated/ur7e_xhand.urdf')
    layouts=json.loads((ROOT/'ur7e_xhand_sim/generated/tactile_layout.json').read_text())[:2]
    result=dict(scope='Actual complete-robot default mesh preprocessing, with original fixed descendants. In this report hull denotes exported engine collision mesh, not an independently generated hull.',
                engine_geometry_sha256=sha(source),finger_geometry={},source_contacts={})
    components_by_finger={}
    for layout in layouts:
        finger=layout['finger']
        components=model.fixed_components(layout['link_name'])
        engine_meshes=[]
        for record in exported['fingers'][finger]:
            mesh=trimesh.Trimesh(vertices=record['init_vertices_geom_frame_m'],faces=record['init_faces'],process=False)
            matrix=np.eye(4)
            quat=np.asarray(record['geom_quaternion_link_wxyz'])
            matrix[:3,:3]=Rotation.from_quat(quat[[1,2,3,0]]).as_matrix()
            matrix[:3,3]=record['geom_position_link_m']
            mesh.apply_transform(matrix)
            engine_meshes.append(mesh)
        costs=np.array([[np.linalg.norm(component['raw'].bounds-engine.bounds) for engine in engine_meshes]
                        for component in components])
        rows,cols=linear_sum_assignment(costs)
        mapping=[]
        for i,j in zip(rows,cols):
            components[i]['hull']=engine_meshes[j]
            mapping.append(dict(source_component=components[i]['name'],engine_geometry_order=int(j),
                                bounds_difference_norm_mm=float(costs[i,j]*1000)))
        result['finger_geometry'][finger]=dict(component_mapping=mapping,layout_audit=audit_layout(layout,components))
        components_by_finger[finger]=components
    for shape in ('sphere','cylinder','cuboid','hex_prism'):
        source=ROOT/'outputs/direct_teacher_multishape_noslip5_20261028_v2'/shape/'attempt01'
        result['source_contacts'][shape]={layout['finger']:source_contact_audit(
            source,layout,components_by_finger[layout['finger']]) for layout in layouts}
    output=base/'geometry/actual_engine_mesh_alignment.json'
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({shape:{finger:item['summary'] for finger,item in group.items()}
                      for shape,group in result['source_contacts'].items()},indent=2))


if __name__=='__main__':
    main()
