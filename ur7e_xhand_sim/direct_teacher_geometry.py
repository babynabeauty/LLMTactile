"""Pure hand FK/mesh construction of a bounded, opposed two-finger teacher.

No Genesis scene, physical search, API call or robot command is issued here.
The later direct teacher is privileged; these geometry inputs are not actor data.
"""
import json
import xml.etree.ElementTree as ET

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
import trimesh

from build_assets import HERE,HAND_JOINTS,ARM_JOINTS
from s8v4 import URDFFrames,sha


class OpposedPatchGeometry:
    SEED_FIRST_SIX=[1.18732,.55169,1.00089,.00044,.70543,.70000]

    def __init__(self):
        self.reference_path=HERE.parent/'outputs/rule_demos/teacher_pinch_v2_20261024/reference.json'
        self.reference=json.loads(self.reference_path.read_text())
        self.quaternion=np.asarray(self.reference['grasp_quaternion_wxyz'])
        self.rotation=Rotation.from_quat(self.quaternion[[1,2,3,0]]).as_matrix()
        self.hand=np.asarray(self.reference['open_hand_target_rad'],float)
        self.hand[:6]=self.SEED_FIRST_SIX
        self.urdf=HERE/'generated/ur7e_xhand.urdf'
        xml=ET.parse(self.urdf).getroot();self.fk=URDFFrames(self.urdf)
        limits=[xml.find(f'joint[@name="{name}"]/limit') for name in HAND_JOINTS]
        self.lower=np.array([float(l.get('lower')) for l in limits])
        self.upper=np.array([float(l.get('upper')) for l in limits])
        self.meshes={};self.mesh_sources={}
        for link in xml.findall('link'):
            name=link.get('name')
            if not name.startswith('right_hand'):continue
            for collision in link.findall('collision'):
                source=collision.find('geometry/mesh')
                if source is None:continue
                from pathlib import Path
                path=Path(source.get('filename'))
                if not path.is_absolute():path=self.urdf.parent/path
                mesh=trimesh.load(path,force='mesh')
                mesh.apply_scale(np.fromstring(source.get('scale','1 1 1'),sep=' '))
                origin=collision.find('origin');transform=np.eye(4)
                if origin is not None:
                    transform[:3,3]=np.fromstring(origin.get('xyz','0 0 0'),sep=' ')
                    transform[:3,:3]=Rotation.from_euler('xyz',np.fromstring(origin.get('rpy','0 0 0'),sep=' ')).as_matrix()
                mesh.apply_transform(transform)
                self.meshes[name]=mesh;self.mesh_sources[str(path.resolve())]=sha(path)
        layouts=json.loads((HERE/'generated/tactile_layout.json').read_text())
        self.layouts=[next(l for l in layouts if l['finger']==finger) for finger in ('thumb','index')]
        self.points=[];self.normals=[];self.masks=[]
        for i,layout in enumerate(self.layouts):
            points=np.asarray(layout['positions_link_m'])
            mesh=self.meshes[layout['link_name']]
            if not mesh.is_watertight or not mesh.is_winding_consistent or mesh.volume<=0:
                raise ValueError('Teacher patch normals require outward closed collision meshes')
            _,distance,faces=mesh.nearest.on_surface(points)
            if distance.max()>.002:raise ValueError('Taxels and collision mesh disagree')
            self.points.append(points);self.normals.append(mesh.face_normals[faces])
            _,normals=self.sensor_points_normals(self.hand,i)
            axis=np.array([-1 if i==0 else 1,0,0])
            self.masks.append(normals@axis>.7)
        if min(mask.sum() for mask in self.masks)<8:
            raise ValueError('Too few covered contact-facing taxels')
        self.patch_centers=np.array([self.patch(self.hand,i)[0] for i in range(2)])
        self.midpoint=self.patch_centers.mean(0)

    def transform(self,hand,link):
        joints=dict(zip(HAND_JOINTS,np.asarray(hand)))
        joints.update(dict.fromkeys(ARM_JOINTS,0.))
        return self.fk.transform('right_hand_link',link,joints)

    def sensor_points_normals(self,hand,finger):
        transform=self.transform(hand,self.layouts[finger]['link_name'])
        points=(self.points[finger]@transform[:3,:3].T+transform[:3,3])@self.rotation.T
        normals=self.normals[finger]@transform[:3,:3].T@self.rotation.T
        return points,normals

    def patch(self,hand,finger):
        points,normals=self.sensor_points_normals(hand,finger)
        mask=self.masks[finger];normal=normals[mask].mean(0)
        return points[mask].mean(0),normal/np.linalg.norm(normal)

    def solve(self,inward_m,patch_offsets_world_m=None):
        """Positive distance moves both patches horizontally toward each other."""
        if not np.isfinite(inward_m) or not -.0100001<=inward_m<=.0050001:
            raise ValueError('Teacher patch displacement is limited to [-10,+5] mm')
        hand=self.hand.copy();errors=[]
        offsets=np.zeros((2,3)) if patch_offsets_world_m is None else np.asarray(patch_offsets_world_m,float)
        if offsets.shape!=(2,3) or not np.isfinite(offsets).all() or np.max(np.abs(offsets))>.006:
            raise ValueError("Bounded patch offsets require finite 2x3 values within6mm")
        for finger,indices in enumerate((np.arange(3),np.arange(3,6))):
            wanted=self.patch_centers[finger]+offsets[finger]+np.array([(-1 if finger==0 else 1)*inward_m,0,0])
            def residual(value):
                trial=self.hand.copy();trial[indices]=value
                return (self.patch(trial,finger)[0]-wanted)*1000
            solution=least_squares(residual,self.hand[indices],
                bounds=(self.lower[indices],self.upper[indices]),
                xtol=1e-12,ftol=1e-12,gtol=1e-12,max_nfev=300)
            hand[indices]=solution.x;errors.append(float(np.linalg.norm(solution.fun)/1000))
        if max(errors)>2e-5 or np.any(hand<self.lower) or np.any(hand>self.upper):
            raise ValueError('Coordinated teacher path violates FK position/limit tolerance')
        return hand,errors

    def measure(self,hand):
        vertices={}
        for name,mesh in self.meshes.items():
            transform=self.transform(hand,name)
            vertices[name]=(mesh.vertices@transform[:3,:3].T+transform[:3,3])@self.rotation.T-self.midpoint
        lowest=min((float(v[:,2].min()),name) for name,v in vertices.items())
        thumb=np.concatenate([v for name,v in vertices.items() if 'thumb' in name])
        index=np.concatenate([v for name,v in vertices.items() if 'index' in name])
        nearest=float(cKDTree(thumb).query(index)[0].min())
        normals=[self.patch(hand,i)[1] for i in range(2)]
        return dict(lowest_hand_surface_relative_object_center_m=lowest[0],lowest_link=lowest[1],
            minimum_thumb_index_vertex_distance_m=nearest,
            opposed_patch_normal_angle_deg=float(np.degrees(np.arccos(np.clip(normals[0]@normals[1],-1,1)))))

    def candidate(self,shape,gain=1.,force_target_N=1.2,contact_height_offset_m=0.,
                  patch_offsets_world_m=None,palm_translation_delta_m=None):
        if shape not in ('sphere','cylinder','hex_prism','cuboid'):raise ValueError('Unsupported opposed-patch teacher shape')
        if gain not in (1.,2.,3.):raise ValueError('Bounded teacher gain must be 1, 2, or 3')
        height_limit=.030 if shape=='cylinder' else .006
        if not np.isfinite(contact_height_offset_m) or not 0<=contact_height_offset_m<=height_limit:
            raise ValueError('Contact-height offset exceeds validated object-specific bounds')
        if not np.isfinite(force_target_N) or not .2<=force_target_N<=4.:
            raise ValueError('Per-finger teacher force target must be [0.2,4] N')
        delta=np.zeros(3) if palm_translation_delta_m is None else np.asarray(palm_translation_delta_m,float)
        if delta.shape!=(3,) or not np.isfinite(delta).all() or np.max(np.abs(delta))>.006:
            raise ValueError("Palm registration correction must remain within6mm")
        coordinates=np.linspace(-.010,.005,31);hands=[];measurements=[];errors=[]
        for coordinate in coordinates:
            hand,error=self.solve(float(coordinate),patch_offsets_world_m)
            hands.append(hand);errors.extend(error);measurements.append(self.measure(hand))
        radius=.045 if shape=='cylinder' else .0225
        clearance=min(m['lowest_hand_surface_relative_object_center_m'] for m in measurements)+radius+contact_height_offset_m+delta[2]
        if clearance<.004:raise ValueError('Teacher hand path has less than 4 mm table clearance')
        if min(m['minimum_thumb_index_vertex_distance_m'] for m in measurements)<.025:
            raise ValueError('Teacher fingers approach too closely')
        offset=-self.midpoint.copy()+delta;offset[2]+=contact_height_offset_m
        return dict(name=f'opposed_patch_ik_{shape}_gain{gain:g}',
            provenance='privileged_direct_teacher; pure hand FK candidate, physically unverified until scored',
            object_spec={'shape':shape,**({'height_m':.09} if shape=='cylinder' else {})},
            hand_open_rad=hands[0].tolist(),hand_closed_rad=self.solve(.002,patch_offsets_world_m)[0].tolist(),
            palm_quaternion_wxyz=self.quaternion.tolist(),palm_offset_m=offset.tolist(),
            palm_offset_frame='world relative to privileged object center',grip_gain=gain,
            closure_mode='symmetric_fixed_patch_position_ik_with_contact_feedback',
            path_inward_m=coordinates.tolist(),path_hand_rad=np.asarray(hands).tolist(),
            force_target_per_finger_N=float(force_target_N),contact_height_offset_m=contact_height_offset_m,
            patch_offsets_world_m=(np.zeros((2,3)) if patch_offsets_world_m is None else np.asarray(patch_offsets_world_m)).tolist(),
            palm_registration_correction_world_m=delta.tolist(),
            max_inward_m=.005,close_increment_m_per_action=.0004,
            preload_increment_m_per_action=.00015,
            geometric_checks={'fixed_patch_taxel_counts':[int(m.sum()) for m in self.masks],
                'nominal_patch_centers_palm_world_axes_m':self.patch_centers.tolist(),
                'nominal_patch_center_difference_m':np.diff(self.patch_centers,axis=0)[0].tolist(),
                'max_fk_position_error_m':max(errors),'minimum_hand_table_clearance_m':clearance,
                'path_measurements':measurements,
                'scope':'Full hand mesh vertices and fixed taxel-patch FK; vertex distance is not complete triangle self-collision certification; physics collision checks remain mandatory'},
            source_sha256={str(self.urdf):sha(self.urdf),str(self.reference_path):sha(self.reference_path),
                **self.mesh_sources})


def hand_at(candidate,coordinates):
    coordinates=np.asarray(coordinates,float)
    joint_groups=candidate.get('finger_joint_indices',[[0,1,2],[3,4,5]])
    if coordinates.shape!=(len(joint_groups),) or not np.isfinite(coordinates).all():
        raise ValueError('Finite per-finger patch coordinates must match candidate joint groups')
    flat=[index for group in joint_groups for index in group]
    if len(flat)!=len(set(flat)) or any(not isinstance(i,int) or i<0 or i>=12 for i in flat):
        raise ValueError('Finger joint groups must be distinct hand indices')
    knots=np.asarray(candidate['path_inward_m']);path=np.asarray(candidate['path_hand_rad'])
    if np.any(coordinates<knots[0]-1e-8) or np.any(coordinates>knots[-1]+1e-8):
        raise ValueError('Teacher coordinate outside prevalidated path')
    hand=path[0].copy()
    for finger,indices in enumerate(joint_groups):
        for index in indices:hand[index]=np.interp(coordinates[finger],knots,path[:,index])
    return hand
