"""Camera-only import from the user-supplied RaiSim configuration."""
import hashlib
from pathlib import Path
import numpy as np
import yaml
from scipy.spatial.transform import Rotation


def load_camera_views():
    path=Path(__file__).parent/'sources/cameras/cfg_reg_xhand.yaml'
    data=yaml.safe_load(path.read_text())['environment']['realtime_obj_pc']
    # Source URDF pedestal is 0.9 m; this scene mounts the same base at 0.2 m.
    # Source external positions/quaternions already include the x/y convention.
    shift=np.array([0.,0.,-.7])
    source={'file':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'world_translation_from_source_m':shift.tolist(),
            'note':'camera fields only; no hardware, controller, hand shape, or material settings imported'}
    def optics(spec,width):
        fx,fy=float(spec['fov_x_deg']),float(spec['fov_y_deg'])
        near,far=float(spec['near']),float(spec['far'])
        if not (0<fx<179 and 0<fy<179 and 0<near<far):raise ValueError('Invalid camera optics')
        # YAML width/height are low-resolution point-cloud ray grids. Use high
        # resolution RGB, selecting aspect ratio to preserve both FOVs to <1 px.
        height=round(width*np.tan(np.deg2rad(fy/2))/np.tan(np.deg2rad(fx/2)))
        actual_fx=float(np.rad2deg(2*np.arctan(width/height*np.tan(np.deg2rad(fy/2)))))
        return {'resolution':[width,height],'fov_y_deg':fy,'fov_x_deg_requested':fx,
                'fov_x_deg_effective':actual_fx,'near_m':near,'far_m':far,'rgb_far_m':20.,
                'source_ray_grid':[spec['width'],spec['height']]}
    views={}
    for name,key,role in [('scene_front','external','front'),('scene_left','external_2','left')]:
        spec=data[key];pos=np.asarray(spec['position'],float)+shift
        R=Rotation.from_quat(spec['quaternion_xyzw']).as_matrix()
        # Optical +Z forward, +Y down; Genesis takes a world up vector.
        look=pos+R[:,2];up=-R[:,1]
        np.testing.assert_allclose(look,np.asarray(spec['look_at'])+shift,atol=1e-6)
        views[name]={**optics(spec,1000),'frame':'world','role':role,
            'position_m':pos.tolist(),'lookat_m':look.tolist(),'up':up.tolist(),
            'optical_quaternion_xyzw':spec['quaternion_xyzw'],'source_key':key}
    pos=np.asarray(data['hand_to_camera_translation'],float)
    rpy=data['hand_to_camera_rotation_rpy'];R=Rotation.from_euler('xyz',rpy).as_matrix()
    views['wrist']={**optics(data['wrist'],640),'frame':'right_hand_link','role':'wrist',
        'position_m':pos.tolist(),'lookat_m':(pos+R[:,2]).tolist(),'up':(-R[:,1]).tolist(),
        'optical_rotation_rpy_rad':rpy,'source_key':'wrist',
        'calibration':'user YAML hand frame mapped to right_hand_link; not independently re-calibrated'}
    return views,source


def camera_kwargs(view):
    return {'res':tuple(view['resolution']),'pos':view['position_m'],
            'lookat':view['lookat_m'],'up':view['up'],'fov':view['fov_y_deg'],
            # YAML near/far bound the ray-depth point cloud, not the RGB image.
            'near':view['near_m'],'far':view['rgb_far_m'],'GUI':False}
