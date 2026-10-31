"""Isolated compiled contact test, not a grasp or force calibration.

Each sample resets the scene and imposes an initial plate pose/velocity, then
advances real collision/dynamics once. No contact buffers or readings are edited.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
import numpy as np
import genesis as gs

ROOT=Path(__file__).resolve().parents[2]

def cpu(x):return x.detach().cpu().numpy() if hasattr(x,'detach') else np.asarray(x)

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--backend',choices=['cpu','gpu'],default='gpu');a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    gs.init(backend=getattr(gs,a.backend),precision='32',seed=23,logging_level='warning')
    scene=gs.Scene(sim_options=gs.options.SimOptions(dt=1/300,gravity=(0,0,0)),
        rigid_options=gs.options.RigidOptions(noslip_iterations=5),show_viewer=False)
    body=scene.add_entity(gs.morphs.Box(size=(.03,.03,.03),fixed=True),material=gs.materials.Rigid(friction=.8))
    plate=scene.add_entity(gs.morphs.Box(size=(.02,.02,.01),pos=(0,0,.1)),
        material=gs.materials.Rigid(rho=12500.,friction=.8))
    sensors={name:scene.add_sensor(gs.sensors.KinematicTaxel(entity_idx=body.idx,
        probe_local_pos=[[0,0,.015]],probe_radius=.0015,history_length=1,
        normal_stiffness=500.,normal_damping=1.,normal_exponent=1.2,shear_scalar=shear,twist_scalar=0.))
        for name,shear in [('production',2.),('normal_only',0.)]}
    scene.build(n_envs=1)
    records=[]
    for depth in [-.005,-.001,0.,.00005,.0001,.00025,.0005]:
        velocities=[0.] if depth!=.0001 else [0.,-.5,.5]
        for velocity in velocities:
            scene.reset();plate.set_pos([0,0,.020-depth],zero_velocity=True)
            plate.set_dofs_velocity([velocity,0,0,0,0,0]);scene.step()
            contacts=body.get_contacts(is_padded=True);valid=cpu(contacts['valid_mask']).reshape(-1).astype(bool)
            la=cpu(contacts['link_a']).reshape(-1);lb=cpu(contacts['link_b']).reshape(-1)
            fa=cpu(contacts['force_a']).reshape(-1,3);mask=valid&(((la==body.base_link.idx)&(lb==plate.base_link.idx))|((lb==body.base_link.idx)&(la==plate.base_link.idx)))
            fixed_forces=np.where((la[mask]==body.base_link.idx)[:,None],fa[mask],-fa[mask])
            point=np.array([0,0,.015]);pos=cpu(plate.get_pos()).reshape(3)
            vel=cpu(plate.get_vel()).reshape(3);ang=cpu(plate.get_ang()).reshape(3)
            point_vel=vel+np.cross(ang,point-pos)
            force={name:cpu(sensor.read().force).reshape(3).tolist() for name,sensor in sensors.items()}
            shear=np.array(force['production'])-force['normal_only']
            records.append({'imposed_overlap_m':depth,'imposed_tangent_velocity_m_s':velocity,
                'plate_position_after_step_m':pos.tolist(),'plate_point_velocity_after_step_m_s':point_vel.tolist(),
                'physical_contact_count':int(mask.sum()),'physical_force_on_sensor_body_N':fixed_forces.sum(0).tolist(),
                'taxel_estimate_nominal_N':force,'isolated_shear_term_nominal_N':shear.tolist(),
                'shear_power_on_relative_motion_proxy':float(shear@(-point_vel))})
    moving=[r for r in records if r['imposed_tangent_velocity_m_s']!=0]
    checks={'no_contact_readings_zero':all(np.linalg.norm(r['taxel_estimate_nominal_N']['production'])<1e-8 for r in records if not r['physical_contact_count']),
        'sliding_cases_really_contact_and_move':all(r['physical_contact_count'] and abs(r['plate_point_velocity_after_step_m_s'][0])>.01 for r in moving),
        'tactile_shear_same_direction_as_physical_friction_on_sensor':all(r['isolated_shear_term_nominal_N'][0]*r['physical_force_on_sensor_body_N'][0]>0 for r in moving),
        'shear_is_dissipative_on_contact_pair':all(r['shear_power_on_relative_motion_proxy']<=1e-9 for r in moving)}
    source=ROOT/'Genesis/genesis/engine/sensors/kinematic_tactile.py'
    out={'scope':__doc__,'backend':a.backend,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'physics_dt_s':1/300,'noslip_iterations':5,'mass_kg':float(cpu(plate.get_mass()).reshape(-1)[0]),
        'checks':checks,'records':records}
    (a.output/'result.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(checks),flush=True);gs.destroy()

if __name__=='__main__':main()
