"""Independent virtual compression fixture: no robot and no FEM state writes.

Drive a prismatic press with PD, compress a 45 mm block, release, and record
mesh validity/recovery. Repeat at different dt and tet sizes for convergence.
"""
import argparse,json,os,time
from pathlib import Path
os.environ.setdefault('PYOPENGL_PLATFORM','egl')
import numpy as np
import genesis as gs
from fem_diagnostics import use_physics_surface,mesh_quality
from ipc_support import weld_ipc_triangle_soups


def main():
    p=argparse.ArgumentParser();p.add_argument('--dt',type=float,default=.005)
    p.add_argument('--maxvolume',type=float,default=1e-7);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();out=a.output;out.mkdir(parents=True,exist_ok=False)
    fixture=out/'press.urdf'
    fixture.write_text('''<robot name="validation_press">
<link name="base"><inertial><mass value="10"/><inertia ixx=".1" ixy="0" ixz="0" iyy=".1" iyz="0" izz=".1"/></inertial><visual><origin xyz="0 0 -.01"/><geometry><box size=".18 .18 .02"/></geometry></visual><collision><origin xyz="0 0 -.01"/><geometry><box size=".18 .18 .02"/></geometry></collision></link>
<link name="platen"><inertial><mass value="1"/><inertia ixx=".001" ixy="0" ixz="0" iyy=".001" iyz="0" izz=".002"/></inertial><visual><geometry><box size=".12 .12 .01"/></geometry></visual><collision><geometry><box size=".12 .12 .01"/></geometry></collision></link>
<joint name="press" type="prismatic"><parent link="base"/><child link="platen"/><origin xyz="0 0 .06"/><axis xyz="0 0 1"/><limit lower="-.04" upper=".02" effort="500" velocity=".2"/></joint></robot>''')
    gs.init(backend=gs.gpu,precision='32',seed=42,logging_level='warning')
    scene=gs.Scene(sim_options=gs.options.SimOptions(dt=a.dt),show_viewer=False,
        coupler_options=gs.options.IPCCouplerOptions(contact_d_hat=.0005,
            newton_tolerance=.005,newton_translation_tolerance=.005,linear_system_tolerance=1e-6,
            n_linesearch_iterations=32,enable_rigid_rigid_contact=False,enable_rigid_ground_contact=False))
    press=scene.add_entity(gs.morphs.URDF(file=str(fixture.resolve()),fixed=True),
        material=gs.materials.Rigid(coup_type='two_way_soft_constraint',coup_friction=.8))
    soft=scene.add_entity(gs.morphs.Box(size=(.045,)*3,pos=(0,0,.024),maxvolume=a.maxvolume,nobisect=False),
        material=gs.materials.FEM.Elastic(E=1e5,nu=.45,rho=1100,model='stable_neohookean',friction_mu=.8),
        surface=gs.surfaces.Default(color=(.95,.4,.65,1)))
    use_physics_surface(soft);weld_ipc_triangle_soups(scene.sim.coupler)
    camera=scene.add_camera(pos=(.16,-.18,.12),lookat=(0,0,.025),res=(640,480),GUI=False)
    scene.build(n_envs=1)
    press.set_dofs_kp([2e5]);press.set_dofs_kv([1500.]);press.set_dofs_force_range([-500.],[500.])
    array=lambda x:x.detach().cpu().numpy() if hasattr(x,'detach') else np.asarray(x)
    rest=array(soft.init_positions).reshape(-1,3).astype(float);tets=array(soft.get_el2v()).astype(int)
    x=rest[tets];Dm=np.stack([x[:,i]-x[:,3] for i in range(3)],axis=-1);inv=np.linalg.inv(Dm);V=np.abs(np.linalg.det(Dm))/6
    np.savez_compressed(out/'mesh.npz',rest=rest,tets=tets,boundary=soft._surface_tri_np)
    (out/'config.json').write_text(json.dumps({'dt':a.dt,'maxvolume':a.maxvolume,'mesh':mesh_quality(rest,tets),
        'E_Pa':1e5,'nu':.45,'rho':1100,'friction':.8,'fixture':'PD prismatic press, no object state writes',
        'deformation_threshold_stops':False,'nominal_max_compression':(0.045-0.025)/.045},indent=2))
    rows=[];start=time.monotonic();stages={}
    with (out/'metrics.jsonl').open('w',buffering=1) as log:
        for i in range(round(4./a.dt)):
            now=(i+1)*a.dt
            if now<=.5:phase='settle';q=0.
            elif now<=1.5:phase='compress';q=-.03*(now-.5)
            elif now<=2.:phase='hold';q=-.03
            elif now<=3.:phase='release';q=-.03*(3.-now)
            else:phase='recover';q=0.
            press.control_dofs_position([q]);scene.step()
            pos=array(soft.get_state().pos).reshape(-1,3);x=pos[tets]
            F=np.stack([x[:,j]-x[:,3] for j in range(3)],axis=-1)@inv;J=np.linalg.det(F)
            actual=float(array(press.get_dofs_position()).ravel()[0]);vel=float(array(press.get_dofs_velocity()).ravel()[0])
            gap=.055+actual
            row={'t':now,'phase':phase,'q_target':q,'q_actual':actual,'plate_gap_m':gap,
                'min_J':float(J.min()),'inverted':int((J<=0).sum()),'volume_ratio':float((J*V).sum()/V.sum()),
                'height_m':float(np.ptp(pos[:,2])),'bottom_m':float(pos[:,2].min()),
                'plate_penetration_m':float(max(0,-pos[:,2].min(),pos[:,2].max()-gap)),
                'servo_force_N':float(np.clip(2e5*(q-actual)-1500*vel,-500,500)),
                'centroid_m':pos.mean(0).tolist(),'finite':bool(np.isfinite(pos).all())}
            rows.append(row);log.write(json.dumps(row)+'\n')
            if any(abs(now-t)<a.dt*.25 for t in (.5,1.5,2.,3.,4.)):
                name=f'{now:.1f}s';np.savez_compressed(out/f'{name}.npz',positions=pos)
                from PIL import Image
                rgb=array(camera.render(rgb=True)[0]);rgb=rgb[0] if rgb.ndim==4 else rgb
                Image.fromarray(rgb.astype(np.uint8)).save(out/f'{name}.png')
                stages[name]=row;print(json.dumps(row),flush=True)
    summary={'min_J':min(r['min_J'] for r in rows),'max_inverted':max(r['inverted'] for r in rows),
        'max_plate_penetration_m':max(r['plate_penetration_m'] for r in rows),
        'final':rows[-1],'stages':stages,'wall_seconds':time.monotonic()-start}
    (out/'summary.json').write_text(json.dumps(summary,indent=2));print('COMPLETE',json.dumps(summary),flush=True)
    scene.destroy()

if __name__=='__main__':main()
