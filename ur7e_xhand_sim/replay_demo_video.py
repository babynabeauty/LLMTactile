"""Independently replay V1 recorded joint targets and render an MP4 with three observation cameras and a hand detail.

No IK, feedback closure, object teleportation or target edits during replay.
The video plays at simulation time (10 fps); metrics use every 60 Hz record.
"""
import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation

from environment import Sandbox
from v1_demo import object_normals, sha
from residual_rl.v1_task import V1Config, hand_shape_joints, opposing_support, score_lift
from contact_guard import assess_contacts


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--demo',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    demo=json.loads((args.demo/'demo.json').read_text())
    trajectory=args.demo/demo['trajectory']
    if sha(trajectory)!=demo['trajectory_sha256']:raise ValueError('Trajectory hash mismatch')
    args.output.mkdir(parents=True,exist_ok=False)
    sim=Sandbox(args.output/'runtime',render=False,cube_mass_kg=.05,cube_friction=.8,
        robot_friction=.8,object_spec=demo['object_spec'],physics_dt_s=1/300,control_substeps=5,
        noslip_iterations=5,deterministic=True)
    sim.cube.set_pos(demo['object_initial_position_m'],zero_velocity=True)
    sim.cube.set_quat(demo['object_initial_quaternion_wxyz'],zero_velocity=True)
    gain_indices=np.asarray(hand_shape_joints(demo['hand_shape']))+6
    rows=[json.loads(line) for line in trajectory.open()]
    evaluations=[];gain=None;anchor=None;max_slip=0.;pre_force=0.;faults=[];samples=[]
    video=args.output/'replay.mp4'
    writer=imageio.get_writer(video,fps=10,codec='libx264',quality=8,macro_block_size=1)
    try:
        for i,row in enumerate(rows):
            if gain!=row['grip_gain']:
                gain=row['grip_gain'];kp=np.asarray(sim.kp,float).copy();kv=np.asarray(sim.kv,float).copy()
                kp[gain_indices]*=gain;kv[gain_indices]*=np.sqrt(gain)
                sim.robot.set_dofs_kp(kp,sim.indices);sim.robot.set_dofs_kv(kv,sim.indices)
            target=np.array(row['sent_target_rad'])
            sim.step({'actions':dict(zip(sim.names,((target-sim.offset)/sim.scale).tolist())),'steps':1})
            e=sim.evaluate();forces=object_normals(e);phase=row['phase']
            pos=sim.array(sim.wrist.get_pos()).reshape(3);q=sim.array(sim.wrist.get_quat()).reshape(4)
            R=Rotation.from_quat(q[[1,2,3,0]]).as_matrix()
            relative=R.T@(np.array(e['object_position_m'])-pos)
            if phase=='test_lift' and anchor is None:anchor=relative.copy()
            if anchor is not None:max_slip=max(max_slip,float(np.linalg.norm(relative-anchor)))
            if phase in ('settle','preshape','approach','align'):pre_force=max(pre_force,float(forces[:6].sum()))
            guard=assess_contacts(e['solver_contacts'],5.)
            if guard['stop']:faults.append({'step':i+1,'reason':guard['reason']})
            if anchor is not None:
                o=np.asarray(e['object_quaternion_wxyz'])
                samples.append((relative,R.T@Rotation.from_quat(o[[1,2,3,0]]).as_matrix(),e['bottom_clearance_m'],
                    bool(opposing_support(forces[None],.05)[0]),bool(guard['stop'])))
            evaluations.append({'step':i+1,'phase':phase,'bottom_clearance_m':e['bottom_clearance_m'],
                'region_normal_N':forces.tolist(),'slip_m':max_slip,'object_position_m':e['object_position_m']})
            if i%6==0 or i==len(rows)-1:
                mount=sim.wrist_camera_mount
                sim.wrist_camera.set_pose(pos=pos+R@np.asarray(mount['position_m']),
                    lookat=pos+R@np.asarray(mount['lookat_m']),up=R@np.asarray(mount['up']))
                center=pos+R@np.array([0.,0.,.09])
                sim.close_camera.set_pose(pos=center+R@np.array([.14,.42,.10]),
                    lookat=center,up=R@np.array([-1.,0.,0.]))
                canvas=Image.new('RGB',(1280,960),'#121922');draw=ImageDraw.Draw(canvas)
                for camera,label,xy in [(sim.camera,'FRONT',(0,40)),(sim.rear_camera,'LEFT',(640,40)),(sim.wrist_camera,'WRIST',(0,500)),(sim.close_camera,'HAND DETAIL',(640,500))]:
                    rgb=sim.array(camera.render(rgb=True)[0]).squeeze().astype('uint8')
                    im=Image.fromarray(rgb);im.thumbnail((632,414))
                    canvas.paste(im,(xy[0]+(640-im.width)//2,xy[1]+26));draw.text((xy[0]+12,xy[1]+5),label,fill='white')
                draw.text((16,12),f"CUBOID / {demo['hand_shape'].upper()}   {phase}   t={(i+1)/60:.2f}s   1x simulation time",fill='white')
                draw.text((660,930),f'Clearance: {e["bottom_clearance_m"]*1000:.1f} mm | Max slip: {max_slip*1000:.2f} mm',fill='white')
                writer.append_data(np.asarray(canvas))
                if phase=='hold':canvas.save(args.output/'held.png')
            if (i+1)%300==0:print(f'{demo["hand_shape"]}: {i+1}/{len(rows)} records, {phase}',flush=True)
    finally:
        writer.close()
        sim.scene.destroy()
    # Same rules as v1_demo: full 3 s hold, V1 outcome from the first lifting command on.
    hold=[e for e in evaluations if e['phase']=='hold'][-180:]
    f=np.array([e['region_normal_N'] for e in hold])
    outcome=score_lift(samples,1/60,V1Config())
    summary={'success':bool(outcome['reason']=='success' and len(hold)==180 and pre_force<=.05 and not faults),
        'v1_outcome':outcome['reason'],'longest_stable_s':outcome['longest_stable_s'],
        'hand_shape':demo['hand_shape'],'trajectory_sha256':sha(trajectory),'control_steps':len(rows),
        'preclose_max_object_normal_N':pre_force,'hold_min_clearance_m':min((e['bottom_clearance_m'] for e in hold),default=None),
        'max_slip_m':max_slip,'faults':faults,
        'hold_mean_region_normal_N':f.mean(0).tolist() if len(f) else [],
        'hold_contact_fraction_by_region':(f>.05).mean(0).tolist() if len(f) else [],
        'region_order':['thumb','index','middle','ring','pinky','palm','arm'],
        'video_fps':10,'video':str(video.resolve()),'independent_replay':True}
    (args.output/'validation.json').write_text(json.dumps(summary,indent=2)+'\n')
    with (args.output/'metrics.jsonl').open('w') as out:
        for e in evaluations:out.write(json.dumps(e)+'\n')
    print(json.dumps(summary),flush=True)
    raise SystemExit(0 if summary['success'] else 1)

if __name__=='__main__':main()
