"""Privileged direct teacher for sphere and upright cylindrical grasps.

Uses simulator contact only in the teacher. All actual commands and 60Hz tactile
samples are recorded by DirectTeacher; no object state is written or attached.
"""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np

from build_assets import HERE
from direct_teacher import DirectTeacher,sha,write
from direct_teacher_geometry import OpposedPatchGeometry,hand_at
from environment import Sandbox



def force_coordinates(coordinates, loads, desired, candidate):
    """Pure bounded teacher force update; no geometry or simulator mutations."""
    coordinates=np.asarray(coordinates,float).copy()
    loads=np.asarray(loads,float);desired=np.asarray(desired,float)
    if coordinates.shape!=loads.shape or loads.shape!=desired.shape or not np.isfinite(np.r_[coordinates,loads,desired]).all():
        raise ValueError('Finite aligned force-controller vectors required')
    step=float(candidate['preload_increment_m_per_action'])
    lower=float(candidate['path_inward_m'][0]);upper=float(candidate['max_inward_m'])
    mode=candidate.get('force_control_mode','tighten_only')
    if mode not in ('tighten_only','bidirectional_target') or not 0<step<=.001:
        raise ValueError('Unknown force mode or unbounded force step')
    for finger in range(len(coordinates)):
        if mode=='bidirectional_target':
            if loads[finger]<desired[finger]*.95:
                coordinates[finger]=min(coordinates[finger]+step,upper)
            elif loads[finger]>desired[finger]*1.12:
                coordinates[finger]=max(coordinates[finger]-step,lower)
        elif loads[finger]<desired[finger]:
            coordinates[finger]=min(coordinates[finger]+step,upper)
    return coordinates



def allocate_normal_targets(normals,thumb_force_N,xy_force_N,min_force_N=.4,max_force_N=4.):
    """Allocate positive three-contact normal magnitudes for a bounded XY wrench."""
    normals=np.asarray(normals,float);xy=np.asarray(xy_force_N,float)
    if normals.shape!=(3,3) or xy.shape!=(2,) or not np.isfinite(np.r_[normals.ravel(),xy]).all():
        raise ValueError('Finite three-contact normals and XY force required')
    lengths=np.linalg.norm(normals,axis=1)
    if np.min(lengths)<1e-6:raise ValueError('Contact normal magnitude is zero')
    normals=normals/lengths[:,None]
    matrix=normals[1:,:2].T
    if np.linalg.cond(matrix)>20:raise ValueError('Contact geometry cannot allocate XY wrench reliably')
    solved=np.linalg.solve(matrix,xy-thumb_force_N*normals[0,:2])
    desired=np.clip(np.r_[thumb_force_N,solved],min_force_N,max_force_N)
    return desired,normals.T@desired


def run_curved_teacher(teacher):
    t,s,c=teacher,teacher.sim,teacher.candidate
    quat=np.asarray(c['palm_quaternion_wxyz']);opened=np.asarray(c['hand_open_rad'])
    for _ in range(10):t.servo(t.home_pos,t.home_quat,s.home[6:],'settle')
    center=s.array(s.cube.get_pos()).reshape(3)
    align=center+np.asarray(c['palm_offset_m'])
    t.alignment=align;t.grasp_quat=quat;t.open_hand=opened.copy()
    write(t.out/'registration.json',dict(source='privileged simulator object center; direct teacher only',
        object_center_world_m=center.tolist(),align_palm_world_m=align.tolist(),
        patch_center_offset_m=c['palm_offset_m']))
    pre=align+np.array([0,0,.12]);high=pre.copy();high[2]=max(pre[2],t.home_pos[2])
    t.move('preshape',t.home_pos,quat,opened,2.5);t.stage('open')
    t.move('approach',high,quat,opened,3.);t.move('approach',pre,quat,opened,1.)
    t.move('align',align,quat,opened,2.);t.stage('aligned')
    t.set_gain(c['grip_gain'])
    fingers=tuple(c.get('contact_fingers',['thumb','index']))
    if not fingers or len(set(fingers))!=len(fingers):raise ValueError('Unique contact fingers required')
    coordinates=np.full(len(fingers),c['path_inward_m'][0],float)
    confirmed=0;events=[];upper=float(c['max_inward_m'])
    initial_align=align.copy()

    def tracking_position(position):
        position=np.asarray(position,float).copy()
        bound=float(c.get('teacher_planar_tracking_bound_m',0.))
        if bound>0:
            object_now=s.array(s.cube.get_pos()).reshape(3)
            desired_xy=object_now[:2]+np.asarray(c['palm_offset_m'])[:2]
            bounded_xy=initial_align[:2]+np.clip(desired_xy-initial_align[:2],-bound,bound)
            delta=np.clip(bounded_xy-align[:2],-.001,.001)
            align[:2]+=delta
            position[:2]=align[:2]
        return position
    for tick in range(90):
        loads=t.finger_loads();load=np.array([loads[f] for f in fingers])
        for finger in range(len(fingers)):
            # A contacted finger holds its target while the other catches up.
            if load[finger]<.10:
                coordinates[finger]=min(coordinates[finger]+c['close_increment_m_per_action'],upper)
        hand=hand_at(c,coordinates)
        t.servo(tracking_position(align),quat,hand,'close')
        load=np.array([t.finger_loads()[f] for f in fingers])
        confirmed=confirmed+1 if np.min(load)>=.10 else 0
        events.append(dict(phase='close',step=s.steps,coordinates_m=coordinates.tolist(),physical_finger_load_N=load.tolist()))
        if confirmed>=3:break
    else:
        write(t.out/'teacher_contact_events.json',events)
        raise ValueError('No persistent physical required-finger contact in bounded coordinated closure')
    t.stage('light_contact')
    desired=np.array([c.get('force_target_by_finger_N',{}).get(f,c['force_target_per_finger_N']) for f in fingers],float)
    if not np.isfinite(desired).all() or np.any(desired<=0) or np.any(desired>4):
        raise ValueError('Per-finger target forces must be finite and within(0,4]N')

    allocation_events=[];last_normals=None

    def adjust_force():
        nonlocal last_normals
        load=np.array([t.finger_loads()[f] for f in fingers])
        active_desired=desired.copy()
        if c.get('force_target_allocation')=='contact_normal_xy_restore':
            if fingers!=('thumb','index','middle'):raise ValueError('XY allocation requires tripod order')
            e=t.evals[-1];normals=np.zeros((3,3));weights=np.zeros(3)
            links={row['finger']:row['link_name'] for row in s.layouts}
            for contact in e['solver_contacts']:
                if 'cube' not in (contact['entity_a'],contact['entity_b']):continue
                sign=1 if contact['entity_a']=='cube' else -1
                for finger,name in enumerate(fingers):
                    if links[name] in (contact['link_a'],contact['link_b']):
                        weight=max(float(contact.get('normal_force_N',0.)),0.)
                        normals[finger]+=sign*weight*np.asarray(contact['normal_world']);weights[finger]+=weight
            measured=bool(np.min(weights)>.01)
            if measured:last_normals=normals/weights[:,None]
            config=c.get('normal_allocation',{})
            position=np.asarray(e['object_position_m'])[:2]
            velocity=np.zeros(2)
            if len(t.evals)>=5:
                previous=t.evals[-5]
                interval=e['sample_time_s']-previous['sample_time_s']
                velocity=(position-np.asarray(previous['object_position_m'])[:2])/interval
            restore=-float(config.get('position_kp_N_m',20.))*(position-center[:2])-float(config.get('velocity_kd_N_s_m',.6))*velocity
            maximum=float(config.get('max_restoring_force_N',.4))
            restore*=min(1.,maximum/max(float(np.linalg.norm(restore)),1e-12))
            allocated=False;wrench=None
            if last_normals is not None:
                try:
                    active_desired,wrench=allocate_normal_targets(last_normals,float(desired[0]),restore)
                    allocated=True
                except ValueError:pass
            allocation_events.append(dict(step=s.steps,phase=t.phase,
                source='privileged objectXY/velocity and solver contact normals; teacher only',
                measured_all_three_normals=measured,allocation_applied=allocated,
                normals_on_object=None if last_normals is None else last_normals.tolist(),
                object_xy_error_m=(position-center[:2]).tolist(),object_xy_velocity_m_s=velocity.tolist(),
                desired_xy_wrench_N=restore.tolist(),normal_force_targets_N=active_desired.tolist(),
                allocated_normal_wrench_N=None if wrench is None else wrench.tolist()))
            write(t.out/'force_allocation.json',allocation_events)
        coordinates[:]=force_coordinates(coordinates,load,active_desired,c)
        return hand_at(c,coordinates)

    for _ in range(35):
        hand=adjust_force();t.servo(tracking_position(align),quat,hand,'preload')
        events.append(dict(phase='preload',step=s.steps,coordinates_m=coordinates.tolist(),
            physical_finger_load_N=[t.finger_loads()[f] for f in fingers]))
    t.stage('closed')
    if min(t.finger_loads()[f] for f in fingers)<.05:
        write(t.out/'teacher_contact_events.json',events)
        raise ValueError('Physical required-finger contact lost during coordinated preload')

    if c.get('lift_control_mode')=='object_progress_limited':
        reference_palm=s.array(s.wrist.get_pos()).reshape(3)
        reference_object_z=float(t.evals[-1]['object_position_m'][2])
        command=reference_palm.copy();progress_events=[]
        lead=float(c.get('palm_max_lead_m',.004))
        max_step=float(c.get('progress_lift_step_m',.0005))
        if not .001<=lead<=.006 or not 0<max_step<=.001:
            raise ValueError('Progress lift exceeds bounded lead or step')

        def progress_lift(phase,clearance_goal,timeout_s):
            nonlocal command,hand
            reached=0
            for _ in range(int(round(timeout_s*15))):
                actual_object_z=float(t.evals[-1]['object_position_m'][2])
                object_progress=actual_object_z-reference_object_z
                allowed_z=reference_palm[2]+object_progress+lead
                # The hand cannot run ahead of the actual object; downward
                # correction is permitted only within the same bounded lead.
                command[2]=min(command[2]+max_step,allowed_z,reference_palm[2]+.12)
                if command[2]<reference_palm[2]-.005:
                    raise ValueError('Object fell below bounded progress-lift region')
                hand=adjust_force();t.servo(command,quat,hand,phase)
                loads=[t.finger_loads()[f] for f in fingers]
                clearance=float(t.evals[-1]['bottom_clearance_m'])
                progress_events.append(dict(step=s.steps,phase=phase,
                    source='privileged true object height; teacher only',
                    initial_object_z_m=reference_object_z,object_z_m=t.evals[-1]['object_position_m'][2],
                    commanded_palm_z_m=float(command[2]),allowed_palm_z_before_command_m=float(allowed_z),
                    object_bottom_clearance_m=clearance,physical_finger_load_N=loads,
                    coordinates_m=coordinates.tolist()))
                write(t.out/'progress_lift_events.json',progress_events)
                events.append(dict(phase=phase,step=s.steps,coordinates_m=coordinates.tolist(),physical_finger_load_N=loads))
                reached=reached+1 if clearance>=clearance_goal and min(loads)>=.10 else 0
                if reached>=3:return
            write(t.out/'teacher_contact_events.json',events)
            raise ValueError('Bounded object-progress lift timed out in '+phase)

        progress_lift('test_lift',.010,float(c.get('test_lift_timeout_s',20.)))
        t.stage('test_lifted')
        progress_lift('lift',.085,float(c.get('full_lift_timeout_s',40.)))
        t.stage('lifted')
        # Demonstrate passive retention after the final measured placement.
        # No object feedback or new hand closure is used during the final hold.
        frozen_hand=hand.copy();frozen_palm=command.copy()
        for _ in range(45):
            t.servo(frozen_palm,quat,frozen_hand,'hold')
            events.append(dict(phase='hold',step=s.steps,coordinates_m=coordinates.tolist(),
                physical_finger_load_N=[t.finger_loads()[f] for f in fingers]))
        t.stage('held');write(t.out/'teacher_contact_events.json',events)
        return

    def lift_segment(phase,goal,duration_s):
        start=s.array(s.wrist.get_pos()).reshape(3)
        for i in range(1,int(round(duration_s*15))+1):
            u=i/(duration_s*15);f=u*u*(3-2*u)
            hand=adjust_force();t.servo(tracking_position(start+f*(goal-start)),quat,hand,phase)
            events.append(dict(phase=phase,step=s.steps,coordinates_m=coordinates.tolist(),
                physical_finger_load_N=[t.finger_loads()[finger] for finger in fingers]))
        for _ in range(8):
            hand=adjust_force();t.servo(tracking_position(goal),quat,hand,phase)

    lift_segment('test_lift',align+np.array([0,0,.02]),2.)
    t.stage('test_lifted')
    clearance=float(t.evals[-1]['bottom_clearance_m'])
    center_progress=float(t.evals[-1]['object_position_m'][2]-center[2])
    support={f:float(t.finger_loads()[f]) for f in fingers}
    followed=clearance>=.005 or (center_progress>=.005 and min(support.values())>=.10)
    write(t.out/'test_lift_gate.json',dict(bottom_clearance_m=clearance,
        center_progress_m=center_progress,physical_finger_load_N=support,passed=followed,
        criterion='bottom>=5mm OR centerprogress>=5mm AND all required-finger physical contacts>=0.1N; final hold still exact bottom>=50mm for3s'))
    if not followed:
        write(t.out/'teacher_contact_events.json',events)
        raise ValueError('Object did not follow coordinated test lift')
    lifted=align+np.array([0,0,.10])
    lift_segment('lift',lifted,3.);t.stage('lifted')
    for _ in range(45):
        hand=adjust_force();t.servo(tracking_position(lifted),quat,hand,'hold')
        events.append(dict(phase='hold',step=s.steps,coordinates_m=coordinates.tolist(),
            physical_finger_load_N=[t.finger_loads()[f] for f in fingers]))
    t.stage('held')
    write(t.out/'teacher_contact_events.json',events)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shape',choices=['sphere','cylinder','hex_prism','cuboid'],required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--gain',type=float,default=1.)
    parser.add_argument('--force-target',type=float,default=1.2)
    parser.add_argument('--contact-height',type=float,default=0.)
    parser.add_argument('--candidate-json',type=Path)
    parser.add_argument('--track-object',action='store_true',help='Privileged teacher XY tracking, bounded to25mm and1mm/action')
    parser.add_argument('--geometry-only',action='store_true')
    args=parser.parse_args()
    candidate=(json.loads(args.candidate_json.read_text()) if args.candidate_json else
        OpposedPatchGeometry().candidate(args.shape,args.gain,args.force_target,args.contact_height))
    if args.track_object:
        candidate['teacher_planar_tracking_bound_m']=.025
        candidate['teacher_planar_tracking_max_step_m']=.001
        candidate['teacher_planar_tracking_source']='privileged solver object center; teacher only'
    if args.geometry_only:
        args.output.parent.mkdir(parents=True,exist_ok=True);write(args.output,candidate)
        print(json.dumps({'candidate':str(args.output),'checks':candidate['geometric_checks']}));return
    sim=Sandbox(output_dir=args.output.parent/f'runtime_{args.shape}',render=False,
        cube_mass_kg=.05,cube_friction=.8,robot_friction=.8,
        object_spec=candidate['object_spec'],physics_dt_s=1/300,control_substeps=5)
    teacher=DirectTeacher(sim,args.output,candidate)
    for source in (Path(__file__),HERE/'direct_teacher_geometry.py',Path('/home/sai/zsq/vlm-policy/urdf_frames.py')):
        teacher.manifest['source_sha256'][str(source.resolve())]=sha(source)
        target=teacher.out/'source_snapshot'/'curved_teacher'/source.name
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
    write(teacher.out/'manifest.json',teacher.manifest)
    score=teacher.run(run_curved_teacher)
    print(json.dumps(score),flush=True)
    raise SystemExit(0 if score['success'] else 1)


if __name__=='__main__':main()
