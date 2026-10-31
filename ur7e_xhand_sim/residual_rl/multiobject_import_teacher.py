"""Freeze direct simulation-teacher evidence and independently verify replay.

The teacher may use privileged geometry. This importer does not describe it as
an Astra demonstration, an API decision, or a learned policy. Conversion is
CPU-only; ``verify`` explicitly builds a fresh 300/60/15 Hz simulation.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, fields
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from ur7e_xhand_sim.build_assets import ARM_JOINTS, HAND_JOINTS, FINGERS
from .core import ROOT, Trajectory
from .multiobject_geometry import ObjectSpec
from .train import write_json
from .solver_profile import normalize_solver_profile, validate_solver_readback, read_solver_readback
from .tactile_provenance import (audit_source_tactile_model, current_tactile_model,
    tactile_models_match, legacy_tactile_model)

JOINT_ORDER=ARM_JOINTS+HAND_JOINTS
HOME_DEGREES=[0.,-89.95,89.96,-89.96,-89.95,-44.98]
PHASE_ORDER=('settle','preshape','approach','align','close','preload','test_lift','lift','hold')
SOURCE_PD={'kp':[4500,4500,3500,2000,2000,2000]+[3.]*12,
           'kv':[450,450,350,200,200,200]+[.08]*12}
TEACHER_DRIVERS=('direct_teacher_search','direct_teacher_frozen_replay')
HISTORICAL_FILES=('control.jsonl','evaluation.jsonl','raw_tactile.npz',
                  'manifest.json','object_scene.json','score.json')


def read_json(path):
    value=json.loads(Path(path).read_text())
    if not isinstance(value,dict):raise ValueError(f'Expected JSON object: {path}')
    return value


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


def spec_from_record(record):
    return ObjectSpec.from_value({f.name:record[f.name] for f in fields(ObjectSpec) if f.name in record})


def finite_array(value, shape, name):
    result=np.asarray(value,dtype=float)
    if result.shape!=shape or not np.isfinite(result).all():
        raise ValueError(f'{name} must be finite with shape {shape}')
    return result


def audit_material(record,spec):
    if not np.isclose(record['object_engine_mass_kg'],.05,rtol=0,atol=1e-7):
        raise ValueError('Teacher material must be physically 50g')
    np.testing.assert_allclose(record['object_engine_inertia_kg_m2'],.05*spec.inertia_per_kg,rtol=2e-5,atol=1e-10)
    np.testing.assert_allclose(record['object_engine_density_kg_m3'],.05/spec.volume_m3,rtol=2e-5)
    np.testing.assert_allclose(record['object_bounds_readback_m'],spec.bounds_size_m,rtol=0,atol=1e-7)


def audit_scene(scene,manifest,spec):
    for value in (scene['object_spec'],manifest['object_spec']):
        if asdict(spec_from_record(value))!=asdict(spec):
            raise ValueError('Recorded object geometry differs from the requested ObjectSpec')
    if scene['joint_order']!=JOINT_ORDER:raise ValueError('Teacher joint order differs')
    for key,expected in (('kp',SOURCE_PD['kp']),('kv',SOURCE_PD['kv']),('home_arm_degrees',HOME_DEGREES)):
        np.testing.assert_allclose(scene[key],expected,rtol=0,atol=1e-7)
    if scene.get('silicone') is not None or scene['table_top_z_m']!=.2:
        raise ValueError('Teacher must use the declared rigid tabletop scene')
    np.testing.assert_allclose(scene['object_spec']['initial_position_m'],spec.initial_position(),rtol=0,atol=1e-7)
    if scene['materials']['cube_friction']!=.8 or scene['materials']['robot_friction']!=.8:
        raise ValueError('Teacher easiest contact friction must be .8')
    audit_material(scene['materials'],spec)
    dt=float(manifest['control_dt_s']);physics_dt=float(manifest['physics_dt_s'])
    if not np.isclose(dt,1/60,rtol=0,atol=1e-10) or not np.isclose(physics_dt,1/300,rtol=0,atol=1e-10):
        raise ValueError('Direct teacher requires 300Hz physics and 60Hz logged samples')
    if not np.isclose(scene['control_dt_s'],dt,rtol=0,atol=1e-10) or not np.isclose(scene['physics_dt_s'],physics_dt,rtol=0,atol=1e-10):
        raise ValueError('Manifest and actual scene clocks disagree')
    return dt


def audit_source_solver(scene,manifest):
    """Historical absence means zero; explicit profiles need physical evidence."""
    declared=['solver_profile' in record for record in (scene,manifest)]
    if any(declared) and not all(declared):
        raise ValueError('Source scene and manifest must both declare the solver profile')
    profile=normalize_solver_profile(scene.get('solver_profile'))
    if normalize_solver_profile(manifest.get('solver_profile'))!=profile:
        raise ValueError('Source scene and manifest solver profiles differ')
    if all(declared):
        readback=validate_solver_readback(profile,scene.get('solver_readback'))
        validate_solver_readback(profile,manifest.get('solver_readback'))
        return profile,readback,False
    # Do not let unlabelled nonzero readbacks masquerade as historical zero.
    for record in (scene,manifest):
        if 'solver_readback' in record:validate_solver_readback(profile,record['solver_readback'])
    return profile,None,True


def audit_rigid_options(source,replay):
    """Allow memory layout/capacity changes, never unreported physics changes."""
    if source is None:return {'available':False,'reason':'legacy source has no complete options snapshot'}
    layout={'batch_links_info','batch_joints_info','batch_dofs_info','max_dynamic_constraints'}
    differences={key:{'source':source.get(key),'replay':replay.get(key)}
        for key in set(source)|set(replay) if source.get(key)!=replay.get(key)}
    physics={key:value for key,value in differences.items() if key not in layout}
    if physics:raise ValueError('Replay rigid physics options differ: '+json.dumps(physics,sort_keys=True))
    return {'available':True,'physical_options_equal':True,'layout_capacity_differences':differences}


def audit_pd(controller,gain,scene):
    if controller.get('common_thumb_index_gain')!=gain:
        raise ValueError('PD readback belongs to a different gain')
    kp=np.asarray(SOURCE_PD['kp'],dtype=float);kv=np.asarray(SOURCE_PD['kv'],dtype=float)
    kp[6:12]*=gain;kv[6:12]*=np.sqrt(gain)
    np.testing.assert_allclose(finite_array(controller['kp_readback'],(18,),'kp'),kp,rtol=1e-6,atol=1e-7)
    np.testing.assert_allclose(finite_array(controller['kv_readback'],(18,),'kv'),kv,rtol=1e-6,atol=1e-7)
    effort=np.asarray([scene['joints'][name]['effort_limit_Nm'] for name in JOINT_ORDER])
    for key,expected in (('effort_lower_readback_Nm',-effort),('effort_upper_readback_Nm',effort)):
        np.testing.assert_allclose(finite_array(controller[key],(18,),key),expected,rtol=1e-6,atol=1e-7)
    if controller.get('effort_limits_unchanged') is not True:
        raise ValueError('Teacher must preserve original actuator force limits')


def convert_commands(path,expected_steps,dt,scene):
    rows=[];phases=[];pd=None;schedule=[];episode=None
    lower=np.asarray([scene['joints'][n]['lower_rad'] for n in JOINT_ORDER])
    upper=np.asarray([scene['joints'][n]['upper_rad'] for n in JOINT_ORDER])
    with Path(path).open() as stream:
        for index,line in enumerate(stream):
            row=json.loads(line);step=index+1
            if row['step']!=step:raise ValueError('Teacher steps must be contiguous from one')
            before=row['before'];after=row['after']
            for observation,time_s in ((before,index*dt),(after,step*dt)):
                if not np.isclose(observation['sim_time_s'],time_s,rtol=0,atol=1e-8):
                    raise ValueError('Teacher observation is not on its logged sample clock')
                if observation.get('fault') is not None or observation.get('terminated',False):
                    raise ValueError('Teacher command has a faulted observation')
                if 'episode' in observation:
                    if episode is None:episode=observation['episode']
                    if observation['episode']!=episode:raise ValueError('Multiple episodes are not one demonstration')
            phase=row['phase']
            if phase not in PHASE_ORDER:raise ValueError('Unknown teacher phase')
            if not phases or phases[-1]!=phase:phases.append(phase)
            target=finite_array(row['sent_target_rad'],(18,),'sent_target_rad')
            if (target<lower-1e-6).any() or (target>upper+1e-6).any():
                raise ValueError('Source target exceeds a robot joint limit')
            gain=row.get('grip_gain',1.)
            if isinstance(gain,bool) or not np.isscalar(gain) or not np.isfinite(gain) or not .2<=gain<=3.:
                raise ValueError('Teacher common grip gain must be an absolute scalar in [.2,3]')
            if 'grip_gain_controller' in row:pd=row['grip_gain_controller']
            if pd is None:raise ValueError('First teacher sample needs physical PD getter readbacks')
            audit_pd(pd,gain,scene)
            if index%4 and (not np.array_equal(target,rows[-1]['sent_target_rad']) or gain!=rows[-1]['grip_gain']):
                raise ValueError('Teacher must hold targets and gain for four 60Hz samples per 15Hz action')
            entry={'step':step,'phase':phase,'before':{'sim_time_s':index*dt},
                'after':{'sim_time_s':step*dt},'sent_target_rad':target.tolist(),'grip_gain':float(gain)}
            rows.append(entry)
            if not schedule or schedule[-1]['gain']!=gain:
                schedule.append({'start_s':index*dt,'end_s':step*dt,'gain':float(gain)})
            else:schedule[-1]['end_s']=step*dt
    if not isinstance(expected_steps,int) or len(rows)!=expected_steps or not rows:
        raise ValueError('Teacher command count differs from score')
    if phases!=list(PHASE_ORDER):raise ValueError('Expected one complete ordered settle-through-hold teacher trajectory')
    hold_count=int(round(3/dt))
    if len(rows)<hold_count or any(row['phase']!='hold' for row in rows[-hold_count:]):
        raise ValueError('Teacher must finish with at least three seconds of contiguous hold')
    residual_gains={row['grip_gain'] for row in rows if row['phase'] in ('preload','test_lift','lift','hold')}
    if len(residual_gains)!=1:raise ValueError('This verification contract requires one source gain throughout preload/lift/hold')
    return rows,{'source_grip_gain':residual_gains.pop(),'command_gain_schedule':schedule,
        'physical_readback_available':True,'source_gain_provenance':'direct teacher/search, not an API decision',
        'readback_timing':'source PD getter snapshot at first sample and each gain change; unchanged snapshots carried forward'}


def audit_raw_tactile(path,rows,dt,control_path,finger_order):
    count=len(rows)
    with np.load(path,allow_pickle=False) as raw:
        times=finite_array(raw['sample_time_s'],(count,),'raw sample_time_s')
        np.testing.assert_allclose(times,np.arange(1,count+1)*dt,rtol=0,atol=1e-8)
        tactile=finite_array(raw['tactile'],(count,5,120,3),'raw tactile')
        q=finite_array(raw['joint_positions_rad'],(count,18),'raw joint_positions_rad')
        target=finite_array(raw['sent_target_rad'],(count,18),'raw sent_target_rad')
        gains=finite_array(raw['grip_gain'],(count,),'raw grip_gain')
        np.testing.assert_allclose(target,[row['sent_target_rad'] for row in rows],rtol=0,atol=1e-6)
        np.testing.assert_allclose(gains,[row['grip_gain'] for row in rows],rtol=0,atol=1e-7)
        with Path(control_path).open() as stream:
            for index,line in enumerate(stream):
                observation=json.loads(line)['after']
                np.testing.assert_allclose(q[index],
                    [observation['joint_positions_rad'][name] for name in JOINT_ORDER],rtol=0,atol=1e-6)
                norms=np.linalg.norm(tactile[index],axis=-1)
                np.testing.assert_allclose(norms.max(-1),
                    [observation['tactile'][name]['peak_force_N'] for name in finger_order],rtol=1e-5,atol=1e-6)
                np.testing.assert_allclose(tactile[index].sum(1),
                    [observation['tactile'][name]['force_sum_sensor_N'] for name in finger_order],rtol=1e-5,atol=1e-6)
        hold=tactile[-round(3/dt):];norm=np.linalg.norm(hold,axis=-1)
        return {'raw_tactile_sha256':sha256(path),'samples':count,'shape':list(tactile.shape),
            'finite':True,'raw_joint_state_finite':bool(np.isfinite(q).all()),
            'joint_states_match_control_observations':True,'taxels_match_control_tactile_summaries':True,
            'hold_taxel_mean_norm_per_finger':norm.mean((0,2)).tolist(),
            'hold_taxel_peak_norm_per_finger':norm.max((0,2)).tolist(),
            'semantics':'uncalibrated simulated taxel signal, not the solver contact force'}


def audit_physics(path,rows,spec,dt):
    clearance=[];loads=[];normals=[];sensor_load=[];fault=False
    links=list(FINGERS.values())
    with Path(path).open() as stream:
        for index,line in enumerate(stream):
            if index>=len(rows):raise ValueError('Too many source physical samples')
            record=json.loads(line)
            if record['step']!=index+1 or record['phase']!=rows[index]['phase']:
                raise ValueError('Control and physical sample indices differ')
            pos=finite_array(record['object_position_m'],(3,),'object position')
            quat=finite_array(record['object_quaternion_wxyz'],(4,),'object quaternion')
            if not np.isclose(np.linalg.norm(quat),1.,rtol=0,atol=1e-4):raise ValueError('Invalid object quaternion')
            value=float(pos[2]-spec.vertical_extent(quat)-.2)
            if not np.isclose(value,record['bottom_clearance_m'],rtol=0,atol=2e-6):
                raise ValueError('Teacher clearance does not match actual primitive geometry')
            audit_material(record['object_material_readback'],spec)
            if record.get('silicone') is not None or record.get('distractor_positions_m'):
                raise ValueError('Only the declared rigid object belongs in the teacher scene')
            load=normal=0.;fingers=np.zeros(5)
            for contact in record['solver_contacts']:
                pair=(contact['entity_a'],contact['entity_b'])
                fault |= pair==('ur7e_xhand','ur7e_xhand') or ('ur7e_xhand' in pair and 'table' in pair)
                if 'ur7e_xhand' not in pair or 'cube' not in pair:continue
                force=finite_array(contact['force_on_a_world_N'],(3,),'contact force')
                n=finite_array(contact['normal_world'],(3,),'contact normal');n/=max(np.linalg.norm(n),1e-12)
                magnitude=float(np.linalg.norm(force));load+=magnitude;normal+=float(abs(force@n))
                link=contact['link_a'] if pair[0]=='ur7e_xhand' else contact['link_b']
                if link in links:fingers[links.index(link)]+=magnitude
            for key,actual in (('all_robot_contact_load_N',load),('all_robot_normal_load_N',normal)):
                if key in record and not np.isclose(record[key],actual,rtol=1e-6,atol=1e-7):
                    raise ValueError('Source contact summary differs from the recorded physical contacts')
            clearance.append(value);loads.append(load);normals.append(normal);sensor_load.append(fingers)
    if len(clearance)!=len(rows):raise ValueError('Missing source physical samples')
    n=round(3/dt);hold=np.asarray(clearance[-n:]);fingers=np.asarray(sensor_load[-n:])
    task=bool((hold>=.05).all() and not fault)
    if not task:raise ValueError('Source teacher fails independent final3s / 5cm / collision criterion')
    return {'task_success':task,'strict_success':bool(task and ((fingers>.05).sum(-1)>=2).all()),
        'hold_min_clearance_m':float(hold.min()),'hold_duration_s':3.,'max_clearance_m':max(clearance),
        'robot_self_or_table_collision':bool(fault),'samples':len(rows),'sample_hz':1/dt,
        'geometry_recomputed':True,'hold_mean_all_robot_contact_force_N':float(np.mean(loads[-n:])),
        'hold_mean_all_robot_normal_force_N':float(np.mean(normals[-n:])),
        'hold_mean_sensor_contact_force_N':float(fingers.sum(-1).mean()),
        'hold_peak_all_robot_contact_force_N':float(np.max(loads[-n:])),
        'hold_sensor_link_force_per_finger_N':fingers.mean(0).tolist()}


def audit_source_snapshots(source,declared):
    """Resolve collector-specific layouts only inside frozen source snapshots."""
    if not declared:raise ValueError('Teacher must record source-code hashes and snapshots')
    mapping={}
    for original,expected in declared.items():
        candidates=[p for p in (source/'source_snapshot').rglob(Path(original).name)
            if p.is_file() and sha256(p)==expected]
        if len(candidates)!=1:
            raise ValueError('Teacher source snapshot must have one same-name file matching its recorded code hash: '+original)
        mapping[original]=str(candidates[0].relative_to(source))
    return mapping


def audit_frozen_replay(source,manifest,scene,rows,spec,dt,tactile_audit,physics_audit,gain_audit):
    """Recompute the new replay evidence and bind it to immutable old commands.

    A successful search source remains supported separately. A frozen replay
    must prove fresh v3 observations and exact historical commands; its saved
    validation alone never substitutes for the fresh-scene ``verify`` step.
    """
    if manifest.get('driver')!='direct_teacher_frozen_replay':return None,{}
    flags={'actual_astra_api':False,'raw_tactile_reused':False,
           'no_ik_or_feedback_decisions':True,'no_attachment_or_object_pose_commands':True}
    if any(manifest.get(key) is not value for key,value in flags.items()):
        raise ValueError('Frozen replay must declare fresh sensing and exact command-only execution')
    profile=manifest.get('tactile_model')
    if (not isinstance(profile,dict) or profile.get('model_version')!='kinematic_force_on_sensor_v3'
            or not tactile_models_match(profile,current_tactile_model())):
        raise ValueError('Frozen replay requires current verified tactile v3 semantics')
    validation=read_json(source/'validation.json')
    required={'accepted':True,'fresh_tactile_captured':True,'raw_tactile_reused':False,
              'exact_source_commands_replayed':True,'source_files_unchanged':True,
              'current_source_unchanged':True}
    if (validation.get('status')!='passed' or validation.get('audit_errors')!=[] or
            any(validation.get(key) is not value for key,value in required.items())):
        raise ValueError('Frozen replay collection validation did not pass')
    for key,expected in (('tactile_model',profile),('tactile_audit',tactile_audit),
                         ('physical_audit',physics_audit),('grip_gain_audit',gain_audit)):
        if validation.get(key)!=expected:
            raise ValueError('Frozen replay validation contradicts recomputed '+key)
    artifacts=read_json(source/'artifact_sha256.json')
    if not set(HISTORICAL_FILES+('validation.json',)).issubset(artifacts):
        raise ValueError('Frozen replay artifact hashes omit required evidence')
    for name,expected in artifacts.items():
        if Path(name).name!=name or not (source/name).is_file() or sha256(source/name)!=expected:
            raise ValueError('Frozen replay artifact hash mismatch: '+name)
    historical=Path(manifest.get('source_run','')).expanduser().resolve()
    if historical==source.resolve() or not historical.is_dir():
        raise ValueError('Frozen replay historical source directory is invalid')
    declared=manifest.get('source_files_sha256')
    if not isinstance(declared,dict) or set(declared)!=set(HISTORICAL_FILES):
        raise ValueError('Frozen replay must hash all six historical source files')
    copies={}
    for name,expected in declared.items():
        original=historical/name
        if not original.is_file() or sha256(original)!=expected:
            raise ValueError('Historical replay source hash mismatch: '+name)
        copies[str(original)]='historical_source_evidence/'+name
    for name in ('control.jsonl','manifest.json','object_scene.json','score.json'):
        local=source/'source_commands'/name
        if not local.is_file() or sha256(local)!=declared[name]:
            raise ValueError('Frozen historical command evidence hash mismatch: '+name)
        copies[str(local)]='source_evidence/source_commands/'+name
    old_manifest=read_json(historical/'manifest.json');old_scene=read_json(historical/'object_scene.json')
    old_score=read_json(historical/'score.json');new_score=read_json(source/'score.json')
    if (old_manifest.get('driver')!='direct_teacher_search' or old_score.get('success') is not True
            or old_score.get('fault') is not None or new_score.get('driver')!='direct_teacher_frozen_replay'):
        raise ValueError('Frozen replay must originate from a successful direct-teacher search')
    old_dt=audit_scene(old_scene,old_manifest,spec)
    old_profile,_,_=audit_source_solver(old_scene,old_manifest)
    if old_dt!=dt or old_profile!=normalize_solver_profile(manifest.get('solver_profile')):
        raise ValueError('Frozen replay source timing or solver profile changed')
    options=audit_rigid_options(old_scene.get('rigid_options_actual'),scene.get('rigid_options_actual'))
    if options.get('available') is not True:
        raise ValueError('Frozen replay requires historical and current rigid-options evidence')
    old_rows,_=convert_commands(historical/'control.jsonl',old_score['control_steps'],dt,old_scene)
    if old_rows!=rows:
        raise ValueError('Frozen replay targets, gain schedule, timestamps or phases changed')
    old_physics=audit_physics(historical/'evaluation.jsonl',old_rows,spec,dt)
    old_model=old_manifest.get('tactile_model',legacy_tactile_model())
    if manifest.get('source_tactile_model')!=old_model:
        raise ValueError('Frozen replay relabelled historical tactile semantics')
    return {'source_run':str(historical),'source_files_sha256':declared,
        'source_tactile_model':old_model,'current_tactile_model':profile,
        'validation_sha256':sha256(source/'validation.json'),
        'exact_historical_commands_recomputed':True,'fresh_tactile_audit_recomputed':True,
        'historical_physics_audit':old_physics,'rigid_options_audit':options,
        'collection_validation_is_independent_replay_verification':False},copies


def convert(source_run,object_spec,output):
    source=Path(source_run).expanduser().resolve();output=Path(output).expanduser().resolve()
    spec=ObjectSpec.from_value(object_spec)
    if output.exists():raise FileExistsError(output)
    if source in output.parents:raise ValueError('Bundle must be outside the immutable source run')
    manifest=read_json(source/'manifest.json');score=read_json(source/'score.json');scene=read_json(source/'object_scene.json')
    driver=manifest.get('driver')
    if driver not in TEACHER_DRIVERS or manifest.get('privileged_teacher') is not True:
        raise ValueError('Expected explicitly privileged direct-teacher/search provenance')
    if score.get('success') is not True or score.get('fault') is not None:
        raise ValueError('Source teacher did not succeed')
    dt=audit_scene(scene,manifest,spec)
    solver_profile,source_solver_readback,legacy_solver=audit_source_solver(scene,manifest)
    rows,gain_audit=convert_commands(source/'control.jsonl',score['control_steps'],dt,scene)
    tactile_audit=audit_raw_tactile(source/'raw_tactile.npz',rows,dt,source/'control.jsonl',
        scene.get('tactile_finger_order',list(FINGERS)))
    physics_audit=audit_physics(source/'evaluation.jsonl',rows,spec,dt)
    replay_audit,historical_copies=audit_frozen_replay(source,manifest,scene,rows,spec,dt,
        tactile_audit,physics_audit,gain_audit)
    names=['manifest.json','object_scene.json','control.jsonl','evaluation.jsonl','score.json','raw_tactile.npz']
    names += [name for name in ('parameters.json','search_history.json','reference.json','stages.json',
        'registration.json','artifact_sha256.json','force_allocation.json','teacher_contact_events.json',
        'test_lift_gate.json','validation.json','status.json') if (source/name).is_file()]
    source_code_mapping=audit_source_snapshots(source,manifest.get('source_sha256'))
    source_tactile_model=audit_source_tactile_model(scene,manifest,manifest.get('source_sha256',{}))
    copies={str(source/name):'source_evidence/'+name for name in names}
    copies.update(historical_copies)
    # Freeze code where the collector supplies an immutable snapshot. Original
    # absolute paths remain provenance and are never executed by the importer.
    for path in sorted((source/'source_snapshot').rglob('*')) if (source/'source_snapshot').is_dir() else []:
        if path.is_file():copies[str(path)]='source_evidence/source_snapshot/'+str(path.relative_to(source/'source_snapshot'))
    hashes={path:sha256(path) for path in copies}
    output.mkdir(parents=True,exist_ok=False)
    for original,relative in copies.items():
        destination=output/relative;destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(original,destination)
        if sha256(destination)!=hashes[original] or sha256(original)!=hashes[original]:
            raise ValueError('Teacher evidence mutated during freezing')
    target=output/'control.jsonl'
    target.write_text(''.join(json.dumps(row,separators=(',',':'),allow_nan=False)+'\n' for row in rows))
    trajectory=Trajectory(target)
    metadata={'schema':'multiobject_direct_teacher_command_trajectory_v1','status':'converted_not_replay_validated',
        'demonstration_source':driver,'privileged_teacher':True,'actual_astra_api':False,
        'object_id':spec.shape,'object_spec':asdict(spec),
        'trajectory_path':str(target),'trajectory_sha256':sha256(target),'source_run':str(source),
        'source_snapshot_paths':copies,'source_sha256':hashes,'source_declared_code_hashes':manifest.get('source_sha256',{}),
        'source_code_snapshot_paths':source_code_mapping,
        'source_tactile_model':source_tactile_model,
        'historical_source_tactile_model':replay_audit['source_tactile_model'] if replay_audit else None,
        'frozen_replay_audit':replay_audit,
        'solver_profile':solver_profile,'source_solver_readback':source_solver_readback,
        'legacy_solver_profile_default_zero':legacy_solver,
        'source_rigid_options_actual':scene.get('rigid_options_actual'),
        'source_physics_audit':physics_audit,'source_tactile_audit':tactile_audit,'source_grip_gain_audit':gain_audit,
        'source_parameters':manifest.get('parameters',{}),'joint_order':JOINT_ORDER,'row_count':len(rows),
        'duration_s':trajectory.duration_s,'phase_starts':trajectory.phase_starts,
        'evaluation_hold_interval_s':[trajectory.duration_s-3.,trajectory.duration_s],
        'material':{'mass_kg':.05,'friction':.8},'rates_hz':{'source_physics':300,'source_samples':60,'source_actions':15,
            'replay_physics':300,'replay_tactile':60,'replay_actions':15},
        'conversion':'Exact original targets, gain samples, timestamps and phase labels. No fabricated hold extension or object-state edits.',
        'replay_verification':None}
    write_json(output/'trajectory_metadata.json',metadata)
    return metadata


def verify(bundle,output,verification_gain=None):
    bundle=Path(bundle).expanduser().resolve();output=Path(output).expanduser().resolve()
    metadata=read_json(bundle/'trajectory_metadata.json')
    if metadata.get('schema')!='multiobject_direct_teacher_command_trajectory_v1':raise ValueError('Not a direct teacher bundle')
    source_gain=float(metadata['source_grip_gain_audit']['source_grip_gain'])
    gain=source_gain if verification_gain is None else float(verification_gain)
    if gain!=source_gain:raise ValueError('Verification gain must be declared by the successful source, not selected from replay outcomes')
    trajectory=bundle/'control.jsonl'
    if sha256(trajectory)!=metadata['trajectory_sha256']:raise ValueError('Teacher trajectory hash changed')
    for path,relative in metadata['source_snapshot_paths'].items():
        if sha256(bundle/relative)!=metadata['source_sha256'][path]:raise ValueError('Frozen teacher evidence hash changed')
    source_scene=read_json(bundle/'source_evidence/object_scene.json')
    source_manifest=read_json(bundle/'source_evidence/manifest.json')
    driver=metadata.get('demonstration_source')
    if driver not in TEACHER_DRIVERS or driver!=source_manifest.get('driver'):
        raise ValueError('Bundle demonstration source contradicts its frozen driver')
    provenance={'demonstration_source':driver,'privileged_teacher':True,'actual_astra_api':False,
        'source_tactile_model':metadata.get('source_tactile_model',legacy_tactile_model()),
        'tactile_model':current_tactile_model()}
    if driver=='direct_teacher_frozen_replay':
        replay_audit=metadata.get('frozen_replay_audit') or {}
        validation=bundle/'source_evidence/validation.json'
        if (not validation.is_file() or sha256(validation)!=replay_audit.get('validation_sha256')
                or read_json(validation).get('accepted') is not True):
            raise ValueError('Frozen replay collection validation is missing or changed')
        if (provenance['source_tactile_model'].get('model_version')!='kinematic_force_on_sensor_v3'
                or not tactile_models_match(provenance['source_tactile_model'],provenance['tactile_model'])):
            raise ValueError('Frozen replay verification must use the same current tactile v3 semantics')
        provenance.update(source_replay_validation_path=str(validation),
            source_replay_validation_sha256=sha256(validation),
            historical_source_tactile_model=metadata.get('historical_source_tactile_model'),
            frozen_replay_audit=replay_audit)
    profile,source_readback,legacy_solver=audit_source_solver(source_scene,source_manifest)
    if normalize_solver_profile(metadata.get('solver_profile'))!=profile:
        raise ValueError('Bundle solver profile differs from its frozen source')
    if not legacy_solver:
        validate_solver_readback(profile,metadata.get('source_solver_readback'))
    from .adaptive_env import AdaptiveConfig
    from .multiobject_env import MultiObjectVecEnv
    from .multiobject_benchmark import evaluate_controller
    from .multiobject_report import trace_statistics,audit_recorded_episode,audit_environment
    from .multiobject_train import source_snapshot
    from .env import cpu
    class AuditedEnv(MultiObjectVecEnv):
        def _apply_gains(self):
            super()._apply_gains()
            kp=cpu(self.robot.get_dofs_kp(self.indices)).reshape(self.num_envs,18)
            kv=cpu(self.robot.get_dofs_kv(self.indices)).reshape(self.num_envs,18)
            expected_kp=np.broadcast_to(self.base_kp,kp.shape).copy();expected_kv=np.broadcast_to(self.base_kv,kv.shape).copy()
            for j,sl in enumerate((slice(6,9),slice(9,12))):
                expected_kp[:,sl]*=self.gains[:,j,None];expected_kv[:,sl]*=np.sqrt(self.gains[:,j,None])
            np.testing.assert_allclose(kp,expected_kp,rtol=1e-6,atol=1e-7)
            np.testing.assert_allclose(kv,expected_kv,rtol=1e-6,atol=1e-7)
            if not hasattr(self,'gain_readbacks'):self.gain_readbacks=[]
            self.gain_readbacks.append({'gains':self.gains.tolist(),'kp_readback':kp.tolist(),'kv_readback':kv.tolist()})
    output.mkdir(parents=True,exist_ok=False)
    gain_record={'source_grip_gain':source_gain,'verification_gain':gain,
        'verification_gain_schedule':{'prefix':'recorded source gains, common to all controllers',
            'residual':'absolute target; .2 maximum gain change per15Hz action; never multiplied by source gain'}}
    write_json(output/'manifest.json',{**provenance,
        'trajectory_sha256':sha256(trajectory),'metadata_sha256':sha256(bundle/'trajectory_metadata.json'),
        'source_sha256':source_snapshot(output),'rates_hz':{'physics':300,'tactile':60,'action':15},
        'solver_profile':profile,'source_solver_readback':source_readback,
        **gain_record,'training':False})
    env=None
    try:
        write_json(output/'status.json',{'status':'initializing'})
        env=AuditedEnv(object_spec=metadata['object_spec'],trajectory_path=trajectory,n_envs=1,seed=7000,
            mode='eval',material_conditions=[[.05,.8]],output_dir=output/'environment',
            noslip_iterations=profile['noslip_iterations'],noslip_tolerance=profile['noslip_tolerance'],
            config=AdaptiveConfig(gain_min=.2,failure_penalty=200.),audit_prefix=True)
        solver_readback=validate_solver_readback(profile,read_solver_readback(env.solver))
        rigid_options_audit=audit_rigid_options(source_scene.get('rigid_options_actual'),
            env.solver._options.model_dump(mode='json'))
        name=f'fixed_{gain:g}'.replace('.','p')
        result,_=evaluate_controller(env,name=name,model=None,fixed_gain=gain,output=output/name,
            conditions=[{'condition_id':'demonstration_easy','mass_kg':.05,'friction':.8}],object_id=metadata['object_id'],
            split='preflight',training_seed=0,method=name)
        with np.load(output/name/'trace.npz',allow_pickle=False) as trace:
            calculated=trace_statistics(trace,result['hold_interval_s'])[0]
            tactile=trace['tactile_before'].reshape(-1,5,120,3)
            tactile_audit={'finite':bool(np.isfinite(tactile).all()),'samples':len(tactile),
                'max_taxel_vector_norm':float(np.linalg.norm(tactile,axis=-1).max()),
                'source_historical_tactile_used_as_policy_input':False}
        physical=read_json(output/'environment/environment_protocol.json')
        if not tactile_models_match(physical.get('tactile_model'),provenance['tactile_model']):
            raise ValueError('Independent replay sensor differs from its recorded tactile provenance')
        errors=audit_recorded_episode(result['episodes'][0],calculated)
        errors+=audit_environment(physical,result['episodes'],{'objects':[{'object_id':metadata['object_id'],
            'object_spec':metadata['object_spec']}]})
        if any(physical['prefix_physics_audit']['collision_per_lane']):errors.append('Robot collision during physical prefix')
        if not tactile_audit['finite']:errors.append('Nonfinite independently generated tactile data')
        validate_solver_readback(profile,read_solver_readback(env.solver))
        readback=output/'controller_readback.json'
        write_json(readback,{'all_applications_verified':True,**gain_record,'applications':env.gain_readbacks})
        accepted=bool(calculated['task_success'] and not errors)
        report={'status':'passed' if accepted else 'failed','accepted':accepted,'audit_errors':errors,
            **provenance,**gain_record,
            'solver_profile':profile,'source_solver_readback':source_readback,'solver_readback':solver_readback,
            'legacy_solver_profile_default_zero':legacy_solver,'rigid_options_audit':rigid_options_audit,
            'trace_recomputed':calculated,'tactile_audit':tactile_audit,
            'controller_readback_path':str(readback),'controller_readback_sha256':sha256(readback),
            'evaluation_path':str(output/name/'evaluation.json'),'evaluation_sha256':sha256(output/name/'evaluation.json')}
        write_json(output/'verification.json',report)
        if accepted:
            evidence={'task_success':True,'strict_success':calculated['strict_success'],'mass_kg':.05,'friction':.8,
                **provenance,**gain_record,
                'solver_profile':profile,'solver_readback':solver_readback,
                'hold_min_clearance_m':calculated['hold_min_clearance_m'],'hold_duration_s':calculated['hold_duration_s'],
                'verification_path':str(output/'verification.json'),'verification_sha256':sha256(output/'verification.json'),
                'trajectory_metadata_path':str(bundle/'trajectory_metadata.json'),'trajectory_metadata_sha256':sha256(bundle/'trajectory_metadata.json'),
                'controller_readback_path':str(readback),'controller_readback_sha256':sha256(readback),
                'source_run':metadata['source_run'],'rates_hz':{'physics':300,'tactile':60,'action':15},
                'criterion':'Original task criterion; strict contact support and force/tactile metrics reported separately'}
            write_json(output/'bind_record.json',{'object_id':metadata['object_id'],'object_spec':metadata['object_spec'],
                'trajectory_path':str(trajectory),'trajectory_sha256':sha256(trajectory),'demonstration_evidence':evidence})
        write_json(output/'status.json',{'status':'completed','accepted':accepted})
        return report
    except BaseException as error:
        write_json(output/'status.json',{'status':'failed','error':repr(error)});raise
    finally:
        if env is not None:env.close()


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    command=sub.add_parser('convert');command.add_argument('--source-run',type=Path,required=True)
    command.add_argument('--object-spec',required=True,help='Shape name or JSON ObjectSpec, including custom dimensions')
    command.add_argument('--output',type=Path,required=True)
    command=sub.add_parser('verify');command.add_argument('--bundle',type=Path,required=True)
    command.add_argument('--output',type=Path,required=True);command.add_argument('--verification-gain',type=float)
    args=parser.parse_args(argv)
    if args.command=='convert':
        spec=json.loads(args.object_spec) if args.object_spec.startswith('{') else args.object_spec
        result=convert(args.source_run,spec,args.output)
    else:result=verify(args.bundle,args.output,args.verification_gain)
    print(json.dumps({k:v for k,v in result.items() if k in ('status','accepted','audit_errors','row_count','duration_s','trajectory_sha256')},indent=2))


if __name__=='__main__':main()
