"""Synthetic CPU evidence contracts; these fixtures are not physics successes."""
import copy
from dataclasses import asdict
import json
import shutil

import numpy as np
import pytest

from .multiobject_import_teacher import (ROOT, FINGERS, HOME_DEGREES, JOINT_ORDER, PHASE_ORDER,
    SOURCE_PD, ObjectSpec, Trajectory, convert, verify, sha256, audit_rigid_options)
from .test_solver_profile import profile,readback


def save_rows(path,rows):path.write_text(''.join(json.dumps(row)+'\n' for row in rows))


def fixture(tmp_path,spec,gain=2.):
    source=tmp_path/'source';source.mkdir();spec=ObjectSpec.from_value(spec)
    phases=[phase for phase in PHASE_ORDER[:-1] for _ in range(4)]+['hold']*180
    material={'object_engine_mass_kg':.05,'object_engine_inertia_kg_m2':(.05*spec.inertia_per_kg).tolist(),
        'object_engine_density_kg_m3':.05/spec.volume_m3,'object_bounds_readback_m':spec.bounds_size_m.tolist()}
    scene={'object_spec':{**spec.protocol(),'initial_position_m':spec.initial_position().tolist()},
        'joint_order':JOINT_ORDER,'kp':SOURCE_PD['kp'],'kv':SOURCE_PD['kv'],'home_arm_degrees':HOME_DEGREES,
        'control_dt_s':1/60,'physics_dt_s':1/300,'table_top_z_m':.2,'silicone':None,
        'joints':{name:{'lower_rad':-10.,'upper_rad':10.,'effort_limit_Nm':3.} for name in JOINT_ORDER},
        'materials':{**material,'cube_friction':.8,'robot_friction':.8}}
    (source/'object_scene.json').write_text(json.dumps(scene))
    snapshot=source/'source_snapshot/fixture_teacher_source.py';snapshot.parent.mkdir();snapshot.write_text('synthetic CPU fixture, never executed')
    manifest={'driver':'direct_teacher_search','privileged_teacher':True,'object_spec':spec.protocol(),
        'control_dt_s':1/60,'physics_dt_s':1/300,'parameters':{'grip_gain':gain},
        'source_sha256':{str(ROOT/'fixture_teacher_source.py'):sha256(snapshot)}}
    (source/'manifest.json').write_text(json.dumps(manifest))
    rows=[];evals=[];previous=None
    for index,phase in enumerate(phases):
        active=phase in ('close','preload','test_lift','lift','hold');current=gain if active else 1.
        target=np.full(18,.001*(index//4));before={'sim_time_s':index/60,'episode':1,'fault':None,'terminated':False}
        after={**before,'sim_time_s':(index+1)/60,
            'joint_positions_rad':dict(zip(JOINT_ORDER,(target+.0001).tolist())),
            'tactile':{name:{'peak_force_N':np.sqrt(3)*.001,'force_sum_sensor_N':[.12]*3} for name in FINGERS}}
        row={'step':index+1,'phase':phase,'before':before,'after':after,
            'sent_target_rad':target.tolist(),'grip_gain':current}
        if previous!=current:
            kp=np.asarray(SOURCE_PD['kp'],dtype=float);kv=np.asarray(SOURCE_PD['kv'],dtype=float)
            kp[6:12]*=current;kv[6:12]*=np.sqrt(current)
            row['grip_gain_controller']={'common_thumb_index_gain':current,
                'kp_readback':kp.tolist(),'kv_readback':kv.tolist(),'effort_lower_readback_Nm':[-3.]*18,
                'effort_upper_readback_Nm':[3.]*18,'effort_limits_unchanged':True}
        previous=current;rows.append(row)
        p=spec.initial_position();clearance=.06 if phase=='hold' else 0.;p[2]+=clearance
        contacts=[{'entity_a':'ur7e_xhand','entity_b':'cube','link_a':link,'link_b':'object',
            'force_on_a_world_N':[sign,0.,0.],'normal_world':[sign,0.,0.]}
            for link,sign in zip(list(FINGERS.values())[:2],(1.,-1.))]
        evals.append({'step':index+1,'phase':phase,'object_position_m':p.tolist(),
            'object_quaternion_wxyz':[1.,0.,0.,0.],'bottom_clearance_m':clearance,
            'object_material_readback':material,'solver_contacts':contacts,
            'all_robot_contact_load_N':2.,'all_robot_normal_load_N':2.,'silicone':None,'distractor_positions_m':{}})
    save_rows(source/'control.jsonl',rows);save_rows(source/'evaluation.jsonl',evals)
    np.savez_compressed(source/'raw_tactile.npz',sample_time_s=np.arange(1,len(rows)+1)/60,
        tactile=np.full((len(rows),5,120,3),.001,np.float32),
        joint_positions_rad=np.asarray([r['sent_target_rad'] for r in rows],np.float32)+.0001,
        sent_target_rad=np.asarray([r['sent_target_rad'] for r in rows],np.float32),
        grip_gain=np.asarray([r['grip_gain'] for r in rows],np.float32))
    (source/'score.json').write_text(json.dumps({'success':True,'fault':None,'control_steps':len(rows)}))
    return source,rows,evals


@pytest.mark.parametrize('spec',['cuboid','hex_prism','sphere',{'shape':'cylinder','height_m':.09}])
def test_four_requested_geometries_and_honest_provenance(tmp_path,spec):
    source,rows,_=fixture(tmp_path,spec);bundle=tmp_path/'bundle'
    metadata=convert(source,spec,bundle)
    trajectory=Trajectory(bundle/'control.jsonl')
    np.testing.assert_array_equal(trajectory.targets,[row['sent_target_rad'] for row in rows])
    np.testing.assert_array_equal(trajectory.command_grip_gains,[row['grip_gain'] for row in rows])
    assert trajectory.phases.tolist()==[row['phase'] for row in rows]
    assert metadata['status']=='converted_not_replay_validated'
    assert metadata['demonstration_source']=='direct_teacher_search'
    assert metadata['privileged_teacher'] is True
    assert metadata['actual_astra_api'] is False
    assert metadata['source_physics_audit']['task_success'] and metadata['source_physics_audit']['strict_success']
    assert metadata['source_physics_audit']['hold_mean_all_robot_contact_force_N']==2.
    assert metadata['source_tactile_audit']['shape']==[len(rows),5,120,3]
    assert metadata['source_grip_gain_audit']['source_grip_gain']==2.
    assert metadata['solver_profile']==profile(0)
    assert metadata['legacy_solver_profile_default_zero'] is True
    assert not (bundle/'bind_record.json').exists()
    assert (bundle/'source_evidence/source_snapshot/fixture_teacher_source.py').exists()
    if isinstance(spec,dict):assert metadata['object_spec']['height_m']==.09
    with pytest.raises(FileExistsError):convert(source,spec,bundle)


@pytest.mark.parametrize('corruption',['clock','target_rate','gain_readback','raw_target','raw_q','raw_tactile','hold','shape','snapshot'])
def test_corrupt_or_incompatible_teacher_evidence_is_rejected(tmp_path,corruption):
    spec={'shape':'cylinder','height_m':.09};source,rows,evals=fixture(tmp_path,spec)
    if corruption=='clock':rows[-1]['after']['sim_time_s']+=.001
    elif corruption=='target_rate':rows[1]['sent_target_rad'][0]+=.01
    elif corruption=='gain_readback':rows[16]['grip_gain_controller']['kp_readback'][9]=3.
    elif corruption in ('raw_target','raw_q','raw_tactile'):
        with np.load(source/'raw_tactile.npz') as raw:data={k:raw[k].copy() for k in raw.files}
        key={'raw_target':'sent_target_rad','raw_q':'joint_positions_rad','raw_tactile':'tactile'}[corruption]
        data[key][-1,0]+=.01;np.savez_compressed(source/'raw_tactile.npz',**data)
    elif corruption=='hold':evals[-1]['object_position_m'][2]-=.04;evals[-1]['bottom_clearance_m']-=.04
    elif corruption=='shape':spec={'shape':'cylinder','height_m':.045}
    elif corruption=='snapshot':(source/'source_snapshot/fixture_teacher_source.py').write_text('changed')
    save_rows(source/'control.jsonl',rows);save_rows(source/'evaluation.jsonl',evals)
    with pytest.raises((ValueError,AssertionError)):convert(source,spec,tmp_path/'bad')
    assert not (tmp_path/'bad').exists()


def test_verification_rejects_post_hoc_gain_without_initializing_genesis(tmp_path):
    source,_,_=fixture(tmp_path,'cuboid',gain=3.);bundle=tmp_path/'bundle'
    convert(source,'cuboid',bundle)
    with pytest.raises(ValueError,match='declared by the successful source'):
        verify(bundle,tmp_path/'verification',verification_gain=2.)
    assert not (tmp_path/'verification').exists()


def test_nested_and_external_teacher_snapshots_are_hash_bound(tmp_path):
    source,_,_=fixture(tmp_path,'cuboid')
    nested=source/'source_snapshot/tripod_teacher';nested.mkdir()
    (source/'source_snapshot/fixture_teacher_source.py').rename(nested/'fixture_teacher_source.py')
    external=nested/'external_frames.py';external.write_text('immutable external helper snapshot')
    path=source/'manifest.json';manifest=json.loads(path.read_text())
    manifest['source_sha256']['/outside/project/external_frames.py']=sha256(external)
    path.write_text(json.dumps(manifest))
    for name in ('force_allocation.json','teacher_contact_events.json'):
        (source/name).write_text(json.dumps({'privileged_teacher':True,'evidence_only':True}))
    metadata=convert(source,'cuboid',tmp_path/'bundle')
    assert metadata['source_code_snapshot_paths']['/outside/project/external_frames.py']=='source_snapshot/tripod_teacher/external_frames.py'
    assert (tmp_path/'bundle/source_evidence/force_allocation.json').exists()
    assert (tmp_path/'bundle/source_evidence/teacher_contact_events.json').exists()
    assert (tmp_path/'bundle/source_evidence/source_snapshot/tripod_teacher/external_frames.py').exists()


def test_duplicate_matching_code_snapshot_is_rejected(tmp_path):
    source,_,_=fixture(tmp_path,'cuboid')
    duplicate=source/'source_snapshot/duplicate';duplicate.mkdir()
    (duplicate/'fixture_teacher_source.py').write_bytes((source/'source_snapshot/fixture_teacher_source.py').read_bytes())
    with pytest.raises(ValueError,match='one same-name file'):
        convert(source,'cuboid',tmp_path/'bundle')


@pytest.mark.parametrize('corruption',[None,'manifest_profile','engine_readback','missing_declaration'])
def test_explicit_solver_profile_requires_source_agreement(tmp_path,corruption):
    source,_,_=fixture(tmp_path,'sphere')
    for name in ('object_scene.json','manifest.json'):
        path=source/name;record=json.loads(path.read_text())
        record.update(solver_profile=profile(),solver_readback=readback())
        if name=='manifest.json' and corruption=='manifest_profile':record['solver_profile']=profile(0)
        if name=='object_scene.json' and corruption=='engine_readback':record['solver_readback']['engine_noslip_iterations']=0
        if name=='manifest.json' and corruption=='missing_declaration':del record['solver_profile']
        path.write_text(json.dumps(record))
    if corruption:
        with pytest.raises(ValueError):convert(source,'sphere',tmp_path/'bundle')
    else:
        metadata=convert(source,'sphere',tmp_path/'bundle')
        assert metadata['solver_profile']==profile()
        assert metadata['source_solver_readback']==readback()
        assert metadata['legacy_solver_profile_default_zero'] is False


def test_only_rigid_memory_layout_can_differ():
    source={'iterations':50,'batch_links_info':False,'max_dynamic_constraints':8}
    replay={**source,'batch_links_info':True,'max_dynamic_constraints':16}
    assert audit_rigid_options(source,replay)['physical_options_equal']
    with pytest.raises(ValueError,match='physics options'):audit_rigid_options(source,{**replay,'iterations':25})


def replay_fixture(tmp_path):
    """Synthetic evidence for CPU integrity tests, never a physical success."""
    from .multiobject_import_teacher import (HISTORICAL_FILES, convert_commands,
        audit_raw_tactile,audit_physics)
    from .tactile_provenance import current_tactile_model,legacy_tactile_model
    historical,_,_=fixture(tmp_path,'cuboid')
    for name in ('manifest.json','object_scene.json'):
        path=historical/name;value=json.loads(path.read_text())
        value.update(solver_profile=profile(),solver_readback=readback())
        if name=='object_scene.json':value['rigid_options_actual']={'iterations':50}
        path.write_text(json.dumps(value))
    source=tmp_path/'replay';shutil.copytree(historical,source)
    current=current_tactile_model();manifest=json.loads((source/'manifest.json').read_text())
    manifest.update(driver='direct_teacher_frozen_replay',actual_astra_api=False,
        raw_tactile_reused=False,no_ik_or_feedback_decisions=True,no_attachment_or_object_pose_commands=True,
        source_run=str(historical),source_files_sha256={name:sha256(historical/name) for name in HISTORICAL_FILES},
        source_tactile_model=legacy_tactile_model(),tactile_model=current)
    for relative,digest in current['source_sha256'].items():
        manifest['source_sha256'][str(ROOT/relative)]=digest
        path=source/'source_snapshot'/relative;path.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/relative,path)
    (source/'manifest.json').write_text(json.dumps(manifest))
    scene=json.loads((source/'object_scene.json').read_text());scene['tactile_model']=current
    (source/'object_scene.json').write_text(json.dumps(scene))
    score=json.loads((source/'score.json').read_text());score['driver']='direct_teacher_frozen_replay'
    (source/'score.json').write_text(json.dumps(score))
    (source/'source_commands').mkdir()
    for name in ('control.jsonl','manifest.json','object_scene.json','score.json'):
        shutil.copy2(historical/name,source/'source_commands'/name)
    rows,gains=convert_commands(source/'control.jsonl',score['control_steps'],1/60,scene)
    tactile=audit_raw_tactile(source/'raw_tactile.npz',rows,1/60,source/'control.jsonl',list(FINGERS))
    physics=audit_physics(source/'evaluation.jsonl',rows,ObjectSpec.from_value('cuboid'),1/60)
    validation=dict(status='passed',accepted=True,fresh_tactile_captured=True,raw_tactile_reused=False,
        audit_errors=[],exact_source_commands_replayed=True,source_files_unchanged=True,current_source_unchanged=True,
        tactile_audit=tactile,physical_audit=physics,grip_gain_audit=gains,tactile_model=current)
    (source/'validation.json').write_text(json.dumps(validation))
    refresh_replay_artifacts(source)
    return source,historical


def refresh_replay_artifacts(source):
    (source/'artifact_sha256.json').write_text(json.dumps({path.name:sha256(path)
        for path in source.iterdir() if path.is_file() and path.name!='artifact_sha256.json'}))


def test_frozen_replay_preserves_honest_source_and_old_sensor_evidence(tmp_path):
    source,historical=replay_fixture(tmp_path)
    metadata=convert(source,'cuboid',tmp_path/'bundle')
    assert metadata['demonstration_source']=='direct_teacher_frozen_replay'
    assert metadata['actual_astra_api'] is False
    assert metadata['source_tactile_model']['model_version']=='kinematic_force_on_sensor_v3'
    assert metadata['historical_source_tactile_model']['model_version']=='legacy_unversioned'
    assert metadata['frozen_replay_audit']['exact_historical_commands_recomputed']
    assert not metadata['frozen_replay_audit']['collection_validation_is_independent_replay_verification']
    assert (tmp_path/'bundle/source_evidence/validation.json').is_file()
    assert sha256(tmp_path/'bundle/historical_source_evidence/raw_tactile.npz')==sha256(historical/'raw_tactile.npz')
    assert not (tmp_path/'bundle/bind_record.json').exists()


@pytest.mark.parametrize('corruption',['validation_failed','validation_false_metrics','historical_hash',
    'frozen_command_copy','relabel_old_sensor','old_v2_sensor','reuse_raw','artifact_hash'])
def test_frozen_replay_rejects_unverified_or_mislabeled_evidence(tmp_path,corruption):
    source,historical=replay_fixture(tmp_path)
    manifest=json.loads((source/'manifest.json').read_text())
    validation=json.loads((source/'validation.json').read_text())
    if corruption=='validation_failed':validation['accepted']=False
    elif corruption=='validation_false_metrics':validation['physical_audit']['hold_min_clearance_m']+=.01
    elif corruption=='historical_hash':manifest['source_files_sha256']['score.json']='0'*64
    elif corruption=='frozen_command_copy':
        (source/'source_commands/control.jsonl').write_text('mutated commands')
    elif corruption=='relabel_old_sensor':manifest['source_tactile_model']=manifest['tactile_model']
    elif corruption=='old_v2_sensor':manifest['tactile_model']['model_version']='kinematic_force_on_sensor_v2'
    elif corruption=='reuse_raw':manifest['raw_tactile_reused']=True
    (source/'manifest.json').write_text(json.dumps(manifest))
    (source/'validation.json').write_text(json.dumps(validation))
    refresh_replay_artifacts(source)
    if corruption=='artifact_hash':
        path=source/'artifact_sha256.json';value=json.loads(path.read_text());value['validation.json']='0'*64
        path.write_text(json.dumps(value))
    with pytest.raises(ValueError):convert(source,'cuboid',tmp_path/'bundle')
    assert not (tmp_path/'bundle').exists()
