"""CPU provenance/clock tests; fixtures are not claimed as simulated successes."""
import copy
from dataclasses import asdict
import json

import numpy as np
import pytest

from .core import Trajectory
from .import_astra_trajectory import CONTROL_DT, HOME_DEGREES, JOINT_ORDER, SOURCE_PD
from .multiobject_geometry import ObjectSpec
from .multiobject_import_astra import convert, audit_api_actions, verification_gain_contract
from .test_import_astra_trajectory import source_rows, write_rows


def source_fixture(tmp_path, extended_hold=False, object_id='cube'):
    source=tmp_path/'source';agent=tmp_path/'agent';source.mkdir();agent.mkdir()
    spec=ObjectSpec(object_id);rows=source_rows()
    if extended_hold:
        for _ in range(40):
            row=copy.deepcopy(rows[-1]);number=len(rows)+1
            row['step']=row['observation']['episode_step']=number
            row['observation']['sim_time_s']=number*CONTROL_DT
            row['target'][10]+=.001
            rows.append(row)
    write_rows(source/'control.jsonl',rows)
    material={'object_engine_mass_kg':.05,'object_engine_inertia_kg_m2':(.05*spec.inertia_per_kg).tolist(),
        'object_engine_density_kg_m3':.05/spec.volume_m3}
    eval_rows=[]
    for index,row in enumerate(rows):
        clearance=.06 if row['phase']=='hold' and index>=len(rows)-120 else 0.
        eval_rows.append({'step':row['step'],'phase':row['phase'],
            'object_position_m':[-.6,-.1333,.2225+clearance],
            'object_quaternion_wxyz':[1.,0.,0.,0.],'bottom_clearance_m':clearance,
            'object_material_readback':material,'silicone':None,'distractor_positions_m':{},'solver_contacts':[]})
    write_rows(source/'evaluation.jsonl',eval_rows)
    manifest={'high_level_driver':'vlm_policy_CodexAppServerPolicy','object_spec':asdict(spec),'source_hashes':{}}
    (source/'manifest.json').write_text(json.dumps(manifest))
    score={'success':True,'fault':None,'control_steps':len(rows)}
    for folder in (source,agent):(folder/'score.json').write_text(json.dumps(score))
    scene={'object_spec':{**asdict(spec),'initial_position_m':spec.initial_position().tolist()},
        'joint_order':JOINT_ORDER,'kp':SOURCE_PD['kp'],'kv':SOURCE_PD['kv'],'home_arm_degrees':HOME_DEGREES,
        'control_dt_s':.025,'physics_dt_s':.005,'table_top_z_m':.2,'silicone':None,
        'materials':{**material,'cube_friction':.8,'robot_friction':.8}}
    (agent/'object_scene.json').write_text(json.dumps(scene))
    (agent/'runner.json').write_text(json.dumps({'driver':'vlm_policy_CodexAppServerPolicy','model':'gpt-6-astra','check_only':False}))
    (agent/'bridge_run.json').write_text(json.dumps({'run_dir':str(source),'thread_id':'fixture-thread'}))
    (agent/'instructions.txt').write_text('Synthetic importer contract test.')
    interactions=[]
    for index,skill in enumerate(['locate','approach','align','close','test_lift','lift','hold','none']):
        action={'status':'done' if skill=='none' else 'continue','skill':skill,'observation_id':str(index)}
        record={'action':action,'metadata':{'model':'gpt-6-astra','api_mode':'codex_app_server',
            'thread_id':'fixture-thread','turn_id':f'fixture-turn-{index}'}}
        (agent/f'model_{index:03d}.json').write_text(json.dumps(record))
        (agent/f'api_input_{index:03d}.json').write_text('[]')
        status='model_done' if skill=='none' else 'located' if skill=='locate' else 'executed'
        interactions.append({'action':action,'result':{'status':status}})
    (source/'interactions.json').write_text(json.dumps(interactions))
    return source,agent,rows


def test_import_preserves_commands_and_does_not_claim_replay_success(tmp_path):
    source,agent,rows=source_fixture(tmp_path)
    bundle=tmp_path/'bundle';metadata=convert(source,agent,'cube',bundle)
    trajectory=Trajectory(bundle/'control.jsonl')
    np.testing.assert_array_equal(trajectory.targets,[row['target'] for row in rows])
    assert metadata['status']=='converted_not_replay_validated'
    assert metadata['replay_verification'] is None
    assert not (bundle/'bind_record.json').exists()
    assert len(metadata['astra_decisions'])==8
    assert (bundle/'source_evidence/agent/api_input_000.json').exists()
    assert 'tactile' not in (bundle/'control.jsonl').read_text()
    with pytest.raises(FileExistsError):convert(source,agent,'cube',bundle)


def test_extended_hold_retains_every_recovery_command_and_original_phase(tmp_path):
    source,agent,rows=source_fixture(tmp_path,extended_hold=True)
    bundle=tmp_path/'bundle';metadata=convert(source,agent,'cube',bundle)
    trajectory=Trajectory(bundle/'control.jsonl')
    assert len(trajectory.targets)==len(rows)==168
    assert np.count_nonzero(trajectory.phases=='hold')==160
    np.testing.assert_array_equal(trajectory.targets,[row['target'] for row in rows])
    assert metadata['source_physics_audit']['source_hold_phase_duration_s']==4.
    assert metadata['evaluation_hold_interval_s']==pytest.approx([1.2,4.2])
    assert trajectory.phase_starts['hold']==pytest.approx(.2)


def test_actual_locate_done_receipts_and_action_identity(tmp_path):
    source,agent,_=source_fixture(tmp_path)
    decisions,_=audit_api_actions(source,agent)
    assert decisions[0]['action']['skill']=='locate'
    path=agent/'model_003.json';record=json.loads(path.read_text());record['action']['skill']='hold'
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError,match='differs'):audit_api_actions(source,agent)


def test_failed_last_hold_or_wrong_material_cannot_be_converted(tmp_path):
    source,agent,_=source_fixture(tmp_path)
    file=source/'evaluation.jsonl';records=[json.loads(line) for line in file.read_text().splitlines()]
    records[-1]['object_position_m'][2]=.23;records[-1]['bottom_clearance_m']=.0075
    write_rows(file,records)
    with pytest.raises(ValueError,match='criterion'):convert(source,agent,'cube',tmp_path/'bad')
    assert not (tmp_path/'bad').exists()


def add_explicit_gain(source,agent,rows,gain):
    action_path=agent/'model_003.json';model=json.loads(action_path.read_text())
    model['action']['grip_gain']=gain;action_path.write_text(json.dumps(model))
    interactions=json.loads((source/'interactions.json').read_text())
    interactions[3]['action']['grip_gain']=gain
    (source/'interactions.json').write_text(json.dumps(interactions))
    for row in rows:
        active=row['phase'] in ('close','preload','test_lift','lift','hold')
        current=gain if active else 1.
        kp=np.asarray(SOURCE_PD['kp'],dtype=float);kv=np.asarray(SOURCE_PD['kv'],dtype=float)
        kp[6:12]*=current;kv[6:12]*=np.sqrt(current)
        row.update(grip_gain=current,grip_gain_selection={'effective_gain':current,
            'requested_gain':gain if active else 0.,'observation_id':'3' if active else None,
            'source':'Astra SkillAct close' if active else 'default'},grip_gain_controller={
            'common_thumb_index_gain':current,'kp_readback':kp.tolist(),'kv_readback':kv.tolist(),
            'effort_lower_readback_Nm':[-1.]*18,'effort_upper_readback_Nm':[1.]*18,
            'effort_limits_unchanged':True})
    write_rows(source/'control.jsonl',rows)


@pytest.mark.parametrize('gain',[1.,2.,3.])
def test_explicit_api_gain_and_physical_schedule_are_preserved(tmp_path,gain):
    source,agent,rows=source_fixture(tmp_path);add_explicit_gain(source,agent,rows,gain)
    bundle=tmp_path/'bundle';metadata=convert(source,agent,'cube',bundle)
    saved=[json.loads(line) for line in (bundle/'control.jsonl').read_text().splitlines()]
    assert [row['grip_gain'] for row in saved]==[row['grip_gain'] for row in rows]
    audit=metadata['source_grip_gain_audit']
    assert audit['source_grip_gain']==gain and audit['physical_readback_available']
    assert audit['legacy_default_gain'] is False
    assert verification_gain_contract(metadata,gain)['verification_gain']==gain
    different=2. if gain==1. else 1.
    with pytest.raises(ValueError,match='source Astra-selected'):
        verification_gain_contract(metadata,different)


@pytest.mark.parametrize('corruption',['gain','kp','effort','missing','decision'])
def test_gain_source_audit_rejects_unreproducible_or_non_api_settings(tmp_path,corruption):
    source,agent,rows=source_fixture(tmp_path);add_explicit_gain(source,agent,rows,2.)
    if corruption=='gain':rows[-1]['grip_gain']=3.
    elif corruption=='kp':rows[-1]['grip_gain_controller']['kp_readback'][10]=3.
    elif corruption=='effort':rows[-1]['grip_gain_controller']['effort_upper_readback_Nm'][10]=2.
    elif corruption=='missing':del rows[-1]['grip_gain']
    else:rows[-1]['grip_gain_selection']['observation_id']='not-astra-decision'
    write_rows(source/'control.jsonl',rows)
    with pytest.raises(ValueError):convert(source,agent,'cube',tmp_path/'bad')
    assert not (tmp_path/'bad').exists()


def test_legacy_cube_verification_gain_remains_one():
    assert verification_gain_contract({},1.)['source_grip_gain']==1.
    for gain in (0.,2.,3.,True):
        with pytest.raises(ValueError):verification_gain_contract({},gain)


@pytest.mark.parametrize('object_id',['cube','cuboid','hex_prism'])
def test_canonical_flat_face_shapes_share_import_contract(tmp_path,object_id):
    source,agent,rows=source_fixture(tmp_path,object_id=object_id)
    add_explicit_gain(source,agent,rows,2.)
    metadata=convert(source,agent,object_id,tmp_path/'bundle')
    assert metadata['object_spec']['shape']==object_id
    assert metadata['source_physics_audit']['task_success']


def test_frozen_cube_scene_without_new_unused_dimensions_remains_valid(tmp_path):
    source,agent,_=source_fixture(tmp_path)
    for path in (source/'manifest.json',agent/'object_scene.json'):
        record=json.loads(path.read_text())
        record['object_spec'].pop('size_m',None)
        record['object_spec'].pop('across_flats_m',None)
        path.write_text(json.dumps(record))
    assert convert(source,agent,'cube',tmp_path/'bundle')['object_id']=='cube'
