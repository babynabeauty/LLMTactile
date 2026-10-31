"""Import actual Astra/S8V4 commands, then independently verify 300/60/15 replay.

``convert`` is CPU/file-only and does not certify replay success. ``verify``
builds the matching rigid scene at 50g/mu=.8 and executes the source-selected
absolute grip gain. Only a
successful, audited replay produces bind_record.json. Historical sensor frames
are retained as provenance, never supplied as current policy observations.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import shutil

import numpy as np

from .core import Trajectory
from .import_astra_trajectory import (CONTROL_DT, HOME_DEGREES, JOINT_ORDER,
    SOURCE_PD, convert_rows, read_json, sha256)
from .multiobject_geometry import ObjectSpec
from .train import write_json


def verify_source_scene(scene, manifest, object_id):
    spec=ObjectSpec(shape=object_id)
    for record in (scene['object_spec'],manifest['object_spec']):
        for key,value in asdict(spec).items():
            # These unused-on-cube fields were added after the already frozen
            # successful cube demonstration; do not rewrite that evidence.
            actual=(value if object_id=='cube' and key in ('size_m','across_flats_m')
                    and key not in record else record.get(key))
            matches=(np.array_equal(actual,value) if isinstance(value,(tuple,list)) else actual==value)
            if not matches:raise ValueError(f'Source object geometry differs: {key}')
    if scene.get('joint_order')!=JOINT_ORDER:
        raise ValueError('Source joint order differs')
    for key,expected in (('kp',SOURCE_PD['kp']),('kv',SOURCE_PD['kv']),
        ('home_arm_degrees',HOME_DEGREES)):
        actual=np.asarray(scene.get(key),dtype=float)
        if actual.shape!=np.shape(expected) or not np.allclose(actual,expected,rtol=0,atol=1e-7):
            raise ValueError(f'Source controller differs: {key}')
    if scene.get('control_dt_s')!=CONTROL_DT or scene.get('physics_dt_s')!=.005:
        raise ValueError('Source must use the declared 40Hz control/200Hz physics contract')
    if scene.get('silicone') is not None or scene.get('table_top_z_m')!=.2:
        raise ValueError('Source must use the rigid tabletop scene')
    if not np.allclose(scene['object_spec']['initial_position_m'],spec.initial_position(),rtol=0,atol=1e-7):
        raise ValueError('Source initial object position differs from the training scene')
    materials=scene['materials']
    if not np.isclose(materials['object_engine_mass_kg'],.05,rtol=0,atol=1e-7):
        raise ValueError('Source easiest mass must be physically 50g')
    for key in ('cube_friction','robot_friction'):
        if materials.get(key)!=.8:raise ValueError('Source easiest friction must be .8')
    np.testing.assert_allclose(materials['object_engine_inertia_kg_m2'],.05*spec.inertia_per_kg,rtol=2e-5,atol=1e-10)
    np.testing.assert_allclose(materials['object_engine_density_kg_m3'],.05/spec.volume_m3,rtol=2e-5)
    return spec


def audit_api_actions(source_run, agent_run):
    runner=read_json(agent_run/'runner.json')
    bridge=read_json(agent_run/'bridge_run.json')
    if runner.get('driver')!='vlm_policy_CodexAppServerPolicy' or runner.get('model')!='gpt-6-astra':
        raise ValueError('Demonstration must originate from the actual Astra policy API')
    if runner.get('check_only') is not False or Path(bridge['run_dir']).resolve()!=source_run:
        raise ValueError('Agent API records do not identify this physical bridge run')
    interactions=json.loads((source_run/'interactions.json').read_text())
    models=sorted(agent_run.glob('model_[0-9][0-9][0-9].json'))
    if not models or len(models)!=len(interactions):
        raise ValueError('Every bridge interaction requires an actual saved API decision')
    decisions=[];files=[];turns=set()
    for index,(path,interaction) in enumerate(zip(models,interactions)):
        if path.name!=f'model_{index:03d}.json':raise ValueError('API decisions are not contiguous')
        record=read_json(path);metadata=record['metadata']
        if metadata.get('model')!='gpt-6-astra' or metadata.get('api_mode')!='codex_app_server':
            raise ValueError('Unsupported/non-Astra decision provenance')
        if metadata.get('thread_id')!=bridge.get('thread_id') or not metadata.get('turn_id') or metadata['turn_id'] in turns:
            raise ValueError('API thread/turn identity differs or repeats')
        turns.add(metadata['turn_id'])
        if record['action']!=interaction['action']:
            raise ValueError('Executed skill action differs from the saved Astra response')
        action=record['action']
        expected_status=('model_done' if action.get('status')=='done' else
                         'located' if action.get('skill')=='locate' else 'executed')
        if interaction.get('result',{}).get('status')!=expected_status:
            raise ValueError('Demonstration contains a rejected/failed skill')
        api=agent_run/f'api_input_{index:03d}.json'
        if not api.is_file():raise ValueError('Missing actual API input record')
        files += [path,api]
        decisions.append({'index':index,'action':record['action'],
            'thread_id':metadata['thread_id'],'turn_id':metadata['turn_id'],
            'api_input_sha256':sha256(api),'model_response_sha256':sha256(path)})
    final=decisions[-1]['action']
    if final.get('status')!='done' or final.get('skill')!='none':
        raise ValueError('Astra must explicitly finish after a successful hold')
    skills=[item['action'].get('skill') for item in decisions[:-1]]
    if skills!=['locate','approach','align','close','test_lift','lift','hold'] or any(
            item['action'].get('status')!='continue' for item in decisions[:-1]):
        raise ValueError('Astra must select the complete ordered S8V4 skill chain')
    return decisions,files


def audit_source_physics(path, rows, spec):
    clearance=[];hold=[];collision=False;count=0
    with Path(path).open() as stream:
        for index,line in enumerate(stream):
            if index>=len(rows):raise ValueError('Too many evaluation rows')
            record=json.loads(line);target=rows[index]
            if record['step']!=index+1 or record['phase']!=target['phase']:
                raise ValueError('Evaluation and control records are not synchronized')
            p=np.asarray(record['object_position_m'],dtype=float)
            q=np.asarray(record['object_quaternion_wxyz'],dtype=float)
            if p.shape!=(3,) or q.shape!=(4,) or not np.isfinite(np.r_[p,q]).all():
                raise ValueError('Nonfinite source object pose')
            value=float(p[2]-spec.vertical_extent(q)-.2)
            if not np.isclose(value,record['bottom_clearance_m'],rtol=0,atol=2e-6):
                raise ValueError('Source clearance differs from primitive pose geometry')
            physical=record['object_material_readback']
            if not np.isclose(physical['object_engine_mass_kg'],.05,rtol=0,atol=1e-7):
                raise ValueError('Source mass changed during the demonstration')
            np.testing.assert_allclose(physical['object_engine_inertia_kg_m2'],.05*spec.inertia_per_kg,rtol=2e-5,atol=1e-10)
            if record.get('silicone') is not None or record.get('distractor_positions_m'):
                raise ValueError('Only the declared rigid target is allowed')
            for contact in record['solver_contacts']:
                pair=(contact['entity_a'],contact['entity_b'])
                collision |= pair==('ur7e_xhand','ur7e_xhand') or ('ur7e_xhand' in pair and 'table' in pair)
            clearance.append(value)
            if target['phase']=='hold':hold.append(value)
            count+=1
    if count!=len(rows) or len(hold)<120 or any(row['phase']!='hold' for row in rows[-120:]):
        raise ValueError('Source must finish with at least3s of uninterrupted hold labels at40Hz')
    final_hold=hold[-120:]
    success=min(final_hold)>=.05 and not collision
    if not success:raise ValueError('Source independent pose/collision audit fails the original task criterion')
    return {'task_success':success,'hold_min_clearance_m':min(final_hold),'hold_duration_s':3.,
        'source_hold_phase_duration_s':len(hold)*CONTROL_DT,
        'max_clearance_m':max(clearance),'robot_self_or_table_collision':bool(collision),
        'samples':count,'sample_hz':40,'geometry_recomputed':True}


def audit_source_grip_gain(path, decisions, rows):
    """Bind the API-selected gain to actual source PD getter readbacks.

    Older successful cube logs have no gain-selection fields and retain gain1.
    New logs require complete per-command metadata; a missing record is never
    interpreted as a nominal gain halfway through an explicit-gain episode.
    """
    close=next(item['action'] for item in decisions if item['action'].get('skill')=='close')
    requested=close.get('grip_gain',0.)
    if isinstance(requested,bool) or requested not in (0.,1.,2.,3.):
        raise ValueError('Astra close grip gain must be 1, 2, or 3 (legacy default0)')
    gain=float(requested or 1.)
    explicit=None;effort_reference=None;schedule=[]
    with Path(path).open() as stream:
        for index,line in enumerate(stream):
            source=json.loads(line);row=rows[index]
            present='grip_gain' in source
            if explicit is None:explicit=present
            if present!=explicit:raise ValueError('Incomplete source grip gain command metadata')
            expected=gain if row['phase'] in ('close','preload','test_lift','lift','hold') else 1.
            if not present:
                if requested:raise ValueError('Explicit source gain lacks physical PD readback')
                continue
            if source['grip_gain']!=expected:
                raise ValueError('Source grip gain differs from the Astra-selected close schedule')
            controller=source['grip_gain_controller'];selection=source['grip_gain_selection']
            if controller['common_thumb_index_gain']!=expected or selection['effective_gain']!=expected:
                raise ValueError('Source gain selection/controller record differs')
            if row['phase'] in ('close','preload','test_lift','lift','hold'):
                if selection['requested_gain']!=requested or selection.get('observation_id')!=close['observation_id']:
                    raise ValueError('Source PD application is not bound to the actual close decision')
                if requested and selection.get('source')!='Astra SkillAct close':
                    raise ValueError('Explicit gain was not selected by Astra close')
            expected_kp=np.asarray(SOURCE_PD['kp'],dtype=float).copy()
            expected_kv=np.asarray(SOURCE_PD['kv'],dtype=float).copy()
            expected_kp[6:12]*=expected;expected_kv[6:12]*=np.sqrt(expected)
            for name,values in (('kp_readback',expected_kp),('kv_readback',expected_kv)):
                actual=np.asarray(controller[name],dtype=float)
                if actual.shape!=(18,) or not np.allclose(actual,values,rtol=1e-6,atol=1e-7):
                    raise ValueError(f'Source physical {name} differs from absolute thumb/index gain')
            lower=np.asarray(controller['effort_lower_readback_Nm'],dtype=float)
            upper=np.asarray(controller['effort_upper_readback_Nm'],dtype=float)
            if (lower.shape!=(18,) or upper.shape!=(18,) or not np.isfinite(np.r_[lower,upper]).all()
                    or not np.allclose(lower,-upper,rtol=0,atol=1e-7) or (upper<=0).any()
                    or controller.get('effort_limits_unchanged') is not True):
                raise ValueError('Invalid source actuator force-limit readback')
            if effort_reference is None:effort_reference=upper
            elif not np.array_equal(upper,effort_reference):
                raise ValueError('Source grip gain changed actuator effort limits')
            row['grip_gain']=expected
            if not schedule or schedule[-1]['gain']!=expected:
                schedule.append({'start_s':row['before']['sim_time_s'],'end_s':row['after']['sim_time_s'],'gain':expected})
            else:schedule[-1]['end_s']=row['after']['sim_time_s']
    return {'source_grip_gain':gain,'source_close_requested_gain':float(requested),
        'source_close_observation_id':close['observation_id'],
        'physical_readback_available':bool(explicit),'command_gain_schedule':schedule,
        'legacy_default_gain':not bool(explicit),
        'readback_timing':'Source records retain the getter results from the most recent PD gain application',
        'replay_schedule':'Recorded source gains are used only for the shared physical prefix; residual controls remain absolute gains'}


def verification_gain_contract(metadata, verification_gain):
    if isinstance(verification_gain,bool) or verification_gain not in (1.,2.,3.):
        raise ValueError('Verification gain must be 1, 2, or 3')
    source=float(metadata.get('source_grip_gain_audit',{}).get('source_grip_gain',1.))
    if source!=verification_gain:
        raise ValueError('Verification gain must equal the source Astra-selected close gain; do not select it from replay outcomes')
    return {'source_grip_gain':source,'verification_gain':float(verification_gain),
        'verification_gain_schedule':{'prefix':'recorded source gains, common to all controllers',
            'residual':'absolute target with max .2/action; not multiplied by source gain',
            'gain_rate_limit_per_action':.2,'policy_gain_bounds':[.2,3.]}}


def convert(source_run, agent_run, object_id, output):
    source_run=Path(source_run).expanduser().resolve();agent_run=Path(agent_run).expanduser().resolve()
    output=Path(output).expanduser().resolve()
    if output.exists():raise FileExistsError(output)
    if source_run in output.parents or agent_run in output.parents:
        raise ValueError('Converted bundle must be outside original evidence directories')
    manifest=read_json(source_run/'manifest.json');score=read_json(source_run/'score.json')
    if manifest.get('high_level_driver')!='vlm_policy_CodexAppServerPolicy':
        raise ValueError('Source is not an Astra-driven S8V4 run')
    if score.get('success') is not True or score.get('fault') is not None:
        raise ValueError('Source run did not complete successfully')
    spec=verify_source_scene(read_json(agent_run/'object_scene.json'),manifest,object_id)
    decisions,api_files=audit_api_actions(source_run,agent_run)
    rows,intervals,feedback=convert_rows(source_run/'control.jsonl',score.get('control_steps'))
    grip_gain=audit_source_grip_gain(source_run/'control.jsonl',decisions,rows)
    physics=audit_source_physics(source_run/'evaluation.jsonl',rows,spec)
    bridge_files=[source_run/name for name in ('control.jsonl','evaluation.jsonl','manifest.json','score.json','interactions.json')]
    if manifest.get('reference_sha256'):
        reference=source_run/'demonstration_reference.json'
        if sha256(reference)!=manifest['reference_sha256']:
            raise ValueError('Historical S8V4 context differs from its recorded hash')
        bridge_files.append(reference)
    for name in ('registration.json','observed_object_cloud.npy'):
        if (source_run/name).is_file():bridge_files.append(source_run/name)
    agent_files=api_files+[agent_run/name for name in ('runner.json','bridge_run.json','object_scene.json','instructions.txt','score.json')]
    copies={**{str(path):'source_evidence/bridge/'+path.name for path in bridge_files},
            **{str(path):'source_evidence/agent/'+path.name for path in agent_files}}
    hashes={path:sha256(path) for path in copies}
    output.mkdir(parents=True,exist_ok=False)
    for path,relative in copies.items():
        destination=output/relative;destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,destination)
        if sha256(destination)!=hashes[path] or sha256(path)!=hashes[path]:
            raise ValueError('Source evidence mutated while freezing')
    target=output/'control.jsonl'
    with target.open('x') as stream:
        for row in rows:stream.write(json.dumps(row,separators=(',',':'),allow_nan=False)+'\n')
    trajectory=Trajectory(target)
    metadata={'schema':'multiobject_astra_command_trajectory_v1','status':'converted_not_replay_validated',
        'object_id':object_id,'object_spec':asdict(spec),'trajectory_path':str(target),
        'trajectory_sha256':sha256(target),'source_run':str(source_run),'agent_run':str(agent_run),
        'source_sha256':hashes,'source_snapshot_paths':copies,'source_declared_code_hashes':manifest.get('source_hashes',{}),
        'astra_decisions':decisions,'source_physics_audit':physics,'source_controller_audit':feedback,
        'source_grip_gain_audit':grip_gain,
        'source_pd':SOURCE_PD,'joint_order':JOINT_ORDER,'row_count':len(rows),
        'duration_s':trajectory.duration_s,'phase_intervals':intervals,
        'evaluation_hold_interval_s':[trajectory.duration_s-3.,trajectory.duration_s],
        'material':{'mass_kg':.05,'friction':.8},
        'rates':{'source_physics_hz':200,'source_action_hz':40,'replay_physics_hz':300,'replay_tactile_hz':60,'replay_action_hz':15},
        'conversion':'Exact logged post-S8V4 commanded targets, with original40Hz interval starts; residual runtime linearly interpolates commands at15Hz and holds each for20physics ticks. No target retuning, state teleportation or historical tactile input.',
        'replay_verification':None}
    write_json(output/'trajectory_metadata.json',metadata)
    if any(sha256(path)!=value for path,value in hashes.items()):
        raise ValueError('Source evidence changed during conversion')
    return metadata


def verify(bundle, output, verification_gain=1.):
    bundle=Path(bundle).expanduser().resolve();output=Path(output).expanduser().resolve()
    metadata=read_json(bundle/'trajectory_metadata.json')
    gain_contract=verification_gain_contract(metadata,verification_gain)
    # Genesis imports happen only for the explicit physical replay subcommand.
    from .adaptive_env import AdaptiveConfig
    from .multiobject_env import MultiObjectVecEnv
    from .multiobject_benchmark import evaluate_controller
    from .multiobject_report import trace_statistics, audit_recorded_episode, audit_environment
    from .multiobject_train import source_snapshot
    from .env import cpu
    class AuditedVerificationEnv(MultiObjectVecEnv):
        def _apply_gains(self):
            super()._apply_gains()
            kp=cpu(self.robot.get_dofs_kp(self.indices)).reshape(self.num_envs,18)
            kv=cpu(self.robot.get_dofs_kv(self.indices)).reshape(self.num_envs,18)
            expected_kp=np.broadcast_to(self.base_kp,kp.shape).copy()
            expected_kv=np.broadcast_to(self.base_kv,kv.shape).copy()
            expected_kp[:,6:9]*=self.gains[:,0,None];expected_kp[:,9:12]*=self.gains[:,1,None]
            expected_kv[:,6:9]*=np.sqrt(self.gains[:,0,None]);expected_kv[:,9:12]*=np.sqrt(self.gains[:,1,None])
            np.testing.assert_allclose(kp,expected_kp,rtol=1e-6,atol=1e-7)
            np.testing.assert_allclose(kv,expected_kv,rtol=1e-6,atol=1e-7)
            if not hasattr(self,'gain_readbacks'):self.gain_readbacks=[]
            self.gain_readbacks.append({'gains':self.gains.tolist(),
                'kp_readback':kp.tolist(),'kv_readback':kv.tolist()})
    trajectory=bundle/'control.jsonl'
    if sha256(trajectory)!=metadata['trajectory_sha256']:raise ValueError('Converted trajectory hash changed')
    for source,relative in metadata['source_snapshot_paths'].items():
        if sha256(bundle/relative)!=metadata['source_sha256'][source]:
            raise ValueError('Frozen API/source evidence hash changed')
    output.mkdir(parents=True,exist_ok=False)
    write_json(output/'manifest.json',{'trajectory_sha256':sha256(trajectory),
        'metadata_sha256':sha256(bundle/'trajectory_metadata.json'),'source_sha256':source_snapshot(output),
        'rates_hz':{'physics':300,'tactile':60,'action':15},'fixed_gain':float(verification_gain),
        **gain_contract,'training':False})
    env=None
    try:
        write_json(output/'status.json',{'status':'initializing'})
        env=AuditedVerificationEnv(object_spec=metadata['object_spec'],trajectory_path=trajectory,
            n_envs=1,seed=7000,mode='eval',material_conditions=[[.05,.8]],
            output_dir=output/'environment',config=AdaptiveConfig(gain_min=.2,failure_penalty=200.),audit_prefix=True)
        controller_name=f'fixed_{verification_gain:g}'
        result,_=evaluate_controller(env,name=controller_name,model=None,fixed_gain=verification_gain,
            output=output/controller_name,conditions=[{'condition_id':'demonstration_easy','mass_kg':.05,'friction':.8}],
            object_id=metadata['object_id'],split='preflight',training_seed=0,method=controller_name)
        readback_path=output/'controller_readback.json'
        write_json(readback_path,{'all_applications_verified':True,**gain_contract,
            'applications':env.gain_readbacks,'readback_timing':'Getter results immediately after every replay PD application'})
        with np.load(output/controller_name/'trace.npz') as trace:
            calculated=trace_statistics(trace,result['hold_interval_s'])[0]
            contact=np.asarray(trace['contact_load_N'])[:,0]
            times=trace['sample_time_s'];mask=(times>result['hold_interval_s'][0]) & (times-1/60<result['hold_interval_s'][1])
            support={'finger_order':['thumb','index','middle','ring','little'],
                'hold_mean_sensor_link_force_N':contact[mask].mean(0).tolist(),
                'hold_min_sensor_link_force_N':contact[mask].min(0).tolist(),
                'thumb_and_index_each_above_005N_every_hold_sample':bool((contact[mask,:2]>.05).all()),
                'other_finger_peak_force_N':float(contact[mask,2:].max())}
        physical=read_json(output/'environment/environment_protocol.json')
        errors=audit_recorded_episode(result['episodes'][0],calculated)
        errors+=audit_environment(physical,result['episodes'],{'objects':[
            {'object_id':metadata['object_id'],'object_spec':metadata['object_spec']}]})
        if any(physical['prefix_physics_audit']['collision_per_lane']):
            errors.append('Robot collision during physically replayed prefix')
        accepted=bool(calculated['task_success'] and not errors)
        report={'status':'passed' if accepted else 'failed','accepted':accepted,
            'audit_errors':errors,'trace_recomputed':calculated,'support_audit':support,
            **gain_contract,'controller_readback_path':str(readback_path),'controller_readback_sha256':sha256(readback_path),
            'evaluation_path':str(output/controller_name/'evaluation.json'),
            'evaluation_sha256':sha256(output/controller_name/'evaluation.json')}
        write_json(output/'verification.json',report)
        if accepted:
            evidence={'task_success':True,'strict_success':calculated['strict_success'],
                'mass_kg':.05,'friction':.8,'hold_min_clearance_m':calculated['hold_min_clearance_m'],
                'hold_duration_s':calculated['hold_duration_s'],
                'verification_path':str(output/'verification.json'),'verification_sha256':sha256(output/'verification.json'),
                'trajectory_metadata_path':str(bundle/'trajectory_metadata.json'),
                'trajectory_metadata_sha256':sha256(bundle/'trajectory_metadata.json'),
                'actual_astra_api':True,'source_run':metadata['source_run'],'agent_run':metadata['agent_run'],
                'rates_hz':{'physics':300,'tactile':60,'action':15},'fixed_gain':float(verification_gain),
                **gain_contract,'controller_readback_path':str(readback_path),'controller_readback_sha256':sha256(readback_path),
                'criterion':'Original task criterion; strict sensor-bearing-link support reported separately'}
            write_json(output/'bind_record.json',{'object_id':metadata['object_id'],
                'trajectory_path':str(trajectory),'trajectory_sha256':sha256(trajectory),
                'demonstration_evidence':evidence})
        write_json(output/'status.json',{'status':'completed','accepted':accepted})
        return report
    except BaseException as exc:
        write_json(output/'status.json',{'status':'failed','error':repr(exc)})
        raise
    finally:
        if env is not None:env.close()


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    convert_parser=sub.add_parser('convert')
    convert_parser.add_argument('--source-run',type=Path,required=True)
    convert_parser.add_argument('--agent-run',type=Path,required=True)
    convert_parser.add_argument('--object-id',choices=['cube','cuboid','hex_prism','cylinder','sphere'],required=True)
    convert_parser.add_argument('--output',type=Path,required=True)
    verify_parser=sub.add_parser('verify')
    verify_parser.add_argument('--bundle',type=Path,required=True)
    verify_parser.add_argument('--output',type=Path,required=True)
    verify_parser.add_argument('--verification-gain',type=float,choices=[1.,2.,3.],default=1.,
        help='Must equal the actual Astra close selection; source prefix gains are replayed unchanged')
    args=parser.parse_args(argv)
    result=(convert(args.source_run,args.agent_run,args.object_id,args.output) if args.command=='convert'
            else verify(args.bundle,args.output,args.verification_gain))
    print(json.dumps({k:v for k,v in result.items() if k in ('status','accepted','audit_errors','trajectory_sha256','row_count','duration_s')},indent=2))


if __name__=='__main__':main()
