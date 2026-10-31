import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from .multiobject_protocol import (make_protocol, validate_protocol, bind_demonstrations,
    amend_demonstration_gains, amend_flatface_shapes, default_object_spec,
    amend_direct_teacher_replay, protocol_shapes, LEGACY_SHAPES,
    DIRECT_TEACHER_SHAPES, OLD_TEST_CONDITIONS)
from .multiobject_report import (HOLD_FIELDS, POLICY_FIELDS, trace_statistics,
    audit_recorded_episode, summarize_group, paired_comparison,
    select_validation_checkpoint, collect, render_html, audit_environment, sha256,
    fixed_control_map, build_selection, primitive_properties)


def synthetic_trace(lanes=2):
    times = np.arange(1,241)/60
    force = np.ones((len(times),lanes))*8
    return dict(sample_time_s=times,clearance_m=np.full_like(force,.06),
        fault=np.zeros_like(force,dtype=bool),
        contact_load_N=np.broadcast_to([3.,3.,0.,0.,0.],force.shape+(5,)).copy(),
        all_robot_contact_load_N=force,all_robot_normal_load_N=force*.75,
        contact_friction_error=np.zeros_like(force))


def direct_replay_mapping(root, protocol):
    """Synthetic hash-bound provenance only; this fixture does not simulate."""
    mapping = {}
    profile = dict(model_version='kinematic_force_on_sensor_v3', fixture=True)
    source_profile = dict(profile,source_semantics_inferred_from_current_code=False,raw_data_relabelled=False)
    source = dict(demonstration_source='direct_teacher_frozen_replay',
                  privileged_teacher=True, actual_astra_api=False)
    for obj in protocol['objects']:
        directory = Path(root)/obj['object_id'];directory.mkdir()
        trajectory = directory/'control.jsonl';trajectory.write_text('synthetic commands\n')
        collection = directory/'collection_validation.json'
        collection.write_text(json.dumps(dict(status='passed',audit_errors=[],accepted=True,
            fresh_tactile_captured=True,raw_tactile_reused=False,exact_source_commands_replayed=True,
            source_files_unchanged=True,current_source_unchanged=True,tactile_model=profile)))
        metadata = directory/'trajectory_metadata.json'
        metadata.write_text(json.dumps(dict(source,object_id=obj['object_id'],object_spec=obj['object_spec'],
            trajectory_sha256=sha256(trajectory),source_grip_gain_audit=dict(source_grip_gain=2.),
            source_tactile_model=source_profile,frozen_replay_audit=dict(
                validation_sha256=sha256(collection),current_tactile_model=profile,
                exact_historical_commands_recomputed=True,fresh_tactile_audit_recomputed=True,
                collection_validation_is_independent_replay_verification=False))))
        readback = directory/'controller_readback.json'
        readback.write_text(json.dumps(dict(all_applications_verified=True)))
        common = dict(source,source_grip_gain=2.,verification_gain=2.,
            source_tactile_model=source_profile,tactile_model=profile,
            source_replay_validation_path=str(collection),source_replay_validation_sha256=sha256(collection),
            controller_readback_path=str(readback),controller_readback_sha256=sha256(readback))
        verification = directory/'verification.json'
        verification.write_text(json.dumps(dict(common,accepted=True)))
        evidence = dict(common,task_success=True,mass_kg=.05,friction=.8,
            trajectory_metadata_path=str(metadata),trajectory_metadata_sha256=sha256(metadata),
            verification_path=str(verification),verification_sha256=sha256(verification))
        mapping[obj['object_id']] = dict(trajectory_path=str(trajectory),demonstration_evidence=evidence)
    return mapping


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.protocol=make_protocol()

    def test_disjoint_and_new_test(self):
        self.assertTrue(validate_protocol(self.protocol))
        test={(r['mass_kg'],r['friction']) for r in self.protocol['splits']['test']}
        self.assertFalse(test & OLD_TEST_CONDITIONS)
        self.assertEqual([len(v) for v in self.protocol['splits'].values()],[12,6,9])
        self.assertEqual(self.protocol['training']['planned_timesteps_per_model_36_lanes'],792576)

    def test_reject_cross_split_pair(self):
        self.protocol['splits']['validation'][0].update(mass_kg=.05,friction=.2)
        with self.assertRaisesRegex(ValueError,'overlap'):
            validate_protocol(self.protocol)

    def test_reject_reused_old_test(self):
        self.protocol['splits']['test'][0].update(mass_kg=.075,friction=.3)
        with self.assertRaisesRegex(ValueError,'previous nine'):
            validate_protocol(self.protocol)

    def test_pending_demonstrations_cannot_run(self):
        with self.assertRaisesRegex(ValueError,'Missing verified'):
            validate_protocol(self.protocol,require_demonstrations=True)

    def test_demonstration_hash_and_evidence_required(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'demo.jsonl';path.write_text('demo\n')
            mapping={shape:dict(trajectory_path=str(path),demonstration_evidence=dict(
                task_success=True,mass_kg=.05,friction=.8)) for shape in protocol_shapes(self.protocol)}
            result=bind_demonstrations(self.protocol,mapping)
            self.assertEqual(result['status'],'frozen_ready_for_training')
            path.write_text('changed')
            with self.assertRaisesRegex(ValueError,'hash mismatch'):
                validate_protocol(result,True)
            mapping['cube']['demonstration_evidence']['task_success']=False
            with self.assertRaisesRegex(ValueError,'evidence'):
                bind_demonstrations(self.protocol,mapping)

    def test_gain_amendment_preserves_experiment_design(self):
        before=copy.deepcopy(self.protocol)
        after=amend_demonstration_gains(before,'v2-sha256')
        for key in ('objects','splits','scene','training','success','evaluation','split_contract'):
            self.assertEqual(after[key],before[key])
        self.assertEqual(before,self.protocol)
        self.assertEqual(after['demonstration_control']['source_gain_choices'],[1.,2.,3.])
        self.assertIn('never multiplied',after['control']['action'])
        self.assertIn('recorded source gain schedule',after['control']['initial_state'])
        self.assertEqual(after['demonstration_gain_amendment']['previous_candidate_sha256'],'v2-sha256')

    def test_flatface_amendment_preserves_design_and_keeps_legacy_readable(self):
        previous=version2_protocol()
        previous['objects'][0]['trajectory_path']='existing-cube-evidence'
        result=amend_flatface_shapes(previous,'v3-sha256')
        self.assertEqual(protocol_shapes(result),('cube','cuboid','hex_prism'))
        for key in ('training','splits','control','scene','evaluation'):
            self.assertEqual(result[key],previous[key])
        self.assertEqual(result['objects'][0]['trajectory_path'],'existing-cube-evidence')
        self.assertEqual(result['objects'][1]['object_spec']['size_m'],[.045,.060,.045])
        self.assertIn('No claim of curved-object',result['split_contract']['meaning'])
        self.assertEqual(protocol_shapes(previous),LEGACY_SHAPES)
        self.assertTrue(validate_protocol(previous))

    def gain_mapping(self,directory,gain=2.):
        root=Path(directory)
        trajectory=root/'control.jsonl';trajectory.write_text('command\n')
        metadata=root/'trajectory_metadata.json'
        metadata.write_text(json.dumps(dict(trajectory_sha256=sha256(trajectory),
            source_grip_gain_audit=dict(source_grip_gain=gain))))
        verification=root/'verification.json'
        verification.write_text(json.dumps(dict(accepted=True,source_grip_gain=gain,verification_gain=gain)))
        readback=root/'controller_readback.json'
        readback.write_text(json.dumps(dict(all_applications_verified=True)))
        evidence=dict(task_success=True,mass_kg=.05,friction=.8,actual_astra_api=True,
            source_grip_gain=gain,verification_gain=gain,fixed_gain=gain,
            trajectory_metadata_path=str(metadata),trajectory_metadata_sha256=sha256(metadata),
            verification_path=str(verification),verification_sha256=sha256(verification),
            controller_readback_path=str(readback),controller_readback_sha256=sha256(readback))
        mapping={shape:dict(trajectory_path=str(trajectory),trajectory_sha256=sha256(trajectory),
            demonstration_evidence=copy.deepcopy(evidence)) for shape in protocol_shapes(self.protocol)}
        # Existing gain1 cube records need not be rewritten to the new schema.
        mapping['cube']['demonstration_evidence']=dict(task_success=True,mass_kg=.05,friction=.8,fixed_gain=1.)
        return mapping

    def test_new_gain_evidence_and_legacy_cube_bind_together(self):
        with tempfile.TemporaryDirectory() as directory:
            mapping=self.gain_mapping(directory)
            result=bind_demonstrations(self.protocol,mapping)
            self.assertTrue(validate_protocol(result,True))
            self.assertEqual(result['objects'][1]['demonstration_evidence']['source_grip_gain'],2.)
            self.assertNotIn('source_grip_gain',result['objects'][0]['demonstration_evidence'])

    def test_reject_posthoc_verification_gain_change(self):
        with tempfile.TemporaryDirectory() as directory:
            mapping=self.gain_mapping(directory)
            mapping['cuboid']['demonstration_evidence']['verification_gain']=3.
            with self.assertRaisesRegex(ValueError,'post-hoc gain search'):
                bind_demonstrations(self.protocol,mapping)

    def test_reject_gain_artifact_tamper_or_unaccepted_verification(self):
        for changed,pattern in [('tamper','hash mismatch'),('unaccepted','did not pass'),
                                ('gain','differs from independent verification')]:
            with self.subTest(changed=changed), tempfile.TemporaryDirectory() as directory:
                mapping=self.gain_mapping(directory)
                evidence=mapping['cuboid']['demonstration_evidence']
                path=Path(evidence['verification_path'])
                record=json.loads(path.read_text())
                if changed=='gain':record['verification_gain']=3.
                else:record['accepted']=False
                path.write_text(json.dumps(record))
                if changed!='tamper':evidence['verification_sha256']=sha256(path)
                with self.assertRaisesRegex(ValueError,pattern):
                    bind_demonstrations(self.protocol,mapping)

    def test_reject_gain_metadata_identifying_other_trajectory(self):
        with tempfile.TemporaryDirectory() as directory:
            mapping=self.gain_mapping(directory)
            evidence=mapping['cuboid']['demonstration_evidence']
            path=Path(evidence['trajectory_metadata_path'])
            record=json.loads(path.read_text());record['trajectory_sha256']='other'
            path.write_text(json.dumps(record));evidence['trajectory_metadata_sha256']=sha256(path)
            with self.assertRaisesRegex(ValueError,'different trajectory'):
                bind_demonstrations(self.protocol,mapping)

    def test_reject_nonlegacy_gain_without_new_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            mapping=self.gain_mapping(directory)
            mapping['cube']['demonstration_evidence']['fixed_gain']=2.
            with self.assertRaisesRegex(ValueError,'requires source and verification'):
                bind_demonstrations(self.protocol,mapping)

    def test_reject_supplied_bind_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            mapping=self.gain_mapping(directory);mapping['cube']['trajectory_sha256']='other'
            with self.assertRaisesRegex(ValueError,'Bind record demonstration hash mismatch'):
                bind_demonstrations(self.protocol,mapping)

    def test_four_shape_amendment_preserves_budgets_splits_and_fixed_controls(self):
        before = version2_protocol()
        result = amend_direct_teacher_replay(before, 'previous-sha')
        self.assertEqual(protocol_shapes(result), DIRECT_TEACHER_SHAPES)
        self.assertEqual(result['objects'][-1]['object_spec']['height_m'], .09)
        self.assertEqual(result['splits'], before['splits'])
        self.assertEqual(result['training']['requested_timesteps_per_model'], 786432)
        self.assertEqual(result['training']['planned_timesteps_per_model'], 792576)
        self.assertEqual(result['training']['proposed_total_envs'], 48)
        for a,b in zip(result['training']['stages'],before['training']['stages']):
            self.assertEqual(a['requested_timesteps'], b['requested_timesteps'])
            self.assertEqual(a['config'], b['config'])
        self.assertEqual(result['evaluation']['fixed_gains'], [.6,1.,2.,3.])
        self.assertEqual(result['evaluation']['expected_test_policy_episodes'], 216)
        self.assertEqual(result['evaluation']['expected_test_fixed_unique_episodes'], 144)
        self.assertIn('24 validation cases', result['training']['checkpoint_selection'])
        self.assertNotIn('planned_timesteps_per_model_36_lanes', result['training'])
        self.assertEqual(result['status'], 'candidate_pending_verified_demonstrations')
        self.assertTrue(all(o['trajectory_path'] is None for o in result['objects']))
        self.assertEqual(protocol_shapes(before), LEGACY_SHAPES)

    def test_four_shape_bind_requires_honest_independent_v3_replay(self):
        protocol = amend_direct_teacher_replay(self.protocol, 'fixture')
        with tempfile.TemporaryDirectory() as directory:
            mapping = direct_replay_mapping(directory, protocol)
            bound = bind_demonstrations(protocol, mapping)
            self.assertTrue(validate_protocol(bound, True))
            self.assertFalse(bound['objects'][0]['demonstration_evidence']['actual_astra_api'])
            mapping['cuboid']['demonstration_evidence']['actual_astra_api'] = True
            with self.assertRaisesRegex(ValueError, 'non-Astra'):
                bind_demonstrations(protocol, mapping)

    def test_four_shape_cannot_fall_back_to_legacy_gain_provenance(self):
        protocol = amend_direct_teacher_replay(self.protocol, 'fixture')
        with tempfile.TemporaryDirectory() as directory:
            mapping = direct_replay_mapping(directory, protocol)
            mapping['cuboid']['demonstration_evidence'] = dict(task_success=True,mass_kg=.05,friction=.8)
            with self.assertRaisesRegex(ValueError, 'explicit direct_teacher_frozen_replay'):
                bind_demonstrations(protocol, mapping)

    def test_frozen_replay_rejects_invalid_provenance(self):
        changes = [('collection_reused', 'fresh-sensing'),('verification_source', 'non-Astra'),
                   ('old_tactile', 'v3 source'),('missing_readback', 'requires both path'),
                   ('collection_as_verification', 'independent'),('wrong_geometry', 'geometry differs')]
        for change,pattern in changes:
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                protocol = amend_direct_teacher_replay(self.protocol, 'fixture')
                mapping = direct_replay_mapping(directory, protocol)
                evidence = mapping['cuboid']['demonstration_evidence']
                metadata_path = Path(evidence['trajectory_metadata_path'])
                metadata = json.loads(metadata_path.read_text())
                verification_path = Path(evidence['verification_path'])
                verification = json.loads(verification_path.read_text())
                if change == 'collection_reused':
                    path = Path(evidence['source_replay_validation_path']);value=json.loads(path.read_text())
                    value['raw_tactile_reused']=True;path.write_text(json.dumps(value))
                    evidence['source_replay_validation_sha256']=sha256(path)
                    metadata['frozen_replay_audit']['validation_sha256']=sha256(path)
                    verification['source_replay_validation_sha256']=sha256(path)
                elif change == 'verification_source':verification['actual_astra_api']=True
                elif change == 'old_tactile':metadata['source_tactile_model']['model_version']='legacy_unversioned'
                elif change == 'missing_readback':del evidence['controller_readback_path']
                elif change == 'collection_as_verification':
                    metadata['frozen_replay_audit']['collection_validation_is_independent_replay_verification']=True
                elif change == 'wrong_geometry':metadata['object_spec']['height_m']=.09
                metadata_path.write_text(json.dumps(metadata));evidence['trajectory_metadata_sha256']=sha256(metadata_path)
                verification_path.write_text(json.dumps(verification));evidence['verification_sha256']=sha256(verification_path)
                with self.assertRaisesRegex(ValueError, pattern):
                    bind_demonstrations(protocol, mapping)


class TraceTests(unittest.TestCase):
    def test_constant_forces_and_success(self):
        result=trace_statistics(synthetic_trace(),(.875,3.875))
        self.assertEqual(len(result),2)
        for row in result:
            self.assertTrue(row['task_success']);self.assertTrue(row['strict_success'])
            self.assertAlmostEqual(row['hold_mean_all_robot_contact_force_N'],8)
            self.assertAlmostEqual(row['hold_mean_all_robot_normal_force_N'],6)
            self.assertAlmostEqual(row['hold_mean_sensor_contact_force_N'],6)
            self.assertFalse(audit_recorded_episode(row,row))

    def test_partial_last_sample_counts_for_height_and_force(self):
        trace=synthetic_trace()
        # Last included interval is (3.8667,3.8833], contributing only .008333s.
        trace['clearance_m'][232,0]=.049
        trace['all_robot_contact_load_N'][232,0]=14
        result=trace_statistics(trace,(.875,3.875))
        self.assertFalse(result[0]['task_success']);self.assertTrue(result[1]['task_success'])
        self.assertAlmostEqual(result[0]['hold_mean_all_robot_contact_force_N'],8+6/120/3)
        self.assertEqual(result[0]['hold_peak_all_robot_contact_force_N'],14)

    def test_sample_after_hold_does_not_fail_success(self):
        trace=synthetic_trace();trace['clearance_m'][233:,0]=0
        self.assertTrue(trace_statistics(trace,(.875,3.875))[0]['task_success'])

    def test_collision_or_missing_support(self):
        trace=synthetic_trace();trace['fault'][5,0]=True
        trace['contact_load_N'][:,1,1]=0
        result=trace_statistics(trace,(.875,3.875))
        self.assertFalse(result[0]['task_success'])
        self.assertTrue(result[1]['task_success']);self.assertFalse(result[1]['strict_success'])

    def test_reject_missing_window_or_bad_friction(self):
        trace=synthetic_trace()
        with self.assertRaisesRegex(ValueError,'complete'):
            trace_statistics(trace,(1.5,4.5))
        trace['contact_friction_error'][1,0]=.01
        with self.assertRaisesRegex(ValueError,'friction'):
            trace_statistics(trace,(.875,3.875))

    def test_reject_nonfinite_force(self):
        trace=synthetic_trace();trace['all_robot_contact_load_N'][3,0]=np.nan
        with self.assertRaisesRegex(ValueError,'Nonfinite'):
            trace_statistics(trace,(.875,3.875))

    def test_wrong_reported_metric_not_silently_replaced(self):
        actual=trace_statistics(synthetic_trace(),(.875,3.875))[0]
        reported=dict(actual,hold_mean_all_robot_contact_force_N=0.)
        self.assertTrue(any('hold_mean_all_robot_contact_force_N' in x for x in audit_recorded_episode(reported,actual)))


def episode(success=True,force=8,condition='test_m080_mu030',seed=1,method='tactile',shape='cube'):
    return dict(task_success=success,strict_success=success,condition_id=condition,
        training_seed=seed,method=method,object_shape=shape,
        **{key:force for key in HOLD_FIELDS+POLICY_FIELDS})


def write_bundle(root,protocol,split,shape,method,seed,stage,force=8,group=None):
    """Write internally consistent synthetic benchmark evidence, never physics."""
    conditions=protocol['splits'][split]
    group=group or stage or 'fixed'
    benchmark=root/'eval'/split/group
    controller=f'seed{seed}_{method}' if method in ('tactile','no_tactile') else method
    out=benchmark/controller/f'shape_{shape}'/'chunk_000';out.mkdir(parents=True)
    trace=synthetic_trace(len(conditions))
    trace['all_robot_contact_load_N'][:]=force
    trace['all_robot_normal_load_N'][:]=force*.75
    trace['contact_load_N'][:]=[force*.35,force*.35,0,0,0]
    np.savez_compressed(out/'trace.npz',**trace)
    rows=[]
    for condition,metric in zip(conditions,trace_statistics(trace,(.875,3.875))):
        rows.append(dict(metric,**condition,object_shape=shape,split=split,
            training_seed=seed,method=method,checkpoint_stage=stage,controller=controller))
    manifest_path=benchmark/'manifest.json'
    manifest=json.loads(manifest_path.read_text()) if manifest_path.exists() else dict(split=split,models={},fixed_gains=[])
    checkpoint=None
    if method in ('tactile','no_tactile'):
        checkpoint=root/'models'/f'{controller}_{stage}.zip';checkpoint.parent.mkdir(exist_ok=True)
        checkpoint.write_text(f'{controller} {stage}')
        manifest['models'][controller]=dict(path=str(checkpoint),sha256=sha256(checkpoint),
            training_seed=seed,method=method,checkpoint_stage=stage)
    else:
        gain=fixed_control_map(protocol['evaluation']['fixed_gains'])[method]
        if gain not in manifest['fixed_gains']:manifest['fixed_gains'].append(gain)
    manifest_path.write_text(json.dumps(manifest))
    spec=next(o['object_spec'] for o in protocol['objects'] if o['object_id']==shape)
    volume,bounds,inertia=primitive_properties(spec)
    bounds=bounds.tolist()
    physical_spec={'shape':shape}
    if shape=='hex_prism':
        from .multiobject_geometry import ObjectSpec
        object_spec=ObjectSpec.from_value(spec)
        physical_spec=object_spec.protocol()
        physical_spec['mesh_asset']=object_spec.mesh_asset(directory=benchmark/'shape-assets',create=True)
    masses=np.array([c['mass_kg'] for c in conditions])
    env=benchmark/'environment'/shape/'environment_protocol.json';env.parent.mkdir(parents=True,exist_ok=True)
    env.write_text(json.dumps(dict(object_spec=physical_spec,mass_readback_kg_per_lane=masses.tolist(),
        density_readback_kg_m3_per_lane=(masses/volume).tolist(),
        inertia_readback_kg_m2_per_lane=(masses[:,None,None]*inertia).tolist(),
        object_bounds_readback_m=bounds,object_initial_bottom_readback_m=[.2]*len(conditions),
        friction_per_lane=[c['friction'] for c in conditions])))
    (out/'prefix_consistency.json').write_text('{"state":0,"tactile":0}')
    (out/'evaluation.json').write_text(json.dumps(dict(episodes=rows,hold_interval_s=[.875,3.875],
        trace_sha256=sha256(out/'trace.npz'),environment_protocol_path=str(env),
        environment_protocol_sha256=sha256(env),split=split,controller=controller,
        method=method,training_seed=seed,checkpoint_stage=stage)))
    return checkpoint


def legacy_protocol():
    protocol=make_protocol()
    protocol['objects']=[dict(object_id=shape,object_spec=default_object_spec(shape),
        trajectory_path=None,trajectory_sha256=None,demonstration_evidence=None) for shape in LEGACY_SHAPES]
    return protocol


def version2_protocol():
    protocol=legacy_protocol()
    for stage in protocol['training']['stages']:stage['config']['gain_min']=.2
    protocol['evaluation']['fixed_gains']=[.6,1.,2.,3.]
    protocol['evaluation']['expected_test_fixed_unique_episodes']=108
    return protocol


class ComparisonTests(unittest.TestCase):
    def test_fractional_canonical_names_and_duplicate_rejection(self):
        self.assertEqual(fixed_control_map([.6,1.,2.,3.]),
                         {'fixed_0p6':.6,'fixed_1':1.,'fixed_2':2.,'fixed_3':3.})
        self.assertEqual(fixed_control_map([.25,.75]),{'fixed_0p25':.25,'fixed_0p75':.75})
        with self.assertRaises(ValueError):fixed_control_map([1.,1])
        with self.assertRaises(ValueError):fixed_control_map([float('nan')])

    def test_failed_zero_force_not_counted_as_efficient(self):
        summary=summarize_group([episode(True,10),episode(False,0)])
        self.assertEqual(summary['task_success_rate'],.5)
        self.assertEqual(summary['successful_hold_forces'][HOLD_FIELDS[0]],10)
        self.assertEqual(summary['all_episode_hold_forces_descriptive_only'][HOLD_FIELDS[0]],5)

    def test_only_joint_success_paired(self):
        left=[episode(True,6),episode(False,0,'other')]
        right=[episode(True,10,method='no_tactile'),episode(True,9,'other',method='no_tactile')]
        pair=paired_comparison(left,right,'no_tactile')
        self.assertEqual(pair['paired_cases'],2);self.assertEqual(pair['jointly_successful_cases'],1)
        self.assertEqual(pair['comparator_only_success'],1)
        self.assertEqual(pair['jointly_successful_force_comparison'][HOLD_FIELDS[0]]['relative_change_percent'],-40)

    def test_fixed_seedless_reused_without_extra_baseline_rows(self):
        left=[episode(seed=i) for i in (1,2,3)]
        baseline=[episode(method='fixed_2',seed=None)]
        pair=paired_comparison(left,baseline,'fixed_2')
        self.assertEqual(pair['paired_cases'],3)
        self.assertEqual(len(baseline),1)

    def test_empty_success_returns_null_not_nan(self):
        summary=summarize_group([episode(False,0)])
        self.assertIsNone(summary['successful_hold_forces'][HOLD_FIELDS[0]])
        json.dumps(summary,allow_nan=False)


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.protocol=make_protocol()
        self.rows={stage:[dict(episode(condition=c['condition_id'],shape=shape),
            split='validation',checkpoint_stage=stage,source_evaluation='synthetic.json')
            for shape in protocol_shapes(self.protocol) for c in self.protocol['splits']['validation']]
            for stage in ('stage1','stage2')}
        self.directory=tempfile.TemporaryDirectory()
        self.paths={stage:Path(self.directory.name)/stage for stage in self.rows}
        for stage,path in self.paths.items():
            path.write_text(stage)
            for row in self.rows[stage]:row['evaluated_checkpoint_sha256']=sha256(path)

    def tearDown(self):self.directory.cleanup()

    def select(self):
        return select_validation_checkpoint(self.rows['stage1'],self.rows['stage2'],self.protocol,self.paths)

    def test_success_count_precedes_force(self):
        self.rows['stage1'][0]['task_success']=False
        for row in self.rows['stage1']:row['hold_mean_all_robot_contact_force_N']=.1
        self.assertEqual(self.select()['selected_stage'],'stage2')

    def test_equal_success_uses_common_cases(self):
        for row in self.rows['stage1']:row['hold_mean_all_robot_contact_force_N']=7
        self.assertEqual(self.select()['selected_stage'],'stage1')
        self.rows['stage1'][0]['task_success']=False
        self.rows['stage2'][1]['task_success']=False
        self.rows['stage2'][0]['hold_mean_all_robot_contact_force_N']=1000
        result=self.select()
        self.assertEqual(result['jointly_successful_cases'],16)
        self.assertEqual(result['joint_success_force_N']['stage2'],8)

    def test_ties_default_stage2(self):
        self.assertEqual(self.select()['selected_stage'],'stage2')

    def test_reject_wrong_validation_checkpoint_hash(self):
        self.paths['stage1'].write_text('unrelated model with same stage label')
        with self.assertRaisesRegex(ValueError,'actually used'):
            self.select()

    def test_reject_test_data_or_incomplete_validation(self):
        self.rows['stage1'][0]['split']='test'
        with self.assertRaisesRegex(ValueError,'validation'):
            self.select()
        self.rows['stage1'][0]['split']='validation';self.rows['stage1'].pop()
        with self.assertRaisesRegex(ValueError,'all18'):
            self.select()

    def test_four_shapes_select_from_all24_validation_cases(self):
        self.protocol = amend_direct_teacher_replay(self.protocol, 'fixture')
        self.rows = {stage:[dict(episode(condition=c['condition_id'],shape=shape),
            split='validation',checkpoint_stage=stage,source_evaluation='synthetic.json',
            evaluated_checkpoint_sha256=sha256(self.paths[stage]))
            for shape in protocol_shapes(self.protocol) for c in self.protocol['splits']['validation']]
            for stage in ('stage1','stage2')}
        self.assertEqual(self.select()['validation_case_count'],24)
        self.rows['stage1'].pop()
        with self.assertRaisesRegex(ValueError,'all24'):
            self.select()


class CollectionTests(unittest.TestCase):
    def test_four_shape_coverage_and_honest_source_report(self):
        protocol=amend_direct_teacher_replay(version2_protocol(),'fixture')
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for obj in protocol['objects']:
                obj['demonstration_evidence']=dict(demonstration_source='direct_teacher_frozen_replay',
                    actual_astra_api=False,privileged_teacher=True)
            for shape in protocol_shapes(protocol):
                for seed in (1,2,3):
                    for method in ('tactile','no_tactile'):
                        write_bundle(root,protocol,'test',shape,method,seed,'stage2',group='selected')
                for method in fixed_control_map(protocol['evaluation']['fixed_gains']):
                    write_bundle(root,protocol,'test',shape,method,0,None,group='selected')
            result=collect(root,protocol)
        self.assertEqual(result['status'],'audited_complete',result['audit_errors'])
        self.assertEqual(result['expected_policy_episodes'],216)
        self.assertEqual(result['expected_fixed_unique_episodes'],144)
        self.assertEqual(result['audited_episode_count'],360)
        self.assertEqual(set(result['by_shape']),set(DIRECT_TEACHER_SHAPES))
        self.assertEqual(len(result['demonstration_provenance']),4)
        self.assertTrue(all(r['actual_astra_api'] is False for r in result['demonstration_provenance']))
        page=render_html(result)
        self.assertIn('不是 Astra API 推理',page)
        self.assertIn('原始零响应未填补',page)

    def test_flatface_full_coverage_and_checkpoint_selection(self):
        protocol=amend_flatface_shapes(version2_protocol(),'v3-fixture');mapping={}
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for stage in ('stage1','stage2'):
                for shape in protocol_shapes(protocol):
                    for seed in (1,2,3):
                        for method in ('tactile','no_tactile'):
                            path=write_bundle(root,protocol,'validation',shape,method,seed,stage)
                            mapping.setdefault(f'seed{seed}_{method}',{})[stage]=path
                            if stage=='stage2':
                                write_bundle(root,protocol,'test',shape,method,seed,stage,group='selected')
                    if stage=='stage1':
                        for method in fixed_control_map(protocol['evaluation']['fixed_gains']):
                            write_bundle(root,protocol,'validation',shape,method,0,None,group='stage1')
                            write_bundle(root,protocol,'test',shape,method,0,None,group='selected')
            result=collect(root,protocol)
            self.assertEqual(result['status'],'audited_complete',result['audit_errors'])
            self.assertEqual(result['audited_episode_count'],270)
            self.assertEqual(set(result['by_shape']),{'cube','cuboid','hex_prism'})
            self.assertIn('正六棱柱',render_html(result))
            selection=build_selection(root,protocol,mapping)
            self.assertEqual(len(selection['selected_models']),6)
            self.assertTrue(all(m['checkpoint_stage']=='stage2' for m in selection['selected_models'].values()))

    def test_flatface_analytic_properties_have_declared_dimensions(self):
        volume,bounds,inertia=primitive_properties(default_object_spec('cuboid'))
        self.assertAlmostEqual(volume,.0001215)
        np.testing.assert_allclose(bounds,[.045,.060,.045])
        np.testing.assert_allclose(np.diag(inertia),[.00046875,.0003375,.00046875])
        volume,bounds,inertia=primitive_properties(default_object_spec('hex_prism'))
        self.assertAlmostEqual(volume,7.891656491985695e-05)
        np.testing.assert_allclose(bounds,[.045,.05196152422706631,.045])
        np.testing.assert_allclose(np.diag(inertia),[.000309375,.000309375,.00028125])

    def test_hex_mesh_mutation_fails_physical_audit(self):
        protocol=amend_flatface_shapes(version2_protocol(),'v3-fixture')
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            write_bundle(root,protocol,'test','hex_prism','tactile',1,'stage2',group='selected')
            evaluation=next((root/'eval').rglob('evaluation.json'))
            bundle=json.loads(evaluation.read_text())
            environment=json.loads(Path(bundle['environment_protocol_path']).read_text())
            mesh=Path(environment['object_spec']['mesh_asset']['path'])
            mesh.write_text(mesh.read_text()+'\n# changed\n')
            self.assertTrue(any('mesh file hash' in error for error in
                audit_environment(environment,bundle['episodes'],protocol)))

    def test_v2_full_test_coverage_includes_fractional_baseline(self):
        protocol=version2_protocol()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for shape in protocol_shapes(protocol):
                for seed in (1,2,3):
                    for method in ('tactile','no_tactile'):
                        write_bundle(root,protocol,'test',shape,method,seed,'stage2',group='selected')
                for method in fixed_control_map(protocol['evaluation']['fixed_gains']):
                    write_bundle(root,protocol,'test',shape,method,0,None,group='selected')
            result=collect(root,protocol)
        self.assertFalse(result['audit_errors'])
        self.assertEqual(result['status'],'audited_complete')
        self.assertEqual(result['audited_episode_count'],270)
        self.assertEqual(result['expected_fixed_unique_episodes'],108)
        self.assertEqual(result['overall']['fixed_0p6']['episodes'],27)
        comparison=next(p for p in result['paired_comparisons']
                        if p['object_shape']=='all' and p['comparator']=='fixed_0p6')
        self.assertEqual(comparison['paired_cases'],81)
        self.assertIn('固定增益 0.6',render_html(result))

    def test_v2_validation_reuses_fixed_once_and_selects_only_policy_stages(self):
        protocol=version2_protocol();mapping={}
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for stage,force in [('stage1',8.),('stage2',7.)]:
                for shape in protocol_shapes(protocol):
                    for seed in (1,2,3):
                        for method in ('tactile','no_tactile'):
                            path=write_bundle(root,protocol,'validation',shape,method,seed,stage,force=force)
                            mapping.setdefault(f'seed{seed}_{method}',{})[stage]=path
                    if stage=='stage1':
                        for method in fixed_control_map(protocol['evaluation']['fixed_gains']):
                            write_bundle(root,protocol,'validation',shape,method,0,None,force=2.4,group='stage1')
            for stage in ('stage1','stage2'):
                report=collect(root,protocol,'validation',stage)
                self.assertEqual(report['status'],'audited_complete',report['audit_errors'])
                self.assertEqual(report['expected_fixed_unique_episodes'],72)
                self.assertEqual(report['overall']['fixed_0p6']['episodes'],18)
            selection=build_selection(root,protocol,mapping)
        self.assertEqual(len(selection['selected_models']),6)
        self.assertTrue(all(m['checkpoint_stage']=='stage2' for m in selection['selected_models'].values()))
        self.assertNotIn('fixed_0p6',selection['selected_models'])
        self.assertFalse(selection['test_outcomes_read'])

    def test_readbacks_use_real_sphere_volume(self):
        protocol=legacy_protocol();r=.0225;mass=.08
        environment=dict(object_spec={'shape':'sphere'},mass_readback_kg_per_lane=[mass],
            density_readback_kg_m3_per_lane=[mass/(4*np.pi*r**3/3)],
            inertia_readback_kg_m2_per_lane=[(np.eye(3)*mass*2*r*r/5).tolist()],
            object_bounds_readback_m=[2*r]*3,object_initial_bottom_readback_m=[.2],friction_per_lane=[.3])
        rows=[dict(object_shape='sphere',mass_kg=mass,friction=.3)]
        self.assertFalse(audit_environment(environment,rows,protocol))
        environment['density_readback_kg_m3_per_lane']=[mass/.045**3]
        self.assertTrue(any('density' in e for e in audit_environment(environment,rows,protocol)))

    def test_collection_audits_one_real_file_bundle(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);out=root/'eval/test/seed1_tactile/shape_cube/chunk_000';out.mkdir(parents=True)
            trace=synthetic_trace(1);np.savez_compressed(out/'trace.npz',**trace)
            metric=trace_statistics(trace,(.875,3.875))[0]
            row=dict(metric,condition_id='test_m080_mu030',object_shape='cube',split='test',
                training_seed=1,method='tactile',mass_kg=.08,friction=.3,checkpoint_stage='stage2',controller='seed1_tactile')
            (root/'manifest.json').write_text(json.dumps(dict(split='test',models={'seed1_tactile':dict(
                path='model.zip',sha256='0'*64,training_seed=1,method='tactile',checkpoint_stage='stage2')})))
            env=root/'environment.json'
            env.write_text(json.dumps(dict(object_spec={'shape':'cube'},mass_readback_kg_per_lane=[.08],
                density_readback_kg_m3_per_lane=[.08/.045**3],
                inertia_readback_kg_m2_per_lane=[(np.eye(3)*.08*.045**2/6).tolist()],
                object_bounds_readback_m=[.045]*3,object_initial_bottom_readback_m=[.2],friction_per_lane=[.3])))
            (out/'prefix_consistency.json').write_text('{"state":0,"tactile":0}')
            (out/'evaluation.json').write_text(json.dumps(dict(episodes=[row],hold_interval_s=[.875,3.875],
                trace_sha256=sha256(out/'trace.npz'),environment_protocol_path=str(env),
                environment_protocol_sha256=sha256(env),split='test',controller='seed1_tactile',
                method='tactile',training_seed=1,checkpoint_stage='stage2')))
            report=collect(root,make_protocol())
        self.assertFalse(report['audit_errors'])
        self.assertEqual(report['audited_episode_count'],1)
        self.assertEqual(report['overall']['tactile']['task_successes'],1)
        self.assertEqual(len(report['missing_policy_cases']),161)

    def test_empty_report_is_incomplete_not_success(self):
        with tempfile.TemporaryDirectory() as directory:
            report=collect(directory,make_protocol())
        self.assertEqual(report['status'],'incomplete_or_audit_failed')
        self.assertEqual(len(report['missing_policy_cases']),162)
        self.assertEqual(len(report['missing_fixed_cases']),81)
        self.assertIn('缺少策略案例 162',render_html(report))


if __name__ == '__main__':
    unittest.main()
