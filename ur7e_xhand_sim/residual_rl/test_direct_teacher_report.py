import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from ur7e_xhand_sim.direct_teacher_report import (sha,source_series,verification_record,
    held_frames,build_report,attempt_records,tactile_coverage,FINGERS,
    audit_report_solver_profile,declared_solver_profile)


class DirectTeacherReportTests(unittest.TestCase):
    def test_legacy_profile_is_preserved_without_relabeling(self):
        self.assertIsNone(declared_solver_profile({}))
        audit=audit_report_solver_profile({},{},{},{},{})
        self.assertFalse(audit['dataset_declared'])
        self.assertIsNone(audit['solver_profile'])

    def test_explicit_solver_profile_requires_same_frozen_source_and_replay(self):
        profile=dict(schema='genesis_rigid_solver_profile_v1',noslip_iterations=5,noslip_tolerance=1e-6)
        record=dict(solver_profile=profile)
        readback=dict(options_noslip_iterations=5,engine_noslip_iterations=5,
            options_noslip_tolerance=1e-6,engine_noslip_tolerance=float(np.float32(1e-6)))
        scene=dict(record,solver_readback=readback)
        metadata=dict(record,source_solver_readback=readback)
        verification=dict(record,solver_readback=readback)
        audit=audit_report_solver_profile(record,metadata,verification,record,scene)
        self.assertEqual(audit['solver_profile'],profile)
        legacy=dict(solver_profile=dict(profile,noslip_iterations=0))
        for wrong in (legacy,{}):
            with self.assertRaisesRegex(ValueError,'profile mismatch'):
                audit_report_solver_profile(record,metadata,wrong,record,scene)
        with self.assertRaisesRegex(ValueError,'explicit dataset'):
            audit_report_solver_profile({},metadata,verification,record,scene)
        with self.assertRaisesRegex(ValueError,'engine_noslip_iterations'):
            audit_report_solver_profile(record,metadata,verification,record,
                dict(record,solver_readback=dict(readback,engine_noslip_iterations=0)))
        with self.assertRaisesRegex(ValueError,'engine_noslip_iterations'):
            audit_report_solver_profile(record,metadata,
                dict(record,solver_readback=dict(readback,engine_noslip_iterations=0)),record,scene)

    def test_pending_new_dataset_names_its_explicit_profile(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'noslip5_v2';root.mkdir()
            profile=dict(schema='genesis_rigid_solver_profile_v1',noslip_iterations=5,noslip_tolerance=1e-6)
            (root/'dataset_manifest.json').write_text(json.dumps(dict(status='collecting',
                objects=[dict(object_id='sphere',diameter_m=.045)],verified={},solver_profile=profile,
                previous_dataset=str(root.parent/'legacy_v1'))))
            report=build_report(root)
            self.assertEqual(report['solver_profile'],profile)
            self.assertEqual(report['cards'][0]['status'],'pending')
            document=(root/'report/index.html').read_text()
            self.assertIn('noslip_iterations=5',document)
            self.assertIn('noslip5_v2',document)
            self.assertIn('没有迁入或重新标记',document)

    def test_sparse_taxel_response_is_separate_from_real_contact_force(self):
        times=np.arange(1,5)/60
        signals=np.zeros((4,5,120));signals[:,0,0]=.01;signals[:,0,1]=.02
        rows=[{'solver_contacts':[dict(entity_a='cube',entity_b='robot',link_a='object',
            link_b='thumb_sensor',force_on_a_world_N=[3.,0.,0.])]} for _ in times]
        series=dict(time=times,hold=np.ones(4,dtype=bool),taxel_norm=signals,evaluation_records=rows)
        scene={'tactile_layout':[{'finger':finger,'link_name':finger+'_sensor'} for finger in FINGERS]}
        report=tactile_coverage(series,scene);thumb=report['fingers'][0]
        self.assertEqual(thumb['physical_sensor_link_mean_contact_force_N'],3.)
        self.assertEqual(thumb['active_taxels_mean'],2.)
        self.assertAlmostEqual(thumb['taxel_norm_sum_mean_uncalibrated_nominal_N'],.03)
        self.assertTrue(thumb['sparse_response_flag'])
        self.assertFalse(report['fingers'][1]['sparse_response_flag'])
        self.assertEqual(thumb['physical_contact_samples'],4)
        self.assertIn('not physical force',report['signal_measurement'])

    def test_external_curved_source_keeps_its_sibling_search_attempts(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'dataset';root.mkdir()
            external=Path(directory)/'curved';external.mkdir()
            for name,shape,success in [('sphere_attempt01','sphere',False),
                    ('sphere_attempt02','sphere',True),('sphere_tripod_attempt01','sphere',False),
                    ('sphere_attempt_other','cylinder',True)]:
                run=external/name;run.mkdir()
                (run/'manifest.json').write_text(json.dumps(dict(driver='direct_teacher_search',
                    object_spec={'shape':shape})))
                (run/'score.json').write_text(json.dumps({'success':success}))
            local=root/'sphere'/'reference_attempt01';local.mkdir(parents=True)
            (local/'manifest.json').write_text(json.dumps(dict(driver='direct_teacher_search',
                object_spec={'shape':'sphere'})))
            (local/'score.json').write_text(json.dumps({'success':False}))
            rows=attempt_records(root,'sphere',external/'sphere_attempt02')
            self.assertEqual({row['name'] for row in rows},{'sphere_attempt01','sphere_attempt02',
                'sphere_tripod_attempt01','reference_attempt01'})
            self.assertEqual(sum(row['source_success'] for row in rows),1)
            pending=attempt_records(root,'sphere',collection_roots=[external])
            self.assertEqual({row['path'] for row in pending},{row['path'] for row in rows})

    def test_raw_pinky_alias_and_clock_alignment(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);times=np.arange(1,5)/60
            rows=[dict(sample_time_s=float(t),bottom_clearance_m=.06,
                all_robot_contact_load_N=8.,all_robot_normal_load_N=7.,phase='hold') for t in times]
            (root/'evaluation.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            tactile=np.zeros((4,5,120,3));tactile[:,1,3,0]=.2
            np.savez(root/'raw_tactile.npz',sample_time_s=times,tactile=tactile,grip_gain=np.ones(4))
            manifest=dict(sensor_finger_order=['thumb','index','middle','ring','pinky'])
            values=source_series(root,manifest)
            self.assertAlmostEqual(values['tactile_peak'][0,1],.2)
            self.assertEqual(values['contact'][0],8.)
            np.savez(root/'raw_tactile.npz',sample_time_s=times+.01,tactile=tactile,grip_gain=np.ones(4))
            with self.assertRaisesRegex(ValueError,'clocks differ'):source_series(root,manifest)

    def test_declaration_without_verified_artifacts_is_not_success(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'dataset_manifest.json').write_text(json.dumps(dict(status='collecting',
                objects=[dict(object_id='sphere',diameter_m=.045)],verified={'sphere':{'bind_record':'missing.json'}})))
            report=build_report(root)
            card=report['cards'][0]
            self.assertEqual(card['status'],'evidence_error')
            self.assertEqual(card['verified_replays'],0)
            self.assertFalse(card['images'])

    def test_verified_json_hash_and_success_are_both_required(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);trajectory=root/'control.jsonl';trajectory.write_text('fixture\n')
            source=root/'source';source.mkdir()
            metadata=root/'metadata.json';metadata.write_text(json.dumps(dict(
                demonstration_source='direct_teacher_search',object_id='sphere',source_run=str(source),
                trajectory_sha256=sha(trajectory))))
            verification=root/'verification.json';verification.write_text(json.dumps(dict(
                accepted=True,audit_errors=[],trace_recomputed={'task_success':True})))
            evidence=dict(demonstration_source='direct_teacher_search',privileged_teacher=True,task_success=True,
                verification_path=str(verification),verification_sha256=sha(verification),
                trajectory_metadata_path=str(metadata),trajectory_metadata_sha256=sha(metadata))
            bind=root/'bind.json';bind.write_text(json.dumps(dict(object_id='sphere',trajectory_path=str(trajectory),
                trajectory_sha256=sha(trajectory),demonstration_evidence=evidence)))
            self.assertEqual(verification_record(root,{'bind_record':str(bind)})['source'],source)
            verification.write_text(json.dumps(dict(accepted=False,trace_recomputed={'task_success':False})))
            with self.assertRaisesRegex(ValueError,'hash mismatch'):
                verification_record(root,{'bind_record':str(bind)})
            record=json.loads(bind.read_text());record['demonstration_evidence']['verification_sha256']=sha(verification)
            bind.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError,'did not pass'):
                verification_record(root,{'bind_record':str(bind)})

    def test_held_images_must_exist_and_keep_actual_timestamp(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'frames').mkdir()
            (root/'stages.json').write_text(json.dumps([dict(name='held',step=240)]))
            (root/'frames/000240_scene_front.jpg').write_bytes(b'test fixture, not rendered')
            (root/'frames/000210_scene_left.jpg').write_bytes(b'test fixture, not rendered')
            images=held_frames(root)
            self.assertEqual(len(images),2)
            self.assertTrue(images[0]['is_exact_held_frame'])
            self.assertFalse(images[1]['is_exact_held_frame'])
            self.assertEqual(images[1]['sample_time_s'],3.5)


if __name__=='__main__':unittest.main()
