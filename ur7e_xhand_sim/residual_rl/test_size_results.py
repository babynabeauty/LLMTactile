"""Independent size-report metrics, provenance and incomplete accounting checks."""
import json
from pathlib import Path
import tempfile
import subprocess
import sys
import unittest

import numpy as np

from .size_results import audit, digest, make_html, paired, trace_metrics


def synthetic_trace(size=30, force=10., dropped=False):
    n=240;t=np.arange(1,n+1)/60
    load=np.zeros((n,1,5));load[:,:,:2]=force/2
    clearance=np.full((n,1),.06)
    if dropped:
        clearance[n//2:]=.001;load[n//2:]=0.
    pos=np.zeros((n,1,3));pos[:,:,2]=.2+size/2000+clearance
    quat=np.zeros((n,1,4));quat[:,:,0]=1
    return dict(sample_time_s=t,clearance_m=clearance,contact_load_N=load,
        all_robot_contact_load_N=load.sum(-1),fault=np.zeros((n,1),bool),
        contact_friction_error=np.zeros((n,1)),cube_position_m=pos,cube_quaternion_wxyz=quat,
        actions=np.zeros((n//4,1,1)),applied_gains=np.ones((n//4,1,2)),
        prefix_state=np.zeros((1,12,79),np.float32),prefix_tactile=np.zeros((1,12,5,120,3),np.float32))


def write_fixture(root, sizes=(30,35), seeds=(1,)):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    cs=[f'seed{s}_{kind}' for s in seeds for kind in ('tactile','no_tactile')]
    models={}
    for c in cs:
        f=root/(c+'.synthetic');f.write_bytes(b'SYNTHETIC; NOT A MODEL\n'+c.encode())
        models[c]={'path':str(f),'sha256':digest(f)}
    trajectory=root/'synthetic_control.jsonl';trajectory.write_text('SYNTHETIC; NOT A TRAJECTORY\n')
    protocol={'synthetic':True,'models':models,'controllers':cs,'sizes_mm':list(sizes),
              'mass_kg':.125,'friction':.5,'trajectory':{'path':str(trajectory),'sha256':digest(trajectory)},
              'size_runs':[{'size_mm':s,'output':f'size_{s:03d}mm/evaluation'} for s in sizes]}
    (root/'protocol.json').write_text(json.dumps(protocol));return protocol


def write_episode(root, size, controller, force=10., dropped=False, prefix_offset=0.):
    root=Path(root);p=json.loads((root/'protocol.json').read_text());base=root/f'size_{size:03d}mm/evaluation'
    out=base/controller;out.mkdir(parents=True,exist_ok=True);(base/'environment').mkdir(exist_ok=True)
    manifest={'models':p['models'],'training':False,'conditions_kg_mu':[[.125,.5]],
              'trajectory':{'sha256':p['trajectory']['sha256']},
              'checkpoint_metadata':{c:{'training_steps':786432,'uses_tactile':'no_tactile' not in c} for c in p['controllers']}}
    (base/'manifest.json').write_text(json.dumps(manifest))
    (base/'status.json').write_text(json.dumps({'status':'completed'}))
    side=size/1000;initial=[-.6,-.1333,.2+side/2]
    env=dict(hold_interval_s=[2,5],start_time_s=1,end_physics_time_s=5,cube_size_m=side,
        cube_size_readback_m=[side]*3,table_top_z_m=.2,mass_kg_per_lane=[.125],friction_per_lane=[.5],
        mass_readback_kg_per_lane=[.125],density_kg_m3_per_lane=[.125/side**3],
        density_readback_kg_m3_per_lane=[.125/side**3],
        inertia_readback_kg_m2_per_lane=(.125*side**2/6*np.eye(3))[None].tolist(),
        cube_initial_position_m=initial,cube_initial_position_readback_m=[initial],
        cube_initial_bottom_readback_m=[.2],
        prefix_physics_audit={'enabled':True,'samples':60,'collision_per_lane':[False]})
    (base/'environment/environment_protocol.json').write_text(json.dumps(env))
    pre=synthetic_trace(size,force=0)
    pre={k:v[:60].copy() for k,v in pre.items() if k in ('sample_time_s','cube_position_m','cube_quaternion_wxyz',
         'contact_load_N','all_robot_contact_load_N','fault','clearance_m','contact_friction_error')}
    pre['cube_position_m'][:]=np.asarray(initial)[None,None];pre['clearance_m'][:]=0
    np.savez_compressed(base/'environment/prefix_physics_trace.npz',**pre)
    a=synthetic_trace(size,force,dropped);a['prefix_state']+=prefix_offset;a['sample_time_s']+=1
    m=trace_metrics(a,size,[2,5],1,5)
    ep={'task_success':m['task_success'],'strict_success':m['strict_success'],
        'mean_contact_force_N':m['sensor_policy_mean_N'],'hold_min_clearance_m':m['hold_min_clearance_m'],
        'hold_duration_s':m['hold_duration_s'],'mass_kg':.125,'friction':.5}
    (out/'evaluation.json').write_text(json.dumps({'controller':controller,'episodes':[ep]}));np.savez_compressed(out/'trace.npz',**a)
    return out


class TestSizeResults(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='rigid_size_SYNTHETIC_');self.root=Path(self.tmp.name)
        self.protocol=write_fixture(self.root)

    def tearDown(self):self.tmp.cleanup()

    def test_load_denominators_keep_dropped_zeros_and_contact_duration(self):
        m=trace_metrics(synthetic_trace(force=10,dropped=True),30,[1,4],0,4)
        self.assertFalse(m['task_success']);self.assertFalse(m['strict_success'])
        self.assertAlmostEqual(m['sensor_policy_mean_N'],5)
        self.assertAlmostEqual(m['sensor_hold_mean_N'],10/3)
        self.assertAlmostEqual(m['sensor_contact_mean_N'],10)
        self.assertAlmostEqual(m['sensor_contact_duration_s'],2)
        self.assertAlmostEqual(m['sensor_hold_contact_duration_s'],1)

    def test_zero_contact_mean_is_unknown_not_zero(self):
        m=trace_metrics(synthetic_trace(force=0),30,[1,4],0,4)
        self.assertTrue(m['task_success']);self.assertFalse(m['strict_success'])
        self.assertIsNone(m['sensor_contact_mean_N']);self.assertEqual(m['sensor_policy_mean_N'],0)

    def test_friction_size_allrobot_and_gain_errors_rejected(self):
        a=synthetic_trace();a['contact_friction_error'][0]=.01
        with self.assertRaisesRegex(ValueError,'friction'):trace_metrics(a,30,[1,4],0,4)
        with self.assertRaisesRegex(ValueError,'Clearance'):trace_metrics(synthetic_trace(),35,[1,4],0,4)
        a=synthetic_trace();a['all_robot_contact_load_N']-=1
        with self.assertRaisesRegex(ValueError,'subset'):trace_metrics(a,30,[1,4],0,4)
        a=synthetic_trace();a['applied_gains'][0]=2
        with self.assertRaisesRegex(ValueError,'rate limit'):trace_metrics(a,30,[1,4],0,4)

    def test_missing_and_running_not_failed(self):
        r=audit(self.root);self.assertEqual(r['counts']['pending'],4);self.assertEqual(r['counts']['task'],0)
        b=self.root/'size_030mm/evaluation';b.mkdir(parents=True)
        (b/'status.json').write_text('{"status":"running"}')
        r=audit(self.root);self.assertEqual(r['counts']['running'],2);self.assertEqual(r['counts']['valid_completed'],0)

    def test_valid_success_and_failed_rows_and_common_success_only(self):
        for s,tf,drop in [(30,2.,False),(35,0.,True)]:
            write_episode(self.root,s,'seed1_tactile',tf,drop)
            write_episode(self.root,s,'seed1_no_tactile',4.)
        r=audit(self.root);self.assertEqual(r['counts']['valid_completed'],4);self.assertEqual(r['counts']['task'],3)
        p=r['paired_task_success'];self.assertEqual(p['aggregate']['pairs'],1)
        self.assertEqual(p['aggregate']['conditions'],[[30,1]])
        self.assertEqual(p['aggregate']['loads']['sensor_policy_mean_N']['tactile'],2.)
        self.assertEqual(p['aggregate']['loads']['sensor_policy_mean_N']['no_tactile'],4.)

    def test_prefix_and_label_mismatch_never_promoted(self):
        write_episode(self.root,30,'seed1_tactile')
        write_episode(self.root,30,'seed1_no_tactile',prefix_offset=.01)
        r=audit(self.root);self.assertEqual(r['counts']['invalid'],1)
        out=write_episode(self.root,35,'seed1_tactile');e=json.loads((out/'evaluation.json').read_text())
        e['episodes'][0]['task_success']=False;(out/'evaluation.json').write_text(json.dumps(e))
        r=audit(self.root);self.assertEqual(r['counts']['invalid'],2)

    def test_hash_mismatch_blocks_validity(self):
        write_episode(self.root,30,'seed1_tactile')
        Path(self.protocol['models']['seed1_tactile']['path']).write_bytes(b'altered')
        r=audit(self.root);self.assertFalse(r['input_integrity_passed']);self.assertEqual(r['counts']['valid_completed'],0)

    def test_numeric_readbacks_verified_independently(self):
        out=write_episode(self.root,30,'seed1_tactile')
        path=out.parent/'environment/environment_protocol.json';env=json.loads(path.read_text())
        for key,value in [('mass_readback_kg_per_lane',[.2]),('cube_size_readback_m',[.045]*3),
                          ('inertia_readback_kg_m2_per_lane',np.ones((1,3,3)).tolist()),
                          ('density_readback_kg_m3_per_lane',[1.]),('cube_initial_bottom_readback_m',[.21])]:
            changed={**env,key:value,'mass_and_inertia_readback_verified':True}
            path.write_text(json.dumps(changed));r=audit(self.root)
            self.assertEqual(r['counts']['invalid'],1,key)

    def test_prefix_collision_does_not_change_task_label(self):
        out=write_episode(self.root,30,'seed1_tactile');base=out.parent/'environment'
        path=base/'prefix_physics_trace.npz'
        with np.load(path) as loaded:a=dict(loaded)
        a['fault'][4:9]=True;a['cube_position_m'][30:,:,0]+=.002
        np.savez_compressed(path,**a)
        env=json.loads((base/'environment_protocol.json').read_text())
        env['prefix_physics_audit']['collision_per_lane']=[True]
        (base/'environment_protocol.json').write_text(json.dumps(env))
        r=audit(self.root);self.assertEqual(r['counts']['task'],1);self.assertEqual(r['counts']['invalid'],0)
        pre=r['prefix_by_size']['30'];self.assertEqual(pre['collision_sample_count'],5)
        self.assertAlmostEqual(pre['cube_max_xy_displacement_m'],.002)
        self.assertTrue(r['warnings']);self.assertIn('前缀碰撞独立展示',make_html(r,b'png',self.root))

    def test_prefix_declared_collision_mismatch_invalid(self):
        out=write_episode(self.root,30,'seed1_tactile');path=out.parent/'environment/prefix_physics_trace.npz'
        with np.load(path) as loaded:a=dict(loaded)
        a['fault'][4]=True;np.savez_compressed(path,**a)
        r=audit(self.root);self.assertEqual(r['counts']['invalid'],1)

    def test_descriptive_all_and_successful_remain_separate(self):
        write_episode(self.root,30,'seed1_tactile',2.)
        write_episode(self.root,35,'seed1_tactile',2.,True)
        g=audit(self.root)['descriptive_loads_by_controller']['seed1_tactile']
        self.assertEqual(g['all_valid_episodes']['episodes'],2)
        self.assertEqual(g['all_valid_episodes']['means']['sensor_policy_mean_N']['value'],1.5)
        self.assertEqual(g['task_successful_episodes']['episodes'],1)
        self.assertEqual(g['task_successful_episodes']['means']['sensor_policy_mean_N']['value'],2.)

    def test_frozen_auxiliary_input_hash_mismatch(self):
        path=self.root/'conditions.synthetic';path.write_text('[[0.125,0.5]]')
        p=json.loads((self.root/'protocol.json').read_text());p['frozen_input_hashes']={str(path):digest(path)}
        (self.root/'protocol.json').write_text(json.dumps(p));path.write_text('[[0.2,0.5]]')
        self.assertFalse(audit(self.root)['input_integrity_passed'])

    def test_final_cli_incomplete_exits_nonzero_after_report(self):
        result=subprocess.run([sys.executable,'-m','ur7e_xhand_sim.residual_rl.size_results',
                               '--root',str(self.root),'--final'],capture_output=True,text=True)
        self.assertEqual(result.returncode,1,result.stderr)
        self.assertTrue((self.root/'results.png').is_file())
        self.assertEqual(json.loads((self.root/'size_audit.json').read_text())['counts']['valid_completed'],0)

    def test_36_scope_and_empty_common_success(self):
        write_fixture(self.root,(30,35,40,50,55,60),(1,2,3))
        r=audit(self.root);self.assertEqual(r['counts']['planned'],36)
        self.assertEqual(r['paired_task_success']['aggregate']['pairs'],0)
        self.assertIsNone(r['paired_task_success']['aggregate']['loads']['sensor_policy_mean_N']['tactile'])

    def test_selfcontained_html_escapes_raw_errors(self):
        r=audit(self.root);r['rows'][0]['errors']=['<script>evil()</script>']
        page=make_html(r,b'synthetic-png',self.root)
        self.assertIn('SYNTHETIC',page);self.assertIn('data:image/png;base64,',page)
        self.assertNotIn('<script>evil()',page);self.assertIn('&lt;script&gt;',page)


if __name__=='__main__':unittest.main()
