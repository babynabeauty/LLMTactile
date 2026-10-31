"""Isolated solver diagnostic, never a demonstration or canonical training source.

Replay exact saved joint targets/gains; temporarily override only RigidOptions
noslip_iterations during construction, restore its factory before stepping.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import shutil
import traceback

import numpy as np
import genesis as gs
from environment import Sandbox
from direct_teacher import sha, write


def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value,dtype='<f8').tobytes()).hexdigest()


@contextmanager
def solver_override(iterations):
    if iterations not in (0,5,10):raise ValueError('Diagnostic accepts only 0,5,10 iterations')
    original=gs.options.RigidOptions
    calls=[]
    def factory(*args,**kwargs):
        if args or 'noslip_iterations' in kwargs:
            raise ValueError('Unexpected construction prevents isolated override')
        result=original(**kwargs,noslip_iterations=iterations)
        calls.append(dict(original_kwargs=kwargs,override={'noslip_iterations':iterations}))
        return result
    gs.options.RigidOptions=factory
    try:yield calls
    finally:gs.options.RigidOptions=original


def set_gain(s,gain):
    kp=np.asarray(s.kp).copy();kv=np.asarray(s.kv).copy();kp[6:12]*=gain;kv[6:12]*=np.sqrt(gain)
    s.robot.set_dofs_kp(kp,s.indices);s.robot.set_dofs_kv(kv,s.indices)
    akp=s.array(s.robot.get_dofs_kp(s.indices)).reshape(-1);akv=s.array(s.robot.get_dofs_kv(s.indices)).reshape(-1)
    lo,hi=s.robot.get_dofs_force_range(s.indices);lo=s.array(lo).reshape(-1);hi=s.array(hi).reshape(-1)
    np.testing.assert_allclose(akp,kp,rtol=1e-6);np.testing.assert_allclose(akv,kv,rtol=1e-6)
    np.testing.assert_allclose(lo,-s.effort,rtol=1e-6);np.testing.assert_allclose(hi,s.effort,rtol=1e-6)
    return dict(gain=float(gain),kp_readback=akp.tolist(),kv_readback=akv.tolist(),effort_lower=lo.tolist(),effort_upper=hi.tolist())


def run(source,output,iterations):
    output.mkdir(parents=True,exist_ok=False)
    source_rows=[json.loads(line) for line in (source/'control.jsonl').read_text().splitlines()]
    source_eval=[json.loads(line) for line in (source/'evaluation.jsonl').read_text().splitlines()]
    source_manifest=json.loads((source/'manifest.json').read_text());source_scene=json.loads((source/'object_scene.json').read_text())
    targets=np.array([r['sent_target_rad'] for r in source_rows]);gains=np.array([r['grip_gain'] for r in source_rows])
    if len(source_rows)!=len(source_eval) or len(source_rows)%4:raise ValueError('Aligned complete 60Hz records required')
    if not np.isfinite(np.r_[targets.ravel(),gains]).all():raise ValueError('Nonfinite source')
    for i in range(0,len(source_rows),4):
        np.testing.assert_array_equal(targets[i:i+4],np.repeat(targets[i:i+1],4,axis=0))
        np.testing.assert_array_equal(gains[i:i+4],np.repeat(gains[i:i+1],4,axis=0))
    baseline_options=gs.options.RigidOptions(enable_self_collision=True,enable_joint_limit=True,enable_neutral_collision=True)
    manifest=dict(schema='isolated_noslip_replay_diagnostic_v1',diagnostic_only=True,canonical_dataset_eligible=False,
        description='Frozen failed source replay with isolated solver override; no IK, feedback control, new trajectory, or object writes.',
        source_run=str(source.resolve()),source_files_sha256={name:sha(source/name) for name in ['control.jsonl','evaluation.jsonl','manifest.json','object_scene.json','raw_tactile.npz']},
        source_target_float64_sha256=array_sha(targets),source_gain_float64_sha256=array_sha(gains),
        target_update_hz=15,tactile_hz=60,physics_hz=300,sample_count=len(source_rows),
        solver_override={'noslip_iterations':iterations},baseline_options=baseline_options.model_dump(mode='json'),
        code_sha256={str(Path(__file__).resolve()):sha(__file__),str(Path('ur7e_xhand_sim/environment.py').resolve()):sha('ur7e_xhand_sim/environment.py')},
        source_scene_unchanged=source_scene,production_defaults_unchanged=True)
    write(output/'manifest.json',manifest);shutil.copy2(__file__,output/'replay_noslip.py')
    write(output/'status.json',{'status':'constructing','completed_samples':0})
    material=source_scene['materials'];s=None;records=[];raw=[];sent=[];used_gains=[];pd=[];failure=None
    try:
        original_factory=gs.options.RigidOptions
        with solver_override(iterations) as factory_calls:
            s=Sandbox(output_dir=output/'runtime',render=False,object_spec=source_manifest['parameters']['object_spec'],
                cube_mass_kg=float(material['cube_mass_kg']),cube_friction=float(material['cube_friction']),
                robot_friction=float(material['robot_friction']),physics_dt_s=1/300,control_substeps=5)
        assert gs.options.RigidOptions is original_factory
        if len(factory_calls)!=1:raise ValueError('Expected exactly one overridden options construction')
        actual=s.scene.rigid_solver._options
        if actual.noslip_iterations!=iterations:raise ValueError('Solver override failed readback')
        manifest.update(factory_calls=factory_calls,factory_restored=True,
            actual_solver_options=actual.model_dump(mode='json'),actual_engine_noslip_iterations=int(s.scene.rigid_solver.rigid_info.noslip_iterations[None]))
        if manifest['actual_engine_noslip_iterations']!=iterations:raise ValueError('Device noslip iteration readback mismatch')
        resolved_baseline=baseline_options.model_dump(mode='json');resolved_actual=actual.model_dump(mode='json')
        # Runtime resolves several baseline None values. Record every difference;
        # only noslip is explicitly changed by construction.
        manifest['resolved_option_differences']={k:{'construction_baseline':v,'actual_resolved':resolved_actual[k]} for k,v in resolved_baseline.items() if resolved_actual[k]!=v}
        manifest['actual_object_scene']=s.describe();manifest['robot_model_sha256']=sha('ur7e_xhand_sim/generated/ur7e_xhand.urdf')
        write(output/'manifest.json',manifest)
        s.reset();gain=None
        with (output/'evaluation.jsonl').open('w',buffering=1) as ew,(output/'control.jsonl').open('w',buffering=1) as cw:
            for i,row in enumerate(source_rows):
                if gain!=gains[i]:
                    gain=float(gains[i]);readback=set_gain(s,gain);readback['before_source_row']=i;pd.append(readback)
                target=targets[i];payload={'actions':dict(zip(s.names,((target-s.offset)/s.scale).tolist())),'steps':1}
                s.step(payload);obs=s.observe(full=True);e=s.evaluate();e.update(step=s.steps,sample_time_s=s.steps/60,phase=row['phase'])
                np.testing.assert_allclose(s.target,target,rtol=0,atol=2e-15)
                if not np.isfinite(np.r_[e['object_position_m'],e['object_quaternion_wxyz'],e['bottom_clearance_m']]).all():raise ValueError('Nonfinite physics')
                records.append(e);sent.append(target.copy());used_gains.append(gain)
                raw.append(np.array([obs['tactile'][f]['force_sensor_N'] for f in s.sensors],np.float32))
                ew.write(json.dumps(e,allow_nan=False)+'\n')
                cw.write(json.dumps(dict(source_row=i,phase=row['phase'],sent_target_rad=target.tolist(),grip_gain=gain,
                    actual_joint_rad=[obs['joint_positions_rad'][n] for n in s.names],actual_wrist_position_m=obs['wrist_position_m']),allow_nan=False)+'\n')
                bad=[c for c in e['solver_contacts'] if 'ur7e_xhand' in [c['entity_a'],c['entity_b']] and ('table' in [c['entity_a'],c['entity_b']] or c['entity_a']==c['entity_b'])]
                if bad:raise ValueError('Robot self/table collision in diagnostic replay')
                if (i+1)%120==0:write(output/'status.json',dict(status='running',completed_samples=i+1,phase=row['phase'],bottom_clearance_m=e['bottom_clearance_m']))
    except Exception as exc:
        failure=f'{type(exc).__name__}: {exc}';(output/'error.txt').write_text(traceback.format_exc())
    finally:
        if raw:np.savez_compressed(output/'raw_tactile.npz',tactile=np.asarray(raw),sent_target_rad=np.asarray(sent),grip_gain=np.asarray(used_gains))
        write(output/'pd_readback.json',pd)
        evaluation=dict(diagnostic_only=True,canonical_dataset_eligible=False,noslip_iterations=iterations,
            completed_samples=len(records),requested_samples=len(source_rows),failure=failure,
            max_bottom_clearance_m=max((r['bottom_clearance_m'] for r in records),default=None),
            final_bottom_clearance_m=records[-1]['bottom_clearance_m'] if records else None,
            original_max_bottom_clearance_m=max(e['bottom_clearance_m'] for e in source_eval),
            original_final_bottom_clearance_m=source_eval[-1]['bottom_clearance_m'],
            exact_full_source_targets_and_gain_replayed=bool(len(sent)==len(targets) and array_sha(sent)==array_sha(targets) and array_sha(used_gains)==array_sha(gains)),
            replay_target_float64_sha256=array_sha(sent),replay_gain_float64_sha256=array_sha(used_gains))
        write(output/'result.json',evaluation);write(output/'status.json',dict(status='failed' if failure else 'completed',**evaluation))
        if s is not None:gs.destroy()
    print(json.dumps(evaluation),flush=True)
    return evaluation


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--noslip',type=int,choices=[0,5,10],required=True)
    args=parser.parse_args();result=run(args.source,args.output,args.noslip)
    raise SystemExit(1 if result['failure'] else 0)

if __name__=='__main__':main()
