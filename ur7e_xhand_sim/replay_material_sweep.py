"""Fixed demonstration joint-command replay; no VLM, IK, or support feedback."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
DEMO=ROOT/'outputs/rule_demos/teacher_pinch_v2_20261024'
CASES={
    'friction_020':{'cube_density':500.,'cube_friction':.2,'robot_friction':.2},
    'friction_050':{'cube_density':500.,'cube_friction':.5,'robot_friction':.5},
    'friction_080':{'cube_density':500.,'cube_friction':.8,'robot_friction':.8},
    'mass_050g':{'cube_mass_kg':.050,'cube_friction':.8,'robot_friction':.8},
    'mass_100g':{'cube_mass_kg':.100,'cube_friction':.8,'robot_friction':.8},
    'mass_200g':{'cube_mass_kg':.200,'cube_friction':.8,'robot_friction':.8},
}
def load(p):return json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def run_case(out,case,sandbox_cls=None):
    import numpy as np
    from scipy.spatial.transform import Rotation
    from environment import Sandbox
    from genesis.utils.misc import qd_to_numpy
    import genesis as gs
    out.mkdir(exist_ok=False)
    ref=load(DEMO/'reference.json')
    assert all(sha(DEMO/name)==h for name,h in ref['files'].items())
    rows=[json.loads(s) for s in (DEMO/'control.jsonl').read_text().splitlines()]
    s=(sandbox_cls or Sandbox)(output_dir=out/'runtime',render=False,scenario='original',**CASES[case])
    assert s.names==ref['arm_joint_order']+ref['hand_joint_order']
    np.testing.assert_allclose(s.values(s.robot.get_dofs_position(s.indices)),
        [rows[0]['before']['joint_positions_rad'][n] for n in s.names],atol=1e-7,rtol=0)
    solver=s.scene.rigid_solver
    expected_mass=s.material_parameters['cube_mass_kg']
    engine_mass=float(s.array(s.cube.get_mass()).reshape(-1)[0])
    assert np.isclose(engine_mass,expected_mass,atol=1e-8,rtol=1e-5)
    assert np.isclose(s.cube.material.rho,s.material_parameters['cube_density_kg_m3'])
    geom_mu=qd_to_numpy(solver.dyn_info.geoms.friction).reshape(-1)
    entity_geoms={name:[g.idx for g in entity.geoms] for name,entity in [('cube',s.cube),('robot',s.robot),('table',s.table)]}
    for name,ids in entity_geoms.items():
        np.testing.assert_allclose(geom_mu[ids],s.material_parameters[name+'_friction'],atol=1e-6,rtol=0)
    verification={'requested':CASES[case],'resolved':s.material_parameters,
        'material_density_kg_m3':float(s.cube.material.rho),'engine_mass_kg':engine_mass,
        'engine_mass_divided_by_volume_kg_m3':engine_mass/.045**3,
        'engine_geom_friction':{name:geom_mu[ids].tolist() for name,ids in entity_geoms.items()},
        'pair_rule':'max(mu_a,mu_b,0.01); robot/cube both set to requested mu; table remains 0.8'}
    write(out/'material_verification.json',verification)
    write(out/'manifest.json',{'case':case,'demonstration':str(DEMO),'reference_sha256':sha(DEMO/'reference.json'),
        'trajectory_sha256':sha(DEMO/'control.jsonl'),'expected_steps':len(rows),'control_dt_s':.025,
        'driver':'fixed recorded sent_target_rad, all 18 joints; original PD only',
        'no_VLM_IK_support_controller_or_tactile_gate':True,'no_object_pose_writes_during_replay':True,
        'scenario':s.scenario,'materials':verification,'joint_order':s.names})
    frame_dir=out/'frames';frame_dir.mkdir()
    def capture(tag):
        s.render_enabled=True;s.render();s.render_enabled=False
        for name in ('scene_front','scene_left','wrist'):(frame_dir/f'{tag}_{name}.jpg').write_bytes(s.image_bytes[name])
    capture('000000')
    values=[];taxels=[];commands=[];actuals=[];fault=None;max_error=0.
    contact_mu={'robot_cube':[],'table_cube':[]};contact_counts={'robot_cube':0,'table_cube':0}
    cube_links={l.idx for l in s.cube.links};robot_links={l.idx for l in s.robot.links};table_links={l.idx for l in s.table.links}
    def audit_contact_mu():
        st=solver.collider.collider_state
        n=int(qd_to_numpy(st.n_contacts).reshape(-1)[0])
        indices=qd_to_numpy(st.contact_sort_idx)[:n,0].astype(int)
        a=qd_to_numpy(st.contact_data.link_a)[indices,0]
        b=qd_to_numpy(st.contact_data.link_b)[indices,0]
        mu=qd_to_numpy(st.contact_data.friction)[indices,0]
        found={key:[] for key in contact_mu}
        for la,lb,m in zip(a,b,mu):
            for key,other in [('robot_cube',robot_links),('table_cube',table_links)]:
                if (la in cube_links and lb in other) or (lb in cube_links and la in other):
                    expected=CASES[case]['cube_friction'] if key=='robot_cube' else .8
                    assert abs(float(m)-expected)<1e-6,(key,m,expected)
                    found[key].append(float(m));contact_mu[key].append(float(m));contact_counts[key]+=1
        return found
    with (out/'control.jsonl').open('w',buffering=1) as control, (out/'evaluation.jsonl').open('w',buffering=1) as evaluation:
        for i,row in enumerate(rows):
            target=np.array(row['sent_target_rad'],dtype=float)
            s.step({'actions':dict(zip(s.names,((target-s.offset)/s.scale).tolist())),'steps':1})
            error=float(np.max(np.abs(s.target-target)));max_error=max(max_error,error)
            assert error<1e-12
            o=s.observe();e=s.evaluate();r=Rotation.from_quat(np.array(e['cube_quaternion_wxyz'])[[1,2,3,0]]).as_matrix()
            e.update(step=s.steps,phase=row['phase'],bottom_clearance_m=float(e['cube_position_m'][2]-.0225*np.abs(r[2]).sum()-.20),
                contact_friction=audit_contact_mu())
            p=Rotation.from_quat(np.array(o['wrist_quaternion_wxyz'])[[1,2,3,0]]).as_matrix()
            e['cube_in_palm_m']=(p.T@(np.array(e['cube_position_m'])-o['wrist_position_m'])).tolist()
            values.append(e)
            control.write(json.dumps({'step':s.steps,'phase':row['phase'],'sent_target_rad':target.tolist(),'observation':o})+'\n')
            evaluation.write(json.dumps(e)+'\n')
            taxels.append(np.stack([s.array(sensor.read().force).reshape(-1,120,3)[0].copy() for sensor in s.sensors.values()]))
            commands.append(s.target.copy());actuals.append(s.values(s.robot.get_dofs_position(s.indices)))
            bad=[x for x in e['solver_contacts'] if (x['entity_a']==x['entity_b']=='ur7e_xhand') or 'table' in (x['entity_a'],x['entity_b'])]
            if bad or s.fault:
                fault=s.fault or 'Robot self/table contact';break
            if s.steps%12==0:capture(f'{s.steps:06d}')
            if i==len(rows)-1 or rows[i+1]['phase']!=row['phase']:
                print(json.dumps({'case':case,'phase':row['phase'],'step':s.steps,'clearance_m':e['bottom_clearance_m'],
                    'tactile_peaks_N':{f:o['tactile'][f]['peak_force_N'] for f in ('thumb','index')}}),flush=True)
    capture(f'{s.steps:06d}')
    final_mass=float(s.array(s.cube.get_mass()).reshape(-1)[0])
    assert np.isclose(final_mass,engine_mass,rtol=0,atol=1e-9)
    assert contact_counts['robot_cube']>0 and contact_counts['table_cube']>0
    verification.update(engine_final_mass_kg=final_mass,
        observed_pair_friction={k:{'min':min(v),'max':max(v),'contact_samples':len(v)} for k,v in contact_mu.items()},
        all_checks_passed=True)
    write(out/'material_verification.json',verification)
    np.savez_compressed(out/'trajectory_tactile.npz',sent_targets_rad=np.array(commands),actual_q_rad=np.array(actuals),
        force_link_N=np.array(taxels,dtype=np.float32),finger_order=np.array(list(s.sensors)),joint_order=np.array(s.names))
    hold=[e for e in values if e['phase']=='hold']
    clearance=[e['bottom_clearance_m'] for e in values]
    relative=np.array([e['cube_in_palm_m'] for e in hold])
    longest=streak=0
    for h in clearance:
        streak=streak+1 if h>=.05 else 0;longest=max(longest,streak)
    score={'case':case,'success':bool(len(hold)==120 and min(e['bottom_clearance_m'] for e in hold)>=.05 and not fault),
        'lifted_above_5cm':max(clearance)>=.05,'fault':fault,'control_steps':len(values),'hold_seconds':len(hold)*.025,
        'hold_min_bottom_clearance_m':min([e['bottom_clearance_m'] for e in hold],default=None),
        'max_bottom_clearance_m':max(clearance),'final_bottom_clearance_m':clearance[-1],
        'longest_continuous_above_5cm_s':longest*.025,
        'hold_cube_relative_displacement_m':float(np.linalg.norm(relative[-1]-relative[0])) if len(relative) else None,
        'max_replay_command_error_rad':max_error,'material_verification_passed':True,
        'actual_mass_g':engine_mass*1000,'density_kg_m3':float(s.cube.material.rho),'contact_mu':CASES[case]['cube_friction'],
        'interpretation':'Open-loop recorded joint targets with unchanged joint PD; tactile recorded, never changes commands or gates lift.'}
    write(out/'score.json',score);print('SCORE '+json.dumps(score),flush=True)
    gs.destroy()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--case',choices=list(CASES));args=parser.parse_args()
    if args.case:run_case(args.output,args.case);return
    args.output.mkdir(parents=True,exist_ok=False)
    paths=[Path(__file__),ROOT/'ur7e_xhand_sim/environment.py',ROOT/'Genesis/genesis/engine/sensors/kinematic_tactile.py']
    write(args.output/'protocol.json',{'cases':CASES,'scenario':'original; no distractors','demonstration':str(DEMO),
        'reference_sha256':sha(DEMO/'reference.json'),'trajectory_sha256':sha(DEMO/'control.jsonl'),
        'one_attempt_per_case':True,'control':'Exact 1275 stored joint targets, .025 seconds each; no adaptive policy',
        'success':'Rotated cube bottom >=5cm above table for all final 120 hold samples (3s)',
        'source_sha256':{str(p):sha(p) for p in paths}})
    snapshot=args.output/'source_snapshot';snapshot.mkdir()
    for p in paths:(snapshot/p.name).write_bytes(p.read_bytes())
    results=[]
    for case in CASES:
        print('START '+case,flush=True)
        with (args.output/(case+'.log')).open('w') as log:
            result=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--output',str((args.output/case).resolve()),'--case',case],stdout=log,stderr=subprocess.STDOUT,cwd=ROOT)
        score=args.output/case/'score.json'
        results.append(load(score) if score.exists() else {'case':case,'execution_error':True,'returncode':result.returncode})
        write(args.output/'results.json',results);print(json.dumps(results[-1]),flush=True)
    if any(x.get('execution_error') for x in results):raise SystemExit(1)

if __name__=='__main__':main()
