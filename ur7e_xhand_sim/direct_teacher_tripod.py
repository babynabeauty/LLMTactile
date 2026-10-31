"""Privileged three-finger source demonstration; middle PD remains nominal.

All three sensor-bearing links must physically contact before lift. The common
residual gain contract is unchanged: only thumb/index six joints are scaled.
"""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
from build_assets import HERE
from direct_teacher import DirectTeacher,sha,write
from direct_teacher_curved import run_curved_teacher
from environment import Sandbox


def validate_tripod_candidate(candidate):
    if candidate.get('contact_fingers')!=['thumb','index','middle']:
        raise ValueError('Tripod teacher requires thumb/index/middle contact order')
    if candidate.get('finger_joint_indices')!=[[0,1,2],[3,4,5],[6,7]]:
        raise ValueError('Tripod teacher finger DOF mapping mismatch')
    if candidate.get('object_spec',{}).get('shape') not in ('sphere','cylinder','cuboid'):
        raise ValueError('This teacher validates sphere, cylinder or cuboid candidates')
    hands=np.asarray(candidate['path_hand_rad'],float)
    knots=np.asarray(candidate['path_inward_m'],float)
    if hands.shape!=(len(knots),12) or not np.isfinite(hands).all() or not np.all(np.diff(knots)>0):
        raise ValueError('Finite ordered 12-DOF path required')
    candidate['gain_scope']='Only thumb/index six joints use common grip_gain; middle remains nominal kp/kv gain1'
    candidate['middle_gain']=1.
    candidate['closure_contact_gate']='All three sensor-bearing links have physical load>=0.1N for3 action frames'
    return candidate


def run_tripod_teacher(teacher):
    validate_tripod_candidate(teacher.candidate)
    run_curved_teacher(teacher)
    s=teacher.sim
    kp=s.array(s.robot.get_dofs_kp(s.indices)).reshape(-1)
    kv=s.array(s.robot.get_dofs_kv(s.indices)).reshape(-1)
    np.testing.assert_allclose(kp[12:14],np.asarray(s.kp)[12:14],rtol=1e-6)
    np.testing.assert_allclose(kv[12:14],np.asarray(s.kv)[12:14],rtol=1e-6)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate-json',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--noslip-iterations',type=int,choices=[0,5],default=0,
        help='Explicit source solver profile; default0 preserves the original study')
    args=parser.parse_args()
    candidate=validate_tripod_candidate(json.loads(args.candidate_json.read_text()))
    sim=Sandbox(output_dir=args.output.parent/'runtime_tripod',render=False,
        cube_mass_kg=.05,cube_friction=.8,robot_friction=.8,
        object_spec=candidate['object_spec'],physics_dt_s=1/300,control_substeps=5,
        noslip_iterations=args.noslip_iterations)
    teacher=DirectTeacher(sim,args.output,candidate)
    sources=[Path(__file__),HERE/'direct_teacher_curved.py',HERE/'direct_teacher_geometry.py',
        HERE/'direct_teacher_tripod_geometry.py',Path('/home/sai/zsq/vlm-policy/urdf_frames.py')]
    for source in sources:
        teacher.manifest['source_sha256'][str(source.resolve())]=sha(source)
        target=teacher.out/'source_snapshot'/'tripod_teacher'/source.name
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
    write(teacher.out/'manifest.json',teacher.manifest)
    score=teacher.run(run_tripod_teacher);print(json.dumps(score),flush=True)
    raise SystemExit(0 if score['success'] else 1)


if __name__=='__main__':main()
