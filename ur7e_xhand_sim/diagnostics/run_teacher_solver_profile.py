"""Explicit alternate-solver full teacher validation, excluded from the dataset.

Construction changes only noslip_iterations=5; production factories/defaults are
restored before execution. This command must be separately authorized to run.
"""
import argparse
import json
from pathlib import Path
import shutil
import genesis as gs
from build_assets import HERE
from direct_teacher import DirectTeacher,sha,write
from direct_teacher_tripod import run_tripod_teacher,validate_tripod_candidate
from diagnostics.replay_noslip import solver_override
from environment import Sandbox


class ProfileDiagnosticTeacher(DirectTeacher):
    def finalize(self):
        result=super().finalize()
        result.update(driver='diagnostic_teacher_solver_profile',diagnostic_only=True,
            canonical_dataset_eligible=False,solver_override={'noslip_iterations':5})
        write(self.out/'score.json',result)
        write(self.out/'artifact_sha256.json',{p.name:sha(p) for p in self.out.iterdir()
            if p.is_file() and p.name!='artifact_sha256.json'})
        return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-candidate-run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if 'solver_profile_validation' not in args.output.parts:
        raise ValueError('Diagnostic output must be inside a solver_profile_validation directory')
    source_manifest=json.loads((args.source_candidate_run/'manifest.json').read_text())
    candidate=validate_tripod_candidate(source_manifest['parameters'].copy())
    if candidate['object_spec']['shape']!='sphere':raise ValueError('This isolated validation is sphere-only')
    original_factory=gs.options.RigidOptions
    with solver_override(5) as factory_calls:
        sim=Sandbox(output_dir=args.output.parent/(args.output.name+'_runtime'),render=False,
            cube_mass_kg=.05,cube_friction=.8,robot_friction=.8,object_spec=candidate['object_spec'],
            physics_dt_s=1/300,control_substeps=5)
    assert gs.options.RigidOptions is original_factory
    actual=sim.scene.rigid_solver._options
    assert actual.noslip_iterations==5 and int(sim.scene.rigid_solver.rigid_info.noslip_iterations[None])==5
    teacher=ProfileDiagnosticTeacher(sim,args.output,candidate)
    teacher.manifest.update(driver='diagnostic_teacher_solver_profile',diagnostic_only=True,
        canonical_dataset_eligible=False,solver_override={'noslip_iterations':5},
        actual_solver_options=actual.model_dump(mode='json'),actual_engine_noslip_iterations=5,
        factory_calls=factory_calls,factory_restored=True,
        original_candidate_source=str(args.source_candidate_run.resolve()),
        original_candidate_manifest_sha256=sha(args.source_candidate_run/'manifest.json'),
        scope='Alternate solver diagnostic full teacher run; never mix with original-physics demonstrations. A future uniform profile requires new explicit provenance and validation for every object.')
    for source in [Path(__file__),HERE/'diagnostics/replay_noslip.py',HERE/'direct_teacher_tripod.py',
        HERE/'direct_teacher_curved.py',HERE/'direct_teacher_geometry.py',HERE/'direct_teacher_tripod_geometry.py']:
        teacher.manifest['source_sha256'][str(source.resolve())]=sha(source)
        target=teacher.out/'source_snapshot'/'solver_diagnostic'/source.name
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
    write(teacher.out/'manifest.json',teacher.manifest)
    score=teacher.run(run_tripod_teacher);gs.destroy();print(json.dumps(score),flush=True)
    raise SystemExit(0 if score['success'] else 1)


if __name__=='__main__':main()
