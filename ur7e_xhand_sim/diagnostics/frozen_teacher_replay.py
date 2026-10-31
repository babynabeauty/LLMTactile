"""Replay a successful direct teacher's exact commands and record fresh taxels.

No IK, policy, tactile feedback decisions, attachments or object pose writes.
The historical source is immutable; new observations carry current provenance.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import traceback

import genesis as gs
import numpy as np

from environment import Sandbox
from direct_teacher import DirectTeacher, compact, sha, write
from build_assets import HERE
from residual_rl.tactile_provenance import current_tactile_model, legacy_tactile_model
from residual_rl.multiobject_import_teacher import (
    audit_scene, audit_source_solver, audit_rigid_options, convert_commands,
    audit_raw_tactile, audit_physics, audit_source_snapshots, spec_from_record,
)
from residual_rl.tactile_provenance import audit_source_tactile_model


SOURCE_FILES = ('control.jsonl', 'evaluation.jsonl', 'raw_tactile.npz',
                'manifest.json', 'object_scene.json', 'score.json')


def read(path):
    return json.loads(Path(path).read_text())


class FrozenTeacherReplay(DirectTeacher):
    def finalize(self):
        score = super().finalize()
        score.update(driver='direct_teacher_frozen_replay',
                     historical_raw_tactile_reused=False,
                     replay_without_ik_or_feedback_decisions=True)
        write(self.out/'score.json', score)
        return score


def replay(teacher, rows):
    s = teacher.sim
    for i, source in enumerate(rows):
        gain = float(source['grip_gain'])
        if gain != teacher.grip_gain:
            teacher.set_gain(gain)
        target = np.asarray(source['sent_target_rad'], float)
        phase = source['phase']
        teacher.phase = phase
        before = s.observe(full=False)
        s.step({'actions':dict(zip(s.names, ((target-s.offset)/s.scale).tolist())), 'steps':1})
        np.testing.assert_allclose(s.target, target, rtol=0, atol=2e-15)
        after = s.observe(full=True)
        e = s.evaluate()
        e.update(step=s.steps, phase=phase, sample_time_s=s.steps/60)
        row = dict(step=s.steps, sample_time_s=s.steps/60, source_row=i, phase=phase,
                   before=before, after=compact(after), sent_target_rad=target.tolist(), grip_gain=gain)
        if teacher.pending_pd:
            row['grip_gain_controller'] = teacher.pd_readback
            teacher.pending_pd = False
        teacher.record.write(json.dumps(row, allow_nan=False)+'\n')
        teacher.eval_record.write(json.dumps(e, allow_nan=False)+'\n')
        teacher.evals.append(e)
        teacher.raw.append(dict(sample_time_s=s.steps/60,
            tactile=np.asarray([after['tactile'][f]['force_sensor_N'] for f in s.sensors], np.float32),
            joint_positions_rad=np.asarray([after['joint_positions_rad'][n] for n in s.names], np.float32),
            sent_target_rad=target.copy(), grip_gain=gain))
        bad = [c for c in e['solver_contacts'] if 'ur7e_xhand' in (c['entity_a'],c['entity_b'])
               and ('table' in (c['entity_a'],c['entity_b']) or c['entity_a']==c['entity_b'])]
        if bad:
            raise ValueError('Robot self/table collision during replay')
        if s.fault:
            raise ValueError('Simulation fault: '+str(s.fault))
        if i == len(rows)-1 or phase != rows[i+1]['phase']:
            teacher.stage(phase+'_end')
        elif s.steps % 120 == 0:
            teacher.capture_frames()
        if s.steps % 120 == 0:
            status = dict(status='running', completed_samples=s.steps, requested_samples=len(rows),
                          phase=phase, bottom_clearance_m=e['bottom_clearance_m'])
            write(teacher.out/'status.json', status)
            print(json.dumps(status), flush=True)


def run(source, output):
    source, output = source.resolve(), output.resolve()
    if output.exists() or source in output.parents:
        raise ValueError('A fresh output outside the frozen source is required')
    hashes = {name:sha(source/name) for name in SOURCE_FILES}
    manifest, scene, score = [read(source/name) for name in ('manifest.json','object_scene.json','score.json')]
    if manifest.get('driver') != 'direct_teacher_search' or not score.get('success'):
        raise ValueError('Expected a successful direct-teacher source')
    spec = spec_from_record(manifest['object_spec'])
    dt = audit_scene(scene, manifest, spec)
    profile, _, _ = audit_source_solver(scene, manifest)
    rows, _ = convert_commands(source/'control.jsonl', score['control_steps'], dt, scene)
    source_audit = audit_physics(source/'evaluation.jsonl', rows, spec, dt)
    audit_source_snapshots(source, manifest['source_sha256'])
    output.parent.mkdir(parents=True, exist_ok=True)
    sim = Sandbox(output_dir=output.parent/'runtime', render=False,
                  object_spec=asdict(spec), cube_mass_kg=float(scene['materials']['cube_mass_kg']),
                  cube_friction=float(scene['materials']['cube_friction']),
                  robot_friction=float(scene['materials']['robot_friction']),
                  physics_dt_s=1/300, control_substeps=5,
                  noslip_iterations=profile['noslip_iterations'], noslip_tolerance=profile['noslip_tolerance'])
    try:
        teacher = FrozenTeacherReplay(sim, output, manifest['parameters'])
        physics_options = audit_rigid_options(scene['rigid_options_actual'], sim.describe()['rigid_options_actual'])
        np.testing.assert_allclose(sim.describe()['object_spec']['initial_position_m'], scene['object_spec']['initial_position_m'], rtol=0, atol=1e-9)
        teacher.manifest.update(driver='direct_teacher_frozen_replay',
            provenance='Exact successful direct-teacher joint targets and gain schedule physically replayed; fresh current-sensor observations.',
            teacher_inputs='Only frozen joint targets, gains and phase labels drive the robot. Physics and tactile observations are recorded, not used for decisions.',
            source_run=str(source), source_files_sha256=hashes,
            source_tactile_model=manifest.get('tactile_model', legacy_tactile_model()),
            source_physics_audit=source_audit, rigid_options_audit=physics_options,
            raw_tactile_reused=False, no_ik_or_feedback_decisions=True,
            formal_training_started=False, tactile_model=current_tactile_model(),
            limitations=['Taxel force amplitudes are uncalibrated', 'Shear is a velocity proxy',
                         'Thumb tip coverage gap remains; mesh SDF remains approximate'])
        for path in (Path(__file__).resolve(), HERE/'residual_rl/multiobject_import_teacher.py'):
            teacher.manifest['source_sha256'][str(path)] = sha(path)
            dest = output/'source_snapshot'/path.relative_to(HERE.parent)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)
        (output/'source_commands').mkdir()
        for name in ('control.jsonl','manifest.json','object_scene.json','score.json'):
            shutil.copy2(source/name, output/'source_commands'/name)
        write(output/'manifest.json', teacher.manifest)
        result = teacher.run(lambda t:replay(t, rows))
        validation = dict(status='failed', accepted=False, fresh_tactile_captured=bool(teacher.raw),
                          raw_tactile_reused=False, audit_errors=[])
        try:
            if not result['success']:
                raise ValueError('Replay did not meet final three seconds / 5 cm success criterion: '+str(result['fault']))
            fresh_scene = read(output/'object_scene.json')
            new_rows, gain_audit = convert_commands(output/'control.jsonl', result['control_steps'], dt, fresh_scene)
            if new_rows != rows:
                raise ValueError('Targets, gain schedule, sample times or phases changed')
            tactile_audit = audit_raw_tactile(output/'raw_tactile.npz', new_rows, dt, output/'control.jsonl', list(sim.sensors))
            physical_audit = audit_physics(output/'evaluation.jsonl', new_rows, spec, dt)
            audit_source_snapshots(output, teacher.manifest['source_sha256'])
            audit_source_tactile_model(fresh_scene, teacher.manifest, teacher.manifest['source_sha256'])
            if current_tactile_model() != teacher.manifest['tactile_model']:
                raise ValueError('Tactile source changed during collection')
            if any(sha(source/name)!=digest for name,digest in hashes.items()):
                raise ValueError('Historical source file changed during replay')
            old_eval = [json.loads(line) for line in (source/'evaluation.jsonl').read_text().splitlines()]
            position_error = max(float(np.max(np.abs(np.asarray(old['object_position_m'])-new['object_position_m'])))
                                 for old,new in zip(old_eval, teacher.evals))
            validation.update(status='passed', accepted=True, exact_source_commands_replayed=True,
                source_files_unchanged=True, current_source_unchanged=True,
                tactile_audit=tactile_audit, physical_audit=physical_audit,
                grip_gain_audit=gain_audit, source_comparison=dict(max_position_component_error_m=position_error),
                tactile_model=teacher.manifest['tactile_model'])
        except Exception as exc:
            validation['audit_errors'].append(f'{type(exc).__name__}: {exc}')
            (output/'validation_error.txt').write_text(traceback.format_exc())
        write(output/'validation.json', validation)
        write(output/'status.json', dict(status='completed' if validation['accepted'] else 'failed',
                                        completed_samples=len(teacher.evals), accepted=validation['accepted']))
        write(output/'artifact_sha256.json', {p.name:sha(p) for p in output.iterdir()
                                            if p.is_file() and p.name!='artifact_sha256.json'})
        print(json.dumps(dict(output=str(output), score=result, validation=validation)), flush=True)
        return validation
    finally:
        gs.destroy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = run(args.source, args.output)
    raise SystemExit(0 if result['accepted'] else 1)


if __name__ == '__main__':
    main()
