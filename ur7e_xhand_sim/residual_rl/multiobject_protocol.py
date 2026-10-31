"""Freeze the material splits and comparison rules before multi-object training.

This module only writes a protocol; it never starts a simulator, training job,
or API request. A candidate containing pending demonstrations is not runnable.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path


SHAPES = ('cube', 'cuboid', 'hex_prism')
LEGACY_SHAPES = ('cube', 'cylinder', 'sphere')
DIRECT_TEACHER_SHAPES = ('cuboid', 'hex_prism', 'sphere', 'cylinder')
DIRECT_TEACHER_DRIVERS = ('direct_teacher_search', 'direct_teacher_frozen_replay')
SEEDS = (1, 2, 3)
METHODS = ('tactile', 'no_tactile')
OLD_TEST_CONDITIONS = {(m, u) for m in (.075, .125, .175) for u in (.3, .5, .7)}
BASE_CONFIG = dict(gain_min=.6, gain_max=3., gain_rate_limit=.2,
                   progress_weight=3., force_weight=.12, slip_weight=1.,
                   unsupported_weight=.5, action_change_weight=.05,
                   success_bonus=25., failure_penalty=200.)


def material_rows(split, masses, frictions):
    return [dict(condition_id=f'{split}_m{round(m*1000):03d}_mu{round(u*100):03d}',
                 mass_kg=m, friction=u) for m in masses for u in frictions]


def condition_key(row):
    return round(float(row['mass_kg']), 8), round(float(row['friction']), 8)


def protocol_shapes(protocol):
    return tuple(obj['object_id'] for obj in protocol['objects'])


def default_object_spec(shape):
    return dict(shape=shape, side_m=.045, radius_m=.0225, height_m=.045,
                size_m=[.045,.060,.045], across_flats_m=.045)


def amend_direct_teacher_replay(protocol, previous_candidate_sha256, *,
                               experiment_id='multiobject_direct_teacher_material_20261028_v1'):
    """Declare the four-object, freshly sensed teacher suite before binding it.

    Preserve material splits, requested transition budgets, rewards and baselines.
    Only shape-dependent lane/coverage counts and demonstration provenance change.
    This produces a candidate; it never imports or freezes demonstration files.
    """
    result = deepcopy(protocol)
    previous_objects = deepcopy(result['objects'])
    result['experiment_id'] = experiment_id
    result['status'] = 'candidate_pending_verified_demonstrations'
    result.pop('frozen_utc', None)
    result['objects'] = []
    for shape in DIRECT_TEACHER_SHAPES:
        spec = default_object_spec(shape)
        if shape == 'cylinder':
            spec['height_m'] = .09
        result['objects'].append(dict(object_id=shape, object_spec=spec,
            trajectory_path=None, trajectory_sha256=None, demonstration_evidence=None))
    result['demonstration_rule'] = (
        'One successful direct-teacher command trajectory per object at 50g/friction .8. '
        'The frozen historical command, gain, phase and time sequence is replayed to collect '
        'fresh kinematic_force_on_sensor_v3 tactile data; historical raw tactile is not reused. '
        'This is privileged simulator demonstration collection, not Astra API inference. '
        'Each imported trajectory must additionally pass a separate fresh-scene replay '
        'at its recorded source gain before binding; collection validation alone is insufficient.')
    result['demonstration_control'] = dict(
        source_gain_choices=[1., 2., 3.],
        source_selection='Gain is recorded in the successful direct teacher commands before import; no post-hoc verification gain search.',
        verification='Independent 300Hz physics/60Hz tactile/15Hz action replay at the recorded source gain, with original task success and controller gain readback.',
        provenance='Explicit direct_teacher_frozen_replay, privileged_teacher=true and actual_astra_api=false in metadata, verification and bind evidence. Hash fresh source validation, source/current tactile models, commands and independent verification.',
        effort_limits='Original URDF effort limits unchanged; increased PD gain is not an increased actuator torque limit.')
    result['split_contract']['assignment'] = (
        'Same material pairs for every protocol shape; no pair crosses train/validation/test.')
    result['split_contract']['meaning'] = (
        'Material-condition interpolation on four known rigid geometries: 45x60x45mm '
        'cuboid, regular hexagonal prism with X across-flats45mm and height45mm, '
        'sphere diameter45mm, and upright cylinder diameter45mm/height90mm. Each has '
        'its own direct-teacher demonstration. No unseen-shape, pose, or real-robot generalization claim.')
    training = result['training']
    lanes = len(result['objects']) * training['proposed_envs_per_shape']
    training['proposed_total_envs'] = lanes
    rollout = lanes * training['n_steps']
    for stage in training['stages']:
        for key in list(stage):
            if key.startswith('planned_timesteps_'):
                del stage[key]
        stage['planned_timesteps'] = math.ceil(stage['requested_timesteps']/rollout)*rollout
    for key in list(training):
        if key.startswith('planned_timesteps_per_model_'):
            del training[key]
    training['planned_timesteps_per_model'] = sum(s['planned_timesteps'] for s in training['stages'])
    training['planned_rollout_envs'] = lanes
    validation_cases = len(result['objects']) * len(result['splits']['validation'])
    training['checkpoint_selection'] = (
        f'For each seed/modality compare stage1 and stage2 on the same {validation_cases} '
        'validation cases: maximize task-success count, then minimize '
        'hold_mean_all_robot_contact_force_N on jointly successful cases; no common '
        'successes or exact force tie defaults to stage2. Freeze the choice and checkpoint '
        'hashes before test. Never choose a seed or checkpoint from test outcomes.')
    training['memory_fallback'] = (
        f'If {lanes} lanes exceed the available allocation, use a declared balanced '
        'scene/chunk schedule with all training conditions per shape. Do not reduce '
        'the condition set or change treatment-specific budgets.')
    evaluation = result['evaluation']
    test_cases = len(result['objects']) * len(result['splits']['test'])
    evaluation['expected_test_policy_episodes'] = test_cases*len(training['seeds'])*len(training['methods'])
    evaluation['expected_test_fixed_unique_episodes'] = test_cases*len(evaluation['fixed_gains'])
    evaluation['significance'] = (
        'Three training seeds are the independent training repetitions; material lanes '
        'are paired test cases, not independent trained policies.')
    result['success']['geometric_extent'] = (
        'Use exact oriented primitive support radii for cuboid, sphere and cylinder; '
        'use the six actual cross-section vertices for the hexagonal prism.')
    gain_min = training['stages'][0]['config']['gain_min']
    gain_max = training['stages'][0]['config']['gain_max']
    result['control']['action'] = (
        f'Absolute common thumb/index PD gain in [{gain_min:g},{gain_max:g}] with max '
        '.2 change/action; unchanged nominal joint targets. Never multiply by source gain.')
    result['control']['initial_state'] = (
        'Each material lane physically replays its own recorded source-gain prefix. '
        'All compared controllers restore the same physical prefix and gain state; '
        'subsequent learned/fixed gains are absolute. Frozen commands do not imply '
        'identical object states between separate physics runs.')
    result['direct_teacher_replay_amendment'] = dict(
        time_utc=datetime.now(timezone.utc).isoformat(),
        previous_candidate_sha256=previous_candidate_sha256,
        previous_objects=previous_objects,
        required_demonstration_source='direct_teacher_frozen_replay',
        new_object_ids=list(DIRECT_TEACHER_SHAPES),
        reason='User authorized simulator-collected successful demonstrations and a shared tactile/no-tactile model across the four shapes.',
        unchanged=['material splits', 'three seeds and both modalities', 'requested transition budgets',
            'optimizer and two-stage rewards', '300/60/15Hz rates', 'fixed-gain controls',
            'success and force metrics', 'validation-only checkpoint selection'],
        tactile_limitations='v3 signals are uncalibrated; shear is a relative-velocity proxy. Thumb coverage gaps are unchanged, including loaded zero-response source samples. Raw zeros remain unfilled.')
    validate_protocol(result)
    return result


def amend_flatface_shapes(protocol, previous_candidate_sha256):
    """Keep the training design, transparently change unverified object scope."""
    result=deepcopy(protocol)
    previous_objects=deepcopy(result['objects'])
    cube=next(obj for obj in previous_objects if obj['object_id']=='cube')
    # Preserve any already-bound first cube success; only normalize its shape spec.
    cube['object_spec']=default_object_spec('cube')
    result['objects']=[cube]+[dict(object_id=shape,object_spec=default_object_spec(shape),
        trajectory_path=None,trajectory_sha256=None,demonstration_evidence=None) for shape in SHAPES[1:]]
    result['split_contract']['meaning']=(
        'Material-condition interpolation on three known flat-face geometries: '
        '45mm cube, 45x60x45mm cuboid and regular hexagonal prism with X across-flats '
        '45mm and height45mm. Each has its own Astra demonstration. No claim of '
        'curved-object, unseen-shape, pose, or real-robot generalization.')
    result['success']['geometric_extent']=(
        'Use the exact oriented primitive support radius for cube/cuboid, cylinder, '
        'sphere or hexagonal prism; for the hexagonal prism use its six actual '
        'cross-section vertices. Success threshold and final3s interval are unchanged.')
    result['flatface_scope_amendment']=dict(
        time_utc=datetime.now(timezone.utc).isoformat(),
        previous_candidate_sha256=previous_candidate_sha256,
        scope='Before formal training; no validation/test outcomes read.',
        reason='Curved cylinder/sphere Astra attempts have not supplied qualifying independently verified demonstrations. Their acquisition would require additional unproven closure changes. Start the known-shape force-learning study on three distinct rigid geometries with accessible opposing flat faces.',
        previous_objects=previous_objects,
        new_object_ids=list(SHAPES),
        dimensions=dict(cube_side_m=.045,cuboid_size_m=[.045,.060,.045],
            hex_prism_across_flats_x_m=.045,hex_prism_height_m=.045,
            hex_prism_circumradius_m=.0225/math.cos(math.pi/6),
            hex_prism_vertex_angles_deg=[30+60*k for k in range(6)]),
        retained='All curved-shape candidates, source attempts, failures and implementation support remain intact. Existing cube demonstration is unchanged.',
        unchanged=['training and validation/test material pairs','all three seeds and both modalities',
            'training budgets, optimizer, rewards and two-stage schedule','300/60/15Hz rates',
            'absolute .2–3 residual gain and source-gain prefix contract',
            'success thresholds and hold interval','fixed .6/1/2/3 controls',
            'validation checkpoint selection and single final test rule'])
    validate_protocol(result)
    return result


def amend_demonstration_gains(protocol, previous_candidate_sha256):
    """Declare source-selected demo gains before training; preserve its design."""
    result = deepcopy(protocol)
    result['demonstration_rule'] = (
        'One actual Astra-generated successful command trajectory per object at '
        '50g/friction .8. Freeze the first source-success trajectory that also '
        'passes independent 300/60/15 replay at its recorded Astra-selected gain; '
        'retain all attempts. Astra chooses common thumb/index gain 1, 2 or 3 '
        'in its close action before executing closure. Do not tune or sweep the '
        'verification gain after a source failure or replay failure. No '
        'rule-generated replacement without an explicit protocol amendment.')
    result['demonstration_control'] = dict(
        source_gain_choices=[1., 2., 3.], legacy_source_gain=1.,
        source_selection='Actual saved Astra close SkillAct; selection precedes closure and is audited against executed gain readback.',
        verification='Independently replay commands at 300Hz physics/60Hz tactile/15Hz action. Verification absolute gain must equal the recorded source close gain; success is recomputed with the unchanged criterion.',
        provenance='Record source_grip_gain and verification_gain separately, with source action/readback and verification artifact hashes. Existing cube gain1 evidence without these new fields remains valid as legacy gain1.',
        first_success='Freeze the first independently verified success per object; preserve earlier failed attempt logs. This rule is evaluated from the complete attempt record, never inferred from the artifact filename.',
        effort_limits='Original URDF effort limits unchanged; increased PD gain is not an increased actuator torque limit.')
    result['control']['action'] = (
        'Absolute common thumb/index PD gain in [.2,3] with max .2 change/action; '
        'unchanged nominal joint position commands. The gain is relative to the '
        'original base PD parameters, never multiplied by the source demonstration gain.')
    result['control']['initial_state'] = (
        'Every material lane physically replays its own demonstration prefix at '
        'its assigned mass/friction and recorded source gain schedule (legacy '
        'records default to gain1). All compared controllers restore that same '
        'physical prefix and gain state. Thereafter learned and fixed controllers '
        'request absolute gains under the same .2/action limit; source gains '
        'are not multiplied into residual actions.')
    check = 'source-selected demonstration gain equals independent verification gain; paired source-gain prefix replay'
    if check not in result['audit']['required_checks']:
        result['audit']['required_checks'].append(check)
    result['demonstration_gain_amendment'] = dict(
        time_utc=datetime.now(timezone.utc).isoformat(),
        previous_candidate_sha256=previous_candidate_sha256,
        scope='Pretraining engineering amendment; no formal training started and no validation/test outcomes read.',
        reason='Curved-shape demonstration attempts at gain1 have not yet produced a verified success. Permit Astra to select a bounded source grasp gain while retaining the original easiest material condition and honest source/replay provenance.',
        changes=['Allow Astra close to choose source common thumb/index PD gain1/2/3 before action execution.',
                 'Verify each source-success trajectory only at its recorded gain; no post-hoc gain search.',
                 'Replay recorded source prefix gains identically for all compared controllers; residual gain remains absolute .2–3.'],
        unchanged=['objects and dimensions', '50g/friction .8 demonstration material',
                   'train/validation/test material pairs', 'training budgets and optimizer settings',
                   'rewards and stage schedule', 'success criteria',
                   'fixed .6/1/2/3 controls', 'checkpoint selection and test-use rules'])
    validate_protocol(result)
    return result


def _verified_json_evidence(evidence, prefix, required=False):
    path_value, expected = evidence.get(prefix+'_path'), evidence.get(prefix+'_sha256')
    if path_value is None and expected is None and not required:
        return None
    if not path_value or not expected:
        raise ValueError(f'Demonstration {prefix} requires both path and SHA256')
    path = Path(path_value).expanduser()
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError(f'Demonstration {prefix} hash mismatch')
    return json.loads(path.read_text())


def validate_demonstration_gains(evidence, trajectory_sha256):
    """Audit optional new gain provenance without invalidating legacy gain1."""
    driver = evidence.get('demonstration_source')
    direct_teacher = driver in DIRECT_TEACHER_DRIVERS
    new_fields = ('source_grip_gain', 'verification_gain')
    if not any(name in evidence for name in new_fields):
        if direct_teacher:
            raise ValueError('Direct-teacher demonstration requires source and verification gain metadata')
        if 'fixed_gain' in evidence and evidence['fixed_gain'] != 1.:
            raise ValueError('Nonlegacy demonstration gain requires source and verification gain metadata')
        return
    gains = []
    for name in new_fields:
        value = evidence.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value not in (1., 2., 3.):
            raise ValueError(f'Demonstration {name} must be an explicit gain1, 2 or 3')
        gains.append(float(value))
    if gains[0] != gains[1]:
        raise ValueError('Demonstration source and verification gain differ; post-hoc gain search is disallowed')
    if 'fixed_gain' in evidence and evidence['fixed_gain'] != gains[1]:
        raise ValueError('Demonstration fixed_gain differs from verification gain')
    if not direct_teacher and evidence.get('actual_astra_api') is not True:
        raise ValueError('Source-selected gain requires actual Astra API evidence')
    metadata = _verified_json_evidence(evidence, 'trajectory_metadata', required=True)
    if metadata.get('trajectory_sha256') != trajectory_sha256:
        raise ValueError('Demonstration gain metadata identifies a different trajectory')
    source = metadata.get('source_grip_gain_audit') or {}
    if source.get('source_grip_gain') != gains[0]:
        raise ValueError('Demonstration source gain differs from trajectory metadata')
    verification = _verified_json_evidence(evidence, 'verification', required=True)
    if verification.get('accepted') is not True:
        raise ValueError('Demonstration independent verification did not pass')
    for name, expected in zip(new_fields, gains):
        if verification.get(name) != expected:
            raise ValueError(f'Demonstration {name} differs from independent verification')
    if direct_teacher:
        _validate_direct_teacher_provenance(evidence, metadata, verification)
    readback = _verified_json_evidence(evidence, 'controller_readback', required=direct_teacher)
    if readback is not None:
        if readback.get('all_applications_verified') is not True:
            raise ValueError('Demonstration controller gain readback did not pass')
        for field in ('controller_readback_path', 'controller_readback_sha256'):
            if field in verification and verification[field] != evidence[field]:
                raise ValueError('Demonstration controller gain readback differs from verification')
    if evidence.get('first_independently_verified_success') is False:
        raise ValueError('Demonstration explicitly violates the first independently verified success rule')


def _validate_direct_teacher_provenance(evidence, metadata, verification):
    """Separate simulator teachers from Astra and collection from verification."""
    driver = evidence['demonstration_source']
    for record in (evidence, metadata, verification):
        if (record.get('demonstration_source') != driver
                or record.get('privileged_teacher') is not True
                or record.get('actual_astra_api') is not False):
            raise ValueError('Direct-teacher provenance must explicitly identify its driver, privilege and non-Astra source')
    if driver != 'direct_teacher_frozen_replay':
        return
    collection = _verified_json_evidence(evidence, 'source_replay_validation', required=True)
    replay_audit = metadata.get('frozen_replay_audit') or {}
    if (replay_audit.get('validation_sha256') != evidence['source_replay_validation_sha256']
            or replay_audit.get('exact_historical_commands_recomputed') is not True
            or replay_audit.get('fresh_tactile_audit_recomputed') is not True
            or replay_audit.get('collection_validation_is_independent_replay_verification') is not False):
        raise ValueError('Frozen replay requires audited fresh collection and a separate independent verification')
    if Path(evidence['verification_path']).resolve() == Path(evidence['source_replay_validation_path']).resolve():
        raise ValueError('Collection validation cannot serve as independent verification')
    flags = dict(accepted=True, fresh_tactile_captured=True, raw_tactile_reused=False,
                 exact_source_commands_replayed=True, source_files_unchanged=True,
                 current_source_unchanged=True)
    if (collection.get('status') != 'passed' or collection.get('audit_errors') != []
            or any(collection.get(key) is not value for key,value in flags.items())):
        raise ValueError('Frozen replay collection validation did not pass fresh-sensing requirements')
    source_model = metadata.get('source_tactile_model')
    current_model = verification.get('tactile_model')
    for model in (source_model, current_model):
        if not isinstance(model, dict) or model.get('model_version') != 'kinematic_force_on_sensor_v3':
            raise ValueError('Frozen replay requires explicit v3 source and verification tactile models')
    audit_fields = ('source_semantics_inferred_from_current_code', 'raw_data_relabelled')
    if any(source_model.get(key) is not False for key in audit_fields):
        raise ValueError('Frozen replay source semantics must be recorded without relabelling raw data')
    # Import adds audit-only flags; compare the actual sensor profile to the
    # collector's unaugmented profile without dropping any physics/hash field.
    source_profile = {key:value for key,value in source_model.items() if key not in audit_fields}
    if (evidence.get('source_tactile_model') != source_model
            or verification.get('source_tactile_model') != source_model
            or collection.get('tactile_model') != source_profile
            or replay_audit.get('current_tactile_model') != source_profile
            or evidence.get('tactile_model') != current_model):
        raise ValueError('Frozen replay tactile provenance differs between collection, metadata and verification')
    for field in ('source_replay_validation_path', 'source_replay_validation_sha256'):
        if verification.get(field) != evidence[field]:
            raise ValueError('Independent verification references different frozen replay collection evidence')


def make_protocol():
    stages = []
    rollout = 36*128
    for name, requested, weight in [('stage1', 131072, .02), ('stage2', 655360, .12)]:
        config = {**BASE_CONFIG, 'force_weight':weight}
        stages.append(dict(name=name, requested_timesteps=requested, config=config,
                           planned_timesteps_36_lanes=math.ceil(requested/rollout)*rollout))
    result = {
        'schema_version':1, 'experiment_id':'multiobject_astra_material_20261027_v1',
        'status':'candidate_pending_verified_demonstrations',
        'created_utc':datetime.now(timezone.utc).isoformat(),
        'objects':[dict(object_id=shape, object_spec=default_object_spec(shape), trajectory_path=None,
                         trajectory_sha256=None, demonstration_evidence=None) for shape in SHAPES],
        'demonstration_condition':dict(mass_kg=.05, friction=.8),
        'demonstration_rule':'One Astra-generated successful command trajectory per object; freeze the first independently verified success, retain all attempt logs. No rule-generated replacement without an explicit protocol amendment.',
        'splits':{
            'train':material_rows('train', (.05,.10,.15,.20), (.2,.5,.8)),
            'validation':material_rows('validation', (.065,.125,.185), (.35,.65)),
            'test':material_rows('test', (.080,.120,.180), (.3,.45,.7))},
        'split_contract':{
            'assignment':'Same material pairs for all three shapes; no pair crosses train/validation/test.',
            'old_test_excluded':[list(pair) for pair in sorted(OLD_TEST_CONDITIONS)],
            'meaning':'Material-condition interpolation on these three known flat-face object shapes, each with its own demonstration. No claim of curved-object, unseen-shape, pose, or real-robot generalization.',
            'test_use':'Freeze all code, demonstrations, reward settings and checkpoints before test outcomes are read; never choose a checkpoint from test results.'},
        'scene':dict(table_top_z_m=.2, initial_xy_m=[-.60,-.1333],
                     initial_orientation_wxyz=[1.,0.,0.,0.],
                     initial_bottom_m=.2, robot_base_position_m=[0.,0.,.2],
                     rigid_only=True, distractors=False),
        'training':dict(seeds=list(SEEDS), methods=list(METHODS), shared_across_shapes=True,
            conditions_per_shape=12, proposed_envs_per_shape=12, proposed_total_envs=36,
            stages=stages, requested_timesteps_per_model=786432,
            planned_timesteps_per_model_36_lanes=sum(s['planned_timesteps_36_lanes'] for s in stages),
            timesteps_definition='Counts individual environment transitions over all shapes, not vector steps or steps per shape; each stage rounds up to full PPO rollouts and records actual counts.',
            n_steps=128, batch_size=256, n_epochs=5, learning_rate=.0003,
            adam_eps=1e-5, use_sde=True, normalize_sde_latent=True,
            action_std_init=.12, sde_sample_freq=30, gamma=1., gae_lambda=.95,
            shared_encoder=True, reward_scale=1., initial_gain=2.75,
            checkpoint_selection='For each seed/modality compare stage1 and stage2 on the same18 validation cases: maximize task-success count, then minimize hold_mean_all_robot_contact_force_N on jointly successful cases; no common successes or exact force tie defaults to stage2. Freeze the choice and checkpoint hashes before test. Never choose a seed or checkpoint from test outcomes.',
            curriculum='Same fixed schedule for both modalities: stage1 force penalty .02, stage2 .12. Final task reward and original success criterion unchanged.',
            balance='Equal lane counts per shape and equal exposure to all training conditions; record actual transitions and completed episodes by shape. Shorter trajectories may produce more episodes.',
            memory_fallback='If 36 lanes exceed the available allocation, use a declared balanced scene/chunk schedule with all 12 conditions per shape. Do not reduce the condition set or change treatment-specific budgets.'),
        'control':dict(physics_hz=300, tactile_hz=60, action_hz=15, history_samples=12,
            action_dimension=1, action='common thumb/index PD gain; unchanged nominal joint position commands',
            actor_inputs='Measured tactile and proprioceptive histories, previous gains, nominal action and phase. No true mass, friction, object pose, solver contact labels, or shape identity is added.',
            no_tactile='Identical policy/control/proprioception; tactile input disabled before every feature path.',
            initial_state='Every material lane physically replays its own demonstration prefix at its assigned mass and friction before residual control. Compared controllers reset to exactly the same prefix.'),
        'success':dict(hold_duration_s=3., minimum_clearance_m=.05,
            task='Actual lowest object surface above table >=.05m at every sample overlapping the final3s, and no robot-table/self collision during the policy interval.',
            strict='task_success plus >=2 sensor-bearing links each >.05N throughout the hold.',
            geometric_extent='Use the exact oriented primitive support radius for cube, cuboid and hexagonal prism; use actual hexagonal cross-section vertices for the prism.',
            training_terminal='task_success; report strict_success separately without changing the old label.'),
        'evaluation':dict(splits=['validation','test'], fixed_gains=[1.,2.,3.],
            deterministic=True, episodes_per_material_per_controller=1,
            paired_policy_seeds=list(SEEDS),
            expected_test_policy_episodes=3*9*2*3,
            expected_test_fixed_unique_episodes=3*9*3,
            force_primary='hold_mean_all_robot_contact_force_N',
            force_secondary=['hold_mean_all_robot_normal_force_N','hold_mean_sensor_contact_force_N',
                             'hold_peak_all_robot_normal_force_N','hold_peak_all_robot_contact_force_N'],
            force_window='Time-weighted final3s; partial 60Hz boundary intervals use exact overlap. Report policy-window means and peaks separately.',
            reporting='All-episode success rates; successful-only force; same-shape/condition/seed jointly successful tactile-versus-no-tactile pairs. Failed post-drop zero force is not evidence of force efficiency.',
            fixed_baseline_rule='Deterministic fixed1/2/3 may be run once per shape/material and shared across seed comparisons; reused baseline records are not independent repetitions.',
            significance='Three training seeds are the independent training repetitions; material lanes are paired test cases, not 81 independent trained policies.'),
        'audit':dict(required_checks=['disjoint material splits','fresh test pairs versus prior9',
            'demonstration SHA256 and success evidence','checkpoint hashes and frozen actor inputs',
            'mass/density/inertia and contact friction readback','paired prefix equality',
            'trace-derived success and force metrics','expected evaluation coverage'],
            trace_keys=['sample_time_s','clearance_m','fault','contact_load_N',
                        'all_robot_contact_load_N','all_robot_normal_load_N','contact_friction_error']),
    }
    validate_protocol(result)
    return result


def validate_protocol(protocol, require_demonstrations=False):
    if protocol.get('schema_version') != 1:
        raise ValueError('Unsupported multi-object protocol version')
    objects = protocol['objects']
    if protocol_shapes(protocol) not in (SHAPES,LEGACY_SHAPES,DIRECT_TEACHER_SHAPES):
        raise ValueError('Protocol must contain a complete canonical flat-face, legacy curved or four-object teacher suite')
    if any(obj['object_spec'].get('shape')!=obj['object_id'] for obj in objects):
        raise ValueError('Protocol object identifier and geometry shape differ')
    identifiers, previous = set(), set()
    for split, count in [('train',12), ('validation',6), ('test',9)]:
        rows = protocol['splits'][split]
        if len(rows) != count:
            raise ValueError(f'{split} requires exactly {count} material pairs')
        keys = set()
        for row in rows:
            pair = condition_key(row)
            if not all(math.isfinite(v) for v in pair) or not (.05 <= pair[0] <= .20 and .2 <= pair[1] <= .8):
                raise ValueError('Material pair outside preregistered mass/friction bounds')
            if row['condition_id'] in identifiers or pair in keys:
                raise ValueError('Duplicate condition identifier or material pair')
            identifiers.add(row['condition_id']); keys.add(pair)
        if keys & previous:
            raise ValueError('Train/validation/test material pairs overlap')
        previous |= keys
        if split == 'test' and keys & OLD_TEST_CONDITIONS:
            raise ValueError('New test must not reuse the previous nine material conditions')
    training = protocol['training']
    if training['seeds'] != list(SEEDS) or training['methods'] != list(METHODS):
        raise ValueError('Both modalities and all three training seeds are required')
    if sum(s['requested_timesteps'] for s in training['stages']) != training['requested_timesteps_per_model']:
        raise ValueError('Training stage budgets do not add up')
    if require_demonstrations:
        for obj in objects:
            path = Path(obj.get('trajectory_path') or '')
            if not path.is_file() or not obj.get('trajectory_sha256'):
                raise ValueError(f'Missing verified demonstration for {obj["object_id"]}')
            if hashlib.sha256(path.read_bytes()).hexdigest() != obj['trajectory_sha256']:
                raise ValueError(f'Demonstration hash mismatch for {obj["object_id"]}')
            evidence = obj.get('demonstration_evidence') or {}
            if evidence.get('task_success') is not True or evidence.get('mass_kg') != .05 or evidence.get('friction') != .8:
                raise ValueError('Demonstration evidence must verify task success at 50g and friction .8')
            if (protocol_shapes(protocol) == DIRECT_TEACHER_SHAPES
                    and evidence.get('demonstration_source') != 'direct_teacher_frozen_replay'):
                raise ValueError('Four-object teacher suite requires explicit direct_teacher_frozen_replay evidence')
            validate_demonstration_gains(evidence, obj['trajectory_sha256'])
            if evidence.get('demonstration_source') in DIRECT_TEACHER_DRIVERS:
                metadata = _verified_json_evidence(evidence, 'trajectory_metadata', required=True)
                if metadata.get('object_id') != obj['object_id'] or metadata.get('object_spec') != obj['object_spec']:
                    raise ValueError('Direct-teacher object geometry differs from protocol')
    return True


def bind_demonstrations(protocol, mapping):
    """Attach explicit verification metadata; never infer success from a filename."""
    result = deepcopy(protocol)
    if set(mapping) != set(protocol_shapes(result)):
        raise ValueError(f'Demonstration mapping requires all {len(result["objects"])} protocol shapes')
    for obj in result['objects']:
        record = mapping[obj['object_id']]
        path = Path(record['trajectory_path']).expanduser().resolve()
        actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        if record.get('trajectory_sha256') and record['trajectory_sha256'] != actual_hash:
            raise ValueError('Bind record demonstration hash mismatch')
        obj.update(trajectory_path=str(path),
                   trajectory_sha256=actual_hash,
                   demonstration_evidence=record['demonstration_evidence'])
    validate_protocol(result, require_demonstrations=True)
    result['status'] = 'frozen_ready_for_training'
    result['frozen_utc'] = datetime.now(timezone.utc).isoformat()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--candidate', type=Path)
    parser.add_argument('--demonstrations-json', type=Path)
    args = parser.parse_args()
    protocol = json.loads(args.candidate.read_text()) if args.candidate else make_protocol()
    if args.demonstrations_json:
        protocol = bind_demonstrations(protocol, json.loads(args.demonstrations_json.read_text()))
    else:
        validate_protocol(protocol)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(protocol, stream, indent=2, ensure_ascii=False)
        stream.write('\n')
    print(json.dumps({'output':str(args.output.resolve()),'status':protocol['status']}))


if __name__ == '__main__':
    main()
