"""CPU evidence tests: signed sensor updates must not relabel historical data."""
import copy
import hashlib
import json

import pytest

from .tactile_provenance import (ROOT, ENGINE_RELATIVE, HELPER_RELATIVE,
    current_tactile_model, audit_source_tactile_model, tactile_models_match,
    trajectory_tactile_evidence, require_checkpoint_tactile_compatibility)


def test_current_profile_hashes_sensor_and_explicit_proxy_semantics():
    profile=current_tactile_model()
    assert profile['engine_source_sha256']==hashlib.sha256((ROOT/ENGINE_RELATIVE).read_bytes()).hexdigest()
    assert profile['force_convention']=='force_on_sensor'
    assert profile['relative_velocity_convention']=='other_minus_sensor_at_probe'
    assert profile['model_version']=='kinematic_force_on_sensor_v3'
    assert 'sensor-local analytic' in profile['distance_query']['box']
    assert 'approximation' in profile['distance_query']['mesh']
    assert not profile['distance_query']['global_solver_sdf_or_contact_dynamics_changed']
    assert not profile['shear_is_calibrated_force']
    assert not profile['static_tangential_load_observable_at_zero_relative_velocity']
    assert HELPER_RELATIVE in profile['source_sha256']
    assert tactile_models_match(profile,profile)


def test_missing_historical_profile_is_not_inferred_from_current_source():
    legacy=audit_source_tactile_model({}, {}, {})
    assert legacy['model_version']=='legacy_unversioned'
    assert not legacy['source_semantics_inferred_from_current_code']
    assert not legacy['raw_data_relabelled']
    assert not tactile_models_match(legacy,current_tactile_model())


def test_new_profile_requires_matching_frozen_source_hashes():
    profile=current_tactile_model()
    declared={str(ROOT/key):value for key,value in profile['source_sha256'].items()}
    a={'tactile_model':profile}
    result=audit_source_tactile_model(a,a,declared)
    assert tactile_models_match(result,profile)
    with pytest.raises(ValueError,match='match exactly'):
        audit_source_tactile_model(a,{},declared)
    with pytest.raises(ValueError,match='declared and snapshotted'):
        audit_source_tactile_model(a,a,{})
    changed=copy.deepcopy(a)
    changed['tactile_model']['engine_source_sha256']='0'*64
    with pytest.raises(ValueError,match='inconsistent'):
        audit_source_tactile_model(changed,changed,declared)


def test_old_commands_can_be_replayed_without_relabelling_raw(tmp_path):
    path=tmp_path/'control.jsonl';path.write_text('frozen commands\n')
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    metadata={'trajectory_sha256':digest}
    frozen=json.dumps(metadata)
    (tmp_path/'trajectory_metadata.json').write_text(frozen)
    record=trajectory_tactile_evidence(path)
    assert record['command_trajectory_reuse_allowed']
    assert not record['source_and_current_tactile_models_match']
    assert not record['source_raw_tactile_reused_as_current_observations']
    assert not record['checkpoint_compatibility_established']
    assert (tmp_path/'trajectory_metadata.json').read_text()==frozen
    path.write_text('mutated commands')
    with pytest.raises(ValueError,match='metadata hash'):
        trajectory_tactile_evidence(path)


def test_resume_rejects_legacy_or_changed_sensor_but_allows_current(tmp_path):
    path=tmp_path/'final_model.zip';path.write_bytes(b'CPU fixture, not a model')
    manifest=tmp_path/'manifest.json'
    require_checkpoint_tactile_compatibility(None)
    with pytest.raises(ValueError,match='Checkpoint tactile semantics'):
        require_checkpoint_tactile_compatibility(path)
    manifest.write_text(json.dumps({'tactile_model':current_tactile_model()}))
    require_checkpoint_tactile_compatibility(path)
    changed=current_tactile_model();changed['engine_source_sha256']='0'*64
    manifest.write_text(json.dumps({'tactile_model':changed}))
    with pytest.raises(ValueError,match='Checkpoint tactile semantics'):
        require_checkpoint_tactile_compatibility(path)


def test_periodic_checkpoint_uses_the_run_manifest(tmp_path):
    (tmp_path/'checkpoints').mkdir()
    path=tmp_path/'checkpoints'/'ppo_000262152.zip';path.write_bytes(b'CPU fixture, not a model')
    with pytest.raises(ValueError,match='Checkpoint tactile semantics'):
        require_checkpoint_tactile_compatibility(path)
    (tmp_path/'manifest.json').write_text(json.dumps({'tactile_model':current_tactile_model()}))
    require_checkpoint_tactile_compatibility(path)
