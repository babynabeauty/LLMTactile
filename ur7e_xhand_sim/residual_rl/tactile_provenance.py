"""CPU-only provenance for taxel semantics; never upgrades frozen observations.

The kinematic taxels are geometry/motion proxies, not a calibrated redistribution
of rigid-solver contact forces. Historical datasets without this record remain
unversioned even when their command trajectories are replayed by current code.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ENGINE_RELATIVE = 'Genesis/genesis/engine/sensors/kinematic_tactile.py'
HELPER_RELATIVE = 'ur7e_xhand_sim/residual_rl/tactile_provenance.py'
SCHEMA = 'xhand_tactile_model_provenance_v1'
MODEL_VERSION = 'kinematic_force_on_sensor_v3'


def legacy_tactile_model():
    return {'schema':SCHEMA,'model_version':'legacy_unversioned',
        'force_convention':'not_declared_in_source',
        'source_semantics_inferred_from_current_code':False,
        'raw_data_relabelled':False}


def source_paths(root=ROOT):
    return [Path(root)/ENGINE_RELATIVE, Path(root)/HELPER_RELATIVE]


def current_tactile_model(root=ROOT):
    """Describe current source, including a digest of the actual sensor engine."""
    root=Path(root).resolve()
    hashes={str(path.relative_to(root)):hashlib.sha256(path.read_bytes()).hexdigest()
            for path in source_paths(root)}
    return {'schema':SCHEMA,'model_version':MODEL_VERSION,
        'force_convention':'force_on_sensor',
        'relative_velocity_convention':'other_minus_sensor_at_probe',
        'normal_model':'probe overlap spring-damper; nominal N, uncalibrated',
        'shear_model':'relative tangential velocity proxy; nominal N, uncalibrated',
        'twist_model':'relative normal angular velocity proxy; nominal Nm, uncalibrated',
        'distance_query':{
            'sphere':'analytic signed distance and normal',
            'plane':'analytic signed distance and normal',
            'box':'sensor-local analytic signed distance and normal, including probes outside cached SDF bounds',
            'mesh':'cached SDF grid approximation; outside-grid conservative fallback remains',
            'global_solver_sdf_or_contact_dynamics_changed':False},
        'shear_is_calibrated_force':False,
        'solver_contact_force_redistribution':False,
        'static_tangential_load_observable_at_zero_relative_velocity':False,
        'engine_source_relative_path':ENGINE_RELATIVE,
        'engine_source_sha256':hashes[ENGINE_RELATIVE],
        'source_sha256':hashes}


def audit_source_tactile_model(scene, manifest, declared_sources):
    """Audit newly recorded semantics, preserving unknown historical semantics.

    This compares frozen evidence only, never a historical profile against the
    current engine. Thus importing old evidence cannot silently relabel taxels.
    Snapshot content integrity is checked by the importer's existing code audit.
    """
    a=scene.get('tactile_model');b=manifest.get('tactile_model')
    if a is None and b is None:
        return legacy_tactile_model()
    if a is None or b is None or a!=b:
        raise ValueError('Teacher scene and manifest tactile_model must match exactly')
    if not isinstance(a,dict) or a.get('schema')!=SCHEMA or not a.get('model_version'):
        raise ValueError('Invalid tactile_model provenance')
    engine=a.get('engine_source_relative_path');digest=a.get('engine_source_sha256')
    hashes=a.get('source_sha256')
    if (engine!=ENGINE_RELATIVE or not isinstance(hashes,dict) or
            hashes.get(engine)!=digest or not isinstance(digest,str) or len(digest)!=64):
        raise ValueError('Tactile engine source hash is missing or inconsistent')
    for relative,expected in hashes.items():
        matches=[value for original,value in declared_sources.items()
                 if Path(original).as_posix().endswith('/'+relative)]
        if matches!=[expected]:
            raise ValueError('Tactile provenance source must be declared and snapshotted: '+relative)
    result=deepcopy(a)
    result['source_semantics_inferred_from_current_code']=False
    result['raw_data_relabelled']=False
    return result


def tactile_models_match(a,b):
    """Unknown provenance never establishes compatibility with a new sensor."""
    keys=('schema','model_version','force_convention','relative_velocity_convention',
          'engine_source_sha256')
    return (isinstance(a,dict) and isinstance(b,dict) and
            all(a.get(key) is not None and a[key]==b.get(key) for key in keys))


def trajectory_tactile_evidence(trajectory_path,current=None):
    """Commands may be reused while source and newly sampled taxels stay distinct."""
    trajectory=Path(trajectory_path).expanduser().resolve()
    metadata_path=trajectory.parent/'trajectory_metadata.json'
    source=legacy_tactile_model()
    if metadata_path.is_file():
        metadata=json.loads(metadata_path.read_text())
        if metadata.get('trajectory_sha256')!=hashlib.sha256(trajectory.read_bytes()).hexdigest():
            raise ValueError('Trajectory differs from its tactile-provenance metadata hash')
        source=metadata.get('source_tactile_model',source)
    current=current_tactile_model() if current is None else current
    return {'source_tactile_model':source,'current_tactile_model':current,
        'source_and_current_tactile_models_match':tactile_models_match(source,current),
        'source_raw_tactile_reused_as_current_observations':False,
        'command_trajectory_reuse_allowed':True,
        'checkpoint_compatibility_established':False,
        'policy':'Replay commands physically and sample current sensor; preserve old raw data and model provenance.'}


def require_checkpoint_tactile_compatibility(checkpoint):
    """Fail before training if resume would silently change sensor semantics."""
    if checkpoint is None:
        return
    run=Path(checkpoint).expanduser().resolve().parent
    # Periodic checkpoints sit in the run's checkpoints/ folder, one level below its manifest.
    if run.name=='checkpoints':
        run=run.parent
    manifest_path=run/'manifest.json'
    saved=json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    if not tactile_models_match(saved.get('tactile_model'),current_tactile_model()):
        raise ValueError('Checkpoint tactile semantics are missing or differ from the current sensor; compatibility must be evaluated before resuming training')
