"""Explicit rigid-solver provenance; absent historical fields always mean zero.

This module is CPU-only and never imports Genesis or initializes a simulator.
"""
import hashlib
import json
import math
from pathlib import Path


SCHEMA='genesis_rigid_solver_profile_v1'


def normalize_solver_profile(value=None):
    if value is None:
        value={'schema':SCHEMA,'noslip_iterations':0,'noslip_tolerance':1e-6}
    if not isinstance(value,dict) or set(value)!={'schema','noslip_iterations','noslip_tolerance'}:
        raise ValueError('Solver profile requires exactly schema, noslip_iterations and noslip_tolerance')
    n=value['noslip_iterations'];tol=value['noslip_tolerance']
    if value['schema']!=SCHEMA or type(n) is not int or n<0:
        raise ValueError('Invalid explicit rigid solver profile')
    if isinstance(tol,bool) or not isinstance(tol,(float,int)) or not math.isfinite(tol) or tol<=0:
        raise ValueError('noslip_tolerance must be finite and positive')
    return {'schema':SCHEMA,'noslip_iterations':n,'noslip_tolerance':float(tol)}


def validate_solver_readback(profile,readback):
    profile=normalize_solver_profile(profile)
    if not isinstance(readback,dict):raise ValueError('Explicit solver profile lacks physical readback')
    for key in ('options_noslip_iterations','engine_noslip_iterations'):
        if type(readback.get(key)) is not int or readback[key]!=profile['noslip_iterations']:
            raise ValueError(f'{key} differs from declared solver profile')
    for key in ('options_noslip_tolerance','engine_noslip_tolerance'):
        value=readback.get(key)
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isclose(
                value,profile['noslip_tolerance'],rel_tol=2e-6,abs_tol=1e-12):
            raise ValueError(f'{key} differs from declared solver profile')
    return dict(readback)


def read_solver_readback(solver):
    return {'options_noslip_iterations':int(solver._options.noslip_iterations),
        'engine_noslip_iterations':int(solver.rigid_info.noslip_iterations[None]),
        'options_noslip_tolerance':float(solver._options.noslip_tolerance),
        'engine_noslip_tolerance':float(solver.rigid_info.noslip_tolerance[None])}


def digest(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):result.update(block)
    return result.hexdigest()


def trajectory_solver_profile(trajectory_path):
    """Hash-bound converted metadata is the only automatic nonzero authority."""
    trajectory=Path(trajectory_path).expanduser().resolve()
    metadata_path=trajectory.parent/'trajectory_metadata.json'
    if not metadata_path.is_file():
        return normalize_solver_profile(),{'legacy_default_zero':True,'metadata_path':None}
    metadata=json.loads(metadata_path.read_text())
    if metadata.get('trajectory_sha256')!=digest(trajectory):
        raise ValueError('Trajectory differs from its solver-profile metadata hash')
    profile=normalize_solver_profile(metadata.get('solver_profile'))
    return profile,{'legacy_default_zero':'solver_profile' not in metadata,
        'metadata_path':str(metadata_path),'metadata_sha256':digest(metadata_path)}


def enforce_trajectory_solver_profile(trajectory_path,kwargs):
    profile,evidence=trajectory_solver_profile(trajectory_path)
    selected=normalize_solver_profile({'schema':SCHEMA,
        'noslip_iterations':kwargs.get('noslip_iterations',profile['noslip_iterations']),
        'noslip_tolerance':kwargs.get('noslip_tolerance',profile['noslip_tolerance'])})
    if selected!=profile:
        raise ValueError('Requested solver profile differs from the frozen source trajectory')
    kwargs['noslip_iterations']=profile['noslip_iterations']
    kwargs['noslip_tolerance']=profile['noslip_tolerance']
    return profile,evidence
