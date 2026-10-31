"""CPU provenance checks; no Genesis scene is constructed."""
import json
from types import SimpleNamespace
import pytest
from .solver_profile import (SCHEMA, normalize_solver_profile, validate_solver_readback,
    read_solver_readback, trajectory_solver_profile, enforce_trajectory_solver_profile, digest)


def profile(n=5):return {'schema':SCHEMA,'noslip_iterations':n,'noslip_tolerance':1e-6}
def readback(n=5):return {'options_noslip_iterations':n,'engine_noslip_iterations':n,
    'options_noslip_tolerance':1e-6,'engine_noslip_tolerance':9.999999974752427e-7}


def test_normalized_profile_and_actual_engine_readback():
    assert normalize_solver_profile()==profile(0)
    actual=readback()
    fake=SimpleNamespace(_options=SimpleNamespace(noslip_iterations=5,noslip_tolerance=1e-6),
        rigid_info=SimpleNamespace(noslip_iterations={None:5},noslip_tolerance={None:actual['engine_noslip_tolerance']}))
    assert validate_solver_readback(profile(),read_solver_readback(fake))==actual
    with pytest.raises(ValueError):validate_solver_readback(profile(),readback(0))


@pytest.mark.parametrize('bad',[{},dict(profile(),schema='other'),dict(profile(),noslip_iterations=True),
    dict(profile(),noslip_iterations=-1),dict(profile(),noslip_tolerance=0),dict(profile(),noslip_tolerance=float('nan'))])
def test_invalid_profiles_rejected(bad):
    with pytest.raises(ValueError):normalize_solver_profile(bad)


def test_frozen_metadata_is_only_nonzero_authority(tmp_path):
    path=tmp_path/'control.jsonl';path.write_text('immutable commands')
    assert trajectory_solver_profile(path)[0]==profile(0)
    with pytest.raises(ValueError):enforce_trajectory_solver_profile(path,{'noslip_iterations':5})
    meta=tmp_path/'trajectory_metadata.json'
    meta.write_text(json.dumps({'trajectory_sha256':digest(path),'solver_profile':profile()}))
    kwargs={};actual,evidence=enforce_trajectory_solver_profile(path,kwargs)
    assert actual==profile() and kwargs=={'noslip_iterations':5,'noslip_tolerance':1e-6}
    assert not evidence['legacy_default_zero']
    with pytest.raises(ValueError):enforce_trajectory_solver_profile(path,{'noslip_iterations':0})
    path.write_text('changed commands')
    with pytest.raises(ValueError,match='metadata hash'):trajectory_solver_profile(path)


def test_uniform_experiment_rejects_mixed_profiles(tmp_path):
    from .multiobject_train import load_experiment
    objects=[]
    for shape,n in [('cube',5),('cuboid',0)]:
        directory=tmp_path/shape;directory.mkdir();trajectory=directory/'control.jsonl';trajectory.write_text(shape)
        (directory/'trajectory_metadata.json').write_text(json.dumps({'trajectory_sha256':digest(trajectory),'solver_profile':profile(n)}))
        objects.append({'object_id':shape,'trajectory_path':str(trajectory),'trajectory_sha256':digest(trajectory)})
    protocol={'objects':objects,'solver_profile':profile(),'splits':{split:[{'condition_id':split,'mass_kg':mass,'friction':.8}]
        for split,mass in [('train',.05),('validation',.1),('test',.2)]}}
    path=tmp_path/'protocol.json';path.write_text(json.dumps(protocol))
    with pytest.raises(ValueError,match='uniform experiment profile'):load_experiment(path)
    protocol['objects']=objects[:1];path.write_text(json.dumps(protocol))
    assert load_experiment(path)['solver_profile']==profile()
    del protocol['solver_profile'];path.write_text(json.dumps(protocol))
    with pytest.raises(ValueError,match='uniform experiment profile'):load_experiment(path)
