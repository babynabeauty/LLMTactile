"""CPU contracts for shared-shape rollout and frozen paired model setup."""
import json
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch
from stable_baselines3.common.vec_env import VecEnv

from .multiobject_train import (ComposedShapeVecEnv, load_experiment, make_model,
                                weight_digest, validate_stage_resume, validate_completed_run)
from .multiobject_benchmark import model_specs, controller_metadata, read_selection
from .train import digest


class FakeScene(VecEnv):
    def __init__(self, period=2, lanes=2):
        self.render_mode=None
        self.config=SimpleNamespace(gain_min=.6,gain_max=3.)
        self.period=period;self.tick=0;self.closes=0;self._closed=False
        super().__init__(lanes,gym.spaces.Dict({
            'state':gym.spaces.Box(-np.inf,np.inf,(12,79),np.float32),
            'tactile':gym.spaces.Box(-np.inf,np.inf,(12,5,120,3),np.float32)}),
            gym.spaces.Box(-1,1,(1,),np.float32))
    def obs(self):
        return {k:np.full((self.num_envs,*s.shape),self.tick,np.float32)
                for k,s in self.observation_space.spaces.items()}
    def reset(self):self.tick=0;return self.obs()
    def step_async(self,actions):self.actions=np.asarray(actions).copy()
    def step_wait(self):
        self.tick+=1;obs=self.obs();done=self.tick==self.period
        infos=[{} for _ in range(self.num_envs)]
        if done:
            for lane,info in enumerate(infos):
                info.update(episode_metrics={'task_success':True},
                    terminal_observation={k:v[lane].copy() for k,v in obs.items()})
            obs=self.reset()
        return obs,np.ones(self.num_envs,np.float32),np.full(self.num_envs,done),infos
    def close(self):self.closes+=1;self._closed=True
    def get_attr(self,name,indices=None):return [getattr(self,name) for _ in self._get_indices(indices)]
    def set_attr(self,name,value,indices=None):setattr(self,name,value)
    def env_method(self,name,*args,indices=None,**kwargs):raise NotImplementedError
    def env_is_wrapped(self,wrapper_class,indices=None):return [False for _ in self._get_indices(indices)]
    def action_for_gain(self,gain):return 2*np.log(np.asarray(gain)/.6)/np.log(5)-1


def compose():
    return ComposedShapeVecEnv([FakeScene(2),FakeScene(3)],['cube','sphere'],
        [{'condition_id':'a'},{'condition_id':'b'}],training_seed=1,method='tactile',split='train')


def test_shapes_reset_independently_with_terminal_observations():
    env=compose();env.reset()
    env.step(np.zeros((4,1)))
    obs,_,done,infos=env.step(np.array([[.1],[.2],[.3],[.4]],np.float32))
    assert done.tolist()==[True,True,False,False]
    assert np.all(obs['state'][:2]==0) and np.all(obs['state'][2:]==2)
    assert np.all(infos[0]['terminal_observation']['state']==2)
    assert infos[1]['episode_metrics']['condition_id']=='b'
    assert infos[1]['episode_metrics']['object_id']=='cube'
    assert env.envs[1].actions[:,0].tolist()==pytest.approx([.3,.4])
    obs,_,done,infos=env.step(np.zeros((4,1)))
    assert done.tolist()==[False,False,True,True]
    assert infos[2]['episode_metrics']['object_id']=='sphere'
    assert set(obs)=={'state','tactile'}


def test_close_destroys_process_runtime_once():
    env=compose();env.close();env.close()
    assert [e.closes for e in env.envs]==[1,0]
    assert all(e._closed for e in env.envs)


def test_composed_attribute_lane_mapping():
    env=compose()
    assert env.get_attr('period')==[2,2,3,3]
    assert env.get_attr('period',[3,0])==[3,2]
    with pytest.raises(ValueError):env.step_async(np.zeros((2,1)))


def test_modalities_have_identical_initial_parameters():
    torch.set_num_threads(2)
    settings={'n_steps':2,'batch_size':4,'n_epochs':1,'initial_gain':2.75}
    tactile,_=make_model(compose(),settings,3,'tactile')
    blind,_=make_model(compose(),settings,3,'no_tactile')
    assert weight_digest(tactile)==weight_digest(blind)
    assert tactile.policy.features_extractor.use_tactile
    assert not blind.policy.features_extractor.use_tactile
    assert tactile.policy.share_features_extractor
    assert tactile.gamma==1 and tactile.gae_lambda==.95
    assert controller_metadata('seed3_no_tactile',blind)==(3,'no_tactile')
    with pytest.raises(ValueError):controller_metadata('seed3_tactile',blind)


def test_ppo_collects_mixed_episode_lengths_and_resumes(tmp_path):
    torch.set_num_threads(2)
    settings={'n_steps':2,'batch_size':4,'n_epochs':1,'initial_gain':2.75}
    model,_=make_model(compose(),settings,1,'tactile')
    model.learn(total_timesteps=8)
    assert model.num_timesteps==8
    path=tmp_path/'model.zip';model.save(path)
    resumed,_=make_model(compose(),settings,1,'tactile',resume=path)
    assert resumed.num_timesteps==8
    assert weight_digest(resumed)==weight_digest(model)


def test_cpu_optimizer_with_non_cpu_default_device():
    settings={'n_steps':2,'batch_size':4,'n_epochs':1,'initial_gain':2.75}
    env=compose()
    with torch.device('meta'):
        model,_=make_model(env,settings,1,'tactile')
        model.learn(total_timesteps=8)
    devices={str(tensor.device) for state in model.policy.optimizer.state.values()
             for tensor in state.values() if isinstance(tensor,torch.Tensor)}
    assert devices=={'cpu'}


def test_material_splits_cannot_overlap(tmp_path):
    protocol={'objects':[{'object_id':'cube','trajectory_path':None}], 'splits':{
        'train':[{'condition_id':'t','mass_kg':.05,'friction':.8}],
        'validation':[{'condition_id':'v','mass_kg':.1,'friction':.6}],
        'test':[{'condition_id':'x','mass_kg':.05,'friction':.8}]}}
    file=tmp_path/'experiment.json';file.write_text(json.dumps(protocol))
    with pytest.raises(ValueError,match='disjoint'):load_experiment(file,require_trajectories=False)
    protocol['splits']['test'][0]['mass_kg']=.2
    file.write_text(json.dumps(protocol))
    assert load_experiment(file,require_trajectories=False)['objects'][0]['object_id']=='cube'
    with pytest.raises(ValueError,match='verified demonstration'):load_experiment(file)


def test_model_listing_uses_final_frozen_stage():
    protocol={'training':{'stages':[{'name':'stage1'},{'name':'stage2'}],
                         'seeds':[1,2,3],'methods':['tactile','no_tactile']}}
    specs=model_specs(protocol,Path('/frozen'))
    assert len(specs)==6
    assert specs[-1]==('seed3_no_tactile',Path('/frozen/seed3_no_tactile_stage2/final_model.zip'))
    with pytest.raises(ValueError):model_specs(protocol,requested=['a=/x','a=/y'])


def completed_stage(tmp_path):
    from .multiobject_protocol import make_protocol
    protocol=make_protocol();path=tmp_path/'protocol.json';path.write_text(json.dumps(protocol))
    stage=protocol['training']['stages'][0];folder=tmp_path/'stage1';folder.mkdir()
    checkpoint=folder/'final_model.zip';checkpoint.write_bytes(b'fake-checkpoint-for-provenance-unit-test')
    manifest={'experiment_sha256':digest(path),'stage':stage,'seed':1,'method':'tactile',
        'diagnostic_max_conditions':None,'requested_timesteps':stage['requested_timesteps'],
        'actual_model_config':{'n_envs':36},'input_checkpoint':None}
    status={'status':'completed','final_checkpoint_sha256':digest(checkpoint),
        'requested_additional_timesteps':stage['requested_timesteps'],'actual_additional_timesteps':133632}
    (folder/'manifest.json').write_text(json.dumps(manifest));(folder/'status.json').write_text(json.dumps(status))
    args=SimpleNamespace(stage='stage2',resume=checkpoint,seed=1,method='tactile',experiment_json=path)
    return protocol,path,folder,args


def test_resume_requires_complete_paired_preceding_stage(tmp_path):
    protocol,path,folder,args=completed_stage(tmp_path)
    validate_stage_resume(args,protocol)
    args.stage='stage1'
    with pytest.raises(ValueError,match='fresh'):validate_stage_resume(args,protocol)
    args.stage='stage2';args.resume=None
    with pytest.raises(ValueError,match='must resume'):validate_stage_resume(args,protocol)
    args.resume=folder/'final_model.zip'
    manifest=json.loads((folder/'manifest.json').read_text());manifest['diagnostic_max_conditions']=2
    (folder/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='provenance'):validate_stage_resume(args,protocol)


def test_completed_suite_entry_cannot_skip_corrupt_checkpoint(tmp_path):
    protocol,path,folder,args=completed_stage(tmp_path)
    (folder/'final_model.zip').write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='checkpoint hash'):
        validate_completed_run(folder,protocol,protocol['training']['stages'][0],1,'tactile',digest(path))


def test_selection_is_bound_to_validation_and_frozen_protocol(tmp_path):
    protocol,path,folder,args=completed_stage(tmp_path)
    selected={'selection_split':'validation','test_outcomes_read':False,'protocol_sha256':digest(path),
        'selected_models':{'seed1_tactile':{'path':str(args.resume),'sha256':digest(args.resume),'checkpoint_stage':'stage1'}}}
    file=tmp_path/'selection.json';file.write_text(json.dumps(selected))
    assert set(read_selection(file,protocol,path))=={'seed1_tactile'}
    for field,value in [('selection_split','test'),('test_outcomes_read',True),('protocol_sha256','wrong')]:
        invalid={**selected,field:value};file.write_text(json.dumps(invalid))
        with pytest.raises(ValueError):read_selection(file,protocol,path)
