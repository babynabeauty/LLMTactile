"""CPU contracts for source gain timing and checkpoint/application consistency."""
import ast
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from .core import RateConfig,Trajectory


def write_trajectory(directory,gains=(1.,2.,3.)):
    rows=[]
    for i,gain in enumerate(gains):
        row=dict(before={'sim_time_s':i*.05},after={'sim_time_s':(i+1)*.05},
            sent_target_rad=[0.]*18,phase='close')
        if gain is not None:row['grip_gain']=gain
        rows.append(row)
    path=Path(directory)/'control.jsonl'
    path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    return path


def environment_class():
    path=Path(__file__).with_name('env.py')
    node=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='ResidualVecEnv')
    keep={'_apply_gains','_source_prefix_gain_tick','_prefix_physics_readbacks',
          '_assert_prefix_gain_readbacks','reset'}
    node.bases=[]
    node.body=[n for n in node.body if isinstance(n,ast.FunctionDef) and n.name in keep]
    ns={'np':np,'cpu':np.asarray}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(path),'exec'),ns)
    return ns['ResidualVecEnv']


Env=environment_class()


class TestSourceGains(unittest.TestCase):
    def test_piecewise_constant_gain_boundary_and_legacy_default(self):
        with tempfile.TemporaryDirectory() as d:
            trajectory=Trajectory(write_trajectory(d))
            np.testing.assert_array_equal(trajectory.grip_gain_at([0,.049999,.05,.099999,.1,.15]),[1,1,2,2,3,3])
            self.assertTrue(trajectory.explicit_grip_gain.all())
            legacy=Trajectory(write_trajectory(d,(None,None,None)))
            np.testing.assert_array_equal(legacy.command_grip_gains,[1,1,1])
            self.assertFalse(legacy.explicit_grip_gain.any())
            with self.assertRaises(ValueError):trajectory.grip_gain_at(np.nan)

    def test_invalid_recorded_gain_rejected(self):
        for value in (0.,-1.,3.1,float('nan'),float('inf'),True,'2',[2,2]):
            with self.subTest(value=value),tempfile.TemporaryDirectory() as d:
                with self.assertRaises(ValueError):Trajectory(write_trajectory(d,(value,)))

    def fixture(self,trajectory):
        env=Env();env.num_envs=2;env.indices=list(range(18));env.rates=RateConfig()
        env.base_kp=np.r_[[4500,4500,3500,2000,2000,2000],[3.]*12].astype(np.float32)
        env.base_kv=np.r_[[450,450,350,200,200,200],[.08]*12].astype(np.float32)
        env.gains=np.ones((2,2),np.float32);env.mass=np.array([.05,.2],np.float32)
        env.effort=np.ones(18,np.float32)*.7
        env.inertia_readback_kg_m2_per_lane=env.mass[:,None,None]*np.eye(3)[None]*.0003
        class Robot:
            calls=0
            def set_dofs_kp(self,value,indices):self.kp=np.array(value);self.calls+=1
            def set_dofs_kv(self,value,indices):self.kv=np.array(value);self.calls+=1
            def get_dofs_kp(self,indices):return self.kp
            def get_dofs_kv(self,indices):return self.kv
            def get_dofs_force_range(self,indices):return -env.effort[None],env.effort[None]
        env.robot=Robot()
        env.cube=SimpleNamespace(get_mass=lambda:env.mass[:,None],
            get_links_inertia=lambda:env.inertia_readback_kg_m2_per_lane[:,None])
        env.trajectory=trajectory;env.replay_source_prefix_gains=True;env.prefix_gain_events=[]
        env._apply_gains()
        return env

    def test_physics_grid_schedule_is_absolute_and_readback_audited(self):
        with tempfile.TemporaryDirectory() as d:
            env=self.fixture(Trajectory(write_trajectory(d)))
            for tick in range(31):env._source_prefix_gain_tick(tick)
            self.assertEqual([e['physics_tick'] for e in env.prefix_gain_events],[0,15,30])
            np.testing.assert_array_equal(env.gains,np.ones((2,2))*3)
            np.testing.assert_allclose(env.robot.kp[:,6:12],9)
            np.testing.assert_allclose(env.robot.kv[:,6:12],.08*np.sqrt(3))
            np.testing.assert_array_equal(env.robot.kp[:,:6],np.broadcast_to(env.base_kp[:6],(2,6)))
            broken=env._prefix_physics_readbacks();broken['mass'][0]+=.01
            with self.assertRaises(AssertionError):env._assert_prefix_gain_readbacks(broken)
            broken=env._prefix_physics_readbacks();broken['kp'][0,6]*=2
            with self.assertRaises(AssertionError):env._assert_prefix_gain_readbacks(broken)

    def test_reset_restores_actual_prefix_gain_without_pd_writes(self):
        with tempfile.TemporaryDirectory() as d:
            env=self.fixture(Trajectory(write_trajectory(d)))
            env._source_prefix_gain_tick(30)
            env.prefix_gains=env.gains.copy();env.prefix_readbacks=env._prefix_physics_readbacks()
            env.prefix_checkpoint={'kp':env.robot.kp.copy(),'kv':env.robot.kv.copy()}
            env.scene=SimpleNamespace(__setstate__=lambda checkpoint:(
                setattr(env.robot,'kp',checkpoint['kp'].copy()),setattr(env.robot,'kv',checkpoint['kv'].copy())))
            env.prefix_history=np.arange(2*12*5*120*3,dtype=np.float32).reshape(2,12,5,120,3)
            env.history=np.zeros_like(env.prefix_history);env.start_tick=31;env.episode_counter=0
            env._obs=lambda:{'gain':env.gains.copy(),'tactile':env.history.copy()}
            env.gains[:]=.4;env._apply_gains();before=env.robot.calls
            obs=env.reset()
            self.assertEqual(env.robot.calls,before)
            np.testing.assert_array_equal(obs['gain'],np.ones((2,2))*3)
            np.testing.assert_array_equal(obs['tactile'],env.prefix_history)
            np.testing.assert_array_equal(env.robot.kp,env.prefix_readbacks['kp'])


if __name__=='__main__':unittest.main()
