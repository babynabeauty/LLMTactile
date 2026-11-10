"""CPU contracts for explicit task force limits and Stage1 transfer."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
import zipfile

import gymnasium as gym
import numpy as np
import torch
from .v1_conditioning import (LEVEL, FORCE_LIMIT, CONDITION_START, condition_features,
                              require_checkpoint_conditioning)
from .v1_migrate_force_limit import migrate_policy, verify_reference_equivalence, input_column_mapping
from .normalized_sde_policy import NormalizedSDEMultiInputPolicy
from .adaptive_policy import policy_kwargs


class ForceConditioningTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_numeric_scale_unlimited_and_ablation(self):
        np.testing.assert_allclose(condition_features(FORCE_LIMIT, [1, 2, 4], [7., 14., np.inf]),
                                   [[.7, 0], [1.4, 0], [0, 1]])
        np.testing.assert_array_equal(condition_features(FORCE_LIMIT, [3, 4], [7., np.inf], False), 0)
        for bad in [None, 0., -1., -np.inf, np.nan]:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                condition_features(FORCE_LIMIT, 3, bad)
        np.testing.assert_array_equal(condition_features(LEVEL, [3, 4]), np.eye(4)[[2, 3]])

    def test_conditioning_mismatch_rejected_even_with_valid_zip(self):
        with TemporaryDirectory() as temp:
            path = Path(temp)/'model.zip'
            with zipfile.ZipFile(path, 'w') as archive:
                archive.writestr('data', json.dumps({}))
            require_checkpoint_conditioning(path, LEVEL)
            with self.assertRaisesRegex(ValueError, 'explicitly migrate'):
                require_checkpoint_conditioning(path, FORCE_LIMIT)

    def test_online_and_training_assemble_identical_numeric_state(self):
        from online_residual import actor_state
        from .v1_env import V1VecEnv
        from .v1_task import V1Config
        from .control_protocol import V1_RATES
        q, qd, nominal = np.linspace(-.1, .1, 18), np.zeros(18), np.zeros(18)
        future, command = np.full(18, .01), np.full(18, .02)
        for force in [7., np.inf]:
            fake = SimpleNamespace(num_envs=1, indices=np.arange(18), rates=V1_RATES,
                robot=SimpleNamespace(get_dofs_position=lambda _:q[None],get_dofs_velocity=lambda _:qd[None]),
                lane_tick=np.array([900]), lane_start=np.array([300]), lane_base=[0],
                _targets=lambda t:nominal[None] if t[0]==3 else future[None],
                bases=[SimpleNamespace(trajectory=SimpleNamespace(phase_at_tick=lambda _:'lift'))],
                astra_level=np.array([3]), use_level=True, actor_conditioning=FORCE_LIMIT,
                f_break=np.array([force]), mid=np.zeros(18),half=np.ones(18),command=command[None],
                gains=np.array([1.2]),finger_mask=np.array([1,1,1,0,0]),residual=np.zeros((1,12)),config=V1Config())
            expected=V1VecEnv._state(fake)[0]
            actual=actor_state(q,qd,command,nominal,future,-np.ones(18),np.ones(18),1.2,'lift',2.,3,True,
                fake.finger_mask,np.zeros(12),.15,actor_conditioning=FORCE_LIMIT,force_limit_N=force)
            self.assertEqual(actual.shape,(98,))
            np.testing.assert_array_equal(actual,expected)

    def test_policy_and_adam_transfer_preserve_reference_behavior(self):
        for joints in [6,12]:
            size=88+joints
            obs=gym.spaces.Dict({'state':gym.spaces.Box(-np.inf,np.inf,(12,size),np.float32),
                                'tactile':gym.spaces.Box(-np.inf,np.inf,(12,5,120,3),np.float32)})
            policy=NormalizedSDEMultiInputPolicy(obs,gym.spaces.Box(-1.,1.,(joints,),np.float32),
                lambda _:3e-4,use_sde=True,**policy_kwargs())
            # Populate actual Adam state before migration, without a simulation run.
            for param in policy.parameters():param.grad=torch.full_like(param,.001)
            policy.optimizer.step();policy.optimizer.zero_grad()
            migrated=migrate_policy(policy,7.)
            verify_reference_equivalence(policy,migrated,7.)
            self.assertEqual(migrated.observation_space['state'].shape,(12,size-2))
            old=dict(policy.named_parameters());new=dict(migrated.named_parameters())
            columns,scale=input_column_mapping(size,7.)
            scale=torch.tensor(scale)
            for name,p in old.items():
                a=policy.optimizer.state[p];b=migrated.optimizer.state[new[name]]
                torch.testing.assert_close(a['step'],b['step'],rtol=0,atol=0)
                if name.endswith('state_encoder.0.weight'):
                    torch.testing.assert_close(b['exp_avg'],a['exp_avg'][:,columns]/scale)
                    torch.testing.assert_close(b['exp_avg_sq'],a['exp_avg_sq'][:,columns]/scale**2)
                else:
                    torch.testing.assert_close(p,new[name],rtol=0,atol=0)
                    torch.testing.assert_close(a['exp_avg'],b['exp_avg'],rtol=0,atol=0)
            # Transferred optimizer accepts another step with the resized layers.
            for p in migrated.parameters():p.grad=torch.full_like(p,.001)
            migrated.optimizer.step()


if __name__=='__main__':unittest.main()
