"""CPU checks for explicit weight transfer followed by exact in-run resume."""
import json
from pathlib import Path
import tempfile
import unittest

import torch
from stable_baselines3.common.vec_env import DummyVecEnv

from .test_v1_exploration_std import ActorEnv
from .tactile_provenance import current_tactile_model
from .v1_bases import TRAINING_DEFAULTS, sha256
from .v1_train import make_model as native_model, weight_digest
from .v1_balanced_transfer_train import make_model
from .v15_train import assert_state_equal


class TransferTest(unittest.TestCase):
    def test_transfer_keeps_weights_resets_adam_and_resume_keeps_both(self):
        old_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        env = DummyVecEnv([lambda: ActorEnv(range(6))])
        try:
            settings = dict(TRAINING_DEFAULTS, n_steps=2, batch_size=2)
            source = native_model(env, settings, 1, 'tactile')
            source.policy.optimizer.zero_grad()
            sum(p.square().sum() for p in source.policy.parameters()).backward()
            source.policy.optimizer.step()
            source.num_timesteps = 1234
            profiles = [dict(level=3, mass_kg=.1, friction=.45, f_break_N=4.)]
            with tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                (directory/'manifest.json').write_text(json.dumps({'tactile_model': current_tactile_model()}))
                checkpoint = directory/'source.zip'
                source.save(checkpoint)
                init = dict(kind='compatible_v1_weights_fresh_adam', checkpoint=str(checkpoint),
                            checkpoint_sha256=sha256(checkpoint), source_training_timesteps=1234)
                lower_lr = dict(settings, learning_rate=1e-4)
                target = make_model(env, lower_lr, 1, 'tactile', output=directory, profiles=profiles, initialization=init)
                self.assertEqual(weight_digest(source), weight_digest(target))
                self.assertEqual(target.num_timesteps, 0)
                self.assertFalse(target.policy.optimizer.state)
                self.assertEqual(target.lr_schedule(1.), 1e-4)
                target.policy.optimizer.zero_grad()
                sum(p.square().sum() for p in target.policy.parameters()).backward()
                target.policy.optimizer.step()
                target.num_timesteps = 24
                target.save(directory/'target.zip')
                resumed = make_model(env, lower_lr, 1, 'tactile', resume=directory/'target.zip',
                                     output=directory, profiles=profiles, initialization=init)
                self.assertEqual(resumed.num_timesteps, 24)
                assert_state_equal(target.policy.state_dict(), resumed.policy.state_dict())
                assert_state_equal(target.policy.optimizer.state_dict(), resumed.policy.optimizer.state_dict())
                with self.assertRaisesRegex(ValueError, 'hash changed'):
                    make_model(env, lower_lr, 1, 'tactile', output=directory, profiles=profiles,
                               initialization=dict(init, checkpoint_sha256='wrong'))
                with self.assertRaisesRegex(ValueError, 'counter'):
                    make_model(env, lower_lr, 1, 'tactile', output=directory, profiles=profiles,
                               initialization=dict(init, source_training_timesteps=9999))
        finally:
            env.close()
            torch.set_num_threads(old_threads)


if __name__ == '__main__':
    unittest.main()
