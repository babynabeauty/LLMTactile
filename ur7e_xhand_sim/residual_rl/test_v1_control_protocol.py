"""CPU regressions for the 60 Hz protocol, without building a Genesis scene."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import zipfile

import numpy as np

from .control_protocol import (V1_RATES, HISTORY_SAMPLES, current_control_protocol,
    elapsed_feature, lookahead_feature, migrate_experiment, require_checkpoint_control_compatibility)
from .core import RateConfig, Trajectory
from .v1_task import V1Config, limit_residual


class V1ControlProtocolTest(unittest.TestCase):
    def test_tactile_and_action_tick_are_identical_and_legacy_is_untouched(self):
        self.assertEqual(V1_RATES.action_stride, 5)
        self.assertEqual(V1_RATES.action_stride, V1_RATES.tactile_stride)
        self.assertEqual(V1_RATES.samples_per_action, 1)
        self.assertEqual(HISTORY_SAMPLES/V1_RATES.tactile_hz, .2)
        self.assertEqual(RateConfig().action_hz, 15)

    def test_slew_speed_and_slew_penalty_keep_physical_scale(self):
        cfg = V1Config()
        self.assertAlmostEqual(cfg.residual_rate_rad*V1_RATES.action_hz, .75)
        residual = np.zeros(1)
        for _ in range(6):
            residual = limit_residual(np.ones(1), residual, cfg)
        np.testing.assert_allclose(residual, [.075])  # 100 ms at the same .75 rad/s
        old_cost = 15*.005*(.075/15/.05)**2
        new_cost = 60*cfg.residual_change_weight*(.075/60/cfg.residual_rate_rad)**2
        self.assertAlmostEqual(old_cost, new_cost)

    def test_legacy_experiment_migration_preserves_time_and_is_idempotent(self):
        original = dict(training=dict(n_steps=128, batch_size=256, sde_sample_freq=30,
            checkpoint_every=65536, gamma=.99, gae_lambda=.95),
            config=dict(residual_rate_rad=.05, residual_change_weight=.005),
            stages=[dict(name='stage1', requested_timesteps=655360)])
        changed = migrate_experiment(original)
        self.assertNotIn('control_protocol', original)
        self.assertEqual(changed['training']['n_steps']/60, original['training']['n_steps']/15)
        self.assertEqual(changed['training']['sde_sample_freq']/60, 2)
        self.assertEqual(changed['stages'][0]['requested_timesteps']/60, 655360/15)
        self.assertAlmostEqual(changed['training']['gamma']**60, .99**15)
        self.assertAlmostEqual(changed['training']['gae_lambda']**60, .95**15)
        self.assertEqual(changed['config']['residual_rate_rad'], .0125)
        self.assertEqual(migrate_experiment(changed), changed)
        changed['control_protocol']['action_hz'] = 15
        with self.assertRaisesRegex(ValueError, 'incompatible'):
            migrate_experiment(changed)

    def test_checkpoint_rejects_missing_or_legacy_protocol_and_accepts_current(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'model.zip'
            legacy = dict(current_control_protocol(), action_hz=15)
            for protocol in (None, legacy, current_control_protocol()):
                data = {} if protocol is None else dict(control_protocol=protocol)
                with zipfile.ZipFile(path, 'w') as archive:
                    archive.writestr('data', json.dumps(data))
                if protocol == current_control_protocol():
                    self.assertEqual(require_checkpoint_control_compatibility(path), data)
                else:
                    with self.assertRaisesRegex(ValueError, 'Checkpoint control protocol'):
                        require_checkpoint_control_compatibility(path)

    def test_elapsed_actor_feature_does_not_need_future_duration(self):
        np.testing.assert_allclose(elapsed_feature([0, 1, 5, 10, 20]), [0, .1, .5, 1, 1])

    def test_lookahead_feature_keeps_its_scale_at_any_action_rate(self):
        # Fastest base closure: .012 rad per 60 Hz tick, four ticks per legacy 15 Hz action.
        np.testing.assert_allclose(lookahead_feature(np.full(3, .012), 0), .48)
        np.testing.assert_allclose(lookahead_feature(np.full(3, .048), 0, 1/15), .48)

    def test_training_saves_online_contract_in_actual_sb3_checkpoint(self):
        import gymnasium as gym
        import torch
        from stable_baselines3.common.vec_env import DummyVecEnv
        from .v1_train import make_model
        from .v1_bases import TRAINING_DEFAULTS

        class ActorEnv(gym.Env):
            def __init__(self):
                self.config = V1Config()
                self.action_joints = np.arange(6)
                self.action_space = gym.spaces.Box(-1., 1., (6,), np.float32)
                self.observation_space = gym.spaces.Dict({
                    'tactile': gym.spaces.Box(-np.inf, np.inf, (12, 5, 120, 3), np.float32),
                    'state': gym.spaces.Box(-np.inf, np.inf, (12, 94), np.float32)})

        threads = torch.get_num_threads()
        torch.set_num_threads(2)
        env = DummyVecEnv([ActorEnv])
        try:
            model = make_model(env, dict(TRAINING_DEFAULTS, n_steps=2, batch_size=2), 1, 'tactile')
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory)/'model.zip'
                model.save(path)
                data = require_checkpoint_control_compatibility(path)
                self.assertEqual(data['action_joints'], list(range(6)))
                self.assertEqual(data['v1_config']['residual_rate_rad'], .0125)
                self.assertEqual(data['method'], 'tactile')
                self.assertTrue(data['use_level'])
        finally:
            env.close()
            torch.set_num_threads(threads)

    def test_every_recorded_command_and_gain_are_held_on_integer_ticks(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'trajectory.jsonl'
            rows, t = [], 0.
            for i in range(600):
                rows.append(dict(before=dict(sim_time_s=t), after=dict(sim_time_s=t+1/60),
                    sent_target_rad=[float(i)], grip_gain=1+i/600, phase=str(i)))
                t += 1/60  # accumulated timestamps deliberately contain rounding error
            path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
            trajectory = Trajectory(path)
        ticks = np.arange(3000)
        indices = ticks//5
        np.testing.assert_array_equal(trajectory.target_at_tick(ticks)[:, 0], indices)
        np.testing.assert_allclose(trajectory.grip_gain_at_tick(ticks), 1+indices/600)
        np.testing.assert_array_equal(trajectory.phase_at_tick(ticks), indices.astype(str))
        with self.assertRaisesRegex(ValueError, 'integer'):
            trajectory.target_at_tick(.5)

    def test_prefixes_capture_each_lane_at_its_own_takeover(self):
        from .v1_env import V1VecEnv
        clock = SimpleNamespace(tick=0)
        command_ticks = []
        robot = SimpleNamespace(
            set_dofs_position=lambda *a, **k: None,
            get_dofs_position=lambda *a: np.full((2, 18), clock.tick),
            control_dofs_position=lambda *a: command_ticks.append(clock.tick))
        cube = SimpleNamespace(set_pos=lambda *a, **k: None, set_quat=lambda *a, **k: None,
            get_pos=lambda: np.full((2, 3), clock.tick), get_quat=lambda: np.zeros((2, 4)))
        env = SimpleNamespace(num_envs=2, robot=robot, cube=cube, home=np.zeros(18), indices=list(range(18)),
            palm=SimpleNamespace(get_quat=lambda: np.zeros((2, 4))), rates=V1_RATES,
            scene=SimpleNamespace(step=lambda: setattr(clock, 'tick', clock.tick+1)),
            _set_materials=lambda *a: None, _set_gains=lambda *a: None,
            _on_prefix_step=lambda *a: None, region_normal_force=lambda: np.zeros((2, 7)))
        trajectory = SimpleNamespace(grip_gain_at_tick=lambda tick: 1., target_at_tick=lambda tick: np.zeros(18))
        jobs = [(SimpleNamespace(takeover_tick=t, trajectory=trajectory), np.zeros(3), np.zeros(4))
                for t in (5, 10)]
        records = V1VecEnv._replay_prefixes(env, jobs)
        np.testing.assert_array_equal(records[0]['q'], 5)
        np.testing.assert_array_equal(records[1]['q'], 10)
        self.assertEqual(command_ticks, [0, 5])


if __name__ == '__main__':
    unittest.main()
