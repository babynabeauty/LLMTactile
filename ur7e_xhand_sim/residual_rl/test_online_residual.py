"""Online actor contracts without building a Genesis scene or calling Astra."""
from types import SimpleNamespace
import unittest
import numpy as np

from online_residual import OnlineResidual, actor_state, strict_online_metrics
from residual_rl.control_protocol import V1_RATES, current_control_protocol
from residual_rl.v1_task import V1Config, FINGERS, PHASES


class FakeSim:
    control_dt_s = 1/60
    steps = 10
    names = [f'j{i}' for i in range(18)]
    lower, upper = -np.ones(18), np.ones(18)*2
    target = np.zeros(18)
    array = staticmethod(np.asarray)

    def __init__(self):
        self.q = np.linspace(-.2, .2, 18)
        self.qd = np.linspace(-1, 1, 18)
        self.sensors = {f: SimpleNamespace(read=lambda i=i: SimpleNamespace(
            force=np.full((120, 3), i+1))) for i, f in enumerate(FINGERS)}

    def observe(self, full=False):
        return dict(joint_positions_rad=dict(zip(self.names, self.q)),
                    joint_velocities_rad_s=dict(zip(self.names, self.qd)))


class FakeModel:
    action_space = SimpleNamespace(shape=(12,))
    observation_space = SimpleNamespace(spaces={
        'tactile': SimpleNamespace(shape=(12, 5, 120, 3)), 'state': SimpleNamespace(shape=(12, 100))})

    def predict(self, observation, deterministic):
        self.observation = observation
        return np.ones((1, 12)), None


class OnlineResidualTest(unittest.TestCase):
    def test_actor_row_matches_training_state(self):
        # Import the real environment, but call its pure state assembler on a fake.
        from residual_rl.v1_env import V1VecEnv
        s = FakeSim()
        cfg = V1Config()
        nominal, future = np.arange(18)*.001, np.arange(18)*.003
        command, residual = nominal+.01, np.arange(12)*.002
        fake = SimpleNamespace(num_envs=1, indices=np.arange(18), rates=V1_RATES,
            robot=SimpleNamespace(get_dofs_position=lambda _: s.q[None], get_dofs_velocity=lambda _: s.qd[None]),
            lane_tick=np.array([900]), lane_start=np.array([300]), lane_base=[0],
            _targets=lambda t: nominal[None] if t[0] == 3 else future[None],
            bases=[SimpleNamespace(trajectory=SimpleNamespace(phase_at_tick=lambda _: 'lift'))],
            astra_level=np.array([2]), use_level=True, mid=(s.lower+s.upper)/2,
            half=(s.upper-s.lower)/2, command=command[None], gains=np.array([1.4]),
            finger_mask=np.array([1, 1, 1, 0, 0]), residual=residual[None], config=cfg)
        expected = V1VecEnv._state(fake)[0]
        actual = actor_state(s.q, s.qd, command, nominal, future, s.lower, s.upper,
            1.4, 'lift', 2, 2, True, fake.finger_mask, residual, cfg.residual_bound_rad)
        np.testing.assert_array_equal(actual, expected)

    def test_tactile_history_slew_mask_and_no_residual_accumulation(self):
        sim, model = FakeSim(), FakeModel()
        adapter = OnlineResidual(sim, 'pinch', 1.2, model=model, metadata={
            'control_protocol': current_control_protocol(), 'action_joints': list(range(12))})
        nominal, future = np.zeros(18), np.ones(18)*.001
        np.testing.assert_array_equal(adapter.apply(nominal, future, 'align'), nominal)
        self.assertFalse(adapter.started)
        first = adapter.apply(nominal, future, 'close')
        self.assertEqual(adapter.inferences, 1)
        np.testing.assert_array_equal(model.observation['tactile'], 0)
        np.testing.assert_array_equal(model.observation['state'][0, 0], model.observation['state'][0, -1])
        np.testing.assert_allclose(first[6:12], .0125)
        np.testing.assert_array_equal(first[12:], 0)
        adapter.after_step()
        sim.steps += 1
        second = adapter.apply(nominal, nominal, 'close')
        np.testing.assert_allclose(second[6:12], .025)
        np.testing.assert_array_equal(second[12:], 0)
        np.testing.assert_array_equal(model.observation['tactile'][0, -1, :, 0, 0], [1, 2, 3, 4, 5])
        np.testing.assert_array_equal(model.observation['tactile'][0, :-1], 0)
        # Next-target delta vanishes when waiting for Astra, but elapsed time advances.
        np.testing.assert_array_equal(model.observation['state'][0, -1, 54:72], 0)
        self.assertAlmostEqual(float(model.observation['state'][0, -1, 78]), 1/600)

    def test_requires_explicit_zero_or_versioned_actor(self):
        with self.assertRaises(ValueError):
            OnlineResidual(FakeSim(), 'pinch', 1)
        with self.assertRaises(ValueError):
            OnlineResidual(FakeSim(), 'pinch', 1, model=FakeModel(), metadata={})
        sim = FakeSim()
        sim.control_dt_s = 1/15
        with self.assertRaises(ValueError):
            OnlineResidual(sim, 'pinch', 1, zero=True)


class ResumableSkillTest(unittest.TestCase):
    def make_executor(self, directory, online):
        from pathlib import Path
        from base_policy_v1 import BasePolicyV1
        b = BasePolicyV1.__new__(BasePolicyV1)
        b.sim = SimpleNamespace(steps=5, episode=1, render_enabled=False,
            target=np.zeros(18), control_dt_s=1/60)
        b.online, b.finished, b.failed = online, False, None
        b.phase, b.index, b.stamp = 'aligned', 2, (1, 5)
        b.offset, b.alignment, b.close_translation = np.zeros(3), np.zeros(3), np.zeros(3)
        b.closed, b.nominal = np.ones(12)*.2, np.zeros(18)
        b.log, b.out = [], Path(directory)
        b.observe = lambda: {'observation_id': '1:5:2'}
        b.capture = lambda: {'observation_id': f'1:{b.sim.steps}:{b.index}'}
        b._move_targets = lambda position, hand: iter([np.ones(18)*i/100 for i in range(3)])
        b.rows = []
        def tick(target, phase, future=None):
            b.sim.steps += 1
            b.nominal = target.copy()
            b.rows.append((target.copy(), phase, future))
        b._tick = tick
        return b

    def action(self):
        return dict(observation_id='1:5:2', status='continue', explanation='close',
            skill='close', offset_xyz_m=[0., 0., 0.], object_roi_xyxy=[0, 0, 0, 0])

    def test_iterator_one_step_per_yield_and_blocking_compatibility(self):
        import tempfile
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as c:
            online, sync = self.make_executor(a, True), self.make_executor(c, False)
            # Accept a still-current captured revision after waiting physics ticks.
            online.sim.steps += 20
            run = online.execute_iter(self.action())
            for i in range(3):
                next(run)
                self.assertEqual(online.sim.steps, 26+i)
            with self.assertRaises(StopIteration) as done:
                next(run)
            self.assertEqual(done.exception.value['receipt']['status'], 'executed')
            self.assertEqual(sync.execute(self.action())['receipt']['status'], 'executed')
            np.testing.assert_array_equal([r[0] for r in online.rows], [r[0] for r in sync.rows])
            self.assertEqual(online.phase, 'contact')
            self.assertFalse(online.sim.render_enabled)
            np.testing.assert_array_equal(online.rows[0][2], online.rows[1][0])
            np.testing.assert_array_equal(online.rows[-1][2], online.rows[-1][0])

    def test_sync_rejects_advanced_stamp_and_online_rejects_old_revision(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            b = self.make_executor(d, False)
            b.sim.steps += 1
            with self.assertRaisesRegex(ValueError, 'Stale observation'):
                b.execute(self.action())
            b.online = True
            action = self.action()
            action['observation_id'] = 'old'
            with self.assertRaisesRegex(ValueError, 'Stale observation'):
                b.execute(action)


class OnlineScoreTest(unittest.TestCase):
    def score_case(self, slip=0., support=True, contact_fault=False, jump_every=None):
        import json
        import tempfile
        from pathlib import Path
        phases = ['close', 'test_lift']+['hold']*180
        evaluations, controls = [], []
        contacts = [dict(entity_a='cube', entity_b='ur7e_xhand',
                        link_a='object', link_b=f'right_hand_{f}_link', normal_force_N=.2)
                    for f in (['thumb', 'index'] if support else ['thumb'])]
        for i, phase in enumerate(phases):
            controls.append(dict(step=i, observation=dict(wrist_position_m=[0, 0, 0],
                                                          wrist_quaternion_wxyz=[1, 0, 0, 0])))
            x = (slip if i > 0 else 0.)+(.003*(i//jump_every) if jump_every else 0.)
            evaluations.append(dict(step=i, phase=phase, object_position_m=[x, 0, .1],
                object_quaternion_wxyz=[1, 0, 0, 0], solver_contacts=contacts, bottom_clearance_m=.08,
                contact_guard={'stop': contact_fault}))
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'control.jsonl'
            path.write_text('\n'.join(json.dumps(r) for r in controls))
            return strict_online_metrics(path, evaluations, 1/60, V1Config())

    def test_score_requires_lift_hold_without_drop_or_fault(self):
        good = self.score_case()
        self.assertTrue(good['strict_success'])
        self.assertEqual(good['stable_hold_s'], 3.)
        self.assertEqual(good['stable_shift_m'], .005)
        self.assertTrue(good['fragility_not_simulated'])
        # Slip before the hold interval does not prevent three seconds of subsequent stability.
        slipped = self.score_case(slip=.006)
        self.assertTrue(slipped['strict_success'])
        self.assertAlmostEqual(slipped['slip_distance_m'], .006)
        self.assertEqual(self.score_case(slip=.04)['termination_reason'], 'dropped')
        drifting = self.score_case(jump_every=60)
        self.assertEqual(drifting['termination_reason'], 'unstable')
        self.assertLess(drifting['longest_stable_s'], 3.)
        unsupported = self.score_case(support=False)
        self.assertEqual(unsupported['termination_reason'], 'success')
        self.assertEqual(unsupported['success_rule'], 'lift_position_hold_v3')
        fault = self.score_case(contact_fault=True)
        self.assertFalse(fault['strict_success'])


if __name__ == '__main__':
    unittest.main()
