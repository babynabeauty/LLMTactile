import unittest

import numpy as np

from .multiobject_geometry import ObjectSpec
from .v1_task import (BASE_MOTION, V1Config, SamplerConfig, BreakMonitor, SlipTracker, StableHold,
    LEVEL_RANGES_N, action_joints, base_draw_weights, break_soft_rate, episode_specs, force_to_level,
    hand_shape_joints, limit_residual, minimum_grip_force, object_offsets, opposing_support, pose_change,
    sample_episode, scaled_object, score_lift, takeover_tick, wrap_yaw)


def turn_z(deg):
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


class SamplerTest(unittest.TestCase):
    def test_every_episode_is_feasible_and_levels_are_balanced(self):
        rng = np.random.default_rng(0)
        cfg = SamplerConfig()
        episodes = [sample_episode(rng, cfg) for _ in range(4000)]
        levels = np.array([e['level'] for e in episodes])
        np.testing.assert_allclose(np.bincount(levels, minlength=5)[1:]/len(levels), .25, atol=.03)
        for e in episodes:
            self.assertTrue(cfg.mass_range_kg[0] <= e['mass_kg'] <= cfg.mass_range_kg[1])
            self.assertTrue(cfg.friction_range[0] <= e['friction'] <= cfg.friction_range[1])
            self.assertGreater(e['f_break_N'], cfg.feasibility_margin*e['f_min_N']-1e-9)
            self.assertLessEqual(e['f_min_N'], cfg.max_grip_force_N+1e-9)
            self.assertEqual(force_to_level(e['f_break_N']), e['level'])
            if e['narrow_window']:
                self.assertTrue(cfg.narrow_ratio[0]-1e-9 <= e['window_ratio'] <= cfg.narrow_ratio[1]+1e-9)
        breakable = [e for e in episodes if e['level'] < 4]
        self.assertAlmostEqual(np.mean([e['narrow_window'] for e in breakable]), .5, delta=.04)

    def test_astra_errors_are_rare_adjacent_and_mostly_conservative(self):
        rng = np.random.default_rng(1)
        episodes = [sample_episode(rng, SamplerConfig(levels=(2, 3))) for _ in range(6000)]
        wrong = [e for e in episodes if e['astra_level'] != e['level']]
        self.assertAlmostEqual(len(wrong)/len(episodes), .15, delta=.02)
        self.assertTrue(all(abs(e['astra_level']-e['level']) == 1 for e in wrong))
        self.assertAlmostEqual(np.mean([e['astra_level'] < e['level'] for e in wrong]), .7, delta=.05)

    def test_t1_only_stage_is_unbreakable(self):
        rng = np.random.default_rng(2)
        e = sample_episode(rng, SamplerConfig(levels=(4,)))
        self.assertEqual(e['level'], 4)
        self.assertTrue(np.isinf(e['f_break_N']))

    def test_t1_materials_reach_the_hold_limit_but_not_beyond(self):
        rng = np.random.default_rng(4)
        cfg = SamplerConfig(levels=(4,))
        f_min = np.array([sample_episode(rng, cfg)['f_min_N'] for _ in range(4000)])
        self.assertLessEqual(f_min.max(), cfg.max_grip_force_N)
        # Some T1 episodes need more grip than the base gives (about 2-3 N per finger).
        self.assertGreater(np.mean(f_min > 2.5), .1)

    def test_evaluation_grid_skips_infeasible_levels(self):
        heavy_slippery = [(.2, .2)]   # F_min = 4.9 N, so levels 1 and 2 are impossible
        specs = episode_specs([0, 1], heavy_slippery, (1, 2, 3, 4), seed=3)
        self.assertEqual(sorted({s['level'] for s in specs}), [3, 4])
        self.assertEqual(len(specs), 4)
        for s in specs:
            lo, hi = LEVEL_RANGES_N[s['level']]
            self.assertTrue(np.isinf(s['f_break_N']) or 1.2*s['f_min_N'] <= s['f_break_N'] <= hi)
        self.assertEqual(specs, episode_specs([0, 1], heavy_slippery, (1, 2, 3, 4), seed=3))

    def test_minimum_grip_force(self):
        self.assertAlmostEqual(float(minimum_grip_force(.05, .8)), .05*9.81/1.6)


class ResidualTest(unittest.TestCase):
    def test_rate_then_bound(self):
        cfg = V1Config()
        r = np.zeros(6)
        for _ in range(8):
            r = limit_residual(np.ones(6), r, cfg)
        np.testing.assert_allclose(r, .1)
        for _ in range(5):
            r = limit_residual(np.ones(6), r, cfg)
        np.testing.assert_allclose(r, .15)
        r = limit_residual(-np.ones(6)*5, r, cfg)
        np.testing.assert_allclose(r, .1375)

    def test_rate_can_cancel_fastest_base_closure(self):
        # Base and tactile policy now both update every 60 Hz tick.
        self.assertGreaterEqual(V1Config().residual_rate_rad, BASE_MOTION['hand_rad_per_tick'])

    def test_rejects_nonfinite(self):
        with self.assertRaises(ValueError):
            limit_residual(np.full(6, np.nan), np.zeros(6), V1Config())


class BreakAndSlipTest(unittest.TestCase):
    def test_break_needs_consecutive_steps_in_any_region(self):
        monitor = BreakMonitor(2, 2)
        f_break = np.array([2., np.inf])
        active = np.ones(2, dtype=bool)
        over = np.array([[0, 2.5, 0, 0, 0, 0], [0, 99, 0, 0, 0, 0]])
        under = np.zeros((2, 6))
        self.assertFalse(monitor.update(over, f_break, active).any())
        self.assertFalse(monitor.update(under, f_break, active).any())
        self.assertFalse(monitor.update(over, f_break, active).any())
        np.testing.assert_array_equal(monitor.update(over, f_break, active), [True, False])
        self.assertFalse(monitor.update(over, f_break, active).any())
        monitor.reset([0])
        self.assertFalse(monitor.broken[0])

    def test_soft_penalty_ramps_from_eighty_percent(self):
        cfg = V1Config()
        forces = np.array([[1.6, 0, 0, 0, 0, 0], [1.8, 0, 0, 0, 0, 0], [3., 0, 0, 0, 0, 0], [50, 0, 0, 0, 0, 0]])
        rate = break_soft_rate(forces, np.array([2., 2., 2., np.inf]), cfg)
        np.testing.assert_allclose(rate, [0, cfg.break_soft_weight/2, cfg.break_soft_weight, 0], atol=1e-9)

    def test_slip_counts_only_growth_of_largest_displacement(self):
        tracker = SlipTracker(1)
        active = np.array([True])
        self.assertEqual(tracker.update(np.zeros((1, 3)), active)[0], 0)
        self.assertAlmostEqual(tracker.update(np.array([[.003, 0, 0]]), active)[0], .003)
        self.assertEqual(tracker.update(np.array([[.001, 0, 0]]), active)[0], 0)
        self.assertAlmostEqual(tracker.update(np.array([[0, .004, 0]]), active)[0], .001)
        self.assertAlmostEqual(tracker.distance[0], .004)
        self.assertEqual(tracker.update(np.array([[0, .02, 0]]), np.array([False]))[0], 0)

    def test_pose_change_measures_translation_and_rotation(self):
        shift, angle = pose_change(np.zeros(3), np.eye(3), np.array([[.003, .004, 0], [0, 0, 0]]),
                                   np.stack([np.eye(3), turn_z(30)]))
        np.testing.assert_allclose(shift, [.005, 0])
        np.testing.assert_allclose(angle, [0, 30], atol=1e-6)

    def test_lift_hold_accepts_exact_5mm_and_ignores_opposing_support(self):
        cfg, tracker = V1Config(), StableHold(1)
        self.assertEqual((cfg.stable_hold_s, cfg.stable_shift_m), (3., .005))
        for k in range(181):
            # A 3-4-5 triangle checks Euclidean displacement, at the inclusive boundary.
            position = np.zeros((1, 3)) if k == 0 else np.array([[.003, .004, 0.]])
            success = tracker.update(np.array([k/60]), np.array([.02]), position,
                np.eye(3)[None], np.array([False]), np.array([True]), cfg)[0]
            self.assertEqual(bool(success), k == 180)
        self.assertTrue(tracker.lifted[0])
        self.assertFalse(tracker.supported[0])
        self.assertAlmostEqual(tracker.longest[0], 3.)
        tracker.reset([0])
        self.assertTrue(np.isnan(tracker.start[0]))
        self.assertFalse(tracker.lifted[0] or tracker.longest[0])

    def test_lift_hold_restarts_from_current_position_when_over_5mm(self):
        cfg, tracker = V1Config(), StableHold(1)
        for k in range(301):
            # Neither per-step displacement nor a moving reference: total drift exceeds 5 mm at 120.
            x = 0. if k < 60 else .003 if k < 120 else .0051
            success = tracker.update(np.array([k/60]), np.array([.02]), np.array([[x, 0, 0]]),
                np.eye(3)[None], np.array([True]), np.array([True]), cfg)[0]
            self.assertEqual(bool(success), k == 300)
            if k == 120:
                self.assertEqual(tracker.start[0], 2.)
                self.assertAlmostEqual(tracker.position[0, 0], .0051)

    def test_lift_hold_ignores_rotation(self):
        cfg, tracker = V1Config(), StableHold(1)
        for k in range(181):
            success = tracker.update(np.array([k/60]), np.array([.05]), np.zeros((1, 3)),
                turn_z(k*.2)[None], np.array([True]), np.array([True]), cfg)[0]
            self.assertEqual(bool(success), k == 180)  # 36 degrees of rotation still qualifies.

    def test_lift_hold_restarts_below_clearance_and_excludes_failed_lanes(self):
        cfg, tracker = V1Config(), StableHold(2)
        for k in range(242):
            clearance = np.array([.0199 if k == 60 else .02, .02])
            success = tracker.update(np.full(2, k/60), clearance, np.zeros((2, 3)),
                np.tile(np.eye(3), (2, 1, 1)), np.zeros(2, bool), np.array([True, False]), cfg)
            self.assertEqual(bool(success[0]), k == 241)
            self.assertFalse(success[1])

    def test_score_lift_decides_like_the_environment(self):
        cfg, dt = V1Config(), 1/60

        def run(x, clearance=lambda k: .05, fault=lambda k: False):
            return score_lift([(np.array([x(k), 0, 0]), np.eye(3), clearance(k), True, fault(k)) for k in range(300)],
                              dt, cfg)
        # 1 cm of slip while lifting: success three seconds after reaching 2 cm clearance.
        slipped = run(lambda k: min(k, 30)/3000, clearance=lambda k: min(k, 30)/1000)
        self.assertEqual((slipped['reason'], slipped['decided_at']), ('success', 200))
        self.assertAlmostEqual(slipped['slip_m'], .01)
        self.assertEqual(run(lambda k: .04*(k >= 10))['reason'], 'dropped')
        self.assertEqual(run(lambda k: 0., fault=lambda k: k == 5)['decided_at'], 5)
        self.assertEqual(run(lambda k: 0., clearance=lambda k: 0.)['reason'], 'not_lifted')
        drifting = run(lambda k: .003*(k//60))
        self.assertEqual((drifting['reason'], drifting['decided_at']), ('unstable', None))
        self.assertEqual(drifting['success_rule'], 'lift_position_hold_v3')
        interrupted = run(lambda k: 0., clearance=lambda k: .01 if k % 60 == 0 else .02)
        self.assertEqual((interrupted['reason'], interrupted['decided_at']), ('unstable', None))
        self.assertLess(interrupted['longest_stable_s'], 1.)
        # Failure wins even on the exact sample that would complete the hold.
        self.assertEqual(run(lambda k: 0., fault=lambda k: k == 180)['reason'], 'collision')
        self.assertEqual(run(lambda k: .04*(k >= 180))['reason'], 'dropped')

    def test_stable_hold_must_be_positive(self):
        with self.assertRaises(ValueError):
            V1Config(stable_hold_s=0.)

    def test_reward_scale_must_be_positive(self):
        with self.assertRaises(ValueError):
            V1Config(reward_scale=0.)

    def test_support_is_thumb_or_palm_against_another_finger(self):
        regions = np.zeros((5, 7))
        regions[0, [0, 1]] = .1        # pinch: thumb + index
        regions[1, 0] = .1             # thumb alone
        regions[2, [1, 2]] = .1        # index + middle, nothing opposing
        regions[3, [5, 3]] = .1        # wrap without the thumb: palm + ring
        regions[4, [0, 6]] = .1        # thumb + arm only
        np.testing.assert_array_equal(opposing_support(regions, .05), [True, False, False, True, False])

    def test_hand_shapes_and_action_union(self):
        self.assertEqual(hand_shape_joints('pinch'), (0, 1, 2, 3, 4, 5))
        self.assertEqual(hand_shape_joints('tripod'), tuple(range(8)))
        self.assertEqual(action_joints(['pinch']), (0, 1, 2, 3, 4, 5))
        self.assertEqual(action_joints(['pinch', 'tripod']), tuple(range(8)))
        self.assertEqual(action_joints(['wrap', 'pinch']), tuple(range(12)))


class TakeoverTest(unittest.TestCase):
    def test_floor_to_action_grid(self):
        self.assertEqual(takeover_tick({'close': 2.0}), 600)
        self.assertEqual(takeover_tick({'close': 2.05}), 615)
        self.assertEqual(takeover_tick({'close': 2.0667}), 620)
        with self.assertRaises(ValueError):
            takeover_tick({'preload': 1.})


class BaseDrawTest(unittest.TestCase):
    def test_r_bases_capped_at_half_of_draws(self):
        np.testing.assert_allclose(base_draw_weights(['S', 'R', 'R', 'R'], .5), [.5, 1/6, 1/6, 1/6])
        for labels in (['S', 'R'], ['S', 'S', 'R'], ['R', 'R'], ['S', 'S']):
            np.testing.assert_allclose(base_draw_weights(labels, .5), np.full(len(labels), 1/len(labels)))

    def test_yaw_error_follows_symmetry(self):
        self.assertAlmostEqual(wrap_yaw(95., 90.), 5.)
        self.assertAlmostEqual(wrap_yaw(-178., 180.), 2.)
        self.assertEqual(wrap_yaw(37., None), 0.)

    def test_object_offsets_stay_horizontal_and_within_bounds(self):
        cfg, rng = SamplerConfig(), np.random.default_rng(3)
        rotation = np.array([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]])   # palm x axis points down
        own = np.array([.002, -.001, .003])
        pool = [([.006, .002, -.004], 3.), ([-.003, .004, .002], 1.)]
        offsets = object_offsets(rng, pool, own, 2., 90., rotation, 20, cfg)
        self.assertEqual(len(offsets), 20)
        for shift, turn in offsets:
            self.assertEqual(shift[2], 0.)
            self.assertLessEqual(np.linalg.norm(own+rotation.T@shift), cfg.offset_max_m)
            self.assertIn(round(turn+2., 9), (-3., 3., -1., 1.))
        self.assertTrue(all(turn == 0. for _, turn in object_offsets(rng, pool, own, 0., None, rotation, 5, cfg)))


class ScaledObjectTest(unittest.TestCase):
    def test_scaled_object_keeps_shape_and_scales_every_length(self):
        object_id, spec = scaled_object('cylinder', {'shape': 'cylinder', 'radius_m': .0225, 'height_m': .09}, .85)
        self.assertEqual(object_id, 'cylinder_x085')
        np.testing.assert_allclose(ObjectSpec.from_value(spec).bounds_size_m, [.03825, .03825, .0765])
        object_id, spec = scaled_object('cuboid', 'cuboid', 1.15)
        self.assertEqual((object_id, spec['shape']), ('cuboid_x115', 'cuboid'))
        np.testing.assert_allclose(ObjectSpec.from_value(spec).bounds_size_m, [.05175, .069, .05175])


if __name__ == '__main__':
    unittest.main()
