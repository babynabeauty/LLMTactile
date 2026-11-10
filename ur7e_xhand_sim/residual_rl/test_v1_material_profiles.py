"""Joint profiles must survive both training draws and the evaluation entry point."""
import types
import unittest
from dataclasses import asdict
import json

import numpy as np

from .v1_eval import assign_specs
from .v1_task import SamplerConfig, episode_specs, sample_episode


PROFILES = (
    dict(level=3, mass_kg=.12, friction=.7, f_break_N=6.),
    dict(level=3, mass_kg=.2, friction=.45, f_break_N=7.),
    dict(level=4, mass_kg=.2, friction=.45, f_break_N=None),
)


class MaterialProfileTest(unittest.TestCase):
    def config(self, **kwargs):
        return SamplerConfig(**dict(dict(levels=(3, 4), material_profiles=PROFILES,
                                         astra_error_rate=0.), **kwargs))

    def test_training_never_crosses_mass_friction_and_force_between_profiles(self):
        rng = np.random.default_rng(42)
        # Legacy narrow-window and range defaults must not perturb explicit profiles.
        cfg = self.config(narrow_fraction=1., mass_range_kg=(.25, .3))
        rows = [sample_episode(rng, cfg) for _ in range(2000)]
        self.assertEqual({(r['mass_kg'], r['friction'], r['f_break_N']) for r in rows},
                         {(.12, .7, 6.), (.2, .45, 7.), (.2, .45, np.inf)})
        self.assertTrue(all(r['astra_level'] == r['level'] for r in rows))
        self.assertAlmostEqual(np.mean([r['level'] == 3 for r in rows]), .5, delta=.04)

    def test_profile_json_roundtrip_and_stage_subset(self):
        cfg = self.config(levels=(4,))
        restored = SamplerConfig(**json.loads(json.dumps(asdict(cfg), allow_nan=False)))
        self.assertEqual(cfg.material_profiles, restored.material_profiles)
        self.assertEqual(cfg.levels, restored.levels)
        row = sample_episode(np.random.default_rng(1), restored)
        self.assertEqual((row['level'], row['mass_kg'], row['friction']), (4, .2, .45))
        self.assertTrue(np.isinf(row['f_break_N']))

    def test_evaluation_uses_exact_profiles_and_rejects_unconfigured_pair(self):
        cfg = self.config()
        specs = episode_specs([0, 1], [[.2, .45]], [3, 4], 1, cfg)
        self.assertEqual(len(specs), 4)
        self.assertEqual([s['f_break_N'] for s in specs], [7., np.inf, 7., np.inf])
        self.assertEqual(specs, episode_specs([0, 1], [[.2, .45]], [3, 4], 1, cfg))
        with self.assertRaisesRegex(ValueError, 'No joint material profile'):
            episode_specs([0], [[.12, .45]], [3, 4], 1, cfg)

    def test_cli_evaluation_assignment_preserves_sampler(self):
        received = []
        child = types.SimpleNamespace(sampler=self.config(),
            bases=[types.SimpleNamespace(base_id='a'), types.SimpleNamespace(base_id='b')],
            set_episode_specs=received.extend)
        assign_specs(types.SimpleNamespace(envs=[child]), [[.2, .45]], [3, 4], 1)
        self.assertEqual([r['f_break_N'] for r in received], [7., np.inf, 7., np.inf])
        self.assertEqual([r['base_id'] for r in received], ['a', 'a', 'b', 'b'])

    def test_invalid_or_infeasible_profiles_fail_before_simulation(self):
        for change in (dict(mass_kg=0), dict(friction=float('nan')), dict(friction='bad'),
                       dict(f_break_N=None), dict(f_break_N=-1), dict(f_break_N=1.),
                       dict(level=4, f_break_N=7.), dict(level=0), dict(mass_kg=2)):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.config(material_profiles=[dict(PROFILES[1], **change)])
        with self.assertRaisesRegex(ValueError, 'cover every'):
            self.config(material_profiles=PROFILES[:2])
        with self.assertRaisesRegex(ValueError, 'distinct'):
            self.config(material_profiles=PROFILES+PROFILES)

    def test_five_profiles_are_equiprobable_without_crossing_materials(self):
        from collections import Counter
        profiles = [dict(level=3, mass_kg=m, friction=mu, f_break_N=f)
                    for m, mu, f in ((.1, .45, 5.), (.1, .45, 4.),
                                     (.25, .5, 10.), (.25, .5, 7.))]
        profiles.append(dict(level=4, mass_kg=.5, friction=.5, f_break_N=None))
        cfg = self.config(material_profiles=profiles, max_grip_force_N=5.,
                          material_profile_sampling='uniform_profiles')
        rng = np.random.default_rng(42)
        counts = Counter()
        for _ in range(10000):
            r = sample_episode(rng, cfg)
            counts[(r['mass_kg'], r['friction'], r['f_break_N'])] += 1
        self.assertEqual(set(counts), {(.1, .45, 5.), (.1, .45, 4.),
                                      (.25, .5, 10.), (.25, .5, 7.), (.5, .5, np.inf)})
        for n in counts.values():
            self.assertAlmostEqual(n / 10000, .2, delta=.02)
        subset = SamplerConfig(**dict(asdict(cfg), levels=(4,)))
        self.assertTrue(all(sample_episode(rng, subset)['level'] == 4 for _ in range(20)))
        restored = SamplerConfig(**json.loads(json.dumps(asdict(cfg))))
        self.assertEqual(restored.material_profiles, cfg.material_profiles)
        self.assertEqual(restored.material_profile_sampling, 'uniform_profiles')
        self.assertEqual(restored.levels, cfg.levels)

    def test_invalid_profile_sampling_modes(self):
        with self.assertRaisesRegex(ValueError, 'Unknown material_profile_sampling'):
            self.config(material_profile_sampling='bad')
        with self.assertRaisesRegex(ValueError, 'requires explicit'):
            SamplerConfig(material_profile_sampling='uniform_profiles')


if __name__ == '__main__':
    unittest.main()
