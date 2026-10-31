"""CPU fixtures for conditional load comparisons without hiding task failures."""
import unittest

import numpy as np

from .adaptive_report import paired_success_load_comparison


CONDITIONS = [[.05, .2], [.1, .4], [.2, .8]]


def episode(material, task, load, strict=None):
    return dict(mass_kg=material[0], friction=material[1], task_success=task,
        strict_success=task if strict is None else strict, mean_contact_force_N=load)


def row(name, episodes):
    return {'controller': name, 'per_lane': episodes}


class PairedReportContracts(unittest.TestCase):
    def compare(self, left, right, conditions=CONDITIONS):
        return paired_success_load_comparison(row('tactile', left), row('baseline', right), conditions)

    def test_failed_zero_load_does_not_create_joint_success_saving(self):
        left = [episode(CONDITIONS[0], False, 0), episode(CONDITIONS[1], True, 12),
                episode(CONDITIONS[2], True, 8)]
        right = [episode(material, True, 10) for material in CONDITIONS]
        result = self.compare(left, right)
        self.assertTrue(result['comparison_valid'])
        self.assertEqual(result['task_success_count_delta'], -1)
        self.assertEqual(result['strict_success_count_delta'], -1)
        self.assertEqual(result['common_task_success_lane_count'], 2)
        self.assertEqual(result['tactile_mean_load_N_on_common_task_success'], 10.)
        self.assertEqual(result['paired_mean_load_delta_N_on_common_task_success'], 0.)
        self.assertEqual(result['paired_mean_load_reduction_fraction_on_common_task_success'], 0.)

    def test_material_reordering_pairs_outcomes_before_subsetting(self):
        left = [episode(CONDITIONS[0], True, 2), episode(CONDITIONS[1], False, 0),
                episode(CONDITIONS[2], True, 6)]
        right = [episode(CONDITIONS[2], False, 0), episode(CONDITIONS[0], True, 4),
                 episode(CONDITIONS[1], True, 8)]
        result = self.compare(left, right)
        self.assertTrue(result['comparison_valid'])
        self.assertEqual(result['material_alignment']['baseline_source_indices'], [1, 2, 0])
        self.assertEqual(result['common_task_success_lane_count'], 1)
        self.assertEqual(result['common_task_success_materials_kg_mu'], [CONDITIONS[0]])
        self.assertEqual(result['paired_mean_load_delta_N_on_common_task_success'], -2.)
        self.assertEqual(result['paired_mean_load_reduction_fraction_on_common_task_success'], .5)

    def test_float32_material_roundoff_is_accepted(self):
        left = [episode(np.asarray(material, np.float32).tolist(), True, 2) for material in CONDITIONS]
        right = [episode(material, True, 4) for material in CONDITIONS]
        self.assertTrue(self.compare(left, right)['comparison_valid'])

    def test_different_or_missing_materials_are_rejected_not_partially_used(self):
        left = [episode(material, True, 2) for material in CONDITIONS]
        for changed in ([CONDITIONS[0], CONDITIONS[1], [.2, .7]], CONDITIONS[:-1]):
            with self.subTest(materials=changed):
                result = self.compare(left, [episode(material, True, 4) for material in changed])
                self.assertFalse(result['comparison_valid'])
                self.assertIsNone(result['common_task_success_lane_count'])
                self.assertIsNone(result['task_success_count_delta'])
                self.assertIsNone(result['paired_mean_load_delta_N_on_common_task_success'])

    def test_duplicate_materials_are_rejected_as_ambiguous(self):
        materials = [CONDITIONS[0], CONDITIONS[0]]
        data = [episode(material, True, 2) for material in materials]
        result = self.compare(data, data, materials)
        self.assertFalse(result['comparison_valid'])
        self.assertIn('ambiguous', result['rejection_reason'])

    def test_no_joint_success_has_zero_count_and_null_load_statistics(self):
        left = [episode(CONDITIONS[0], True, 2), episode(CONDITIONS[1], False, 0)]
        right = [episode(CONDITIONS[0], False, 0), episode(CONDITIONS[1], True, 4)]
        result = self.compare(left, right, CONDITIONS[:2])
        self.assertTrue(result['comparison_valid'])
        self.assertEqual(result['task_success_count_delta'], 0)
        self.assertEqual(result['common_task_success_lane_count'], 0)
        for name in ('tactile_mean_load_N_on_common_task_success',
                     'baseline_mean_load_N_on_common_task_success',
                     'paired_mean_load_delta_N_on_common_task_success',
                     'paired_mean_load_reduction_fraction_on_common_task_success'):
            self.assertIsNone(result[name])

    def test_task_subset_keeps_strict_failure_visible(self):
        left = [episode(CONDITIONS[0], True, 2, strict=False)]
        right = [episode(CONDITIONS[0], True, 4)]
        result = self.compare(left, right, CONDITIONS[:1])
        self.assertEqual(result['common_task_success_lane_count'], 1)
        self.assertEqual(result['task_success_count_delta'], 0)
        self.assertEqual(result['strict_success_count_delta'], -1)
        self.assertEqual(result['strict_success_fraction_delta'], -1.)

    def test_reduction_is_ratio_of_common_means_not_mean_ratio(self):
        left = [episode(CONDITIONS[0], True, 1), episode(CONDITIONS[1], True, 8)]
        right = [episode(CONDITIONS[0], True, 2), episode(CONDITIONS[1], True, 8)]
        result = self.compare(left, right, CONDITIONS[:2])
        self.assertAlmostEqual(result['paired_mean_load_reduction_fraction_on_common_task_success'], .1)
        self.assertEqual(result['paired_mean_load_delta_N_on_common_task_success'], -.5)

    def test_zero_baseline_denominator_is_not_an_infinite_saving(self):
        left = [episode(CONDITIONS[0], True, 2)]
        right = [episode(CONDITIONS[0], True, 0)]
        result = self.compare(left, right, CONDITIONS[:1])
        self.assertEqual(result['paired_mean_load_delta_N_on_common_task_success'], 2.)
        self.assertIsNone(result['paired_mean_load_reduction_fraction_on_common_task_success'])

    def test_missing_labels_or_invalid_common_load_reject_comparison(self):
        right = [episode(CONDITIONS[0], True, 2)]
        for invalid in ({'mass_kg': .05, 'friction': .2},
                        episode(CONDITIONS[0], True, float('nan'))):
            with self.subTest(episode=invalid):
                self.assertFalse(self.compare([invalid], right, CONDITIONS[:1])['comparison_valid'])


if __name__ == '__main__':
    unittest.main()
