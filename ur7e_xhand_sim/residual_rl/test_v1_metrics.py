import unittest

import numpy as np

from .v1_metrics import PostContactResidual


class PostContactResidualTest(unittest.TestCase):
    def test_latches_contact_averages_signed_commands_and_excludes_failed_lanes(self):
        stats = PostContactResidual(2, [2, 0, 1])
        stats.update([[.05], [0.]], [[1., 1., 1.], [1., 1., 1.]], .1, np.array([True, True]))
        self.assertIsNone(stats.metrics(0)['post_contact_mean_thumb_flexion_rad'])
        stats.update([[.06], [.06]], [[.09, 0., -.03], [.15, 0., .15]], .1, np.array([True, False]))
        # Even if contact is subsequently lost, it remains part of the post-contact interval.
        stats.update([[0.], [0.]], [[-.03, 0., .09], [1., 1., 1.]], .2, np.array([True, False]))
        row = stats.metrics(0)
        self.assertAlmostEqual(row['post_contact_duration_s'], .3)
        self.assertAlmostEqual(row['post_contact_mean_residual_rad'][2], .01)
        self.assertAlmostEqual(row['post_contact_mean_residual_rad'][1], .05)
        self.assertAlmostEqual(row['post_contact_mean_thumb_flexion_rad'], .03)
        self.assertEqual(stats.metrics(1)['post_contact_duration_s'], 0.)
        stats.reset([0])
        self.assertIsNone(stats.metrics(0)['post_contact_mean_thumb_flexion_rad'])
        np.testing.assert_array_equal(stats.integral, 0.)


if __name__ == '__main__':
    unittest.main()
