"""CPU contracts for geometric tripod candidates; never builds a simulator."""
import unittest
import json
from pathlib import Path
import tempfile

import numpy as np

from direct_teacher_tripod_geometry import TripodPatchGeometry, hand_at, translated_candidate


class TripodGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sphere = TripodPatchGeometry('sphere', .004)
        cls.sphere_candidate = cls.sphere.candidate(2., 3., .005)

    def test_sphere_height_uses_spherical_cross_section_and_true_bottom(self):
        g, c = self.sphere, self.sphere_candidate
        np.testing.assert_allclose(np.linalg.norm(g.target_centers, axis=1), .0225, atol=1e-10)
        np.testing.assert_allclose(g.target_centers[:, 2], .004, atol=1e-10)
        self.assertLess(max(c['geometric_checks']['nominal_fit_error_m']), .0005)
        self.assertGreater(c['geometric_checks']['minimum_hand_table_clearance_m'], .010)
        self.assertEqual(len(c['path_inward_m']), 41)
        np.testing.assert_allclose(np.diff(c['path_inward_m']), .00025, atol=1e-12)
        for q in c['path_hand_rad']:
            self.assertTrue(np.all(q >= g.lower))
            self.assertTrue(np.all(q <= g.upper))

    def test_independent_three_finger_coordinates_and_midpoint_interpolation(self):
        g, c = self.sphere, self.sphere_candidate
        coordinates = np.array([-.004875, .002125, .004875])
        q = hand_at(c, coordinates)
        for finger in range(3):
            expected = g.patch_centers[finger].copy()
            expected[0] += (-1 if finger == 0 else 1)*coordinates[finger]
            np.testing.assert_allclose(g.patch(q, finger)[0], expected, atol=2e-5)
        np.testing.assert_array_equal(q[8:], c['hand_open_rad'][8:])
        self.assertEqual(c['contact_fingers'], ['thumb', 'index', 'middle'])
        self.assertEqual(c['gain_scope']['middle_absolute_gain'], 1.)
        self.assertEqual(c['force_target_by_finger_N'], {'thumb':3., 'index':1.5, 'middle':1.5})
        self.assertEqual(len(c['geometric_checks']['independent_coordinate_corner_measurements']), 8)
        self.assertGreater(min(c['geometric_checks']['minimum_interfinger_vertex_distances_m'].values()), .006)

    def test_raised_cuboid_patch_preserves_table_clearance(self):
        g=TripodPatchGeometry('cuboid',.012)
        c=g.candidate(2.,3.,.004)
        np.testing.assert_allclose(g.target_centers[:,2],.012)
        self.assertGreater(c['geometric_checks']['minimum_hand_table_clearance_m'],.015)
        with self.assertRaises(ValueError):TripodPatchGeometry('cuboid',.013)

    def test_tall_cylinder_has_explicit_90mm_spec_and_radial_contact_targets(self):
        g = TripodPatchGeometry('cylinder', .025)
        c = g.candidate(2., 3., .002)
        self.assertEqual(c['object_spec']['height_m'], .09)
        np.testing.assert_allclose(np.linalg.norm(g.target_centers[:, :2], axis=1), .0225, atol=1e-10)
        np.testing.assert_allclose(g.target_centers[:, 2], .025)
        self.assertGreater(c['geometric_checks']['minimum_hand_table_clearance_m'], .052)
        self.assertIn('not force-closure proof', c['geometric_checks']['scope'])

    def test_thumb_height_correction_preserves_palm_and_extended_path_limits(self):
        g = self.sphere
        offsets = np.array([[0., 0., .007], [0., 0., 0.], [0., 0., 0.]])
        palm = g.palm_offset.copy()
        c = g.candidate(2., 3., .008, offsets)
        np.testing.assert_array_equal(c['palm_offset_m'], palm)
        np.testing.assert_array_equal(c['patch_offsets_world_m'], offsets)
        self.assertEqual(len(c['path_inward_m']), 53)
        for coordinate in (-.005, 0., .008):
            q, errors = g.solve(coordinate, offsets)
            self.assertLess(max(errors), 2e-5)
            self.assertTrue(np.all(q >= g.lower))
            self.assertTrue(np.all(q <= g.upper))
            for finger in range(3):
                expected = g.patch_centers[finger]+offsets[finger]
                expected[0] += (-1 if finger == 0 else 1)*coordinate
                np.testing.assert_allclose(g.patch(q, finger)[0], expected, atol=2e-5)
        self.assertGreater(c['geometric_checks']['minimum_hand_table_clearance_m'], .010)
        self.assertGreater(min(c['geometric_checks']['minimum_interfinger_vertex_distances_m'].values()), .006)
        for bad in ([[0., 0., .00801], [0., 0., 0.], [0., 0., 0.]], [[0., 0., 0.]]):
            with self.assertRaises(ValueError):
                g.solve(0., bad)

    def test_rigid_placement_keeps_joint_path_exact_and_recomputes_clearance(self):
        base = self.sphere_candidate
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'base.json'
            path.write_text(json.dumps(base))
            base = json.loads(path.read_text())
            moved = translated_candidate(path, [0., 0., -.006])
            self.assertEqual(moved['path_hand_rad'], base['path_hand_rad'])
            self.assertEqual(moved['hand_open_rad'], base['hand_open_rad'])
            self.assertEqual(moved['hand_closed_rad'], base['hand_closed_rad'])
            self.assertEqual(moved['palm_quaternion_wxyz'], base['palm_quaternion_wxyz'])
            self.assertEqual(moved['object_spec'], base['object_spec'])
            self.assertAlmostEqual(moved['contact_height_offset_m'], -.002)
            self.assertFalse(moved['placement_amendment']['exact_new_surface_fit'])
            self.assertAlmostEqual(moved['geometric_checks']['minimum_hand_table_clearance_m'],
                base['geometric_checks']['minimum_hand_table_clearance_m']-.006, places=10)
            self.assertGreater(moved['geometric_checks']['minimum_hand_table_clearance_m'], .004)
            for shift in ([0., 0., -.0061], [.001, 0., 0.], [0., 0., float('nan')]):
                with self.assertRaises(ValueError):
                    translated_candidate(path, shift)
            path.write_text(json.dumps(moved))
            with self.assertRaisesRegex(ValueError, 'single placement'):
                translated_candidate(path, [0., 0., 0.])

    def test_narrow_hex_and_unbounded_actions_rejected(self):
        with self.assertRaisesRegex(ValueError, 'too narrow'):
            TripodPatchGeometry('hex_prism')
        for coordinates in ([0, 0], [0, 0, .0051], [0, 0, float('nan')]):
            with self.assertRaises(ValueError):
                hand_at(self.sphere_candidate, coordinates)
        for gain, force, upper in [(4., 3., .002), (2., .2, .002), (2., 3., .0051), (2., 3., .0021)]:
            with self.assertRaises(ValueError):
                self.sphere.candidate(gain, force, upper)


if __name__ == '__main__':
    unittest.main()
