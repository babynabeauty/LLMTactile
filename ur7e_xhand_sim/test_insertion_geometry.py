"""CPU checks for true insertion frames and physically important false positives."""
import json
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from ur7e_xhand_sim.insertion_geometry import (evaluate_insertion_pose, insertion_geometry_spec,
                                 target_object_pose)


def quat(rotation):
    return rotation.as_quat()[[3, 0, 1, 2]]


class InsertionGeometryTests(unittest.TestCase):
    def evaluate_goal(self, task, socket_pos=(0., 0., 0.), socket_quat=(1., 0., 0., 0.)):
        goal = target_object_pose(task, socket_pos, socket_quat)
        return evaluate_insertion_pose(task, goal['position_m'], goal['quaternion_wxyz'],
                                       socket_pos, socket_quat)

    def test_nominal_goals_and_json_protocol(self):
        for task in ('tube_insert', 'usb_insert'):
            with self.subTest(task=task):
                result = self.evaluate_goal(task)
                self.assertTrue(result['geometric_success'])
                self.assertGreater(result['clearance_margin_m'], 0.)
                self.assertAlmostEqual(result['insertion_depth_m'],
                                       insertion_geometry_spec(task)['target_depth_m'])
                json.dumps(result, allow_nan=False)

    def test_socket_rigid_transform_invariance(self):
        q = quat(Rotation.from_euler('xyz', [37., -21., 113.], degrees=True))
        for task in ('tube_insert', 'usb_insert'):
            plain = self.evaluate_goal(task)
            rotated = self.evaluate_goal(task, [-.4, .2, .8], q)
            self.assertTrue(rotated['geometric_success'])
            self.assertAlmostEqual(plain['insertion_depth_m'], rotated['insertion_depth_m'])
            self.assertAlmostEqual(plain['clearance_margin_m'], rotated['clearance_margin_m'])

    def test_usb_same_axis_wrong_keying_is_not_success(self):
        # Tip is seated and Z axes align, but the two plastic inserts overlap.
        result = evaluate_insertion_pose('usb_insert', [0., 0., .001], [1., 0., 0., 0.],
                                         [0., 0., 0.], [1., 0., 0., 0.])
        self.assertAlmostEqual(result['axis_error_deg'], 0.)
        self.assertAlmostEqual(result['orientation_error_deg'], 180.)
        self.assertLess(result['tongue_to_empty_cavity_clearance_m'], 0.)
        self.assertFalse(result['geometric_success'])

    def test_tube_axial_spin_is_free(self):
        goal = target_object_pose('tube_insert', [0., 0., 0.], [1., 0., 0., 0.])
        r = Rotation.from_quat(np.asarray(goal['quaternion_wxyz'])[[1, 2, 3, 0]])
        r = r*Rotation.from_euler('z', 127., degrees=True)
        result = evaluate_insertion_pose('tube_insert', goal['position_m'], quat(r),
                                         [0., 0., 0.], [1., 0., 0., 0.])
        self.assertTrue(result['geometric_success'])
        self.assertAlmostEqual(result['full_orientation_error_deg'], 127.)

    def test_tube_subdegree_tilt_can_still_hit_narrow_bore(self):
        goal = target_object_pose('tube_insert', [0., 0., 0.], [1., 0., 0., 0.])
        r = Rotation.from_quat(np.asarray(goal['quaternion_wxyz'])[[1, 2, 3, 0]])
        r = r*Rotation.from_euler('x', .9, degrees=True)
        result = evaluate_insertion_pose('tube_insert', goal['position_m'], quat(r),
                                         [0., 0., 0.], [1., 0., 0., 0.])
        self.assertTrue(result['orientation_ok'])
        self.assertFalse(result['alignment_ok'])
        self.assertFalse(result['geometric_success'])

    def test_usb_tilted_tip_corner_cannot_penetrate_floor(self):
        r = Rotation.from_euler('z', 180., degrees=True)*Rotation.from_euler('y', .2, degrees=True)
        result = evaluate_insertion_pose('usb_insert', [0., 0., 0.], quat(r),
                                         [0., 0., 0.], [1., 0., 0., 0.])
        self.assertTrue(result['depth_ok'])
        self.assertTrue(result['orientation_ok'])
        self.assertLess(result['metal_tip_to_socket_floor_clearance_m'], 0.)
        self.assertFalse(result['geometric_success'])

    def test_above_beside_and_below_base_are_not_success(self):
        for task in ('tube_insert', 'usb_insert'):
            goal = target_object_pose(task, [0., 0., 0.], [1., 0., 0., 0.])
            r = Rotation.from_quat(np.asarray(goal['quaternion_wxyz'])[[1, 2, 3, 0]])
            for displacement in ([0., 0., .1], [.01, 0., 0.], [0., 0., -.1]):
                p = np.asarray(goal['position_m'])+r.apply(displacement)
                result = evaluate_insertion_pose(task, p, goal['quaternion_wxyz'],
                                                 [0., 0., 0.], [1., 0., 0., 0.])
                self.assertFalse(result['geometric_success'], (task, displacement))

    def test_validation_and_threshold_copy(self):
        spec = insertion_geometry_spec('insert_usb')
        spec['success_depth_range_m'][0] = 99.
        self.assertEqual(insertion_geometry_spec('usb_insert')['success_depth_range_m'][0], .016)
        with self.assertRaises(ValueError):
            target_object_pose('usb_insert', [0., 0., 0.], [0., 0., 0., 0.])
        with self.assertRaises(ValueError):
            target_object_pose('grasp', [0., 0., 0.], [1., 0., 0., 0.])


if __name__ == '__main__':
    unittest.main()
