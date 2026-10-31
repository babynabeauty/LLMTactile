"""CPU-only teacher geometry contracts; never constructs a physics scene."""
import unittest
import numpy as np
from direct_teacher_geometry import OpposedPatchGeometry, hand_at


class DirectTeacherGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.geometry=OpposedPatchGeometry()

    def test_surface_patch_offsets_preserve_tangential_position(self):
        g=self.geometry; offsets=np.array([[0,0,.003],[0,0,0.]])
        for distance in (-.010,0.,.005):
            hand,errors=g.solve(distance,offsets)
            self.assertLess(max(errors),2e-5)
            for finger in range(2):
                expected=g.patch_centers[finger]+offsets[finger]
                expected[0]+=(-1 if finger==0 else 1)*distance
                np.testing.assert_allclose(g.patch(hand,finger)[0],expected,atol=2e-5)
            self.assertTrue(np.all(hand>=g.lower));self.assertTrue(np.all(hand<=g.upper))

    def test_invalid_unbounded_teacher_parameters_rejected(self):
        g=self.geometry
        for offsets in ([[0,0,.007],[0,0,0]],[[0,0,float('nan')],[0,0,0]]):
            with self.assertRaises(ValueError):g.solve(0,offsets)
        for shape,height in [('sphere',.025),('cylinder',.031)]:
            with self.assertRaises(ValueError):g.candidate(shape,contact_height_offset_m=height)
        for force in (0.,float('inf'),4.01):
            with self.assertRaises(ValueError):g.candidate('sphere',force_target_N=force)

    def test_tall_cylinder_registration_and_per_finger_interpolation(self):
        c=self.geometry.candidate('cylinder',2.,2.,.025)
        self.assertEqual(c['object_spec'],{'shape':'cylinder','height_m':.09})
        self.assertGreater(c['geometric_checks']['minimum_hand_table_clearance_m'],.05)
        q=hand_at(c,[-.010,.005]);path=np.asarray(c['path_hand_rad'])
        np.testing.assert_array_equal(q[:3],path[0,:3]);np.testing.assert_array_equal(q[3:6],path[-1,3:6])
        with self.assertRaises(ValueError):hand_at(c,[-.011,0])

    def test_three_finger_path_maps_independent_coordinates(self):
        path=np.stack([np.arange(12),np.arange(12)+2.]).astype(float)
        c={'path_inward_m':[-.005,.005],'path_hand_rad':path.tolist(),
           'finger_joint_indices':[[0,1,2],[3,4,5],[6,7]]}
        q=hand_at(c,[-.005,0.,.005])
        np.testing.assert_allclose(q[:3],path[0,:3])
        np.testing.assert_allclose(q[3:6],path[0,3:6]+1.)
        np.testing.assert_allclose(q[6:8],path[-1,6:8])
        np.testing.assert_allclose(q[8:],path[0,8:])
        c['finger_joint_indices']=[[0,1,2],[2,3,4],[6,7]]
        with self.assertRaises(ValueError):hand_at(c,[0,0,0])

    def test_flatface_geometry_keeps_45mm_height(self):
        c=self.geometry.candidate('hex_prism')
        self.assertEqual(c['object_spec'],{'shape':'hex_prism'})
        self.assertGreater(c['geometric_checks']['minimum_hand_table_clearance_m'],.004)
        self.assertLess(c['geometric_checks']['minimum_hand_table_clearance_m'],.01)


if __name__=='__main__':unittest.main()
