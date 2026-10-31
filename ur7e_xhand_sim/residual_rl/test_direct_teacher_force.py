import unittest
import numpy as np
from direct_teacher_curved import force_coordinates, allocate_normal_targets

class BoundedForceUpdateTests(unittest.TestCase):
    def setUp(self):
        self.c={'preload_increment_m_per_action':.000125,'path_inward_m':[-.005,.005],
                'max_inward_m':.005,'force_control_mode':'bidirectional_target'}
    def test_relaxes_overloaded_contact_and_closes_missing_contact(self):
        q=force_coordinates([0,0,0],[3,2.9,0],[3,1.5,1.5],self.c)
        np.testing.assert_allclose(q,[0,-.000125,.000125])
    def test_deadband_preserves_coordinates(self):
        np.testing.assert_array_equal(force_coordinates([0,0,0],[2.9,1.55,1.6],[3,1.5,1.5],self.c),[0,0,0])
    def test_both_bounds_are_hard(self):
        np.testing.assert_allclose(force_coordinates([-.005,.005],[10,0],[1.5,1.5],self.c),[-.005,.005])
    def test_old_mode_never_relaxes(self):
        self.c.pop('force_control_mode')
        np.testing.assert_allclose(force_coordinates([0,0],[5,0],[1.5,1.5],self.c),[0,.000125])
    def test_contact_normal_allocator_balances_requested_xy_force(self):
        normals=np.array([[-.999,-.043,0],[.917,.398,0],[.915,-.403,0]])
        targets,wrench=allocate_normal_targets(normals,3.,[.1,-.05])
        self.assertTrue(np.all(targets>=.4));self.assertTrue(np.all(targets<=4.))
        np.testing.assert_allclose(wrench[:2],[.1,-.05],atol=1e-10)
    def test_allocator_refuses_degenerate_contact_geometry(self):
        with self.assertRaises(ValueError):
            allocate_normal_targets([[-1,0,0],[1,0,0],[1,0,0]],3.,[0,0])
    def test_allocator_hard_limits_target_forces(self):
        target,_=allocate_normal_targets([[-1,0,0],[.9,.4,0],[.9,-.4,0]],3.,[20,0])
        self.assertTrue(np.all(target<=4.));self.assertTrue(np.all(target>=.4))

    def test_unknown_mode_rejected(self):
        self.c['force_control_mode']='invalid'
        with self.assertRaises(ValueError):force_coordinates([0],[0],[1],self.c)

if __name__=='__main__':unittest.main()
