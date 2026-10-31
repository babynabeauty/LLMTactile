"""Force ownership/scaling tests; physical force checks live in recorded probes."""
import unittest
import numpy as np
from .soft_contact import aggregate_gradient


class TestIPCContactOwnership(unittest.TestCase):
    def test_dt_squared_and_action_reaction(self):
        owners=np.array([0,0,0,1]);dt=.002
        force=np.array([0,0,2.])
        grad=np.array([force/3,force/3,force/3,-force])*dt**2
        result,balance,ignored=aggregate_gradient('PT+N',[3,0,1,2],grad[[3,0,1,2]],owners,1,dt)
        np.testing.assert_allclose(result[0]['normal_force_on_soft_N'],force)
        self.assertAlmostEqual(result[0]['normal_primitive_load_N'],2.)
        self.assertLess(balance,1e-12);self.assertEqual(ignored,0)

    def test_unrelated_and_self_contacts_excluded(self):
        owners=np.array([0,0,1,1,2,2])
        ids=[0,1,2,3,0,4,2,4]
        grads=np.tile([[1.,0,0],[-1.,0,0]],(4,1))
        result,_,ignored=aggregate_gradient('PP+N',ids,grads,owners,1,1.)
        self.assertEqual(ignored,3);self.assertEqual(set(result),{2})
        self.assertEqual(result[2]['normal_primitive_count'],1)

    def test_friction_is_separate(self):
        result,_,_=aggregate_gradient('PP+F',[0,1],[[1.,0,0],[-1.,0,0]],np.array([0,1]),1,1.)
        self.assertEqual(result[0]['normal_primitive_load_N'],0.)
        self.assertEqual(result[0]['friction_primitive_load_N'],1.)

    def test_bad_contact_ownership_fails(self):
        with self.assertRaises(ValueError):
            aggregate_gradient('PE+N',[0,1,2],np.zeros((3,3)),np.array([0,1,2]),1,1.)

    def test_empty_gradient(self):
        out,err,ignored=aggregate_gradient('PT+N',[],np.empty((0,3)),np.array([0]),0,.002)
        self.assertEqual(out,{});self.assertEqual((err,ignored),(0,0))


if __name__=='__main__':unittest.main()
