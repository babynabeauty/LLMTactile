import unittest
import math
import eden
import torch
from eden.managers.terms.utils import soft_dof_pos_violation
from eden.utils.geom import axis_angle_from_quat, quat_error_magnitude, quat_mul


class LimitUtilityTests(unittest.TestCase):
    def test_batched_lower_inside_upper_and_exact_boundary(self):
        limits = torch.tensor([[-1.0, 1.0], [0.0, 2.0]])
        positions = torch.tensor([[-2.0, 3.0], [0.0, 1.0], [-1.0, 2.0]])
        expected = torch.tensor([[1.0, 1.0], [0.0, 0.0], [0.0, 0.0]])
        torch.testing.assert_close(soft_dof_pos_violation(positions, limits), expected)

    def test_quaternion_geometry(self):
        s = math.sqrt(0.5)
        identity = torch.tensor([[1.0, 0, 0, 0]])
        x = torch.tensor([[s, s, 0, 0]])
        y = torch.tensor([[s, 0, s, 0]])
        torch.testing.assert_close(axis_angle_from_quat(identity), torch.zeros(1, 3))
        torch.testing.assert_close(axis_angle_from_quat(x), torch.tensor([[math.pi / 2, 0, 0]]))
        torch.testing.assert_close(quat_error_magnitude(x, -x), torch.zeros(1), atol=1e-6, rtol=0)
        torch.testing.assert_close(quat_mul(x, y), torch.tensor([[0.5, 0.5, 0.5, 0.5]]))
        torch.testing.assert_close(quat_error_magnitude(identity, x), torch.tensor([math.pi / 2]))


if __name__ == "__main__":
    unittest.main()
