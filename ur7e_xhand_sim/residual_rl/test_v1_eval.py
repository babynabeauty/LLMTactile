"""CPU tests for the repair-grid controller of v1_eval."""
import contextlib
import io
import types
import unittest

import numpy as np

from .v1_eval import grid_action, grid_vectors, print_grid
from .v1_label import candidate_names
from .v1_task import V1Config, action_joints

SQUEEZE = candidate_names().index('thumb +0.150, fingers +0.150')


def scene(shape, peak_region):
    return types.SimpleNamespace(hand_shape=shape, action_joints=np.array(action_joints(['pinch', 'wrap'])),
                                 config=V1Config(), num_envs=len(peak_region), peak_region=np.array(peak_region))


class ResidualGridTest(unittest.TestCase):
    def test_candidates_map_onto_the_shared_action_space(self):
        pinch, wrap = grid_vectors(scene('pinch', [0.])), grid_vectors(scene('wrap', [0.]))
        self.assertEqual(len(pinch), len(candidate_names()))
        # Thumb flexion (joints 1, 2) and the opposing fingers' last two joints; the rest stay at the base.
        np.testing.assert_array_equal(pinch[SQUEEZE], [0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0])
        np.testing.assert_array_equal(wrap[SQUEEZE], [0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1])

    def test_contact_mode_waits_for_touch_in_each_lane(self):
        env = types.SimpleNamespace(envs=[scene('pinch', [0., .2]), scene('wrap', [.06])])
        vectors = [grid_vectors(s) for s in env.envs]
        after = grid_action(env, vectors, SQUEEZE, after_contact=True)
        np.testing.assert_array_equal(after[0], 0)
        np.testing.assert_array_equal(after[1], vectors[0][SQUEEZE])
        np.testing.assert_array_equal(after[2], vectors[1][SQUEEZE])
        np.testing.assert_array_equal(grid_action(env, vectors, SQUEEZE, after_contact=False)[0], vectors[0][SQUEEZE])

    def test_report_counts_episodes_any_candidate_solved(self):
        row = lambda base, success: dict(hand_shape='pinch', level=4, mass_kg=.2, friction=.3, base_id=base,
                                         f_break_N=None, success=success,
                                         termination_reason='success' if success else 'dropped')
        controllers = {f'grid_{k:02d}': [row('a', False), row('b', False)] for k in range(len(candidate_names()))}
        controllers[f'grid_{SQUEEZE:02d}'] = [row('a', True), row('b', False)]
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            print_grid(controllers)
        self.assertIn('pinch       4  0.20  0.30   1/2    thumb +0.150, fingers +0.150 (1)', out.getvalue())


if __name__ == '__main__':
    unittest.main()
