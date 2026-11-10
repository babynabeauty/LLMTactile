"""CPU checks for an opt-in override; no scene creation or physics."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from .v1_env import V1VecEnv, diagnostic_armature


class Robot:
    def __init__(self):
        self.values = np.arange(80, dtype=np.float32).reshape(2, 40) / 100
        self.calls = []

    def get_dofs_armature(self, indices):
        return self.values[:, indices].copy()

    def set_dofs_armature(self, values, indices):
        self.calls.append((values.copy(), list(indices)))
        self.values[:, indices] = values


class ArmatureDiagnosticTest(unittest.TestCase):
    def fixture(self, value):
        # Noncontiguous DOF IDs ensure we use actual robot indices, not columns.
        indices = [1, 3, 5, 7, 9, 11] + list(range(15, 39, 2))
        return SimpleNamespace(robot=Robot(), indices=indices, num_envs=2,
            names=[f'joint_{i}' for i in range(18)], finger_armature=value,
            action_joints=np.arange(6))  # Pinch action space must not narrow the override.

    def test_none_never_calls_setter(self):
        env = self.fixture(None)
        before = env.robot.values.copy()
        V1VecEnv._configure_finger_armature(env)
        self.assertEqual(env.robot.calls, [])
        np.testing.assert_array_equal(env.robot.values, before)
        self.assertEqual(env.armature_readback['before_kg_m2'], env.armature_readback['after_kg_m2'])

    def test_all_twelve_fingers_all_lanes_and_no_other_dofs(self):
        for value in (.1, .01, .001, .003, 0.):
            with self.subTest(value=value):
                env = self.fixture(value)
                before = env.robot.values.copy()
                V1VecEnv._configure_finger_armature(env)
                values, indices = env.robot.calls[0]
                self.assertEqual(indices, env.indices[6:])
                self.assertEqual(values.shape, (2, 12))
                expected = before.copy()
                expected[:, env.indices[6:]] = value
                np.testing.assert_array_equal(env.robot.values, expected)
                np.testing.assert_array_equal(env.armature_readback['after_kg_m2'], expected[:, env.indices])

    def test_invalid_values(self):
        self.assertIsNone(diagnostic_armature(None))
        self.assertEqual(diagnostic_armature(.001), .001)
        for value in (-.1, np.nan, np.inf, True, np.bool_(True), [.1]):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'finger_armature'):
                diagnostic_armature(value)

    def test_readback_detects_ignored_setter(self):
        env = self.fixture(.001)
        env.robot.set_dofs_armature = lambda values, indices: None
        with self.assertRaisesRegex(AssertionError, 'readback differs'):
            V1VecEnv._configure_finger_armature(env)

    def test_environment_builder_uses_recorded_armature_and_allows_diagnostic_override(self):
        from . import v1_env, v1_train
        experiment = dict(objects=[dict(object_id='cuboid_pinch', base_library='unused.json')],
                          lanes_per_object=2, finger_armature_kg_m2=.003)
        with patch.object(v1_train, 'load_library', return_value=(
                dict(hand_shape='pinch', object_spec={'shape': 'cuboid'}), [])), \
             patch.object(v1_env, 'V1VecEnv') as constructor, \
             patch.object(v1_env, 'V1ComposedVecEnv'):
            for value, expected in ((None, .003), (.01, .01)):
                v1_train.build_environment(experiment, {'levels': [4]}, 'unused',
                    seed=1, method='tactile', finger_armature=value)
                self.assertEqual(constructor.call_args.kwargs['finger_armature'], expected)
            del experiment['finger_armature_kg_m2']
            v1_train.build_environment(experiment, {'levels': [4]}, 'unused', seed=1, method='tactile')
            self.assertIsNone(constructor.call_args.kwargs['finger_armature'])


if __name__ == '__main__':
    unittest.main()
