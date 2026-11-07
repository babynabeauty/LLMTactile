"""CPU tests for demo armature overrides and their exported provenance; no physics scene."""
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v1_demo import SpecDemo, VIEWS, validate_finger_armature, write


class Robot:
    def __init__(self, events):
        self.events = events
        self.values = np.arange(40, dtype=np.float32)/100
        self.calls = []

    def get_dofs_armature(self, indices):
        return self.values[indices].copy()

    def set_dofs_armature(self, values, indices):
        self.events.append('armature')
        self.calls.append((values.copy(), list(indices)))
        self.values[indices] = values

    def set_dofs_kp(self, values, indices):
        self.events.append('gain')
        self.kp = values.copy()

    def set_dofs_kv(self, values, indices):
        pass

    def get_dofs_kp(self, indices):
        return self.kp.copy()


class DemoArmatureTest(unittest.TestCase):
    def fixture(self):
        events = []
        robot = Robot(events)
        # Deliberately noncontiguous IDs: twelve fingers must be selected by
        # the robot's DOF mapping, including for a two-finger demonstration.
        indices = [1, 3, 5, 7, 9, 11]+list(range(15, 39, 2))
        def reset():
            events.append('reset')
            robot.values[:] = np.arange(40, dtype=np.float32)/100
        sim = SimpleNamespace(robot=robot, indices=indices, names=[f'joint_{i}' for i in range(18)],
            array=np.asarray, reset=reset, kp=np.ones(18), kv=np.ones(18), solver_profile={},
            wrist=SimpleNamespace(get_quat=lambda: [1., 0., 0., 0.], get_pos=lambda: [0., 0., 0.]),
            object_spec=SimpleNamespace(protocol=lambda: {'shape': 'cuboid'}))
        spec = dict(object_id='cuboid', object_spec={'shape': 'cuboid'}, hand_shape='pinch',
            palm_offset_m=[0., 0., .1], palm_quaternion_wxyz=[1., 0., 0., 0.],
            open_hand_rad=[0.]*12, closed_hand_rad=[.5]*12)
        return sim, spec, events

    def make_demo(self, path, value):
        sim, spec, events = self.fixture()
        demo = SpecDemo(sim, path, spec, finger_armature=value)
        self.addCleanup(demo.record.close)
        self.addCleanup(demo.eval_record.close)
        return demo, sim, events

    def test_none_preserves_loaded_values_without_setter(self):
        with TemporaryDirectory() as tmp:
            demo, sim, events = self.make_demo(Path(tmp)/'demo', None)
            self.assertEqual(sim.robot.calls, [])
            self.assertEqual(events, ['reset', 'gain'])
            np.testing.assert_array_equal(sim.robot.values, np.arange(40, dtype=np.float32)/100)
            self.assertEqual(demo.armature_readback['before_kg_m2'], demo.armature_readback['after_kg_m2'])
            manifest = json.loads((demo.out/'manifest.json').read_text())
            self.assertIsNone(manifest['finger_armature_override_kg_m2'])

    def test_override_after_reset_all_twelve_fingers_before_gain(self):
        for value in (.003, 0.):
            with self.subTest(value=value), TemporaryDirectory() as tmp:
                demo, sim, events = self.make_demo(Path(tmp)/'demo', value)
                self.assertEqual(events, ['reset', 'armature', 'gain'])
                values, indices = sim.robot.calls[0]
                self.assertEqual(indices, sim.indices[6:])
                self.assertEqual(values.shape, (12,))
                expected = np.arange(40, dtype=np.float32)/100
                expected[indices] = value
                np.testing.assert_array_equal(sim.robot.values, expected)
                np.testing.assert_array_equal(demo.armature_readback['after_kg_m2'], expected[sim.indices])

    def test_invalid_inputs_fail_before_reset_or_output_creation(self):
        for value in (-.1, np.nan, np.inf, -np.inf, True, np.bool_(False), [.003], 'invalid'):
            with self.subTest(value=value), TemporaryDirectory() as tmp:
                sim, spec, events = self.fixture()
                output = Path(tmp)/'demo'
                with self.assertRaisesRegex(ValueError, 'finger_armature.*finite nonnegative'):
                    SpecDemo(sim, output, spec, finger_armature=value)
                self.assertFalse(output.exists())
                self.assertEqual(events, [])
        self.assertEqual(validate_finger_armature(np.float32(.003)), float(np.float32(.003)))

    def test_readback_rejects_ignored_setter_and_arm_changes(self):
        for faulty in ('ignored', 'arm_changed'):
            with self.subTest(faulty=faulty):
                sim, _, _ = self.fixture()
                demo = SpecDemo.__new__(SpecDemo)
                demo.sim, demo.finger_armature = sim, .003
                def setter(values, indices):
                    if faulty == 'arm_changed':
                        sim.robot.values[indices] = values
                        sim.robot.values[sim.indices[0]] = 0.
                sim.robot.set_dofs_armature = setter
                with self.assertRaisesRegex(AssertionError,
                        'readback differs' if faulty == 'ignored' else 'changed arm armatures'):
                    demo._configure_finger_armature()

    def test_all_exported_physics_provenance_matches_readback(self):
        with TemporaryDirectory() as tmp:
            demo, sim, _ = self.make_demo(Path(tmp)/'demo', .003)
            demo.initial_object = ([0., 0., 0.], [1., 0., 0., 0.])
            demo.stages = [dict(name=name, image_dir=f'stage_{name}',
                palm_pose_world=np.eye(4).tolist(), object_position_m=[0., 0., 0.],
                sent_joint_target_rad=[0.]*18) for name in ('aligned', 'light_contact')]
            for stage in demo.stages:
                folder = demo.out/stage['image_dir']
                folder.mkdir()
                for view in VIEWS:
                    (folder/f'{view}.jpg').write_bytes(b'test image')
            write(demo.out/'stages.json', demo.stages)
            write(demo.out/'score.json', {'success': True})
            demo.export({'success': True})
            for name in ('manifest.json', 'reference.json', 'demo.json', 'trajectory_metadata.json'):
                with self.subTest(file=name):
                    record = json.loads((demo.out/name).read_text())
                    self.assertEqual(record['finger_armature_override_kg_m2'], .003)
                    self.assertEqual(record['armature_readback'], demo.armature_readback)


if __name__ == '__main__':
    unittest.main()
