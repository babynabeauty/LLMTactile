"""CPU geometry/material/diagnostic contracts; never construct or step Genesis."""
import ast
import itertools
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from . import test_adaptive as adaptive_tests


def contracts():
    path = Path(__file__).with_name('env.py')
    tree = ast.parse(path.read_text())
    selected = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id in {'DEFAULT_CUBE_SIZE_M', 'TABLE_TOP_Z_M'}
                for t in node.targets):
            selected.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in {
                'cpu', 'cube_geometry', 'cube_vertical_extent'}:
            selected.append(node)
        elif isinstance(node, ast.ClassDef) and node.name == 'ResidualVecEnv':
            node.bases = []
            node.body = [method for method in node.body if isinstance(method, ast.FunctionDef)
                         and method.name in {'_apply_materials', '_metrics'}]
            for method in node.body:
                method.body = [statement for statement in method.body
                               if not isinstance(statement, ast.ImportFrom)]
            selected.append(node)
    ns = {'np': np, 'torch': torch, 'qd_to_torch': lambda value, **_: value}
    exec(compile(ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[])),
                 str(path), 'exec'), ns)
    return ns


CONTRACT = contracts()


class TestCubeSize(unittest.TestCase):
    def test_six_sizes_keep_xy_and_place_bottom_on_table(self):
        for side in (.030, .035, .040, .050, .055, .060):
            with self.subTest(side=side):
                source = np.array([-.60, -.1333, .225])
                size, position = CONTRACT['cube_geometry'](side, source)
                self.assertEqual(size, side)
                np.testing.assert_array_equal(position[:2], source[:2])
                self.assertAlmostEqual(position[2]-size/2, .2, places=14)
                np.testing.assert_array_equal(source, [-.60, -.1333, .225])

    def test_default_keeps_original_training_pose_and_explicit_position(self):
        size, position = CONTRACT['cube_geometry']()
        self.assertEqual(size, .045)
        np.testing.assert_array_equal(position, [-.60, -.1333, .225])
        np.testing.assert_array_equal(CONTRACT['cube_geometry'](.045, [-.4, 0, .3])[1], [-.4, 0, .3])

    def test_invalid_side_cannot_create_scene(self):
        for value in (0., -.03, np.nan, np.inf, True, '0.03', [.03], [.03, .04, .05]):
            with self.subTest(value=value), self.assertRaises(ValueError):
                CONTRACT['cube_geometry'](value)

    def test_rotated_extent_matches_transformed_eight_corners(self):
        rotations = Rotation.random(10, random_state=7)
        rotations = Rotation.concatenate([Rotation.identity(), rotations])
        xyzw = rotations.as_quat()
        wxyz = torch.tensor(xyzw[:, [3, 0, 1, 2]], dtype=torch.float64)
        for side in (.03, .045, .06):
            corners = np.array(list(itertools.product((-side/2, side/2), repeat=3)))
            expected = np.array([rotation.apply(corners)[:, 2].max() for rotation in rotations])
            actual = CONTRACT['cube_vertical_extent'](wxyz, side).numpy()
            np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-14)

    def material_fixture(self, side, wrong_inertia_side=None):
        env = CONTRACT['ResidualVecEnv']()
        env.num_envs = 2
        env.cube_size_m = side
        env.cube_size_readback_m = np.full(3, side)
        env.mass = np.array([.125, .2], np.float32)
        env.mu = np.array([.5, .8], np.float32)
        cube = SimpleNamespace()
        cube.set_mass = lambda value: setattr(cube, 'mass', value.copy())
        cube.get_mass = lambda: cube.mass
        cube.set_friction_ratio = lambda value: setattr(cube, 'friction', value.copy())
        inertia_side = wrong_inertia_side or side
        cube.get_links_inertia = lambda: (
            np.eye(3)[None, None] * cube.mass[:, :, None, None]
            * (inertia_side**2 + inertia_side**2)/12)
        env.cube = cube
        env.robot = SimpleNamespace(n_links=17)
        env.robot.set_friction_ratio = lambda value: setattr(env.robot, 'friction', value.copy())
        return env

    def test_actual_mass_inertia_density_and_contact_ratio_for_each_size(self):
        for side in (.03, .035, .04, .045, .05, .055, .06):
            with self.subTest(side=side):
                env = self.material_fixture(side)
                env._apply_materials()
                np.testing.assert_array_equal(env.mass_readback_kg_per_lane, env.mass)
                expected_inertia = np.eye(3)[None] * env.mass[:, None, None] * (side**2 + side**2)/12
                np.testing.assert_allclose(env.inertia_readback_kg_m2_per_lane, expected_inertia)
                np.testing.assert_allclose(env.density_readback_kg_m3_per_lane,
                                           env.mass / np.prod([side, side, side]))
                np.testing.assert_array_equal(env.cube.friction[:, 0], env.mu/.8)
                np.testing.assert_array_equal(env.robot.friction[:, 0], env.mu/.8)

    def test_wrong_original_size_inertia_is_rejected(self):
        env = self.material_fixture(.03, wrong_inertia_side=.045)
        with self.assertRaises(AssertionError):
            env._apply_materials()

    def test_all_robot_load_excludes_table_other_objects_invalid_and_self_contacts(self):
        env = CONTRACT['ResidualVecEnv']()
        env.num_envs, env.device, env.cube_size_m = 1, 'cpu', .06
        env.mu = np.array([.5], np.float32)
        # Physical pair load: sensor=3, non-sensor hand=4, arm=5 N. Exclude
        # table=100, other object=70, invalid=90, robot self-contact=7 N.
        force = torch.tensor([3., 4., 100., 70., 90., 5., 7.])
        contacts = {
            'link_a': torch.tensor([[11, 30, 0, 31, 12, 19, 17]]),
            'link_b': torch.tensor([[30, 18, 30, 30, 30, 30, 11]]),
            'valid_mask': torch.tensor([[True, True, True, True, False, True, True]]),
            'force_a': torch.stack([force, torch.zeros(7), torch.zeros(7)], -1)[None],
            'position': torch.zeros(1, 7, 3),
            'normal': torch.tensor([0., 0., 1.]).expand(1, 7, 3)}
        env.robot = SimpleNamespace(link_start=10, link_end=20, get_contacts=lambda **_: contacts)
        env.table = SimpleNamespace(base_link=SimpleNamespace(idx=0))
        position = torch.tensor([[.1, .2, .3]])
        quaternion = torch.tensor([[1., 0., 0., 0.]])
        env.cube = SimpleNamespace(base_link=SimpleNamespace(idx=30),
                                  get_pos=lambda: position, get_quat=lambda: quaternion)
        env.finger_links = [11, 12, 13, 14, 15]
        env.solver = SimpleNamespace(
            get_links_pos=lambda **_: torch.zeros(1, 32, 3),
            get_links_vel=lambda **_: torch.zeros(1, 32, 3),
            get_links_ang=lambda: torch.zeros(1, 32, 3),
            collider=SimpleNamespace(collider_state=SimpleNamespace(
                contact_data=SimpleNamespace(friction=torch.full((1, 7), .5)),
                contact_sort_idx=torch.arange(7)[None])))
        clearance, sensor_load, slip, fault = env._metrics()
        np.testing.assert_allclose(clearance, [.07], atol=2e-8)
        np.testing.assert_array_equal(sensor_load, [[3., 0., 0., 0., 0.]])
        np.testing.assert_array_equal(env.last_contact_diagnostics['all_robot_contact_load_N'], [12.])
        np.testing.assert_array_equal(env.last_contact_diagnostics['contact_friction_error'], [0.])
        np.testing.assert_array_equal(env.last_contact_diagnostics['cube_quaternion_wxyz'], [[1., 0., 0., 0.]])
        np.testing.assert_array_equal(slip, [0.])
        self.assertTrue(fault[0])
        position[:] = 9
        self.assertLess(env.last_contact_diagnostics['cube_position_m'].max(), 1)

    def test_read_only_diagnostics_follow_four_samples_without_entering_actor(self):
        fixture = adaptive_tests.TestAdaptiveEnvironmentContract()
        env = fixture.fixture()
        metrics = env._metrics
        def sample_metrics():
            env.last_contact_diagnostics = {
                'cube_position_m': np.full((2, 3), env.clock/300),
                'cube_quaternion_wxyz': np.tile([1., 0., 0., 0.], (2, 1)),
                'contact_friction_error': np.zeros(2),
                'all_robot_contact_load_N': np.full(2, env.clock)}
            return metrics()
        env._metrics = sample_metrics
        env.step_async(np.zeros((2, 1), np.float32))
        obs, _, _, _ = env.step_wait()
        self.assertEqual(env.last_interval_metrics['cube_position_m'].shape, (4, 2, 3))
        np.testing.assert_allclose(env.last_interval_metrics['cube_position_m'][:, 0, 0],
                                   (env.start_tick + np.arange(5, 21, 5))/300)
        self.assertEqual(set(obs), {'state', 'tactile'})
        self.assertEqual(obs['state'].shape, (2, 12, 79))
        for value in env.last_contact_diagnostics.values():
            value[:] = 999
        for key, value in env._obs().items():
            np.testing.assert_array_equal(value, obs[key])
        self.assertFalse((env.last_interval_metrics['cube_position_m'] == 999).any())


if __name__ == '__main__':
    unittest.main()
