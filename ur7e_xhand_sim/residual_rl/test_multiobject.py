"""CPU contracts for shapes, material mass properties and fair force reporting."""
import ast
import itertools
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
from scipy.spatial.transform import Rotation
import torch

from .multiobject_geometry import ObjectSpec
from . import test_adaptive
from . import test_cube_size


def multiobject_harness():
    path = Path(__file__).with_name('multiobject_env.py')
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
    cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef)
                and n.name not in {'__init__', '_create_grasp_object'}]
    namespace = {'np':np, 'ObjectSpec':ObjectSpec, 'TABLE_TOP_Z_M':.2,
                 'AdaptiveVecEnv':test_adaptive.Harness}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])),
                 str(path), 'exec'), namespace)
    return namespace['MultiObjectVecEnv']


MultiHarness = multiobject_harness()


class TestObjectGeometry(unittest.TestCase):
    def test_volume_mass_inertia_for_analytic_primitives(self):
        r, h, s = .0225, .045, .045
        expected = {
            'cube':(s**3, np.repeat(s*s/6, 3)),
            'cuboid':(.045*.060*.045,np.array([.060**2+.045**2,.045**2+.045**2,.045**2+.060**2])/12),
            'hex_prism':(3*np.sqrt(3)/2*(.0225/np.cos(np.pi/6))**2*h,
                np.array([5*(.0225/np.cos(np.pi/6))**2/24+h*h/12]*2+[5*(.0225/np.cos(np.pi/6))**2/12])),
            'cylinder':(np.pi*r*r*h, np.array([(3*r*r+h*h)/12]*2+[r*r/2])),
            'sphere':(4*np.pi*r**3/3, np.repeat(2*r*r/5, 3))}
        for shape, (volume, diagonal) in expected.items():
            spec = ObjectSpec(shape)
            self.assertAlmostEqual(spec.volume_m3, volume, places=14)
            np.testing.assert_allclose(spec.inertia_per_kg, np.diag(diagonal), rtol=1e-14)
            np.testing.assert_allclose(spec.initial_position(), [-.6, -.1333, .2225])
            for mass in (.05, .125, .2):
                self.assertAlmostEqual((mass/spec.volume_m3)*spec.volume_m3, mass, places=14)

    def test_exact_support_matches_rotated_geometric_samples(self):
        rotations = Rotation.random(20, random_state=17)
        q = rotations.as_quat()[:, [3,0,1,2]]
        angle = np.linspace(0, 2*np.pi, 16384, endpoint=False)
        vertices = {
            'cube':np.array(list(itertools.product((-.0225,.0225), repeat=3))),
            'cuboid':np.array(list(itertools.product((-.0225,.0225),(-.030,.030),(-.0225,.0225)))),
            'hex_prism':ObjectSpec('hex_prism').vertices_local_m,
            'cylinder':np.concatenate([np.column_stack([.0225*np.cos(angle),
                .0225*np.sin(angle), np.full(len(angle), z)]) for z in (-.0225,.0225)])}
        for shape, points in vertices.items():
            expected = np.array([rotation.apply(points)[:,2].max() for rotation in rotations])
            spec = ObjectSpec(shape)
            np.testing.assert_allclose(spec.vertical_extent(q), expected, atol=5e-10, rtol=0)
            np.testing.assert_allclose(spec.vertical_extent(torch.tensor(q)).numpy(), expected, atol=5e-10, rtol=0)
        np.testing.assert_array_equal(ObjectSpec('sphere').vertical_extent(q), np.full(20,.0225))

    def test_material_readback_uses_primitive_volume_and_inertia(self):
        factory = test_cube_size.TestCubeSize()
        for shape in ('cube','cuboid','hex_prism','cylinder','sphere'):
            spec = ObjectSpec(shape)
            env = factory.material_fixture(.045)
            env.object_volume_m3 = spec.volume_m3
            env.object_inertia_per_kg = spec.inertia_per_kg
            env.cube.get_links_inertia = lambda: env.mass[:,None,None,None]*spec.inertia_per_kg[None,None]
            env._apply_materials()
            np.testing.assert_allclose(env.density_readback_kg_m3_per_lane, env.mass/spec.volume_m3)
            np.testing.assert_allclose(env.inertia_readback_kg_m2_per_lane,
                                       env.mass[:,None,None]*spec.inertia_per_kg)

    def test_invalid_specs_are_rejected_before_scene_construction(self):
        for value in ({'shape':'soft'}, {'radius_m':0}, {'side_m':float('nan')},
                      {'height_m':True}, {'side_m':'0.045'}, {'across_flats_m':0},
                      {'size_m':[.04,0,.04]}, {'size_m':[.04,.04]},
                      {'size_m':[.04,float('nan'),.04]}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                ObjectSpec.from_value(value)

    def test_hex_mesh_is_closed_outward_exact_and_content_addressed(self):
        import trimesh
        spec=ObjectSpec('hex_prism')
        with tempfile.TemporaryDirectory() as directory:
            metadata=spec.mesh_asset(directory)
            self.assertFalse(Path(metadata['path']).exists())
            created=spec.mesh_asset(directory,create=True)
            self.assertEqual(metadata,created)
            self.assertEqual(created,spec.mesh_asset(directory,create=True))
            mesh=trimesh.load(created['path'],force='mesh',process=False)
            self.assertTrue(mesh.is_watertight and mesh.is_winding_consistent)
            self.assertGreater(mesh.volume,0)
            self.assertEqual(len(mesh.vertices),12)
            self.assertEqual(len(mesh.faces),20)
            np.testing.assert_allclose(mesh.vertices,spec.vertices_local_m,atol=1e-16,rtol=0)
            np.testing.assert_allclose(mesh.extents,spec.bounds_size_m,atol=1e-15,rtol=0)
            self.assertAlmostEqual(mesh.volume,spec.volume_m3,places=15)
            np.testing.assert_allclose(mesh.moment_inertia/mesh.mass,spec.inertia_per_kg,atol=1e-16,rtol=1e-12)
            np.testing.assert_allclose(mesh.center_mass,0,atol=1e-16)
            # Two opposing side faces are exactly perpendicular to world X.
            np.testing.assert_array_equal(np.isclose(np.abs(mesh.face_normals[:,0]),1).sum(),4)
            np.testing.assert_allclose(mesh.bounds[:,0],[-.0225,.0225],atol=1e-16)
            Path(created['path']).write_text('corrupted')
            with self.assertRaisesRegex(ValueError,'content hash'):
                spec.mesh_asset(directory,create=True)
        self.assertNotEqual(ObjectSpec('hex_prism',across_flats_m=.05).mesh_asset()['sha256'],spec.mesh_asset()['sha256'])

    def test_new_dimensions_round_trip_json_and_legacy_cube_defaults(self):
        from dataclasses import asdict
        for shape in ('cube','cuboid','hex_prism','cylinder','sphere'):
            spec=ObjectSpec(shape)
            restored=ObjectSpec.from_value(json.loads(json.dumps(asdict(spec))))
            self.assertEqual(spec,restored)
        old=ObjectSpec.from_value(dict(shape='cube',side_m=.045,radius_m=.0225,height_m=.045))
        self.assertEqual(old,ObjectSpec('cube'))


class TestMultiObjectMeasurements(unittest.TestCase):
    def test_normal_load_is_projection_and_excludes_nonrobot_pairs(self):
        env = test_cube_size.CONTRACT['ResidualVecEnv']()
        env.num_envs, env.device, env.cube_size_m = 1, 'cpu', .045
        env.mu = np.array([.5],np.float32)
        # Two robot contacts have total loads 5+13 N and normal loads 3+5 N.
        # The 100 N table contact and an invalid 200 N hand contact do not count.
        contacts = {
            'link_a':torch.tensor([[11,30,0,12]]),
            'link_b':torch.tensor([[30,18,30,30]]),
            'valid_mask':torch.tensor([[True,True,True,False]]),
            'force_a':torch.tensor([[[3.,4.,0.],[0.,12.,5.],[100.,0.,0.],[200.,0.,0.]]]),
            'position':torch.zeros(1,4,3),
            'normal':torch.tensor([[[2.,0.,0.],[0.,0.,-3.],[1.,0.,0.],[1.,0.,0.]]])}
        env.robot = SimpleNamespace(link_start=10,link_end=20,get_contacts=lambda **_:contacts)
        env.table = SimpleNamespace(base_link=SimpleNamespace(idx=0))
        env.cube = SimpleNamespace(base_link=SimpleNamespace(idx=30),
            get_pos=lambda:torch.tensor([[0.,0.,.3]]),
            get_quat=lambda:torch.tensor([[1.,0.,0.,0.]]))
        env.finger_links = [11,12,13,14,15]
        env.solver = SimpleNamespace(
            get_links_pos=lambda **_:torch.zeros(1,31,3),
            get_links_vel=lambda **_:torch.zeros(1,31,3),
            get_links_ang=lambda:torch.zeros(1,31,3),
            collider=SimpleNamespace(collider_state=SimpleNamespace(
                contact_data=SimpleNamespace(friction=torch.full((1,4),.5)),
                contact_sort_idx=torch.arange(4)[None])))
        _, legacy_load, _, _ = env._metrics()
        np.testing.assert_allclose(legacy_load, [[5.,0.,0.,0.,0.]])
        np.testing.assert_allclose(env.last_contact_diagnostics['all_robot_contact_load_N'], [18.])
        np.testing.assert_allclose(env.last_contact_diagnostics['all_robot_normal_load_N'], [8.])

    def fixture(self):
        env = test_adaptive.TestAdaptiveEnvironmentContract().fixture()
        env.__class__ = MultiHarness
        env.object_spec = ObjectSpec('sphere')
        env.cube_initial_position = np.array([-.6,-.1333,.225])
        env._configure_object_geometry()
        env.density_readback_kg_m3_per_lane = env.mass/env.object_volume_m3
        env.last_contact_diagnostics = {
            'all_robot_contact_load_N':np.array([10.,20.]),
            'all_robot_normal_load_N':np.array([8.,16.])}
        env.reset()
        return env

    def test_all_contact_reward_and_normal_measurements_without_actor_leak(self):
        env = self.fixture()
        action = np.full((2,1), env.action_for_gain(1.), np.float32)
        env.step_async(action)
        obs, reward, _, _ = env.step_wait()
        np.testing.assert_allclose(env.reward_components[:,1], -.12*np.array([10.,20.])/15)
        np.testing.assert_allclose(env.all_force_integral, np.array([10.,20.])/15)
        np.testing.assert_allclose(env.all_normal_integral, np.array([8.,16.])/15)
        # Legacy five-link contact force remains five Newtons in this fixture.
        np.testing.assert_allclose(env.load_integral, np.full(2,5/15))
        self.assertEqual(obs['state'].shape, (2,12,79))
        self.assertEqual(obs['tactile'].shape, (2,12,5,120,3))
        before = env._obs()
        env.last_contact_diagnostics['all_robot_contact_load_N'][:] = 999
        env.object_spec = ObjectSpec('cylinder')
        for key, value in env._obs().items():
            np.testing.assert_array_equal(value, before[key])

    def test_terminal_metrics_use_exact_hold_duration_and_survive_autoreset(self):
        env = self.fixture()
        _, _, infos = test_adaptive.TestAdaptiveEnvironmentContract().finish_hold(env)
        for lane, info in enumerate(infos):
            m = info['episode_metrics']
            self.assertTrue(m['task_success'] and m['strict_success'])
            self.assertAlmostEqual(m['hold_force_measurement_duration_s'],3.,places=10)
            self.assertAlmostEqual(m['hold_mean_all_robot_contact_force_N'],10*(lane+1),places=10)
            self.assertAlmostEqual(m['hold_mean_all_robot_normal_force_N'],8*(lane+1),places=10)
            self.assertAlmostEqual(m['mean_all_robot_contact_force_N'],10*(lane+1),places=10)
            self.assertAlmostEqual(m['hold_mean_sensor_contact_force_N'],5.,places=10)
            self.assertAlmostEqual(m['hold_peak_sensor_contact_force_N'],5.,places=10)
        np.testing.assert_array_equal(env.all_force_integral, [0.,0.])
        self.assertEqual(env.extra_hold_duration, 0.)


if __name__ == '__main__':
    unittest.main()
