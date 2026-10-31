"""Sandbox shape and evaluation contracts, with no Genesis initialization."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from .multiobject_geometry import ObjectSpec


def sandbox_harness():
    path = Path(__file__).parents[1]/'environment.py'
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    cls.body = [node for node in cls.body if isinstance(node, ast.FunctionDef)
                and node.name in {'array','values','_initial_object_position','_object_morph',
                                  '_object_material_readback','evaluate'}]
    fake_gs = SimpleNamespace(morphs=SimpleNamespace(**{
        name:(lambda _name=name, **kw:{'type':_name,**kw}) for name in ('Box','Cylinder','Sphere')}))
    ns = {'np':np,'ObjectSpec':ObjectSpec,'gs':fake_gs}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls],type_ignores=[])),str(path),'exec'),ns)
    return ns['Sandbox']


Sandbox = sandbox_harness()


class TestMultiObjectSandbox(unittest.TestCase):
    def test_default_pose_unchanged_and_explicit_shapes_rest_on_table(self):
        sim = Sandbox()
        sim.object_spec = ObjectSpec('cube')
        sim.legacy_object_placement = True
        np.testing.assert_array_equal(sim._initial_object_position(),[-.60,-.1333,.225])
        sim.legacy_object_placement = False
        for shape in ('cube','cylinder','sphere'):
            sim.object_spec = ObjectSpec(shape)
            p = sim._initial_object_position((.03,.02,0))
            np.testing.assert_allclose(p,[-.57,-.1133,.2225],atol=1e-15)
            self.assertAlmostEqual(p[2]-sim.object_spec.bounds_size_m[2]/2,.2,places=14)
            morph = sim._object_morph(p)
            self.assertEqual(morph['type'],{'cube':'Box','cylinder':'Cylinder','sphere':'Sphere'}[shape])
            np.testing.assert_array_equal(morph['pos'],p)

    def fixture(self, shape):
        sim = Sandbox()
        sim.object_spec = ObjectSpec(shape)
        sim.legacy_object_placement = False
        sim.distractors = {}
        sim.fixture = None
        sim.silicone_state = lambda:None
        sim.indices = list(range(18))
        robot = SimpleNamespace(idx=2,get_dofs_force=lambda _:np.zeros(18))
        table = SimpleNamespace(idx=1)
        obj = SimpleNamespace(idx=3)
        links = [SimpleNamespace(idx=i,name=f'link_{i}',entity=e)
                 for i,e in ((11,robot),(18,robot),(30,obj),(0,table))]
        sim.scene = SimpleNamespace(rigid_solver=SimpleNamespace(links=links))
        angle = Rotation.from_euler('x',45,degrees=True).as_quat()[[3,0,1,2]]
        spec = sim.object_spec
        sim.cube = SimpleNamespace(idx=3,
            base_link=SimpleNamespace(idx=30,geoms=[SimpleNamespace(
                init_verts=np.stack([-spec.bounds_size_m/2,spec.bounds_size_m/2]))]),
            get_mass=lambda:np.array([.05]),
            get_links_inertia=lambda:spec.inertia_per_kg[None,None]*.05,
            get_pos=lambda:np.array([[-.6,-.1333,.3]]),get_quat=lambda:angle[None])
        sim.table, sim.robot = table, robot
        robot.get_contacts = lambda:{
            'link_a':np.array([[11,30,0]]),'link_b':np.array([[30,18,30]]),
            'valid_mask':np.array([[True,True,False]]),
            'force_a':np.array([[[3.,4.,0.],[0.,12.,5.],[100.,0.,0.]]]),
            'position':np.zeros((1,3,3)),
            'normal':np.array([[[2.,0.,0.],[0.,0.,-3.],[1.,0.,0.]]])}
        return sim

    def test_described_mass_properties_use_actual_primitive_volume(self):
        for shape in ('cube','cylinder','sphere'):
            sim = self.fixture(shape)
            readback = sim._object_material_readback()
            self.assertEqual(readback['object_engine_mass_kg'],.05)
            self.assertAlmostEqual(readback['object_engine_density_kg_m3'],.05/sim.object_spec.volume_m3)
            np.testing.assert_allclose(readback['object_engine_inertia_kg_m2'],.05*sim.object_spec.inertia_per_kg)

    def test_evaluation_has_generic_lowest_surface_and_legacy_aliases(self):
        expected_extent = {'cube':.0225*np.sqrt(2),'cylinder':.0225*np.sqrt(2),'sphere':.0225}
        for shape in ('cube','cylinder','sphere'):
            sim = self.fixture(shape)
            result = sim.evaluate()
            self.assertAlmostEqual(result['bottom_clearance_m'],.1-expected_extent[shape],places=14)
            self.assertEqual(result['object_position_m'],result['cube_position_m'])
            self.assertEqual(result['object_quaternion_wxyz'],result['cube_quaternion_wxyz'])
            self.assertEqual(result['object_spec']['shape'],shape)
            self.assertAlmostEqual(result['all_robot_contact_load_N'],18.)
            self.assertAlmostEqual(result['all_robot_normal_load_N'],8.)
            self.assertEqual(len(result['solver_contacts']),2)
            # Keep the target entity tag stable for existing S8V4 contact filters.
            self.assertEqual(result['solver_contacts'][0]['entity_b'],'cube')


if __name__ == '__main__':
    unittest.main()
