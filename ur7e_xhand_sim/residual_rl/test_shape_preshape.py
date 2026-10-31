"""Pure FK and action-binding tests; no physics scene, API or robot actions."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

from shape_s8v4 import build_preshape_catalog, HERE
from .test_shape_s8v4 import Shape, configure_grip_fixture
from .test_skill_preload import SkillAct, close_action


class TestPreshapeGeometry(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reference=json.loads((HERE.parent/'outputs/rule_demos/teacher_pinch_v2_20261024/reference.json').read_text())
        cls.layouts=json.loads((HERE/'generated/tactile_layout.json').read_text())
        cls.urdf=HERE/'generated/ur7e_xhand.urdf'
        root=ET.parse(cls.urdf).getroot()
        limits=[root.find(f'joint[@name="{name}"]/limit') for name in cls.reference['hand_joint_order']]
        cls.lower=np.array([float(l.get('lower')) for l in limits])
        cls.upper=np.array([float(l.get('upper')) for l in limits])
        cls.catalog=cls.build()

    @classmethod
    def build(cls,upper=None):
        return build_preshape_catalog(cls.urdf,cls.layouts,
            cls.reference['open_hand_target_rad'],cls.reference['closed_hand_target_rad'],
            cls.lower,cls.upper if upper is None else upper,.15)

    def fixture(self,out):
        owner=Shape()
        owner.preshape_catalog=deepcopy(self.catalog)
        owner.preshape_selection={'preshape_id':'reference'}
        owner.preload_selection={}
        owner.closure_state={}
        owner.motion_recovery_state={}
        owner.CLOSURE=dict(Shape.CLOSURE)
        owner.open=np.array(self.reference['open_hand_target_rad'])
        owner.closed=np.array(self.reference['closed_hand_target_rad'])
        owner.relative_alignment=np.array([.096,.027,.108])
        owner.reference_relative_alignment=owner.relative_alignment.copy()
        owner.rotation=Rotation.from_euler('xyz',[.3,-.7,.8]).as_matrix()
        owner.sim=SimpleNamespace(lower=np.r_[np.zeros(6),self.lower],upper=np.r_[np.ones(6),self.upper],
            episode=0,steps=0,object_spec=SimpleNamespace(shape='cylinder'))
        configure_grip_fixture(owner)
        owner.out=out
        (out/'manifest.json').write_text('{}')
        owner.prior=SimpleNamespace(provenance={'unchanged_reference_prior':True})
        owner.prior_rebuild_calls=[]
        def rebuild(closed):
            owner.prior_rebuild_calls.append(closed.copy())
            return SimpleNamespace(provenance={'nominal_hand_target_rad':closed.tolist()})
        owner._new_preshape_prior=rebuild
        owner.finished=False;owner.failed=None;owner.stamp=(0,0);owner.phase='unregistered'
        return owner

    def locate(self,name):
        payload=close_action()
        payload.update(skill='locate',object_roi_xyxy=[1,1,30,30],preshape_id=name)
        return payload

    def test_catalog_is_bounded_and_geometry_has_no_object_input(self):
        self.assertEqual(list(self.catalog),['reference','reference_hold_face',
            'opposed_mild_hold_face','opposed_wide_hold_face'])
        for candidate in self.catalog.values():
            for key in ('open_hand_target_rad','nominal_closed_hand_target_rad'):
                hand=np.array(candidate[key])
                self.assertTrue((hand>=self.lower).all() and (hand<=self.upper).all())
            self.assertLessEqual(candidate['commanded_fk_orientation_error'],1e-8)
            self.assertIn('unvalidated',candidate['physical_validation'])
            self.assertIn('not complete',candidate['mechanical_check'])
        reference=self.catalog['reference']['closed_geometry']
        wide=self.catalog['opposed_wide_hold_face']['closed_geometry']
        self.assertGreater(wide['facing_taxel_counts'][0],reference['facing_taxel_counts'][0])
        self.assertGreater(wide['facing_patch_normal_angle_deg'],reference['facing_patch_normal_angle_deg'])
        self.assertLess(wide['facing_patch_tangential_offset_m'],.01)

    def test_bad_runtime_limits_reject_catalog(self):
        upper=self.upper.copy();upper[0]=self.reference['closed_hand_target_rad'][0]-.01
        with self.assertRaisesRegex(ValueError,'limits'):
            self.build(upper)

    def test_optional_out_of_limit_candidate_is_omitted(self):
        upper=self.upper.copy()
        upper[0]=self.reference['closed_hand_target_rad'][0]+.20
        catalog=self.build(upper)
        self.assertIn('reference',catalog)
        self.assertNotIn('opposed_wide_hold_face',catalog)
        np.testing.assert_array_equal(catalog['reference']['nominal_closed_hand_target_rad'],
                                      self.reference['closed_hand_target_rad'])

    def test_reference_keeps_prior_targets_and_registration_exactly(self):
        for name in ('reference','keep'):
            with tempfile.TemporaryDirectory() as directory:
                owner=self.fixture(Path(directory))
                old_prior=owner.prior
                before=(owner.open.copy(),owner.closed.copy(),owner.relative_alignment.copy())
                owner.execute(self.locate(name))
                self.assertIs(owner.prior,old_prior)
                self.assertEqual(owner.prior_rebuild_calls,[])
                for actual,expected in zip((owner.open,owner.closed,owner.relative_alignment),before):
                    np.testing.assert_array_equal(actual,expected)
                self.assertEqual(owner.CLOSURE,Shape.CLOSURE)

    def test_locate_binds_new_prior_and_compensates_in_world_before_parent(self):
        with tempfile.TemporaryDirectory() as directory:
            owner=self.fixture(Path(directory))
            old_center=np.array(self.catalog['reference']['closed_geometry']['pad_midpoint_palm_m'])
            previous=owner.relative_alignment+owner.rotation@old_center
            payload=self.locate('opposed_wide_hold_face')
            returned=owner.execute(payload)
            self.assertEqual(returned['preshape_id'],'opposed_wide_hold_face')
            self.assertEqual(len(owner.prior_rebuild_calls),1)
            np.testing.assert_array_equal(owner.prior_rebuild_calls[0],owner.closed)
            new_center=np.array(self.catalog['opposed_wide_hold_face']['closed_geometry']['pad_midpoint_palm_m'])
            np.testing.assert_allclose(owner.relative_alignment+owner.rotation@new_center,previous,atol=1e-12)
            self.assertTrue(owner.CLOSURE['post_contact_distal_compensation'])
            manifest=json.loads((owner.out/'manifest.json').read_text())
            self.assertEqual(manifest['preshape_selection']['requested_preshape_id'],'opposed_wide_hold_face')
            self.assertEqual(manifest['prior']['nominal_hand_target_rad'],owner.closed.tolist())

    def test_stale_locate_cannot_change_hand_or_score_prior(self):
        with tempfile.TemporaryDirectory() as directory:
            owner=self.fixture(Path(directory));owner.stamp=(0,1)
            with self.assertRaisesRegex(ValueError,'Stale'):
                owner.execute(self.locate('opposed_mild_hold_face'))
            self.assertFalse(owner.prior_rebuild_calls)
            self.assertEqual(owner.preshape_selection['preshape_id'],'reference')


if __name__=='__main__':unittest.main()
