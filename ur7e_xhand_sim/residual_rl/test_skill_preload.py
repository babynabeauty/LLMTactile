"""Exercise the real action schema/legacy gates without loading GPU modules."""
import ast
from pathlib import Path
from typing import Literal
import unittest

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator


def preload_models():
    root = Path(__file__).parents[1]
    upstream = Path('/home/sai/zsq/vlm-policy')
    namespace = dict(np=np,Literal=Literal,BaseModel=BaseModel,ConfigDict=ConfigDict,
                     Field=Field,model_validator=model_validator)
    for path,name in ((upstream/'core.py','StrictModel'),
                      (upstream/'real_variants/config.py','SkillAct')):
        node = next(n for n in ast.parse(path.read_text()).body
                    if isinstance(n,ast.ClassDef) and n.name==name)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),
                     str(path),'exec'),namespace)
    namespace['ProjectSkillAct'] = namespace['SkillAct']
    path = root/'s8v4.py'
    classes = [n for n in ast.parse(path.read_text()).body if isinstance(n,ast.ClassDef)]
    schema = next(n for n in classes if n.name=='SkillAct')
    legacy = next(n for n in classes if n.name=='S8V4')
    legacy.body = [n for n in legacy.body if isinstance(n,ast.Assign) or
                  isinstance(n,ast.FunctionDef) and n.name in ('execute','_validate_preload_capability')]
    exec(compile(ast.fix_missing_locations(ast.Module(body=[schema,legacy],type_ignores=[])),
                 str(path),'exec'),namespace)
    path = root/'verified_s8v4.py'
    verified = next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.ClassDef))
    verified.body = [n for n in verified.body if isinstance(n,ast.FunctionDef) and n.name=='execute']
    exec(compile(ast.fix_missing_locations(ast.Module(body=[verified],type_ignores=[])),
                 str(path),'exec'),namespace)
    return namespace['SkillAct'],namespace['S8V4'],namespace['VerifiedS8V4']


SkillAct,Legacy,Verified = preload_models()


def close_action(**changes):
    return dict(observation_id='0:0:0',status='continue',explanation='bounded shape closure',
                skill='close',offset_xyz_m=[0.,0.,0.],object_roi_xyxy=[0,0,0,0],**changes)


class TestPreloadSchema(unittest.TestCase):
    def test_default_preserves_old_payload(self):
        action = SkillAct.model_validate(close_action())
        self.assertEqual(action.preload_increment_rad,0.)
        self.assertEqual(action.preshape_id,'keep')
        self.assertEqual(action.grip_gain,0.)
        self.assertEqual(SkillAct.model_json_schema()['properties']['preload_increment_rad']['default'],0.)

    def test_strict_api_schema_requires_every_property(self):
        schema = SkillAct.model_json_schema()
        self.assertEqual(set(schema['required']),set(schema['properties']))
        self.assertIn('preload_increment_rad',schema['required'])
        self.assertIn('preshape_id',schema['required'])
        self.assertIn('grip_gain',schema['required'])
        self.assertFalse(schema['additionalProperties'])
        self.assertEqual(SkillAct.model_validate(close_action()).preload_increment_rad,0.)

    def test_only_discrete_bounded_close_options(self):
        for value in (0.,.05,.10,.15,.20):
            self.assertEqual(SkillAct.model_validate(close_action(preload_increment_rad=value)).preload_increment_rad,value)
        for value in (-.05,.025,.151,.25,float('nan'),float('inf')):
            with self.subTest(value=value),self.assertRaises(ValueError):
                SkillAct.model_validate(close_action(preload_increment_rad=value))
        for skill in ('align','hold','lift','none'):
            payload = close_action(preload_increment_rad=.15)
            payload.update(skill=skill,status='done' if skill=='none' else 'continue')
            with self.subTest(skill=skill),self.assertRaisesRegex(ValueError,'Only the close'):
                SkillAct.model_validate(payload)

    def test_upstream_validation_is_retained(self):
        payload = close_action(preload_increment_rad=.15)
        payload['offset_xyz_m'] = [.03,0.,0.]
        with self.assertRaisesRegex(ValueError,'offset'):
            SkillAct.model_validate(payload)

    def test_grip_gain_only_close_and_legacy_rejected(self):
        for gain in (0.,1.,2.,3.):
            self.assertEqual(SkillAct.model_validate(close_action(grip_gain=gain)).grip_gain,gain)
        for gain in (-1.,.6,1.5,4.,float('nan'),float('inf')):
            with self.subTest(gain=gain),self.assertRaises(ValueError):
                SkillAct.model_validate(close_action(grip_gain=gain))
        for skill in ('locate','align','test_lift','hold','none'):
            payload=close_action(grip_gain=2.)
            payload.update(skill=skill,status='done' if skill=='none' else 'continue')
            if skill=='locate':payload['object_roi_xyxy']=[1,1,20,20]
            with self.subTest(skill=skill),self.assertRaisesRegex(ValueError,'Only the close'):
                SkillAct.model_validate(payload)
        for cls in (Legacy,Verified):
            owner=cls();owner.finished=False;owner.failed=None
            with self.assertRaisesRegex(ValueError,'requires explicit shape-study'):
                owner.execute(close_action(grip_gain=2.))

    def test_legacy_classes_reject_nonzero_before_any_motion(self):
        for cls in (Legacy,Verified):
            owner = cls()
            owner.finished = False
            owner.failed = None
            with self.subTest(cls=cls.__name__),self.assertRaisesRegex(ValueError,'requires explicit shape-study'):
                owner.execute(close_action(preload_increment_rad=.15))
            owner._validate_preload_capability(SkillAct.model_validate(close_action()))

    def test_preshape_allowed_only_at_locate_and_old_modes_reject_explicit_selection(self):
        payload=close_action()
        payload.update(skill='locate',object_roi_xyxy=[1,1,20,20],preshape_id='opposed_mild_hold_face')
        self.assertEqual(SkillAct.model_validate(payload).preshape_id,'opposed_mild_hold_face')
        for cls in (Legacy,Verified):
            owner=cls();owner.finished=False;owner.failed=None
            with self.assertRaisesRegex(ValueError,'requires explicit shape-study'):
                owner.execute(payload)
        for skill in ('close','approach','hold','none'):
            wrong={**payload,'skill':skill,'status':'done' if skill=='none' else 'continue'}
            with self.assertRaisesRegex(ValueError,'Only the locate'):
                SkillAct.model_validate(wrong)


if __name__ == '__main__':
    unittest.main()
