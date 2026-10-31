"""CPU-only contracts for bounded shape closure; no API or robot actions."""
import ast
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
from build_assets import HAND_JOINTS
from .test_skill_preload import SkillAct, close_action


def harness():
    path = Path(__file__).parents[1]/'shape_s8v4.py'
    classes = [node for node in ast.parse(path.read_text()).body if isinstance(node,ast.ClassDef)]
    shape = classes[1]
    shape.body = [node for node in shape.body
                  if not isinstance(node,ast.FunctionDef) or node.name != '__init__']
    class Parent:
        def observe(self):
            return {'base_observation':True,'observation_id':'0:0:0'}
        def execute(self,payload):
            self.received_action = SkillAct.model_validate(payload).model_dump()
            return self.received_action
        def servo(self,*args):
            self.parent_calls += 1
            self.sim.steps += 1
            self.motion_calls.append(args)
            if getattr(self,'support_sequence',None):
                self.support_value = self.support_sequence.pop(0)
            elif getattr(self,'lose_support_during_servo',False):
                self.support_value = False
    namespace = {'np':np,'VerifiedS8V4':Parent,'SkillAct':SkillAct,'json':json,'deepcopy':deepcopy,
                 'HAND_JOINTS':HAND_JOINTS,
                 'write':lambda path,value:Path(path).write_text(json.dumps(value)) if Path(path).parent.exists() else None}
    exec(compile(ast.fix_missing_locations(ast.Module(body=classes,type_ignores=[])),str(path),'exec'),namespace)
    return namespace['ShapeS8V4'],namespace['BoundedShapeCandidates']


Shape,Candidates = harness()


def configure_grip_fixture(owner):
    class Robot:
        def set_dofs_kp(self,value,indices):self.kp=np.array(value);self.kp_calls+=1
        def set_dofs_kv(self,value,indices):self.kv=np.array(value);self.kv_calls+=1
        def get_dofs_kp(self,indices):return self.kp[None]
        def get_dofs_kv(self,indices):return self.kv[None]
        def get_dofs_force_range(self,indices):return -owner.sim.effort,owner.sim.effort
    owner.base_grip_kp=np.r_[[4500,4500,3500,2000,2000,2000],[3.]*12]
    owner.base_grip_kv=np.r_[[450,450,350,200,200,200],[.08]*12]
    owner.sim.indices=list(range(18));owner.sim.effort=np.ones(18)*.7
    owner.sim.array=np.asarray;owner.sim.robot=Robot()
    owner.sim.robot.kp_calls=owner.sim.robot.kv_calls=0
    owner.grip_gain_selection={'source':'default','requested_gain':0.,'effective_gain':1.}
    owner._apply_grip_gain(1.)


class TestShapeCandidates(unittest.TestCase):
    def test_preserves_prior_order_and_scores_and_adds_explicit_unranked_choices(self):
        original = {'candidates':[{'offset_xyz_m':[0,0,0],'energy':.2},
                                   {'offset_xyz_m':[.005,0,0],'energy':.1}], 'diagnostic':'from prior'}
        calls = []
        prior = SimpleNamespace(propose=lambda *args,**kw:(calls.append((args,kw)) or original),
                                provenance={'weights':'original'})
        wrapped = Candidates(prior)
        result = wrapped.propose('cloud','pose',.2)
        self.assertEqual(len(calls),1)
        self.assertEqual(result['candidates'][:2],original['candidates'])
        self.assertEqual(len(original['candidates']),2)
        self.assertEqual(len(result['candidates']),9)
        for candidate in result['candidates'][2:]:
            self.assertIsNone(candidate['energy'])
            self.assertIn('not GenDex ranked',candidate['source'])
            self.assertLessEqual(np.linalg.norm(candidate['offset_xyz_m']),.01)
            self.assertEqual(np.count_nonzero(candidate['offset_xyz_m']),1)
        self.assertEqual(wrapped.provenance,prior.provenance)
        self.assertFalse(result['additional_candidate_policy']['automatic_selection'])


class TestBoundedShapeClosure(unittest.TestCase):
    def fixture(self):
        owner = Shape()
        owner.CLOSURE = dict(Shape.CLOSURE)
        owner.preload_selection = {'source':'default','requested_increment_rad':0.,'effective_increment_rad':.05}
        owner.preshape_catalog = {}
        owner.preshape_selection = {'preshape_id':'reference'}
        owner.sim = SimpleNamespace(target=np.zeros(18),lower=np.zeros(18),upper=np.full(18,2.),
            object_spec=SimpleNamespace(shape='sphere'),episode=0,steps=0,array=np.asarray,
            wrist=SimpleNamespace(get_pos=lambda:np.array([[.1,.2,.4]]),
                                  get_quat=lambda:np.array([[1.,0.,0.,0.]])))
        owner.controller = SimpleNamespace(deactivate=lambda:None,active=True)
        owner.closure_state = {'stage':'not_started','close_ticks':0,'confirmed_samples':0}
        owner.motion_recovery_state = {'stage':'idle','events':0,'total_ticks':0,
                                       'consecutive_supported_samples':0}
        owner.motion_recovery_events = []
        owner.motion_calls = []
        owner.out = Path('/unused_cpu_fixture')
        owner.support_history = [True]*50
        owner.parent_calls = 0
        owner.support_value = False
        owner.finished = False
        owner.failed = None
        owner.stamp = (0,0)
        owner.phase = 'aligned'
        owner.offset = np.zeros(3)
        owner.support = lambda:{finger:{'tactile_peak_nominal_N':.02 if owner.support_value else 0.}
                                for finger in ('thumb','index')}
        configure_grip_fixture(owner)
        return owner

    def test_astra_gain_scales_same_six_dofs_as_rl_and_is_absolute(self):
        owner=self.fixture()
        baseline_kp=owner.base_grip_kp.copy();baseline_kv=owner.base_grip_kv.copy()
        with tempfile.TemporaryDirectory() as temp:
            owner.out=Path(temp);(owner.out/'manifest.json').write_text('{}')
            result=owner.execute(close_action(grip_gain=3.))
            self.assertEqual(result['grip_gain'],3.)
            expect_kp=baseline_kp.copy();expect_kv=baseline_kv.copy()
            expect_kp[6:12]*=3;expect_kv[6:12]*=np.sqrt(3)
            np.testing.assert_allclose(owner.sim.robot.kp,expect_kp)
            np.testing.assert_allclose(owner.sim.robot.kv,expect_kv)
            np.testing.assert_array_equal(owner.base_grip_kp,baseline_kp)
            np.testing.assert_array_equal(owner.base_grip_kv,baseline_kv)
            meta=owner._command_metadata()
            self.assertEqual(meta['grip_gain'],3.)
            self.assertEqual(meta['grip_gain_selection']['source'],'Astra SkillAct close')
            self.assertTrue(meta['grip_gain_controller']['effort_limits_unchanged'])
            np.testing.assert_array_equal(meta['grip_gain_controller']['effort_upper_readback_Nm'],owner.sim.effort)
            self.assertEqual(json.loads((owner.out/'manifest.json').read_text())['grip_gain_selection']['effective_gain'],3.)
            owner._apply_grip_gain(2.)
            np.testing.assert_allclose(owner.sim.robot.kp[6:12],baseline_kp[6:12]*2)

    def test_nonclose_keeps_gain_and_stale_close_cannot_apply_gain(self):
        owner=self.fixture();owner._apply_grip_gain(2.)
        before=owner.sim.robot.kp_calls
        payload=close_action();payload['skill']='hold'
        owner.execute(payload)
        self.assertEqual(owner.sim.robot.kp_calls,before)
        self.assertEqual(owner.grip_gain,2.)
        owner.stamp=(0,1)
        with self.assertRaisesRegex(ValueError,'Stale'):
            owner.execute(close_action(grip_gain=3.))
        self.assertEqual(owner.sim.robot.kp_calls,before)

    def test_default_gain_is_one_and_failed_readback_is_rejected(self):
        owner=self.fixture()
        with tempfile.TemporaryDirectory() as temp:
            owner.out=Path(temp);(owner.out/'manifest.json').write_text('{}')
            owner.execute(close_action())
            self.assertEqual(owner.grip_gain,1.)
        owner.sim.robot.get_dofs_kp=lambda indices:np.zeros(18)
        with self.assertRaises(AssertionError):owner._apply_grip_gain(2.)

    def test_astra_selects_preload_and_lazy_generator_reads_actual_choice(self):
        owner,other = self.fixture(),self.fixture()
        generator = owner._bounded_closure()
        with tempfile.TemporaryDirectory() as temp:
            owner.out = Path(temp)
            (owner.out/'manifest.json').write_text('{}')
            result = owner.execute(close_action(preload_increment_rad=.15))
            self.assertEqual(result['preload_increment_rad'],.15)
            manifest = json.loads((owner.out/'manifest.json').read_text())
            self.assertEqual(manifest['closure']['preload_increment_rad'],.15)
            self.assertEqual(manifest['preload_selection']['source'],'Astra SkillAct close')
            self.assertEqual(other.CLOSURE['preload_increment_rad'],.05)
            self.assertEqual(Shape.CLOSURE['preload_increment_rad'],.05)
            owner.support_value = True
            rows = list(generator)
            self.assertAlmostEqual(rows[-1]['sent_target_rad'][10],.15)
            self.assertEqual(owner.observe()['preload_selection']['effective_increment_rad'],.15)

    def test_invalid_close_does_not_mutate_choice_or_write_manifest(self):
        for changed in ('stale','wrong_phase','wrong_offset','ended'):
            owner = self.fixture()
            if changed=='stale':owner.stamp = (0,1)
            elif changed=='wrong_phase':owner.phase = 'contact'
            elif changed=='wrong_offset':owner.offset[0] = .005
            else:owner.failed = 'previous fault'
            with self.subTest(case=changed),self.assertRaises(ValueError):
                owner.execute(close_action(preload_increment_rad=.20))
            self.assertEqual(owner.CLOSURE['preload_increment_rad'],.05)

    def test_zero_close_selects_explicit_default_and_joint_limit_still_applies(self):
        owner = self.fixture()
        with tempfile.TemporaryDirectory() as temp:
            owner.out = Path(temp)
            (owner.out/'manifest.json').write_text('{}')
            owner.execute(close_action())
            self.assertEqual(owner.preload_selection['effective_increment_rad'],.05)
            self.assertEqual(owner.preload_selection['source'],'close default')
            owner.execute(close_action(preload_increment_rad=.20))
            owner.sim.upper[10] = .1
            owner.support_value = True
            rows = list(owner._bounded_closure())
            self.assertAlmostEqual(rows[-1]['sent_target_rad'][10],.1)

    def test_closure_waits_for_contact_then_preloads_only_index(self):
        owner = self.fixture()
        rows = []
        for row in owner._bounded_closure():
            rows.append(row)
            owner.sim.target = np.asarray(row['sent_target_rad'])
            owner.support_value = owner.sim.target[10]>=.04-1e-12
        close = [row for row in rows if row['phase']=='close']
        preload = [row for row in rows if row['phase']=='preload']
        self.assertEqual(len(close),14)
        self.assertEqual(len(preload),60)
        self.assertAlmostEqual(close[-1]['sent_target_rad'][10],.04)
        self.assertAlmostEqual(preload[-1]['sent_target_rad'][10],.09)
        targets = np.array([row['sent_target_rad'] for row in rows])
        self.assertLessEqual(np.diff(targets[:,10]).max(),.004+1e-12)
        np.testing.assert_array_equal(np.delete(targets,10,axis=1),0.)
        self.assertEqual(owner.closure_state['stage'],'ready_for_test_lift')

    def test_selected_synergy_starts_after_contact_and_preserves_joint_sum(self):
        owner = self.fixture()
        owner.CLOSURE.update(post_contact_distal_compensation=True,preload_increment_rad=.15)
        owner.sim.target[11] = 1.55
        rows=[]
        for row in owner._bounded_closure():
            rows.append(row)
            owner.sim.target=np.asarray(row['sent_target_rad'])
            owner.support_value=owner.sim.target[10]>=.04-1e-12
        closing=np.array([r['sent_target_rad'] for r in rows if r['phase']=='close'])
        preload=np.array([r['sent_target_rad'] for r in rows if r['phase']=='preload'])
        np.testing.assert_allclose(closing[:,11],1.55)
        np.testing.assert_allclose(preload[:,10]+preload[:,11],.04+1.55)
        self.assertAlmostEqual(preload[-1,10],.19)
        self.assertAlmostEqual(preload[-1,11],1.40)

    def test_synergy_clamps_shared_increment_at_distal_limit(self):
        owner=self.fixture()
        owner.CLOSURE.update(post_contact_distal_compensation=True,preload_increment_rad=.20)
        owner.sim.target[11]=.08
        owner.support_value=True
        rows=list(owner._bounded_closure())
        final=np.asarray(rows[-1]['sent_target_rad'])
        self.assertAlmostEqual(final[10],.08)
        self.assertAlmostEqual(final[11],0.)
        self.assertTrue(all(r['sent_target_rad'][11]>=0 for r in rows))

    def test_persistent_support_requires_five_new_samples(self):
        owner = self.fixture()
        owner.support_value = True
        rows = list(owner._bounded_closure())
        self.assertEqual(sum(row['phase']=='close' for row in rows),5)
        self.assertEqual(owner.closure_state['confirmed_samples'],5)

    def test_missing_contact_times_out_at_joint_bound_and_cannot_preload(self):
        owner = self.fixture()
        owner.sim.upper[10] = .5
        rows = []
        with self.assertRaisesRegex(ValueError,'lift blocked'):
            for row in owner._bounded_closure():
                rows.append(row)
                owner.sim.target = np.array(row['sent_target_rad'])
        self.assertEqual(len(rows),300)
        self.assertTrue(all(row['phase']=='close' for row in rows))
        self.assertAlmostEqual(max(row['sent_target_rad'][10] for row in rows),.5)

    def test_transient_preload_loss_is_allowed_only_while_stationary_and_reconfirmed(self):
        owner = self.fixture()
        owner.support_value = True
        preload_count = 0
        for row in owner._bounded_closure():
            owner.sim.target = np.array(row['sent_target_rad'])
            if row['phase']=='preload':
                preload_count += 1
                owner.support_value = preload_count>4
        self.assertEqual(preload_count,60)
        self.assertEqual(owner.closure_state['preload_support_gap_ticks'],4)
        self.assertGreaterEqual(owner.closure_state['preload_final_supported_samples'],5)
        self.assertEqual(owner.closure_state['recovery_ticks'],0)
        self.assertEqual(owner.closure_state['stage'],'ready_for_test_lift')

    def test_persistent_preload_contact_loss_remains_failure_with_bounded_recovery(self):
        owner = self.fixture()
        owner.support_value = True
        preload_count = 0
        with self.assertRaisesRegex(ValueError,'not reestablished.*lift blocked'):
            for row in owner._bounded_closure():
                owner.sim.target = np.array(row['sent_target_rad'])
                if row['phase']=='preload':
                    preload_count += 1
                    owner.support_value = False
        self.assertEqual(preload_count,120)
        self.assertEqual(owner.closure_state['recovery_ticks'],60)
        self.assertAlmostEqual(owner.sim.target[10],.1)
        self.assertLessEqual(owner.closure_state['recovery_increment_rad'],.05+1e-12)
        self.assertEqual(owner.closure_state['preload_final_supported_samples'],0)

    def test_bounded_reclosure_can_restore_support_without_changing_other_targets(self):
        owner = self.fixture()
        owner.support_value = True
        preload_count = 0
        for row in owner._bounded_closure():
            owner.sim.target = np.array(row['sent_target_rad'])
            if row['phase']=='preload':
                preload_count += 1
                owner.support_value = preload_count>60 and owner.sim.target[10]>=.08
        self.assertEqual(owner.closure_state['preload_final_supported_samples'],5)
        self.assertLessEqual(owner.closure_state['recovery_ticks'],60)
        self.assertGreater(owner.closure_state['recovery_ticks'],5)
        self.assertLessEqual(owner.sim.target[10],.1)
        np.testing.assert_array_equal(np.delete(owner.sim.target,10),0.)
        self.assertEqual(owner.closure_state['stage'],'ready_for_test_lift')

    def test_support_loss_before_upward_tick_freezes_pose_until_five_new_samples(self):
        owner = self.fixture()
        owner.support_sequence = [False,True,True,True,True,True,True]
        destination = np.array([.1,.2,.5])
        owner.servo(destination,np.array([1.,0.,0.,0.]),np.zeros(12),'lift')
        self.assertEqual(owner.parent_calls,7)
        for call in owner.motion_calls[:6]:
            np.testing.assert_array_equal(call[0],[.1,.2,.4])
            self.assertEqual(call[3],'lift')
        np.testing.assert_array_equal(owner.motion_calls[-1][0],destination)
        self.assertEqual(owner.motion_recovery_events[0]['ticks'],6)
        self.assertEqual(owner.motion_recovery_events[0]['result'],'recovered')

    def test_support_loss_after_upward_tick_recovers_without_recursive_motion(self):
        owner = self.fixture()
        owner.support_value = True
        owner.support_sequence = [False,True,True,True,True,True]
        destination = np.array([.1,.2,.5])
        owner.servo(destination,np.array([1.,0.,0.,0.]),np.zeros(12),'hold')
        self.assertEqual(owner.parent_calls,6)
        np.testing.assert_array_equal(owner.motion_calls[0][0],destination)
        for call in owner.motion_calls[1:]:
            np.testing.assert_array_equal(call[0],[.1,.2,.4])
            self.assertEqual(call[3],'hold')
        self.assertEqual(owner.motion_recovery_events[0]['start_step'],1)
        self.assertEqual(owner.motion_recovery_events[0]['end_step'],6)

    def test_persistent_motion_support_loss_times_out_without_upward_progress(self):
        owner = self.fixture()
        with self.assertRaisesRegex(ValueError,'not recovered at fixed palm pose'):
            owner.servo(np.array([.1,.2,.5]),None,None,'lift')
        self.assertEqual(owner.parent_calls,60)
        self.assertEqual(owner.motion_recovery_state['total_ticks'],60)
        for call in owner.motion_calls:
            np.testing.assert_array_equal(call[0],[.1,.2,.4])
        self.assertEqual(owner.motion_recovery_events[0]['result'],'timeout')

    def test_total_recovery_budget_cannot_reset_per_event(self):
        owner = self.fixture()
        owner.motion_recovery_state['total_ticks'] = 179
        owner.support_sequence = [True]
        with self.assertRaisesRegex(ValueError,'not recovered at fixed palm pose'):
            owner.servo(np.array([.1,.2,.5]),None,None,'test_lift')
        self.assertEqual(owner.parent_calls,1)
        self.assertEqual(owner.motion_recovery_state['total_ticks'],180)
        owner.support_value = False
        with self.assertRaisesRegex(ValueError,'budget exhausted'):
            owner.servo(np.array([.1,.2,.5]),None,None,'test_lift')
        self.assertEqual(owner.parent_calls,1)

    def test_observation_discloses_shape_and_actual_closure_mode(self):
        owner = self.fixture()
        result = owner.observe()
        self.assertEqual(result['object_shape'],'sphere')
        self.assertEqual(result['closure_mode'],'bounded_index_proximal_tactile_closure_v3')
        result['closure_state']['stage'] = 'modified'
        self.assertEqual(owner.closure_state['stage'],'not_started')


if __name__ == '__main__':
    unittest.main()
