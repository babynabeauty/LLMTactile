"""CPU contracts for training-only preflight; no Genesis scene is created."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from .multiobject_preflight import (FiniteReplayGuard, audit_result, durable_json,
    finite_tree, load_frozen_experiment, run, summarize)
from .multiobject_protocol import make_protocol, bind_demonstrations
from .multiobject_report import trace_statistics
from .test_multiobject_protocol_report import synthetic_trace


class TestPreflight(unittest.TestCase):
    def frozen_fixture(self, root):
        trajectory = root/'demo.jsonl'
        trajectory.write_text('frozen synthetic test data; never simulated\n')
        candidate = make_protocol()
        mapping = {shape:dict(trajectory_path=str(trajectory),demonstration_evidence=dict(
            task_success=True,mass_kg=.05,friction=.8,actual_astra_api=True))
            for shape in [obj['object_id'] for obj in candidate['objects']]}
        protocol = bind_demonstrations(candidate,mapping)
        path = root/'protocol.json'
        path.write_text(json.dumps(protocol))
        return path,protocol

    def test_requires_all_frozen_astra_demos_and_exact_train_pairs(self):
        with tempfile.TemporaryDirectory() as directory:
            path,protocol = self.frozen_fixture(Path(directory))
            loaded = load_frozen_experiment(path)
            pairs = {(r['mass_kg'],r['friction']) for r in loaded['splits']['train']}
            self.assertEqual(len(pairs),12)
            for split in ('validation','test'):
                self.assertFalse(pairs & {(r['mass_kg'],r['friction']) for r in loaded['splits'][split]})
            protocol['objects'][1]['demonstration_evidence']['actual_astra_api'] = False
            path.write_text(json.dumps(protocol))
            with self.assertRaisesRegex(ValueError,'actual Astra'):
                load_frozen_experiment(path)

    def test_changed_demo_is_rejected_before_simulation(self):
        with tempfile.TemporaryDirectory() as directory:
            path,protocol = self.frozen_fixture(Path(directory))
            Path(protocol['objects'][0]['trajectory_path']).write_text('mutated')
            with self.assertRaisesRegex(ValueError,'hash mismatch'):
                load_frozen_experiment(path)

    def test_nan_and_friction_errors_stop_but_grasp_failure_does_not(self):
        env = SimpleNamespace(last_interval_metrics={'contact_friction_error':np.zeros((4,12))})
        env.reset = lambda:{'state':np.zeros((12,12,79))}
        result = ({'state':np.zeros((12,12,79))},np.zeros(12),np.ones(12,dtype=bool),
                  [{'episode_metrics':{'task_success':False}}]*12)
        env.step = lambda action:result
        guarded = FiniteReplayGuard(env)
        guarded.reset()
        self.assertIs(guarded.step(np.zeros((12,1))),result)
        env.last_interval_metrics['contact_friction_error'][0,0] = .1
        with self.assertRaisesRegex(ValueError,'contact friction'):
            guarded.step(np.zeros((12,1)))
        env.last_interval_metrics['contact_friction_error'][0,0] = np.nan
        with self.assertRaises(FloatingPointError):
            guarded.step(np.zeros((12,1)))
        with self.assertRaises(FloatingPointError):
            finite_tree({'state':[1.,np.inf]},'test')

    def test_grasp_failures_remain_denominator_and_gain3_is_only_one_policy(self):
        trace = synthetic_trace()
        trace['clearance_m'][:,1] = 0.
        episodes = trace_statistics(trace,(1.,4.))
        results = []
        for name in ('fixed_1','fixed_3'):
            rows = copy.deepcopy(episodes)
            for lane,row in enumerate(rows):
                row.update(object_id='cube',object_shape='cube',condition_id=f'train_{lane}',
                           mass_kg=.05+.05*lane,friction=.2)
            results.append(dict(object_id='cube',controller=name,episodes=rows,num_episodes=2))
        summary = summarize(results,{})
        self.assertEqual(summary['grasp_failures'],2)
        self.assertEqual(summary['groups']['cube/fixed_3']['task_success_rate'],.5)
        self.assertFalse(summary['cases'][1]['fixed_gain_3_task_success'])
        self.assertIn('does not prove',summary['interpretation'])
        self.assertEqual(summary['split'],'train_preflight')

    def test_final_hold_criterion_not_relaxed_by_preflight(self):
        trace = synthetic_trace()
        trace['clearance_m'][-1,0] = .049
        episodes = trace_statistics(trace,(1.,4.))
        self.assertFalse(episodes[0]['task_success'])
        self.assertTrue(episodes[1]['task_success'])
        trace['contact_load_N'][:,1,1] = 0.
        episodes = trace_statistics(trace,(1.,4.))
        self.assertTrue(episodes[1]['task_success'])
        self.assertFalse(episodes[1]['strict_success'])

    def test_saved_trace_audit_checks_readbacks_and_prefix_without_changing_failure(self):
        trace = synthetic_trace()
        trace['clearance_m'][:,1] = 0.
        episodes = trace_statistics(trace,(1.,4.))
        masses = np.array([.05,.10])
        for lane,row in enumerate(episodes):
            row.update(object_shape='cube',mass_kg=float(masses[lane]),friction=.2)
        side = .045
        physical = dict(object_spec={'shape':'cube'},mass_readback_kg_per_lane=masses.tolist(),
            density_readback_kg_m3_per_lane=(masses/side**3).tolist(),
            inertia_readback_kg_m2_per_lane=(masses[:,None,None]*np.eye(3)*side**2/6).tolist(),
            object_bounds_readback_m=[side]*3,object_initial_bottom_readback_m=[.2,.2],
            friction_per_lane=[.2,.2])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            environment = root/'environment_protocol.json'
            environment.write_text(json.dumps(physical))
            np.savez_compressed(root/'trace.npz',**trace)
            prefix = root/'prefix_consistency.json'
            prefix.write_text(json.dumps({'state':0.,'tactile':0.}))
            result = dict(episodes=episodes,hold_interval_s=[1.,4.],environment_protocol_path=str(environment))
            report = audit_result(result,root,make_protocol())
            self.assertTrue(report['passed'])
            self.assertFalse(report['trace_recomputed'][1]['task_success'])
            prefix.write_text(json.dumps({'state':.001,'tactile':0.}))
            with self.assertRaisesRegex(ValueError,'unequal'):
                audit_result(result,root,make_protocol())
            physical['mass_readback_kg_per_lane'][0] = .2
            environment.write_text(json.dumps(physical))
            with self.assertRaisesRegex(ValueError,'physical values'):
                audit_result(result,root,make_protocol())

    def test_invalid_protocol_persists_engineering_error_without_gpu(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate = root/'candidate.json'
            candidate.write_text(json.dumps(make_protocol()))
            output = root/'preflight'
            with self.assertRaisesRegex(ValueError,'frozen demonstrations'):
                run(candidate,output)
            status = json.loads((output/'status.json').read_text())
            self.assertEqual(status['status'],'engineering_error')
            self.assertFalse(status['engineering_valid'])
            self.assertEqual(status['completed_batches'],0)
            self.assertFalse((output/'status.json.tmp').exists())
            with self.assertRaises(FileExistsError):
                run(candidate,output)

    def test_atomic_json_refuses_nonfinite_values(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'status.json'
            durable_json(path,{'status':'running'})
            with self.assertRaises(ValueError):
                durable_json(path,{'bad':float('nan')})
            self.assertEqual(json.loads(path.read_text()),{'status':'running'})


if __name__ == '__main__':
    unittest.main()
