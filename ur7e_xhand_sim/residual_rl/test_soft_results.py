"""Report accounting checks: unknown outcomes and paired loads are not failures."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from .soft_results import normalize, aggregate, paired, make_html


class TestSoftResults(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='soft_results_synthetic_test_')
        self.root = Path(self.tmp.name)
        self.protocol = {'synthetic': True, 'controllers': ['tactile_seed1','no_tactile_seed1'],
                         'conditions_kg_mu': [[.075,.3],[.125,.5],[.175,.7]]}
        (self.root/'protocol.json').write_text(json.dumps(self.protocol))

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, rows, **extra):
        summary={'synthetic':True,'episodes':rows,'input_integrity_passed':True,
                 'protocol_sha256':hashlib.sha256((self.root/'protocol.json').read_bytes()).hexdigest(),**extra}
        (self.root/'experiment_summary.json').write_text(json.dumps(summary))

    def test_no_summary_means_pending_not_failed(self):
        report=normalize(self.root)
        self.assertEqual(aggregate(report['rows']),dict(planned=6,completed=0,task=0,strict=0,pending=6,running=0,awaiting_audit=0,invalid=0))
        self.assertTrue(report['synthetic'])

    def test_unknown_running_and_invalid_not_success_denominator(self):
        self.write([{'condition_index':0,'controller':'tactile_seed1','valid_completed':True,'task_success':False,'strict_success':False},
                    {'condition_index':0,'controller':'no_tactile_seed1','valid_completed':False,'status':'running'},
                    {'condition_index':1,'controller':'tactile_seed1','valid_completed':False,'status':'invalid_artifact'}])
        stats=aggregate(normalize(self.root)['rows'])
        self.assertEqual((stats['completed'],stats['task'],stats['running'],stats['invalid'],stats['pending']),(1,0,1,1,3))

    def test_common_success_load_pairs_exclude_failed_zero_load(self):
        rows=[]
        for ci,(tv,bv,ts,bs) in enumerate([(2.,4.,True,True),(0.,6.,False,True),(4.,0.,True,False)]):
            for c,load,success in [('tactile_seed1',tv,ts),('no_tactile_seed1',bv,bs)]:
                rows.append(dict(condition_index=ci,controller=c,valid_completed=True,task_success=success,strict_success=success,mean_normal_load_N=load))
        self.write(rows)
        result=paired(normalize(self.root))[0]
        self.assertEqual(result['common'],[0]);self.assertEqual(result['load_pairs'],1)
        self.assertEqual((result['touch_N'],result['blind_N'],result['reduction']),(2.,4.,.5))

    def test_no_common_success_is_missing_not_zero(self):
        result=paired(normalize(self.root))[0]
        self.assertEqual(result['common'],[]);self.assertIsNone(result['touch_N']);self.assertIsNone(result['reduction'])

    def test_live_progress_overrides_unfinished_execution_label(self):
        self.write([dict(condition_index=0,controller='tactile_seed1',valid_completed=False,status='execution_failed_or_interrupted')])
        (self.root/'suite_status.json').write_text(json.dumps(dict(status='running',episodes=[dict(condition_index=0,controller='tactile_seed1',active=True)])))
        result=aggregate(normalize(self.root)['rows'])
        self.assertEqual((result['running'],result['invalid']),(1,0))

    def test_duplicate_or_strict_without_task_rejected(self):
        row=dict(condition_index=0,controller='tactile_seed1',valid_completed=True,task_success=False,strict_success=True)
        self.write([row])
        with self.assertRaises(ValueError):normalize(self.root)
        self.write([dict(condition_index=0,controller='tactile_seed1')]*2)
        with self.assertRaises(ValueError):normalize(self.root)

    def test_error_is_escaped_and_html_self_contained(self):
        self.write([dict(condition_index=0,controller='tactile_seed1',valid_completed=False,status='invalid_artifact',failure_reason='<script>alert(1)</script>')])
        page=make_html(normalize(self.root),b'not-a-real-png',self.root)
        self.assertNotIn('<script>alert(1)</script>',page)
        self.assertIn('&lt;script&gt;',page);self.assertIn('data:image/png;base64,',page)
        self.assertIn('SYNTHETIC',page)

    def test_seed1_scope_is_18_not_72_and_excludes_other_controllers(self):
        self.protocol.update(conditions_kg_mu=[[m,f] for m in (.075,.125,.175) for f in (.3,.5,.7)],
                             official_model_episodes=18,official_fixed_baseline_episodes=0)
        (self.root/'protocol.json').write_text(json.dumps(self.protocol))
        self.write([dict(condition_index=0,controller='tactile_seed2',status='pending')])
        report=normalize(self.root);stats=aggregate(report['rows'])
        self.assertEqual((stats['planned'],stats['pending']),(18,18))
        self.assertEqual(len(report['controllers']),2)
        self.assertEqual(report['seed_ids'],[1]);self.assertEqual(len(report['excluded_nonplanned_rows']),1)
        self.assertEqual([x['seed'] for x in paired(report)],[1])
        self.assertIn('本轮仅种子 1',report['scope'])

    def test_full_72_scope_remains_compatible(self):
        self.protocol.update(controllers=['fixed_2','fixed_3']+[f'{kind}_seed{s}' for s in (1,2,3) for kind in ('tactile','no_tactile')],
            conditions_kg_mu=[[m,f] for m in (.075,.125,.175) for f in (.3,.5,.7)],
            official_model_episodes=54,official_fixed_baseline_episodes=18)
        (self.root/'protocol.json').write_text(json.dumps(self.protocol))
        report=normalize(self.root)
        self.assertEqual(aggregate(report['rows'])['planned'],72)
        self.assertEqual(report['seed_ids'],[1,2,3]);self.assertEqual(len(report['controllers']),8)

    def test_predeclared_bounded_prefix_differences_are_accepted_not_exact(self):
        budgets={'state':5e-5,'tactile':1e-5,'vertices':2e-6,'q':1e-5}
        self.protocol['prefix_tolerance_abs']=budgets
        (self.root/'protocol.json').write_text(json.dumps(self.protocol))
        rows=[dict(condition_index=i,controller=c,valid_completed=True,task_success=True,strict_success=True) for i in range(3) for c in self.protocol['controllers']]
        self.write(rows,all_prefixes_exact=False,all_prefixes_within_tolerance=True,prefix_tolerance_abs=budgets)
        report=normalize(self.root)
        self.assertFalse(report['warnings']);self.assertTrue(report['prefix_bounded_accepted'])
        self.assertEqual(report['status'],'已完成（接受了有限数值差异）')
        self.assertIn('并非逐值完全相同',report['notes'][0])
        self.write(rows,all_prefixes_exact=False,all_prefixes_within_tolerance=True,prefix_tolerance_abs={**budgets,'state':.5})
        report=normalize(self.root)
        self.assertFalse(report['prefix_bounded_accepted']);self.assertTrue(report['warnings'])


if __name__=='__main__':unittest.main()
