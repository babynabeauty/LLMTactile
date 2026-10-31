"""CPU metadata-only resume contracts; no active runs/checkpoints are read."""
from pathlib import Path
import tempfile
import unittest

from .adaptive_report import (completed_independent_seeds, continuation_audits,
                              continuation_markdown, continuation_parent_contrasts,
                              training_check_applicability)


class ContinuationReportContracts(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / 'outputs' / 'experiment'

    def run_record(self, name, penalty=75, *, resumed=False, start=0,
                   total=266432, parent=None, sha='parent-sha'):
        return {'name': name, 'resumed': resumed, 'uses_tactile': True, 'seed': 0,
            'execution_status': 'completed', 'model_config': {'timesteps_at_run_start': start},
            'timesteps_at_run_start': start, 'timesteps': total,
            'requested_additional_timesteps': 266432, 'actual_additional_timesteps': None,
            'input_checkpoint': str(self.root / parent / 'final_model.zip') if parent else None,
            'input_checkpoint_sha256': sha if parent else None,
            'final_checkpoint_sha256': sha,
            'training_reward_config': {'failure_penalty': penalty,
                'complete_resolved_and_environment_config_verified': penalty is not None}}

    def test_two_stage_budget_penalty_and_parent_identity(self):
        parent = self.run_record('parent')
        child = self.run_record('child', 200, resumed=True, start=266432, total=532864, parent='parent')
        audit, = continuation_audits(self.root, [parent, child])
        self.assertEqual(audit['stage_failure_penalties'], [75, 200])
        self.assertEqual(audit['current_stage']['requested_additional_timesteps'], 266432)
        self.assertEqual(audit['current_stage']['actual_additional_timesteps'], 266432)
        self.assertEqual(audit['current_stage']['requested_cumulative_timesteps'], 532864)
        self.assertEqual(audit['current_stage']['recorded_cumulative_timesteps'], 532864)
        self.assertTrue(audit['parent_final_checkpoint_hash_matches'])
        self.assertTrue(audit['parent_final_timestep_matches_start'])
        self.assertEqual(audit['origin_run'], 'parent')
        text = '\n'.join(continuation_markdown([audit]))
        self.assertIn('266432 / 266432 / 532864', text)
        self.assertIn('75（已核验） → 200（已核验）', text)
        self.assertIn('已训练父模型', text)

    def test_running_budget_is_not_reported_as_completed(self):
        parent = self.run_record('parent')
        child = self.run_record('child', resumed=True, start=266432, total=272224, parent='parent')
        child['execution_status'] = 'running'
        audit, = continuation_audits(self.root, [parent, child])
        self.assertEqual(audit['current_stage']['actual_additional_timesteps'], 5792)
        self.assertEqual(audit['current_stage']['requested_cumulative_timesteps'], 532864)
        self.assertEqual(audit['current_stage']['recorded_cumulative_timesteps'], 272224)
        self.assertEqual(audit['current_stage']['execution_status'], 'running')

    def test_missing_parent_or_start_metadata_is_not_guessed_from_name(self):
        child = self.run_record('continued_p200_533k', None, resumed=True, parent='absent')
        child['model_config'] = {}
        audit, = continuation_audits(self.root, [child])
        self.assertIsNone(audit['origin_run'])
        self.assertFalse(audit['lineage_complete'])
        self.assertIsNone(audit['parent_final_checkpoint_hash_matches'])
        self.assertIsNone(audit['current_stage']['requested_cumulative_timesteps'])
        self.assertIsNone(audit['current_stage']['actual_additional_timesteps'])
        self.assertEqual(audit['stage_failure_penalties'], [None])

    def test_conflicting_parent_hash_and_timestep_are_not_verified(self):
        parent = self.run_record('parent')
        child = self.run_record('child', resumed=True, start=266000, total=532432, parent='parent', sha='wrong')
        audit, = continuation_audits(self.root, [parent, child])
        self.assertFalse(audit['parent_final_checkpoint_hash_matches'])
        self.assertFalse(audit['parent_final_timestep_matches_start'])

    def test_relative_project_path_and_periodic_checkpoint_are_resolved(self):
        parent = self.run_record('parent')
        child = self.run_record('child', resumed=True, start=100000, parent='parent')
        child['input_checkpoint'] = 'outputs/experiment/parent/checkpoints/policy_100000_steps.zip'
        audit, = continuation_audits(self.root, [parent, child])
        self.assertEqual(audit['parent_run'], 'parent')
        self.assertIsNone(audit['parent_final_checkpoint_hash_matches'])
        self.assertIsNone(audit['parent_final_timestep_matches_start'])

    def test_cycle_is_not_a_fresh_origin(self):
        a = self.run_record('a', resumed=True, parent='b')
        b = self.run_record('b', resumed=True, parent='a')
        for audit in continuation_audits(self.root, [a, b]):
            self.assertFalse(audit['lineage_complete'])
            self.assertIsNone(audit['origin_run'])

    def test_fresh_behavior_and_independent_seed_count_unchanged(self):
        parent = self.run_record('parent')
        self.assertEqual(continuation_audits(self.root, [parent]), [])
        self.assertEqual(continuation_markdown([]), [])
        child = self.run_record('child', resumed=True, start=266432, total=532864, parent='parent')
        child['seed'] = 7  # A different restart RNG does not create independent learned weights.
        self.assertEqual(completed_independent_seeds(self.root, [parent, child]), [0])
        self.assertEqual(child['independent_origin_run'], 'parent')

    def test_fresh_checks_remain_applicable_and_resume_preserves_raw_differences(self):
        raw = {'same_initial_action_noise_scale': False, 'same_policy_initialization': False,
            'same_actual_learning_rate': False}
        original = dict(raw)
        fresh = training_check_applicability({'resumed': False}, {'resumed': False}, raw)
        self.assertEqual(fresh['not_applicable_as_fresh_initialization_checks'], [])
        self.assertEqual(set(fresh['raw_false_other_checks']), set(raw))
        resumed = training_check_applicability({'resumed': True}, {'resumed': True}, raw)
        self.assertEqual(set(resumed['not_applicable_as_fresh_initialization_checks']),
            {'same_initial_action_noise_scale', 'same_policy_initialization'})
        self.assertEqual(resumed['raw_false_other_checks'], ['same_actual_learning_rate'])
        self.assertEqual(raw, original)
        self.assertFalse(all(raw.values()))

    def test_same_ablation_penalty_contrast_checks_parent_identity_not_fresh_noise(self):
        parent = self.run_record('parent')
        a = self.run_record('p75', resumed=True, start=266432, total=532864, parent='parent')
        b = self.run_record('p200', 200, resumed=True, start=266432, total=532864, parent='parent')
        runs = [parent, a, b]
        contrasts = continuation_parent_contrasts(runs, continuation_audits(self.root, runs))
        self.assertEqual(len(contrasts), 1)
        self.assertEqual(contrasts[0]['parent_identity_status'], 'matched_parent_identity')
        self.assertTrue(contrasts[0]['uses_tactile'])
        for sha, expected in [('wrong', 'parent_identity_mismatch'), (None, 'unverified_parent_identity')]:
            b['input_checkpoint_sha256'] = sha
            contrast, = continuation_parent_contrasts(runs, continuation_audits(self.root, runs))
            self.assertEqual(contrast['parent_identity_status'], expected)
        b['uses_tactile'] = False
        self.assertEqual(continuation_parent_contrasts(runs, continuation_audits(self.root, runs)), [])


if __name__ == '__main__':
    unittest.main()
