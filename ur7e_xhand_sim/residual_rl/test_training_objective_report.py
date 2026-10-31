"""CPU fixture contracts for LR/config provenance; no real running archives."""
import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

import torch

from .adaptive_report import (Reader, read_learning_rate, training_reward_config_audit,
                              training_objective_match_checks)


class TrainingObjectiveReportContracts(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.run = self.root / 'run'
        self.run.mkdir()
        self.reader = Reader(self.root)

    def checkpoint(self, name, lr, group_lr=None):
        data = {} if lr is None else {'learning_rate': lr}
        buffer = io.BytesIO()
        torch.save({'param_groups': [{'lr': lr if group_lr is None else group_lr}]}, buffer)
        with zipfile.ZipFile(self.run / name, 'w') as archive:
            archive.writestr('data', json.dumps(data))
            archive.writestr('policy.optimizer.pth', buffer.getvalue())

    def test_saved_lr_overrides_requested_value_and_matches_actual_metadata(self):
        for name in ('initial_model.zip', 'final_model.zip'):
            self.checkpoint(name, 1e-4)
        audit = read_learning_rate(self.reader, self.run, {'learning_rate': 1e-4},
            {'learning_rate': 1e-4}, requested=3e-4)
        self.assertEqual(audit['actual_learning_rate'], 1e-4)
        self.assertEqual(audit['requested_learning_rate'], 3e-4)
        self.assertTrue(audit['initial_final_learning_rate_verified'])
        self.assertFalse(self.reader.warnings)

    def test_missing_lr_is_never_defaulted_from_cli(self):
        audit = read_learning_rate(self.reader, self.run, {}, requested=3e-4)
        self.assertIsNone(audit['actual_learning_rate'])
        self.assertFalse(audit['initial_final_learning_rate_verified'])
        for name in ('initial_model.zip', 'final_model.zip'):
            self.checkpoint(name, None, 3e-4)
        audit = read_learning_rate(self.reader, self.run, {}, requested=3e-4)
        self.assertIsNone(audit['actual_learning_rate'])
        self.assertFalse(audit['initial_final_learning_rate_verified'])

    def test_initial_final_or_optimizer_conflict_is_not_verified(self):
        self.checkpoint('initial_model.zip', 3e-4)
        self.checkpoint('final_model.zip', 1e-4)
        audit = read_learning_rate(self.reader, self.run, {})
        self.assertIsNone(audit['actual_learning_rate'])
        self.assertFalse(audit['initial_final_learning_rate_verified'])
        self.checkpoint('final_model.zip', 3e-4, 1e-4)
        audit = read_learning_rate(self.reader, self.run, {'learning_rate': 3e-4})
        self.assertFalse(audit['initial_final_learning_rate_verified'])

    def test_conflicting_model_and_manifest_metadata_is_rejected(self):
        for name in ('initial_model.zip', 'final_model.zip'):
            self.checkpoint(name, 1e-4)
        audit = read_learning_rate(self.reader, self.run, {'learning_rate': 1e-4}, {'learning_rate': 3e-4})
        self.assertFalse(audit['initial_final_learning_rate_verified'])

    def test_failure_penalty_requires_runtime_sources_not_directory_default(self):
        config = {'failure_penalty': 200., 'force_weight': .12}
        audit = training_reward_config_audit(config, config, config, config.keys())
        self.assertEqual(audit['failure_penalty'], 200.)
        self.assertTrue(audit['complete_resolved_and_environment_config_verified'])
        absent = training_reward_config_audit(None, None, config, config.keys())
        self.assertIsNone(absent['failure_penalty'])
        self.assertFalse(absent['complete_resolved_and_environment_config_verified'])

    def test_config_conflict_or_incomplete_fields_cannot_be_matched(self):
        config = {'failure_penalty': 200., 'force_weight': .12}
        for env in ({'failure_penalty': 75., 'force_weight': .12}, {'failure_penalty': 200.}):
            with self.subTest(environment=env):
                audit = training_reward_config_audit(config, env, config, config.keys())
                self.assertFalse(audit['complete_resolved_and_environment_config_verified'])

    def test_pair_rejects_lr_or_full_config_difference(self):
        def run(penalty, lr):
            config = {'failure_penalty': penalty, 'force_weight': .12}
            return {'learning_rate_audit': {'actual_learning_rate': lr, 'initial_final_learning_rate_verified': True},
                'training_reward_config': training_reward_config_audit(config, config, config, config.keys())}
        reference = run(75, 3e-4)
        self.assertTrue(all(training_objective_match_checks(reference, reference).values()))
        penalty = training_objective_match_checks(reference, run(200, 3e-4))
        self.assertFalse(penalty['same_complete_runtime_reward_config'])
        self.assertTrue(penalty['same_actual_learning_rate'])
        smaller_lr = training_objective_match_checks(reference, run(75, 1e-4))
        self.assertFalse(smaller_lr['same_actual_learning_rate'])
        self.assertTrue(smaller_lr['same_complete_runtime_reward_config'])


if __name__ == '__main__':
    unittest.main()
