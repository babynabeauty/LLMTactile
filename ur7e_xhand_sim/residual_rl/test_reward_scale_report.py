"""Archive-only reward-scale reporting contracts; no policy, training or physics."""
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from .adaptive_report import Reader, read_training_reward_scale, reward_scale_match_checks


class RewardScaleReportContracts(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.run = self.root / 'run'
        self.run.mkdir()
        self.reader = Reader(self.root)

    def checkpoint(self, name, data):
        with zipfile.ZipFile(self.run / name, 'w') as archive:
            archive.writestr('data', json.dumps(data))

    def pair(self, initial, final):
        self.checkpoint('initial_model.zip', initial)
        self.checkpoint('final_model.zip', final)

    def audit(self, model=None, requested=None):
        return read_training_reward_scale(self.reader, self.run, model or {}, requested)

    def test_read_legacy_archives_resolves_one_and_records_provenance(self):
        self.pair({}, {})
        result = self.audit()
        self.assertEqual(result['training_reward_scale'], 1.)
        self.assertTrue(result['initial_final_reward_scale_verified'])
        self.assertTrue(all(c['legacy_missing_attribute_uses_one'] for c in result['saved_checkpoints']))
        self.assertEqual(len(self.reader.sources), 2)
        self.assertFalse(self.reader.warnings)

    def test_explicit_saved_scale_matches_metadata_and_pair(self):
        self.pair({'training_reward_scale': .01}, {'training_reward_scale': .01})
        result = self.audit({'training_reward_scale': .01, 'reward_scale_requested': .01})
        self.assertEqual(result['training_reward_scale'], .01)
        self.assertTrue(result['initial_final_reward_scale_verified'])
        self.assertTrue(all(reward_scale_match_checks(result, result).values()))
        self.assertIn('value_loss', result['logging_semantics']['optimization'])

    def test_legacy_archive_does_not_verify_claimed_scaled_metadata(self):
        self.pair({}, {})
        result = self.audit({'training_reward_scale': .01})
        self.assertEqual(result['training_reward_scale'], 1.)
        self.assertFalse(result['saved_reward_scale_verified'])
        self.assertTrue(self.reader.warnings)

    def test_initial_final_mismatch_does_not_select_either_value(self):
        self.pair({'training_reward_scale': 1.}, {'training_reward_scale': .01})
        result = self.audit()
        self.assertIsNone(result['training_reward_scale'])
        self.assertFalse(result['initial_final_reward_scale_verified'])
        self.assertFalse(all(reward_scale_match_checks(result, result).values()))

    def test_manifest_and_model_conflict_cannot_verify_matching_archive(self):
        self.pair({'training_reward_scale': .01}, {'training_reward_scale': .01})
        result = read_training_reward_scale(self.reader, self.run,
            {'training_reward_scale': .01}, manifest_model={'training_reward_scale': 1.})
        self.assertEqual(result['training_reward_scale'], .01)
        self.assertFalse(result['saved_reward_scale_verified'])
        self.assertTrue(self.reader.warnings)

    def test_no_archive_never_verifies_a_default_or_request(self):
        result = self.audit(requested=.01)
        self.assertIsNone(result['training_reward_scale'])
        self.assertEqual(result['reward_scale_requested'], .01)
        self.assertFalse(result['saved_reward_scale_verified'])
        self.assertFalse(all(reward_scale_match_checks(result, result).values()))
        declared = self.audit({'training_reward_scale': .01}, requested=.01)
        self.assertEqual(declared['training_reward_scale'], .01)
        self.assertFalse(declared['saved_reward_scale_verified'])

    def test_initial_only_is_observed_but_not_finally_verified(self):
        self.checkpoint('initial_model.zip', {'training_reward_scale': .01})
        result = self.audit()
        self.assertTrue(result['saved_reward_scale_verified'])
        self.assertFalse(result['initial_final_reward_scale_verified'])
        self.assertFalse(all(reward_scale_match_checks(result, result).values()))

    def test_failed_final_read_cannot_leave_initial_verified(self):
        self.checkpoint('initial_model.zip', {'training_reward_scale': .01})
        (self.run / 'final_model.zip').write_bytes(b'incomplete checkpoint')
        result = self.audit()
        self.assertFalse(result['saved_reward_scale_verified'])
        self.assertFalse(result['initial_final_reward_scale_verified'])
        self.assertEqual(len(result['saved_checkpoints']), 2)

    def test_invalid_saved_scalars_are_not_legacy_defaults(self):
        for value in (None, True, 0, -1, float('nan'), float('inf'), '0.01', {'value': .01}):
            with self.subTest(value=value):
                self.pair({'training_reward_scale': value}, {'training_reward_scale': value})
                result = self.audit()
                self.assertIsNone(result['training_reward_scale'])
                self.assertFalse(result['saved_reward_scale_verified'])

    def test_different_verified_scales_are_not_a_matched_tactile_pair(self):
        self.pair({}, {})
        raw = self.audit()
        self.pair({'training_reward_scale': .01}, {'training_reward_scale': .01})
        scaled = self.audit()
        checks = reward_scale_match_checks(raw, scaled)
        self.assertFalse(checks['same_actual_training_reward_scale'])
        self.assertTrue(checks['saved_training_reward_scale_verified'])
        self.assertTrue(checks['initial_final_training_reward_scale_verified'])


if __name__ == '__main__':
    unittest.main()
