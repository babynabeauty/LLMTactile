"""Relabeling configuration checks without starting a physics scene."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from . import v1_label
from .v1_bases import load_experiment


class LabelProtocolTest(unittest.TestCase):
    def test_train_only_labels_preserve_armature_in_generated_experiment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            demo = root/'demo.json'
            demo.write_text('{}')
            runs = [dict(run_id=f'pinch_{split}', split=split, object_id='cuboid', hand_shape='pinch',
                         demo=str(demo), placement={}, object_spec={'shape': 'cuboid'},
                         trajectory_path='unused.jsonl', trajectory_sha256='unused',
                         object_initial_position_m=[0, 0, 0], object_initial_quaternion_wxyz=[1, 0, 0, 0])
                    for split in ('train', 'test')]
            (root/'runs.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in runs))
            output = root/'labels'
            with patch.object(v1_label, 'label_group', return_value={
                    'pinch_train': dict(usable=True, label='S')}) as label:
                v1_label.main(['--collection', str(root), '--output', str(output),
                               '--split', 'train', '--finger-armature', '.003', '--lanes', '2'])
                self.assertEqual([r['run_id'] for r in label.call_args.args[0]], ['pinch_train'])
                self.assertEqual(label.call_args.kwargs['finger_armature'], .003)
            experiment = load_experiment(output/'experiment_train.json')
            self.assertEqual(experiment['finger_armature_kg_m2'], .003)
            self.assertEqual(experiment['success_rule'], 'lift_position_hold_v3')
            self.assertFalse((output/'experiment_test.json').exists())
            self.assertEqual(json.loads((output/'label_protocol.json').read_text())['split'], 'train')


if __name__ == '__main__':
    unittest.main()
