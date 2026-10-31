import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from . import multiobject_campaign as campaign
from .multiobject_protocol import make_protocol,bind_demonstrations,protocol_shapes


class CampaignTests(unittest.TestCase):
    def setUp(self):
        self.directory=tempfile.TemporaryDirectory();self.base=Path(self.directory.name)
        self.source=self.base/'source';(self.source/'ur7e_xhand_sim').mkdir(parents=True)
        (self.source/'ur7e_xhand_sim/example.py').write_text('value=1\n')
        demo=self.base/'control.jsonl';demo.write_text('verified fixture\n')
        mapping={shape:dict(trajectory_path=str(demo),demonstration_evidence=dict(
            task_success=True,mass_kg=.05,friction=.8)) for shape in protocol_shapes(make_protocol())}
        self.protocol=bind_demonstrations(make_protocol(),mapping)
        self.input=self.base/'protocol.json';self.input.write_text(json.dumps(self.protocol))
        self.output=self.base/'campaign'

    def tearDown(self):self.directory.cleanup()

    def initialize(self):
        return campaign.initialize_campaign(self.input,self.output,root=self.source)

    def test_commands_put_test_after_validation_selection(self):
        self.assertLess(campaign.STEPS.index('selection'),campaign.STEPS.index('test'))
        val1=campaign.step_command('validation_stage1',self.output,python='python')
        val2=campaign.step_command('validation_stage2',self.output,python='python')
        test=campaign.step_command('test',self.output,python='python')
        self.assertNotIn('--fixed-gains',val1)
        self.assertEqual(val2[-1],'--fixed-gains')
        self.assertIn('--selection-json',test)
        self.assertNotIn('--training-root',test)
        self.assertNotIn('--models',test)

    def test_freeze_and_exact_resume(self):
        manifest=self.initialize()
        self.assertEqual(manifest['protocol_sha256'],campaign.sha256(self.input))
        with self.assertRaises(FileExistsError):self.initialize()
        resumed=campaign.initialize_campaign(self.input,self.output,resume=True,root=self.source)
        self.assertEqual(manifest,resumed)
        (self.source/'ur7e_xhand_sim/example.py').write_text('value=2\n')
        with self.assertRaisesRegex(ValueError,'source tree changed'):
            campaign.initialize_campaign(self.input,self.output,resume=True,root=self.source)

    def test_demonstration_change_blocks_resume(self):
        self.initialize();Path(self.protocol['objects'][0]['trajectory_path']).write_text('changed')
        with self.assertRaisesRegex(ValueError,'hash mismatch'):
            campaign.initialize_campaign(self.input,self.output,resume=True,root=self.source)

    def test_generated_rigid_mesh_is_frozen_and_mutation_blocks_resume(self):
        relative='ur7e_xhand_sim/generated/rigid_shapes/hex_prism.obj'
        mesh=self.source/relative;mesh.parent.mkdir(parents=True);mesh.write_text('v 0 0 0\n')
        manifest=self.initialize()
        self.assertIn(relative,manifest['source_sha256'])
        self.assertEqual((self.output/'source_snapshot'/relative).read_bytes(),mesh.read_bytes())
        mesh.write_text('v 1 0 0\n')
        with self.assertRaisesRegex(ValueError,'source tree changed'):
            campaign.initialize_campaign(self.input,self.output,resume=True,root=self.source)

    def test_unfinished_directory_is_archived_only_explicitly(self):
        self.initialize();folder=self.output/'eval/test';folder.mkdir(parents=True)
        (folder/'evidence.txt').write_text('keep partial evidence')
        with self.assertRaises(FileExistsError):
            campaign.archive_incomplete_outputs(self.output,'test',retry_incomplete=False,protocol=self.protocol)
        campaign.archive_incomplete_outputs(self.output,'test',retry_incomplete=True,protocol=self.protocol)
        saved=list((self.output/'incomplete_attempts').rglob('evidence.txt'))
        self.assertEqual(len(saved),1);self.assertEqual(saved[0].read_text(),'keep partial evidence')
        self.assertFalse(folder.exists())

    def run_mock_campaign(self, fail_step=None, recovered=()):
        manifest=self.initialize();produced=set(recovered);called=[]
        def artifacts(output,step,protocol):
            if step not in produced:raise FileNotFoundError(step)
            return {'fixture':step}
        def commands(step,output):
            return None if step.startswith('audit_') or step=='selection' else [step]
        def process(command,log_path,**kwargs):
            step=command[0];called.append(step);log_path.write_text('fixture log\n')
            if step==fail_step:raise subprocess.CalledProcessError(7,command)
            produced.add(step)
        def local(step,output,protocol):
            called.append(step)
            if step==fail_step:raise ValueError('audit rejected')
            produced.add(step)
        with patch.object(campaign,'artifact_hashes',side_effect=artifacts),patch.object(campaign,'step_command',side_effect=commands):
            if fail_step:
                with self.assertRaises((subprocess.CalledProcessError,ValueError)):
                    campaign.execute_campaign(self.output,manifest,root=self.source,
                        process_runner=process,local_runner=local)
            else:
                campaign.execute_campaign(self.output,manifest,root=self.source,
                    process_runner=process,local_runner=local)
        return called,json.loads((self.output/'status.json').read_text())

    def test_full_order_and_only_complete_after_report(self):
        called,state=self.run_mock_campaign()
        self.assertEqual(called,list(campaign.STEPS))
        self.assertEqual(state['status'],'completed')
        self.assertEqual(state['completed_steps'],list(campaign.STEPS))

    def test_child_failure_stops_before_test(self):
        called,state=self.run_mock_campaign('validation_stage2')
        self.assertEqual(state['status'],'failed')
        self.assertNotIn('selection',called);self.assertNotIn('test',called)
        self.assertEqual(state['completed_steps'],list(campaign.STEPS[:3]))

    def test_validation_audit_failure_stops_before_test(self):
        called,state=self.run_mock_campaign('audit_validation_stage1')
        self.assertEqual(state['status'],'failed');self.assertNotIn('test',called)

    def test_recover_completed_child_does_not_repeat_test(self):
        called,state=self.run_mock_campaign(recovered=campaign.STEPS[:-1])
        self.assertEqual(called,['report']);self.assertEqual(state['status'],'completed')
        marker=json.loads((self.output/'steps/test.json').read_text())
        self.assertTrue(marker['recovered_completed_artifacts'])


if __name__=='__main__':unittest.main()
