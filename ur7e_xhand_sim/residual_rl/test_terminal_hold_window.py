"""CPU contracts for final-three-second evaluation with intact command logs."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

from .core import interval_overlap


path = Path(__file__).with_name('env.py')
node = next(node for node in ast.parse(path.read_text()).body
            if isinstance(node,ast.FunctionDef) and node.name=='trajectory_hold_window')
namespace = {'np':np}
exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(path),'exec'),namespace)
hold_window = namespace['trajectory_hold_window']


class TestTerminalHoldWindow(unittest.TestCase):
    def trajectory(self, hold_seconds=3.):
        times = np.arange(round((3+hold_seconds)*40))/40
        phases = np.where(times<1,'preload',np.where(times<2,'test_lift',
                  np.where(times<3,'lift','hold')))
        return SimpleNamespace(times=times,phases=phases,
            duration_s=3+hold_seconds,
            targets=np.arange(len(times)*18).reshape(-1,18),
            phase_starts={'preload':0.,'test_lift':1.,'lift':2.,'hold':3.})

    def test_legacy_exact_hold_is_unchanged_and_long_hold_still_rejected(self):
        self.assertEqual(hold_window(self.trajectory()),(3.,3.))
        with self.assertRaisesRegex(ValueError,'final 3-second hold'):
            hold_window(self.trajectory(4.))

    def test_explicit_terminal_window_preserves_every_command_and_phase(self):
        trajectory = self.trajectory(4.5)
        before = {key:getattr(trajectory,key).copy() for key in ('times','phases','targets')}
        recorded,evaluated = hold_window(trajectory,3.)
        self.assertEqual((recorded,evaluated),(3.,4.5))
        for key,value in before.items():
            np.testing.assert_array_equal(getattr(trajectory,key),value)
        self.assertEqual(len(trajectory.times),300)
        self.assertEqual(np.count_nonzero(trajectory.phases=='hold'),180)

    def test_final_window_counts_exact_three_seconds_on_60hz_sampling(self):
        trajectory = self.trajectory(4.525)
        _,start = hold_window(trajectory,3.)
        sample_end = np.arange(1,int(np.ceil(trajectory.duration_s*60))+1)/60
        overlap = interval_overlap(sample_end-1/60,sample_end,start,trajectory.duration_s)
        self.assertAlmostEqual(float(overlap.sum()),3.,places=12)
        # A large force in early recorded hold is absent from the final window.
        force = np.where(sample_end<start,100.,2.)
        self.assertAlmostEqual(float((force*overlap).sum()/overlap.sum()),2.,places=12)

    def test_short_or_interrupted_hold_cannot_pass(self):
        with self.assertRaisesRegex(ValueError,'shorter'):
            hold_window(self.trajectory(2.975),3.)
        for changed_time in (3.5,4.5,6.5):
            trajectory = self.trajectory(4.)
            trajectory.phases[np.isclose(trajectory.times,changed_time)]='lift'
            with self.subTest(time=changed_time),self.assertRaisesRegex(ValueError,'uninterrupted'):
                hold_window(trajectory,3.)

    def test_criterion_duration_cannot_be_silently_relaxed(self):
        for duration in (0.,1.,2.,4.,np.nan,np.inf,True,'3',[3.]):
            with self.subTest(duration=duration),self.assertRaises(ValueError):
                hold_window(self.trajectory(),duration)

    def test_roundoff_does_not_select_previous_lift_command(self):
        trajectory = self.trajectory()
        trajectory.duration_s -= 1e-14
        _,start = hold_window(trajectory,3.)
        self.assertAlmostEqual(start,3.,places=12)


if __name__ == '__main__':
    unittest.main()
