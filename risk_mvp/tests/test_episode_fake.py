"""run_trial bookkeeping on a stand-in scene (no physics): settle is not judged, in-hand rotation and
allocator status are logged, phases and lift timing line up."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from risk_mvp.episode import FINGERS, TrialConfig, run_trial  # noqa: E402


class FakePart:
    k = 3

    def region_of(self, p):
        return np.zeros(len(np.atleast_2d(p)), int)


class FakeScene:
    """Each finger reports force = its index in the step counter schedule; the object turns about the
    wrist z axis at 10 deg/s once the settle is over."""

    def __init__(self, E, settle_spike=3.):
        self.E, self.mass_kg, self.t = E, .15, 0
        self.target = np.zeros((E, 18))
        self.kp_env = np.full((E, 18), 3.)
        self.effort = np.ones(18)
        self.lower, self.upper = -np.ones(18)*3, np.ones(18)*3
        self.finger_cols = {'thumb': [6, 7], 'index': [8, 9], 'middle': [10, 11]}
        self.spike = settle_spike
        self.pts = {'thumb': [.03, 0, 0], 'index': [-.03, -.012, 0], 'middle': [-.03, .012, 0]}

    def restore(self, snap):
        self.t = 0

    def wrist_pose(self):
        return np.zeros((self.E, 3)), np.tile([1., 0, 0, 0], (self.E, 1))

    def ik(self, pos, quat):
        return np.zeros((self.E, 6)), np.zeros((self.E, 6))

    def object_pose(self):
        ang = np.radians(10.)*max(0, self.t-15)/60
        R = np.array([[np.cos(ang), -np.sin(ang), 0], [np.sin(ang), np.cos(ang), 0], [0, 0, 1]])
        p = np.zeros((self.E, 3))
        p[:, 2] = .05*min(1., max(0, self.t-15-30)/60)
        return p, np.tile(R, (self.E, 1, 1))

    def q(self):
        return np.zeros((self.E, 18))

    def finger_jacobians(self):
        J = np.zeros((self.E, 6, 2))
        J[:, 0, 0] = 1
        return {f: (J, np.zeros((self.E, 3))) for f in FINGERS}

    def finger_contacts(self):
        fn = self.spike if self.t == 5 else .8
        return {f: dict(fn=np.full(self.E, fn), point=np.tile(self.pts[f], (self.E, 1)),
                        normal=np.tile(-np.array(self.pts[f])/.03, (self.E, 1)), has=np.ones(self.E, bool))
                for f in FINGERS}

    def step(self, tgt):
        self.target = tgt
        self.t += 1


def _snap(sc):
    return dict(contacts={f: dict(point_obj=sc.pts[f], normal_obj=(-np.array(sc.pts[f])/.03).tolist(), region=i)
                          for i, f in enumerate(FINGERS)})


def test_settle_not_judged_and_inhand_logged():
    E = 2
    sc = FakeScene(E)
    cfg = TrialConfig()
    thr = np.full((E, 3), 2.)          # the settle spike (3 N at tick 5, 3 ticks min-window would need 3 ticks)
    sc.spike = 3.
    res = run_trial(sc, _snap(sc), FakePart(), ['equal', 'oracle'], thr, cfg, mass_kg=.15)
    n = cfg.settle_ticks+cfg.preload_ticks+cfg.lift_ticks+cfg.hold_ticks
    assert len(res.log['phase']) == n and res.log['phase'][:cfg.settle_ticks] == ['settle']*cfg.settle_ticks
    assert not res.broke.any()
    assert np.allclose(res.log['settle_peak_N'], 3.)
    # 10 deg/s from the end of the settle: last tick ~ (n-1-15)/60*10 deg
    expect = 10*(n-cfg.settle_ticks)/60
    assert abs(res.log['obj_rot_deg'][-1, 0]-expect) < .5, res.log['obj_rot_deg'][-1, 0]
    assert res.log['obj_rot_deg'][:cfg.settle_ticks].max() == 0
    assert res.log['alloc_ok'].shape == (n, E) and res.log['eq_slack'].shape == (n, E)


def test_sustained_force_after_settle_breaks():
    E = 1
    sc = FakeScene(E, settle_spike=.8)
    thr = np.full((E, 3), .5)            # reported .8 N everywhere: breaks on the first judged ticks
    res = run_trial(sc, _snap(sc), FakePart(), ['equal'], thr, TrialConfig(), mass_kg=.15)
    assert res.broke.all()
    assert res.events[0]['step'] >= TrialConfig().settle_ticks


def test_shake_and_rest_phases():
    E = 1
    sc = FakeScene(E, settle_spike=.8)
    cfg = TrialConfig(shake_ticks=60, risk_safety=4.)
    res = run_trial(sc, _snap(sc), FakePart(), ['oracle'], np.full((E, 3), 5.), cfg, mass_kg=.15)
    ph = res.log['phase']
    n = cfg.settle_ticks+cfg.preload_ticks+cfg.lift_ticks+cfg.hold_ticks
    assert ph.count('shake') == 60 and ph.count('rest') == cfg.rest_ticks and len(ph) == n+60+cfg.rest_ticks
    assert ph[n] == 'shake' and ph[-1] == 'rest'
