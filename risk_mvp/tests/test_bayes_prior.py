"""bayes_prior: the handwritten log-normal belief updates correctly from censored outcomes."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from risk_mvp import bayes_prior as B  # noqa: E402

VLM = {'1': {'label': 'fragile', 'confidence': .9}, '0': {'label': 'normal', 'confidence': .8}}


def test_prior_follows_astra():
    m = B.median_threshold(VLM, {'safe_force_N': [5, 10], 'confidence': .7}, 5, [])
    assert abs(m[0]-np.sqrt(50)) < .3 and m[1] < .6*m[0]


def test_break_pulls_every_region_down_and_survival_up():
    strong = {'safe_force_N': [5, 10], 'confidence': .7}
    m0 = B.median_threshold(VLM, strong, 5, [])
    m1 = B.median_threshold(VLM, strong, 5, [dict(region=0, broke=True, lower=1.4, upper=1.5)])
    assert 1.35 < m1[0] < 1.6 and (m1[2:] < .5*m0[2:]).all()          # shared object scale
    weak = {'safe_force_N': [1, 3]}
    w0 = B.median_threshold(VLM, weak, 5, [])
    w1 = B.median_threshold(VLM, weak, 5, [dict(region=0, broke=False, lower=4., upper=np.inf)])
    assert w1[0] > 4. and (w1[2:] > 2*w0[2:]).all()


def test_hazard_is_allocator_ready():
    H = B.hazard(VLM, {'safe_force_N': [1, 4]}, 5, [dict(region=1, broke=False, lower=.5, upper=np.inf)])
    d = np.diff(H, axis=1)
    assert H.shape == (5, 101) and (H[:, 0] == 0).all() and (d >= -1e-12).all() and (np.diff(d, axis=1) >= -1e-9).all()
