"""CPU tests for regions, damage and the allocator (no Genesis). Run: python -m pytest risk_mvp/tests -q"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from risk_mvp.regions import object_mesh, partition, sample_instance          # noqa: E402
from risk_mvp.damage import DamageJudge, region_forces                         # noqa: E402
from risk_mvp.allocator import Allocator, Contacts, GRID_N, gravity_wrench    # noqa: E402

CUBOID = dict(shape='cuboid', size_m=[.045, .06, .045])
# Tripod hold contacts measured on the 5090 (risk_mvp/probe_env.py, cuboid_tripod demo), object frame.
PROBE = {'thumb': [.0222, -.0007, .0025], 'index': [-.0223, -.013, .0035], 'middle': [-.0223, .011, .0035]}


@pytest.mark.parametrize('spec', [CUBOID, dict(shape='cylinder', radius_m=.03, height_m=.09),
                                  dict(shape='sphere', radius_m=.03), dict(shape='flat_box', size_m=[.08, .06, .025]),
                                  dict(shape='hex_prism', across_flats_m=.045, height_m=.045)])
@pytest.mark.parametrize('k', [3, 6])
def test_partition_covers_surface(spec, k):
    m = object_mesh(spec)
    p = partition(m, k, seed=1)
    assert p.k == k and np.isclose(p.area.sum(), m.area, rtol=1e-6) and (p.area > 0).all()
    assert np.allclose(np.linalg.norm(p.normal, axis=1), 1)
    # contact labelling uses the same rule as face labelling
    assert (p.region_of(m.triangles_center) == p.face_labels).all()
    assert p.features().shape == (k, 8)


def test_partition_separates_index_and_middle():
    p = partition(object_mesh(CUBOID), 4, seed=0, separate=[PROBE['index'], PROBE['middle']])
    assert p.region_of(PROBE['index']) != p.region_of(PROBE['middle'])


def test_instances_are_automatic_and_correlated():
    p = partition(object_mesh(CUBOID), 6, seed=0)
    rng = np.random.default_rng(0)
    inst = [sample_instance(p, rng) for _ in range(2000)]
    base = np.array([i.base_N for i in inst])
    assert .5 <= base.min() and base.max() <= 10
    weak = np.mean([i.weak_region is not None for i in inst])
    assert .15 < weak < .25
    # neighbouring regions are more alike than far ones
    logc = np.log(np.stack([i.coeff for i in inst if i.weak_region is None]))
    d = np.linalg.norm(p.centers[:, None]-p.centers[None], axis=-1)
    iu = np.triu_indices(6, 1)
    corr = np.corrcoef(logc.T)[iu]
    assert np.corrcoef(d[iu], corr)[0, 1] < -.5
    forced = sample_instance(p, rng, weak_region=2)
    assert forced.weak_region == 2 and .3 <= forced.coeff[2] <= .6


def test_slow_press_breaks_only_that_region_near_threshold():
    thr = np.array([[2., 5., 8.]])
    j = DamageJudge(thr)
    for t, f in enumerate(np.arange(0, 6, .01)):
        F = np.array([[f, .3*f, .1*f]])
        if j.update(F, t).any():
            break
    ev = j.event[0]
    assert ev['region'] == 0 and 2. < ev['sustained_N'] <= 2.03 and ev['lower_N'] <= 2.
    rows = j.archive(0)
    assert rows[0]['outcome'] == 'broke' and rows[1]['outcome'] == 'survived'
    assert rows[1]['lower_N'] < 5


def test_spike_shorter_than_hold_does_not_break():
    j = DamageJudge(np.array([[2.]]))
    seq = [1, 1, 9, 9, 1, 1, 1]
    assert not any(j.update(np.array([[f]]), t).any() for t, f in enumerate(seq))
    assert j.archive(0)[0]['outcome'] == 'survived' and j.archive(0)[0]['lower_N'] == 1


def test_region_forces_batched():
    r = np.array([[0, 1, 1], [2, 2, 0]])
    fn = np.array([[1., 2., 3.], [4., 5., 6.]])
    v = np.array([[1, 1, 0], [1, 1, 1]], bool)
    assert np.allclose(region_forces(r, fn, v, 3), [[1, 2, 0], [6, 0, 9]])


def tripod_contacts(region):
    names = ['thumb', 'index', 'middle']
    pos = np.array([PROBE[n] for n in names])
    normal = np.array([[-1., 0, 0], [1., 0, 0], [1., 0, 0]])
    return Contacts(pos, normal, np.asarray(region), names)


def weibull_hazard(theta, k=3.):
    return (GRID_N[None, :]/np.asarray(theta, float)[:, None])**k


def symmetric_contacts():
    pos = np.array([[.0225, 0., 0.], [-.0225, -.012, 0.], [-.0225, .012, 0.]])
    return Contacts(pos, np.array([[-1., 0, 0], [1., 0, 0], [1., 0, 0]]), np.array([0, 1, 2]),
                    ['thumb', 'index', 'middle'])


def test_slip_margin_lets_load_follow_the_weak_side():
    """Scheme 1 on the measured tripod contacts: a weak index or middle region gives up slip margin first,
    the load moves to the other finger, and the two cases mirror each other (the swap experiment)."""
    w = gravity_wrench(.2, np.eye(3))
    base = Allocator(tripod_contacts([0, 1, 2]), 3, mode='tactile').solve(w, mu=.8, fmax=15.)
    assert base['feasible'] and max(base['slip_margin_deficit'].values()) < 1e-6
    out = {}
    for weak in ('index', 'middle'):
        theta = np.array([30., 3. if weak == 'index' else 30., 3. if weak == 'middle' else 30.])
        a = Allocator(tripod_contacts([0, 1, 2]), 3, mode='risk')
        times, reps = [], []
        for _ in range(20):
            reps.append(a.solve(w, mu=.8, fmax=15., margin=.1, hazard=weibull_hazard(theta), eps=.05))
            times.append(reps[-1]['solve_s'])
        rep = reps[-1]
        assert rep['feasible'] and rep['damage_prob_bound'] <= .05+1e-6
        out[weak] = rep['finger_normal']
        print(weak, 'weak:', {k: round(v, 3) for k, v in rep['finger_normal'].items()},
              'baseline:', {k: round(v, 3) for k, v in base['finger_normal'].items()},
              'warm solve ms %.2f' % (1e3*np.median(times[1:])))
        assert np.median(times[1:]) < .01
    b = base['finger_normal']
    assert out['index']['index'] < b['index']-.2 and out['index']['middle'] > b['middle']+.2
    assert out['middle']['middle'] < b['middle']-.2 and out['middle']['index'] > b['index']+.2


def test_probe_geometry_limits_the_shift():
    """Measured tripod contacts, point contacts: rotation about the grasp axis fixes how the weight is
    shared between index and middle, so each has a floor of (its weight share)/mu."""
    w = gravity_wrench(.2, np.eye(3))
    rep = Allocator(tripod_contacts([0, 1, 2]), 3, mode='risk').solve(
        w, mu=.5, fmax=15., margin=.1, hazard=weibull_hazard([30., 30., 1.5]))
    assert not rep['feasible'] and rep['infeasible']['binding_region'] == 2


def test_equal_and_tactile_ignore_weakness_and_oracle_respects_thresholds():
    w = gravity_wrench(.2, np.eye(3))
    eq = Allocator(tripod_contacts([0, 1, 2]), 3, mode='equal').solve(w, mu=.5, fmax=15.)
    assert eq['feasible'] and np.isclose(eq['finger_normal']['index'], eq['finger_normal']['middle'], atol=1e-4)
    tac = Allocator(tripod_contacts([0, 1, 2]), 3, mode='tactile').solve(w, mu=.5, fmax=15.)
    assert tac['feasible'] and tac['fn'].sum() <= eq['fn'].sum()+1e-4
    orc = Allocator(tripod_contacts([0, 1, 2]), 3, mode='oracle').solve(w, mu=.8, fmax=15., margin=.1,
                                                                       limit=[10., .8, 10.])
    assert orc['feasible'] and orc['region_force'][1] <= .7+1e-4


def test_infeasible_reports_slack():
    w = gravity_wrench(2., np.eye(3))  # 2 kg with mu .3 and a 1 N cap cannot be held
    rep = Allocator(tripod_contacts([0, 1, 2]), 3, mode='risk').solve(
        w, mu=.3, fmax=1., hazard=weibull_hazard([3., 3., 3.]))
    assert not rep['feasible'] and 'infeasible' in rep
