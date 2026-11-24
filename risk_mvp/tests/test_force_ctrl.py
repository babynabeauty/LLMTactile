import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from risk_mvp.force_ctrl import FingerForceController, contact_jacobian, finger_contact  # noqa: E402


def planar_two_link(q, l1=.05, l2=.04):
    """Joints about +y, links along +z; returns link-2 origin, a tip point and the origin Jacobian."""
    a1, a12 = q[0], q[0]+q[1]
    o2 = np.array([l1*np.sin(a1), 0, l1*np.cos(a1)])
    tip = o2+np.array([l2*np.sin(a12), 0, l2*np.cos(a12)])
    # origin of link 2 moves with joint 1 only; angular velocity is y for both joints
    J = np.zeros((6, 2))
    J[:3, 0] = np.cross([0, 1, 0], o2)
    J[4, :] = 1.
    return o2, tip, J


def test_contact_jacobian_matches_finite_difference():
    q = np.array([.3, .5])
    o2, tip, J = planar_two_link(q)
    Jc = contact_jacobian(J, o2, tip)
    for k in range(2):
        dq = np.zeros(2)
        dq[k] = 1e-6
        fd = (planar_two_link(q+dq)[1]-tip)/1e-6
        assert np.allclose(Jc[:, k], fd, atol=1e-6)


def test_force_step_is_jacobian_transpose_and_pi_corrects():
    q = np.array([.3, .5])
    o2, tip, J = planar_two_link(q)
    c = FingerForceController(fingers=('index',))
    n = np.array([1., 0, 0])
    kw = dict(q=q, kp=np.array([9., 9.]), effort=np.array([1.1, .4]), lower=-np.ones(2)*3, upper=np.ones(2)*3,
              J_origin=J, link_pos=o2, contact_point=tip, normal_in=n)
    out = c.step('index', 2., measured_N=2., **kw)
    assert np.allclose(out['tau'], np.clip(contact_jacobian(J, o2, tip).T@(2*n), -kw['effort'], kw['effort']))
    assert np.allclose(out['q_target'], q+out['tau']/9.)
    lo = c.step('index', 2., measured_N=1., **kw)['f_cmd']
    assert lo > 2.          # under-force raises the command
    for _ in range(100):
        last = c.step('index', 2., measured_N=1., **kw)['f_cmd']
    assert last <= 2.+.3*1+2.+1e-9   # integral is clamped


def test_approach_closes_until_contact():
    c = FingerForceController(fingers=('middle',))
    q = np.zeros(2)
    kw = dict(kp=np.ones(2), effort=np.ones(2), lower=-np.ones(2), upper=np.ones(2), closed_q=np.array([.8, .6]))
    out = c.step('middle', 1., q, **kw)
    assert out['phase'] == 'approach' and np.allclose(out['q_target'], [.4/60, .4/60])


def test_finger_contact_orients_normal_and_sign():
    rows = [dict(entity_a='ur7e_xhand', entity_b='cube', link_a='right_hand_mid_link2', link_b='cube',
                 force_on_a_world_N=[-2., 0, .1], normal_world=[-1., 0, 0], position_world_m=[-.02, 0, 0])]
    fn, p, n = finger_contact(rows, 'right_hand_mid_link2', object_center=np.zeros(3))
    assert np.isclose(fn, 2.) and np.allclose(n, [1, 0, 0])


def test_batch_controller_matches_single():
    from risk_mvp.force_ctrl import BatchForceController
    q = np.array([.3, .5])
    o2, tip, J = planar_two_link(q)
    single = FingerForceController(fingers=('index',))
    out = single.step('index', 2., measured_N=1.5, q=q, kp=np.array([9., 9.]), effort=np.array([1.1, .4]),
                      lower=-np.ones(2)*3, upper=np.ones(2)*3, J_origin=J, link_pos=o2, contact_point=tip,
                      normal_in=np.array([1., 0, 0]))
    E = 3
    b = BatchForceController(E)
    rep = lambda x: np.repeat(np.asarray(x, float)[None], E, 0)
    qt, info = b.step(rep(2.), rep(q), rep([9., 9.]), rep([1.1, .4]), rep(-np.ones(2)*3), rep(np.ones(2)*3),
                      rep(J), rep(o2), rep(tip), rep([1., 0, 0]), rep(1.5), np.ones(E, bool), rep([.8, .6]))
    assert np.allclose(qt, out['q_target'][None]) and np.allclose(info['tau'], out['tau'][None])
    # no contact -> approach step toward the closed posture
    b2 = BatchForceController(E)
    qt2, _ = b2.step(rep(2.), rep(q), rep([9., 9.]), rep([1.1, .4]), rep(-np.ones(2)*3), rep(np.ones(2)*3),
                     rep(J), rep(o2), rep(tip), rep([1., 0, 0]), rep(0.), np.zeros(E, bool), rep([.8, .6]))
    assert np.allclose(qt2, q+np.clip(np.array([.8, .6])-q, -.4/60, .4/60))
