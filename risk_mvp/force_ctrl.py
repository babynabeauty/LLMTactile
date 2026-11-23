"""Target normal force per finger -> finger joint position targets (MVP step 2).

The fingers are position-PD joints (Sandbox: kp 3 N m/rad x grip gain). Commanding
    q_target = q + tau / kp,   tau = J_c^T (f_cmd n)
makes the PD torque equal tau whatever the deflection, i.e. a joint-torque source that
presses the contact point with f_cmd along the inward normal n. J_c is the contact-point
Jacobian built from the link-origin Jacobian: J_c = J_lin - [p - o]x J_ang.
A per-finger PI on the measured solver normal force corrects model errors (finger gravity,
damping, Jacobian/normal errors). Before first contact the finger closes at a fixed joint
speed toward the demonstration's closed posture.

Pure numpy, no Genesis: the simulator adapter passes Jacobians, contacts and joint states.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import numpy as np

# Flexion joints driven per finger; the thumb rotation (bend) and index abduction stay at the demo pose.
FINGER_JOINTS = {
    'thumb': ('right_hand_thumb_rota_joint1', 'right_hand_thumb_rota_joint2'),
    'index': ('right_hand_index_joint1', 'right_hand_index_joint2'),
    'middle': ('right_hand_mid_joint1', 'right_hand_mid_joint2'),
}
FINGER_LINKS = {'thumb': 'right_hand_thumb_rota_link2', 'index': 'right_hand_index_rota_link2',
                'middle': 'right_hand_mid_link2'}


def skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def contact_jacobian(J_origin: np.ndarray, link_pos: np.ndarray, point: np.ndarray) -> np.ndarray:
    """(3, n) linear Jacobian of a point rigidly attached to the link; J_origin is (6, n), rows [lin; ang]."""
    return J_origin[:3]-skew(np.asarray(point)-np.asarray(link_pos))@J_origin[3:]


@dataclass
class PIGains:
    kp: float = .3          # proportional on force error (dimensionless, added to the target)
    ki: float = 3.          # 1/s
    i_limit_N: float = 2.   # anti-windup on the integral term
    close_speed_rad_s: float = .4


@dataclass
class FingerState:
    integ: float = 0.
    in_contact: bool = False


@dataclass
class FingerForceController:
    fingers: Sequence[str] = ('thumb', 'index', 'middle')
    gains: PIGains = field(default_factory=PIGains)
    dt: float = 1/60

    def __post_init__(self):
        self.state = {f: FingerState() for f in self.fingers}

    def reset(self):
        self.state = {f: FingerState() for f in self.fingers}

    def step(self, finger: str, target_N: float, q: np.ndarray, kp: np.ndarray, effort: np.ndarray,
             lower: np.ndarray, upper: np.ndarray, J_origin: Optional[np.ndarray] = None,
             link_pos: Optional[np.ndarray] = None, contact_point: Optional[np.ndarray] = None,
             normal_in: Optional[np.ndarray] = None, measured_N: float = 0., closed_q: Optional[np.ndarray] = None,
             contact_min_N: float = .05) -> Dict:
        """One 60 Hz update for one finger. q/kp/effort/lower/upper/closed_q are over the finger's driven
        joints; J_origin is (6, n_driven) for those joints. Returns the joint targets and diagnostics."""
        st, g = self.state[finger], self.gains
        q = np.asarray(q, float)
        touching = contact_point is not None and measured_N > contact_min_N
        if touching:
            st.in_contact = True
        if not st.in_contact or contact_point is None:
            # approach: close toward the demo's closed posture at a fixed joint speed
            step = g.close_speed_rad_s*self.dt
            goal = q if closed_q is None else np.asarray(closed_q, float)
            q_t = q+np.clip(goal-q, -step, step) if closed_q is not None else q
            return dict(q_target=np.clip(q_t, lower, upper), tau=np.zeros_like(q), f_cmd=0., phase='approach')
        err = target_N-measured_N
        st.integ = float(np.clip(st.integ+g.ki*err*self.dt, -g.i_limit_N, g.i_limit_N))
        f_cmd = max(0., target_N+g.kp*err+st.integ)
        Jc = contact_jacobian(J_origin, link_pos, contact_point)
        n = np.asarray(normal_in, float)/max(np.linalg.norm(normal_in), 1e-12)
        tau = np.clip(Jc.T@(f_cmd*n), -effort, effort)
        q_t = np.clip(q+tau/np.asarray(kp, float), lower, upper)
        return dict(q_target=q_t, tau=tau, f_cmd=f_cmd, phase='force', err=err)


def finger_contact(rows: Sequence[dict], link_name: str, object_entity: str = 'cube',
                   object_center: Optional[np.ndarray] = None):
    """From Sandbox.evaluate()['solver_contacts']: (normal force N, point, inward unit normal) of all
    contacts between one finger link and the object, force-weighted. None if no contact."""
    pts, forces = [], []
    for c in rows:
        if {c['entity_a'], c['entity_b']} != {object_entity, 'ur7e_xhand'}:
            continue
        if link_name not in (c['link_a'], c['link_b']):
            continue
        on_a = np.asarray(c['force_on_a_world_N'], float)
        on_object = -on_a if c['link_a'] == link_name else on_a   # force the finger applies to the object
        pts.append(np.asarray(c['position_world_m'], float))
        forces.append(on_object)
    if not pts:
        return 0., None, None
    pts, forces = np.asarray(pts), np.asarray(forces)
    total = forces.sum(0)
    mag = np.linalg.norm(forces, axis=1)
    point = (pts*mag[:, None]).sum(0)/max(mag.sum(), 1e-12)
    # inward normal: the solver normal of the strongest contact, oriented into the object
    rows_on = [c for c in rows if link_name in (c['link_a'], c['link_b'])
               and {c['entity_a'], c['entity_b']} == {object_entity, 'ur7e_xhand'}]
    n = np.asarray(rows_on[int(np.argmax(mag))]['normal_world'], float)
    ref = (object_center-point) if object_center is not None else total
    if n@ref < 0:
        n = -n
    return float(max(total@n, 0.)), point, n
