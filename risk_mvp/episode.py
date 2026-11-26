"""One batched trial from a snapshot (MVP step 3): settle -> allocate/preload -> lift 5 cm -> hold 2 s.

Every env starts from the same already-touching grasp (snapshot.py). Per env it gets an allocation
method, hidden region thresholds, object mass and contact friction. At 60 Hz:
  - every `alloc_every` ticks the env's Allocator gives per-finger normal targets (the contact geometry
    and regions are those of the snapshot, in the object frame; the external wrench is gravity in the
    current object frame, mass x (1 + mass_margin));
  - settle (0.25 s): the force controller holds the touch force so the restored contacts settle; the
    restore made index/middle drop to 0 N and re-hit at ~1.1 N within the first 0.25 s (065-a). Not
    judged for damage; its peak region force is logged (settle_peak_N) to check it stays harmless;
  - targets ramp from the touch force to the allocation during the preload, then the arm lifts and holds;
  - BatchForceController tracks the targets; the solver's per-finger normal forces are summed per region
    (each contact point mapped to its region in the object frame) and fed to DamageJudge.
Also logged per tick: the object pose relative to the wrist since the start of the preload (in-hand
rotation obj_rot_deg and slip obj_slip_mm; the allocator re-plans with the current object orientation, so
a slow in-hand rotation shows up as drifting targets) and the allocator report (alloc_ok, eq_slack = wrench
it could not supply within the friction cone, risk_slack = excess over a damage limit).
Outcome per env: broke (first damage event), dropped (lifted less than 60 % of the planned height or a
finger lost contact in the last 0.5 s), success = neither. The archive rows follow damage.DamageJudge.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from risk_mvp.allocator import Allocator, Contacts, gravity_wrench
from risk_mvp.damage import DamageJudge
from risk_mvp.force_ctrl import BatchForceController

FINGERS = ('thumb', 'index', 'middle')


def safe_normal(n, point_obj):
    """Unit inward normal; if the stored one is missing/degenerate, point horizontally from the contact
    toward the object's vertical axis."""
    n = np.asarray(n, float)
    if np.all(np.isfinite(n)) and np.linalg.norm(n) > .5:
        return n/np.linalg.norm(n)
    d = -np.asarray(point_obj, float)
    d[2] = 0.
    return d/max(np.linalg.norm(d), 1e-9)


@dataclass
class TrialConfig:
    mu_hat: float = .5            # allocator's slip-limit friction estimate
    safety: float = 2.            # slip-margin factor (soft cone mu_hat / safety)
    min_safety: float = 1.        # hard cone at mu_hat / min_safety: a slip margin no method may give up.
                                  # With 1 the risk allocators (and the oracle) gave up the whole margin when
                                  # the belief was tight and dropped 19-36 % of the grasps (eval_seq): the point
                                  # contact model leaves out in-hand rotation, lift acceleration, tracking error
    risk_safety: float = None     # soft-cone target for the belief-aware modes (risk / oracle / cap): with a higher
                                  # target they take as much slip margin as the damage budget allows (an accurate
                                  # belief buys margin, a pessimistic one does not). None: safety, as equal/tactile.
                                  # With it set, these modes weigh missing margin at risk_slip_weight (vs 10) and
                                  # the damage-budget slack at 1000: hold the object >> stay within the budget >>
                                  # buy margin, so margin is bought until the budget binds (CPU check: strong
                                  # belief thumb 2.4 N, pessimistic 1.3 N, weak side still unloaded)
    risk_slip_weight: float = 100.
    margin_N: float = .1          # force-estimate margin in the damage constraint
    oracle_margin_N: float = None  # the oracle's margin under the true thresholds (None: margin_N). 0.1 N is
                                   # below the force-tracking error: the oracle broke 11-17 % (eval_seq)
    eps: float = .05              # damage budget for the risk mode
    mass_margin: float = .1
    fmax_N: float = 8.
    touch_N: float = .5            # snapshot touch force (snapshot.py --touch; thumb twice), used only when the
                                   # snapshot has no per-finger contact force
    settle_ticks: int = 15         # hold the touch force after the restore, not judged
    start_thumb_N: float = .4      # settle/preload start: the snapshot's finger forces scaled down together so
                                   # the thumb starts here. The snapshot closes at thumb ~1.1 N, more than the
                                   # weakest objects take (cups from 1 N broke at the first judged tick).
                                   # Scaling all three keeps the forces balanced (halving only the thumb
                                   # made it lose contact). <= 0: the snapshot forces as they are.
    preload_ticks: int = 60        # 1 s: with 0.5 s the thumb ramping up pushed index/middle ~0.1 N over target
                                   # before the force controller caught up (065-d oracle broke 5/16)
    lift_ticks: int = 60
    shake_ticks: int = 0           # disturbance after the hold: the wrist shakes sideways (x) and vertically (z),
    shake_hz: float = 3.           # a quarter period apart, shake_mm amplitude (8 mm at 3 Hz: ~2.8 m/s^2 peak),
    shake_mm: float = 8.           # then rest_ticks still before the drop check. Holding still only never punished
    rest_ticks: int = 30           # a grip too loose to survive a bump, so the most cautious method always won.
    hold_ticks: int = 120
    lift_m: float = .05
    alloc_every: int = 4


@dataclass
class TrialResult:
    broke: np.ndarray
    dropped: np.ndarray
    success: np.ndarray
    events: list
    archive: list
    log: dict = field(default_factory=dict)


def run_trial(sc, snap: dict, part, modes: Sequence[str], thresholds: np.ndarray, cfg: TrialConfig = TrialConfig(),
              mass_kg=None, hazard: Optional[np.ndarray] = None, cap_N: Optional[np.ndarray] = None,
              scale: Optional[np.ndarray] = None) -> TrialResult:
    """scale (E,): multiplies the allocated finger forces (data collection: squeeze harder than the
    allocator would, so thresholds get crossed and the archive has breaks as well as survivals)."""
    E = sc.E
    scale = np.ones(E) if scale is None else np.broadcast_to(np.asarray(scale, float), (E,))
    assert len(modes) == E and thresholds.shape == (E, part.k)
    sc.restore(snap)
    mass = np.broadcast_to(np.asarray(sc.mass_kg if mass_kg is None else mass_kg, float), (E,))
    pos = np.array([snap['contacts'][f]['point_obj'] for f in FINGERS])
    nrm = np.array([snap['contacts'][f]['normal_obj'] for f in FINGERS], float)
    nrm = np.array([safe_normal(n, p) for n, p in zip(nrm, pos)])
    reg = np.array([snap['contacts'][f]['region'] for f in FINGERS])
    def make_alloc(m):
        kw = dict(slip_weight=cfg.risk_slip_weight, risk_slack_weight=1000.) \
            if cfg.risk_safety and m in ('risk', 'oracle', 'cap') else {}
        return Allocator(Contacts(pos, nrm, reg, list(FINGERS)), part.k, mode=m, **kw)
    allocs = [make_alloc(m) for m in modes]
    ctrl = {f: BatchForceController(E) for f in FINGERS}
    for f in FINGERS:
        ctrl[f].in_contact[:] = True
    judge = DamageJudge(thresholds)
    # Start from the forces the snapshot was closed with (thumb ~2x index/middle). Settling all three at
    # touch_N let the thumb drop to 0 N and pushed index/middle over their targets when it came back.
    start_N = np.array([snap['contacts'][f].get('fn', np.nan) for f in FINGERS], float)
    start_N = np.where(np.isfinite(start_N) & (start_N > .1), start_N, cfg.touch_N*np.array([2., 1., 1.]))
    if cfg.start_thumb_N > 0:
        start_N = start_N*min(1., cfg.start_thumb_N/start_N[0])
    targets = np.tile(start_N, (E, 1))
    alloc_fn = targets.copy()
    feasible = np.ones(E, bool)
    # Lift from the commanded arm target, not the measured (sagging) joints: starting from the measured
    # ones dropped the hand by the ~4 mm sag at lift start and made the fingers lose and re-hit the object.
    q_arm0 = sc.target[:, :6].copy()
    wpos, wquat = sc.wrist_pose()
    q_here, _ = sc.ik(wpos, wquat)
    q_up, _ = sc.ik(wpos+np.array([0, 0, cfg.lift_m]), wquat)
    q_lift = q_arm0+(q_up-q_here)
    S = cfg.settle_ticks
    n_shake = cfg.shake_ticks+(cfg.rest_ticks if cfg.shake_ticks else 0)
    n_ticks = S+cfg.preload_ticks+cfg.lift_ticks+cfg.hold_ticks+n_shake
    t_shake = S+cfg.preload_ticks+cfg.lift_ticks+cfg.hold_ticks
    log = dict(live_region_mismatch=np.zeros((n_ticks, E, 3), bool), fn=np.zeros((n_ticks, E, 3)), ft=np.zeros((n_ticks, E, 3)), target=np.zeros((n_ticks, E, 3)), region_F=np.zeros((n_ticks, E, part.k)),
               has=np.zeros((n_ticks, E, 3), bool), phase=[], obj_rot_deg=np.zeros((n_ticks, E)),
               obj_slip_mm=np.zeros((n_ticks, E)), alloc_ok=np.ones((n_ticks, E), bool),
               eq_slack=np.zeros((n_ticks, E)), risk_slack=np.zeros((n_ticks, E)), settle_peak_N=np.zeros(E))
    eq_slack, risk_slack, ok_now = np.zeros(E), np.zeros(E), np.ones(E, bool)

    from risk_mvp.scene import quat_to_mat

    def in_hand():
        """Object pose in the wrist frame: (p (E,3), R (E,3,3))."""
        pw, qw = sc.wrist_pose()
        Rw = quat_to_mat(qw)
        p_o, R_o = sc.object_pose()
        return np.einsum('eji,ej->ei', Rw, p_o-pw), np.einsum('eji,ejk->eik', Rw, R_o)

    start_center = None
    for t in range(n_ticks):
        if t < S:
            phase = 'settle'
        else:
            u = t-S
            phase = ('preload' if u < cfg.preload_ticks else 'lift' if u < cfg.preload_ticks+cfg.lift_ticks
                     else 'hold' if t < t_shake else 'shake' if t < t_shake+cfg.shake_ticks else 'rest')
        if t == t_shake and cfg.shake_ticks:
            # joint offsets for a small wrist displacement along x and z (linearised around the hold pose)
            wp, wq = sc.wrist_pose()
            q0, _ = sc.ik(wp, wq)
            d = cfg.shake_mm/1000.
            dq_x = sc.ik(wp+np.array([d, 0, 0]), wq)[0]-q0
            dq_z = sc.ik(wp+np.array([0, 0, d]), wq)[0]-q0
            q_hold = sc.target[:, :6].copy()
        if t == S:
            start_center, _ = sc.object_pose()
            p_h0, R_h0 = in_hand()
        c = sc.finger_contacts()
        center, R = sc.object_pose()
        if t >= S and (t-S) % cfg.alloc_every == 0:
            for e in range(E):
                w = gravity_wrench(mass[e]*(1+cfg.mass_margin), R[e])
                soft = cfg.risk_safety if cfg.risk_safety and modes[e] in ('risk', 'oracle', 'cap') else cfg.safety
                kw = dict(w_ext=w, mu=cfg.mu_hat/cfg.min_safety, fmax=cfg.fmax_N, margin=cfg.margin_N,
                          safety=max(1., soft/cfg.min_safety))             # soft cone at mu_hat / soft
                if modes[e] == 'risk':
                    kw.update(hazard=hazard[e], eps=cfg.eps)
                elif modes[e] == 'oracle':
                    kw.update(limit=thresholds[e])
                    if cfg.oracle_margin_N is not None:
                        kw.update(margin=cfg.oracle_margin_N)
                elif modes[e] == 'cap':
                    kw.update(limit=np.full(part.k, cap_N[e] if cap_N is not None else 2.))
                rep = allocs[e].solve(**kw)
                alloc_fn[e] = [scale[e]*rep['finger_normal'][f] for f in FINGERS]
                feasible[e] &= rep['feasible']
                ok_now[e], eq_slack[e], risk_slack[e] = rep['feasible'], rep['eq_slack'], rep['risk_slack']
        ramp = 0. if t < S else min(1., (t-S+1)/cfg.preload_ticks)
        targets = start_N+ramp*(alloc_fn-start_N)
        # force control
        J = sc.finger_jacobians()
        q = sc.q()
        tgt = sc.target.copy()
        for i, f in enumerate(FINGERS):
            cols = sc.finger_cols[f]
            qt, _ = ctrl[f].step(targets[:, i], q[:, cols], sc.kp_env[:, cols], sc.effort[cols], sc.lower[cols],
                                 sc.upper[cols], J[f][0], J[f][1], c[f]['point'], c[f]['normal'], c[f]['fn'],
                                 c[f]['has'], np.tile(sc.upper[cols], (E, 1)))
            tgt[:, cols] = qt
        if phase == 'lift':
            s = (t-S-cfg.preload_ticks+1)/cfg.lift_ticks
            tgt[:, :6] = q_arm0+(s*s*(3-2*s))*(q_lift-q_arm0)
        elif phase == 'shake':
            tau = (t-t_shake+1)/60.
            env = min(1., (t-t_shake+1)/10., (t_shake+cfg.shake_ticks-t)/10.)     # 1/6 s fade in and out
            w2 = 2*np.pi*cfg.shake_hz*tau
            tgt[:, :6] = q_hold+env*(np.sin(w2)*dq_x+np.sin(w2+np.pi/2)*dq_z)
        elif phase == 'rest':
            tgt[:, :6] = q_hold
        sc.step(tgt)
        # damage: per-finger solver force at the region of its contact point
        c = sc.finger_contacts()
        center, R = sc.object_pose()
        F = np.zeros((E, part.k))
        for i, f in enumerate(FINGERS):
            p_obj = np.einsum('eji,ej->ei', R, c[f]['point']-center)
            # Each finger's force goes to its snapshot region (the allocator plans with the same regions).
            # Mapping the live contact point instead flipped regions when a contact sat near a boundary and
            # summed two fingers into one region (spurious breaks at 0.1 s with every force below threshold).
            F[:, reg[i]] += np.where(c[f]['has'], c[f]['fn'], 0.)
            log['live_region_mismatch'][t, :, i] = c[f]['has'] & (part.region_of(p_obj) != reg[i])
            log['fn'][t, :, i] = c[f]['fn']
            log['ft'][t, :, i] = c[f].get('tangential', np.zeros(E))
            log['has'][t, :, i] = c[f]['has']
        if t < S:
            log['settle_peak_N'] = np.maximum(log['settle_peak_N'], F.max(1))
        else:
            judge.update(F, t)
        if t >= S:
            p_h, R_h = in_hand()
            cosang = (np.einsum('eji,eji->e', R_h0, R_h)-1)/2      # trace(R_h0^T R_h)
            log['obj_rot_deg'][t] = np.degrees(np.arccos(np.clip(cosang, -1., 1.)))
            log['obj_slip_mm'][t] = 1000*np.linalg.norm(p_h-p_h0, axis=1)
        log['alloc_ok'][t], log['eq_slack'][t], log['risk_slack'][t] = ok_now, eq_slack, risk_slack
        log['target'][t] = targets
        log['region_F'][t] = F
        log['phase'].append(phase)
    end_center, _ = sc.object_pose()
    lifted = end_center[:, 2]-start_center[:, 2]
    lost = ~log['has'][-30:].all(axis=(0, 2))
    dropped = (lifted < .6*cfg.lift_m) | lost
    broke = judge.broken.copy()
    log.update(lifted_m=lifted, feasible=feasible)
    return TrialResult(broke=broke, dropped=dropped & ~broke, success=~broke & ~dropped, events=judge.event,
                       archive=[judge.archive(e) for e in range(E)], log=log)
