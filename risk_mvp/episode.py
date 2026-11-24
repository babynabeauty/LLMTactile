"""One batched trial from a snapshot (MVP step 3): allocate -> preload -> lift 5 cm -> hold 2 s.

Every env starts from the same already-touching grasp (snapshot.py). Per env it gets an allocation
method, hidden region thresholds, object mass and contact friction. At 60 Hz:
  - every `alloc_every` ticks the env's Allocator gives per-finger normal targets (the contact geometry
    and regions are those of the snapshot, in the object frame; the external wrench is gravity in the
    current object frame, mass x (1 + mass_margin));
  - targets ramp from the touch force to the allocation during the preload, then the arm lifts and holds;
  - BatchForceController tracks the targets; the solver's per-finger normal forces are summed per region
    (each contact point mapped to its region in the object frame) and fed to DamageJudge.
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
    margin_N: float = .1          # force-estimate margin in the damage constraint
    eps: float = .05              # damage budget for the risk mode
    mass_margin: float = .1
    fmax_N: float = 8.
    touch_N: float = .2
    preload_ticks: int = 30
    lift_ticks: int = 60
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
              mass_kg=None, hazard: Optional[np.ndarray] = None, cap_N: Optional[np.ndarray] = None) -> TrialResult:
    E = sc.E
    assert len(modes) == E and thresholds.shape == (E, part.k)
    sc.restore(snap)
    mass = np.broadcast_to(np.asarray(sc.mass_kg if mass_kg is None else mass_kg, float), (E,))
    pos = np.array([snap['contacts'][f]['point_obj'] for f in FINGERS])
    nrm = np.array([snap['contacts'][f]['normal_obj'] for f in FINGERS], float)
    nrm = np.array([safe_normal(n, p) for n, p in zip(nrm, pos)])
    reg = np.array([snap['contacts'][f]['region'] for f in FINGERS])
    allocs = [Allocator(Contacts(pos, nrm, reg, list(FINGERS)), part.k, mode=m) for m in modes]
    ctrl = {f: BatchForceController(E) for f in FINGERS}
    for f in FINGERS:
        ctrl[f].in_contact[:] = True
    judge = DamageJudge(thresholds)
    targets = np.full((E, 3), cfg.touch_N)
    alloc_fn = targets.copy()
    feasible = np.ones(E, bool)
    q_arm0 = sc.q()[:, :6]
    wpos, wquat = sc.wrist_pose()
    q_lift, _ = sc.ik(wpos+np.array([0, 0, cfg.lift_m]), wquat)
    start_center, _ = sc.object_pose()
    n_ticks = cfg.preload_ticks+cfg.lift_ticks+cfg.hold_ticks
    log = dict(fn=np.zeros((n_ticks, E, 3)), target=np.zeros((n_ticks, E, 3)), region_F=np.zeros((n_ticks, E, part.k)),
               has=np.zeros((n_ticks, E, 3), bool), phase=[])
    for t in range(n_ticks):
        phase = 'preload' if t < cfg.preload_ticks else 'lift' if t < cfg.preload_ticks+cfg.lift_ticks else 'hold'
        c = sc.finger_contacts()
        center, R = sc.object_pose()
        if t % cfg.alloc_every == 0:
            for e in range(E):
                w = gravity_wrench(mass[e]*(1+cfg.mass_margin), R[e])
                kw = dict(w_ext=w, mu=cfg.mu_hat, fmax=cfg.fmax_N, margin=cfg.margin_N, safety=cfg.safety)
                if modes[e] == 'risk':
                    kw.update(hazard=hazard[e], eps=cfg.eps)
                elif modes[e] == 'oracle':
                    kw.update(limit=thresholds[e])
                elif modes[e] == 'cap':
                    kw.update(limit=np.full(part.k, cap_N[e] if cap_N is not None else 2.))
                rep = allocs[e].solve(**kw)
                alloc_fn[e] = [rep['finger_normal'][f] for f in FINGERS]
                feasible[e] &= rep['feasible']
        ramp = min(1., (t+1)/cfg.preload_ticks)
        targets = cfg.touch_N+ramp*(alloc_fn-cfg.touch_N)
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
            s = (t-cfg.preload_ticks+1)/cfg.lift_ticks
            tgt[:, :6] = q_arm0+(s*s*(3-2*s))*(q_lift-q_arm0)
        sc.step(tgt)
        # damage: per-finger solver force at the region of its contact point
        c = sc.finger_contacts()
        center, R = sc.object_pose()
        F = np.zeros((E, part.k))
        for i, f in enumerate(FINGERS):
            p_obj = np.einsum('eji,ej->ei', R, c[f]['point']-center)
            r = part.region_of(p_obj)
            np.add.at(F, (np.arange(E), r), np.where(c[f]['has'], c[f]['fn'], 0.))
            log['fn'][t, :, i] = c[f]['fn']
            log['has'][t, :, i] = c[f]['has']
        judge.update(F, t)
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
