"""Chance-constrained force allocation (MVP step 7), one SOCP per control step.

Variables: a force vector f_i on the object at every contact i (object frame).
  (1) wrench balance        G f = -w_ext          (gravity + lift inertia, force & torque about COM)
  (2) friction cone         ||f_i - (n_i.f_i) n_i|| <= mu (n_i.f_i)
  (3) damage budget         sum_r H_r(F_r + margin) <= -log(1 - eps),   F_r = sum_{i in r} n_i.f_i
      H_r: cumulative hazard on a force grid, convex increasing piecewise-linear -> max of affine pieces
  (4) actuator limit        n_i.f_i <= fmax_i     (joint-torque limit mapped to the fingertip)
Objective: sum_i n_i.f_i + lam ||f - f_prev||^2.

(1) and (3) carry non-negative slacks with a large penalty, so the problem is always
feasible; a positive slack is the structured "cannot hold safely" report.

Modes (all share (1), (2), (4), the objective and the same contacts):
  'risk'    our method / handwritten belief: pass hazard curves
  'tactile' TacDexGrasp-style: no (3)
  'equal'   no (3), the listed contacts carry equal normal force
  'cap'     static cap: F_r <= cap instead of (3)
  'oracle'  known thresholds: F_r <= thr_r - margin instead of (3)

The contact set and its region assignment are fixed per grasp, so the problem is
built once (cvxpy DPP) and re-solved every step with new parameters.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Sequence

import cvxpy as cp
import numpy as np

GRID_N = np.linspace(0., 10., 101)   # 100 bins over 0-10 N, matches the risk model output


def skew(p):
    return np.array([[0, -p[2], p[1]], [p[2], 0, -p[0]], [-p[1], p[0], 0]])


def pwl_from_hazard(H: np.ndarray, grid: np.ndarray = GRID_N):
    """Slopes/intercepts of the convex PWL through (grid, H) per region, (K, S) each.
    H must be non-decreasing and convex along the grid (the risk model guarantees it);
    beyond the grid the last slope is extended."""
    H = np.asarray(H, float)
    a = np.diff(H, axis=-1)/np.diff(grid)
    b = H[..., :-1]-a*grid[:-1]
    return a, b


@dataclass
class Contacts:
    pos: np.ndarray        # (n, 3) contact points relative to the object COM, object frame
    normal: np.ndarray     # (n, 3) unit normals pointing INTO the object
    region: np.ndarray     # (n,) region index
    finger: Sequence[str]  # (n,) finger name per contact


class Allocator:
    def __init__(self, contacts: Contacts, n_regions: int, mode: str = 'risk', lam: float = 1.,
                 equal_group: Sequence[str] = ('index', 'middle'), slack_weight: float = 1e4,
                 torque_scale_m: float = .02):
        if mode not in ('risk', 'tactile', 'equal', 'cap', 'oracle'):
            raise ValueError(mode)
        self.c, self.K, self.mode = contacts, n_regions, mode
        n = len(contacts.pos)
        N = np.asarray(contacts.normal, float)
        N = N/np.linalg.norm(N, axis=1, keepdims=True)
        G = np.zeros((6, 3*n))
        for i in range(n):
            G[:3, 3*i:3*i+3] = np.eye(3)
            G[3:, 3*i:3*i+3] = skew(contacts.pos[i])
        # Torque rows divided by a grasp length so the balance slack is in newtons on every row.
        G[3:] /= torque_scale_m
        self.torque_scale_m = torque_scale_m
        M = np.zeros((n_regions, n))
        M[contacts.region, np.arange(n)] = 1.

        self.f = cp.Variable((n, 3))
        fn = cp.sum(cp.multiply(self.f, N), axis=1)          # normal components
        self.fn_expr = fn
        self.w = cp.Parameter(6)
        self.f_prev = cp.Parameter((n, 3))
        self.mu = cp.Parameter(nonneg=True)
        self.fmax = cp.Parameter(n, nonneg=True)
        self.margin = cp.Parameter(nonneg=True)
        s_eq = cp.Variable(6)
        cons = [G@cp.reshape(self.f, (3*n,), order='C')+self.w == s_eq, fn >= 0, fn <= self.fmax]
        for i in range(n):
            P = np.eye(3)-np.outer(N[i], N[i])
            cons.append(cp.norm(P@self.f[i]) <= self.mu*fn[i])
        F = M@fn
        obj = cp.sum(fn)+lam*cp.sum_squares(self.f-self.f_prev)+slack_weight*cp.norm1(s_eq)
        self.s_eq, self.s_risk = s_eq, None
        if mode == 'risk':
            S = len(GRID_N)-1
            self.A = cp.Parameter((n_regions, S), nonneg=True)
            self.B = cp.Parameter((n_regions, S))
            self.budget = cp.Parameter(nonneg=True)
            t = cp.Variable(n_regions)
            self.s_risk = cp.Variable(nonneg=True)
            # The margin enters through B (B + A*margin) so the problem stays DPP.
            for r in range(n_regions):
                cons.append(t[r] >= cp.multiply(self.A[r], F[r])+self.B[r])
            cons += [t >= 0, cp.sum(t) <= self.budget+self.s_risk]
            self.t = t
            obj = obj+slack_weight*self.s_risk
        elif mode in ('cap', 'oracle'):
            self.limit = cp.Parameter(n_regions, nonneg=True)
            self.s_risk = cp.Variable(n_regions, nonneg=True)
            cons.append(F+self.margin <= self.limit+self.s_risk)
            obj = obj+slack_weight*cp.sum(self.s_risk)
        elif mode == 'equal':
            idx = [i for i, f in enumerate(contacts.finger) if f in equal_group]
            cons += [fn[idx[j]] == fn[idx[0]] for j in range(1, len(idx))]
        self.F = F
        self.prob = cp.Problem(cp.Minimize(obj), cons)
        self.N = N
        self.f_prev.value = np.zeros((n, 3))

    def solve(self, w_ext: np.ndarray, mu: float, fmax, margin: float = 0., hazard: Optional[np.ndarray] = None,
              eps: float = .05, limit: Optional[np.ndarray] = None, solver: str = 'CLARABEL') -> dict:
        n = len(self.c.pos)
        w_ext = np.asarray(w_ext, float).copy()
        w_ext[3:] /= self.torque_scale_m
        self.w.value = w_ext
        self.mu.value = float(mu)
        self.fmax.value = np.broadcast_to(np.asarray(fmax, float), (n,)).copy()
        self.margin.value = float(margin)
        if self.mode == 'risk':
            a, b = pwl_from_hazard(hazard)
            a = np.maximum(a, 0.)
            self.A.value, self.B.value = a, b+a*float(margin)
            self.budget.value = -np.log(1-eps)
        elif self.mode in ('cap', 'oracle'):
            self.limit.value = np.broadcast_to(np.asarray(limit, float), (self.K,)).copy()
        t0 = time.perf_counter()
        self.prob.solve(solver=solver)
        dt = time.perf_counter()-t0
        f = self.f.value
        self.f_prev.value = f
        fn = (f*self.N).sum(1)
        F = np.zeros(self.K)
        np.add.at(F, self.c.region, fn)
        report = dict(status=self.prob.status, solve_s=dt, solver_s=self.prob.solver_stats.solve_time,
                      f=f, fn=fn, region_force=F, finger_normal={g: float(fn[i]) for i, g in enumerate(self.c.finger)})
        eq_slack = float(np.abs(self.s_eq.value).sum())
        risk_slack = 0. if self.s_risk is None else float(np.sum(self.s_risk.value))
        report.update(eq_slack=eq_slack, risk_slack=risk_slack, feasible=eq_slack < 1e-4 and risk_slack < 1e-5)
        if self.mode == 'risk':
            report['region_hazard'] = self.t.value.copy()
            report['damage_prob_bound'] = float(1-np.exp(-self.t.value.sum()))
        if not report['feasible']:
            report['infeasible'] = dict(
                missing_wrench=(-self.s_eq.value).tolist(),
                binding_region=int(np.argmax(self.t.value if self.mode == 'risk' else self.s_risk.value))
                if self.s_risk is not None else None)
        return report


def gravity_wrench(mass_kg: float, R_obj_to_world: np.ndarray, accel_world=(0., 0., 0.), g=9.81) -> np.ndarray:
    """External wrench on the object (object frame) from gravity and the commanded lift acceleration
    (d'Alembert): w = m (g_vec - a). Applied at the COM, so the torque part is zero."""
    fw = mass_kg*(np.array([0., 0., -g])-np.asarray(accel_world, float))
    return np.r_[R_obj_to_world.T@fw, np.zeros(3)]
