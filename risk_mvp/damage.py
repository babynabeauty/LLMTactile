"""Region force aggregation and virtual damage (MVP step 1). Batched over envs, numpy.

Damage rule: the region's summed normal force stays above its threshold for
`hold_steps` consecutive physics samples (3 by default, filters solver spikes).
Equivalently, with the sustained force s_t = min(F_t, ..., F_{t-hold+1}),
damage happens at the first t with s_t > threshold. That makes the archive exact
interval censoring:
  survived  ->  threshold >= max_t s_t                      (right-censored)
  broke at t*  ->  max_{t<t*} s_t <= threshold < s_{t*}     (interval-censored)
"""
from __future__ import annotations

import numpy as np


def region_forces(region_idx: np.ndarray, normal_N: np.ndarray, valid: np.ndarray, k: int) -> np.ndarray:
    """Sum contact normal forces per region. Inputs (E, C); output (E, k)."""
    E = region_idx.shape[0]
    out = np.zeros((E, k))
    r = np.where(valid, region_idx, 0)
    np.add.at(out, (np.repeat(np.arange(E), region_idx.shape[1]).reshape(E, -1), r), np.where(valid, normal_N, 0.))
    return out


class DamageJudge:
    def __init__(self, thresholds: np.ndarray, hold_steps: int = 3):
        self.thr = np.asarray(thresholds, float)            # (E, K)
        self.hold = hold_steps
        E, K = self.thr.shape
        self.window = np.zeros((hold_steps, E, K))
        self.n = 0
        self.max_sustained = np.zeros((E, K))   # max s_t over steps before damage
        self.max_raw = np.zeros((E, K))
        self.broken = np.zeros(E, bool)
        self.event = [None]*E

    def update(self, F: np.ndarray, step: int) -> np.ndarray:
        """F (E, K) region normal forces at this physics sample. Returns envs newly broken."""
        F = np.asarray(F, float)
        self.window[self.n % self.hold] = F
        self.n += 1
        live = ~self.broken
        self.max_raw[live] = np.maximum(self.max_raw[live], F[live])
        if self.n < self.hold:
            return np.zeros(len(F), bool)
        s = self.window.min(0)
        over = (s > self.thr) & live[:, None]
        newly = over.any(1)
        for e in np.flatnonzero(newly):
            ratio = np.where(over[e], s[e]/self.thr[e], -np.inf)
            r = int(np.argmax(ratio))
            self.event[e] = dict(step=int(step), region=r, sustained_N=float(s[e, r]),
                                 lower_N=float(self.max_sustained[e, r]), raw_N=float(F[e, r]))
        self.broken |= newly
        keep = live & ~newly
        self.max_sustained[keep] = np.maximum(self.max_sustained[keep], s[keep])
        return newly

    def archive(self, e: int, touched_min_N: float = .05) -> list:
        """Archive rows for env e: one per region that was touched (or broke)."""
        rows = []
        ev = self.event[e]
        for r in range(self.thr.shape[1]):
            if ev is not None and ev['region'] == r:
                rows.append(dict(region=r, outcome='broke', lower_N=ev['lower_N'], upper_N=ev['sustained_N'],
                                 max_force_N=ev['raw_N']))
            elif self.max_sustained[e, r] > touched_min_N:
                rows.append(dict(region=r, outcome='survived', lower_N=float(self.max_sustained[e, r]),
                                 upper_N=None, max_force_N=float(self.max_raw[e, r])))
        return rows
