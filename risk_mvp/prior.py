"""Handwritten log-normal belief from the VLM labels (the 'handwritten log-normal' baseline, and the prior
the risk allocator uses before any trained model).

For the image the robot actually sees (the object as it is, mark or no mark):
  object median  m_obj = geometric mean of the VLM's safe_force_N range, or a class median
                 (fragile 1 N, normal 3.2 N, strong 7 N) when it gave no range
  region median  m_r = m_obj x {fragile 0.45, normal 1, strong 1.4}[label_r]; regions without a label
                 (not visible) get m_obj
  spread         log-normal sigma (default 0.5), widened when the VLM is unsure:
                 sigma_r = sigma x (1 + (1 - confidence_r))
  hazard         H_r(f) = -log(1 - Phi((ln f - ln m_r)/sigma_r)) on allocator.GRID_N. A log-normal hazard
                 rate falls again far in the tail, so H is made convex by taking the running maximum of its
                 slopes (never below the log-normal H): what allocator.pwl_from_hazard needs.
"""
from __future__ import annotations

from math import erf, log, sqrt
from typing import Optional

import numpy as np

from risk_mvp.allocator import GRID_N

CLASS_MEDIAN_N = dict(fragile=1., normal=3.2, strong=7.)
REGION_FACTOR = dict(fragile=.45, normal=1., strong=1.4)


def _phi(z):
    return .5*(1+np.vectorize(erf)(np.asarray(z, float)/sqrt(2)))


def region_medians(label: dict, k: int):
    """(medians (k,), confidences (k,)) from one labels.json entry (vlm_query.merge output)."""
    o = label.get('object') or {}
    rng = o.get('safe_force_N')
    if rng and min(rng) > 0:
        m_obj = float(np.sqrt(rng[0]*rng[1]))
    else:
        m_obj = CLASS_MEDIAN_N.get(o.get('strength'), CLASS_MEDIAN_N['normal'])
    med, conf = np.full(k, m_obj), np.full(k, .5)
    for r, reg in (label.get('regions') or {}).items():
        r = int(r)
        if r < k:
            med[r] = m_obj*REGION_FACTOR[reg['label']]
            conf[r] = float(reg.get('confidence', .5))
    return med, conf


def lognormal_hazard(median: np.ndarray, sigma: np.ndarray, grid: np.ndarray = GRID_N) -> np.ndarray:
    """(k, len(grid)) cumulative hazard, convex and non-decreasing along the grid."""
    f = np.maximum(grid, 1e-6)[None, :]
    z = (np.log(f)-np.log(np.asarray(median, float))[:, None])/np.asarray(sigma, float)[:, None]
    S = np.clip(1-_phi(z), 1e-12, 1.)
    H = -np.log(S)
    H[:, 0] = 0.
    slope = np.maximum.accumulate(np.diff(H, axis=1)/np.diff(grid), axis=1)
    return np.concatenate([np.zeros((H.shape[0], 1)), np.cumsum(slope*np.diff(grid), axis=1)], 1)


def hazard_from_label(label: dict, k: int, sigma: float = .5, grid: np.ndarray = GRID_N) -> np.ndarray:
    med, conf = region_medians(label, k)
    return lognormal_hazard(med, sigma*(1+(1-conf)), grid)


def safe_force(H: np.ndarray, eps: float = .05, grid: np.ndarray = GRID_N) -> np.ndarray:
    """Per region, the largest grid force with damage probability <= eps under H (for reports)."""
    lim = -log(1-eps)
    return np.array([grid[np.flatnonzero(h <= lim)].max() if (h <= lim).any() else 0. for h in H])


def label_for(labels: dict, oid: str, variant: str) -> Optional[dict]:
    """The labels.json entry, or None when there is none or it has no answer (every call failed): an
    empty entry would silently give the default class median for every region."""
    lab = labels.get(f'{oid}/{variant}')
    return lab if lab and lab.get('n_answers', 1) > 0 and lab.get('regions') else None
