"""Handwritten log-normal belief with exact censored Bayesian updates (the strong handwritten baseline).

Model (log thresholds, natural log):
  log theta_r = s + d_r
  s    ~ N(log m_obj, sigma_obj)          object scale: Astra's safe-force range (geometric mean), or a class
                                          median; sigma_obj widened when Astra is unsure
  d_r  ~ N(log factor_r, sigma_region)    Astra's region label: fragile 0.45, normal 1, strong 1.4
                                          (unlabelled regions: factor 1)
Observations from earlier grasps of the same instance (DamageJudge archive rows), soft edges of width
--edge (log units) for force noise:
  survived at L        P = Phi((log theta - log L) / edge)
  broke in (L, U]      P = Phi((log theta - log L) / edge) - Phi((log theta - log U) / edge)
The object scale is shared, so a survival on one region raises the belief on all of them. Everything is
exact on a grid in log force (no sampling). The hazard handed to the allocator is -log P(theta_r > f) on
allocator.GRID_N, made convex like prior.py.
"""
from __future__ import annotations

from math import erf, sqrt

import numpy as np

from risk_mvp.allocator import GRID_N
from risk_mvp.prior import CLASS_MEDIAN_N, REGION_FACTOR

X = np.linspace(np.log(.02), np.log(60.), 260)          # log-threshold grid
DX = X[1]-X[0]
_erf = np.vectorize(erf)


def _phi(z):
    return .5*(1+_erf(np.asarray(z, float)/sqrt(2)))


def _norm(x, mu, sd):
    return np.exp(-.5*((x-mu)/sd)**2)


def prior_params(vlm: dict, vlm_object: dict, k: int, sigma_obj=.5, sigma_region=.3):
    o = vlm_object or {}
    rng = o.get('safe_force_N')
    m_obj = float(np.sqrt(rng[0]*rng[1])) if rng and min(rng) > 0 else CLASS_MEDIAN_N.get(o.get('strength'), 3.2)
    conf_obj = float(o.get('confidence', .5)) if o else 0.
    mu_d, sd_d = np.zeros(k), np.full(k, sigma_region*2.)
    for r, v in (vlm or {}).items():
        r = int(r)
        if r < k:
            mu_d[r] = np.log(REGION_FACTOR[v['label']])
            sd_d[r] = sigma_region*(1+(1-float(v.get('confidence', .5))))
    return np.log(m_obj), sigma_obj*(1+(1-conf_obj)), mu_d, sd_d


def likelihood(rows, k, edge=.05):
    """(k, len(X)) likelihood of each region's log threshold given the archive rows."""
    L = np.ones((k, len(X)))
    for row in rows:
        r = row['region']
        lo = np.log(max(row['lower'], 1e-3))
        if row['broke']:
            up = np.log(max(row['upper'], row['lower']*1.001, 1e-3))
            L[r] *= np.clip(_phi((X-lo)/edge)-_phi((X-up)/edge), 1e-12, None)
        else:
            L[r] *= np.clip(_phi((X-lo)/edge), 1e-12, None)
    return L


def posterior(vlm, vlm_object, k, rows, sigma_obj=.5, sigma_region=.3, edge=.05):
    """(k, len(X)) posterior over each region's log threshold."""
    mu_s, sd_s, mu_d, sd_d = prior_params(vlm, vlm_object, k, sigma_obj, sigma_region)
    ps = _norm(X, mu_s, sd_s)
    ps /= ps.sum()
    L = likelihood(rows, k, edge)
    # K[r][s, x] = p(log theta_r = x | s) = N(x - s; mu_d, sd_d)
    D = X[None, :]-X[:, None]                                   # (s, x)
    K = [_norm(D, mu_d[r], sd_d[r]) for r in range(k)]
    K = [Kr/np.maximum(Kr.sum(1, keepdims=True), 1e-300) for Kr in K]
    ev = np.stack([K[r]@L[r] for r in range(k)])               # (k, s): evidence of region r given s
    log_ev = np.log(np.maximum(ev, 1e-300))
    tot = log_ev.sum(0)
    post = np.zeros((k, len(X)))
    for r in range(k):
        w = np.log(np.maximum(ps, 1e-300))+tot-log_ev[r]        # p(s) * prod_{q != r} evidence_q
        w = np.exp(w-w.max())
        px = (w@K[r])*L[r]
        post[r] = px/max(px.sum(), 1e-300)
    return post


def hazard(vlm, vlm_object, k, rows, grid=GRID_N, **kw) -> np.ndarray:
    """(k, len(grid)) convex non-decreasing cumulative hazard for the allocator."""
    post = posterior(vlm, vlm_object, k, rows, **kw)
    surv = 1-np.cumsum(post, 1)                                 # P(log theta > X_j)
    lf = np.log(np.maximum(grid, 1e-6))
    S = np.stack([np.interp(lf, X, surv[r], left=1., right=0.) for r in range(k)])
    H = -np.log(np.clip(S, 1e-9, 1.))
    H[:, 0] = 0.
    slope = np.maximum.accumulate(np.maximum(np.diff(H, axis=1), 0)/np.diff(grid), axis=1)
    return np.concatenate([np.zeros((k, 1)), np.cumsum(slope*np.diff(grid), axis=1)], 1)


def median_threshold(vlm, vlm_object, k, rows, **kw) -> np.ndarray:
    post = posterior(vlm, vlm_object, k, rows, **kw)
    c = np.cumsum(post, 1)
    return np.exp(np.array([X[min(np.searchsorted(c[r], .5), len(X)-1)] for r in range(k)]))
