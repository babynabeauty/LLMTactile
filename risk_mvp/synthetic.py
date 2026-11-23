"""Synthetic grasp histories for CPU checks of the risk model (no simulator).

Same tensors as the real pipeline will produce (collect.py), drawn from a simple generative story:
- object: a real partition (regions.py) of one of five shapes, K = 3..6 regions;
- instance: regions.sample_instance (log-uniform strength, correlated coefficients, hidden weak spot);
- VLM: labels region strength from what is visible (the hidden weak spot is labelled like its
  neighbours). 'good' objects: label right with p = 0.8; 'bad' objects (p = 0.25): labels inverted;
- grasp: 1-3 touched regions ramped together to random peak forces (log-uniform 0.3-12 N); the first
  region to cross its threshold breaks (interval (L, U] from a 5 % ramp resolution), the others survive
  at the force they had reached. Force history = the first 30 % of the planned ramp (no outcome leak).
"""
from __future__ import annotations

import numpy as np

from .regions import object_mesh, partition, sample_instance
from .risk_model import GEOM_DIM, HIST_LEN, N_LABELS, TACT_DIM, RiskConfig

K_MAX, A_MAX = 6, 10
SHAPES = [dict(shape='cuboid', size_m=[.045, .06, .045]), dict(shape='cylinder', radius_m=.03, height_m=.09),
          dict(shape='sphere', radius_m=.03), dict(shape='flat_box', size_m=[.08, .06, .025]),
          dict(shape='hex_prism', across_flats_m=.045, height_m=.045)]


def vlm_label(coeff):
    return np.where(coeff < .75, 0, np.where(coeff > 1.3, 2, 1))   # 0 fragile, 1 normal, 2 strong


class World:
    def __init__(self, seed=0, cfg: RiskConfig = RiskConfig(), p_bad=.25):
        self.rng = np.random.default_rng(seed)
        self.cfg, self.p_bad = cfg, p_bad
        self.parts = [partition(object_mesh(s), k, seed=0) for s in SHAPES for k in range(3, 7)]
        self.text = np.random.default_rng(123).normal(0, 1, (N_LABELS, cfg.text_dim))

    def instance(self, weak_region=None, bad=None):
        rng = self.rng
        part = self.parts[rng.integers(len(self.parts))]
        inst = sample_instance(part, rng, weak_region=weak_region if weak_region is None or weak_region < part.k else None)
        visible = inst.coeff.copy()
        if inst.weak_region is not None:   # the weak spot cannot be seen: it looks like its neighbours
            others = np.delete(np.arange(part.k), inst.weak_region)
            visible[inst.weak_region] = np.exp(np.log(visible[others]).mean())
        true_lab = vlm_label(visible)
        bad = (rng.random() < self.p_bad) if bad is None else bad
        if bad:
            lab = 2-true_lab
            conf = rng.uniform(.5, .9, part.k)
        else:
            flip = rng.random(part.k) > .8
            lab = np.where(flip, rng.integers(0, 3, part.k), true_lab)
            conf = rng.uniform(.6, 1., part.k)
        return dict(part=part, thr=inst.thresholds, weak=inst.weak_region, label=lab, conf=conf, bad=bool(bad))

    def grasp(self, obj):
        rng = self.rng
        k = obj['part'].k
        touched = rng.choice(k, size=rng.integers(1, min(3, k)+1), replace=False)
        peak = np.exp(rng.uniform(np.log(.3), np.log(12.), len(touched)))
        t_cross = obj['thr'][touched]/peak
        rows = []
        if t_cross.min() < 1:
            j = int(np.argmin(t_cross))
            t_end = t_cross[j]
        else:
            j, t_end = None, 1.
        for m, r in enumerate(touched):
            reached = peak[m]*t_end
            # what the model may see: the first 30 % of the planned ramp (fixed before the outcome)
            hist = np.linspace(0, .3*peak[m], HIST_LEN)
            if m == j:
                th = obj['thr'][r]
                rows.append(dict(region=int(r), broke=True, lower=th*(1-rng.uniform(0, .05)),
                                 upper=th*(1+rng.uniform(0, .05)), max_force=th, hist=hist))
            else:
                rows.append(dict(region=int(r), broke=False, lower=reached, upper=np.inf, max_force=reached, hist=hist))
        return rows

    # ---- tensors ----
    def region_cards(self, obj):
        p, k = obj['part'], obj['part'].k
        lab = np.eye(N_LABELS)[obj['label']]
        text = self.text[obj['label']]+self.rng.normal(0, .3, (k, self.cfg.text_dim))
        tact = self.rng.normal(0, .1, (k, TACT_DIM))
        return np.concatenate([lab, obj['conf'][:, None], text, p.features(), tact], 1)

    def archive_row(self, obj, row):
        r = row['region']
        geom = obj['part'].features()[r]
        lab = np.eye(N_LABELS)[obj['label'][r]]
        up = row['upper'] if row['broke'] else 0.
        return np.concatenate([geom, lab, [obj['conf'][r], row['max_force']/10, float(row['broke']),
                                           np.log(max(row['lower'], 1e-3)), np.log(up) if up else 0.,
                                           float(not row['broke'])]])

    def samples(self, n_instances, grasps=10, n_archive=None, bad=None, weak=None):
        """List of per-sample dicts. n_archive None: uniform 0..10 per sample."""
        out = []
        for _ in range(n_instances):
            obj = self.instance(weak_region=weak, bad=bad)
            hist = [self.grasp(obj) for _ in range(grasps)]
            cards = self.region_cards(obj)
            for g in range(grasps):
                pool = [row for h, rows in enumerate(hist) if h != g for row in rows]
                n = self.rng.integers(0, min(A_MAX, len(pool))+1) if n_archive is None else min(n_archive, len(pool))
                pick = self.rng.choice(len(pool), size=n, replace=False) if n else []
                out.append(dict(obj=obj, cards=cards, rows=hist[g], archive=[self.archive_row(obj, pool[i]) for i in pick]))
        return out


def collate(samples, cfg: RiskConfig = RiskConfig()):
    import torch
    B = len(samples)
    region = np.zeros((B, K_MAX, cfg.region_dim), np.float32)
    hist = np.zeros((B, K_MAX, HIST_LEN), np.float32)
    reg_valid = np.zeros((B, K_MAX), bool)
    archive = np.zeros((B, A_MAX, cfg.archive_dim), np.float32)
    arc_valid = np.zeros((B, A_MAX), bool)
    lower = np.ones((B, K_MAX), np.float32)
    upper = np.ones((B, K_MAX), np.float32)
    broke = np.zeros((B, K_MAX), bool)
    lab = np.zeros((B, K_MAX), np.float32)
    thr = np.ones((B, K_MAX), np.float32)
    vlm = np.zeros((B, K_MAX), np.int64)
    for b, s in enumerate(samples):
        k = s['cards'].shape[0]
        region[b, :k] = s['cards']
        reg_valid[b, :k] = True
        thr[b, :k] = s['obj']['thr']
        vlm[b, :k] = s['obj']['label']
        for row in s['rows']:
            r = row['region']
            hist[b, r] = row['hist']
            lower[b, r] = row['lower']
            upper[b, r] = row['upper'] if row['broke'] else row['lower']
            broke[b, r] = row['broke']
            lab[b, r] = 1.
        for i, a in enumerate(s['archive']):
            archive[b, i] = a
            arc_valid[b, i] = True
    t = lambda x: torch.as_tensor(x)
    return dict(region=t(region), hist=t(hist), reg_valid=t(reg_valid), archive=t(archive), arc_valid=t(arc_valid),
                lower=t(lower), upper=t(upper), broke=t(broke), mask=t(lab), thr=t(thr), vlm=t(vlm))
