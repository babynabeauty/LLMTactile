import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from risk_mvp.risk_model import (GRID, RiskConfig, RiskModel, build_attention_mask, censored_nll,  # noqa: E402
                                 hazard_at, median_threshold)
from risk_mvp.synthetic import World, collate  # noqa: E402
from risk_mvp.allocator import pwl_from_hazard  # noqa: E402


def test_mask_structure():
    m = build_attention_mask(2, 3, torch.tensor([[True, False]]), torch.tensor([[True, True, False]]), 1)[0]
    ok = m == 0
    T, A0, A1, R0, R1, R2 = range(6)
    assert ok[A0, A0] and not ok[A0, A1] and not ok[A0, T] and not ok[A0, R0]   # archive sees only valid archive
    assert ok[T, T] and ok[T, A0] and not ok[T, R0]                             # trust reads the archive
    assert ok[R0, R1] and ok[R0, A0] and ok[R0, T] and not ok[R0, R2]          # regions see all valid tokens
    assert ok[A1, A1] and ok[R2, R2]                                           # padded rows see themselves


def test_hazard_monotone_convex_and_archive_independent_of_regions():
    cfg = RiskConfig()
    w = World(seed=0, cfg=cfg)
    b = collate(w.samples(3), cfg)
    model = RiskModel(cfg).eval()
    with torch.no_grad():
        H = model(b['region'], b['hist'], b['reg_valid'], b['archive'], b['arc_valid'])
    d = torch.diff(H, dim=-1)
    assert (H[..., 0] == 0).all() and (d >= -1e-6).all() and (torch.diff(d, dim=-1) >= -1e-6).all()
    a, _ = pwl_from_hazard(H[0, 0].numpy())
    assert (np.diff(a) >= -1e-5).all()
    # empty archive works
    with torch.no_grad():
        H0 = model(b['region'], b['hist'], b['reg_valid'], b['archive'][:, :0], b['arc_valid'][:, :0])
    assert torch.isfinite(H0).all()


def test_censored_loss_prefers_consistent_curves():
    H_lo = (GRID/2.)[None, None]**2      # median about 1.7 N
    H_hi = (GRID/8.)[None, None]**2      # median about 6.7 N
    one = lambda v: torch.tensor([[v]])
    # survived at 4 N: the strong curve explains it better
    surv = [censored_nll(H, one(4.), one(4.), one(False), one(1.)) for H in (H_lo, H_hi)]
    assert surv[1] < surv[0]
    # broke in (1.5, 1.8]: the weak curve explains it better
    brk = [censored_nll(H, one(1.5), one(1.8), one(True), one(1.)) for H in (H_lo, H_hi)]
    assert brk[0] < brk[1]
    assert abs(median_threshold(H_lo).item()-2*np.sqrt(np.log(2))) < .05
    v, s = hazard_at(H_hi, torch.tensor([[12.]]))
    assert v.item() > H_hi[0, 0, -1].item() and s.item() > 0
