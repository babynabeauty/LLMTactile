"""Card Transformer risk model with a censored likelihood (MVP step 6).

Tokens
  region cards  (one per region of the object in this grasp): VLM label (fragile/normal/strong one-hot),
                VLM confidence, frozen text embedding of the VLM reason, geometry (regions.Partition
                .features), tactile features, and a GRU encoding of the region's force history so far.
  archive cards (0-10, unordered, no positional encoding): earlier outcomes on this object instance:
                the region's geometry, the VLM label given for it, the max force, and the censoring
                result (survived above L, or broke in (L, U]). Never the queried grasp's own outcome.
  trust token   learnable; reads the archive (VLM label vs what actually happened).
Attention (allowed): archive<->archive; trust->archive,trust; region->region,archive,trust.
The archive never attends to regions or trust, so its states can be cached per instance.

Output per region: a cumulative hazard H(f) on GRID (0-10 N, 101 points). Slopes are cumulative sums of
softplus increments, so H is non-decreasing and convex (what allocator.pwl_from_hazard needs). Past 10 N
the last slope is extended. Damage probability at force f is 1 - exp(-H(f)).

Loss (censored NLL, per labelled region):
  survived at L       -log S(L)          = H(L)
  broke in (L, U]     -log(S(L) - S(U))  = H(L) - log(1 - exp(-(H(U) - H(L))))
  exact (open-book)   -log f(t)          = H(t) - log h(t)         (evaluation baseline only)
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

GRID = torch.linspace(0., 10., 101)
N_LABELS = 3                # fragile / normal / strong
GEOM_DIM = 8                # regions.Partition.features
TACT_DIM = 3                # tangential/normal ratio, contact count, slip indicator (MVP)
HIST_LEN = 30


@dataclass
class RiskConfig:
    text_dim: int = 32
    extra_dim: int = 0          # extra region-card features appended at the end (collect data: the VLM's
                                # object-level answer and the handwritten prior's median / spread per region)
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = .1
    max_archive: int = 10

    @property
    def region_dim(self):
        return N_LABELS+1+self.text_dim+GEOM_DIM+TACT_DIM+self.extra_dim

    @property
    def archive_dim(self):
        # geometry, VLM label + confidence, max force, broke flag, log L, log U (0 if open), U-is-open flag
        return GEOM_DIM+N_LABELS+1+1+1+1+1+1


def build_attention_mask(n_arc: int, n_reg: int, arc_valid: torch.Tensor, reg_valid: torch.Tensor,
                         n_heads: int) -> torch.Tensor:
    """Float mask (B*heads, L, L), 0 = allowed, -inf = blocked; sequence order [trust, archive, regions]."""
    B = arc_valid.shape[0]
    L = 1+n_arc+n_reg
    kind = torch.cat([torch.zeros(1), torch.ones(n_arc), 2*torch.ones(n_reg)]).long()   # 0 trust, 1 arc, 2 reg
    q, k = kind[:, None], kind[None, :]
    allowed = ((q == 1) & (k == 1)) | ((q == 0) & (k <= 1)) | (q == 2)
    key_valid = torch.cat([torch.ones(B, 1, dtype=torch.bool), arc_valid, reg_valid], 1)   # (B, L)
    allowed = allowed[None] & key_valid[:, None, :]
    allowed = allowed | torch.eye(L, dtype=torch.bool)[None]   # every token sees itself (no empty rows)
    mask = torch.zeros(B, L, L).masked_fill(~allowed, float('-inf'))
    return mask.repeat_interleave(n_heads, 0)


class RiskModel(nn.Module):
    def __init__(self, cfg: RiskConfig = RiskConfig()):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.hist = nn.GRU(1, 32, batch_first=True)
        self.region_in = nn.Sequential(nn.Linear(cfg.region_dim+32, d), nn.GELU(), nn.Linear(d, d))
        self.archive_in = nn.Sequential(nn.Linear(cfg.archive_dim, d), nn.GELU(), nn.Linear(d, d))
        self.trust = nn.Parameter(torch.zeros(1, 1, d))
        self.kind_emb = nn.Embedding(3, d)
        layer = nn.TransformerEncoderLayer(d, cfg.n_heads, 4*d, cfg.dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, cfg.n_layers, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, len(GRID)-1+1))

    def forward(self, region, hist, reg_valid, archive, arc_valid):
        """region (B,K,region_dim), hist (B,K,T), reg_valid (B,K), archive (B,A,archive_dim), arc_valid (B,A).
        Returns H on GRID, (B, K, 101)."""
        B, K, _ = region.shape
        A = archive.shape[1]
        _, h = self.hist(hist.reshape(B*K, -1, 1))
        r = self.region_in(torch.cat([region, h[-1].reshape(B, K, -1)], -1))+self.kind_emb.weight[2]
        a = self.archive_in(archive)+self.kind_emb.weight[1] if A else archive.new_zeros(B, 0, self.cfg.d_model)
        x = torch.cat([self.trust.expand(B, 1, -1)+self.kind_emb.weight[0], a, r], 1)
        mask = build_attention_mask(A, K, arc_valid, reg_valid, self.cfg.n_heads).to(x.device)
        z = self.encoder(x, mask=mask)[:, 1+A:]
        out = self.head(z)
        # slope_0 = softplus(out[0]) (scale), increments softplus(out[1:]) -> non-decreasing slopes
        incr = F.softplus(out[..., 1:]-3.)
        slopes = .02*F.softplus(out[..., :1])+torch.cumsum(incr, -1)*.05
        dx = GRID[1]-GRID[0]
        H = torch.cat([torch.zeros_like(slopes[..., :1]), torch.cumsum(slopes*dx, -1)], -1)
        return H


def hazard_at(H: torch.Tensor, f: torch.Tensor):
    """Linear interpolation of H (..., 101) at forces f (...), with the last slope past 10 N.
    Returns (H(f), h(f)) where h is the local slope."""
    g = GRID.to(H.device)
    dx = g[1]-g[0]
    pos = (f/dx).clamp(min=0)
    i = pos.floor().long().clamp(max=len(g)-2)
    w = pos-i
    H0 = torch.gather(H, -1, i.unsqueeze(-1)).squeeze(-1)
    H1 = torch.gather(H, -1, (i+1).unsqueeze(-1)).squeeze(-1)
    slope = (H1-H0)/dx
    return H0+w*(H1-H0), slope


def censored_nll(H, lower, upper, broke, mask, exact=None):
    """Per-region censored NLL averaged over labelled regions.
    lower/upper/broke/mask (B, K); upper ignored where not broke. exact: thresholds for the open-book loss."""
    if exact is not None:
        Ht, ht = hazard_at(H, exact)
        nll = Ht-torch.log(ht.clamp_min(1e-6))
    else:
        HL, _ = hazard_at(H, lower)
        HU, _ = hazard_at(H, torch.where(broke, upper, lower))
        gap = (HU-HL).clamp_min(1e-6)
        nll = HL-torch.where(broke, torch.log(-torch.expm1(-gap)), torch.zeros_like(gap))
    return (nll*mask).sum()/mask.sum().clamp_min(1)


def median_threshold(H: torch.Tensor) -> torch.Tensor:
    """Force where damage probability reaches 1/2 (H = ln 2), extrapolating past 10 N."""
    g = GRID.to(H.device)
    target = math.log(2.)
    above = (H >= target)
    idx = torch.where(above.any(-1), above.float().argmax(-1), torch.full_like(H[..., 0], len(g)-1).long())
    idx = idx.clamp(min=1)
    H1 = torch.gather(H, -1, idx.unsqueeze(-1)).squeeze(-1)
    H0 = torch.gather(H, -1, (idx-1).unsqueeze(-1)).squeeze(-1)
    slope = ((H1-H0)/(g[1]-g[0])).clamp_min(1e-6)
    return g[idx-1]+(target-H0)/slope
