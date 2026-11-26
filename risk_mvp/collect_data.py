"""Risk-model samples from collect.py episodes (same dicts as synthetic.World.samples, so synthetic.collate,
the model and the loss are unchanged).

Per episode (one grasp of one instance), the query is every region of the object:
  region card   Astra's label for the variant this instance shows (one-hot) and confidence (0 where the
                region was not visible or not answered), a fixed hashed bag-of-words embedding of Astra's
                reason, the region geometry (Partition.features from the appearance manifest), and tactile
                features of the finger on that region during the settle phase (tangential/normal ratio,
                fraction of ticks in contact, 0)
  force history the region force during the settle phase (0.25 s at the start force, never judged, the same
                for every data policy), resampled to HIST_LEN. Deliberately not the allocation targets: the
                oracle's targets are the thresholds minus a margin and would leak the answer.
  labels        the episode's archive rows (touched regions): survived above L, or broke in (L, U]
  archive cards rows from OTHER grasps of the same instance (random subset of 0..10, or a fixed number)

Episodes whose object turned more than --max-rot-deg in the hand are dropped: their fingers no longer sit on
the snapshot regions the forces are booked to. Train / validation split by instance.
"""
from __future__ import annotations

import json
import zlib
from collections import defaultdict
from pathlib import Path

import numpy as np

from risk_mvp.risk_model import GEOM_DIM, HIST_LEN, N_LABELS, TACT_DIM, RiskConfig

EXTRA_DIM = 6      # object strength one-hot (3), log object median, log prior median of the region, prior sigma


def collect_config(**kw) -> RiskConfig:
    return RiskConfig(extra_dim=EXTRA_DIM, **kw)


def prior_features(obj: dict, k: int, sigma: float = .5) -> np.ndarray:
    """(k, EXTRA_DIM): what the handwritten prior believes, so the model learns a correction to it
    (without it the model had to recover the absolute strength from geometry alone and did worse than the
    prior)."""
    from risk_mvp import prior as P
    lab = dict(object=obj['vlm_object'], regions=obj['vlm'])
    med, conf = P.region_medians(lab, k)
    o = obj['vlm_object'] or {}
    rng = o.get('safe_force_N')
    m_obj = float(np.sqrt(rng[0]*rng[1])) if rng and min(rng) > 0 else P.CLASS_MEDIAN_N.get(o.get('strength'), 3.2)
    onehot = np.eye(N_LABELS)[LABEL_ID.get(o.get('strength'), 1)]
    return np.concatenate([np.tile(onehot, (k, 1)), np.full((k, 1), np.log(m_obj)), np.log(med)[:, None],
                           (sigma*(1+(1-conf)))[:, None]], 1)

LABEL_ID = dict(fragile=0, normal=1, strong=2)
FINGERS = ('thumb', 'index', 'middle')
SETTLE_SAMPLES = 8          # collect.py keeps every 2nd tick; settle = 15 ticks


def text_embedding(text: str, dim: int) -> np.ndarray:
    """Fixed hashed bag-of-words vector (a stand-in for a frozen sentence encoder)."""
    words = [w for w in ''.join(c.lower() if c.isalnum() else ' ' for c in text).split() if len(w) > 2]
    if not words:
        return np.zeros(dim)
    v = np.mean([np.random.default_rng(zlib.crc32(w.encode())).normal(0, 1, dim) for w in words], 0)
    return v/max(np.linalg.norm(v), 1e-9)


def region_features(manifest_obj: dict) -> np.ndarray:
    """(K, 8) = regions.Partition.features from the manifest's partition json."""
    r = manifest_obj['regions']
    area = np.array(r['area'])[:, None]*1e4
    return np.concatenate([area, np.array(r['flatness'])[:, None], np.array(r['normal']),
                           np.array(r['centroid'])*100], 1)


def resample(x: np.ndarray, n: int) -> np.ndarray:
    if len(x) == 0:
        return np.zeros(n)
    return np.interp(np.linspace(0, len(x)-1, n), np.arange(len(x)), x)


def load_episodes(collect_dir: Path, max_rot_deg: float = 20.):
    collect_dir = Path(collect_dir)
    inst = json.loads((collect_dir/'instances.json').read_text())
    eps, traces = [], {}
    for f in sorted(collect_dir.glob('*.jsonl')):
        oid = f.stem
        tr = np.load(collect_dir/f'{oid}_traces.npz')
        traces[oid] = dict(region_F=tr['region_F'], fn=tr['fn'], ft=tr['ft'])
        for line in open(f):
            e = json.loads(line)
            if e['max_rot_deg'] <= max_rot_deg:
                eps.append(e)
    return eps, inst, traces


def build(collect_dir: Path, appearance: Path, cfg: RiskConfig = RiskConfig(), max_rot_deg: float = 20.,
          val_frac: float = .2, seed: int = 0, n_archive=None, sigma: float = .5):
    """Returns (train samples, val samples, info). n_archive None: random 0..10 per sample."""
    eps, inst, traces = load_episodes(collect_dir, max_rot_deg)
    manifest = json.loads((Path(appearance)/'manifest.json').read_text())
    rng = np.random.default_rng(seed)
    by_inst = defaultdict(list)
    for e in eps:
        by_inst[(e['object'], e['instance'])].append(e)
    keys = sorted(by_inst)
    perm = rng.permutation(len(keys))
    n_val = int(round(val_frac*len(keys)))
    val_keys = {keys[i] for i in perm[:n_val]}
    geom = {oid: region_features(m) for oid, m in manifest.items()}
    train, val = [], []
    for key in keys:
        oid, i = key
        rec = inst[oid][i]
        k = len(rec['thresholds_N'])
        label = np.full(k, LABEL_ID['normal'])
        conf = np.zeros(k)
        text = np.zeros((k, cfg.text_dim))
        for r, v in rec['vlm'].items():
            r = int(r)
            label[r], conf[r] = LABEL_ID[v['label']], v['confidence']
            text[r] = text_embedding(v.get('reason', ''), cfg.text_dim)
        obj = dict(thr=np.array(rec['thresholds_N']), label=label, conf=conf, condition=rec['condition'],
                   oid=oid, instance=i, variant=rec['variant'], vlm=rec['vlm'], vlm_object=rec['vlm_object'])
        G = geom[oid]
        extra = prior_features(obj, k, sigma) if cfg.extra_dim else np.zeros((k, 0))
        if G.shape[0] != k:
            raise ValueError(f'{oid}: manifest has {G.shape[0]} regions, instances {k}')
        grasps = by_inst[key]
        rows_of = []
        cards_of = []
        for e in grasps:
            tr = traces[oid]
            t = e['trace_index']
            regF = tr['region_F'][t].astype(float)                      # (T/2, K)
            fn, ft = tr['fn'][t].astype(float), tr['ft'][t].astype(float)
            tact = np.zeros((k, TACT_DIM))
            for j, f in enumerate(FINGERS):
                r = e['finger_region'][f]
                s_fn, s_ft = fn[:SETTLE_SAMPLES, j], ft[:SETTLE_SAMPLES, j]
                tact[r] = [np.mean(s_ft/np.maximum(s_fn, 1e-3)), np.mean(s_fn > .05), 0.]
            cards = np.concatenate([np.eye(N_LABELS)[label], conf[:, None], text, G, tact, extra], 1).astype(np.float32)
            hist = {r: resample(regF[:SETTLE_SAMPLES, r], HIST_LEN) for r in range(k)}
            rows = []
            for a in e['archive']:
                r = a['region']
                broke = a['outcome'] == 'broke'
                rows.append(dict(region=r, broke=broke, lower=max(a['lower_N'], 1e-3),
                                 upper=a['upper_N'] if broke else np.inf, max_force=a['max_force_N'],
                                 hist=hist[r]))
            rows_of.append(rows)
            cards_of.append(cards)
        for g in range(len(grasps)):
            pool = [row for h, rows in enumerate(rows_of) if h != g for row in rows]
            n = rng.integers(0, min(10, len(pool))+1) if n_archive is None else min(n_archive, len(pool))
            pick = rng.choice(len(pool), size=n, replace=False) if n else []
            arc = [archive_row(G, label, conf, pool[j]) for j in pick]
            s = dict(obj=obj, cards=cards_of[g], rows=rows_of[g], archive=arc, policy=grasps[g]['policy'],
                     scale=grasps[g]['scale'],
                     pool=[archive_row(G, label, conf, row) for row in pool])   # to redraw the archive per batch
            (val if key in val_keys else train).append(s)
    info = dict(episodes=len(eps), instances=len(keys), val_instances=len(val_keys),
                train_samples=len(train), val_samples=len(val))
    return train, val, info


def archive_row(G, label, conf, row) -> np.ndarray:
    r = row['region']
    up = row['upper'] if row['broke'] else 0.
    return np.concatenate([G[r], np.eye(N_LABELS)[label[r]],
                           [conf[r], row['max_force']/10, float(row['broke']), np.log(max(row['lower'], 1e-3)),
                            np.log(up) if up else 0., float(not row['broke'])]]).astype(np.float32)


def prior_hazard(obj: dict, k: int, sigma: float = .5) -> np.ndarray:
    """The handwritten log-normal belief (prior.py) for this instance's variant, on the model's grid."""
    from risk_mvp import prior as P
    lab = dict(object=obj['vlm_object'], regions=obj['vlm'])
    return P.hazard_from_label(lab, k, sigma=sigma)


assert GEOM_DIM == 8
