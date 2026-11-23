"""Train the risk model and run the three step-6 self-checks.

  python risk_mvp/train_risk.py --data synthetic --out outputs/risk_mvp/risk_synth   (CPU, minutes)

Self-checks (validation set, labelled regions, error = |log(predicted median threshold / true threshold)|):
  (a) censored training vs open-book training (true thresholds as labels): the gap between them;
  (b) error with an archive of 0, 1, 2, 4, 6, 8, 10 cards;
  (c) objects whose VLM labels are systematically wrong: how much the prediction moves when the query
      region's VLM label is flipped fragile<->strong, with 0 vs 10 archive cards (should shrink with 10).
--data collect will read the collected episodes once collect.py exists; the tensors are the same.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from risk_mvp.risk_model import RiskConfig, RiskModel, censored_nll, median_threshold, N_LABELS  # noqa: E402
from risk_mvp.synthetic import World, collate  # noqa: E402


def run_model(model, batch):
    return model(batch['region'], batch['hist'], batch['reg_valid'], batch['archive'], batch['arc_valid'])


def train(samples, cfg, epochs, open_book=False, batch_size=128, lr=3e-4, seed=0, log=None):
    torch.manual_seed(seed)
    model = RiskModel(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    steps = epochs*int(np.ceil(len(samples)/batch_size))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    for ep in range(epochs):
        order = rng.permutation(len(samples))
        losses = []
        model.train()
        for i in range(0, len(order), batch_size):
            b = collate([samples[j] for j in order[i:i+batch_size]], cfg)
            H = run_model(model, b)
            if open_book:
                loss = censored_nll(H, None, None, None, b['mask'], exact=b['thr'])
            else:
                loss = censored_nll(H, b['lower'], b['upper'], b['broke'], b['mask'])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            opt.step()
            sched.step()
            losses.append(loss.item())
        if log:
            log(f'{"open-book" if open_book else "censored"} epoch {ep+1}/{epochs} loss {np.mean(losses):.4f}')
    return model.eval()


@torch.no_grad()
def log_error(model, samples, cfg, which='labelled'):
    errs = []
    for i in range(0, len(samples), 256):
        b = collate(samples[i:i+256], cfg)
        med = median_threshold(run_model(model, b))
        e = torch.abs(torch.log(med.clamp_min(1e-3)/b['thr']))
        m = b['mask'] > 0 if which == 'labelled' else b['reg_valid']
        errs.append(e[m])
    return float(torch.cat(errs).median())


@torch.no_grad()
def label_sensitivity(model, samples, cfg):
    """Median |log ratio| of predicted median when the labelled regions' VLM label is flipped (0<->2)."""
    b = collate(samples, cfg)
    base = median_threshold(run_model(model, b))
    region = b['region'].clone()
    lab = region[..., :N_LABELS]
    region[..., :N_LABELS] = lab.flip(-1)    # fragile <-> strong, normal stays
    b2 = dict(b, region=region)
    flipped = median_threshold(run_model(model, b2))
    m = (b['mask'] > 0) & (b['vlm'] != 1)
    return float(torch.abs(torch.log(flipped/base))[m].median())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', choices=['synthetic', 'collect'], default='synthetic')
    ap.add_argument('--out', type=Path, default=Path('outputs/risk_mvp/risk_synth'))
    ap.add_argument('--train-instances', type=int, default=1500)
    ap.add_argument('--val-instances', type=int, default=200)
    ap.add_argument('--epochs', type=int, default=8)
    a = ap.parse_args()
    if a.data == 'collect':
        raise SystemExit('collect data loader comes with collect.py (step 5)')
    a.out.mkdir(parents=True, exist_ok=True)
    cfg = RiskConfig()
    t0 = time.time()
    log = lambda m: print(f'[{time.time()-t0:6.0f}s] {m}', flush=True)
    world = World(seed=0, cfg=cfg)
    train_s = world.samples(a.train_instances)
    val_world = World(seed=1, cfg=cfg)
    val_s = val_world.samples(a.val_instances)
    log(f'{len(train_s)} train samples, {len(val_s)} val samples')

    cens = train(train_s, cfg, a.epochs, log=log)
    book = train(train_s, cfg, a.epochs, open_book=True, log=log)
    report = dict(config={k: str(v) for k, v in vars(a).items()} | dict(model=cfg.__dict__), checks={})
    report['checks']['a_censored_vs_open_book'] = dict(
        censored_median_abs_log_err=log_error(cens, val_s, cfg), open_book_median_abs_log_err=log_error(book, val_s, cfg))

    curve = {}
    for n in (0, 1, 2, 4, 6, 8, 10):
        s = World(seed=2, cfg=cfg).samples(a.val_instances, n_archive=n)
        curve[n] = log_error(cens, s, cfg)
    report['checks']['b_error_vs_archive_size'] = curve

    bad0 = World(seed=3, cfg=cfg).samples(a.val_instances, n_archive=0, bad=True)
    bad10 = World(seed=3, cfg=cfg).samples(a.val_instances, n_archive=10, bad=True)
    good10 = World(seed=4, cfg=cfg).samples(a.val_instances, n_archive=10, bad=False)
    report['checks']['c_vlm_label_dependence'] = dict(
        wrong_vlm_archive0=label_sensitivity(cens, bad0, cfg), wrong_vlm_archive10=label_sensitivity(cens, bad10, cfg),
        right_vlm_archive10=label_sensitivity(cens, good10, cfg),
        wrong_vlm_err_archive0=log_error(cens, bad0, cfg), wrong_vlm_err_archive10=log_error(cens, bad10, cfg))
    c = report['checks']
    report['pass'] = dict(
        a=c['a_censored_vs_open_book']['censored_median_abs_log_err'] < 1.5*c['a_censored_vs_open_book']['open_book_median_abs_log_err']+.05,
        b=curve[10] < curve[0],
        c=c['c_vlm_label_dependence']['wrong_vlm_archive10'] < c['c_vlm_label_dependence']['wrong_vlm_archive0'])
    torch.save(dict(state=cens.state_dict(), cfg=cfg.__dict__), a.out/'risk_censored.pt')
    (a.out/'report.json').write_text(json.dumps(report, indent=1)+'\n')
    log(json.dumps(report['checks'], indent=1))
    log(f'pass: {report["pass"]}')


if __name__ == '__main__':
    main()
