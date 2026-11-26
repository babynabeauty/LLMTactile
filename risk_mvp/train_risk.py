"""Train the risk model and run the three step-6 self-checks.

  python risk_mvp/train_risk.py --data synthetic --out outputs/risk_mvp/risk_synth   (CPU, minutes)

Self-checks (validation set, labelled regions, error = |log(predicted median threshold / true threshold)|):
  (a) censored training vs open-book training (true thresholds as labels): the gap between them;
  (b) error with an archive of 0, 1, 2, 4, 6, 8, 10 cards;
  (c) objects whose VLM labels are systematically wrong: how much the prediction moves when the query
      region's VLM label is flipped fragile<->strong, with 0 vs 10 archive cards (should shrink with 10).
--data collect: episodes from collect.py (collect_data.py builds the same tensors), split by instance.
  python risk_mvp/train_risk.py --data collect --collect outputs/risk_mvp/collect \
      --appearance outputs/risk_mvp/appearance --out outputs/risk_mvp/risk_collect
  Compares the trained model with the handwritten log-normal prior from Astra's labels (prior.py) on the
  held-out instances: censored NLL and |log(predicted median / true threshold)| on the labelled regions,
  per condition (plain / visible / hidden / decoy) and versus the number of archive cards (0..10).
  Writes report.json and compare.png.
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
    ap.add_argument('--collect', type=Path, default=Path('outputs/risk_mvp/collect'))
    ap.add_argument('--appearance', type=Path, default=Path('outputs/risk_mvp/appearance'))
    ap.add_argument('--max-rot-deg', type=float, default=20.)
    a = ap.parse_args()
    if a.data == 'collect':
        return main_collect(a)
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


@torch.no_grad()
def compare(model, samples, cfg, sigma=.5):
    """Per labelled region of every sample: model and prior NLL and |log median error|, with the sample's
    condition and archive size. Returns a list of dicts."""
    from risk_mvp.collect_data import prior_hazard
    from risk_mvp.risk_model import GRID
    out = []
    for i in range(0, len(samples), 256):
        ss = samples[i:i+256]
        b = collate(ss, cfg)
        H = run_model(model, b)
        Hp = torch.zeros_like(H)
        for j, s in enumerate(ss):
            k = s['cards'].shape[0]
            Hp[j, :k] = torch.as_tensor(prior_hazard(s['obj'], k, sigma), dtype=H.dtype)
        for name, HH in (('model', H), ('prior', Hp)):
            med = median_threshold(HH)
            err = torch.abs(torch.log(med.clamp_min(1e-3)/b['thr']))
            for j, s in enumerate(ss):
                for r in torch.nonzero(b['mask'][j] > 0).flatten().tolist():
                    nll = censored_nll(HH[j:j+1, r:r+1], b['lower'][j:j+1, r:r+1], b['upper'][j:j+1, r:r+1],
                                       b['broke'][j:j+1, r:r+1], b['mask'][j:j+1, r:r+1]).item()
                    out.append(dict(who=name, nll=nll, err=float(err[j, r]), condition=s['obj']['condition'],
                                    archive=len(s['archive']), oid=s['obj']['oid'],
                                    weak=s['obj']['thr'][r] < .7*np.median(s['obj']['thr'])))
    assert len(GRID) == 101
    return out


def summarise_rows(rows, keys):
    groups = {}
    for r in rows:
        groups.setdefault(tuple(r[k] for k in keys), []).append(r)
    return {'/'.join(map(str, g)): dict(n=len(v), nll=float(np.mean([x['nll'] for x in v])),
                                         median_log_err=float(np.median([x['err'] for x in v])))
            for g, v in sorted(groups.items())}


def main_collect(a):
    from risk_mvp.collect_data import build
    a.out.mkdir(parents=True, exist_ok=True)
    cfg = RiskConfig()
    t0 = time.time()
    log = lambda m: print(f'[{time.time()-t0:6.0f}s] {m}', flush=True)
    train_s, val_s, info = build(a.collect, a.appearance, cfg, max_rot_deg=a.max_rot_deg)
    log(f'data: {info}')
    model = train(train_s, cfg, a.epochs, log=log)
    report = dict(config={k: str(v) for k, v in vars(a).items()}, data=info)
    rows = compare(model, val_s, cfg)
    report['overall'] = summarise_rows(rows, ['who'])
    report['by_condition'] = summarise_rows(rows, ['who', 'condition'])
    report['by_object'] = summarise_rows(rows, ['who', 'oid'])
    curve = {}
    for n in (0, 1, 2, 4, 6, 10):
        _, vs, _ = build(a.collect, a.appearance, cfg, max_rot_deg=a.max_rot_deg, n_archive=n)
        r = compare(model, vs, cfg)
        curve[n] = dict(model=summarise_rows([x for x in r if x['who'] == 'model'], ['who'])['model'],
                        model_by_condition=summarise_rows([x for x in r if x['who'] == 'model'], ['condition']),
                        prior=summarise_rows([x for x in r if x['who'] == 'prior'], ['who'])['prior'])
        log(f"archive {n:2d}: model err {curve[n]['model']['median_log_err']:.3f} nll {curve[n]['model']['nll']:.3f}"
            f" | prior err {curve[n]['prior']['median_log_err']:.3f} nll {curve[n]['prior']['nll']:.3f}")
    report['archive_curve'] = curve
    torch.save(dict(state=model.state_dict(), cfg=cfg.__dict__), a.out/'risk_collect.pt')
    (a.out/'report.json').write_text(json.dumps(report, indent=1)+'\n')
    print(f"{'':22s}{'model err':>10s}{'prior err':>10s}{'model nll':>10s}{'prior nll':>10s}   n")
    for c in ('plain', 'visible', 'hidden', 'decoy'):
        m, p_ = report['by_condition'].get(f'model/{c}'), report['by_condition'].get(f'prior/{c}')
        if m:
            print(f'{c:22s}{m["median_log_err"]:10.3f}{p_["median_log_err"]:10.3f}{m["nll"]:10.3f}{p_["nll"]:10.3f}   {m["n"]}')
    m, p_ = report['overall']['model'], report['overall']['prior']
    print(f'{"all":22s}{m["median_log_err"]:10.3f}{p_["median_log_err"]:10.3f}{m["nll"]:10.3f}{p_["nll"]:10.3f}   {m["n"]}')
    plot_compare(report, a.out/'compare.png')
    log(f'WROTE {a.out}')


def plot_compare(report, path):
    import os
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/risk_mvp_mpl')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    conds = ['plain', 'visible', 'hidden', 'decoy']
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    for ax, key, lab in ((axs[0], 'median_log_err', 'median |log(pred / true threshold)|'), (axs[1], 'nll', 'censored NLL')):
        x = np.arange(len(conds))
        for k, who in enumerate(('prior', 'model')):
            v = [report['by_condition'].get(f'{who}/{c}', {}).get(key, np.nan) for c in conds]
            ax.bar(x+(k-.5)*.38, v, .38, label='handwritten prior (Astra labels)' if who == 'prior' else 'trained model')
        ax.set_xticks(x, conds)
        ax.set_ylabel(lab)
        ax.legend(fontsize=8)
        ax.set_title(f'held-out instances, labelled regions ({lab.split(" ")[0]} lower is better)', fontsize=9)
    ax = axs[2]
    cur = report['archive_curve']
    ns = sorted(int(n) for n in cur)
    for c in conds:
        ax.plot(ns, [cur[n]['model_by_condition'].get(c, {}).get('median_log_err', np.nan) for n in ns], 'o-', label=f'model, {c}')
    ax.plot(ns, [cur[n]['prior']['median_log_err'] for n in ns], 'k--', label='prior (no archive)')
    ax.set_xlabel('archive cards (earlier grasps of this instance)')
    ax.set_ylabel('median |log error|')
    ax.set_title('does the model improve with grasps of the same object?', fontsize=9)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


if __name__ == '__main__':
    main()
