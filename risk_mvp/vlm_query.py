"""Ask Astra (through the Codex CLI, `codex exec`) how strong each numbered region of an object looks.

Input: the region-numbered 4-view sheets from objects_check.py (<id>_plain.png, <id>_cue.png and its
summary.json: which regions are visible, which region carries the painted cue). For every object x
variant the sheet is sent --repeats times; each answer is JSON:
  object:  strength fragile|normal|strong, a force range in N it expects the object to take without
           damage, material guess, confidence
  regions: per visible region id: label fragile|normal|strong, confidence 0-1, short visual reason
The repeats give a self-consistency score per region (fraction agreeing with the majority label), which
is what risk_model's "VLM confidence" input gets together with the stated confidence.

  python risk_mvp/vlm_query.py --images outputs/risk_mvp/objects --out outputs/risk_mvp/vlm
  python risk_mvp/vlm_query.py --objects 065-a_cups 018_plum --variants cue --repeats 1   # quick try
  python risk_mvp/vlm_query.py --images outputs/risk_mvp/appearance --out outputs/risk_mvp/vlm_cues
                                 # every variant of appearance.py (plain + one mark per contactable region)
Writes <id>_<variant>.json (parsed answers + raw text), labels.json (merged, one entry per object x
variant: majority label, agreement, mean confidence, reason per region; object-level answer) and
vlm_labels.png. Send back labels.json and vlm_labels.png (and a raw .json if anything failed to parse).

Codex flags differ between versions: `codex exec --help` is read once and only flags it lists are used
(-i/--image for the picture, --output-last-message for the answer file, --output-schema for strict JSON,
--skip-git-repo-check, -s read-only). Without --output-last-message the JSON is taken from stdout.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

LABELS = ('fragile', 'normal', 'strong')

PROMPT = """You are helping a robot hand grasp objects without crushing or bruising them.

The image shows ONE object from four directions (2x2 grid, the camera circles the object; 35 degrees above
the table). The object's surface is divided into numbered regions; a number is drawn on a region where
that region is visible in that view. Regions that appear in no view: {unseen}.
Visible region numbers: {seen}.

Judge from what you can see (shape, material, wall thickness, colour, marks, bruises, creases, dents)
how much squeezing force each visible region can take before it is damaged, RELATIVE to the rest of this
object: "fragile" (clearly weaker than the rest), "normal", or "strong" (clearly stronger than the rest).
Most regions of an undamaged uniform object are "normal". Then judge the whole object by the fingertip
force at which you expect it to be damaged: "fragile" (below about 2 N), "normal" (about 2-5 N) or
"strong" (above about 5 N), and give the fingertip force range in newtons you expect it to take without
damage. Decide from what you see; do not default to the middle class.

Answer with JSON only, no other text, exactly this shape:
{{"object": {{"name": "<what the object is>", "material": "<material>", "strength": "fragile|normal|strong",
  "safe_force_N": [<low>, <high>], "confidence": <0-1>}},
 "regions": [{{"id": <region number>, "label": "fragile|normal|strong", "confidence": <0-1>,
   "reason": "<at most 15 words: what you see>"}}, ...]}}
Include every visible region number exactly once."""

SCHEMA = {
    'type': 'object', 'additionalProperties': False, 'required': ['object', 'regions'],
    'properties': {
        'object': {'type': 'object', 'additionalProperties': False,
                   'required': ['name', 'material', 'strength', 'safe_force_N', 'confidence'],
                   'properties': {'name': {'type': 'string'}, 'material': {'type': 'string'},
                                  'strength': {'type': 'string', 'enum': list(LABELS)},
                                  'safe_force_N': {'type': 'array', 'items': {'type': 'number'}},
                                  'confidence': {'type': 'number'}}},
        'regions': {'type': 'array', 'items': {
            'type': 'object', 'additionalProperties': False, 'required': ['id', 'label', 'confidence', 'reason'],
            'properties': {'id': {'type': 'integer'}, 'label': {'type': 'string', 'enum': list(LABELS)},
                           'confidence': {'type': 'number'}, 'reason': {'type': 'string'}}}}}}

# what the category should be called at the object level (objects.CATEGORIES). The prompt defines the
# classes by force only: an earlier prompt named example objects matching these categories (soft fruit, thin
# cup, foam brick, tennis ball), so the object-level answer could be copied from the prompt.
EXPECTED_OBJECT = {'soft_fruit': 'fragile', 'firm_fruit': 'normal', 'thin_cup': 'fragile', 'sturdy': 'strong'}


# ---------------------------------------------------------------- codex call
def codex_flags(codex: str) -> set:
    """Long and short options listed by `codex exec --help`."""
    try:
        h = subprocess.run([codex, 'exec', '--help'], capture_output=True, text=True, timeout=60)
        text = h.stdout+h.stderr
    except Exception as exc:                          # noqa: BLE001
        raise SystemExit(f'cannot run {codex} exec --help: {exc}')
    return set(re.findall(r'(?<![\w-])(--?[a-zA-Z][\w-]*)', text))


def ask(codex: str, flags: set, image: Path, prompt: str, model=None, timeout=600, log: Path = None,
        effort=None) -> str:
    """One `codex exec` call with the image attached; returns the final message text. codex's own
    progress output goes to `log` (tail -f it to see what a call is doing)."""
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        cmd = [codex, 'exec']
        if '--skip-git-repo-check' in flags:
            cmd.append('--skip-git-repo-check')
        if '--sandbox' in flags:
            cmd += ['--sandbox', 'read-only']
        if model and ('--model' in flags or '-m' in flags):
            cmd += ['--model', model] if '--model' in flags else ['-m', model]
        if effort and ('--config' in flags or '-c' in flags):
            cmd += ['-c', f'model_reasoning_effort="{effort}"']
        if '--image' in flags:
            cmd += ['--image', str(image.resolve())]
        elif '-i' in flags:
            cmd += ['-i', str(image.resolve())]
        else:
            raise SystemExit('this codex exec has no --image/-i option; update codex')
        out_file = td/'last.txt'
        if '--output-last-message' in flags:
            cmd += ['--output-last-message', str(out_file)]
        if '--output-schema' in flags:
            (td/'schema.json').write_text(json.dumps(SCHEMA))
            cmd += ['--output-schema', str(td/'schema.json')]
        cmd.append(prompt)
        log = log or td/'codex.log'
        with open(log, 'w') as fh:
            fh.write(' '.join(cmd[:-1])+' <prompt>\n\n')
            fh.flush()
            try:
                r = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, text=True, timeout=timeout, cwd=td,
                                   stdin=subprocess.DEVNULL)
            except subprocess.TimeoutExpired:
                raise RuntimeError(f'codex did not answer within {timeout} s (see {log})')
        if out_file.exists() and out_file.read_text().strip():
            return out_file.read_text()
        text = Path(log).read_text()
        if r.returncode != 0:
            raise RuntimeError(f'codex exit {r.returncode}: {text[-1500:]}')
        return text


def preflight(codex: str, flags: set, model=None, effort=None, timeout=180, log: Path = None):
    """One small text-only call before the real ones, so a codex that cannot answer (no network, a CLI
    too old for the configured model, not logged in) stops the run at once with its own message."""
    cmd = [codex, 'exec']
    if '--skip-git-repo-check' in flags:
        cmd.append('--skip-git-repo-check')
    if '--sandbox' in flags:
        cmd += ['--sandbox', 'read-only']
    if model:
        cmd += ['--model', model] if '--model' in flags else ['-m', model]
    if effort and ('--config' in flags or '-c' in flags):
        cmd += ['-c', f'model_reasoning_effort="{effort}"']
    cmd.append('Reply with the single word: ok')
    with tempfile.TemporaryDirectory() as td:
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=td, stdin=subprocess.DEVNULL)
            text = r.stdout+r.stderr
        except subprocess.TimeoutExpired as exc:
            text = ((exc.stdout or b'').decode(errors='replace') if isinstance(exc.stdout, bytes) else (exc.stdout or '')) + \
                   ((exc.stderr or b'').decode(errors='replace') if isinstance(exc.stderr, bytes) else (exc.stderr or ''))
            r = None
    if log:
        log.write_text(' '.join(cmd)+'\n\n'+text)
    failed = r is None or r.returncode != 0 or '"type":"error"' in text or 'requires a newer version' in text
    if failed:
        hint = ''
        if 'requires a newer version of Codex' in text:
            hint = ('\n-> the configured model needs a newer codex CLI: upgrade codex, or pass --model with a model '
                    'this codex supports')
        elif 'Reconnecting' in text or r is None:
            hint = '\n-> codex cannot reach the server: check the network / proxy (https_proxy) in this shell'
        elif 'login' in text.lower() or '401' in text:
            hint = '\n-> codex is not logged in: run codex login'
        tail = '\n'.join(text.strip().splitlines()[-12:])
        raise SystemExit(f'codex preflight failed ({"timeout" if r is None else f"exit {r.returncode}"}):\n{tail}{hint}')
    print('codex preflight ok', flush=True)


def parse_answer(text: str, seen: list) -> dict:
    """JSON object from the answer (the last {...} block if there is extra text), checked and cleaned:
    labels in LABELS, confidences clipped to [0, 1], only visible region ids, one entry per id."""
    s = text.strip()
    s = re.sub(r'^```(?:json)?\s*|\s*```$', '', s)
    try:
        d = json.loads(s)
    except json.JSONDecodeError:
        start = [m.start() for m in re.finditer(r'\{\s*"object"', s)]
        if not start:
            raise ValueError('no JSON object in the answer')
        dec = json.JSONDecoder()
        d, _ = dec.raw_decode(s[start[-1]:])
    o = d['object']
    if o.get('strength') not in LABELS:
        raise ValueError(f"object strength {o.get('strength')!r}")
    lo_hi = [float(x) for x in o.get('safe_force_N', [])][:2]
    obj = dict(name=str(o.get('name', '')), material=str(o.get('material', '')), strength=o['strength'],
               safe_force_N=sorted(lo_hi) if len(lo_hi) == 2 else None,
               confidence=min(1., max(0., float(o.get('confidence', .5)))))
    regions = {}
    for r in d['regions']:
        rid = int(r['id'])
        if rid not in seen or rid in regions or r.get('label') not in LABELS:
            continue
        regions[rid] = dict(label=r['label'], confidence=min(1., max(0., float(r.get('confidence', .5)))),
                            reason=str(r.get('reason', ''))[:200])
    return dict(object=obj, regions=regions, missing=[i for i in seen if i not in regions])


def merge(answers: list, seen: list) -> dict:
    """Majority label per region over the repeats, agreement = share of repeats that gave it."""
    out = dict(regions={}, object={})
    for rid in seen:
        got = [a['regions'][rid] for a in answers if rid in a['regions']]
        if not got:
            continue
        cnt = Counter(g['label'] for g in got)
        lab, n = max(cnt.items(), key=lambda kv: (kv[1], -LABELS.index(kv[0])))   # ties -> more fragile
        agree = n/len(answers)
        conf = sum(g['confidence'] for g in got if g['label'] == lab)/n
        out['regions'][rid] = dict(label=lab, agreement=round(agree, 3), stated_confidence=round(conf, 3),
                                   confidence=round(agree*conf, 3), votes=dict(cnt),
                                   reason=next(g['reason'] for g in got if g['label'] == lab))
    objs = [a['object'] for a in answers]
    if objs:
        cnt = Counter(o['strength'] for o in objs)
        lab, n = max(cnt.items(), key=lambda kv: (kv[1], -LABELS.index(kv[0])))
        rng = [o['safe_force_N'] for o in objs if o['safe_force_N']]
        out['object'] = dict(strength=lab, agreement=round(n/len(objs), 3), votes=dict(cnt),
                             safe_force_N=[round(sorted(x[0] for x in rng)[len(rng)//2], 3),
                                           round(sorted(x[1] for x in rng)[len(rng)//2], 3)] if rng else None,
                             confidence=round(sum(o['confidence'] for o in objs)/len(objs), 3),
                             name=Counter(o['name'] for o in objs).most_common(1)[0][0],
                             material=Counter(o['material'] for o in objs).most_common(1)[0][0])
    return out


def mark_region(obj_summary: dict, variant: str):
    """Region carrying the painted mark in this variant (None for plain). appearance.py stores it per
    variant; objects_check.py has one cue region per object for its 'cue' variant."""
    view = obj_summary['views'][variant]
    if 'cue_region' in view:
        return view['cue_region']
    return obj_summary.get('cue_region') if variant == 'cue' else None


# ---------------------------------------------------------------- picture
def plot_labels(labels: dict, summary: dict, k: int, path: Path):
    import os
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/risk_mvp_mpl')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    colour = dict(fragile='#d62728', normal='#f2c14e', strong='#2ca02c')
    rows = list(labels)
    fig, ax = plt.subplots(figsize=(2.2+1.05*(k+2), .45*len(rows)+1.4))
    for i, key in enumerate(rows):
        oid, variant = key.rsplit('/', 1)
        L = labels[key]
        y = len(rows)-1-i
        for r in range(k):
            reg = L['regions'].get(str(r)) or L['regions'].get(r)
            if reg is None:
                vis = r in L.get('regions_seen', [])
                ax.add_patch(Rectangle((r, y), .95, .9, color='#bbbbbb' if vis else '#e6e6e6'))
                ax.text(r+.47, y+.45, 'no answer' if vis else 'not visible', ha='center', va='center', fontsize=6,
                        color='#333' if vis else '#888')
            else:
                ax.add_patch(Rectangle((r, y), .95, .9, color=colour[reg['label']], alpha=.35+.65*reg['confidence']))
                ax.text(r+.47, y+.45, f"{reg['label'][:4]}\n{reg['agreement']:.0%}", ha='center', va='center', fontsize=6)
            cue = L.get('cue_region')
            if cue is not None and cue == r:
                ax.add_patch(Rectangle((r-.02, y-.02), .99, .94, fill=False, ec='k', lw=2))
        o = L['object']
        if o:
            exp = EXPECTED_OBJECT.get(summary.get(oid, {}).get('category'))
            ax.add_patch(Rectangle((k+.3, y), 1.6, .9, color=colour[o['strength']], alpha=.6))
            rng = o['safe_force_N']
            ax.text(k+1.1, y+.45, f"{o['strength']}" + (f" {rng[0]:g}-{rng[1]:g} N" if rng else '')
                    + ('' if exp is None else (' ok' if exp == o['strength'] else f' (exp {exp})')),
                    ha='center', va='center', fontsize=6)
    ax.set_xlim(-.1, k+2)
    ax.set_ylim(-.2, len(rows))
    ax.set_yticks([len(rows)-1-i+.45 for i in range(len(rows))], rows, fontsize=7)
    ax.set_xticks([r+.47 for r in range(k)]+[k+1.1], [f'region {r}' for r in range(k)]+['whole object'], fontsize=7)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title('Astra region labels (colour = label, shade = confidence, % = agreement over repeats; '
                 'black box = painted cue region)', fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--images', type=Path, default=ROOT/'outputs/risk_mvp/objects')
    ap.add_argument('--out', type=Path, default=ROOT/'outputs/risk_mvp/vlm')
    ap.add_argument('--objects', nargs='*', default=None)
    ap.add_argument('--variants', nargs='*', default=None,
                    help='default: every variant in summary.json (objects_check: plain, cue; appearance.py: '
                         'plain, cue_r<r>)')
    ap.add_argument('--repeats', type=int, default=3)
    ap.add_argument('--jobs', type=int, default=4, help='parallel codex calls')
    ap.add_argument('--codex', default=shutil.which('codex') or '/usr/local/bin/codex')
    ap.add_argument('--model', default=None, help='codex model (default: codex config)')
    ap.add_argument('--timeout', type=int, default=600, help='seconds per codex call')
    ap.add_argument('--from-logs', action='store_true',
                    help='no codex calls: build labels.json from the answers already in <out>/logs (e.g. after '
                         'stopping a run); variants without any answer are left out')
    ap.add_argument('--fresh', action='store_true', help='ask again even where an earlier answer is saved')
    ap.add_argument('--no-preflight', action='store_true', help='skip the one text-only test call')
    ap.add_argument('--effort', default=None, help='codex reasoning effort, e.g. low / medium (default: codex config)')
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    summary = json.loads((a.images/'summary.json').read_text())
    flags = set() if a.from_logs else codex_flags(a.codex)
    print('codex', a.codex, 'uses', sorted(f for f in flags if f in {'--image', '-i', '--output-last-message',
                                                                     '--output-schema', '--skip-git-repo-check',
                                                                     '--sandbox', '--model', '-c'}), flush=True)
    jobs = []
    for oid in a.objects or list(summary):
        for v in a.variants or list(summary[oid]['views']):
            img = a.images/f'{oid}_{v}.png'
            if not img.exists():
                print('missing', img, flush=True)
                continue
            seen = summary[oid]['views'][v]['regions_seen']
            k = len(summary[oid]['regions']['centroid']) if 'centroid' in summary[oid].get('regions', {}) else None
            unseen = [r for r in range(k)] if k else []
            unseen = [r for r in unseen if r not in seen]
            prompt = PROMPT.format(seen=', '.join(map(str, seen)), unseen=', '.join(map(str, unseen)) or 'none')
            for rep in range(a.repeats):
                jobs.append((oid, v, rep, img, seen, prompt))

    logs = a.out/'logs'
    logs.mkdir(parents=True, exist_ok=True)

    def recover(job):
        """An earlier answer to this call: the saved answer file, else the answer at the end of the codex log
        of a run that was stopped (the prompt echoed at the top of the log is not valid JSON, so only a
        finished answer parses)."""
        oid, v, rep, img, seen, prompt = job
        for path in (logs/f'{oid}_{v}_{rep}.answer.txt', logs/f'{oid}_{v}_{rep}.log'):
            if path.exists():
                try:
                    text = path.read_text(errors='replace')
                    return text, parse_answer(text, seen)
                except Exception:                     # noqa: BLE001
                    continue
        return None

    results = {}
    todo = []
    for job in jobs:
        got = None if a.fresh else recover(job)
        if got is not None:
            results.setdefault(job[:2], []).append(dict(repeat=job[2], raw=got[0], parsed=got[1], error=None,
                                                        reused=True))
        elif not a.from_logs:
            todo.append(job)
    n_reused = sum(map(len, results.values()))
    print(f'{len(jobs)} calls planned, {n_reused} answered earlier (reused)'
          + ('' if a.from_logs else f', {len(todo)} to ask'), flush=True)
    if todo and not a.no_preflight:
        preflight(a.codex, flags, a.model, a.effort, log=logs/'preflight.log')
    if todo:
        print(f'{len(todo)} codex calls, {a.jobs} at a time; each can take a few minutes. Every answer is saved '
              f'as it arrives: stopping and rerunning the same command continues where it stopped. '
              f'Live codex output: tail -f {logs}/<object>_<variant>_<repeat>.log', flush=True)
    t0 = time.time()

    def run(job):
        oid, v, rep, img, seen, prompt = job
        log = logs/f'{oid}_{v}_{rep}.log'
        print(f'[{time.time()-t0:6.0f} s] asking {oid}/{v} repeat {rep}  (log {log})', flush=True)
        try:
            text = ask(a.codex, flags, img, prompt, a.model, a.timeout, log=log, effort=a.effort)
        except Exception as exc:                      # noqa: BLE001
            return job, None, f'{type(exc).__name__}: {exc}'
        (logs/f'{oid}_{v}_{rep}.answer.txt').write_text(text)
        try:
            return job, (text, parse_answer(text, seen)), None
        except Exception as exc:                      # noqa: BLE001
            return job, (text, None), f'parse: {type(exc).__name__}: {exc}'

    with ThreadPoolExecutor(max(1, a.jobs)) as ex:
        for fut in as_completed([ex.submit(run, j) for j in todo]):
            job, res, err = fut.result()
            oid, v, rep = job[:3]
            results.setdefault((oid, v), []).append(dict(repeat=rep, raw=res[0] if res else None,
                                                         parsed=res[1] if res else None, error=err))
            print(f'[{time.time()-t0:6.0f} s] {oid}/{v} repeat {rep}: ' + (err or 'ok'), flush=True)
    labels = {}
    k_max = 0
    for (oid, v), rs in sorted(results.items()):
        seen = summary[oid]['views'][v]['regions_seen']
        ok = [r['parsed'] for r in rs if r['parsed']]
        (a.out/f'{oid}_{v}.json').write_text(json.dumps(
            [dict(r, parsed=None if r['parsed'] is None else dict(r['parsed'], regions={str(i): x for i, x in r['parsed']['regions'].items()}))
             for r in rs], indent=1)+'\n')
        m = merge(ok, seen) if ok else dict(regions={}, object={})
        m['regions'] = {str(i): x for i, x in m['regions'].items()}
        m.update(n_answers=len(ok), n_asked=len(rs), regions_seen=seen, cue_region=mark_region(summary[oid], v),
                 category=summary[oid].get('category'))
        labels[f'{oid}/{v}'] = m
        k_max = max(k_max, len(summary[oid]['regions']['centroid']) if 'centroid' in summary[oid].get('regions', {})
                    else max(seen)+1)
    (a.out/'labels.json').write_text(json.dumps(labels, indent=1)+'\n')
    # console check: is the cue region called fragile, is the object strength the expected one
    print(f"{'object/variant':24s} answers  object(expected)          cue region label")
    for key, m in labels.items():
        o = m['object']
        exp = EXPECTED_OBJECT.get(m['category'])
        cue = None if m['cue_region'] is None else m['regions'].get(str(m['cue_region']))
        print(f"{key:24s} {m['n_answers']}/{m['n_asked']}     {(o.get('strength', '-')+' ('+str(exp)+')'):25s} "
              + ('-' if cue is None else f"{cue['label']} ({cue['agreement']:.0%})"))
    if labels:
        plot_labels(labels, summary, k_max, a.out/'vlm_labels.png')
    print('WROTE', a.out)


if __name__ == '__main__':
    main()
