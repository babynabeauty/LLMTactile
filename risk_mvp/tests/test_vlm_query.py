"""vlm_query against a fake `codex` executable: flag detection, answer parsing, merging, outputs."""
import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from risk_mvp.vlm_query import merge, parse_answer  # noqa: E402

FAKE = r'''#!/usr/bin/env python3
import json, sys, os
args = sys.argv[1:]
if args[:2] == ['exec', '--help']:
    print("Usage: codex exec [OPTIONS] [PROMPT]\n  -i, --image <FILE>...\n  -m, --model <MODEL>\n"
          "  -s, --sandbox <MODE>\n  --skip-git-repo-check\n  --output-schema <FILE>\n"
          "  -o, --output-last-message <FILE>")
    sys.exit(0)
if os.environ.get('FAKE_MODE') == 'old':
    print('ERROR: {"type":"error","status":400,"error":{"message":"The model requires a newer version of Codex."}}')
    sys.exit(1)
if '--image' not in args:          # preflight
    print('ok')
    sys.exit(0)
assert args[0] == 'exec' and '--output-schema' in args
img = args[args.index('--image')+1]
assert os.path.isabs(img) and os.path.exists(img)
json.load(open(args[args.index('--output-schema')+1]))
mode = os.environ.get('FAKE_MODE', 'json')
cue = 'cue' in os.path.basename(img)
ans = {"object": {"name": "cup", "material": "plastic", "strength": "fragile", "safe_force_N": [0.5, 2],
                  "confidence": 0.7},
       "regions": [{"id": 0, "label": "fragile" if cue else "normal", "confidence": 0.8, "reason": "crease"},
                   {"id": 2, "label": "normal", "confidence": 1.4, "reason": "smooth wall"},
                   {"id": 9, "label": "strong", "confidence": 0.5, "reason": "not a visible id"}]}
text = json.dumps(ans) if mode == 'json' else "Here you go:\n```json\n" + json.dumps(ans) + "\n```"
open(args[args.index('--output-last-message')+1], 'w').write(text)
'''


def test_parse_cleans_and_merges():
    text = 'blah {"object": {"name": "x", "material": "y", "strength": "strong", "safe_force_N": [9, 5], ' \
           '"confidence": 3}, "regions": [{"id": 1, "label": "fragile", "confidence": 0.9, "reason": "dent"}, ' \
           '{"id": 1, "label": "strong", "confidence": 0.9, "reason": "dup"}, ' \
           '{"id": 4, "label": "weird", "confidence": 0.9, "reason": "bad label"}]}'
    p = parse_answer(text, seen=[1, 2, 4])
    assert p['object']['safe_force_N'] == [5., 9.] and p['object']['confidence'] == 1.
    assert list(p['regions']) == [1] and p['regions'][1]['label'] == 'fragile'
    assert p['missing'] == [2, 4]
    q = parse_answer(text.replace('"fragile", "confidence": 0.9, "reason": "dent"',
                                  '"normal", "confidence": 0.5, "reason": "flat"'), seen=[1, 2, 4])
    m = merge([p, p, q], seen=[1, 2, 4])
    assert m['regions'][1]['label'] == 'fragile' and abs(m['regions'][1]['agreement']-2/3) < 1e-3
    assert 2 not in m['regions'] and m['object']['strength'] == 'strong'
    with pytest.raises(ValueError):
        parse_answer('no json here', seen=[0])


@pytest.mark.parametrize('mode', ['json', 'fenced'])
def test_end_to_end_with_fake_codex(tmp_path, mode):
    codex = tmp_path/'codex'
    codex.write_text(FAKE)
    codex.chmod(codex.stat().st_mode | stat.S_IEXEC)
    img = tmp_path/'imgs'
    img.mkdir()
    from PIL import Image
    for v in ('plain', 'cue'):
        Image.new('RGB', (40, 30)).save(img/f'cupx_{v}.png')
    summary = {'cupx': dict(category='thin_cup', cue_region=0,
                            regions=dict(centroid=[[0, 0, 0]]*4),
                            views=dict(plain=dict(regions_seen=[0, 2]), cue=dict(regions_seen=[0, 2, 3])))}
    (img/'summary.json').write_text(json.dumps(summary))
    out = tmp_path/'out'
    env = dict(os.environ, FAKE_MODE=mode, MPLCONFIGDIR=str(tmp_path/'mpl'))
    r = subprocess.run([sys.executable, str(ROOT/'risk_mvp/vlm_query.py'), '--images', str(img), '--out', str(out),
                        '--codex', str(codex), '--repeats', '2', '--jobs', '2'],
                       capture_output=True, text=True, env=env, timeout=300)
    assert r.returncode == 0, r.stdout+r.stderr
    labels = json.loads((out/'labels.json').read_text())
    cue = labels['cupx/cue']
    assert cue['n_answers'] == 2 and cue['regions']['0']['label'] == 'fragile'
    assert cue['regions']['2']['stated_confidence'] == 1.0 and '3' not in cue['regions']
    assert labels['cupx/plain']['regions']['0']['label'] == 'normal'
    assert cue['object']['strength'] == 'fragile' and cue['object']['safe_force_N'] == [0.5, 2.0]
    assert (out/'vlm_labels.png').exists() and (out/'cupx_cue.json').exists()


def test_preflight_stops_an_old_codex(tmp_path):
    codex = tmp_path/'codex'
    codex.write_text(FAKE)
    codex.chmod(codex.stat().st_mode | stat.S_IEXEC)
    img = tmp_path/'imgs'
    img.mkdir()
    from PIL import Image
    Image.new('RGB', (40, 30)).save(img/'cupx_plain.png')
    (img/'summary.json').write_text(json.dumps({'cupx': dict(category='thin_cup', views=dict(plain=dict(regions_seen=[0])))}))
    env = dict(os.environ, FAKE_MODE='old')
    r = subprocess.run([sys.executable, str(ROOT/'risk_mvp/vlm_query.py'), '--images', str(img), '--out',
                        str(tmp_path/'out'), '--codex', str(codex)], capture_output=True, text=True, env=env, timeout=120)
    assert r.returncode != 0 and 'newer codex CLI' in r.stderr, r.stdout+r.stderr


def test_from_logs_and_resume(tmp_path):
    """A stopped run: finished calls have the answer after the echoed prompt in their log, unfinished ones
    only the prompt. --from-logs builds labels from the finished ones; a rerun asks only the rest."""
    from risk_mvp.vlm_query import PROMPT
    codex = tmp_path/'codex'
    codex.write_text(FAKE)
    codex.chmod(codex.stat().st_mode | stat.S_IEXEC)
    img = tmp_path/'imgs'
    img.mkdir()
    from PIL import Image
    for v in ('plain', 'cue_r0'):
        Image.new('RGB', (40, 30)).save(img/f'cupx_{v}.png')
    summary = {'cupx': dict(category='thin_cup', regions=dict(centroid=[[0, 0, 0]]*4),
                            views=dict(plain=dict(regions_seen=[0, 2], cue_region=None),
                                       cue_r0=dict(regions_seen=[0, 2], cue_region=0)))}
    (img/'summary.json').write_text(json.dumps(summary))
    out = tmp_path/'out'
    logs = out/'logs'
    logs.mkdir(parents=True)
    prompt = PROMPT.format(seen='0, 2', unseen='1, 3')
    ans = json.dumps({"object": {"name": "cup", "material": "plastic", "strength": "fragile",
                                 "safe_force_N": [1, 3], "confidence": .6},
                      "regions": [{"id": 0, "label": "fragile", "confidence": .9, "reason": "mark"},
                                  {"id": 2, "label": "normal", "confidence": .8, "reason": "wall"}]})
    (logs/'cupx_cue_r0_0.log').write_text('codex exec ... <prompt>\n\nuser\n'+prompt+'\n\ncodex\n'+ans+'\n')
    (logs/'cupx_cue_r0_1.log').write_text('codex exec ... <prompt>\n\nuser\n'+prompt+'\nthinking...\n')
    env = dict(os.environ, MPLCONFIGDIR=str(tmp_path/'mpl'))
    base = [sys.executable, str(ROOT/'risk_mvp/vlm_query.py'), '--images', str(img), '--out', str(out),
            '--codex', str(codex), '--repeats', '2']
    r = subprocess.run(base+['--from-logs'], capture_output=True, text=True, env=env, timeout=300)
    assert r.returncode == 0, r.stdout+r.stderr
    labels = json.loads((out/'labels.json').read_text())
    assert list(labels) == ['cupx/cue_r0'] and labels['cupx/cue_r0']['n_answers'] == 1
    assert labels['cupx/cue_r0']['regions']['0']['label'] == 'fragile'
    r = subprocess.run(base, capture_output=True, text=True, env=env, timeout=300)
    assert r.returncode == 0, r.stdout+r.stderr
    assert '1 answered earlier (reused), 3 to ask' in r.stdout, r.stdout
    labels = json.loads((out/'labels.json').read_text())
    assert labels['cupx/cue_r0']['n_answers'] == 2 and labels['cupx/plain']['n_answers'] == 2
    r = subprocess.run(base, capture_output=True, text=True, env=env, timeout=300)
    assert '4 answered earlier (reused), 0 to ask' in r.stdout, r.stdout


def test_rerendered_image_is_asked_again(tmp_path):
    """An answer older than its image (the image was re-rendered) is not reused."""
    codex = tmp_path/'codex'
    codex.write_text(FAKE)
    codex.chmod(codex.stat().st_mode | stat.S_IEXEC)
    img = tmp_path/'imgs'
    img.mkdir()
    from PIL import Image
    Image.new('RGB', (40, 30)).save(img/'cupx_plain.png')
    (img/'summary.json').write_text(json.dumps({'cupx': dict(category='thin_cup', regions=dict(centroid=[[0, 0, 0]]*4),
                                                             views=dict(plain=dict(regions_seen=[0, 2])))}))
    out = tmp_path/'out'
    env = dict(os.environ, MPLCONFIGDIR=str(tmp_path/'mpl'))
    base = [sys.executable, str(ROOT/'risk_mvp/vlm_query.py'), '--images', str(img), '--out', str(out),
            '--codex', str(codex), '--repeats', '1']
    assert subprocess.run(base, capture_output=True, text=True, env=env, timeout=300).returncode == 0
    r = subprocess.run(base, capture_output=True, text=True, env=env, timeout=300)
    assert '1 answered earlier (reused), 0 to ask' in r.stdout, r.stdout
    import time
    time.sleep(1.1)
    Image.new('RGB', (40, 30), (9, 9, 9)).save(img/'cupx_plain.png')       # re-rendered
    r = subprocess.run(base, capture_output=True, text=True, env=env, timeout=300)
    assert '0 answered earlier (reused), 1 to ask' in r.stdout, r.stdout
