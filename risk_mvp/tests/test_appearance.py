"""appearance.py: instances follow the mark, cue assets and the VLM summary share one partition.
Needs the YCB meshes (python risk_mvp/objects.py download); skipped without them."""
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from risk_mvp import appearance as A  # noqa: E402
from risk_mvp import objects as O     # noqa: E402

OID = '065-a_cups'
pytestmark = pytest.mark.skipif(not O.mesh_path(OID).exists(), reason='YCB meshes not downloaded')


def test_conditions_follow_the_mark():
    part, info = A.canonical_partition(OID, None)
    rng = np.random.default_rng(0)
    insts = [A.sample_instance(OID, part, rng) for _ in range(2000)]
    frac = {k: v/len(insts) for k, v in Counter(i.condition for i in insts).items()}
    for k, p in A.DEFAULT_MIX.items():
        assert abs(frac[k]-p) < .05, frac
    lo, hi = O.CATEGORIES[O.CATALOG[OID]['category']]
    assert all(lo <= i.base_N <= hi for i in insts)
    cand = set(A.mark_regions(part))
    for i in insts:
        if i.condition == 'visible':
            assert i.weak_region == i.mark_region and i.weak_region in cand and .3 <= i.coeff[i.weak_region] <= .6
        elif i.condition == 'hidden':
            assert i.mark_region is None and i.weak_region in cand and i.coeff[i.weak_region] <= .6
        elif i.condition == 'decoy':
            assert i.weak_region is None and i.mark_region in cand and i.coeff[i.mark_region] >= 1
        else:
            assert i.weak_region is None and i.mark_region is None
        assert i.variant == ('plain' if i.mark_region is None else f'cue_r{i.mark_region}')
    pinned = A.sample_instance(OID, part, rng, condition='visible', weak_region=2)
    assert pinned.weak_region == pinned.mark_region == 2


def test_assets_manifest_and_vlm_summary(tmp_path):
    r = subprocess.run([sys.executable, str(ROOT/'risk_mvp/appearance.py'), '--objects', OID, '--no-render',
                        '--snapshots', str(tmp_path/'none'), '--out', str(tmp_path/'app')],
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout+r.stderr
    m = json.loads((tmp_path/'app/manifest.json').read_text())[OID]
    part, info = A.canonical_partition(OID, None)
    assert m['partition']['fingerprint'] == info['fingerprint']
    assert set(m['variants']) == {'plain'} | {f'cue_r{r}' for r in A.mark_regions(part)}
    for v, d in m['variants'].items():
        assert Path(d['mesh']).exists()
        assert d['mark_region'] == (None if v == 'plain' else int(v[5:]))
    inst = A.sample_instance(OID, part, np.random.default_rng(1), condition='visible')
    assert inst.mesh_path(json.loads((tmp_path/'app/manifest.json').read_text())) == m['variants'][inst.variant]['mesh']
    s = json.loads((tmp_path/'app/summary.json').read_text())[OID]
    from risk_mvp.vlm_query import mark_region
    assert all(mark_region(s, v) == m['variants'][v]['mark_region'] for v in s['views'])
