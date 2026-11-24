"""Real objects for the MVP: a YCB subset with textures, and their visual-cue variants.

Why: virtual thresholds drawn independently of appearance give the VLM nothing to read. Here the
object identity carries the overall strength (a category prior), and a hidden weak region can be made
visible by painting a decal (bruise on fruit, crease/dent mark on cups and other objects) onto the
texture inside that region ("cue" condition), or left unmarked ("no cue" condition).

Download (5090, needs internet; ~6-10 MB per object):
  python risk_mvp/objects.py download            # -> risk_mvp/assets/ycb/<id>/google_16k/
  python risk_mvp/objects.py summary             # sizes, masses, categories

The category table below is the one hand-set input of the threshold generator: it fixes the range the
object-level strength is drawn from (log-uniform), never a per-region value. Region coefficients and the
weak spot stay random (regions.sample_instance). Edit CATEGORIES/CATALOG to change the object set.
"""
from __future__ import annotations

import json
import sys
import tarfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

import numpy as np
import trimesh

ASSETS = Path(__file__).resolve().parent/'assets'/'ycb'
URL = 'https://ycb-benchmarks.s3.amazonaws.com/data/google/{}_google_16k.tgz'

# Object-level strength range (N) per category, log-uniform within the range.
CATEGORIES = {
    'soft_fruit': (.5, 2.),     # strawberry, plum, peach: bruise easily
    'firm_fruit': (2., 5.),     # lemon
    'thin_cup': (1., 4.),       # thin plastic cups: the wall buckles
    'sturdy': (5., 10.),        # tennis ball, foam brick, can
}
# id -> category, mass (kg, YCB reference masses rounded), cue style for the visible weak spot
CATALOG = {
    '012_strawberry': dict(category='soft_fruit', mass_kg=.018, cue='bruise'),
    '018_plum': dict(category='soft_fruit', mass_kg=.025, cue='bruise'),
    '015_peach': dict(category='soft_fruit', mass_kg=.033, cue='bruise'),
    '014_lemon': dict(category='firm_fruit', mass_kg=.029, cue='bruise'),
    '065-a_cups': dict(category='thin_cup', mass_kg=.013, cue='crease'),
    '065-d_cups': dict(category='thin_cup', mass_kg=.015, cue='crease'),
    '061_foam_brick': dict(category='sturdy', mass_kg=.028, cue='crease'),
    '056_tennis_ball': dict(category='sturdy', mass_kg=.058, cue='crease'),
}


def fetch(url: str, dst: Path, tries: int = 8, chunk: int = 1 << 16):
    """Download with resume (HTTP Range) and retries; checks the final size against Content-Length."""
    import time
    part = dst.with_suffix(dst.suffix+'.part')
    total = None
    for attempt in range(tries):
        have = part.stat().st_size if part.exists() else 0
        req = urllib.request.Request(url, headers={'Range': f'bytes={have}-'} if have else {})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                if r.status == 200 and have:      # server ignored the range: start over
                    have = 0
                    part.unlink()
                length = int(r.headers.get('Content-Length', 0))
                total = have+length if length else total
                with part.open('ab') as f:
                    while True:
                        buf = r.read(chunk)
                        if not buf:
                            break
                        f.write(buf)
            if total is None or part.stat().st_size >= total:
                part.rename(dst)
                return
        except (OSError, urllib.error.URLError) as exc:
            print(f'  retry {attempt+1}/{tries} after {type(exc).__name__}: {exc}', flush=True)
        time.sleep(min(2**attempt, 30))
    raise RuntimeError(f'download failed after {tries} tries: {url}')


def download(ids=None, root: Path = ASSETS):
    root.mkdir(parents=True, exist_ok=True)
    for oid in ids or CATALOG:
        target = root/oid/'google_16k'/'textured.obj'
        if target.exists():
            continue
        tgz = root/f'{oid}.tgz'
        print('downloading', oid, flush=True)
        fetch(URL.format(oid), tgz)
        with tarfile.open(tgz) as t:
            members = [m for m in t.getmembers() if m.name.startswith(f'{oid}/google_16k/')
                       and '..' not in Path(m.name).parts and (m.isfile() or m.isdir())]
            t.extractall(root, members=members)
        tgz.unlink()


def mesh_path(oid: str, root: Path = ASSETS) -> Path:
    return root/oid/'google_16k'/'textured.obj'


def load(oid: str, root: Path = ASSETS) -> dict:
    """Textured mesh and the object frame used everywhere in risk_mvp: origin at the bounding-box centre,
    z up as in the YCB google_16k files. `offset` is the translation from the file frame to that frame
    (place the Genesis morph at pos + offset_world so the object frame lands at pos)."""
    path = mesh_path(oid, root)
    m = trimesh.load(path, force='mesh', process=False)
    center = m.bounds.mean(0)
    m.apply_translation(-center)
    info = dict(CATALOG[oid])
    info.update(id=oid, path=str(path), mesh=m, offset=(-center).tolist(),
                extent_m=(m.bounds[1]-m.bounds[0]).tolist(), strength_range_N=CATEGORIES[info['category']])
    return info


def region_spec(oid: str, root: Path = ASSETS) -> dict:
    """regions.object_mesh spec for this object (geometry only, object frame)."""
    return dict(mesh_path=str(mesh_path(oid, root)))


def paint_cue(oid: str, part, region: int, out_dir: Path, root: Path = ASSETS, style: Optional[str] = None,
              strength: float = .75, seed: int = 0) -> Path:
    """Copy of the textured OBJ whose texture is marked inside `region` (regions.Partition computed on
    regions.object_mesh(region_spec(oid)), i.e. the same object frame). Returns the new OBJ path.

    The mark is drawn in UV space over the faces of the region nearest the region centre (about half the
    region's area), so it is visible on the rendered surface and stays inside the region."""
    from PIL import Image, ImageDraw, ImageFilter
    style = style or CATALOG[oid]['cue']
    info = load(oid, root)
    m = info['mesh']
    uv = np.asarray(m.visual.uv)
    tex = m.visual.material.image.convert('RGB')
    W, H = tex.size
    fc = m.triangles_center
    lab = part.region_of(fc)
    sel = np.flatnonzero(lab == region)
    d = np.linalg.norm(fc[sel]-part.centroid[region], axis=1)
    sel = sel[d <= np.quantile(d, .5)]
    mask = Image.new('L', (W, H), 0)
    draw = ImageDraw.Draw(mask)
    for f in sel:
        pts = [(float(u*W), float((1-v)*H)) for u, v in uv[m.faces[f]]]
        draw.polygon(pts, fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(2))
    rng = np.random.default_rng(seed)
    arr = np.asarray(tex).astype(np.float32)
    a = (np.asarray(mask).astype(np.float32)/255.)[..., None]*strength
    if style == 'bruise':
        colour = np.array([70., 45., 25.])+rng.normal(0, 5, 3)
        noise = rng.normal(1., .08, arr.shape[:2])[..., None]
        arr = arr*(1-a)+colour*noise*a
    else:   # crease: darken with streaks
        streak = (np.sin(np.arange(W)[None, :]/3.+rng.uniform(0, 6))*.5+.5)[..., None]
        arr = arr*(1-a*(.55+.35*streak))
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f'{oid}_cue_r{region}'
    Image.fromarray(arr.clip(0, 255).astype(np.uint8)).save(out_dir/f'{stem}.png')
    (out_dir/f'{stem}.mtl').write_text(f'newmtl material_0\nmap_Kd {stem}.png\n')
    src = Path(info['path']).read_text().splitlines()
    lines = [f'mtllib {stem}.mtl' if l.startswith('mtllib') else l for l in src]
    (out_dir/f'{stem}.obj').write_text('\n'.join(lines)+'\n')
    return out_dir/f'{stem}.obj'


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'summary'
    if cmd == 'download':
        download()
    for oid in CATALOG:
        if mesh_path(oid).exists():
            i = load(oid)
            print(oid.ljust(18), i['category'].ljust(11), 'extent mm', np.round(np.array(i['extent_m'])*1000),
                  'mass g', i['mass_kg']*1000, 'strength N', i['strength_range_N'])
        else:
            print(oid, 'not downloaded')
