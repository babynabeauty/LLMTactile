"""Appearance-linked object instances: the hidden thresholds follow what the object looks like.

Before this, the thresholds were drawn independently of the rendered object (any strength 0.5-10 N for
any object, the weak spot in a random region, the painted cue only on the check images), so a VLM that
reads the picture correctly still carried no information about the thresholds.

Here, per object:
  partition   ONE canonical partition, the one the grasp snapshot uses (snapshot.py: k, seed, index and
              middle contacts in different regions). Cue painting, the region numbers shown to the VLM, the
              thresholds and the episode all use it, so "region 3" means the same patch everywhere.
  cue assets  one textured mesh per contactable region with a painted mark there (objects.paint_cue);
              the physics geometry is unchanged, only the texture differs.
  instance    base strength log-uniform in the object's category range (objects.CATEGORIES), per-region
              coefficients log-normal around 1 (small spread, spatially correlated), and one of:
                visible  a weak region (coefficient 0.3-0.6) carrying the painted mark
                hidden   a weak region without any mark (the VLM cannot see it)
                decoy    a mark on a region that is not weak (the VLM is misled)
                plain    no mark, no weak region
              The instance says which mesh to load (mark or plain).

  python risk_mvp/appearance.py --snapshots outputs/risk_mvp/snapshots --out outputs/risk_mvp/appearance
writes, per object, the cue meshes, a 4-view region-numbered sheet per variant (plain, cue_r<r>) for the
VLM, manifest.json (partition, variants, mesh paths, which regions each view shows) and sheet.png (all
variants of all objects, small). Region numbers are drawn plainly: the earlier check images ringed the cue
region's number in red, which the VLM could have used instead of the mark itself.
Then: python risk_mvp/vlm_query.py --images outputs/risk_mvp/appearance --out outputs/risk_mvp/vlm_cues
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from risk_mvp import objects as O                                       # noqa: E402
from risk_mvp.regions import object_mesh, partition                     # noqa: E402

CONDITIONS = ('visible', 'hidden', 'decoy', 'plain')
DEFAULT_MIX = dict(visible=.4, hidden=.2, decoy=.2, plain=.2)


# ---------------------------------------------------------------- partition and assets
def canonical_partition(oid: str, snapshots: Optional[Path] = None):
    """(Partition, info) of the object: the snapshot's if there is one, else seed 0 without constraints."""
    hull = object_mesh(O.region_spec(oid))
    snap_path = None if snapshots is None else Path(snapshots)/f'{oid}.json'
    if snap_path is not None and snap_path.exists():
        snap = json.loads(snap_path.read_text())
        part = partition(hull, snap['k_regions'], seed=snap['partition_seed'], separate=snap['partition_separate'])
        info = dict(source='snapshot', k=snap['k_regions'], seed=snap['partition_seed'],
                    separate=snap['partition_separate'],
                    contacts={f: snap['contacts'][f]['region'] for f in ('thumb', 'index', 'middle')})
    else:
        part = partition(hull, 5, seed=0)
        info = dict(source='default (no snapshot)', k=5, seed=0, separate=None, contacts=None)
    info['fingerprint'] = fingerprint(part)
    return part, info


def fingerprint(part) -> str:
    """Short id of a partition (region centres rounded to 0.1 mm), to catch mixing two partitions."""
    import hashlib
    return hashlib.sha1(np.round(np.asarray(part.centers)*1e4).astype(np.int64).tobytes()).hexdigest()[:10]


def mark_regions(part) -> list:
    """Regions that can carry a mark / weak spot: the ones a side grasp can touch."""
    r = np.flatnonzero(part.contactable())
    return [int(x) for x in (r if len(r) else np.arange(part.k))]


def make_cue_assets(oid: str, part, out_dir: Path, regions: Optional[Sequence[int]] = None):
    """({region: obj path}, {region: mark contrast}) with a painted mark on each region (regenerated every
    call, cheap)."""
    paths, contrast = {}, {}
    for r in (mark_regions(part) if regions is None else regions):
        paths[int(r)] = str(O.paint_cue(oid, part, int(r), Path(out_dir)))
        contrast[int(r)] = O.paint_cue.last_contrast
    return paths, contrast


# ---------------------------------------------------------------- instances
@dataclass
class AppearanceInstance:
    oid: str
    condition: str
    base_N: float
    coeff: np.ndarray
    weak_region: Optional[int]
    mark_region: Optional[int]
    partition_fingerprint: str
    extra: dict = field(default_factory=dict)

    @property
    def thresholds(self) -> np.ndarray:
        return self.base_N*self.coeff

    @property
    def variant(self) -> str:
        """Which rendered variant the robot (and the VLM) sees: 'plain' or 'cue_r<r>'."""
        return 'plain' if self.mark_region is None else f'cue_r{self.mark_region}'

    def mesh_path(self, manifest: dict) -> str:
        m = manifest[self.oid]
        if m['partition']['fingerprint'] != self.partition_fingerprint:
            raise ValueError(f'{self.oid}: instance and manifest use different partitions')
        return m['variants'][self.variant]['mesh']

    def to_json(self) -> dict:
        return dict(oid=self.oid, condition=self.condition, base_N=self.base_N, coeff=self.coeff.tolist(),
                    thresholds_N=self.thresholds.tolist(), weak_region=self.weak_region,
                    mark_region=self.mark_region, variant=self.variant,
                    partition_fingerprint=self.partition_fingerprint, **self.extra)


def sample_instance(oid: str, part, rng: np.random.Generator, condition: Optional[str] = None,
                    mix: dict = DEFAULT_MIX, weak_region: Optional[int] = None, mark_region: Optional[int] = None,
                    weak_factor=(.3, .6), sigma=.15, base_N: Optional[float] = None) -> AppearanceInstance:
    """One instance. condition None -> drawn from `mix`. weak_region / mark_region pin the region (the swap
    experiment puts the weak spot under the index or the middle finger); otherwise drawn from the
    contactable regions. For 'decoy' the mark goes on a region other than any weak one (there is none)."""
    if condition is None:
        names = list(mix)
        p = np.array([mix[n] for n in names], float)
        condition = names[rng.choice(len(names), p=p/p.sum())]
    if condition not in CONDITIONS:
        raise ValueError(f'condition {condition!r}')
    lo, hi = O.CATEGORIES[O.CATALOG[oid]['category']]
    base = float(np.exp(rng.uniform(np.log(lo), np.log(hi)))) if base_N is None else float(base_N)
    # spatially correlated log-normal spread around 1 (regions.sample_instance's kernel), no weak spot yet
    c = part.centers
    ell = .5*float(np.ptp(c, 0).max() or .01)
    cov = sigma**2*np.exp(-.5*((c[:, None]-c[None])**2).sum(-1)/ell**2)+1e-9*np.eye(part.k)
    coeff = np.exp(rng.multivariate_normal(np.zeros(part.k), cov))
    cand = mark_regions(part)
    weak = mark = None
    if condition in ('visible', 'hidden'):
        weak = int(rng.choice(cand)) if weak_region is None else int(weak_region)
        coeff[weak] = rng.uniform(*weak_factor)
        mark = weak if condition == 'visible' else None
    elif condition == 'decoy':
        mark = int(rng.choice(cand)) if mark_region is None else int(mark_region)
        coeff[mark] = max(coeff[mark], 1.)          # a mark that is not a weak spot
    return AppearanceInstance(oid, condition, base, coeff, weak, mark, fingerprint(part))


# ---------------------------------------------------------------- rendering for the VLM
def render_variants(items: list, out: Path, backend='gpu', res=(640, 480), fov=30.):
    """items: [(oid, part, {variant: mesh path})]. 4 views per variant, region numbers drawn where the
    region is visible (no highlight). Returns {oid: {variant: dict(image, regions_seen)}}."""
    import genesis as gs
    from risk_mvp import render as RD
    from risk_mvp.objects_check import quat_mat
    if not gs._initialized:
        gs.init(backend=getattr(gs, backend), precision='32', logging_level='warning')
    views = {}
    for oid, part, variants in items:
        info = O.load(oid)
        hull = object_mesh(O.region_spec(oid))
        rho = info['mass_kg']/info['mesh'].convex_hull.volume
        scene = gs.Scene(sim_options=gs.options.SimOptions(dt=1/300), show_viewer=False,
                         vis_options=gs.options.VisOptions(background_color=(.9, .9, .9), ambient_light=(.5, .5, .5)))
        scene.add_entity(gs.morphs.Plane(), surface=gs.surfaces.Default(color=(.55, .55, .58, 1)))
        ents = []
        for j, (v, path) in enumerate(variants.items()):
            center = np.array([j*.6, 0., info['extent_m'][2]/2+.003])
            ent = scene.add_entity(gs.morphs.Mesh(file=path, pos=tuple(center+np.asarray(info['offset'])), convexify=True),
                                   material=gs.materials.Rigid(rho=rho, friction=.8))
            cams = [(scene.add_camera(res=res, pos=tuple(p), lookat=tuple(center), fov=fov, GUI=False), p)
                    for p in RD.orbit(center, dist=RD.fit_distance(info['extent_m'], fov))]
            ents.append((v, ent, cams, center))
        scene.build()
        for _ in range(300):
            scene.step()
        views[oid] = {}
        for v, ent, cams, center in ents:
            Rq = quat_mat(np.asarray(ent.get_quat().cpu()).reshape(4))
            p_obj = np.asarray(ent.get_pos().cpu()).reshape(3)-Rq@np.asarray(info['offset'])
            ims, seen = [], set()
            for cam, cam_pos in cams:
                rgb = cam.render(rgb=True)[0]
                rgb = np.asarray(rgb.cpu() if hasattr(rgb, 'cpu') else rgb)
                labels = RD.visible_labels(hull, part, Rq, p_obj, cam_pos)
                seen |= set(labels)
                ims.append(RD.draw_labels(rgb, labels, cam_pos, RD.look_at(cam_pos, center), res, fov))
            name = f'{oid}_{v}.png'
            RD.montage(ims).save(out/name)
            views[oid][v] = dict(image=name, regions_seen=sorted(int(s) for s in seen),
                                 tilt_deg=float(np.degrees(np.arccos(np.clip(Rq[2, 2], -1, 1)))))
            print(f'  {oid} {v}: regions seen {sorted(seen)}', flush=True)
    return views


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--snapshots', type=Path, default=ROOT/'outputs/risk_mvp/snapshots')
    ap.add_argument('--out', type=Path, default=ROOT/'outputs/risk_mvp/appearance')
    ap.add_argument('--objects', nargs='*', default=None)
    ap.add_argument('--backend', default='gpu')
    ap.add_argument('--no-render', action='store_true', help='only partitions and cue meshes')
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    O.download()
    manifest, items = {}, []
    for oid in a.objects or list(O.CATALOG):
        part, pinfo = canonical_partition(oid, a.snapshots)
        cues, cue_contrast = make_cue_assets(oid, part, a.out/'meshes'/oid)
        variants = {'plain': O.mesh_path(oid).as_posix()}
        variants.update({f'cue_r{r}': path for r, path in cues.items()})
        contrast = {f'cue_r{r}': c for r, c in cue_contrast.items()}
        manifest[oid] = dict(category=O.CATALOG[oid]['category'], strength_range_N=O.CATEGORIES[O.CATALOG[oid]['category']],
                             partition=pinfo, regions=part.to_json(), mark_regions=mark_regions(part),
                             variants={v: dict(mesh=str(p), mark_region=None if v == 'plain' else int(v[5:]),
                                               contrast=contrast.get(v)) for v, p in variants.items()})
        items.append((oid, part, variants))
        print(oid, 'partition', pinfo['source'], 'seed', pinfo['seed'], 'mark regions', mark_regions(part),
              'contacts', pinfo['contacts'], flush=True)
    if not a.no_render:
        views = render_variants(items, a.out, backend=a.backend)
        for oid, vs in views.items():
            for v, d in vs.items():
                manifest[oid]['variants'][v].update(d)
        from risk_mvp import render as RD
        from PIL import Image
        tiles = [Image.open(a.out/d['image']) for oid in views for d in views[oid].values()]
        if tiles:
            RD.montage(tiles, cols=max(len(v) for v in views.values()), size=(320, 240)).save(a.out/'sheet.png')
    # merge with objects rendered earlier into the same --out (render only new objects with --objects)
    old = a.out/'manifest.json'
    if old.exists():
        prev = json.loads(old.read_text())
        prev.update(manifest)
        manifest = prev
    (a.out/'manifest.json').write_text(json.dumps(manifest, indent=1)+'\n')
    # summary.json in the shape vlm_query reads (per object: category, regions, views with regions_seen)
    summary = {oid: dict(category=m['category'], regions=m['regions'],
                         views={v: dict(regions_seen=d.get('regions_seen', list(range(len(m['regions']['centers'])))),
                                        cue_region=d['mark_region'])
                                for v, d in m['variants'].items()})
               for oid, m in manifest.items()}
    (a.out/'summary.json').write_text(json.dumps(summary, indent=1)+'\n')
    print('WROTE', a.out)


if __name__ == '__main__':
    main()
