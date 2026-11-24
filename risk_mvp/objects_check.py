"""5090 check of the YCB object set: load, rest on a table, render 4 views with region numbers.

  python risk_mvp/objects.py download
  python risk_mvp/objects_check.py --out outputs/risk_mvp/objects

For every catalogue object: partition the exterior into K regions (regions.py), pick the cue region as
the most side-facing contactable region (a side grasp can reach it), load the plain and the cue-painted
textured mesh in Genesis (convexified collision), let them settle on a plane for 2 s, report drift and
tilt, and render 4 views around each (azimuth 0/90/180/270, 35 deg elevation). Region numbers are drawn
where the region is actually visible (render.visible_labels); the cue region's label has a red ring.
Send back summary.json and sheet_plain.png / sheet_cue.png.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from risk_mvp import objects as O  # noqa: E402
from risk_mvp import render as RD  # noqa: E402
from risk_mvp.regions import object_mesh, partition  # noqa: E402


def quat_mat(q):
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)], [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                     [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, default=ROOT/'outputs/risk_mvp/objects')
    ap.add_argument('--k', type=int, default=5)
    ap.add_argument('--backend', default='gpu')
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    O.download()
    import genesis as gs
    gs.init(backend=getattr(gs, a.backend), precision='32', logging_level='warning')
    scene = gs.Scene(sim_options=gs.options.SimOptions(dt=1/300), show_viewer=False,
                     vis_options=gs.options.VisOptions(background_color=(.9, .9, .9), ambient_light=(.5, .5, .5)))
    scene.add_entity(gs.morphs.Plane(), surface=gs.surfaces.Default(color=(.55, .55, .58, 1)))
    res, fov = (640, 480), 40.
    items = []
    for i, oid in enumerate(O.CATALOG):
        info = O.load(oid)
        hull = object_mesh(O.region_spec(oid))
        part = partition(hull, a.k, seed=0)
        side = np.flatnonzero(part.contactable())
        cue_region = int(side[np.argmin(np.abs(part.normal[side, 2]))]) if len(side) else 0
        cue_obj = O.paint_cue(oid, part, cue_region, a.out/'cue_assets')
        rho = info['mass_kg']/info['mesh'].convex_hull.volume
        variants = []
        for j, path in enumerate([info['path'], str(cue_obj)]):
            center = np.array([i*.6, j*.6, info['extent_m'][2]/2+.003])
            ent = scene.add_entity(gs.morphs.Mesh(file=path, pos=tuple(center+np.asarray(info['offset'])), convexify=True),
                                   material=gs.materials.Rigid(rho=rho, friction=.8))
            cams = [(scene.add_camera(res=res, pos=tuple(p), lookat=tuple(center), fov=fov, GUI=False), p)
                    for p in RD.orbit(center)]
            variants.append(dict(entity=ent, cams=cams, center0=center))
        items.append(dict(oid=oid, info=info, part=part, hull=hull, cue_region=cue_region, variants=variants))
    scene.build()
    for _ in range(600):
        scene.step()
    summary, sheets = {}, {'plain': [], 'cue': []}
    for it in items:
        oid, part = it['oid'], it['part']
        rec = dict(category=it['info']['category'], mass_kg=it['info']['mass_kg'],
                   extent_mm=np.round(np.array(it['info']['extent_m'])*1000).tolist(),
                   strength_range_N=it['info']['strength_range_N'], cue_region=it['cue_region'],
                   contactable=np.flatnonzero(part.contactable()).tolist(), regions=part.to_json(), views={})
        for name, v in zip(('plain', 'cue'), it['variants']):
            ent = v['entity']
            Rq = quat_mat(np.asarray(ent.get_quat().cpu()).reshape(4))
            p_obj = np.asarray(ent.get_pos().cpu()).reshape(3)-Rq@np.asarray(it['info']['offset'])
            ims, seen = [], set()
            for cam, cam_pos in v['cams']:
                rgb = cam.render(rgb=True)[0]
                rgb = np.asarray(rgb.cpu() if hasattr(rgb, 'cpu') else rgb)
                labels = RD.visible_labels(it['hull'], part, Rq, p_obj, cam_pos)
                seen |= set(labels)
                ims.append(RD.draw_labels(rgb, labels, cam_pos, RD.look_at(cam_pos, v['center0']), res, fov,
                                          highlight=(it['cue_region'],) if name == 'cue' else ()))
            sheet = RD.montage(ims)
            sheet.save(a.out/f'{oid}_{name}.png')
            sheets[name].append(sheet)
            rec['views'][name] = dict(image=f'{oid}_{name}.png', regions_seen=sorted(seen),
                                      settled_shift_mm=float(np.linalg.norm(p_obj[:2]-v['center0'][:2])*1000),
                                      tilt_deg=float(np.degrees(np.arccos(np.clip(Rq[2, 2], -1, 1)))))
        summary[oid] = rec
        print(oid, 'cue region', it['cue_region'], json.dumps(rec['views']), flush=True)
    (a.out/'summary.json').write_text(json.dumps(summary, indent=1)+'\n')
    for name, ss in sheets.items():
        RD.montage(ss, cols=4, size=(400, 300)).save(a.out/f'sheet_{name}.png')
    print('WROTE', a.out)


if __name__ == '__main__':
    main()
