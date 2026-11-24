"""5090 check of the YCB object set: load, rest on a table, render with region numbers (plain and cue).

  python risk_mvp/objects.py download
  python risk_mvp/objects_check.py --out outputs/risk_mvp/objects

For every catalogue object: partition the exterior into K regions (regions.py), load the textured mesh
in Genesis (convexified collision), drop it on a plane, let it settle for 2 s and report how far it
moved/rolled, then render two images: plain texture and the "cue" variant whose region 2 carries the
painted weak-spot mark, both with region numbers projected onto the picture (the image the VLM will be
asked about). Send back summary.json and the contact sheet sheet.png.
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
from risk_mvp.regions import object_mesh, partition  # noqa: E402


def look_at(pos, lookat, up=(0, 0, 1)):
    f = np.asarray(lookat, float)-pos
    f /= np.linalg.norm(f)
    r = np.cross(f, up)
    r /= np.linalg.norm(r)
    d = np.cross(f, r)
    return np.stack([r, d, f])     # rows: OpenCV camera axes x right, y down, z forward


def project(points, cam_pos, R, res, fov_deg):
    W, H = res
    f = .5*H/np.tan(np.deg2rad(fov_deg)/2)
    pc = (np.asarray(points)-cam_pos)@R.T
    return np.stack([f*pc[:, 0]/pc[:, 2]+W/2, f*pc[:, 1]/pc[:, 2]+H/2], 1), pc[:, 2]


def quat_mat(q):
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)], [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                     [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])


def annotate(rgb, part, R_obj, p_obj, cam_pos, R_cam, res, fov, weak=None):
    from PIL import Image, ImageDraw, ImageFont
    im = Image.fromarray(rgb[..., :3].astype(np.uint8))
    dr = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype('DejaVuSans-Bold.ttf', 22)
    except OSError:
        font = ImageFont.load_default()
    cw = part.centroid@R_obj.T+p_obj
    nw = part.normal@R_obj.T
    uv, depth = project(cw, cam_pos, R_cam, res, fov)
    shown = []
    for r in range(part.k):
        facing = nw[r]@(cam_pos-cw[r]) > 0
        if depth[r] <= 0 or not facing:
            continue
        u, v = uv[r]
        txt = str(r)
        dr.ellipse([u-14, v-14, u+14, v+14], fill=(255, 255, 255), outline=(0, 0, 0), width=2)
        dr.text((u-7, v-13), txt, fill=(0, 0, 0), font=font)
        shown.append(r)
    return im, shown


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
        part = partition(object_mesh(O.region_spec(oid)), a.k, seed=0)
        cue_obj = O.paint_cue(oid, part, 2, a.out/'cue_assets')
        hull = info['mesh'].convex_hull
        rho = info['mass_kg']/hull.volume
        row = []
        for j, path in enumerate([info['path'], str(cue_obj)]):
            center = np.array([i*.4, j*.4, info['extent_m'][2]/2+.003])
            ent = scene.add_entity(gs.morphs.Mesh(file=path, pos=tuple(center+np.asarray(info['offset'])), convexify=True),
                                   material=gs.materials.Rigid(rho=rho, friction=.8))
            cam_pos = center+np.array([.16, -.20, .16])
            cam = scene.add_camera(res=res, pos=tuple(cam_pos), lookat=tuple(center), fov=fov, GUI=False)
            row.append(dict(entity=ent, cam=cam, cam_pos=cam_pos, center0=center, path=path))
        items.append(dict(oid=oid, info=info, part=part, row=row, rho=rho))
    scene.build()
    for _ in range(600):
        scene.step()
    summary = {}
    tiles = []
    for it in items:
        oid, part = it['oid'], it['part']
        rec = dict(category=it['info']['category'], mass_kg=it['info']['mass_kg'], extent_mm=np.round(np.array(it['info']['extent_m'])*1000).tolist(),
                   strength_range_N=it['info']['strength_range_N'], regions=part.to_json(), views={})
        for name, r in zip(('plain', 'cue_region2'), it['row']):
            ent = r['entity']
            # Genesis reports the pose of the file frame; the object frame sits at file frame - offset
            q = np.asarray(ent.get_quat().cpu()).reshape(4)
            Rq = quat_mat(q)
            p_file = np.asarray(ent.get_pos().cpu()).reshape(3)
            p_obj = p_file-Rq@np.asarray(it['info']['offset'])
            tilt = float(np.degrees(np.arccos(np.clip(Rq[2, 2], -1, 1))))
            rgb = r['cam'].render(rgb=True)[0]
            rgb = np.asarray(rgb.cpu() if hasattr(rgb, 'cpu') else rgb)
            im, shown = annotate(rgb, part, Rq, p_obj, r['cam_pos'], look_at(r['cam_pos'], r['center0']), res, fov)
            f = a.out/f'{oid}_{name}.png'
            im.save(f)
            tiles.append(im)
            rec['views'][name] = dict(image=f.name, settled_shift_mm=float(np.linalg.norm(p_obj[:2]-r['center0'][:2])*1000),
                                      tilt_deg=tilt, regions_visible=shown)
        summary[oid] = rec
        print(oid, json.dumps({k: v for k, v in rec['views'].items()}), flush=True)
    (a.out/'summary.json').write_text(json.dumps(summary, indent=1)+'\n')
    from PIL import Image
    cols = 4
    th = [t.resize((320, 240)) for t in tiles]
    sheet = Image.new('RGB', (320*cols, 240*int(np.ceil(len(th)/cols))), (255, 255, 255))
    for k, t in enumerate(th):
        sheet.paste(t, (320*(k % cols), 240*(k//cols)))
    sheet.save(a.out/'sheet.png')
    print('WROTE', a.out/'summary.json', a.out/'sheet.png')


if __name__ == '__main__':
    main()
