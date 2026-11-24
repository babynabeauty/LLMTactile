"""Region-numbered multi-view images (objects_check.py now, vlm_regions.py later). numpy + PIL only.

Cameras follow the OpenCV convention Genesis uses for its extrinsics (x right, y down, z forward),
fov is the vertical field of view. A region number is drawn where that region is actually visible:
at the area-weighted centre of its exterior (convex hull) faces that face the camera, and only if at
least `min_visible` of the region's area does. On a convex surface front faces are not self-occluded.
"""
from __future__ import annotations

import numpy as np

AZIMUTHS = (0., 90., 180., 270.)


def fit_distance(extent_m, fov_deg=30., fill=.55):
    """Camera distance at which the object's largest extent spans about `fill` of the image height."""
    return float(max(extent_m))/fill/(2*np.tan(np.deg2rad(fov_deg)/2))


def orbit(center, dist=.26, elev_deg=35., azimuths=AZIMUTHS):
    """Camera positions around `center` (world), azimuth 0 = +x side."""
    out = []
    for az in azimuths:
        a, e = np.deg2rad(az), np.deg2rad(elev_deg)
        out.append(np.asarray(center, float)+dist*np.array([np.cos(e)*np.cos(a), np.cos(e)*np.sin(a), np.sin(e)]))
    return out


def look_at(pos, lookat, up=(0, 0, 1)):
    f = np.asarray(lookat, float)-pos
    f /= np.linalg.norm(f)
    r = np.cross(f, up)
    r /= np.linalg.norm(r)
    return np.stack([r, np.cross(f, r), f])


def project(points, cam_pos, R_cam, res, fov_deg):
    W, H = res
    f = .5*H/np.tan(np.deg2rad(fov_deg)/2)
    pc = (np.atleast_2d(points)-cam_pos)@R_cam.T
    return np.stack([f*pc[:, 0]/pc[:, 2]+W/2, f*pc[:, 1]/pc[:, 2]+H/2], 1), pc[:, 2]


def visible_labels(hull, part, R_obj, p_obj, cam_pos, min_visible=.15, min_cos=.2):
    """{region: world point to label} for regions visible from cam_pos. hull: object-frame trimesh."""
    c = hull.triangles_center@R_obj.T+p_obj
    n = hull.face_normals@R_obj.T
    v = cam_pos-c
    facing = (n*v).sum(1)/np.linalg.norm(v, axis=1) > min_cos
    lab = part.region_of(hull.triangles_center)
    out = {}
    for r in range(part.k):
        tot = hull.area_faces[lab == r].sum()
        m = facing & (lab == r)
        if tot <= 0 or hull.area_faces[m].sum() < min_visible*tot:
            continue
        w = hull.area_faces[m]
        out[r] = (c[m]*w[:, None]).sum(0)/w.sum()
    return out


def draw_labels(rgb, labels, cam_pos, R_cam, res, fov, highlight=()):
    from PIL import Image, ImageDraw, ImageFont
    im = Image.fromarray(np.asarray(rgb)[..., :3].astype(np.uint8))
    dr = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype('DejaVuSans-Bold.ttf', 20)
    except OSError:
        font = ImageFont.load_default()
    for r, p in labels.items():
        (u, v), = project(p, cam_pos, R_cam, res, fov)[0]
        dr.ellipse([u-13, v-13, u+13, v+13], fill=(255, 255, 255), outline=(200, 0, 0) if r in highlight else (0, 0, 0),
                   width=2)
        dr.text((u-6, v-12), str(r), fill=(0, 0, 0), font=font)
    return im


def montage(images, cols=2, size=(400, 300)):
    from PIL import Image
    ims = [i.resize(size) for i in images]
    rows = int(np.ceil(len(ims)/cols))
    sheet = Image.new('RGB', (size[0]*cols, size[1]*rows), (255, 255, 255))
    for k, im in enumerate(ims):
        sheet.paste(im, (size[0]*(k % cols), size[1]*(k//cols)))
    return sheet
