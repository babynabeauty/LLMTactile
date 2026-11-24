"""Plots shared by the 5090 scripts (matplotlib Agg; every function writes one PNG)."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

os.environ.setdefault('MPLCONFIGDIR', '/tmp/risk_mvp_mpl')


def _plt():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


def grid_heatmap(rows, path: Path, title: str):
    """Pregrasp grid: one panel per dy, x = dx, y = dz; colour = outcome, text = drift / failure reason."""
    plt = _plt()
    off = np.array([r['offset_m'] for r in rows])
    dxs, dys, dzs = (np.unique(off[:, i]) for i in range(3))
    fig, axs = plt.subplots(1, len(dys), figsize=(4.2*len(dys), 3.6), squeeze=False)
    for k, dy in enumerate(dys):
        ax = axs[0, k]
        img = np.full((len(dzs), len(dxs)), np.nan)
        for r, o in zip(rows, off):
            if o[1] != dy:
                continue
            i, j = np.searchsorted(dzs, o[2]), np.searchsorted(dxs, o[0])
            if r['success']:
                img[i, j] = 2
                txt = f"ok\n{r['drift_mm']:.1f}mm"
            else:
                img[i, j] = 0 if r['pre_touch'] else 1
                why = ('touch' if r['pre_touch'] else 'finger' if min(r['hold_contact_frac'].values()) <= .95
                       else 'lift' if r['lift_m'] <= .035 else 'slip')
                txt = why
            ax.text(j, i, txt, ha='center', va='center', fontsize=7)
        ax.imshow(img, origin='lower', cmap='RdYlGn', vmin=0, vmax=2)
        ax.set_xticks(range(len(dxs)), [f'{v*1000:+.1f}' for v in dxs], fontsize=7)
        ax.set_yticks(range(len(dzs)), [f'{v*1000:+.0f}' for v in dzs], fontsize=7)
        ax.set(xlabel='dx mm', ylabel='dz mm', title=f'dy {dy*1000:+.0f} mm')
    fig.suptitle(f'{title}  (green: lifted and held; red: touched before closing; yellow: other failure)')
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def contacts_on_regions(part, hull, rows, path: Path, title: str):
    """The two sides of the grasp axis (x): fingers side (x < 0) and thumb side (x > 0), each seen face-on
    as (y, z), surface coloured by region, with the hold contact points of the successful envs."""
    plt = _plt()
    pts = hull.sample(6000)
    lab = part.region_of(pts)
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.8))
    marks = dict(thumb=('k', '^'), index=('b', 'o'), middle=('m', 's'))
    for ax, (side, name, fingers) in zip(axs, [(-1, 'index/middle side (x < 0), seen from -x', ('index', 'middle')),
                                               (1, 'thumb side (x > 0), seen from +x', ('thumb',))]):
        m = side*pts[:, 0] > 0
        y = pts[m, 1]*1000*(-side)          # mirror so both views read as seen from outside
        ax.scatter(y, pts[m, 2]*1000, c=lab[m], cmap='tab10', vmin=0, vmax=9, s=6, alpha=.5)
        for r in range(part.k):
            if side*part.centroid[r, 0] > 0:
                ax.text(part.centroid[r, 1]*1000*(-side), part.centroid[r, 2]*1000, str(r), fontsize=14,
                        weight='bold', ha='center', va='center')
        for f in fingers:
            col, mk = marks[f]
            p = np.array([r['contact_obj_m'][f] for r in rows if r['success'] and f in r['contact_obj_m']])
            if len(p):
                ax.scatter(p[:, 1]*1000*(-side), p[:, 2]*1000, c=col, marker=mk, s=40, label=f, edgecolors='w')
        ax.set(aspect='equal', title=name, xlabel='horizontal mm', ylabel='height mm')
        ax.legend(loc='upper right', fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def force_trace(times, forces, phases, path: Path, title: str, targets=None):
    """forces: {finger: array over ticks}; phases: [(t_start, name)]."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(10, 3.4))
    for f, col in zip(forces, ('C0', 'C1', 'C2', 'C3')):
        ax.plot(times, forces[f], col, label=f)
        if targets and f in targets:
            ax.axhline(targets[f], color=col, ls='--', lw=.8)
    for t, name in phases:
        ax.axvline(t, color='gray', lw=.6)
        ax.text(t, ax.get_ylim()[1]*.95, ' '+name, fontsize=8, va='top', color='gray')
    ax.set(xlabel='s', ylabel='normal force N', title=title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def keyframes(frames, labels, path: Path, cols=None, size=(320, 240)):
    """frames: list of HxWx3 uint8; labels: caption per frame."""
    from PIL import Image, ImageDraw
    cols = cols or len(frames)
    rows = int(np.ceil(len(frames)/cols))
    sheet = Image.new('RGB', (size[0]*cols, size[1]*rows), (255, 255, 255))
    for k, (f, lab) in enumerate(zip(frames, labels)):
        im = Image.fromarray(f).resize(size)
        ImageDraw.Draw(im).text((6, 6), lab, fill=(0, 0, 0))
        sheet.paste(im, (size[0]*(k % cols), size[1]*(k//cols)))
    sheet.save(path)
    return sheet
