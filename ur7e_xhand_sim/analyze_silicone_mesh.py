"""Plot saved physical boundary deformation after removing fitted rigid motion."""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def aligned_displacement(rest, positions):
    x = rest - rest.mean(0)
    y = positions - positions.mean(0)
    u, _, vt = np.linalg.svd(x.T @ y)
    correction = np.eye(3)
    correction[-1, -1] = np.linalg.det(u @ vt)
    rotation = u @ correction @ vt
    fitted = y @ rotation.T
    return fitted, np.linalg.norm(fitted - x, axis=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    mesh = np.load(args.run / 'mesh.npz')
    rest, tets, boundary = (mesh[k] for k in ('rest', 'tets', 'boundary'))
    paths = sorted(args.run.glob('**/fem_positions.npz'))
    if not paths:
        raise ValueError('No saved stage mesh positions')
    x = rest[tets]
    inv_rest = np.linalg.inv(np.stack([x[:,i]-x[:,3] for i in range(3)], axis=-1))
    stages = json.loads((args.run/'stages.json').read_text())
    names = {Path(s['image_dir']).name:s['name'] for s in stages}
    snapshots = []
    for path in paths:
        positions = np.load(path)['positions']
        aligned, displacement = aligned_displacement(rest, positions)
        x = positions[tets]
        J = np.linalg.det(np.stack([x[:,i]-x[:,3] for i in range(3)], axis=-1) @ inv_rest)
        snapshots.append((names.get(path.parent.name,path.parent.name), aligned, displacement, J))
    report = [{'stage':name, 'rigid_fit_residual_max_mm':float(d.max()*1000),
               'rigid_fit_residual_rms_mm':float(np.sqrt(np.mean(d*d))*1000),
               'minimum_J':float(J.min()), 'inverted_tet_count':int((J<0).sum())}
              for name, _, d, J in snapshots]
    (args.run / 'mesh_deformation_analysis.json').write_text(json.dumps({
        'definition':'Displacement from rest after a best-fit proper rotation and translation; not force or whole-block compression.',
        'stages':report}, indent=2))
    n = len(snapshots)
    fig = plt.figure(figsize=(4*min(n,4), 4*((n+3)//4)))
    norm = Normalize(0, max(max(s[2].max()*1000 for s in snapshots), .1))
    axes = []
    for i,(name, aligned, displacement, J) in enumerate(snapshots):
        ax = fig.add_subplot((n+3)//4, min(n,4), i+1, projection='3d'); axes.append(ax)
        colors = plt.cm.viridis(norm(displacement[boundary].mean(1)*1000))
        ax.add_collection3d(Poly3DCollection(aligned[boundary]*1000, facecolors=colors,
                                           edgecolors='#444444', linewidths=.15))
        limit = max(26., float(np.abs(aligned).max()*1100))
        ax.set(xlim=(-limit,limit), ylim=(-limit,limit), zlim=(-limit,limit),
               title=f'{name}\nmin J={J.min():.3f}; inverted={int((J<0).sum())}',
               xlabel='X (mm)', ylabel='Y (mm)', zlabel='Z (mm)')
        ax.set_box_aspect((1,1,1)); ax.view_init(elev=20, azim=-110)
    fig.suptitle('Actual FEM surface; rigid motion removed; deformation not magnified')
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='viridis'), ax=axes, shrink=.5,
                 label='Residual displacement from rest (mm)')
    fig.savefig(args.run/'physical_mesh_stages.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
