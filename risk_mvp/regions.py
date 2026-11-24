"""Object surface regions and the automatic damage-threshold generator (MVP step 1).

Regions: sample the surface, farthest-point-sample K centres, label every face by its
nearest centre. A point (e.g. a solver contact in the object frame) belongs to the
region of its nearest centre, so face labels and contact labels use the same rule.

Thresholds (never hand-set): object strength log-uniform in [0.5, 10] N, a spatially
correlated log-normal coefficient per region (neighbouring regions share material),
and with probability 0.2 one hidden weak region whose coefficient is 0.3-0.6.
Every call with a new rng draw is a new object instance.

CPU only (numpy + trimesh); no Genesis import.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import trimesh

SHAPES = ('cube', 'cuboid', 'flat_box', 'cylinder', 'sphere', 'hex_prism')


def object_mesh(spec: dict, max_edge_m: float = .004) -> trimesh.Trimesh:
    """Mesh in the object frame (origin at the centre, z up), subdivided so faces are small.

    spec follows residual_rl.multiobject_geometry.ObjectSpec fields; 'flat_box' is a
    cuboid by size_m (kept separate only as a name). A 'mesh_path' key loads a file
    (e.g. a YCB object) and recentres it on its bounding-box centre.
    """
    if 'mesh_path' in spec:
        m = trimesh.load(spec['mesh_path'], force='mesh')
        m.apply_translation(-m.bounds.mean(0))
        if spec.get('exterior', True):
            # Fingers only touch the outside: partition the convex hull, so thin-walled objects (cups)
            # do not mix inner and outer wall in one region and region normals stay meaningful.
            m = m.convex_hull
    else:
        shape = spec['shape']
        if shape == 'cube':
            m = trimesh.creation.box(extents=[spec.get('side_m', .045)]*3)
        elif shape in ('cuboid', 'flat_box'):
            m = trimesh.creation.box(extents=spec['size_m'])
        elif shape == 'cylinder':
            m = trimesh.creation.cylinder(radius=spec.get('radius_m', .0225), height=spec.get('height_m', .045),
                                          sections=48)
        elif shape == 'sphere':
            m = trimesh.creation.icosphere(subdivisions=3, radius=spec.get('radius_m', .0225))
        elif shape == 'hex_prism':
            af, h = spec.get('across_flats_m', .045), spec.get('height_m', .045)
            r = af/(2*np.cos(np.pi/6))
            m = trimesh.creation.cylinder(radius=r, height=h, sections=6)
            m.apply_transform(trimesh.transformations.rotation_matrix(np.deg2rad(30), [0, 0, 1]))
        else:
            raise ValueError(f'unknown shape {shape}')
    v, f = trimesh.remesh.subdivide_to_size(m.vertices, m.faces, max_edge=max_edge_m)
    return trimesh.Trimesh(v, f, process=True)


def farthest_point_sampling(points: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    idx = [int(rng.integers(len(points)))]
    d = np.linalg.norm(points-points[idx[0]], axis=1)
    for _ in range(k-1):
        idx.append(int(np.argmax(d)))
        d = np.minimum(d, np.linalg.norm(points-points[idx[-1]], axis=1))
    return np.asarray(idx)


@dataclass
class Partition:
    centers: np.ndarray            # (K, 3) object frame
    face_labels: np.ndarray        # (F,)
    area: np.ndarray               # (K,) m^2
    centroid: np.ndarray           # (K, 3) area-weighted face centroid
    normal: np.ndarray             # (K, 3) unit mean outward normal
    flatness: np.ndarray           # (K,) |area-weighted mean normal|; 1 = flat, lower = curved/edge
    seed: int = 0
    meta: dict = field(default_factory=dict)

    @property
    def k(self) -> int:
        return len(self.centers)

    def region_of(self, points: np.ndarray) -> np.ndarray:
        """Region index of each point (..., 3) in the object frame."""
        p = np.asarray(points, float)
        d = np.linalg.norm(p[..., None, :]-self.centers, axis=-1)
        return d.argmin(-1)

    def features(self) -> np.ndarray:
        """(K, 8) geometry features for the risk model: area (cm^2), flatness, normal, centroid (cm)."""
        return np.concatenate([self.area[:, None]*1e4, self.flatness[:, None], self.normal, self.centroid*100], 1)

    def to_json(self) -> dict:
        return dict(centers=self.centers.tolist(), area=self.area.tolist(), centroid=self.centroid.tolist(),
                    normal=self.normal.tolist(), flatness=self.flatness.tolist(), seed=self.seed, meta=self.meta)


def partition(mesh: trimesh.Trimesh, k: int, seed: int = 0, n_samples: int = 4000,
              separate: Optional[Sequence[Sequence[float]]] = None, max_tries: int = 200) -> Partition:
    """K regions by FPS. `separate`: object-frame points that must fall in pairwise
    different regions (e.g. the index and middle contacts of the swap experiment);
    seeds are tried in order from `seed` until that holds."""
    if not 2 <= k <= 12:
        raise ValueError('k must be in 2..12')
    for s in range(seed, seed+max_tries):
        rng = np.random.default_rng(s)
        pts, _ = trimesh.sample.sample_surface_even(mesh, n_samples, seed=s)
        centers = pts[farthest_point_sampling(pts, k, rng)]
        if separate is not None:
            lab = np.linalg.norm(np.asarray(separate, float)[:, None]-centers, axis=-1).argmin(-1)
            if len(set(lab.tolist())) < len(separate):
                continue
        tc = mesh.triangles_center
        labels = np.linalg.norm(tc[:, None]-centers, axis=-1).argmin(-1)
        area = np.bincount(labels, weights=mesh.area_faces, minlength=k)
        if (area <= 0).any():
            continue
        w = mesh.area_faces[:, None]
        centroid = np.stack([(tc*w)[labels == r].sum(0)/area[r] for r in range(k)])
        nsum = np.stack([(mesh.face_normals*w)[labels == r].sum(0)/area[r] for r in range(k)])
        flat = np.linalg.norm(nsum, axis=1)
        return Partition(centers, labels, area, centroid, nsum/np.maximum(flat[:, None], 1e-12), flat, seed=s,
                         meta=dict(k=k, n_samples=n_samples, separate=None if separate is None else np.asarray(separate).tolist()))
    raise RuntimeError(f'no seed in {seed}..{seed+max_tries} separates the given points into different regions')


@dataclass
class Instance:
    """One object instance: hidden per-region thresholds (evaluation only, never a model input)."""
    base_N: float
    coeff: np.ndarray
    weak_region: Optional[int]

    @property
    def thresholds(self) -> np.ndarray:
        return self.base_N*self.coeff

    def to_json(self) -> dict:
        return dict(base_N=self.base_N, coeff=self.coeff.tolist(), thresholds_N=self.thresholds.tolist(),
                    weak_region=self.weak_region)


def sample_instance(part: Partition, rng: np.random.Generator, base_range=(.5, 10.), sigma=.25,
                    length_scale_m: Optional[float] = None, p_weak=.2, weak_range=(.3, .6),
                    weak_region: Optional[int] = None) -> Instance:
    """Correlated region coefficients: log c ~ N(0, sigma^2 K_rbf(centers)). Passing
    weak_region forces a weak spot there (swap experiment); otherwise one appears with p_weak."""
    base = float(np.exp(rng.uniform(*np.log(base_range))))
    c = part.centers
    ell = length_scale_m or .5*float(np.ptp(c, 0).max() or .01)
    d2 = ((c[:, None]-c[None])**2).sum(-1)
    cov = sigma**2*np.exp(-.5*d2/ell**2)+1e-9*np.eye(part.k)
    coeff = np.exp(rng.multivariate_normal(np.zeros(part.k), cov))
    if weak_region is None and rng.random() < p_weak:
        weak_region = int(rng.integers(part.k))
    if weak_region is not None:
        coeff[weak_region] = rng.uniform(*weak_range)
    return Instance(base, coeff, weak_region)
