"""Virtual XHand taxels on a deforming, closed FEM boundary mesh.

This is a geometric spring/damper *estimate*, not a FEM contact-force sensor.
It deliberately mirrors ``kinematic_tactile._func_kinematic_spring_damper``:
``F = n * (k * penetration**exponent + damping * penetration**exponent * vn)
     - shear * vt`` with relative velocity ``material - probe``.  In particular,
the shear sign is the existing rigid virtual sensor's convention, not a claim
that this is the true traction acting on the fingertip.  Stress and strain are
never treated as force.  Do not use this estimate as independent contact truth.

The input vertices retain material-node identity between 60 Hz samples.  The
nearest triangle's CURRENT barycentric coordinates interpolate both current and
previous vertex positions; differencing independently chosen closest points
would falsely report tangential material motion as the contact patch migrates.
Link rotations map link-frame column vectors into world-frame column vectors.
All geometry is in meters, time in seconds, and force estimates in newtons.
"""

from __future__ import annotations

import copy
from typing import Sequence

import numpy as np
import trimesh


def _signed_nearest_distance(mesh, points, closest, distance, triangle_id):
    """Reuse closest queries with Trimesh's face-projection/raycast sign method.

    The sign is unambiguous when the orthogonal projection lies inside the
    closest face. Edges/corners use closed-mesh ray containment instead of a
    potentially incorrect nearest-face half-space. This is equivalent to the
    strategy in ``trimesh.proximity.signed_distance`` (with outside-positive
    convention), without running the costly closest-point query twice.
    """
    normals = mesh.face_normals[triangle_id]
    dot = np.einsum('pi,pi->p', points - closest, normals)
    projection = points - dot[:, None] * normals
    projected_weights = trimesh.triangles.points_to_barycentric(mesh.triangles[triangle_id], projection)
    on_face = ((projected_weights >= -1e-10) & (projected_weights <= 1 + 1e-10)).all(axis=1)
    signed = np.where(dot < 0., -distance, distance)
    ambiguous = (~on_face) & (distance > 1e-12)
    if ambiguous.any():
        inside = mesh.contains(points[ambiguous])
        signed[ambiguous] = np.where(inside, -distance[ambiguous], distance[ambiguous])
    return signed


class SoftTaxelAdapter:
    """Sample the existing five-finger, 120-taxel-per-finger layout.

    ``layouts`` is the existing list of layout dictionaries, or an array with
    shape (5, 120, 3) of probe positions in each corresponding link frame.
    The boundary must be watertight, winding-consistent, and outward oriented;
    rejected malformed geometry must be repaired upstream, never silently used.
    Faces and material vertex indices cannot change until ``reset()`` is called.
    AABB tests only prune definitely distant probes; every potential contact is
    checked against the nearest triangle and closed-mesh inside/outside test.
    """

    def __init__(self, layouts: Sequence | np.ndarray, *, probe_radius: float = .0015,
                 normal_stiffness: float = 500., normal_damping: float = 1.,
                 normal_exponent: float = 1.2, shear_scalar: float = 2.):
        if isinstance(layouts, (list, tuple)) and layouts and isinstance(layouts[0], dict):
            layouts = [item['positions_link_m'] for item in layouts]
        self.probe_local = np.asarray(layouts, dtype=np.float64).copy()
        if self.probe_local.shape != (5, 120, 3) or not np.isfinite(self.probe_local).all():
            raise ValueError('layouts must contain finite (5, 120, 3) positions in meters')
        values = np.array([probe_radius, normal_stiffness, normal_damping,
                           normal_exponent, shear_scalar], dtype=float)
        if (not np.isfinite(values).all() or np.any(values < 0)
                or probe_radius == 0 or normal_exponent == 0):
            raise ValueError('sensor parameters must be finite and nonnegative; radius/exponent positive')
        self.probe_radius = float(probe_radius)
        self.normal_stiffness = float(normal_stiffness)
        self.normal_damping = float(normal_damping)
        self.normal_exponent = float(normal_exponent)
        self.shear_scalar = float(shear_scalar)
        self.reset()

    def reset(self):
        """Invalidate history at an episode reset or discontinuous teleport."""
        self._previous_vertices = None
        self._previous_probes = None
        self._faces = None
        self._sample_count = 0

    def get_state(self) -> dict:
        """Copy causal history along with the simulator's prefix checkpoint."""
        return copy.deepcopy({
            'vertices': self._previous_vertices, 'probes': self._previous_probes,
            'faces': self._faces, 'sample_count': self._sample_count,
        })

    def set_state(self, state: dict):
        """Restore a checkpoint produced by this adapter's ``get_state``."""
        copied = copy.deepcopy(state)
        self._previous_vertices = copied['vertices']
        self._previous_probes = copied['probes']
        self._faces = copied['faces']
        self._sample_count = int(copied['sample_count'])

    def sample(self, vertices, faces, link_positions, link_rotations, dt: float = 1 / 60,
               *, contact_mask=None) -> dict:
        """Return virtual forces and diagnostic geometry without advancing physics.

        ``queried`` marks probes for which a true nearest-surface query ran.
        ``closest_surface_world_m``, ``barycentric`` and ``signed_distance_m``
        contain NaN at AABB-pruned probes, and ``triangle_id`` is -1 there.
        Their force and penetration are exactly zero.  ``finite`` covers all
        physical outputs plus geometry ONLY where queried.  First-sample
        velocities are zero; subsequent samples use backward differences only.

        ``contact_mask`` is an optional boolean (5,) array indicating whether
        each sensor-bearing link has a physical contact pair with this object.
        The production runner must supply it, matching the rigid taxel's
        opposing-geometry contact prefilter. Only pair presence belongs here:
        never pass force magnitudes or the task's 0.05 N support threshold.
        A false entry zeros force/penetration/active but retains geometric
        diagnostics in ``raw_penetration_m`` and updates causal history as usual.
        None disables the pair gate for stand-alone geometry validation only.
        """
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError('dt must be a positive finite sample interval')
        vertices = np.asarray(vertices, dtype=np.float64)
        raw_faces = np.asarray(faces)
        positions = np.asarray(link_positions, dtype=np.float64)
        rotations = np.asarray(link_rotations, dtype=np.float64)
        if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 4 or not np.isfinite(vertices).all():
            raise ValueError('vertices must be finite (N, 3) material-node positions')
        if (raw_faces.ndim != 2 or raw_faces.shape[1] != 3 or len(raw_faces) < 4
                or not np.issubdtype(raw_faces.dtype, np.integer)):
            raise ValueError('faces must be an integer (M, 3) closed boundary')
        faces = raw_faces.astype(np.int64, copy=False)
        if faces.min() < 0 or faces.max() >= len(vertices):
            raise ValueError('face index outside material vertices')
        if positions.shape != (5, 3) or not np.isfinite(positions).all():
            raise ValueError('link_positions must be finite (5, 3)')
        if (rotations.shape != (5, 3, 3) or not np.isfinite(rotations).all()
                or not np.allclose(rotations @ rotations.transpose(0, 2, 1), np.eye(3), atol=2e-6)
                or not np.allclose(np.linalg.det(rotations), 1., atol=2e-6)):
            raise ValueError('link_rotations must be proper (5, 3, 3) link-to-world rotations')
        mask_applied = contact_mask is not None
        if mask_applied:
            pair_mask = np.asarray(contact_mask)
            if pair_mask.shape != (5,) or pair_mask.dtype != np.dtype(bool):
                raise ValueError('contact_mask must be a boolean (5,) contact-presence array, not force values')
            pair_mask = pair_mask.copy()
        else:
            pair_mask = np.ones(5, dtype=bool)
        has_history = self._previous_vertices is not None
        if has_history and (vertices.shape != self._previous_vertices.shape or not np.array_equal(faces, self._faces)):
            raise ValueError('material-node count or boundary topology changed; reset history explicitly')

        # process=False preserves material vertex and face identity.
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        if not has_history and (not mesh.is_watertight or not mesh.is_winding_consistent):
            raise ValueError('FEM boundary must be watertight and winding-consistent')
        if (not np.isfinite(mesh.volume) or mesh.volume <= 0
                or np.any(mesh.area_faces <= 1e-16)):
            raise ValueError('FEM boundary has reversed orientation, collapsed faces, or nonpositive volume')
        probes = np.einsum('fij,fpj->fpi', rotations, self.probe_local) + positions[:, None, :]
        flat = probes.reshape(-1, 3)
        bounds = mesh.bounds
        queried = np.all((flat >= bounds[0] - self.probe_radius)
                         & (flat <= bounds[1] + self.probe_radius), axis=1)
        chosen = np.flatnonzero(queried)
        n = len(flat)
        closest = np.full((n, 3), np.nan)
        barycentric = np.full((n, 3), np.nan)
        signed = np.full(n, np.nan)
        triangle_ids = np.full(n, -1, dtype=np.int64)
        penetration = np.zeros(n)
        normal = np.zeros((n, 3))
        material_velocity = np.zeros((n, 3))
        probe_velocity = np.zeros((n, 3))
        relative_velocity = np.zeros((n, 3))
        if has_history:
            probe_velocity = (flat - self._previous_probes.reshape(-1, 3)) / dt
        if len(chosen):
            near, distance, triangle_id = trimesh.proximity.closest_point(mesh, flat[chosen])
            sd = _signed_nearest_distance(mesh, flat[chosen], near, distance, triangle_id)
            weights = trimesh.triangles.points_to_barycentric(mesh.triangles[triangle_id], near)
            if not np.isfinite(weights).all():
                raise ValueError('nearest boundary triangle is degenerate')
            closest[chosen] = near
            barycentric[chosen] = weights
            signed[chosen] = sd
            triangle_ids[chosen] = triangle_id
            penetration[chosen] = np.maximum(0., self.probe_radius - sd)
            normal[chosen] = mesh.face_normals[triangle_id]
            if has_history:
                node_displacement = (vertices - self._previous_vertices)[faces[triangle_id]]
                material_velocity[chosen] = np.einsum('pi,pij->pj', weights, node_displacement) / dt
                relative_velocity[chosen] = material_velocity[chosen] - probe_velocity[chosen]

        raw_penetration = penetration.copy()
        penetration[~np.repeat(pair_mask, 120)] = 0.
        active = penetration > 0
        vn = np.einsum('pi,pi->p', relative_velocity, normal)
        tangent = relative_velocity - vn[:, None] * normal
        scale = penetration ** self.normal_exponent
        force_world = (normal * ((self.normal_stiffness + self.normal_damping * vn) * scale)[:, None]
                       - self.shear_scalar * tangent)
        force_world[~active] = 0.
        force_link = np.einsum('fji,fpj->fpi', rotations, force_world.reshape(5, 120, 3))
        finite = bool(np.isfinite(force_link).all() and np.isfinite(penetration).all()
                      and np.isfinite(relative_velocity).all()
                      and np.isfinite(closest[queried]).all()
                      and np.isfinite(barycentric[queried]).all()
                      and np.isfinite(signed[queried]).all())
        if not finite:
            raise FloatingPointError('nonfinite soft virtual taxel output')
        self._previous_vertices = vertices.copy()
        self._previous_probes = probes.copy()
        self._faces = faces.copy()
        self._sample_count += 1
        return {
            'force_link_N': force_link.astype(np.float32),
            'penetration_m': penetration.reshape(5, 120),
            'raw_penetration_m': raw_penetration.reshape(5, 120),
            'active': active.reshape(5, 120),
            'contact_mask_applied': mask_applied,
            'contact_pair_mask': pair_mask,
            'queried': queried.reshape(5, 120),
            'probe_world_m': probes,
            'closest_surface_world_m': closest.reshape(5, 120, 3),
            'signed_distance_m': signed.reshape(5, 120),
            'triangle_id': triangle_ids.reshape(5, 120),
            'barycentric': barycentric.reshape(5, 120, 3),
            'normal_world': normal.reshape(5, 120, 3),
            'material_velocity_world_m_s': material_velocity.reshape(5, 120, 3),
            'probe_velocity_world_m_s': probe_velocity.reshape(5, 120, 3),
            'relative_velocity_world_m_s': relative_velocity.reshape(5, 120, 3),
            'relative_tangent_speed_m_s': np.linalg.norm(tangent, axis=1).reshape(5, 120),
            'has_velocity_history': has_history,
            'sample_count': self._sample_count,
            'finite': finite,
            'queried_probes': int(len(chosen)),
            'active_taxels': int(active.sum()),
            'mesh_volume_m3': float(mesh.volume),
            'measurement_kind': 'geometric_spring_damper_estimate_not_FEM_contact_force',
        }
