"""CPU-only three-finger direct-teacher geometry, with no physics execution.

These are privileged geometric candidates, never successful demonstrations by
construction. Independent physical contact/collision checks remain mandatory.
"""
from dataclasses import asdict
from copy import deepcopy
from itertools import product
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

from build_assets import HERE
from direct_teacher_geometry import OpposedPatchGeometry
from residual_rl.multiobject_geometry import ObjectSpec
from s8v4 import sha


FINGERS = ('thumb', 'index', 'middle')
JOINTS = ((0, 1, 2), (3, 4, 5), (6, 7))


class TripodPatchGeometry(OpposedPatchGeometry):
    """Fit three fixed surface patches to a known, fixed-size rigid object."""

    def __init__(self, shape='sphere', contact_height_offset_m=0.):
        if shape not in ('sphere', 'cylinder', 'cube', 'cuboid'):
            raise ValueError('Tripod geometry supports sphere/cylinder/cube/cuboid; hex faces are too narrow')
        height_limit = .030 if shape == 'cylinder' else (.012 if shape in ('cube','cuboid') else .006)
        height_minimum = -.002 if shape == 'sphere' else 0.
        if not np.isfinite(contact_height_offset_m) or not height_minimum <= contact_height_offset_m <= height_limit:
            raise ValueError('Contact height is outside bounded object-specific geometry')
        super().__init__()
        self.shape = shape
        self.contact_height = float(contact_height_offset_m)
        self.object_spec = ObjectSpec(shape=shape, height_m=.09 if shape == 'cylinder' else .045)
        layout_path = HERE/'generated/tactile_layout.json'
        self.layout_path = layout_path
        layouts = json.loads(layout_path.read_text())
        layout = next(row for row in layouts if row['finger'] == 'middle')
        self.layouts.append(layout)
        self.hand[6:8] = [.65, .7]
        points = np.asarray(layout['positions_link_m'])
        mesh = self.meshes[layout['link_name']]
        if not mesh.is_watertight or not mesh.is_winding_consistent or mesh.volume <= 0:
            raise ValueError('Middle patch requires a closed outward-wound collision mesh')
        _, distance, faces = mesh.nearest.on_surface(points)
        if distance.max() > .002:
            raise ValueError('Middle taxels and collision mesh disagree')
        self.points.append(points)
        self.normals.append(mesh.face_normals[faces])
        _, normals = self.sensor_points_normals(self.hand, 2)
        self.masks.append(normals @ np.array([1., 0., 0.]) > .7)
        if min(mask.sum() for mask in self.masks) < 8:
            raise ValueError('Too few fixed facing-patch taxels')

        # Y separation leaves meaningful index/middle clearance. A hexagonal
        # prism's 45 mm across-flats face cannot safely contain these patches.
        radius, y = .0225, .012
        if shape in ('sphere', 'cylinder'):
            horizontal_radius = np.sqrt(radius**2-self.contact_height**2) if shape == 'sphere' else radius
            negative_x = -np.sqrt(horizontal_radius**2-y**2)
            wanted = np.array([[horizontal_radius, 0., self.contact_height],
                               [negative_x, -y, self.contact_height],
                               [negative_x, y, self.contact_height]])
            radial = wanted.copy()
            if shape == 'cylinder':
                radial[:, 2] = 0.
            desired_normals = -radial/np.linalg.norm(radial, axis=1, keepdims=True)
            position_scale = .0005
        else:
            wanted = np.array([[radius, 0., self.contact_height],
                               [-radius, -y, self.contact_height],
                               [-radius, y, self.contact_height]])
            desired_normals = np.array([[-1., 0., 0.], [1., 0., 0.], [1., 0., 0.]])
            position_scale = .001
        seed = self.hand.copy()
        initial_centers = np.array([self.patch(seed, i)[0] for i in range(3)])
        upper = self.upper[:8].copy()
        upper[4] = min(upper[4], .8)  # retain index proximal closing room

        def residual(values):
            hand = seed.copy()
            hand[:8] = values[:8]
            centers = np.array([self.patch(hand, i)[0] for i in range(3)])
            directions = np.array([self.patch(hand, i)[1] for i in range(3)])
            return np.r_[((centers+values[8:]-wanted)/position_scale).ravel(),
                         ((directions-desired_normals)/.15).ravel(),
                         .03*(hand[:8]-seed[:8])]

        fit = least_squares(residual, np.r_[seed[:8], wanted.mean(0)-initial_centers.mean(0)],
            bounds=(np.r_[self.lower[:8], [-.2]*3], np.r_[upper, [.2]*3]),
            xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=2000)
        self.hand = seed.copy()
        self.hand[:8] = fit.x[:8]
        self.palm_offset = fit.x[8:]
        self.patch_centers = np.array([self.patch(self.hand, i)[0] for i in range(3)])
        self.target_centers = wanted
        self.desired_normals = desired_normals
        self.nominal_position_error = np.linalg.norm(self.patch_centers+self.palm_offset-wanted, axis=1)
        if not fit.success or self.nominal_position_error.max() > .0005:
            raise ValueError('Nominal tripod fit exceeds 0.5 mm geometric tolerance')
        if np.any(self.hand < self.lower) or np.any(self.hand > self.upper):
            raise ValueError('Nominal tripod fit violates joint limits')

    def solve(self, inward_m, patch_offsets_world_m=None):
        if not np.isfinite(inward_m) or not -.0050001 <= inward_m <= .0080001:
            raise ValueError('Tripod patch coordinate must remain in [-5,+8] mm')
        offsets = np.zeros((3, 3)) if patch_offsets_world_m is None else np.asarray(patch_offsets_world_m, float)
        if offsets.shape != (3, 3) or not np.isfinite(offsets).all() or np.max(np.abs(offsets)) > .008:
            raise ValueError('Tripod patch corrections require finite 3x3 values within 8 mm')
        hand = self.hand.copy()
        errors = []
        for finger, ids in enumerate(JOINTS):
            indices = np.asarray(ids)
            target = self.patch_centers[finger]+offsets[finger]+np.array([(-1 if finger == 0 else 1)*inward_m, 0., 0.])

            def residual(value):
                trial = self.hand.copy()
                trial[indices] = value
                return (self.patch(trial, finger)[0]-target)*1000

            solution = least_squares(residual, self.hand[indices],
                bounds=(self.lower[indices], self.upper[indices]),
                xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=500)
            hand[indices] = solution.x
            errors.append(float(np.linalg.norm(solution.fun)/1000))
        if max(errors) > 2e-5 or np.any(hand < self.lower) or np.any(hand > self.upper):
            raise ValueError('Tripod path violates FK position or joint limits')
        return hand, errors

    def measure(self, hand):
        vertices = {}
        for name, mesh in self.meshes.items():
            transform = self.transform(hand, name)
            vertices[name] = (mesh.vertices @ transform[:3, :3].T+transform[:3, 3]) @ self.rotation.T+self.palm_offset
        lowest = min((float(v[:, 2].min()), name) for name, v in vertices.items())
        # Names use 'mid', whereas the public finger label is 'middle'.
        grouped = [np.concatenate([v for name, v in vertices.items() if token in name])
                   for token in ('thumb', 'index', '_mid_')]
        distances = {}
        for i, j in ((0, 1), (0, 2), (1, 2)):
            distances[f'{FINGERS[i]}_{FINGERS[j]}'] = float(cKDTree(grouped[i]).query(grouped[j])[0].min())
        normals = np.array([self.patch(hand, i)[1] for i in range(3)])
        angles = np.degrees(np.arccos(np.clip(np.sum(normals*self.desired_normals, axis=1), -1, 1)))
        return dict(lowest_hand_surface_relative_object_center_m=lowest[0], lowest_link=lowest[1],
            minimum_hand_table_clearance_m=float(lowest[0]+self.object_spec.bounds_size_m[2]/2),
            minimum_interfinger_vertex_distances_m=distances,
            patch_normal_mismatch_deg=angles.tolist())

    def candidate(self, gain=2., force_target_N=3., max_inward_m=.002, patch_offsets_world_m=None):
        if gain not in (1., 2., 3.):
            raise ValueError('Common thumb/index gain must be 1, 2, or 3')
        if not np.isfinite(force_target_N) or not .4 <= force_target_N <= 4.:
            raise ValueError('Thumb force target must lie in [0.4,4] N')
        if not np.isfinite(max_inward_m) or not .002 <= max_inward_m <= .008:
            raise ValueError('Maximum inward coordinate must lie in [2,8] mm')
        intervals = int(round((max_inward_m+.005)/.00025))
        if not np.isclose(intervals*.00025, max_inward_m+.005, atol=1e-10, rtol=0):
            raise ValueError('Path endpoint must lie on the 0.25 mm grid')
        coordinates = np.linspace(-.005, max_inward_m, intervals+1)
        hands, errors, measures = [], [], []
        for coordinate in coordinates:
            hand, error = self.solve(float(coordinate), patch_offsets_world_m)
            hands.append(hand)
            errors.extend(error)
            measures.append(self.measure(hand))
        partial = dict(path_inward_m=coordinates.tolist(), path_hand_rad=np.asarray(hands).tolist(),
                       finger_joint_indices=[list(ids) for ids in JOINTS])
        corner_measures = [self.measure(hand_at(partial, values))
                           for values in product((coordinates[0], coordinates[-1]), repeat=3)]
        all_measures = measures+corner_measures
        clearance = min(m['minimum_hand_table_clearance_m'] for m in all_measures)
        pair_min = {pair:min(m['minimum_interfinger_vertex_distances_m'][pair] for m in all_measures)
                    for pair in measures[0]['minimum_interfinger_vertex_distances_m']}
        if clearance < .004 or min(pair_min.values()) < .003:
            raise ValueError('Tripod mesh path fails table clearance or finger vertex-separation proxy')
        source_paths = [Path(__file__), HERE/'direct_teacher_geometry.py', self.urdf,
                        self.reference_path, self.layout_path, Path('/home/sai/zsq/vlm-policy/urdf_frames.py'),
                        HERE/'residual_rl/multiobject_geometry.py']
        return dict(name=f'tripod_fixed_patch_ik_{self.shape}_gain{gain:g}',
            provenance='privileged_direct_teacher; CPU geometry only, physically unverified until scored',
            object_spec=asdict(self.object_spec), contact_fingers=list(FINGERS),
            hand_open_rad=hands[0].tolist(), hand_closed_rad=self.solve(.002, patch_offsets_world_m)[0].tolist(),
            palm_quaternion_wxyz=self.quaternion.tolist(), palm_offset_m=self.palm_offset.tolist(),
            palm_offset_frame='world relative to privileged object center', grip_gain=float(gain),
            gain_scope={'common_scaled_fingers':['thumb','index'], 'middle_absolute_gain':1.,
                        'middle_gain_policy':'unchanged nominal kp/kv; physical readback required'},
            closure_mode='three_independent_fixed_patch_position_ik_with_physical_contact_feedback',
            force_target_per_finger_N=float(force_target_N),
            force_target_by_finger_N={'thumb':float(force_target_N),
                                     'index':float(force_target_N/2), 'middle':float(force_target_N/2)},
            contact_height_offset_m=self.contact_height, max_inward_m=float(max_inward_m),
            patch_offsets_world_m=(np.zeros((3, 3)) if patch_offsets_world_m is None else np.asarray(patch_offsets_world_m)).tolist(),
            patch_correction_scope='Fixed-patch position corrections only; palm registration and object pose remain unchanged. Corrected patch targets need not lie on the nominal object surface.',
            close_increment_m_per_action=.00025, preload_increment_m_per_action=.000125,
            **partial,
            geometric_checks={'fixed_patch_taxel_counts':[int(mask.sum()) for mask in self.masks],
                'target_contact_centers_relative_object_center_m':self.target_centers.tolist(),
                'nominal_patch_centers_relative_object_center_m':(self.patch_centers+self.palm_offset).tolist(),
                'nominal_patch_centers_palm_world_axes_m':self.patch_centers.tolist(),
                'nominal_fit_error_m':self.nominal_position_error.tolist(),
                'maximum_path_fk_error_m':max(errors), 'minimum_hand_table_clearance_m':clearance,
                'minimum_interfinger_vertex_distances_m':pair_min,
                'nominal_geometry':self.measure(self.solve(0., patch_offsets_world_m)[0]), 'path_measurements':measures,
                'independent_coordinate_corner_measurements':corner_measures,
                'scope':'Fixed taxel-patch FK and full hand mesh vertices; not force-closure proof or complete triangle collision certification. Physical three-finger contact, self/table collision, and hold tests remain mandatory.'},
            source_sha256={**{str(path.resolve()):sha(path) for path in source_paths}, **self.mesh_sources})


def hand_at(candidate, coordinates):
    coordinates = np.asarray(coordinates, float)
    if coordinates.shape != (3,) or not np.isfinite(coordinates).all():
        raise ValueError('Three finite independent patch coordinates required')
    knots = np.asarray(candidate['path_inward_m'], float)
    path = np.asarray(candidate['path_hand_rad'], float)
    if path.shape != (len(knots), 12) or len(knots) < 2 or np.any(np.diff(knots) <= 0):
        raise ValueError('Invalid tripod path dimensions or knot order')
    if not np.isfinite(path).all() or np.any(coordinates < knots[0]-1e-8) or np.any(coordinates > knots[-1]+1e-8):
        raise ValueError('Teacher coordinate outside finite prevalidated path')
    if candidate.get('finger_joint_indices') != [list(ids) for ids in JOINTS]:
        raise ValueError('Unexpected three-finger joint map')
    hand = path[0].copy()
    for finger, indices in enumerate(JOINTS):
        for index in indices:
            hand[index] = np.interp(coordinates[finger], knots, path[:, index])
    return hand


def candidate(shape, gain=2., force_target=3., contact_height_offset_m=0., max_inward_m=.002,
              patch_offsets_world_m=None):
    """Public CPU constructor matching the direct-teacher candidate dictionary."""
    return TripodPatchGeometry(shape, contact_height_offset_m).candidate(gain, force_target, max_inward_m,
                                                                      patch_offsets_world_m)


def translated_candidate(source_candidate_path, translation_world_m):
    """Translate a frozen sphere hand path, without refitting or moving the object.

    This bounded placement experiment preserves the old finger path exactly.
    Its shifted patches are explicitly not a newly fitted spherical surface.
    """
    path = Path(source_candidate_path).resolve()
    base = json.loads(path.read_text())
    delta = np.asarray(translation_world_m, float)
    if delta.shape != (3,) or not np.isfinite(delta).all() or np.any(delta[:2] != 0.) or abs(delta[2]) > .006:
        raise ValueError('Rigid placement amendment permits only a finite Z shift within 6 mm')
    if base['object_spec']['shape'] != 'sphere' or base.get('placement_amendment'):
        raise ValueError('A single placement amendment of an original sphere candidate is supported')
    new_height = float(base['contact_height_offset_m']+delta[2])
    if not -.0020000001 <= new_height <= .006:
        raise ValueError('Amended sphere contact plane must remain within [-2,+6] mm')
    geometry = TripodPatchGeometry('sphere', float(base['contact_height_offset_m']))
    np.testing.assert_allclose(base['palm_quaternion_wxyz'], geometry.quaternion, rtol=0, atol=1e-12)
    geometry.palm_offset = np.asarray(base['palm_offset_m'], float)+delta
    hands = np.asarray(base['path_hand_rad'], float)
    if np.any(hands < geometry.lower) or np.any(hands > geometry.upper):
        raise ValueError('Saved path violates joint limits')
    knots = np.asarray(base['path_inward_m'], float)
    measures = [geometry.measure(hand) for hand in hands]
    corners = [geometry.measure(hand_at(base, values))
               for values in product((knots[0], knots[-1]), repeat=3)]
    all_measures = measures+corners
    clearance = min(m['minimum_hand_table_clearance_m'] for m in all_measures)
    pairs = {key:min(m['minimum_interfinger_vertex_distances_m'][key] for m in all_measures)
             for key in measures[0]['minimum_interfinger_vertex_distances_m']}
    if clearance < .004 or min(pairs.values()) < .003:
        raise ValueError('Translated hand path fails the unchanged table/finger geometry gates')
    result = deepcopy(base)
    result.update(name=base['name']+f'_rigid_shift_z_{delta[2]*1000:g}mm',
        palm_offset_m=geometry.palm_offset.tolist(), contact_height_offset_m=new_height,
        placement_amendment={'source_candidate_path':str(path), 'source_candidate_sha256':sha(path),
            'translation_world_m':delta.tolist(),
            'base_source_sha256':base['source_sha256'],
            'base_geometric_checks':base['geometric_checks'],
            'joint_path_and_quaternion_unchanged':True,
            'exact_new_surface_fit':False,
            'scope':'Rigid hand placement only; no object state change. Original hand path and quaternion are retained exactly. Shifted patches are not refitted to the new sphere cross-section.'})
    nominal_hand = hand_at(base, [0., 0., 0.])
    result['geometric_checks'] = {
        'fixed_patch_taxel_counts':[int(mask.sum()) for mask in geometry.masks],
        'nominal_patch_centers_relative_object_center_m':
            [(geometry.patch(nominal_hand, i)[0]+geometry.palm_offset).tolist() for i in range(3)],
        'minimum_hand_table_clearance_m':clearance,
        'minimum_interfinger_vertex_distances_m':pairs,
        'nominal_geometry':geometry.measure(nominal_hand),
        'path_measurements':measures, 'independent_coordinate_corner_measurements':corners,
        'scope':'Full mesh recomputation after bounded rigid hand translation; unchanged joint path and limits. Not exact new surface fit, force-closure proof, or triangle collision certification. Actual physical contacts/collision/hold gates remain mandatory.'}
    result['source_sha256'] = {key:sha(Path(key)) for key in base['source_sha256']}
    result['source_sha256'][str(path)] = sha(path)
    return result
