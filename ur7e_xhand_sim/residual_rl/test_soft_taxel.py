"""CPU geometry tests; these do not validate FEM physics or hardware calibration.

Run: .venv/bin/python -m unittest ur7e_xhand_sim.residual_rl.test_soft_taxel -v
Benchmark: .venv/bin/python -m ur7e_xhand_sim.residual_rl.test_soft_taxel --benchmark
"""

import json
import sys
import time
import unittest

import numpy as np
import trimesh

from .soft_taxel import SoftTaxelAdapter


def layout(point=(.001, .002, .0105)):
    return np.tile(point, (5, 120, 1)).astype(float)


class TestSoftTaxel(unittest.TestCase):
    def setUp(self):
        self.mesh = trimesh.creation.box(extents=[.02, .02, .02])
        self.positions = np.zeros((5, 3))
        self.rotations = np.tile(np.eye(3), (5, 1, 1))

    def sample(self, adapter, vertices=None, faces=None, positions=None, rotations=None, dt=1 / 60,
               contact_mask=None):
        return adapter.sample(self.mesh.vertices if vertices is None else vertices,
                              self.mesh.faces if faces is None else faces,
                              self.positions if positions is None else positions,
                              self.rotations if rotations is None else rotations, dt,
                              contact_mask=contact_mask)

    def test_analytic_plane_face_depth_and_force(self):
        result = self.sample(SoftTaxelAdapter(layout()))
        np.testing.assert_allclose(result['signed_distance_m'], .0005, atol=1e-12)
        np.testing.assert_allclose(result['penetration_m'], .001, atol=1e-12)
        np.testing.assert_allclose(result['force_link_N'],
                                   np.tile([0, 0, 500 * .001 ** 1.2], (5, 120, 1)), rtol=1e-6)
        self.assertEqual(result['force_link_N'].dtype, np.float32)
        self.assertEqual(result['active_taxels'], 600)
        self.assertTrue(result['finite'])

    def test_first_frame_has_no_velocity_including_after_reset(self):
        adapter = SoftTaxelAdapter(layout())
        result = self.sample(adapter)
        self.assertFalse(result['has_velocity_history'])
        np.testing.assert_array_equal(result['relative_velocity_world_m_s'], 0)
        self.sample(adapter, vertices=self.mesh.vertices + [0, 0, .0001])
        adapter.reset()
        result = self.sample(adapter, vertices=self.mesh.vertices + [0, 0, .0002])
        self.assertFalse(result['has_velocity_history'])
        np.testing.assert_array_equal(result['relative_velocity_world_m_s'], 0)

    def test_no_contact_even_with_motion(self):
        adapter = SoftTaxelAdapter(layout((.5, .3, .2)))
        self.sample(adapter)
        result = self.sample(adapter, vertices=self.mesh.vertices + [.1, .1, .1])
        self.assertEqual(result['queried_probes'], 0)
        self.assertEqual(result['active_taxels'], 0)
        np.testing.assert_array_equal(result['force_link_N'], 0)
        self.assertTrue(np.isnan(result['closest_surface_world_m']).all())
        self.assertTrue(result['finite'])

    def test_aabb_is_not_contact_at_corner(self):
        # Both x/y are within expanded AABB, but Euclidean gap exceeds radius.
        result = self.sample(SoftTaxelAdapter(layout((.0114, .0114, 0))))
        self.assertEqual(result['queried_probes'], 600)
        self.assertEqual(result['active_taxels'], 0)
        np.testing.assert_allclose(result['signed_distance_m'], np.sqrt(2) * .0014, atol=1e-12)
        np.testing.assert_array_equal(result['force_link_N'], 0)

    def test_inside_signed_distance_and_outward_normal(self):
        result = self.sample(SoftTaxelAdapter(layout((.001, .002, .009))))
        np.testing.assert_allclose(result['signed_distance_m'], -.001, atol=1e-12)
        np.testing.assert_allclose(result['penetration_m'], .0025, atol=1e-12)
        np.testing.assert_allclose(result['normal_world'], np.tile([0, 0, 1], (5, 120, 1)))

    def test_inside_aabb_of_nonconvex_mesh_is_not_inside_material(self):
        annulus = trimesh.creation.annulus(r_min=.006, r_max=.012, height=.02, sections=24)
        result = self.sample(SoftTaxelAdapter(layout((0, 0, 0))),
                             vertices=annulus.vertices, faces=annulus.faces)
        self.assertEqual(result['queried_probes'], 600)
        self.assertEqual(result['active_taxels'], 0)
        self.assertTrue((result['signed_distance_m'] > .005).all())

    def test_signed_distance_matches_trimesh_for_random_box_points(self):
        rng = np.random.default_rng(15)
        points = rng.uniform(-.0115, .0115, (5, 120, 3))
        result = self.sample(SoftTaxelAdapter(points))
        reference = -trimesh.proximity.signed_distance(self.mesh, points.reshape(-1, 3))
        np.testing.assert_allclose(result['signed_distance_m'].ravel(), reference, atol=1e-12)

    def test_nearest_triangle_barycentric_reconstructs_surface(self):
        result = self.sample(SoftTaxelAdapter(layout()))
        triangles = self.mesh.vertices[self.mesh.faces[result['triangle_id']]]
        reconstructed = np.einsum('fpi,fpij->fpj', result['barycentric'], triangles)
        np.testing.assert_allclose(reconstructed, result['closest_surface_world_m'], atol=1e-14)
        np.testing.assert_allclose(result['barycentric'].sum(axis=-1), 1., atol=1e-14)

    def test_stationary_probe_and_translating_material_sign(self):
        adapter = SoftTaxelAdapter(layout())
        self.sample(adapter)
        result = self.sample(adapter, vertices=self.mesh.vertices + [.0001, 0, .0001])
        # Existing rigid sensor uses material - probe and negative shear*vt.
        np.testing.assert_allclose(result['material_velocity_world_m_s'],
                                   np.tile([.006, 0, .006], (5, 120, 1)), atol=1e-13)
        expected_z = (500 + .006) * .0011 ** 1.2
        np.testing.assert_allclose(result['force_link_N'], np.tile([-.012, 0, expected_z], (5, 120, 1)), rtol=1e-6)

    def test_moving_probe_on_stationary_mesh_has_opposite_shear(self):
        adapter = SoftTaxelAdapter(layout())
        self.sample(adapter)
        result = self.sample(adapter, positions=self.positions + [.0001, 0, 0])
        np.testing.assert_allclose(result['material_velocity_world_m_s'], 0, atol=1e-14)
        np.testing.assert_allclose(result['relative_tangent_speed_m_s'], .006, atol=1e-13)
        np.testing.assert_allclose(result['force_link_N'][..., 0], .012, rtol=1e-6)

    def test_common_rigid_translation_produces_no_false_shear(self):
        adapter = SoftTaxelAdapter(layout())
        first = self.sample(adapter)
        displacement = [.001, -.002, .003]
        result = self.sample(adapter, vertices=self.mesh.vertices + displacement,
                             positions=self.positions + displacement)
        np.testing.assert_allclose(result['relative_velocity_world_m_s'], 0, atol=1e-13)
        np.testing.assert_allclose(result['force_link_N'], first['force_link_N'], rtol=1e-6, atol=1e-13)

    def test_global_frame_rotation_preserves_link_frame_force(self):
        adapter = SoftTaxelAdapter(layout())
        reference = self.sample(adapter)
        # Rotate the complete scene so old +z becomes world +x.
        rotation = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=float)
        rotated = self.sample(SoftTaxelAdapter(layout()), vertices=self.mesh.vertices @ rotation.T,
                              rotations=np.tile(rotation, (5, 1, 1)))
        np.testing.assert_allclose(rotated['force_link_N'], reference['force_link_N'], atol=1e-7)
        np.testing.assert_allclose(rotated['normal_world'], np.tile([1, 0, 0], (5, 120, 1)))

    def test_current_barycentric_tracks_material_not_closest_point_motion(self):
        adapter = SoftTaxelAdapter(layout((-.002, .001, .0105)))
        self.sample(adapter)
        result = self.sample(adapter, positions=self.positions + [.003, 0, 0])
        # The closest point moves 3 mm across a fixed surface; material does not.
        np.testing.assert_allclose(result['material_velocity_world_m_s'], 0, atol=1e-14)
        np.testing.assert_allclose(result['relative_velocity_world_m_s'][..., 0], -.18, atol=1e-12)
        np.testing.assert_allclose(result['force_link_N'][..., 0], .36, rtol=1e-6)

    def test_deforming_material_velocity_is_current_barycentric_difference(self):
        adapter = SoftTaxelAdapter(layout())
        self.sample(adapter)
        deformed = self.mesh.vertices.copy()
        deformed[:, 0] *= 1.01
        result = self.sample(adapter, vertices=deformed)
        # Current closest x=.001 originated at x=.001/1.01 on affine material.
        expected_vx = (.001 - .001 / 1.01) * 60
        np.testing.assert_allclose(result['material_velocity_world_m_s'][..., 0], expected_vx, atol=1e-13)
        np.testing.assert_allclose(result['force_link_N'][..., 0], -2 * expected_vx, rtol=1e-6)

    def test_patch_moves_between_taxels(self):
        positions = layout((0, 0, .0105))
        positions[:, 60:, 0] = .03
        adapter = SoftTaxelAdapter(positions)
        first = self.sample(adapter)
        second = self.sample(adapter, vertices=self.mesh.vertices + [.03, 0, 0])
        self.assertTrue(first['active'][:, :60].all())
        self.assertFalse(first['active'][:, 60:].any())
        self.assertFalse(second['active'][:, :60].any())
        self.assertTrue(second['active'][:, 60:].all())

    def test_checkpoint_restores_causal_history_exactly(self):
        adapter = SoftTaxelAdapter(layout())
        self.sample(adapter)
        checkpoint = adapter.get_state()
        expected = self.sample(adapter, vertices=self.mesh.vertices + [.0001, 0, 0])
        self.sample(adapter, vertices=self.mesh.vertices + [.0005, 0, 0])
        adapter.set_state(checkpoint)
        restored = self.sample(adapter, vertices=self.mesh.vertices + [.0001, 0, 0])
        for key in ['force_link_N', 'relative_velocity_world_m_s', 'penetration_m']:
            np.testing.assert_array_equal(expected[key], restored[key])
        self.assertEqual(restored['sample_count'], 2)

    def test_history_is_not_aliased_with_input_or_output(self):
        adapter = SoftTaxelAdapter(layout())
        vertices = self.mesh.vertices.copy()
        result = self.sample(adapter, vertices=vertices)
        vertices[:] += 1
        result['probe_world_m'][:] += 1
        checkpoint = adapter.get_state()
        checkpoint['vertices'][:] += 1
        result = self.sample(adapter)
        np.testing.assert_allclose(result['relative_velocity_world_m_s'], 0, atol=1e-14)

    def test_invalid_topology_or_geometry_is_rejected(self):
        adapter = SoftTaxelAdapter(layout())
        with self.assertRaisesRegex(ValueError, 'watertight'):
            self.sample(adapter, faces=self.mesh.faces[:-1])
        with self.assertRaisesRegex(ValueError, 'orientation'):
            self.sample(adapter, faces=self.mesh.faces[:, ::-1])
        self.sample(adapter)
        with self.assertRaisesRegex(ValueError, 'topology'):
            self.sample(adapter, faces=self.mesh.faces[::-1])

    def test_invalid_input_and_interval_are_rejected(self):
        adapter = SoftTaxelAdapter(layout())
        for dt in [0, -1, float('nan'), float('inf')]:
            with self.subTest(dt=dt), self.assertRaises(ValueError):
                self.sample(adapter, dt=dt)
        with self.assertRaises(ValueError):
            self.sample(adapter, rotations=np.zeros((5, 3, 3)))
        with self.assertRaises(ValueError):
            self.sample(adapter, vertices=np.full_like(self.mesh.vertices, np.nan))
        with self.assertRaises(ValueError):
            SoftTaxelAdapter(np.zeros((2, 120, 3)))

    def test_pair_gate_zeros_outputs_but_preserves_geometric_diagnostics(self):
        unmasked = self.sample(SoftTaxelAdapter(layout()))
        mask = np.array([True, False, True, False, False])
        gated = self.sample(SoftTaxelAdapter(layout()), contact_mask=mask)
        self.assertTrue(gated['contact_mask_applied'])
        np.testing.assert_array_equal(gated['contact_pair_mask'], mask)
        np.testing.assert_array_equal(gated['force_link_N'][~mask], 0)
        np.testing.assert_array_equal(gated['penetration_m'][~mask], 0)
        np.testing.assert_array_equal(gated['active'][~mask], False)
        np.testing.assert_array_equal(gated['force_link_N'][mask], unmasked['force_link_N'][mask])
        for key in ['raw_penetration_m', 'closest_surface_world_m', 'signed_distance_m',
                    'normal_world', 'barycentric', 'triangle_id', 'queried']:
            np.testing.assert_array_equal(gated[key], unmasked[key])

    def test_pair_closure_does_not_leak_previous_frame_force(self):
        adapter = SoftTaxelAdapter(layout())
        first = self.sample(adapter, contact_mask=np.ones(5, dtype=bool))
        self.assertTrue((np.linalg.norm(first['force_link_N'], axis=-1) > 0).all())
        closed = self.sample(adapter, vertices=self.mesh.vertices + [.0001, 0, .0001],
                             contact_mask=np.zeros(5, dtype=bool))
        np.testing.assert_array_equal(closed['force_link_N'], 0)
        np.testing.assert_array_equal(closed['penetration_m'], 0)
        np.testing.assert_array_equal(closed['active'], False)
        self.assertTrue((closed['raw_penetration_m'] > 0).all())
        self.assertTrue(closed['has_velocity_history'])
        self.assertEqual(closed['sample_count'], 2)

    def test_pair_reopens_with_continuous_causal_velocity(self):
        adapter = SoftTaxelAdapter(layout())
        reference = SoftTaxelAdapter(layout())
        for step in range(4):
            vertices = self.mesh.vertices + [step * .0001, 0, 0]
            expected = self.sample(reference, vertices=vertices)
            result = self.sample(adapter, vertices=vertices,
                                 contact_mask=np.full(5, step == 3, dtype=bool))
            np.testing.assert_array_equal(result['relative_velocity_world_m_s'],
                                          expected['relative_velocity_world_m_s'])
            if step < 3:
                np.testing.assert_array_equal(result['force_link_N'], 0)
        np.testing.assert_array_equal(result['force_link_N'], expected['force_link_N'])
        np.testing.assert_allclose(result['material_velocity_world_m_s'][..., 0], .006, atol=1e-13)
        self.assertEqual(result['sample_count'], 4)

    def test_pair_gate_rejects_force_values_and_wrong_shape(self):
        for value in [np.ones(5), np.zeros(5, dtype=int), np.ones((5, 1), dtype=bool),
                      np.ones(4, dtype=bool), [.05, .05, 0., 0., 0.]]:
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'boolean'):
                self.sample(SoftTaxelAdapter(layout()), contact_mask=value)


def benchmark(repeats=40):
    """Report wall time only, never assert hardware-dependent real-time speed."""
    mesh = trimesh.creation.icosphere(subdivisions=2, radius=.0225)
    rng = np.random.default_rng(3)
    directions = rng.normal(size=(5, 120, 3))
    directions /= np.linalg.norm(directions, axis=-1, keepdims=True)
    rotations = np.tile(np.eye(3), (5, 1, 1))
    result = {'benchmark_kind': 'CPU_mesh_only_not_Genesis',
              'mesh_vertices': len(mesh.vertices), 'mesh_faces': len(mesh.faces),
              'probes': 600, 'repeats': repeats, 'requested_sample_hz': 60,
              'numpy_version': np.__version__, 'trimesh_version': trimesh.__version__}
    for name, points in [('all_near_surface', directions * .0228), ('all_distant', directions * .2)]:
        adapter = SoftTaxelAdapter(points)
        adapter.sample(mesh.vertices, mesh.faces, np.zeros((5, 3)), rotations)
        times = []
        for i in range(repeats):
            displacement = np.array([i * 1e-6, 0, 0])
            start = time.perf_counter()
            sampled = adapter.sample(mesh.vertices + displacement, mesh.faces,
                                     np.tile(displacement, (5, 1)), rotations)
            times.append((time.perf_counter() - start) * 1000)
        result[name] = {'mean_ms': float(np.mean(times)), 'p95_ms': float(np.quantile(times, .95)),
                        'max_ms': float(np.max(times)), 'queried_probes': sampled['queried_probes'],
                        'active_taxels': sampled['active_taxels']}
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    if '--benchmark' in sys.argv:
        benchmark()
    else:
        unittest.main()
