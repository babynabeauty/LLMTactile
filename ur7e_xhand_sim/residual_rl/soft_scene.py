"""Single-environment UR7e/XHand scene with an IPC elastic grasp target.

This module creates physics and exposes read-only state. It does not choose a
trajectory, produce tactile signals, reset/checkpoint an episode, or control real
hardware. Every controller evaluation must construct a fresh scene/process.
"""
from pathlib import Path
import hashlib
import fcntl
import shutil
import json
import os
import sys
import xml.etree.ElementTree as ET

os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
os.environ.setdefault('OMP_NUM_THREADS', '4')

import numpy as np
from scipy.spatial.transform import Rotation
import genesis as gs

# Existing scene helpers use sibling absolute imports. Keep this compatibility
# local rather than changing the older Sandbox or the rigid training pipeline.
_SIM_DIR = Path(__file__).resolve().parents[1]
if str(_SIM_DIR) not in sys.path:
    sys.path.insert(0, str(_SIM_DIR))
from build_assets import build, HERE, ARM_JOINTS, HAND_JOINTS
from fem_diagnostics import use_physics_surface, mesh_quality
from ipc_support import weld_ipc_triangle_soups
from .soft_geometry import add_robot_with_exact_sensor_hulls, verify_ipc_sensor_hulls
from .soft_coupling import install_ipc_substep_reaction


def array(value):
    return value.detach().cpu().numpy() if hasattr(value, 'detach') else np.asarray(value)


class SoftScene:
    """Fresh large-translation scene, at 300 Hz with two IPC substeps by default.

    ``set_gain`` accepts either one common thumb/index gain or two finger gains.
    Gain/rate limits are the caller's policy contract; no hidden gain clipping or
    target correction is applied here. Other finger and arm gains remain fixed.
    """
    size_m = .045
    target_position = np.array([-.45, -.0333, .225], dtype=float)
    physics_dt = 1. / 300.

    def __init__(self, output_dir, mass_kg=.1002375, friction=.8, substeps=2,
                 maxvolume=1e-7, render=False, seed=7000, newton_tolerance=.005,
                 exact_sensor_hulls=False):
        if not np.isfinite(mass_kg) or mass_kg <= 0:
            raise ValueError('mass_kg must be finite and positive')
        if not np.isfinite(friction) or not .01 <= friction <= 5:
            raise ValueError('friction must be finite and within [.01, 5]')
        if not isinstance(substeps, (int, np.integer)) or substeps < 1:
            raise ValueError('substeps must be a positive integer')
        if not np.isfinite(maxvolume) or maxvolume <= 0:
            raise ValueError('maxvolume must be finite and positive')
        if not np.isfinite(newton_tolerance) or newton_tolerance <= 0:
            raise ValueError('newton_tolerance must be finite and positive')
        if gs._initialized:
            raise RuntimeError('SoftScene requires its own Genesis lifetime/process')
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.mass_kg = float(mass_kg)
        self.friction = float(friction)
        self.density = self.mass_kg / self.size_m**3
        self.substeps = int(substeps)
        self.newton_tolerance = float(newton_tolerance)
        self.render_enabled = bool(render)
        self.exact_sensor_hulls = bool(exact_sensor_hulls)
        self.seed = int(seed)
        self._closed = False
        self.names = ARM_JOINTS + HAND_JOINTS
        self.home = np.r_[np.deg2rad([0, -89.95, 89.96, -89.96, -89.95, -44.98]),
                          np.zeros(12)]
        self.kp = np.asarray([4500, 4500, 3500, 2000, 2000, 2000] + [3.]*12,
                             dtype=np.float32)
        self.kv = np.asarray([450, 450, 350, 200, 200, 200] + [.08]*12,
                             dtype=np.float32)
        self.base_kp, self.base_kv = self.kp.copy(), self.kv.copy()
        # build() rewrites shared generated assets. Serialize soft workers and
        # copy the URDF while locked, so later parsing reads a private snapshot.
        (HERE/'generated').mkdir(parents=True, exist_ok=True)
        with (HERE/'generated'/'.soft_scene_build.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            model, self.layouts = build()
            self.model = self.output_dir/'ur7e_xhand_scene.urdf'
            shutil.copyfile(model, self.model)
            fcntl.flock(lock, fcntl.LOCK_UN)
        gs.init(backend=gs.gpu, precision='32', seed=self.seed,
                logging_level='warning', use_deterministic_algorithms=True)
        try:
            self._build(float(maxvolume))
        except BaseException:
            self.close()
            raise

    def _build(self, maxvolume):
        coupling = gs.options.IPCCouplerOptions(
            contact_d_hat=.0005, newton_tolerance=self.newton_tolerance,
            newton_translation_tolerance=self.newton_tolerance, linear_system_tolerance=1e-6,
            n_linesearch_iterations=32, enable_rigid_ground_contact=False,
            enable_rigid_rigid_contact=False, enable_rigid_dofs_sync=True)
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.physics_dt, substeps=self.substeps),
            coupler_options=coupling,
            fem_options=gs.options.FEMOptions(use_implicit_solver=True,
                                             damping_alpha=5., damping_beta=5e-4),
            rigid_options=gs.options.RigidOptions(enable_self_collision=True,
                enable_joint_limit=True, enable_neutral_collision=True,
                batch_links_info=False, batch_dofs_info=False),
            vis_options=gs.options.VisOptions(background_color=(.10, .13, .18),
                                              ambient_light=(.5, .5, .5)),
            show_viewer=False)
        self.ground = self.scene.add_entity(gs.morphs.Plane(),
            material=gs.materials.Rigid(friction=.8, coup_friction=.8),
            surface=gs.surfaces.Default(color=(.22, .25, .30, 1)))
        self.table = self.scene.add_entity(
            gs.morphs.Box(size=(1.3, 1., .04), pos=(-.4, 0, .18), fixed=True),
            material=gs.materials.Rigid(friction=.8, coup_friction=.8**2/self.friction, coup_softness=.002),
            surface=gs.surfaces.Default(color=(.40, .30, .20, 1)))
        robot_morph = gs.morphs.URDF(file=str(self.model), fixed=True, pos=(0, 0, .2),
            merge_fixed_links=True, links_to_keep=['right_hand_link', 'tool0'])
        robot_material = gs.materials.Rigid(friction=self.friction, coup_friction=self.friction,
            coup_softness=.002, coup_type='two_way_soft_constraint')
        if self.exact_sensor_hulls:
            self.robot, self.sensor_collision_geometry = add_robot_with_exact_sensor_hulls(
                self.scene, robot_morph, robot_material, self.layouts,
                self.output_dir/'sensor_collision_geometry')
        else:
            self.robot = self.scene.add_entity(robot_morph, material=robot_material)
            self.sensor_collision_geometry = {
                'method': 'standard Genesis collision preprocessing; source fixed sublinks retained',
                'exact_source_hulls_enabled': False,
                'note': 'Exact hulls are a diagnostic option, not a validated tactile-coverage fix.'}
        # There is no rigid duplicate target and no distractor in this scene.
        self.soft = self.scene.add_entity(
            gs.morphs.Box(size=(self.size_m,)*3, pos=tuple(self.target_position),
                          maxvolume=maxvolume, nobisect=False),
            material=gs.materials.FEM.Elastic(E=1e5, nu=.45, rho=self.density,
                model='stable_neohookean', friction_mu=self.friction),
            surface=gs.surfaces.Default(color=(.95, .40, .65, 1)))
        surface_info = use_physics_surface(self.soft)
        self.cameras = {}
        self.camera_source = None
        if self.render_enabled:
            from cameras import load_camera_views, camera_kwargs
            views, self.camera_source = load_camera_views()
            self.cameras = {name: self.scene.add_camera(**camera_kwargs(views[name]))
                            for name in ('scene_front', 'scene_left')}
        # Install before build so both Genesis and IPC see the same home pose.
        # Welding changes IPC import topology, not authored collision geometry.
        self.ipc_mesh_import = weld_ipc_triangle_soups(
            self.scene.sim.coupler, self.robot, dict(zip(self.names, self.home)))
        self.scene.build(n_envs=1)
        self.ipc_reaction_fix = install_ipc_substep_reaction(self.scene.sim.coupler)
        if self.exact_sensor_hulls:
            verify_ipc_sensor_hulls(self.scene, self.robot, self.layouts,
                self.sensor_collision_geometry, self.output_dir/'sensor_collision_geometry')
        self.solver = self.scene.rigid_solver
        self.indices = [self.robot.get_joint(n).dofs_idx_local[0] for n in self.names]
        if len(set(self.indices)) != 18 or self.robot.n_dofs != 18:
            raise RuntimeError('Expected the recorded 18-DOF UR7e/XHand joint layout')
        self.arm_indices, self.hand_indices = self.indices[:6], self.indices[6:]
        self.finger_link_objects = [self.robot.get_link(x['link_name']) for x in self.layouts]
        self.finger_links = [link.idx for link in self.finger_link_objects]
        self.links = self.finger_link_objects
        self.wrist = self.robot.get_link('right_hand_link')
        lo, up = self.robot.get_dofs_limit(self.indices)
        self.low, self.up = array(lo).reshape(18), array(up).reshape(18)
        self.lower, self.upper = self.low, self.up
        self.mid, self.half = (self.low+self.up)/2, (self.up-self.low)/2
        xml = ET.parse(self.model).getroot()
        self.effort = np.asarray([float(xml.find(f"joint[@name='{n}']/limit").get('effort'))
                                 for n in self.names], dtype=np.float32)
        self.set_gain(1.)
        self.robot.set_dofs_force_range(-self.effort, self.effort, self.indices)
        # The mesh hook initialized the state before IPC construction. This is
        # a target command only; do not reset or teleport after IPC build.
        self.apply_target(self.home)
        home_readback, _ = self.joints()
        np.testing.assert_allclose(home_readback, self.home, rtol=0, atol=1e-6)
        self.rest_vertices = array(self.soft.init_positions).reshape(-1, 3).astype(float)
        self.tets = array(self.soft.get_el2v()).astype(np.int64)
        self.surface_triangles = np.asarray(self.soft._surface_tri_np, dtype=np.int64).copy()
        self.boundary = self.surface_triangles
        quality = mesh_quality(self.rest_vertices, self.tets)
        if quality['degenerate_tet_count']:
            raise RuntimeError('FEM rest mesh contains degenerate tetrahedra')
        rest_center = (self.rest_vertices.min(0)+self.rest_vertices.max(0))/2
        np.testing.assert_allclose(rest_center, self.target_position, rtol=0, atol=1e-7)
        np.testing.assert_allclose(np.ptp(self.rest_vertices, axis=0), self.size_m,
                                   rtol=0, atol=1e-7)
        # The reference vertex positions are float32: tolerate their geometric
        # roundoff, but fail if density/mass assignment or mesh scale is wrong.
        tet_mass = quality['volume_m3'] * float(self.soft.material.rho)
        np.testing.assert_allclose(tet_mass, self.mass_kg, rtol=1e-5, atol=1e-10)
        fem = self.scene.fem_solver
        elem_slice = slice(self.soft._el_start, self.soft._el_start+self.soft.n_elements)
        engine_mass = float(np.asarray(fem.elements_i.mass_scaled.to_numpy())[elem_slice].sum()
                            / fem._vol_scale)
        engine_volume = float(np.asarray(fem.elements_i.V.to_numpy())[elem_slice].sum())
        np.testing.assert_allclose(engine_mass, self.mass_kg, rtol=1e-5, atol=1e-10)
        np.testing.assert_allclose(engine_volume, quality['volume_m3'], rtol=1e-5, atol=1e-12)
        # Query the actual registered IPC pair, rather than just restating the
        # requested material values. IPC combines coefficients geometrically.
        coupler = self.scene.sim.coupler
        soft_contact = coupler._ipc_fem_contacts[self.soft]
        friction_readback = {}
        for name, entity, expected in (
                ('hand_soft', self.robot, self.friction),
                ('table_soft', self.table, .8)):
            rigid_contact = coupler._ipc_abd_contacts[entity]
            pair = coupler._ipc_contact_tabular.at(soft_contact.id(), rigid_contact.id())
            actual = float(pair.friction_rate())
            if not pair.is_enabled():
                raise RuntimeError(f'{name} IPC contact is disabled')
            np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)
            friction_readback[name] = {'friction': actual, 'enabled': True,
                'contact_element_ids': [soft_contact.id(), rigid_contact.id()]}
        np.savez_compressed(self.output_dir/'soft_mesh.npz', rest=self.rest_vertices,
                            tets=self.tets, boundary=self.surface_triangles)
        self.metadata = {
            'model_path': str(self.model),
            'model_sha256': hashlib.sha256(Path(self.model).read_bytes()).hexdigest(),
            'joint_order': self.names, 'home_rad': self.home.tolist(),
            'home_readback_rad': home_readback.tolist(),
            'base_kp': self.kp.tolist(), 'base_kv': self.kv.tolist(),
            'effort_limits': self.effort.tolist(),
            'gain_dof_slices': {'thumb': [6, 9], 'index': [9, 12]},
            'gain_damping_rule': 'base_kv * sqrt(gain)',
            'scene': {'target_initial_position_m': self.target_position.tolist(),
                'target_initial_center_readback_m': rest_center.tolist(),
                'table_top_m': .2, 'robot_base_m': [0, 0, .2],
                'target_type': 'FEM only; no rigid duplicate', 'distractors': []},
            'material': {'model': self.soft.material.model, 'E_Pa': float(self.soft.material.E),
                'nu': float(self.soft.material.nu), 'size_m': [self.size_m]*3,
                'requested_mass_kg': self.mass_kg, 'density_kg_m3': self.density,
                'density_readback_kg_m3': float(self.soft.material.rho),
                'mass_from_actual_tet_volume_kg': tet_mass,
                'genesis_fem_element_mass_readback_kg': engine_mass,
                'genesis_fem_rest_volume_readback_m3': engine_volume,
                'mass_readback_relative_tolerance': 1e-5, 'mass_checks_passed': True,
                'fem_friction_mu': float(self.soft.material.friction_mu),
                'robot_coup_friction': float(self.robot.material.coup_friction),
                'table_coup_friction': float(self.table.material.coup_friction),
                'ipc_pair_combination': 'sqrt(mu_a * mu_b)',
                'ipc_pair_readback': friction_readback,
                'calibrated_to_real_silicone': False},
            'physics': {'dt_s': self.physics_dt, 'substeps': self.substeps,
                'substep_dt_s': self.physics_dt/self.substeps, 'seed': self.seed,
                'single_environment': True, 'coupler': 'IPC',
                'coupler_options': coupling.model_dump(),
                'robot_coupling_type': self.robot.material.coup_type,
                'ipc_substep_reaction_fix': self.ipc_reaction_fix.metadata(),
                'rigid_self_and_table_contacts': 'Genesis rigid solver enabled',
                'batch_links_info': False, 'batch_dofs_info': False,
                'checkpoint_or_mid_episode_object_pose_writes': False,
                'fem_damping_options': {'alpha': 5., 'beta': 5e-4},
                'fem_damping_note': 'Legacy FEM options retained; IPC owns soft dynamics'},
            'mesh': {**quality, 'maxvolume_m3': maxvolume,
                'surface_rendering': surface_info, 'ipc_rigid_mesh_import': self.ipc_mesh_import,
                'sensor_collision_geometry': self.sensor_collision_geometry},
            'render_enabled': self.render_enabled, 'camera_source': self.camera_source,
        }
        (self.output_dir/'soft_scene_protocol.json').write_text(
            json.dumps(self.metadata, indent=2)+'\n')

    def apply_target(self, q):
        q = np.asarray(q, dtype=float)
        if q.shape != (18,) or not np.isfinite(q).all():
            raise ValueError('Target must be a finite 18-joint vector')
        if np.any(q < self.low-1e-6) or np.any(q > self.up+1e-6):
            raise ValueError('Recorded target exceeds a robot joint limit')
        self.robot.control_dofs_position(q, self.indices)

    def set_gain(self, gain):
        gain = np.asarray(gain, dtype=float)
        if gain.ndim == 0:
            gain = np.repeat(gain, 2)
        if gain.shape != (2,) or not np.isfinite(gain).all() or np.any(gain <= 0):
            raise ValueError('Gain must be one or two finite positive values')
        kp, kv = self.kp.copy(), self.kv.copy()
        for value, sl in zip(gain, (slice(6, 9), slice(9, 12))):
            kp[sl] *= value
            kv[sl] *= np.sqrt(value)
        self.robot.set_dofs_kp(kp, self.indices)
        self.robot.set_dofs_kv(kv, self.indices)
        self.gains = gain.copy()
        self.applied_kp, self.applied_kv = kp, kv

    def joints(self):
        """Return ordered position and velocity arrays, each shape (18,)."""
        return (array(self.robot.get_dofs_position(self.indices)).reshape(18).copy(),
                array(self.robot.get_dofs_velocity(self.indices)).reshape(18).copy())

    def finger_poses(self):
        """Authored sensor-link frames and world-frame link velocities."""
        positions = np.stack([array(x.get_pos()).reshape(3) for x in self.finger_link_objects])
        quats = np.stack([array(x.get_quat()).reshape(4) for x in self.finger_link_objects])
        return {'positions': positions,
            'rotations': Rotation.from_quat(quats[:, [1, 2, 3, 0]]).as_matrix(),
            'linear_velocities': np.stack([array(x.get_vel()).reshape(3)
                                           for x in self.finger_link_objects]),
            'angular_velocities': np.stack([array(x.get_ang()).reshape(3)
                                            for x in self.finger_link_objects])}

    def vertices(self):
        """Current FEM world positions; no rigid pose approximation or velocity claim."""
        return array(self.soft.get_state().pos).reshape(-1, 3).copy()

    def rigid_collision_fault(self):
        """Original rigid self/table collision criterion, separate from soft contact."""
        contacts = self.robot.get_contacts(is_padded=True)
        a, b = array(contacts['link_a']), array(contacts['link_b'])
        valid = array(contacts['valid_mask']).astype(bool)
        a_robot = (a >= self.robot.link_start) & (a < self.robot.link_end)
        b_robot = (b >= self.robot.link_start) & (b < self.robot.link_end)
        bad = valid & ((a_robot & b_robot) | (a == self.table.base_link.idx)
                       | (b == self.table.base_link.idx))
        bad &= np.linalg.norm(array(contacts['force_a']), axis=-1) > .01
        return bool(bad.any())

    def capture(self, tag):
        """Save requested stage images only when cameras were explicitly enabled."""
        if not self.render_enabled:
            return {}
        if not tag or Path(str(tag)).name != str(tag):
            raise ValueError('Capture tag must be a nonempty filename component')
        from PIL import Image
        paths = {}
        for name, camera in self.cameras.items():
            rgb = array(camera.render(rgb=True)[0])
            if rgb.ndim == 4:
                rgb = rgb[0]
            path = self.output_dir/f'{tag}_{name}.png'
            Image.fromarray(rgb.astype(np.uint8)).save(path)
            paths[name] = str(path)
        return paths

    def close(self):
        if not self._closed:
            if gs._initialized:
                gs.destroy()
            self._closed = True
