"""Actual Genesis rigid-buffer tests; no IPC world or full robot is advanced.

Run ``.venv/bin/python -m ur7e_xhand_sim.residual_rl.test_soft_coupling``.
The fixture invokes the real IPCCoupler reaction method with prescribed ABD
transforms, reads the actual Quadrants force buffers and advances the real rigid
integrator. It tests routing/consumption, not FEM convergence or contact physics.
"""
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
os.environ.setdefault('OMP_NUM_THREADS', '2')

import numpy as np
import genesis as gs
from genesis.engine.couplers.ipc_coupler.coupler import IPCCoupler
from genesis.engine.couplers.ipc_coupler.data import COUPLING_TYPE
from .soft_coupling import install_ipc_substep_reaction


_URDF = '''<robot name="offset_mass"><link name="body">
<inertial><origin xyz="0.03 0.01 0.02"/><mass value="1"/>
<inertia ixx="0.01" iyy="0.01" izz="0.01" ixy="0" ixz="0" iyz="0"/></inertial>
<visual><geometry><box size="0.02 0.02 0.02"/></geometry></visual>
<collision><geometry><box size="0.02 0.02 0.02"/></geometry></collision>
</link></robot>'''


class TestRealRigidBuffers(unittest.TestCase):
    report = {'scope': 'real rigid buffers + original IPC reaction method; prescribed transforms, no FEM solve',
              'cases': []}

    @classmethod
    def setUpClass(cls):
        backend = os.environ.get('SOFT_COUPLING_TEST_BACKEND', 'cpu')
        gs.init(backend=getattr(gs, backend), precision='32', logging_level='warning', seed=7000)
        cls.report['backend'] = backend
        cls.temp = tempfile.TemporaryDirectory()
        cls.urdf = Path(cls.temp.name)/'offset.urdf'
        cls.urdf.write_text(_URDF)

    @classmethod
    def tearDownClass(cls):
        gs.destroy()
        cls.temp.cleanup()
        output = os.environ.get('SOFT_COUPLING_TEST_REPORT')
        if output:
            path = Path(output)
            if path.exists():
                raise RuntimeError(f'Refusing to overwrite evidence {path}')
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(cls.report, indent=2))

    def fixture(self, substeps, patched):
        scene = gs.Scene(sim_options=gs.options.SimOptions(dt=.004, substeps=substeps,
                                                         gravity=(0, 0, 0)), show_viewer=False)
        entity = scene.add_entity(gs.morphs.URDF(file=str(self.urdf), fixed=False))
        other = scene.add_entity(gs.morphs.Box(size=(.02,)*3, pos=(1, 0, 0)))
        scene.build(n_envs=1)
        solver = scene.rigid_solver
        coupler = IPCCoupler(scene.sim, gs.options.IPCCouplerOptions())
        link = entity.base_link
        coupler._entities_by_coup_type = {COUPLING_TYPE.TWO_WAY_SOFT_CONSTRAINT: [entity]}
        coupler._abd_data_by_link = {link: [object()]}
        coupler._constraint_strength_translation_scaled = 100 / scene.sim.substep_dt**2
        coupler._constraint_strength_rotation_scaled = 100 / scene.sim.substep_dt**2
        aim = np.eye(4)[None, None].copy()
        ipc = aim.copy()
        force = np.array([.3, -.2, 2.])
        ipc[0, 0, :3, 3] = force/coupler._constraint_strength_translation_scaled
        coupler._coupling_data = SimpleNamespace(
            links_idx=[link.idx], links_mass=np.ones(1), links_inertia_i=np.eye(3)[None],
            aim_transforms=aim, ipc_transforms=ipc,
            out_forces=np.zeros((1, 1, 3), np.float32),
            out_torques=np.zeros((1, 1, 3), np.float32))
        hook = install_ipc_substep_reaction(coupler) if patched else None
        return scene, solver, coupler, hook, link.idx, other.base_link.idx, force

    @staticmethod
    def buffers(solver):
        state = solver.dyn_state.links
        return {name: np.asarray(getattr(state, name).to_numpy()).copy()
                for name in ('cfrc_applied_vel', 'cfrc_applied_ang',
                             'cfrc_coupling_vel', 'cfrc_coupling_ang')}

    def test_original_accumulates_and_fixed_1_2_4_do_not(self):
        for n in (1, 2, 4):
            final_state = []
            for patched in (False, True):
                scene, solver, coupler, hook, link, other, force = self.fixture(n, patched)
                consumed = []
                for outer in range(2):
                    solver.process_input()
                    # Public user wrench on both a coupled and an unrelated link.
                    solver.apply_links_external_wrench([[.1, .2, -.3]], [[.02, -.03, .01]],
                                                       links_idx=[link])
                    solver.apply_links_external_wrench([[.4, .5, .6]], [[.01, .02, .03]],
                                                       links_idx=[other])
                    user = self.buffers(solver)
                    for substep in range(n):
                        old = self.buffers(solver)
                        pos = solver.dyn_state.links.pos.to_numpy()[link, 0]
                        root = solver.dyn_state.links.root_COM.to_numpy()[link, 0]
                        moment = np.cross(pos-root, force)
                        coupler._apply_abd_coupling_forces()
                        before = self.buffers(solver)
                        if patched:
                            for suffix in ('vel', 'ang'):
                                np.testing.assert_array_equal(before['cfrc_applied_'+suffix],
                                                              user['cfrc_applied_'+suffix])
                            np.testing.assert_allclose(before['cfrc_coupling_vel'][link, 0],
                                                       -force, atol=3e-7)
                            np.testing.assert_allclose(before['cfrc_coupling_ang'][link, 0],
                                                       -moment, atol=3e-7)
                        else:
                            np.testing.assert_allclose(before['cfrc_applied_vel'][link, 0],
                                user['cfrc_applied_vel'][link, 0]-(substep+1)*force, atol=1e-6)
                            np.testing.assert_array_equal(before['cfrc_coupling_vel'], 0)
                        for suffix in ('vel', 'ang'):
                            np.testing.assert_array_equal(before['cfrc_applied_'+suffix][other],
                                                          old['cfrc_applied_'+suffix][other])
                        total = before['cfrc_applied_vel'] + before['cfrc_coupling_vel']
                        consumed.append(float(-(total-user['cfrc_applied_vel'])[link, 0, 2]))
                        solver.substep(scene.sim.cur_substep_local)
                        scene.sim._cur_substep_global += 1
                        after = self.buffers(solver)
                        # Actual native integrator consumed and cleared the substep buffer.
                        np.testing.assert_array_equal(after['cfrc_coupling_vel'], 0)
                        np.testing.assert_array_equal(after['cfrc_coupling_ang'], 0)
                        np.testing.assert_array_equal(after['cfrc_applied_vel'], before['cfrc_applied_vel'])
                    solver.clear_external_force()
                    np.testing.assert_array_equal(self.buffers(solver)['cfrc_applied_vel'], 0)
                self.report['cases'].append({'substeps': n, 'patched': patched,
                    'prescribed_Fz_N': 2., 'actual_consumed_Fz_N': consumed,
                    'user_force_and_torque_preserved': True,
                    'noncoupled_link_unchanged': True,
                    'actual_integrator_cleared_substep_channel': True})
                if patched:
                    self.assertEqual(hook.call_count, 2*n)
                final_state.append(solver.dyn_state.dofs.vel.to_numpy().copy())
            if n == 1:
                # No accumulation occurs with one substep. The real dynamics
                # therefore must be unchanged by routing through the other channel.
                np.testing.assert_allclose(final_state[0], final_state[1], atol=2e-7, rtol=1e-6)
                self.report['single_substep_velocity_equivalent'] = True

    def test_existing_coupling_contribution_is_preserved(self):
        scene, solver, coupler, hook, link, other, force = self.fixture(2, True)
        state = solver.dyn_state.links
        before = self.buffers(solver)
        before['cfrc_coupling_vel'][link, 0] = [.5, .3, .1]
        before['cfrc_coupling_ang'][link, 0] = [.02, .03, .04]
        before['cfrc_coupling_vel'][other, 0] = [.1, .4, .7]
        before['cfrc_coupling_ang'][other, 0] = [.03, .06, .09]
        state.cfrc_coupling_vel.from_numpy(before['cfrc_coupling_vel'])
        state.cfrc_coupling_ang.from_numpy(before['cfrc_coupling_ang'])
        pos = state.pos.to_numpy()[link, 0]
        root = state.root_COM.to_numpy()[link, 0]
        coupler._apply_abd_coupling_forces()
        after = self.buffers(solver)
        np.testing.assert_allclose(after['cfrc_coupling_vel'][link, 0],
                                   before['cfrc_coupling_vel'][link, 0]-force, atol=3e-7)
        np.testing.assert_allclose(after['cfrc_coupling_ang'][link, 0],
                                   before['cfrc_coupling_ang'][link, 0]-np.cross(pos-root, force), atol=3e-7)
        for suffix in ('vel', 'ang'):
            np.testing.assert_array_equal(after['cfrc_coupling_'+suffix][other],
                                          before['cfrc_coupling_'+suffix][other])
            np.testing.assert_array_equal(after['cfrc_applied_'+suffix], before['cfrc_applied_'+suffix])
        self.report['other_native_coupling_contributions_preserved'] = True

    def test_repeated_call_and_double_install_fail(self):
        scene, solver, coupler, hook, *_ = self.fixture(2, True)
        with self.assertRaisesRegex(RuntimeError, 'already installed'):
            install_ipc_substep_reaction(coupler)
        coupler._apply_abd_coupling_forces()
        before = self.buffers(solver)
        with self.assertRaisesRegex(RuntimeError, 'twice'):
            coupler._apply_abd_coupling_forces()
        for name, value in self.buffers(solver).items():
            np.testing.assert_array_equal(value, before[name])

    def test_exception_restores_user_wrench(self):
        scene, solver, coupler, hook, link, *_ = self.fixture(2, True)
        solver.apply_links_external_wrench([[1., 2., 3.]], links_idx=[link])
        before = self.buffers(solver)
        def partial_failure():
            solver.apply_links_external_wrench([[5., 6., 7.]], links_idx=[link])
            raise ValueError('intentional test failure')
        hook._original = partial_failure
        with self.assertRaisesRegex(ValueError, 'intentional'):
            coupler._apply_abd_coupling_forces()
        for name, value in self.buffers(solver).items():
            np.testing.assert_array_equal(value, before[name])


if __name__ == '__main__':
    unittest.main(verbosity=2)
