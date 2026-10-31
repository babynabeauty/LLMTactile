"""Instance-local IPC reaction fix for the elastic-target experiments.

Genesis' IPC coupler sends a new reaction through ``apply_links_external_wrench``
on every substep. That API accumulates in ``cfrc_applied_*`` until the *outer*
scene step ends. Consequently an n-substep scene can consume F, 2F, ..., nF.
The engine already has a separate ``cfrc_coupling_*`` channel which the rigid
integrator consumes and clears on every substep.

This hook preserves the original IPC force calculation and public wrench API
(including wake-up and the moment about root COM). It transfers only the raw
delta just written by that call into the native substep coupling channel, and
restores the exact prior external-wrench buffers. No material, force magnitude,
controller, global class, or other scene is modified. Install after scene.build,
before the first scene.step; this is not a checkpoint/backpropagation hook.
"""

from types import MethodType

import numpy as np
import quadrants as qd

import genesis as gs
from genesis.utils import array_class


@qd.kernel
def _snapshot(links_idx: qd.types.ndarray(), state: array_class.DynState,
              linear: qd.Tensor, angular: qd.Tensor):
    for j, b in qd.ndrange(links_idx.shape[0], linear.shape[1]):
        link = links_idx[j]
        linear[j, b] = state.links.cfrc_applied_vel[link, b]
        angular[j, b] = state.links.cfrc_applied_ang[link, b]


@qd.kernel
def _restore_and_transfer(links_idx: qd.types.ndarray(), state: array_class.DynState,
                          linear: qd.Tensor, angular: qd.Tensor,
                          transfer: qd.template()):
    for j, b in qd.ndrange(links_idx.shape[0], linear.shape[1]):
        link = links_idx[j]
        if qd.static(transfer):
            state.links.cfrc_coupling_vel[link, b] += (
                state.links.cfrc_applied_vel[link, b] - linear[j, b])
            state.links.cfrc_coupling_ang[link, b] += (
                state.links.cfrc_applied_ang[link, b] - angular[j, b])
        state.links.cfrc_applied_vel[link, b] = linear[j, b]
        state.links.cfrc_applied_ang[link, b] = angular[j, b]


class IPCSubstepReaction:
    """Installed hook; ``metadata()`` records the numerical provenance.

    The snapshots contain the already transformed engine buffers, so a changed
    link pose/root COM does not require cancelling an old force at a new arm.
    Other links and any pre-existing native coupling contributions are intact.
    The original engine handles scene-step external-force clearing as before.
    """

    def __init__(self, coupler):
        from genesis.engine.couplers.ipc_coupler.coupler import IPCCoupler
        from genesis.engine.couplers.ipc_coupler.data import COUPLING_TYPE

        if not isinstance(coupler, IPCCoupler):
            raise TypeError('Expected a built IPCCoupler instance')
        if getattr(coupler, '_soft_substep_reaction', None) is not None:
            raise RuntimeError('IPC substep reaction hook is already installed')
        if coupler._coupling_data is None or not coupler.rigid_solver.is_built:
            raise RuntimeError('Install the reaction hook after scene.build')
        if not coupler.options.two_way_coupling:
            raise ValueError('Two-way coupling is disabled')
        if COUPLING_TYPE.TWO_WAY_SOFT_CONSTRAINT not in coupler._entities_by_coup_type:
            raise ValueError('No two-way soft-constraint entity in this scene')
        self.coupler = coupler
        self.solver = coupler.rigid_solver
        self.links_idx = np.asarray(coupler._coupling_data.links_idx, dtype=np.int32)
        if (self.links_idx.ndim != 1 or not len(self.links_idx)
                or len(np.unique(self.links_idx)) != len(self.links_idx)
                or self.links_idx.min() < 0 or self.links_idx.max() >= self.solver.n_links):
            raise ValueError('Coupling links must be a nonempty unique valid index list')
        shape = (len(self.links_idx), int(coupler.sim._B))
        self._linear = array_class.V_VEC(3, dtype=gs.qd_float, shape=shape)
        self._angular = array_class.V_VEC(3, dtype=gs.qd_float, shape=shape)
        self._original = coupler._apply_abd_coupling_forces
        self._last_substep = None
        self.call_count = 0
        self._installed = True

        def apply(instance):
            return self.apply()

        self._wrapper = MethodType(apply, coupler)
        coupler._apply_abd_coupling_forces = self._wrapper
        coupler._soft_substep_reaction = self

    def apply(self):
        step = int(self.coupler.sim.cur_substep_global)
        if self._last_substep is not None and step <= self._last_substep:
            raise RuntimeError('IPC reaction invoked twice or rewound without a fresh scene')
        if list(self.coupler._coupling_data.links_idx) != self.links_idx.tolist():
            raise RuntimeError('Coupling link topology changed after hook installation')
        _snapshot(self.links_idx, self.solver.dyn_state, self._linear, self._angular)
        try:
            self._original()
        except BaseException:
            # Do not leave a partially applied reaction in the external channel.
            _restore_and_transfer(self.links_idx, self.solver.dyn_state,
                                  self._linear, self._angular, False)
            raise
        _restore_and_transfer(self.links_idx, self.solver.dyn_state,
                              self._linear, self._angular, True)
        self._last_substep = step
        self.call_count += 1

    def metadata(self):
        return {'name': 'ipc_reaction_native_substep_channel', 'version': 1,
                'installed': self._installed, 'links_idx': self.links_idx.tolist(),
                'substep_dt_s': float(self.coupler.sim.substep_dt),
                'call_count': self.call_count,
                'scope': 'this coupler instance only',
                'force_calculation': 'unchanged original IPC _apply_abd_coupling_forces',
                'buffer_transfer': 'new applied raw delta to coupling; restore previous applied',
                'clear_semantics': 'native rigid integrator clears coupling each substep',
                'limitations': ['fresh scene only; no rewind/checkpoint/backpropagation',
                                'does not remove the explicit strength / substep_dt**2 scaling',
                                'does not calibrate virtual tactile forces']}


def install_ipc_substep_reaction(coupler):
    """Install once after build and return the scene-local provenance handle."""
    return IPCSubstepReaction(coupler)
