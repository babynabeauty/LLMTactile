"""Read-only IPC contact audit, independent of the virtual XHand taxel model.

ContactSystemFeature exports the solver's stored incremental-potential gradient.
Its conversion to force is -gradient / IPC_substep_dt**2.  This is the last
assembled Newton gradient, not a fresh force solve at the retrieved final pose.
The falling-block validation checks normal force sign, units and ownership.
Tangential exports fail horizontal-load balance at Newton tolerances .005, but
pass at 1e-5 with a 1/600 s IPC step.  The reference-validity fields are explicitly
conditioned on that tested setting; they do not guarantee every dynamic contact.
Never use energy.topo to pair these rows: its ordering can differ from gradient.i.
"""
from __future__ import annotations

import numpy as np


_WIDTH = {'PT': 4, 'EE': 4, 'PE': 3, 'PP': 2}


def _empty():
    return {'normal_force_on_soft_N': np.zeros(3),
            'friction_force_on_soft_N': np.zeros(3),
            'normal_primitive_load_N': 0., 'friction_primitive_load_N': 0.,
            'normal_primitive_count': 0, 'friction_primitive_count': 0}


def _finish(value):
    out = dict(value)
    out['force_on_soft_N'] = (value['normal_force_on_soft_N'] + value['friction_force_on_soft_N']).tolist()
    out['net_force_magnitude_N'] = float(np.linalg.norm(out['force_on_soft_N']))
    # Normal and friction contributions need not share primitive ordering.
    # Their sum of magnitudes is explicitly an upper-bound load, not net force.
    out['contact_load_upper_bound_N'] = value['normal_primitive_load_N'] + value['friction_primitive_load_N']
    out['normal_force_on_soft_N'] = value['normal_force_on_soft_N'].tolist()
    out['friction_force_on_soft_N'] = value['friction_force_on_soft_N'].tolist()
    return out


def aggregate_gradient(primitive, indices, gradients, owners, soft_owner, dt):
    """Return per-counterpart forces from ordered, unmerged IPC gradient rows.

    ``owners`` maps global vertex IDs to semantic body IDs; -1 is unmapped.
    Only primitives containing the selected soft body and one other body count.
    Self-contact, other objects and purely rigid contacts cannot contaminate it.
    """
    kind, component = primitive.split('+')
    if kind not in _WIDTH or component not in ('N', 'F'):
        raise ValueError(f'Unsupported pair primitive {primitive}')
    width = _WIDTH[kind]
    indices = np.asarray(indices, dtype=np.int64).reshape(-1, width)
    gradients = np.asarray(gradients, dtype=float).reshape(-1, width, 3)
    if len(indices) != len(gradients) or not np.isfinite(gradients).all():
        raise ValueError('Invalid IPC contact gradient')
    if indices.size and (indices.min() < 0 or indices.max() >= len(owners)):
        raise ValueError('Unmapped IPC global vertex ID')
    result = {}; balance = 0.; ignored = 0
    category = 'normal' if component == 'N' else 'friction'
    for ids, grad in zip(indices, gradients):
        body = owners[ids]
        selected = body == soft_owner
        if not selected.any() or selected.all():
            ignored += 1
            continue
        others = np.unique(body[~selected])
        if len(others) != 1 or others[0] < 0:
            raise ValueError(f'IPC primitive spans unknown/multiple counterpart bodies: {body}')
        force = -grad[selected].sum(axis=0) / dt**2
        err = float(np.linalg.norm(grad.sum(axis=0)) / dt**2)
        balance = max(balance, err)
        entry = result.setdefault(int(others[0]), _empty())
        entry[f'{category}_force_on_soft_N'] += force
        entry[f'{category}_primitive_load_N'] += float(np.linalg.norm(force))
        entry[f'{category}_primitive_count'] += int(np.linalg.norm(force) > 1e-12)
    return result, balance, ignored


class IPCContactAudit:
    """Inspect a built single-environment IPC scene without modifying its state.

    ``finger_links`` maps finger names to one link or a collection of links.
    Include all phalanges to measure whole-finger support, or only sensor-bearing
    links when an explicitly fingertip-only support metric is desired.
    """
    def __init__(self, scene, soft, robot=None, table=None, finger_links=None,
                 env_index=0, support_threshold_N=.05):
        import uipc
        from uipc.backend import SceneVisitor
        from uipc.geometry import SimplicialComplexSlot, Geometry
        from genesis.engine.couplers.ipc_coupler.utils import read_ipc_geometry_metadata

        self.scene = scene; self.soft = soft; self.robot = robot; self.table = table
        self.coupler = scene.sim.coupler
        self.dt = float(scene.sim.substep_dt)
        self.solver_tolerances = {name: float(getattr(self.coupler.options, name))
                                 for name in ('newton_tolerance', 'newton_translation_tolerance')}
        self.friction_reference_validated = bool(
            np.isclose(self.dt, 1/600, rtol=0, atol=1e-12)
            and all(np.isclose(v, 1e-5, rtol=0, atol=1e-12) for v in self.solver_tolerances.values()))
        self.threshold = float(support_threshold_N)
        if not self.dt > 0 or self.threshold < 0:
            raise ValueError('Invalid contact audit time step or support threshold')
        world = self.coupler._ipc_world
        self.feature = world.features().find(uipc.core.ContactSystemFeature)
        if self.feature is None:
            raise RuntimeError('IPC ContactSystemFeature is unavailable')
        self.types = tuple(self.feature.contact_primitive_types())
        self.buffers = {kind: Geometry() for kind in self.types}
        self.records = []; self.soft_owner = None; self.links = {}
        for slot in SceneVisitor(self.coupler._ipc_scene).geometries():
            if not isinstance(slot, SimplicialComplexSlot):
                continue
            geo = slot.geometry(); meta = read_ipc_geometry_metadata(geo)
            if meta is None or meta[1] != env_index:
                continue
            attr = geo.meta().find(uipc.builtin.global_vertex_offset)
            if attr is None:
                raise RuntimeError('IPC geometry has no global vertex offset')
            start = int(np.asarray(attr.view()).reshape(-1)[0])
            count = int(geo.vertices().size())
            owner = len(self.records)
            rec = dict(owner=owner, solver_type=meta[0], index=meta[2],
                       global_vertex_start=start, vertex_count=count)
            if meta[0] == 'rigid':
                link = scene.rigid_solver.links[meta[2]]
                self.links[owner] = link
                rec.update(link_name=link.name, entity_idx=int(link.entity.idx))
            elif meta[0] == 'fem' and scene.fem_solver.entities[meta[2]] is soft:
                if self.soft_owner is not None:
                    raise RuntimeError('Multiple target FEM geometries')
                self.soft_owner = owner
                fem_offset = int(np.asarray(geo.meta().find(uipc.builtin.backend_fem_vertex_offset).view()).reshape(-1)[0])
                self.fem_feature = world.features().find(uipc.core.FiniteElementStateAccessorFeature)
                self.fem_buffer = self.fem_feature.create_geometry(fem_offset, count)
                self.fem_buffer.vertices().create(uipc.builtin.position, np.zeros(3))
                self.fem_buffer.vertices().create(uipc.builtin.velocity, np.zeros(3))
            self.records.append(rec)
        if self.soft_owner is None:
            raise RuntimeError('Target soft body is absent from the IPC world')
        self.owners = np.full(max(r['global_vertex_start']+r['vertex_count'] for r in self.records), -1, dtype=int)
        for r in self.records:
            section = slice(r['global_vertex_start'], r['global_vertex_start']+r['vertex_count'])
            if (self.owners[section] >= 0).any():
                raise RuntimeError('Overlapping IPC vertex ownership ranges')
            self.owners[section] = r['owner']
        self.groups = {'robot': {i for i, link in self.links.items() if link.entity is robot},
                       'table': {i for i, link in self.links.items() if link.entity is table}}
        self.finger_names = tuple((finger_links or {}).keys())
        for finger, links in (finger_links or {}).items():
            if hasattr(links, 'idx'):
                links = [links]
            ids = {link.idx for link in links}
            self.groups[finger] = {i for i, link in self.links.items() if link.idx in ids}

    def friction_readback(self):
        soft_id = self.coupler._ipc_fem_contacts[self.soft].id()
        out = {}
        for name, entity in (('robot', self.robot), ('table', self.table)):
            if entity is None:
                continue
            counterpart = self.coupler._ipc_abd_contacts[entity].id()
            model = self.coupler._ipc_contact_tabular.at(soft_id, counterpart)
            out[name] = {'friction': float(model.friction_rate()),
                         'enabled': bool(model.is_enabled()),
                         'element_ids': np.asarray(model.topo()).reshape(-1).tolist()}
        return out

    def soft_vertex_state(self):
        """Direct IPC positions and velocities; avoids stale Genesis FEM velocity."""
        import uipc
        self.fem_feature.copy_to(self.fem_buffer)
        return {name: np.array(self.fem_buffer.vertices().find(attribute).view()).reshape(-1, 3)
                for name, attribute in (('position_m', uipc.builtin.position),
                                        ('velocity_m_s', uipc.builtin.velocity))}

    def read(self):
        """Call after scene.step; samples the last IPC substep, not its average."""
        per_link = {}; balance = 0.; ignored = 0; plane_force = np.zeros(3)
        for kind, buffer in self.buffers.items():
            self.feature.contact_gradient(kind, buffer)
            ia = buffer.instances().find('i'); ga = buffer.instances().find('grad')
            if ia is None or ga is None:
                continue
            ids = np.asarray(ia.view()).reshape(-1)
            grad = np.asarray(ga.view()).reshape(-1, 3)
            if kind.startswith('PH+'):
                # Half-plane gradients contain only the moving vertex. This is
                # not a finger pair; preserve as explicitly unattributed support.
                if ids.size:
                    if ids.max() >= len(self.owners) or ids.min() < 0:
                        raise RuntimeError('Unknown half-plane contact vertex')
                    plane_force -= grad[self.owners[ids] == self.soft_owner].sum(0)/self.dt**2
                continue
            values, err, skipped = aggregate_gradient(kind, ids, grad, self.owners, self.soft_owner, self.dt)
            balance = max(balance, err); ignored += skipped
            for owner, values_i in values.items():
                target = per_link.setdefault(owner, _empty())
                for key, value in values_i.items():
                    target[key] += value
        groups = {}
        for name, owners in self.groups.items():
            total = _empty()
            for owner in owners:
                for key, value in per_link.get(owner, {}).items():
                    total[key] += value
            groups[name] = _finish(total)
        fingers = {name: groups[name] for name in self.finger_names}
        supported = {name: data['normal_primitive_load_N'] > self.threshold for name, data in fingers.items()}
        return {'groups': groups, 'fingers': fingers,
                'finger_order': list(self.finger_names),
                'finger_load_N': [fingers[name]['contact_load_upper_bound_N'] for name in self.finger_names],
                'finger_normal_load_N': [fingers[name]['normal_primitive_load_N'] for name in self.finger_names],
                'finger_net_force_world_N': [fingers[name]['force_on_soft_N'] for name in self.finger_names],
                'table_force_world_N': groups['table']['force_on_soft_N'],
                'robot_force_world_N': groups['robot']['force_on_soft_N'],
                'force_vector_convention': 'world frame; force exerted ON the target soft body',
                'load_definition': 'sum of normal primitive magnitudes plus sum of friction primitive magnitudes; an upper bound, not the rigid benchmark combined-force norm',
                'links': {self.records[i].get('link_name', f"{self.records[i]['solver_type']}_{self.records[i]['index']}"): _finish(data) for i, data in per_link.items()},
                'supported_fingers': supported,
                'thumb_index_supported': bool(supported.get('thumb', False) and supported.get('index', False)),
                'support_normal_load_threshold_N': self.threshold,
                'friction_readback': self.friction_readback(),
                'half_plane_force_on_soft_N': plane_force.tolist(),
                'max_pair_force_balance_error_N': balance, 'ignored_other_or_self_primitives': ignored,
                'ipc_substep_dt_s': self.dt,
                'source': 'IPC last-assembled normal/friction contact gradient divided by negative substep_dt squared',
                'normal_force_validated': True,
                'friction_force_validated': self.friction_reference_validated,
                'combined_force_validated': self.friction_reference_validated,
                'solver_tolerances': self.solver_tolerances,
                'validation_scope': 'Static falling-block reference only. Normal force verified at 2.5 ms and 1.667 ms. Tangential equilibrium passed at 1.667 ms with both Newton tolerances 1e-5; it failed at .005 because the stored gradient lagged the solution. Dynamic accuracy still requires scrutiny. Use normal_primitive_load_N for support.',
                'virtual_taxels_used_for_audit': False}
