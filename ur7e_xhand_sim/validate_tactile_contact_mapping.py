"""Offline frozen-state regression; never connects to a running simulation/server."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

from environment import Sandbox
import genesis as gs
from genesis.engine.sensors import kinematic_tactile as fixed
from genesis.utils.misc import qd_to_torch
from genesis.engine.solvers.rigid import rigid_solver

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/tactile_contact_audit_20261024'
data = json.loads((OUT / 'live_debug.json').read_text())
spec = importlib.util.spec_from_file_location('before_prefilter', OUT / 'before_prefilter.py')
before = importlib.util.module_from_spec(spec)
spec.loader.exec_module(before)
sim = Sandbox(render=False, scenario='small_clutter', output_dir=OUT / 'offline_runtime')
solver = sim.scene.rigid_solver
state = solver.collider.collider_state
m = next(iter(sim.sensors.values()))._shared_metadata
sim.robot.set_dofs_position([data['observation']['joint_positions_rad'][n] for n in sim.names], sim.indices, zero_velocity=True)
sim.cube.set_pos(data['evaluation']['cube_position_m'])
sim.cube.set_quat(data['evaluation']['cube_quaternion_wxyz'])
rigid_solver.kernel_update_geom_aabbs(solver.geoms_init_AABB, solver.dyn_state, solver.rigid_config)

def put(field, values):
    dest = qd_to_torch(field, copy=False)
    dest.copy_(torch.as_tensor(values, dtype=dest.dtype, device=dest.device))

def load_contacts(identity=False, reverse=False, empty=False):
    n = data['raw_contact_count']
    permutation = data['contact_sort_idx']
    for key, vals in data['raw_contact_indices'].items():
        field = getattr(state.contact_data, key)
        arr = np.full(tuple(qd_to_torch(field).shape), -1, dtype=np.int32)
        arr[:n] = vals
        arr[permutation] = data['mapped_contact_indices'][key]
        if identity:
            arr[:n] = data['mapped_contact_indices'][key]
        put(field, arr)
    arr = np.zeros(tuple(qd_to_torch(state.contact_sort_idx).shape), dtype=np.int32)
    order = list(range(n)) if identity else permutation
    arr[:n, 0] = order[::-1] if reverse else order
    put(state.contact_sort_idx, arr)
    put(state.n_contacts, [0 if empty else n])

def prefilter(module, filter_links=None):
    filt = m.filter_links_idx if filter_links is None else filter_links
    geoms = torch.zeros_like(m.sensor_geoms_idx)
    ng = torch.zeros_like(m.sensor_n_geoms)
    contacts = torch.zeros_like(m.sensor_contacts_idx)
    nc = torch.zeros_like(m.sensor_n_contacts)
    module._kernel_build_sensor_geom_idx(m.links_idx, filt, geoms, ng, state)
    module._kernel_build_sensor_contact_idx(m.links_idx, filt, contacts, nc, state)
    result = {
        'geoms': [geoms[0,i,:int(ng[0,i])].cpu().tolist() for i in range(len(m.links_idx))],
        'contacts': [contacts[0,i,:int(nc[0,i])].cpu().tolist() for i in range(len(m.links_idx))],
    }
    return result, geoms, ng

def forces(geoms, counts):
    gt = torch.zeros((sum(m.cache_sizes), 1), dtype=gs.tc_float, device=gs.device)
    measured = torch.zeros_like(gt)
    fixed._kernel_kinematic_taxel(m.probe_sensor_idx, m.links_idx, m.sensor_cache_start,
        m.sensor_probe_start, geoms, m.probe_positions, m.probe_radii, m.probe_radii_noise,
        m.probe_gains, m.normal_stiffness, m.normal_damping, m.normal_exponent,
        m.shear_scalar, m.twist_scalar, m.n_probes_per_sensor, counts, gt, measured,
        solver.dyn_state, solver.dyn_info, solver.rigid_info, solver.collider.collider_info,
        solver.collider.collider_config, gs.EPS, 1)
    result = {}
    for i, name in enumerate(sim.sensors):
        start, n = int(m.sensor_cache_start[i]), int(m.n_probes_per_sensor[i])
        values = measured[start:start+n*3,0].reshape(n,3).cpu().numpy()
        norms = np.linalg.norm(values, axis=1)
        result[name] = {'peak_N': float(norms.max()),
            'active_ids': np.flatnonzero(norms > 1e-8).tolist(), 'forces_N': values.tolist()}
    return result

load_contacts()
old, old_geoms, old_counts = prefilter(before)
new, new_geoms, new_counts = prefilter(fixed)
old_forces, new_forces = forces(old_geoms, old_counts), forces(new_geoms, new_counts)
assert old['geoms'][:2] == [[], []]
assert new['geoms'][:2] == [[44], [44]]
assert sorted(new['contacts'][0] + new['contacts'][1]) == [21,22,23]
assert old_forces['thumb']['peak_N'] == old_forces['index']['peak_N'] == 0
assert new_forces['thumb']['peak_N'] > 0 and new_forces['index']['peak_N'] > 0

load_contacts(identity=True)
identity_old = prefilter(before)[0]
identity_new = prefilter(fixed)[0]
assert identity_old == identity_new
load_contacts(reverse=True)
reverse_new = prefilter(fixed)[0]
for key in ('geoms', 'contacts'):
    assert [sorted(x) for x in reverse_new[key]] == [sorted(x) for x in new[key]]
load_contacts(empty=True)
empty_new = prefilter(fixed)[0]
assert all(not x for values in empty_new.values() for x in values)
load_contacts()
filtered_new = prefilter(fixed, torch.full_like(m.filter_links_idx, 23))[0]
assert all(not x for values in filtered_new.values() for x in values)

report = {'purpose':'Frozen state sensor regression, not a grasp attempt; no scene.step calls',
    'source_step':data['step'], 'before':old, 'after':new,
    'before_forces':old_forces, 'after_forces':new_forces,
    'checks':{'captured_sparse_mapping':True, 'identity_mapping':True,
        'permutation_invariance':True, 'empty_contacts':True, 'counterpart_filter':True,
        'deduplication':True, 'both_contact_and_geom_kernels':True},
    'notes':'Restored recorded joint and cube poses; velocities zero. Force values are static frozen-state diagnostics, not the original dynamic reading.'}
(OUT / 'validation.json').write_text(json.dumps(report, indent=2))
print(json.dumps({'checks':report['checks'], 'before':old, 'after':new,
    'forces_after':{k:{'peak_N':v['peak_N'],'active_ids':v['active_ids']} for k,v in new_forces.items()}},indent=2),flush=True)
gs.destroy()
