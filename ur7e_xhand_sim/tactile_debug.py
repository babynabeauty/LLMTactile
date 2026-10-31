"""Read-only sensor/contact internals, never included in policy observations."""
import numpy as np
from genesis.utils.misc import qd_to_numpy

def snapshot(sim):
    observation=sim.observe(full=True)
    solver=sim.scene.rigid_solver
    meta=next(iter(sim.sensors.values()))._shared_metadata
    state=solver.collider.collider_state
    n=int(np.asarray(qd_to_numpy(state.n_contacts)).reshape(-1)[0])
    contacts={k:qd_to_numpy(getattr(state.contact_data,k))[:n].tolist() for k in ('geom_a','geom_b','link_a','link_b')}
    mapping=qd_to_numpy(state.contact_sort_idx)[:n].reshape(-1).astype(int)
    mapped={k:qd_to_numpy(getattr(state.contact_data,k))[mapping].tolist() for k in ('geom_a','geom_b','link_a','link_b')}
    counts=sim.array(meta.sensor_n_geoms)
    geom_ids=sim.array(meta.sensor_geoms_idx)
    pref=[geom_ids[0,i,:int(counts[0,i])].tolist() for i in range(counts.shape[1])]
    fingers={}
    for layout in sim.layouts:
        link=sim.robot.get_link(layout['link_name'])
        fingers[layout['finger']]={'global_link_idx':link.idx,'position_m':sim.values(link.get_pos()),'quaternion_wxyz':sim.values(link.get_quat()),'probe_local_m':layout['positions_link_m']}
    return {'step':sim.steps,'observation':observation,'evaluation':sim.evaluate(),
        'raw_contact_count':n,'raw_contact_indices':contacts,
        'contact_sort_idx':mapping.tolist(),'mapped_contact_indices':mapped,
        'sensor_links':sim.array(meta.links_idx).tolist(),'prefilter_geom_ids':pref,
        'filter_links_idx':sim.array(meta.filter_links_idx).tolist(),
        'geoms':[{'idx':g.idx,'link_idx':g.link.idx,'link_name':g.link.name} for g in solver.geoms],
        'fingers':fingers}
