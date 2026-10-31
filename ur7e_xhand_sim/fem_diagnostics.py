"""Read-only FEM diagnostics and direct boundary rendering for the silicone prototype."""
import numpy as np
import quadrants as qd
import genesis as gs
from genesis.utils import sdf
from genesis.utils import array_class


def skip_inactive_pcg_iterations(solver):
    """Skip only no-op iterations after the solver's EXISTING convergence flags clear.

    This changes neither residual tolerance nor iteration budget. Every active
    iteration still uses the unmodified Genesis kernel. Installed per scene.
    """
    from types import MethodType
    stats = {'solves':0, 'kernel_iterations':0, 'max_kernel_iterations':0,
             'unconverged_at_budget':0}

    def pcg_solve(self):
        self.init_pcg_solve()
        count = 0
        active = bool(self.batch_pcg_active.to_numpy().any())
        while active and count < self._n_pcg_iterations:
            for _ in range(min(8, self._n_pcg_iterations-count)):
                self.one_pcg_iter()
                count += 1
            active = bool(self.batch_pcg_active.to_numpy().any())
        stats['solves'] += 1
        stats['kernel_iterations'] += count
        stats['max_kernel_iterations'] = max(stats['max_kernel_iterations'],count)
        stats['unconverged_at_budget'] += int(active)

    solver.pcg_solve = MethodType(pcg_solve, solver)
    return stats


def use_physics_surface(entity):
    """Before scene.build, render the actual tet boundary without changing any tet."""
    import trimesh
    from genesis.engine.entities.fem_entity import FEMVisGeom
    old_vertices=entity.n_vverts;old_faces=entity.n_vfaces
    ids,tri=np.unique(entity._surface_tri_np,return_inverse=True)
    rest=entity.init_positions.detach().cpu().numpy()
    mesh=gs.Mesh.from_trimesh(trimesh.Trimesh(rest[ids],tri.reshape(-1,3),process=False),surface=entity.surface)
    entity._vgeoms=gs.List([FEMVisGeom(entity,entity._vvert_start,entity._vface_start,mesh,ids.astype(gs.np_int))])
    return {'old_render_vertices':old_vertices,'old_render_triangles':old_faces,
            'render_vertices':len(ids),'render_triangles':tri.size//3,
            'physical_boundary_vertices':len(ids),'physics_mesh_unchanged':True}


def mesh_quality(rest,tets):
    x=rest[tets];D=np.stack([x[:,i]-x[:,3] for i in range(3)],axis=-1)
    volume=np.abs(np.linalg.det(D))/6
    edges=np.stack([np.linalg.norm(x[:,i]-x[:,j],axis=1) for i in range(4) for j in range(i)],axis=1)
    q=12*np.power(3*volume,2/3)/(edges**2).sum(1)
    return {'vertices':len(rest),'tetrahedra':len(tets),'volume_m3':float(volume.sum()),
        'minimum_tet_volume_m3':float(volume.min()),'minimum_edge_m':float(edges.min()),
        'mean_ratio_quality_percentiles':dict(zip(['min','p01','p05','p50','p95','max'],np.percentile(q,[0,1,5,50,95,100]).tolist())),
        'quality_below_0_1_count':int((q<.1).sum()),'rest_matrix_condition_max':float(np.linalg.cond(D).max()),
        'degenerate_tet_count':int((volume<=1e-16).sum())}


@qd.kernel
def _surface_sdf(points:qd.types.ndarray(),minimum:qd.types.ndarray(),nearest:qd.types.ndarray(),
                 geoms:array_class.GeomsState,info:array_class.GeomsInfo,sdf_info:array_class.SDFInfo,
                 n_geoms:qd.i32):
    for i in range(points.shape[0]):
        p=qd.Vector([points[i,0],points[i,1],points[i,2]])
        best=gs.qd_float(1000.0);idx=-1
        for g in range(n_geoms):
            if info.needs_coup[g]:
                d=sdf.sdf_func_world(g,0,p,geoms,info,sdf_info)
                if d<best:best=d;idx=g
        minimum[i]=best;nearest[i]=idx


def contact_diagnostics(sim,positions):
    import torch
    r=sim.scene.rigid_solver
    ids=np.unique(sim.silicone._surface_tri_np)
    points=torch.tensor(positions[ids],dtype=gs.tc_float,device=gs.device)
    distance=torch.empty(len(ids),dtype=gs.tc_float,device=gs.device)
    nearest=torch.empty(len(ids),dtype=gs.tc_int,device=gs.device)
    _surface_sdf(points,distance,nearest,r.dyn_state.geoms,r.dyn_info.geoms,r.collider.collider_info.sdf,r.n_geoms)
    d=distance.cpu().numpy();g=nearest.cpu().numpy()
    groups={}
    for i in np.unique(g[d<.001]):
        geo=r.geoms[int(i)];sel=(g==i)&(d<.001)
        groups[str(int(i))]={'link':geo.link.name,'entity_idx':geo.entity.idx,
            'near_surface_vertices':int(sel.sum()),'penetrating_vertices':int(((d<0)&sel).sum()),
            'min_signed_distance_m':float(d[sel].min())}
    return {'minimum_signed_distance_m':float(d.min()),'nearby_geometries':groups}
