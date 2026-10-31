"""Scene-local IPC import fixes; preserve collision surfaces and original assets."""
from types import MethodType
import numpy as np
import trimesh


def weld_ipc_triangle_soups(coupler, initial_robot=None, initial_joint_positions=None):
    """Repair IPC topology, then restore Genesis arrays.

    Genesis convex collision meshes can be triangle soups with repeated vertices.
    IPC needs shared edge topology to compute ABD volumes. Some decimated convex
    finger meshes are missing one triangular face. Fill only simple holes using
    existing vertices; never convexify, scale, or disable a contact surface.
    """
    original = coupler._add_rigid_geoms_to_ipc
    records = []

    def add(self):
        saved = []
        try:
            if initial_robot is not None:
                # Initialize both worlds consistently BEFORE creating IPC bodies.
                # Runtime motion remains ordinary PD; this is initial state only.
                indices=[initial_robot.get_joint(n).dofs_idx[0] for n in initial_joint_positions]
                self.rigid_solver.set_dofs_position(list(initial_joint_positions.values()),indices)
            for geom in self.rigid_solver.geoms:
                if not geom.n_verts or len(geom.init_faces) == 0:
                    continue
                mesh = trimesh.Trimesh(geom.init_verts,geom.init_faces,process=False)
                before=len(mesh.vertices)
                mesh.merge_vertices(digits_vertex=12)
                before_faces=len(mesh.faces)
                if not mesh.is_watertight and geom.is_convex:
                    trimesh.repair.fill_holes(mesh)
                if not mesh.is_watertight:
                    raise ValueError(f'IPC collision mesh remains open: {geom.link.name}, geom {geom.idx}')
                mesh.fix_normals(multibody=True)
                saved.append((geom,geom._init_verts,geom._init_faces))
                geom._init_verts=np.asarray(mesh.vertices,dtype=geom.init_verts.dtype)
                geom._init_faces=np.asarray(mesh.faces,dtype=geom.init_faces.dtype)
                records.append({'geom':geom.idx,'link':geom.link.name,
                    'before_vertices':before,'welded_vertices':len(mesh.vertices),
                    'faces':len(mesh.faces),'filled_triangles':len(mesh.faces)-before_faces,'watertight':True})
            original()
        finally:
            for geom,verts,faces in saved:
                geom._init_verts=verts;geom._init_faces=faces

    coupler._add_rigid_geoms_to_ipc=MethodType(add,coupler)
    return records
