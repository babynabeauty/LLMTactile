"""Tabletop layouts for the imported insertion assets (meters, wxyz quaternions)."""
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from insertion_assets import load_insertion_assets
from platform_asset import TABLE_TOP_Z, TABLETOP_XY_M


TASKS = ('grasp', 'tube_insert', 'usb_insert')


def rotation(quaternion):
    return Rotation.from_quat(np.asarray(quaternion)[[1, 2, 3, 0]]).as_matrix()


class ImportedObjectSpec:
    """Geometry in the source mesh frame; its origin need not be its center."""

    def __init__(self, task, asset):
        self.shape = task.removesuffix('_insert')
        self.asset = asset
        mesh = trimesh.load(asset['visual_mesh_path'], force='mesh', process=False)
        self.vertices_local_m = np.asarray(mesh.vertices)
        self.bounds_size_m = np.ptp(self.vertices_local_m, axis=0)
        self.volume_m3 = float(asset['volume_m3'])
        self.inertia_per_kg = np.asarray(asset['inertia_per_kg'])

    def vertical_extent(self, quaternion_wxyz):
        # evaluate() subtracts this from the body-frame origin's world height.
        return -float((self.vertices_local_m @ rotation(quaternion_wxyz).T)[:, 2].min())

    def protocol(self):
        return dict(shape=self.shape, geometry='imported_mesh',
                    bounds_size_m=self.bounds_size_m.tolist(), volume_m3=self.volume_m3,
                    inertia_per_kg_m2=self.inertia_per_kg.tolist(),
                    center_of_mass_local_m=self.asset['center_of_mass_m'],
                    visual_mesh_path=self.asset['visual_mesh_path'],
                    lowest_surface='minimum world Z of all transformed source mesh vertices',
                    frame='original source asset origin; not the bounding-box center')


class InsertionScene:
    def __init__(self, task):
        if task not in TASKS[1:]:
            raise ValueError(f'Unknown insertion task: {task}')
        self.task = task
        self.assets = load_insertion_assets(task)
        self.object_spec = ImportedObjectSpec(task, self.assets['movable'])
        self.quaternion = np.array([1., 0., 0., 0.])
        self.fixed_poses = {}
        for asset in self.assets['fixed']:
            xy = (-.50, -.1333) if asset['name'] == 'holder' else (-.50, -.36)
            self.fixed_poses[asset['name']] = dict(
                position_m=[*xy, TABLE_TOP_Z-float(asset['bounds_m'][0][2])],
                quaternion_wxyz=[1., 0., 0., 0.])
        if task == 'tube_insert':
            # Same support-to-tube offset as UniVTAC's create_actors, on our table.
            holder = self.fixed_poses['holder']['position_m']
            self.position = np.asarray(holder)+[0., 0., .002]
        else:
            # Start with the USB resting flat, requiring pickup before insertion.
            self.quaternion = Rotation.from_euler('y', 90, degrees=True).as_quat()[[3, 0, 1, 2]]
            verts = self.object_spec.vertices_local_m @ rotation(self.quaternion).T
            center_xy = (verts.min(0)+verts.max(0))[:2]/2
            self.position = np.r_[np.array([-.50, -.1333])-center_xy,
                                  TABLE_TOP_Z-verts[:, 2].min()+.0005]
        self._validate_table_bounds()

    def _validate_table_bounds(self):
        bounds = np.asarray(TABLETOP_XY_M)
        vertices = self.object_spec.vertices_local_m @ rotation(self.quaternion).T+self.position
        boxes = [np.stack([vertices.min(0), vertices.max(0)])]
        boxes.extend(np.asarray(asset['bounds_m'])+self.fixed_poses[asset['name']]['position_m']
                     for asset in self.assets['fixed'])
        for box in boxes:
            if np.any(box[0, :2] < bounds[:, 0]+.02) or np.any(box[1, :2] > bounds[:, 1]-.02):
                raise ValueError('Insertion asset must remain at least 2 cm inside the tabletop')

    def add_entities(self, scene, gs, friction):
        def add(asset, pos, quat, fixed):
            return scene.add_entity(
                gs.morphs.URDF(file=asset['urdf_path'], pos=pos, quat=quat, fixed=fixed,
                               **asset.get('morph_options', {})),
                material=gs.materials.Rigid(friction=friction,
                                            **asset.get('material_options', {})),
                surface=gs.surfaces.Default(smooth=False))
        movable = add(self.assets['movable'], self.position, self.quaternion, False)
        fixtures = {}
        for asset in self.assets['fixed']:
            pose = self.fixed_poses[asset['name']]
            fixtures[asset['name']] = add(asset, pose['position_m'], pose['quaternion_wxyz'], True)
        return movable, fixtures

    def describe(self):
        from insertion_geometry import insertion_geometry_spec, target_object_pose
        socket = self.fixed_poses['socket']
        return dict(task=self.task, units='meters', asset_scale=1.0,
                    initial_object_position_m=self.position.tolist(),
                    initial_object_quaternion_wxyz=self.quaternion.tolist(),
                    fixtures=self.fixed_poses, assets=self.assets,
                    insertion_geometry=insertion_geometry_spec(self.task),
                    goal_object_pose=target_object_pose(self.task, socket['position_m'], socket['quaternion_wxyz']),
                    phases=['approach', 'grasp', 'lift', 'align', 'insert'],
                    policy_status='scene and geometric evaluation only; insertion demonstrations and policy pending')

    def evaluate(self, sim):
        from insertion_geometry import evaluate_insertion_pose
        socket = sim.task_fixtures['socket']
        result = evaluate_insertion_pose(
            self.task, sim.values(sim.cube.get_pos()), sim.values(sim.cube.get_quat()),
            sim.values(socket.get_pos()), sim.values(socket.get_quat()))
        contacts = {k: sim.array(v) for k, v in sim.cube.get_contacts().items()}
        links = {link.idx: link for link in sim.scene.rigid_solver.links}
        entities = {sim.cube.idx: self.object_spec.shape, sim.robot.idx: 'ur7e_xhand', sim.table.idx: 'table'}
        entities.update({entity.idx: name for name, entity in sim.task_fixtures.items()})
        valid = contacts.get('valid_mask', np.ones(contacts['link_a'].shape, dtype=bool)).reshape(-1)
        rows = []
        for i in np.flatnonzero(valid):
            a = int(contacts['link_a'].reshape(-1)[i])
            b = int(contacts['link_b'].reshape(-1)[i])
            rows.append(dict(entity_a=entities.get(links[a].entity.idx, 'environment'),
                             entity_b=entities.get(links[b].entity.idx, 'environment'),
                             force_on_a_world_N=contacts['force_a'].reshape(-1, 3)[i].tolist(),
                             position_world_m=contacts['position'].reshape(-1, 3)[i].tolist()))
        result['object_contacts'] = rows
        return result
