"""UR7e + XHAND1 rigid-contact development scene (no hardware connection)."""
from pathlib import Path
import io
import json
import os
import sys

os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
os.environ.setdefault('OMP_NUM_THREADS', '8')
import numpy as np
from scipy.spatial.transform import Rotation
from PIL import Image
import genesis as gs

from build_assets import HERE, ARM_JOINTS, HAND_JOINTS, build
from cameras import load_camera_views, camera_kwargs
from platform_asset import build_platform, PLATFORM_POSITION, FLOOR_Z, TABLE_TOP_Z
from residual_rl.multiobject_geometry import ObjectSpec
from residual_rl.solver_profile import normalize_solver_profile, read_solver_readback, validate_solver_readback
from residual_rl.tactile_provenance import current_tactile_model
sys.path.insert(0, str(HERE.parent/'astra_bridge'))
from protocol import validate_step


class Sandbox:
    def __init__(self, output_dir=None, render=True, test_fixture=False, scenario='original',
                 cube_density=None, cube_mass_kg=None, cube_friction=.8, robot_friction=.8,
                 silicone=False, silicone_substeps=None, silicone_contact_softness=.002,
                 silicone_skip_idle_pcg=True, silicone_solver='ipc', silicone_maxvolume=1e-7,
                 object_spec=None, physics_dt_s=.005, control_substeps=5,
                 noslip_iterations=0, noslip_tolerance=1e-6, deterministic=False,
                 table_contact_force_limit_N=5.0, overview=False):
        from contact_guard import validate_table_force_limit
        self.table_contact_force_limit_N=validate_table_force_limit(table_contact_force_limit_N)
        if (isinstance(physics_dt_s,bool) or not np.isfinite(physics_dt_s)
                or not .001 <= physics_dt_s <= .01
                or type(control_substeps) is not int or not 1 <= control_substeps <= 20):
            raise ValueError('Invalid explicit simulation timing')
        self.physics_dt_s=float(physics_dt_s)
        self.control_substeps=control_substeps
        self.control_dt_s=self.physics_dt_s*self.control_substeps
        self.solver_profile=normalize_solver_profile({'schema':'genesis_rigid_solver_profile_v1',
            'noslip_iterations':noslip_iterations,'noslip_tolerance':noslip_tolerance})
        self.object_spec = ObjectSpec.from_value('cube' if object_spec is None else object_spec)
        self.legacy_object_placement = object_spec is None
        if object_spec is not None and silicone:
            raise ValueError('The multi-object study is rigid-only; silicone cannot be enabled')
        if silicone_solver not in ('legacy','ipc'):
            raise ValueError('Unknown silicone solver')
        ipc = silicone and silicone_solver == 'ipc'
        if silicone_substeps is None:
            silicone_substeps = 2 if ipc else 20
        if silicone_substeps < 1:
            raise ValueError('silicone_substeps must be positive')
        if cube_density is not None and cube_mass_kg is not None:
            raise ValueError('Specify cube density or mass, not both')
        density = (float(cube_mass_kg)/self.object_spec.volume_m3 if cube_mass_kg is not None
                   else float(cube_density) if cube_density is not None else 500.)
        if not np.isfinite(density) or density <= 0:
            raise ValueError('Cube density/mass must be finite and positive')
        if any(not np.isfinite(v) or not .01 <= v <= 5 for v in (cube_friction,robot_friction)):
            raise ValueError('Friction must be finite and between .01 and 5')
        self.material_parameters = {'cube_size_m':self.object_spec.bounds_size_m.tolist(),'cube_density_kg_m3':density,
            'cube_mass_kg':density*self.object_spec.volume_m3,'cube_friction':float(cube_friction),
            'robot_friction':float(robot_friction),'table_friction':.8,
            'mass_input_kg':cube_mass_kg,'density_input_kg_m3':cube_density,
            'object_shape':self.object_spec.shape,'object_volume_m3':self.object_spec.volume_m3,
            'legacy_cube_field_aliases':'cube_* material keys describe the selected rigid object'}
        self.output_dir = Path(output_dir or HERE/'runtime')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.render_enabled = render
        model, self.layouts = build()
        gs.init(backend=gs.gpu, precision='32', seed=42, logging_level='warning',
                **({'use_deterministic_algorithms':True} if deterministic else {}))
        self.silicone = None
        self.silicone_parameters = None
        coupling = gs.options.IPCCouplerOptions(contact_d_hat=.0005,
            newton_tolerance=.005, newton_translation_tolerance=.005,
            linear_system_tolerance=1e-6, n_linesearch_iterations=32,
            enable_rigid_ground_contact=False,enable_rigid_rigid_contact=False,
            enable_rigid_dofs_sync=True) if ipc else gs.options.LegacyCouplerOptions()
        self.scene = gs.Scene(sim_options=gs.options.SimOptions(dt=self.physics_dt_s, substeps=silicone_substeps if silicone else 1),
            coupler_options=coupling,
            fem_options=gs.options.FEMOptions(use_implicit_solver=True, damping_alpha=5., damping_beta=5e-4),
            rigid_options=gs.options.RigidOptions(enable_self_collision=True, enable_joint_limit=True,
                                                 enable_neutral_collision=True,
                                                 noslip_iterations=self.solver_profile['noslip_iterations'],
                                                 noslip_tolerance=self.solver_profile['noslip_tolerance']),
            vis_options=gs.options.VisOptions(background_color=(.10,.13,.18),ambient_light=(.5,.5,.5)),
            show_viewer=False)
        self.scene.add_entity(gs.morphs.Plane(pos=(0,0,FLOOR_Z)),surface=gs.surfaces.Default(color=(.22,.25,.30,1)))
        platform_model, self.platform_metadata = build_platform()
        self.table = self.scene.add_entity(gs.morphs.URDF(file=str(platform_model),pos=PLATFORM_POSITION,fixed=True),
            material=gs.materials.Rigid(friction=.8,coup_friction=.8,coup_softness=silicone_contact_softness),surface=gs.surfaces.Default())
        self.robot = self.scene.add_entity(gs.morphs.URDF(file=str(model),fixed=True,pos=(0,0,.2),
            merge_fixed_links=True,links_to_keep=['right_hand_link','tool0']),
            material=gs.materials.Rigid(friction=robot_friction,coup_friction=robot_friction,coup_softness=silicone_contact_softness,
                **({'coup_type':'two_way_soft_constraint'} if ipc else {})))
        offsets={'original':(0,0,0),'small':(.03,.02,0),'large':(.15,.10,0),'clutter':(0,0,0),'small_clutter':(.03,.02,0)}
        if scenario not in offsets:raise ValueError('Unknown scenario')
        position=self._initial_object_position(offsets[scenario])
        self.scenario={'name':scenario,'target_initial_position_m':position.tolist(),'translation_m':offsets[scenario],'distractors':[]}
        self.scenario['object_spec']=self.object_spec.protocol()
        self.cube = self.scene.add_entity(self._object_morph(position),
            material=gs.materials.Rigid(rho=density,friction=cube_friction,
                **({'coup_type':'two_way_soft_constraint'} if ipc else {})),surface=gs.surfaces.Default(color=(.95,.36,.10,1)))
        self.distractors={}
        if scenario in ('clutter','small_clutter'):
            specs=[('blue_box','box',(.12,.12),(.1,.25,.9,1)),
                ('green_box','box',(-.12,.10),(.15,.8,.22,1)),
                ('purple_box','box',(-.14,-.10),(.65,.15,.8,1)),
                ('cyan_ball','ball',(.13,-.12),(.1,.8,.85,1)),
                ('red_ball','ball',(-.02,.16),(.9,.08,.1,1)),
                ('yellow_ball','ball',(-.02,-.17),(.95,.85,.1,1))]
            for name,shape,xy,color in specs:
                pos=position+np.array([*xy,0])
                morph=gs.morphs.Box(size=(.045,)*3,pos=pos) if shape=='box' else gs.morphs.Sphere(radius=.0225,pos=pos)
                self.distractors[name]=self.scene.add_entity(morph,material=gs.materials.Rigid(rho=500,friction=.8,
                    **({'coup_type':'two_way_soft_constraint'} if ipc else {})),surface=gs.surfaces.Default(color=color))
                self.scenario['distractors'].append({'name':name,'shape':shape,'position_m':pos.tolist(),'color':color})
        if silicone:
            self.silicone_parameters = {
                'size_m':[.045]*3, 'initial_position_m':(position+np.array([.18,-.18,.01])).tolist(),
                'young_modulus_Pa':1e5, 'poisson_ratio':.45, 'density_kg_m3':1100.,
                'nominal_mass_kg':1100.*.045**3, 'model':'stable_neohookean' if ipc else 'linear_corotated',
                'coupler':'IPCCoupler' if ipc else 'LegacyCoupler', 'rigid_surface_coupling_friction':.8,
                'calibrated_to_real_silicone':False, 'tactile_supported':False,
                'physics_substep_s':.005/silicone_substeps,'contact_softness_m':None if ipc else silicone_contact_softness,
                'tet_maxvolume_m3':silicone_maxvolume,'coupler_options':coupling.model_dump()}
            self.silicone = self.scene.add_entity(
                gs.morphs.Box(size=(.045,)*3,pos=self.silicone_parameters['initial_position_m'],
                              maxvolume=silicone_maxvolume,nobisect=False),
                material=gs.materials.FEM.Elastic(E=1e5,nu=.45,rho=1100.,
                                                 model=self.silicone_parameters['model'],friction_mu=.8),
                surface=gs.surfaces.Default(color=(.95,.40,.65,1)))
            from fem_diagnostics import use_physics_surface
            self.silicone_parameters['surface_rendering']=use_physics_surface(self.silicone)
        self.fixture = None
        if test_fixture:
            self.fixture = self.scene.add_entity(gs.morphs.Sphere(radius=.006,pos=(2,2,2),fixed=True),
                surface=gs.surfaces.Default(color=(.2,.8,.7,1)))
        self.sensors = {}
        for layout in self.layouts:
            link = self.robot.get_link(layout['link_name'])
            self.sensors[layout['finger']] = self.scene.add_sensor(gs.sensors.KinematicTaxel(
                entity_idx=self.robot.idx,link_idx_local=link.idx_local,
                probe_local_pos=layout['positions_link_m'],probe_radius=.0015,history_length=5,
                normal_stiffness=500.,normal_damping=1.,normal_exponent=1.2,
                shear_scalar=2.,twist_scalar=0.,draw_debug=False))
        self.camera_views,self.camera_source=load_camera_views()
        self.camera=self.scene.add_camera(**camera_kwargs(self.camera_views['scene_front']))
        self.rear_camera=self.scene.add_camera(**camera_kwargs(self.camera_views['scene_left']))
        self.wrist_camera=self.scene.add_camera(**camera_kwargs(self.camera_views['wrist']))
        self.wrist_camera_mount=self.camera_views['wrist']
        # Offsets and up vectors live in the palm frame (+X palm normal, +Z fingers).
        # Update from measured link pose, including rotation, on every rendered frame.
        self.hand_view_specs = {
            'palm': ((.42,0,.08),(0,0,1)),
            'back': ((-.42,0,.12),(0,0,1)),
            'thumb': ((.14,.42,.10),(-1,0,0)),
            'little': ((.14,-.42,.10),(-1,0,0)),
        }
        self.hand_cameras = {name:self.scene.add_camera(res=(480,480),
            pos=(1,1,1),lookat=(0,0,0),fov=50,GUI=False) for name in self.hand_view_specs}
        self.close_camera = self.hand_cameras['thumb']
        if ipc:
            from ipc_support import weld_ipc_triangle_soups
            initial=dict(zip(ARM_JOINTS+HAND_JOINTS,np.r_[np.deg2rad([0.,-89.95,89.96,-89.96,-89.95,-44.98]),np.zeros(12)]))
            self.silicone_parameters['ipc_mesh_import']=weld_ipc_triangle_soups(self.scene.sim.coupler,self.robot,initial)
        if overview:
            self.overview_camera=self.scene.add_camera(res=(1200,900),
                pos=(-2.1,-2.5,1.7),lookat=(-.12,-.2,.15),fov=48,GUI=False)
        self.scene.build(n_envs=1)
        self.solver_readback=validate_solver_readback(self.solver_profile,
            read_solver_readback(self.scene.rigid_solver))
        initial_materials = self._object_material_readback()
        np.testing.assert_allclose(initial_materials['object_engine_mass_kg'],
            self.material_parameters['cube_mass_kg'],rtol=1e-5)
        np.testing.assert_allclose(initial_materials['object_engine_inertia_kg_m2'],
            self.material_parameters['cube_mass_kg']*self.object_spec.inertia_per_kg,rtol=1e-4,atol=1e-9)
        np.testing.assert_allclose(initial_materials['object_bounds_readback_m'],
            self.object_spec.bounds_size_m,rtol=0,atol=1e-7)
        self.fem_pcg_stats = None
        if silicone and silicone_skip_idle_pcg and not ipc:
            from fem_diagnostics import skip_inactive_pcg_iterations
            self.fem_pcg_stats=skip_inactive_pcg_iterations(self.scene.fem_solver)
        if silicone:
            self.silicone_parameters['skip_inactive_pcg_iterations']=bool(silicone_skip_idle_pcg and not ipc)
            self.silicone_parameters['pcg_residual_threshold']=self.scene.fem_solver._pcg_threshold
            self.silicone_parameters['pcg_max_iterations']=self.scene.fem_solver._n_pcg_iterations
        self.names = ARM_JOINTS + HAND_JOINTS
        self.indices = [self.robot.get_joint(n).dofs_idx_local[0] for n in self.names]
        self.arm_indices = self.indices[:6]
        self.hand_indices = self.indices[6:]
        assert len(set(self.indices)) == self.robot.n_dofs == 18
        self.wrist = self.robot.get_link('right_hand_link')
        lo,hi = self.robot.get_dofs_limit(self.indices)
        self.lower,self.upper = self.array(lo).reshape(-1),self.array(hi).reshape(-1)
        self.offset,self.scale = (self.lower+self.upper)/2,(self.upper-self.lower)/2
        self.kp = [4500,4500,3500,2000,2000,2000]+[3.0]*12
        self.kv = [450,450,350,200,200,200]+[.08]*12
        self.robot.set_dofs_kp(self.kp,self.indices)
        self.robot.set_dofs_kv(self.kv,self.indices)
        # Explicitly enforce original UR/vendor effort limits; PD gains are simulation-only.
        import xml.etree.ElementTree as ET
        xml = ET.parse(model).getroot()
        effort = [float(xml.find(f"joint[@name='{n}']/limit").get('effort')) for n in self.names]
        self.effort = np.array(effort)
        self.robot.set_dofs_force_range(-self.effort,self.effort,self.indices)
        # Transcribed from the user's teach-pendant joint-position photo (2026-10-24).
        # Joint values are degrees; the separate TCP orientation fields are radians
        # and are not joint angles. Preserve the source mount and base convention.
        self.home_arm_degrees = [0.00,-89.95,89.96,-89.96,-89.95,-44.98]
        self.home = np.r_[np.deg2rad(self.home_arm_degrees),np.zeros(12)]
        self.episode = -1
        self.image_bytes = {}
        self.reset()

    @staticmethod
    def array(value):
        return value.detach().cpu().numpy() if hasattr(value,'detach') else np.asarray(value)

    def _initial_object_position(self, translation=(0,0,0)):
        initial = (np.array([-.60,-.1333,.225]) if self.legacy_object_placement
                   else self.object_spec.initial_position())
        return initial+np.asarray(translation,dtype=float)

    def _object_morph(self, position):
        spec = self.object_spec
        if spec.shape in ('cube','cuboid'):
            return gs.morphs.Box(size=tuple(spec.bounds_size_m),pos=position)
        if spec.shape=='hex_prism':
            asset=spec.mesh_asset(create=True)
            return gs.morphs.Mesh(file=asset['path'],convexify=True,decimate=False,pos=position)
        if spec.shape == 'cylinder':
            return gs.morphs.Cylinder(radius=spec.radius_m,height=spec.height_m,pos=position)
        return gs.morphs.Sphere(radius=spec.radius_m,pos=position)

    def _object_material_readback(self):
        mass = float(self.array(self.cube.get_mass()).reshape(-1)[0])
        inertia = self.array(self.cube.get_links_inertia()).reshape(-1,3,3)[0]
        bounds = np.ptp(self.array(self.cube.base_link.geoms[0].init_verts),axis=0)
        return {'cube_engine_mass_kg':mass,'object_engine_mass_kg':mass,
                'object_engine_inertia_kg_m2':inertia.tolist(),
                'object_engine_density_kg_m3':mass/self.object_spec.volume_m3,
                'object_bounds_readback_m':bounds.tolist(),
                'density_readback_definition':'solver mass divided by analytic primitive volume'}

    def values(self,value):
        return self.array(value).reshape(-1).astype(float).tolist()

    def render(self, force_render=False):
        if not self.render_enabled:
            return
        pos = self.array(self.wrist.get_pos()).reshape(3)
        quat = self.array(self.wrist.get_quat()).reshape(4)
        rotation = Rotation.from_quat(quat[[1,2,3,0]]).as_matrix()
        mount = self.wrist_camera_mount
        self.wrist_camera.set_pose(pos=pos+rotation@np.asarray(mount['position_m']),
            lookat=pos+rotation@np.asarray(mount['lookat_m']),up=rotation@np.asarray(mount['up']))
        center = pos + rotation @ np.array([0,0,.09])
        for name,(offset,up) in self.hand_view_specs.items():
            camera_pos = center+rotation@np.asarray(offset)
            # Keep the viewpoint above the tabletop at low grasping poses, while
            # retaining its distance to the hand (avoid a clipped, magnified view).
            if camera_pos[2]<.23:
                direction = camera_pos[:2]-center[:2]
                norm = np.linalg.norm(direction)
                direction = direction/norm if norm>1e-6 else np.array([1.,0.])
                radius = np.linalg.norm(offset)
                horizontal = np.sqrt(max(radius**2-(.23-center[2])**2,.01))
                camera_pos = np.r_[center[:2]+horizontal*direction,.23]
            self.hand_cameras[name].set_pose(pos=camera_pos,
                lookat=center,up=rotation@np.asarray(up))
        views = [('scene',self.camera),('scene_left',self.rear_camera),('wrist',self.wrist_camera)]+[(f'hand_{n}',c) for n,c in self.hand_cameras.items()]
        for name,camera in views:
            rgb = self.array(camera.render(rgb=True, force_render=force_render)[0])
            if rgb.ndim==4:
                rgb=rgb[0]
            img = Image.fromarray(rgb.astype(np.uint8))
            buffer = io.BytesIO()
            img.save(buffer,format='JPEG',quality=90)
            self.image_bytes[name] = buffer.getvalue()
            img.save(self.output_dir/f'{name}.png')
            if name=='scene':
                self.image_bytes['scene_front']=self.image_bytes[name]
                img.save(self.output_dir/'scene_front.png')
            if name=='scene_left':
                self.image_bytes['scene_rear']=self.image_bytes[name]  # legacy endpoint
            if name=='hand_thumb':
                self.image_bytes['hand'] = self.image_bytes[name]
                img.save(self.output_dir/'hand.png')

    def reset(self,seed=42):
        if type(seed) is not int:
            raise ValueError('seed must be an integer; scene currently deterministic')
        self.scene.reset()
        self.robot.set_dofs_position(self.home,self.indices,zero_velocity=True)
        self.target=self.home.copy()
        self.robot.control_dofs_position(self.target,self.indices)
        self.actions=((self.target-self.offset)/self.scale).tolist()
        self.episode+=1
        self.steps=0
        self.fault=None
        self.render()
        return self.observe()

    def status(self):
        return {'episode':self.episode,'episode_step':self.steps,'sim_time_s':self.steps*self.control_dt_s,
                'paused':True,'terminated':self.fault is not None,'fault':self.fault}

    def describe(self):
        return {'robot':'UR7e (850 mm) + XHAND1 right','joint_order':self.names,
                'online_table_contact_guard':{'limit_N':self.table_contact_force_limit_N,
                    'metric':'sum of robot/table solver normal forces', 'stop_when':'strictly greater than limit',
                    'self_collision':'stop on any self contact','sample_dt_s':self.control_dt_s},
                'arm_joints':ARM_JOINTS,'hand_joints':HAND_JOINTS,'hand_order':'SDK joint IDs 0..11',
                'control_dt_s':self.control_dt_s,'physics_dt_s':self.physics_dt_s,
                'solver_profile':self.solver_profile,'solver_readback':self.solver_readback,
                'rigid_options_actual':self.scene.rigid_solver._options.model_dump(mode='json'),
                'action_formula':'target_rad = joint_midpoint + normalized_action * half_joint_range',
                'joints':{n:{'index':i,'lower_rad':float(self.lower[i]),'upper_rad':float(self.upper[i]),
                    'offset_rad':float(self.offset[i]),'scale_rad':float(self.scale[i]),
                    'effort_limit_Nm':float(self.effort[i])} for i,n in enumerate(self.names)},
                'tactile_taxels':600,'tactile_shape':[5,120,3],
                'tactile_model':current_tactile_model(),
                'tactile_finger_order':list(self.sensors),'tactile_layout':self.layouts,
                'tactile_frame':'force_sensor_N: sensor body XYZ; force_link_N: attachment link XYZ',
                'tactile_semantics':'Uncalibrated geometric force estimate in nominal N, not SDK LSB or solver contact forces',
                'tactile_parameters':{'probe_radius_m':.0015,'normal_stiffness':500.,'normal_damping':1.,
                                      'normal_exponent':1.2,'shear_scalar':2.},
                'table_top_z_m':TABLE_TOP_Z,'platform_asset':self.platform_metadata,'mount':'local ur5_xhand adapter and camera geometry; vendor XHAND URDF',
                'materials':{**self.material_parameters,**self._object_material_readback()},
                'object_spec':{**self.object_spec.protocol(),
                    'initial_position_m':self.scenario['target_initial_position_m'],
                    'legacy_initial_gap_m':.0025 if self.legacy_object_placement else 0.},
                'silicone':self.silicone_parameters,
                'reset':'fixed scene; seed accepted, no domain randomization',
                'home_arm_degrees':self.home_arm_degrees,
                'home_source':'User teach-pendant joint-position photo, 2026-10-24; nominal URDF kinematics',
                'cameras':self.camera_views,'camera_source':self.camera_source,
                'controller':'simulation PD, uncalibrated to hardware','kp':self.kp,'kv':self.kv,
                'hardware_connected':False,'training_ready':False}

    def observe(self,full=False):
        tactile={}
        for layout in self.layouts:
            name=layout['finger']
            # Genesis history index 0 is the newest physics substep.
            force = self.array(self.sensors[name].read().force).reshape(-1,120,3)[0]
            if not np.isfinite(force).all():
                raise RuntimeError('Nonfinite tactile reading')
            sensor_force = force @ np.array(layout['sensor_to_link_rotation'])
            norms=np.linalg.norm(force,axis=-1)
            tactile[name]={'taxels':120,'active_taxels':int((norms>1e-5).sum()),
                          'peak_force_N':float(norms.max()),'force_sum_sensor_N':sensor_force.sum(0).tolist()}
            if full:
                tactile[name].update(force_sensor_N=sensor_force.tolist(),force_link_N=force.tolist())
        result={**self.status(),'joint_positions_rad':dict(zip(self.names,self.values(self.robot.get_dofs_position(self.indices)))),
            'joint_velocities_rad_s':dict(zip(self.names,self.values(self.robot.get_dofs_velocity(self.indices)))),
            'last_actions':dict(zip(self.names,self.actions)),'tactile':tactile,
            'wrist_position_m':self.values(self.wrist.get_pos()),'wrist_quaternion_wxyz':self.values(self.wrist.get_quat()),
            'rgb_path':str(self.output_dir/'scene.png'),'hand_rgb_path':str(self.output_dir/'hand.png'),
            'overview_rgb_paths':{n:str(self.output_dir/f'scene_{n}.png') for n in ('front','left')},
            'wrist_rgb_path':str(self.output_dir/'wrist.png'),
            'hand_rgb_paths':{n:str(self.output_dir/f'hand_{n}.png') for n in self.hand_cameras}}
        json.dumps(result,allow_nan=False)
        return result

    def evaluate(self):
        """Privileged physical state, separate from policy observations."""
        contacts = self.robot.get_contacts()
        data={k:self.array(v) for k,v in contacts.items()}
        valid=data.get('valid_mask',np.ones(data['link_a'].shape,dtype=bool)).reshape(-1)
        links={l.idx:l for l in self.scene.rigid_solver.links}
        entities={self.robot.idx:'ur7e_xhand',self.table.idx:'table',self.cube.idx:'cube'}
        entities.update({entity.idx:name for name,entity in self.distractors.items()})
        if self.fixture is not None:
            entities[self.fixture.idx]='test_sphere'
        rows=[]
        all_object_load = 0.
        all_object_normal_load = 0.
        for i in np.flatnonzero(valid):
            a=int(data['link_a'].reshape(-1)[i]); b=int(data['link_b'].reshape(-1)[i])
            force=data['force_a'].reshape(-1,3)[i]
            normal=data['normal'].reshape(-1,3)[i]
            normal=normal/max(np.linalg.norm(normal),1e-12)
            normal_load=float(abs(np.dot(force,normal)))
            if (a == self.cube.base_link.idx and links[b].entity.idx == self.robot.idx
                    or b == self.cube.base_link.idx and links[a].entity.idx == self.robot.idx):
                all_object_load += float(np.linalg.norm(force))
                all_object_normal_load += normal_load
            rows.append({'link_a':links[a].name,'link_b':links[b].name,
                         'entity_a':entities.get(links[a].entity.idx,'environment'),
                         'entity_b':entities.get(links[b].entity.idx,'environment'),
                         'link_a_global_index':a,'link_b_global_index':b,
                         'force_on_a_world_N':force.tolist(),
                         'normal_world':normal.tolist(),'normal_force_N':normal_load,
                         'position_world_m':data['position'].reshape(-1,3)[i].tolist()})
        position=self.values(self.cube.get_pos())
        quaternion=self.values(self.cube.get_quat())
        bottom=float(position[2]-self.object_spec.vertical_extent(np.asarray(quaternion)))
        return {'cube_position_m':position,'cube_quaternion_wxyz':quaternion,
                'object_position_m':position.copy(),'object_quaternion_wxyz':quaternion.copy(),
                'object_spec':self.object_spec.protocol(),
                'bottom_world_z_m':bottom,'bottom_clearance_m':bottom-.2,
                'all_robot_contact_load_N':all_object_load,
                'all_robot_normal_load_N':all_object_normal_load,
                'object_material_readback':self._object_material_readback(),
                'distractor_positions_m':{name:self.values(entity.get_pos()) for name,entity in self.distractors.items()},
                'silicone':self.silicone_state(),
                'solver_contacts':rows,'joint_force_Nm':self.values(self.robot.get_dofs_force(self.indices))}

    def silicone_state(self):
        """Privileged FEM diagnostics; never represented as XHand tactile force."""
        if self.silicone is None:
            return None
        state = self.silicone.get_state()
        pos = self.array(state.pos).reshape(-1,3)
        vel = self.array(state.vel).reshape(-1,3)
        finite = bool(np.isfinite(pos).all() and np.isfinite(vel).all())
        return {'finite':finite,'vertex_count':len(pos),'tactile_supported':False,
                'bounds_min_m':pos.min(0).tolist() if finite else None,
                'bounds_max_m':pos.max(0).tolist() if finite else None,
                'extent_m':np.ptp(pos,axis=0).tolist() if finite else None,
                'velocity_available':self.silicone_parameters['coupler'] != 'IPCCoupler',
                'max_vertex_speed_m_s':float(np.linalg.norm(vel,axis=1).max())
                    if finite and self.silicone_parameters['coupler'] != 'IPCCoupler' else None}

    def step(self,payload):
        if self.fault:
            raise ValueError('Reset after simulation fault')
        for key,expected in [('expected_episode',self.episode),('expected_step',self.steps)]:
            if key in payload and (type(payload[key]) is not int or payload[key]!=expected):
                raise ValueError('Stale state')
        actions,count=validate_step(payload,self.names,self.actions)
        self.actions=actions
        self.target=self.offset+np.asarray(actions)*self.scale
        self.robot.control_dofs_position(self.target,self.indices)
        for _ in range(count):
            for _ in range(self.control_substeps):
                self.scene.step()
            self.steps+=1
            if not np.isfinite(self.array(self.robot.get_dofs_position(self.indices))).all():
                self.fault='Nonfinite joint state'
                break
            if self.silicone is not None and not self.silicone_state()['finite']:
                self.fault='Nonfinite silicone FEM state'
                break
        self.render()
        return self.observe()

    def move(self,payload):
        delta=np.asarray(payload.get('delta',[0,0,0]),dtype=float)
        steps=payload.get('steps',10)
        if delta.shape!=(3,) or not np.isfinite(delta).all() or np.linalg.norm(delta)>.05:
            raise ValueError('delta must be finite xyz with length <= 5 cm')
        if type(steps) is not int or not 1<=steps<=10:
            raise ValueError('steps must be 1..10')
        pos=self.array(self.wrist.get_pos()).reshape(3)+delta
        if not (-.8<=pos[0]<=.1 and -.5<=pos[1]<=.5 and .25<=pos[2]<=.9):
            raise ValueError('Wrist target outside interactive workspace')
        q,err=self.robot.inverse_kinematics(self.wrist,pos=pos,quat=self.wrist.get_quat(),
            dofs_idx_local=self.arm_indices,return_error=True,max_solver_iters=100,seed=42)
        err=self.array(err).reshape(-1)
        if np.linalg.norm(err[:3])>.003 or np.linalg.norm(err[3:])>.03:
            raise ValueError('IK target unreachable')
        desired=self.array(q).reshape(-1)[self.indices]
        normalized=(desired-self.offset)/self.scale
        return self.step({'actions':dict(zip(ARM_JOINTS,normalized[:6].tolist())),'steps':steps})

    def hand(self,payload):
        closure=payload.get('closure',0)
        if isinstance(closure,bool) or not isinstance(closure,(int,float)) or not np.isfinite(closure) or not 0<=closure<=1:
            raise ValueError('closure must be in [0,1]')
        desired=np.array([.5,.3,.6,0,.7,.7,.7,.7,.7,.7,.7,.7])*closure
        normalized=(desired-self.offset[6:])/self.scale[6:]
        return self.step({'actions':dict(zip(HAND_JOINTS,normalized.tolist())),'steps':payload.get('steps',10)})
