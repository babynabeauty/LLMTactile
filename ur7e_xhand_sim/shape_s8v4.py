"""Shape-study S8V4: Astra SkillAct decisions plus bounded tactile closure.

The historical reference supplies preshape, orientation and registration only.
No agent-selected candidate or low-level motion is issued by this module until
an actual SkillAct from the API driver requests the corresponding skill.
"""
from pathlib import Path
from copy import deepcopy
import json
import xml.etree.ElementTree as ET

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from verified_s8v4 import VerifiedS8V4
from s8v4 import HERE, write, sha, SkillAct, URDFFrames, HAND_JOINTS, ARM_JOINTS, XHandContactPrior, GENDEX


def hand_pad_geometry(fk, layouts, normals, hand):
    """Object-free FK proxy; mesh outward normals, never sensor-axis normals."""
    joints = dict(zip(HAND_JOINTS,np.asarray(hand,dtype=float)))
    joints.update(dict.fromkeys(ARM_JOINTS,0.))
    points,directions = [],[]
    for layout,normal in zip(layouts,normals):
        transform = fk.transform('right_hand_link',layout['link_name'],joints)
        points.append(np.asarray(layout['positions_link_m'])@transform[:3,:3].T+transform[:3,3])
        directions.append(normal@transform[:3,:3].T)
    centers = np.array([p.mean(0) for p in points])
    axis = centers[1]-centers[0]
    gap = float(np.linalg.norm(axis))
    if not np.isfinite(centers).all() or not .02 < gap < .18:
        raise ValueError('Preshape pad geometry is nonfinite or outside bounded separation')
    axis /= gap
    facing = [directions[0]@axis > .7,directions[1]@(-axis) > .7]
    counts = [int(mask.sum()) for mask in facing]
    selected_normals = []
    for normal,mask in zip(directions,facing):
        if not mask.any():
            raise ValueError('Preshape has no geometrically facing sensor patch')
        mean = normal[mask].mean(0)
        selected_normals.append(mean/np.linalg.norm(mean))
    patch_centers = np.array([p[mask].mean(0) for p,mask in zip(points,facing)])
    patch_delta = patch_centers[1]-patch_centers[0]
    minimum = float(np.linalg.norm(points[0][:,None]-points[1][None,:],axis=-1).min())
    if minimum < .008:
        raise ValueError('Preshape sensor point clouds are too close for this bounded catalog')
    return dict(pad_centers_palm_m=centers.tolist(),pad_midpoint_palm_m=centers.mean(0).tolist(),
        pad_center_gap_m=gap,minimum_interpad_taxel_distance_m=minimum,
        facing_taxel_counts=counts,facing_cosine_threshold=.7,
        facing_patch_centers_palm_m=patch_centers.tolist(),
        facing_patch_axial_gap_m=float(patch_delta@axis),
        facing_patch_tangential_offset_m=float(np.linalg.norm(patch_delta-axis*(patch_delta@axis))),
        facing_patch_mean_normals_palm= [n.tolist() for n in selected_normals],
        facing_patch_normal_angle_deg=float(np.degrees(np.arccos(np.clip(np.dot(*selected_normals),-1,1)))),
        interpretation='Hand-only geometric proxy using nearest outward collision-mesh face normals. Counts are not physical object coverage or force-closure evidence.')


def build_preshape_catalog(urdf, layouts, open_hand, closed_hand, lower, upper, nominal_preload):
    """Return in-limit declared choices; the reference itself must be valid."""
    import trimesh
    root = ET.parse(urdf).getroot()
    fk = URDFFrames(urdf)
    layouts = [next(row for row in layouts if row['finger']==finger) for finger in ('thumb','index')]
    urdf_lower,urdf_upper = [],[]
    for name in HAND_JOINTS:
        limit = root.find(f'joint[@name="{name}"]/limit')
        urdf_lower.append(float(limit.get('lower')))
        urdf_upper.append(float(limit.get('upper')))
    lower = np.maximum(lower,urdf_lower)
    upper = np.minimum(upper,urdf_upper)
    normals,mesh_sources = [],[]
    for layout in layouts:
        collision = root.find(f'link[@name="{layout["link_name"]}"]/collision')
        source = collision.find('geometry/mesh')
        path = Path(source.get('filename'))
        if not path.is_absolute():path=Path(urdf).parent/path
        mesh = trimesh.load(path,force='mesh')
        mesh.apply_scale(np.fromstring(source.get('scale','1 1 1'),sep=' '))
        origin = collision.find('origin')
        transform = np.eye(4)
        if origin is not None:
            transform[:3,:3] = Rotation.from_euler('xyz',np.fromstring(origin.get('rpy','0 0 0'),sep=' ')).as_matrix()
            transform[:3,3] = np.fromstring(origin.get('xyz','0 0 0'),sep=' ')
        mesh.apply_transform(transform)
        if not mesh.is_watertight or not mesh.is_winding_consistent or mesh.volume <= 0:
            raise ValueError('Preshape normal proxy requires outward, watertight finger collision mesh')
        _,distance,faces = mesh.nearest.on_surface(np.asarray(layout['positions_link_m']))
        if not np.isfinite(distance).all() or distance.max() > .002:
            raise ValueError('Taxel coordinates do not match the selected collision link')
        normals.append(np.asarray(mesh.face_normals[faces]))
        mesh_sources.append(dict(path=str(path),sha256=sha(path),
            taxel_mesh_distance_quantiles_m=np.quantile(distance,[0.,.5,1.]).tolist()))
    if not np.isfinite(nominal_preload) or not 0 <= nominal_preload <= .25:
        raise ValueError('Reference preload increment is invalid')
    choices = [('reference',[0.,0.,0.],False),
               ('reference_hold_face',[0.,0.,0.],True),
               ('opposed_mild_hold_face',[.15,-.10,.05],True),
               ('opposed_wide_hold_face',[.25,-.20,.15],True)]
    result = {}
    for name,offset,synergy in choices:
        opened,closed = np.array(open_hand,dtype=float),np.array(closed_hand,dtype=float)
        opened[[0,1,3]] += offset
        closed[[0,1,3]] += offset
        if synergy:closed[5] -= nominal_preload
        invalid = any(hand.shape != (12,) or not np.isfinite(hand).all() or
                      np.any(hand < lower-1e-8) or np.any(hand > upper+1e-8)
                      for hand in (opened,closed))
        if invalid:
            if name == 'reference':
                raise ValueError('Preshape violates URDF/runtime limits: '+name)
            # An optional offset may exceed limits for a different demonstration.
            # Omit it rather than clip targets or reject a valid reference.
            continue
        geometry = hand_pad_geometry(fk,layouts,normals,closed)
        opening = hand_pad_geometry(fk,layouts,normals,opened)
        orientation_error = 0.
        if synergy:
            # Verify commanded orientation invariance from the complete URDF
            # chain, including every joint origin; matching axis names is not
            # sufficient. This is not a claim about loaded physical tracking.
            for delta in (0.,.05,.15,.25):
                first=closed.copy();second=closed.copy()
                first[4]=.4;second[4]=.4+delta;second[5]-=delta
                a=dict(zip(HAND_JOINTS,first));b=dict(zip(HAND_JOINTS,second))
                a.update(dict.fromkeys(ARM_JOINTS,0.));b.update(dict.fromkeys(ARM_JOINTS,0.))
                r0=fk.transform('right_hand_link',layouts[1]['link_name'],a)[:3,:3]
                r1=fk.transform('right_hand_link',layouts[1]['link_name'],b)[:3,:3]
                orientation_error=max(orientation_error,float(np.linalg.norm(r0-r1)))
            if orientation_error > 1e-8:
                raise ValueError('URDF does not support the declared orientation-hold synergy')
        result[name] = dict(preshape_id=name,open_hand_target_rad=opened.tolist(),
            nominal_closed_hand_target_rad=closed.tolist(),
            joint_offsets_rad=dict(zip([HAND_JOINTS[i] for i in (0,1,3)],offset)),
            post_contact_synergy='index_proximal_plus_distal_minus' if synergy else 'reference_proximal_only',
            nominal_scoring_preload_increment_rad=nominal_preload if synergy else None,
            commanded_fk_orientation_error=orientation_error,
            closed_geometry=geometry,open_geometry=opening,
            physical_validation='unvalidated candidate; no physics, object state or grasp search used',
            mechanical_check='URDF/runtime limits, finite FK and inter-taxel separation only; not complete mesh self-collision certification',
            mesh_normal_sources=mesh_sources)
    return result


class BoundedShapeCandidates:
    """Retain GenDex rankings and explicitly mark extra unranked choices."""
    SOURCE = 'bounded world-X/Z shape-study offset; not GenDex ranked; Astra must select'

    def __init__(self, prior):
        self.prior = prior

    def __getattr__(self, name):
        return getattr(self.prior, name)

    def propose(self, *args, **kwargs):
        result = self.prior.propose(*args, **kwargs)
        result = {**result, 'candidates':[dict(candidate) for candidate in result['candidates']]}
        for axis in (0,2):
            for distance in (-.01,-.005,.005,.01):
                offset = np.zeros(3)
                offset[axis] = distance
                if any(np.allclose(offset, candidate['offset_xyz_m'], atol=1e-8,rtol=0)
                       for candidate in result['candidates']):
                    continue
                result['candidates'].append({'offset_xyz_m':offset.tolist(),
                                             'energy':None,'source':self.SOURCE})
        result['additional_candidate_policy'] = {
            'source':self.SOURCE,'axes':'world X or world Z, one axis at a time',
            'distances_m':[-.01,-.005,.005,.01],'automatic_selection':False}
        return result


class ShapeS8V4(VerifiedS8V4):
    supports_preload_selection = True
    supports_preshape_selection = True
    supports_grip_gain_selection = True
    GRIP_GAIN_CHOICES = (1.,2.,3.)
    PRELOAD_CHOICES_RAD = (.05, .10, .15, .20)
    DEFAULT_PRELOAD_INCREMENT_RAD = .05
    CLOSURE = {
        'mode':'bounded_index_proximal_tactile_closure_v3',
        'hand_joint_index':4,'close_increment_rad_per_tick':.004,
        'index_proximal_limit_rad':.85,'max_close_ticks':300,
        'tactile_threshold_nominal_N':.01,'confirmation_samples':5,
        'preload_increment_rad':.05,'preload_ramp_ticks':30,'preload_hold_ticks':30,
        'preload_recovery_increment_limit_rad':.05,'preload_recovery_max_ticks':60,
        'control_tick_s':.025,'support_fingers':['thumb','index']}
    MOTION_RECOVERY = {
        'controller':'original RuleSupportController, unchanged threshold/rate/increment/anchor bounds',
        'max_ticks_per_event':60,'max_ticks_per_trial':180,'max_events_per_trial':8,
        'confirmation_samples':5,'control_tick_s':.025,
        'palm_target':'measured palm position and orientation at support-loss detection',
        'resume_gate':'five new consecutive thumb/index tactile samples >=.01 nominal N',
        'recording':'every recovery command retains its current phase and is logged normally'}

    def __init__(self, sim, label, reference):
        if not reference:
            raise ValueError('Shape-study S8V4 requires a historical demonstration reference')
        if getattr(sim,'legacy_object_placement',True):
            raise ValueError('Select an explicit rigid object_spec for ShapeS8V4')
        self.sim=sim
        self.base_grip_kp=np.asarray(sim.kp,dtype=float).copy()
        self.base_grip_kv=np.asarray(sim.kv,dtype=float).copy()
        self.grip_gain_selection={'source':'default; close not yet requested',
            'requested_gain':0.,'effective_gain':1.}
        self._apply_grip_gain(1.)
        self.CLOSURE = dict(type(self).CLOSURE)
        self.CLOSURE['control_tick_s']=sim.control_dt_s
        self.preshape_catalog = {}
        self.preshape_selection = {'preshape_id':'reference','source':'default; locate not yet requested'}
        self.preload_selection = {'source':'default; close not yet requested',
                                  'requested_increment_rad':0.,
                                  'effective_increment_rad':self.DEFAULT_PRELOAD_INCREMENT_RAD}
        self.closure_state = {'stage':'not_started','close_ticks':0,'confirmed_samples':0}
        self.motion_recovery_state = {'stage':'idle','events':0,'total_ticks':0,
                                      'consecutive_supported_samples':0}
        self.motion_recovery_events = []
        super().__init__(sim,label,reference)
        reference_close = [row for row in self.closure if row['phase']=='close']
        reference_preload = [row for row in self.closure if row['phase']=='preload']
        if not reference_close or not reference_preload:
            raise ValueError('Reference must retain actual close and preload commands')
        nominal_preload = (reference_preload[-1]['sent_target_rad'][10]-
                           reference_close[-1]['sent_target_rad'][10])
        self.reference_relative_alignment = self.relative_alignment.copy()
        self.recorded_tripod = self.ref.get('execution_mode') == 'recorded_tripod_closure_v1'
        if self.recorded_tripod:
            if self.ref.get('contact_fingers') != ['thumb','index','middle']:
                raise ValueError('Recorded tripod requires explicit thumb/index/middle support')
            self.recorded_closure = list(self.closure)
            self.controller.fingers = (0,1,2)
            self.CLOSURE = {**self.CLOSURE,'mode':'recorded_tripod_closure_v1'}
            self.preshape_catalog = {'reference':dict(preshape_id='reference',
                open_hand_target_rad=self.open.tolist(),nominal_closed_hand_target_rad=self.closed.tolist(),
                physical_validation='recorded successful three-finger source; new mass/friction unverified')}
        else:
            self.preshape_catalog = build_preshape_catalog(HERE/'generated/ur7e_xhand.urdf',
                sim.layouts,self.open,self.closed,sim.lower[6:],sim.upper[6:],nominal_preload)
        self.prior = BoundedShapeCandidates(self.prior)
        # VerifiedS8V4 iterates this only when Astra requests the close skill.
        # Rows are generated from new contact measurements, not the old demo.
        self.closure = self._recorded_tripod_closure() if self.recorded_tripod else self._bounded_closure()
        manifest_path = self.out/'manifest.json'
        manifest = json.loads(manifest_path.read_text())
        manifest.update(
            method='UR7e/XHand S8V4 shape-study port with bounded tactile closure',
            object_spec=sim.object_spec.protocol(),
            components=['historical multimodal reference for preshape/orientation/registration',
                'live RGB-D registration','GenDex CMap-CVAE candidate scoring',
                'explicitly unranked bounded world-X/Z choices','independent Astra SkillAct',
                'online bounded tactile close/preload','RuleSupportController after closure'],
            closure=self.CLOSURE,
            preload_selection=self.preload_selection,
            preload_choices_rad=[0.] if getattr(self,'recorded_tripod',False) else list(self.PRELOAD_CHOICES_RAD),
            grip_gain_choices=list(self.GRIP_GAIN_CHOICES),
            grip_gain_selection=self.grip_gain_selection,
            grip_gain_controller=self.grip_gain_controller,
            preshape_catalog=self.preshape_catalog,
            preshape_selection=self.preshape_selection,
            motion_recovery=self.MOTION_RECOVERY,
            demonstration_usage='Historical reference preshape/orientation/registration, with only explicitly selected bounded hand-only FK alternatives; historical close/preload sequence is not replayed',
            candidate_selection='Astra API selects from supplied candidates; no automatic grid search or scripted candidate selection',
            support_gate='five consecutive NEW close samples with both thumb/index tactile >=.01 nominal N, followed by bounded preload/settling and a fresh five-sample support gate; support loss pauses upward progression at the current measured palm pose while the original bounded RuleSupportController may recover; resume only after five new supported samples, otherwise fail at the recovery limit',
            support_rule='Original UR/XHand RuleSupportController activates after new bounded closure; Astra close may select common thumb/index PD gain 1/2/3; joint/effort limits unchanged',
            limits='Fixed-orientation reference transfer, finite preshape catalog and bounded index closure/preload synergy; original post-closure support rule remains unchanged; not a full grasp optimizer')
        if self.recorded_tripod:
            manifest.update(method='Astra registered three-finger recorded closure with frozen PPO',
                demonstration_usage='Recorded close/preload hand targets at original sample rate; live Astra registration and skill decisions',
                support_gate='All thumb/index/middle tactile >=.01 nominal N for five samples before lift; motion recovery unchanged',
                support_rule='RuleSupportController for thumb/index/middle; PPO still scales only thumb/index',
                components=['historical three-finger multimodal example','live RGB-D registration','Astra skill decisions',
                    'recorded three-finger close/preload','three-finger support rule','frozen thumb/index gain PPO'],
                limits='Experimental transfer to a new three-finger trajectory; PPO was not retrained; middle gain stays nominal')
        for path in (Path(__file__),HERE/'server.py',HERE/'residual_rl/multiobject_geometry.py'):
            manifest['source_hashes'][str(path)] = sha(path)
        write(manifest_path,manifest)

    def observe(self):
        result = super().observe()
        result.update(object_shape=self.sim.object_spec.shape,
            closure_mode=self.CLOSURE['mode'],closure_parameters=dict(self.CLOSURE),
            preload_choices_rad=[0.] if getattr(self,'recorded_tripod',False) else list(self.PRELOAD_CHOICES_RAD),
            preload_default_increment_rad=self.DEFAULT_PRELOAD_INCREMENT_RAD,
            preload_selection=dict(self.preload_selection),
            grip_gain_choices=list(self.GRIP_GAIN_CHOICES),
            grip_gain_selection=dict(self.grip_gain_selection),
            grip_gain_controller=deepcopy(self.grip_gain_controller),
            preshape_catalog=deepcopy(self.preshape_catalog),
            preshape_selection=deepcopy(self.preshape_selection),
            closure_state=dict(self.closure_state),
            motion_recovery_parameters=dict(self.MOTION_RECOVERY),
            motion_recovery_state=dict(self.motion_recovery_state),
            candidate_note='GenDex proposals and clearly marked unranked X/Z offsets; Astra chooses the offset')
        return result

    def execute(self,payload):
        # Select only after validating the actual Astra request and the same
        # preconditions used by the inherited close skill. The lazy generator
        # reads this per-instance configuration when close actually starts.
        action = SkillAct.model_validate(payload)
        if action.status == 'continue' and action.skill == 'locate':
            if self.finished or self.failed:
                raise ValueError('Trial already ended')
            if (action.observation_id != self.observe()['observation_id'] or
                    self.stamp != (self.sim.episode,self.sim.steps)):
                raise ValueError('Stale observation')
            if self.phase != 'unregistered':
                raise ValueError('Already registered')
            self._select_preshape(action)
        if action.status == 'continue' and action.skill == 'close':
            if self.finished or self.failed:
                raise ValueError('Trial already ended')
            if (action.observation_id != self.observe()['observation_id'] or
                    self.stamp != (self.sim.episode,self.sim.steps)):
                raise ValueError('Stale observation')
            if self.phase != 'aligned':
                raise ValueError('Skill invalid in phase '+self.phase)
            if not np.allclose(action.offset_xyz_m,self.offset):
                raise ValueError('Preserve offset')
            if getattr(self,'recorded_tripod',False) and action.preload_increment_rad != 0:
                raise ValueError('Recorded tripod uses demonstration preload; choose preload_increment_rad=0')
            effective = (0. if getattr(self,'recorded_tripod',False) else
                         action.preload_increment_rad or self.DEFAULT_PRELOAD_INCREMENT_RAD)
            self.CLOSURE = {**self.CLOSURE,'preload_increment_rad':effective}
            self.preload_selection = {
                'source':('recorded tripod preload' if getattr(self,'recorded_tripod',False) else
                          'Astra SkillAct close' if action.preload_increment_rad else 'close default'),
                'observation_id':action.observation_id,
                'requested_increment_rad':action.preload_increment_rad,
                'effective_increment_rad':effective}
            self._apply_grip_gain(action.grip_gain or 1.)
            self.grip_gain_selection={
                'source':'Astra SkillAct close' if action.grip_gain else 'close default',
                'observation_id':action.observation_id,
                'requested_gain':action.grip_gain,
                'effective_gain':action.grip_gain or 1.}
            manifest_path = self.out/'manifest.json'
            manifest = json.loads(manifest_path.read_text())
            manifest.update(closure=self.CLOSURE,preload_selection=self.preload_selection,
                grip_gain_selection=self.grip_gain_selection,
                grip_gain_controller=self.grip_gain_controller)
            write(manifest_path,manifest)
        return super().execute(payload)

    def _apply_grip_gain(self,gain):
        """Same absolute PD scaling as ResidualVecEnv._apply_gains; no motion."""
        if gain not in self.GRIP_GAIN_CHOICES:
            raise ValueError('Choose a bounded common grip gain 1, 2, or 3')
        kp,kv=self.base_grip_kp.copy(),self.base_grip_kv.copy()
        if kp.shape!=(18,) or kv.shape!=(18,) or not np.isfinite(np.r_[kp,kv]).all():
            raise ValueError('Invalid nominal robot PD gains')
        kp[6:12]*=gain
        kv[6:12]*=np.sqrt(gain)
        s=self.sim
        s.robot.set_dofs_kp(kp,s.indices)
        s.robot.set_dofs_kv(kv,s.indices)
        read_kp=s.array(s.robot.get_dofs_kp(s.indices)).reshape(-1)
        read_kv=s.array(s.robot.get_dofs_kv(s.indices)).reshape(-1)
        effort_lower,effort_upper=s.robot.get_dofs_force_range(s.indices)
        read_lower=s.array(effort_lower).reshape(-1)
        read_upper=s.array(effort_upper).reshape(-1)
        np.testing.assert_allclose(read_kp,kp,rtol=1e-6,atol=1e-7)
        np.testing.assert_allclose(read_kv,kv,rtol=1e-6,atol=1e-7)
        np.testing.assert_allclose(read_lower,-s.effort,rtol=1e-6,atol=1e-7)
        np.testing.assert_allclose(read_upper,s.effort,rtol=1e-6,atol=1e-7)
        self.grip_gain=float(gain)
        self.grip_gain_controller=dict(common_thumb_index_gain=float(gain),
            scaled_joint_names=HAND_JOINTS[:6],kp_rule='base * gain',
            kv_rule='base * sqrt(gain)',kp_readback=read_kp.tolist(),
            kv_readback=read_kv.tolist(),effort_lower_readback_Nm=read_lower.tolist(),
            effort_upper_readback_Nm=read_upper.tolist(),effort_limits_unchanged=True,
            applied_at_episode=int(s.episode),applied_at_episode_step=int(s.steps),
            readback_timing='immediately after applying this gain; command rows retain the latest application readback',
            schedule='absolute gain applied before close and held through preload/test_lift/lift/hold; no RL rate ramp in demonstration')

    def _command_metadata(self):
        return dict(grip_gain=self.grip_gain,
            grip_gain_selection=dict(self.grip_gain_selection),
            grip_gain_controller=deepcopy(self.grip_gain_controller))

    def _new_preshape_prior(self,closed):
        with torch.device('cpu'):
            prior = XHandContactPrior(GENDEX,HERE/'generated/ur7e_xhand.urdf',
                HERE/'sources/xhand/meshes',closed,[0]*6)
        return BoundedShapeCandidates(prior)

    def _select_preshape(self,action):
        name = 'reference' if action.preshape_id == 'keep' else action.preshape_id
        if name not in self.preshape_catalog:
            raise ValueError('Choose an available bounded preshape')
        if getattr(self,'recorded_tripod',False):
            self.preshape_selection = dict(preshape_id='reference',source='Astra selected recorded tripod reference')
            return
        candidate = self.preshape_catalog[name]
        opened=np.array(candidate['open_hand_target_rad'])
        closed=np.array(candidate['nominal_closed_hand_target_rad'])
        for hand in (opened,closed):
            if not np.isfinite(hand).all() or np.any(hand<self.sim.lower[6:]) or np.any(hand>self.sim.upper[6:]):
                raise ValueError('Selected preshape violates runtime limits')
        reference_center=np.array(self.preshape_catalog['reference']['closed_geometry']['pad_midpoint_palm_m'])
        candidate_center=np.array(candidate['closed_geometry']['pad_midpoint_palm_m'])
        compensation=self.rotation@(reference_center-candidate_center)
        if not np.isfinite(compensation).all() or np.linalg.norm(compensation)>.02:
            raise ValueError('Preshape registration compensation exceeds 2 cm bound')
        if name != 'reference':
            # Rebuild actual sampled XHand mesh before inherited locate invokes
            # GenDex scoring. The existing reference prior is untouched when
            # reference/keep is selected.
            prior=self._new_preshape_prior(closed)
            self.open=opened
            self.closed=closed
            self.relative_alignment=self.reference_relative_alignment+compensation
            self.prior=prior
            self.baseline_candidate_source=(f'{name} FK-compensated nominal alignment; '
                'unvalidated preshape, not a verified demonstration baseline or GenDex-ranked offset')
        hold_face=candidate['post_contact_synergy']=='index_proximal_plus_distal_minus'
        if hold_face:
            self.CLOSURE={**self.CLOSURE,
                'mode':'bounded_index_closure_postcontact_orientation_hold_v1',
                'post_contact_distal_compensation':True}
        self.preshape_selection=dict(preshape_id=name,requested_preshape_id=action.preshape_id,
            source='Astra SkillAct locate' if action.preshape_id!='keep' else 'locate default reference',
            observation_id=action.observation_id,
            registration_compensation_world_m=compensation.tolist(),
            reference_relative_alignment_world_m=self.reference_relative_alignment.tolist(),
            selected_relative_alignment_world_m=self.relative_alignment.tolist(),
            post_contact_synergy=candidate['post_contact_synergy'],
            prior_nominal_hand_target_rad=self.closed.tolist(),
            nominal_scoring_preload_increment_rad=candidate['nominal_scoring_preload_increment_rad'],
            actual_close_note='Live contact angle and Astra close preload may differ from nominal scoring hand; actual commands are logged.',
            physical_validation=candidate['physical_validation'])
        manifest_path=self.out/'manifest.json'
        manifest=json.loads(manifest_path.read_text())
        manifest.update(preshape_selection=self.preshape_selection,closure=self.CLOSURE,
                        prior=self.prior.provenance)
        write(manifest_path,manifest)

    def _both_tactile_supported(self):
        return all(value['tactile_peak_nominal_N'] >= self.CLOSURE['tactile_threshold_nominal_N']
                   for value in self.support().values())

    def _closure_row(self, hand, phase):
        return {'phase':phase,'sent_target_rad':np.r_[self.sim.target[:6],hand].tolist()}

    def _recorded_tripod_closure(self):
        self.controller.deactivate()
        for row in self.recorded_closure:
            hand=np.asarray(row['sent_target_rad'][6:],float)
            if np.any(hand < self.sim.lower[6:]-1e-6) or np.any(hand > self.sim.upper[6:]+1e-6):
                raise ValueError('Recorded tripod hand exceeds limits')
            self.closure_state['stage']=row['phase']
            yield self._closure_row(hand,row['phase'])
        if len(self.support_history)<5 or not all(self.support_history[-5:]):
            raise ValueError('Recorded tripod closure lacks persistent three-finger tactile; lift blocked')
        self.closure_state['stage']='ready_for_test_lift'

    def _bounded_closure(self):
        cfg = self.CLOSURE
        self.controller.deactivate()
        hand = self.sim.target[6:].copy()
        joint = cfg['hand_joint_index']
        upper = min(cfg['index_proximal_limit_rad'],float(self.sim.upper[6+joint]))
        lower = float(self.sim.lower[6+joint])
        if hand[joint] < lower-1e-8 or hand[joint] > upper+1e-8:
            raise ValueError('Initial index proximal target lies outside bounded closure range')
        confirmed = 0
        self.closure_state.update(stage='close',index_limit_rad=upper)
        # Count only new servo observations from this closure, not alignment's
        # history. On first contact hold the target while confirming persistence.
        for tick in range(cfg['max_close_ticks']):
            if not self._both_tactile_supported():
                hand[joint] = min(hand[joint]+cfg['close_increment_rad_per_tick'],upper)
            self.closure_state['close_ticks'] = tick+1
            yield self._closure_row(hand,'close')
            confirmed = confirmed+1 if self._both_tactile_supported() else 0
            self.closure_state['confirmed_samples'] = confirmed
            if confirmed >= cfg['confirmation_samples']:
                break
        else:
            self.closure_state['stage'] = 'failed_no_persistent_contact'
            raise ValueError('No persistent thumb/index tactile within bounded shape closure; lift blocked')

        start = hand.copy()
        goal = hand.copy()
        hold_face = cfg.get('post_contact_distal_compensation',False)
        if hold_face:
            # Keep q_proximal + q_distal constant only after real dual contact.
            # Restrict the shared increment before interpolation; never clip
            # distal separately, which would silently break the synergy.
            upper=min(upper,start[joint]+start[5]-float(self.sim.lower[11]))
        goal[joint] = min(goal[joint]+cfg['preload_increment_rad'],upper)
        if hold_face:goal[5]=start[5]-(goal[joint]-start[joint])
        self.closure_state.update(stage='preload',contact_index_target_rad=float(start[joint]),
            preload_index_target_rad=float(goal[joint]),preload_ticks=0,
            contact_index_distal_target_rad=float(start[5]),
            preload_index_distal_target_rad=float(goal[5]),
            post_contact_distal_compensation=bool(hold_face),
            preload_support_gap_ticks=0,preload_final_supported_samples=0,
            recovery_ticks=0,recovery_increment_rad=0.)
        # The object is still on the table and the palm position is fixed.
        # Complete the bounded preload and let its contact response settle;
        # a single missing contact sample cannot authorize lift, but need not
        # abort this stationary contact-establishment stage either.
        settled_support = 0
        for tick in range(1,cfg['preload_ramp_ticks']+1):
            hand = start+(goal-start)*tick/cfg['preload_ramp_ticks']
            yield self._closure_row(hand,'preload')
            supported = self._both_tactile_supported()
            settled_support = settled_support+1 if supported else 0
            self._record_preload_sample(supported,settled_support)
        for _ in range(cfg['preload_hold_ticks']):
            yield self._closure_row(hand,'preload')
            supported = self._both_tactile_supported()
            settled_support = settled_support+1 if supported else 0
            self._record_preload_sample(supported,settled_support)
        if settled_support < cfg['confirmation_samples']:
            # Recovery uses the same one-joint closure law and remains in the
            # preload phase, with an independent increment and duration cap.
            self.closure_state['stage'] = 'preload_contact_recovery'
            recovery_upper = min(goal[joint]+cfg['preload_recovery_increment_limit_rad'],upper)
            for tick in range(cfg['preload_recovery_max_ticks']):
                if not self._both_tactile_supported():
                    hand[joint] = min(hand[joint]+cfg['close_increment_rad_per_tick'],recovery_upper)
                    if hold_face:hand[5]=start[5]-(hand[joint]-start[joint])
                self.closure_state['recovery_ticks'] = tick+1
                self.closure_state['recovery_increment_rad'] = float(hand[joint]-goal[joint])
                yield self._closure_row(hand,'preload')
                supported = self._both_tactile_supported()
                settled_support = settled_support+1 if supported else 0
                self._record_preload_sample(supported,settled_support)
                if settled_support >= cfg['confirmation_samples']:
                    break
            else:
                self.closure_state['stage'] = 'failed_preload_support_not_reestablished'
                raise ValueError('Persistent thumb/index tactile not reestablished within bounded preload recovery; lift blocked')
        if settled_support < cfg['confirmation_samples']:
            raise ValueError('Final preload lacks five consecutive supported samples; lift blocked')
        self.closure_state['stage'] = 'ready_for_test_lift'

    def _record_preload_sample(self,supported,consecutive):
        self.closure_state['preload_ticks'] += 1
        self.closure_state['preload_support_gap_ticks'] += int(not supported)
        self.closure_state['preload_final_supported_samples'] = consecutive

    def servo(self,pos,quat,hand,phase):
        upward_or_hold = phase in ('test_lift','lift','hold')
        if upward_or_hold and not self._both_tactile_supported():
            self._recover_motion_support(phase)
        super().servo(pos,quat,hand,phase)
        if upward_or_hold and not self._both_tactile_supported():
            self._recover_motion_support(phase)

    def _recover_motion_support(self,phase):
        """Pause Cartesian progression; run the existing rule without recursion."""
        cfg,state = self.MOTION_RECOVERY,self.motion_recovery_state
        if not self.controller.active:
            raise ValueError('Motion support recovery requires an active RuleSupportController; lift blocked')
        if state['events'] >= cfg['max_events_per_trial'] or state['total_ticks'] >= cfg['max_ticks_per_trial']:
            state['stage'] = 'failed_budget_exhausted'
            raise ValueError('Motion support recovery budget exhausted; further motion blocked')
        s = self.sim
        frozen_pos = s.array(s.wrist.get_pos()).reshape(3).copy()
        frozen_quat = s.array(s.wrist.get_quat()).reshape(4).copy()
        state.update(stage='paused_for_support_recovery',events=state['events']+1,
                     consecutive_supported_samples=0)
        event = {'event':state['events'],'phase':phase,'start_step':int(s.steps),
                 'frozen_palm_position_m':frozen_pos.tolist(),
                 'frozen_palm_quaternion_wxyz':frozen_quat.tolist(),
                 'ticks':0,'result':'running'}
        self.motion_recovery_events.append(event)
        remaining = min(cfg['max_ticks_per_event'],cfg['max_ticks_per_trial']-state['total_ticks'])
        try:
            for _ in range(remaining):
                # Explicit parent call avoids recursively entering this gate.
                # Its ordinary _tick applies and acknowledges the real support
                # rule and writes the actual command/observation/evaluation.
                super().servo(frozen_pos,frozen_quat,s.target[6:].copy(),phase)
                state['total_ticks'] += 1
                event['ticks'] += 1
                supported = self._both_tactile_supported()
                state['consecutive_supported_samples'] = (
                    state['consecutive_supported_samples']+1 if supported else 0)
                if state['consecutive_supported_samples'] >= cfg['confirmation_samples']:
                    state['stage'] = 'support_reestablished'
                    event['result'] = 'recovered'
                    return
            state['stage'] = 'failed_persistent_support_loss'
            event['result'] = 'timeout'
            raise ValueError('Persistent thumb/index tactile not recovered at fixed palm pose; further motion blocked')
        except Exception as exc:
            if event['result'] == 'running':
                state['stage'] = 'failed_recovery_motion'
                event['result'] = 'motion_error'
            event['error'] = str(exc)
            raise
        finally:
            event['end_step'] = int(s.steps)
            event['final_consecutive_supported_samples'] = state['consecutive_supported_samples']
            write(self.out/'motion_recovery_events.json',self.motion_recovery_events)
