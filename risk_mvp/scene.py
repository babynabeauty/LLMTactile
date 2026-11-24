"""Batched grasp scene for the risk MVP: UR7e + XHand, the Sandbox platform, one YCB object per scene.

Same robot model, platform, floor, PD gains, effort limits and solver settings as environment.Sandbox
(1/300 s physics, 5 noslip iterations, 60 Hz control = 5 substeps), with the finger armature fixed at
0.003 (PLAN_v1). One object type per scene (Genesis cannot vary meshes across envs); per-env object
pose, contact friction and finger joint friction.

Friction: Genesis combines the two geoms' coefficients, so the object AND the hand are both set to
the same per-env mu (friction ratio relative to the 0.8 material). Joint friction (frictionloss on the
12 finger joints) is 0 in the vendor URDF and in Sandbox; here it is a per-env option, see README.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'ur7e_xhand_sim'))
sys.path.insert(0, str(ROOT))

from build_assets import ARM_JOINTS, HAND_JOINTS, build                       # noqa: E402
from platform_asset import FLOOR_Z, PLATFORM_POSITION, TABLE_TOP_Z, build_platform  # noqa: E402
from risk_mvp import objects as O                                             # noqa: E402
from risk_mvp.force_ctrl import FINGER_JOINTS, FINGER_LINKS                   # noqa: E402

BASE_FRICTION = .8
OBJECT_XY = (-.50, -.1333)          # demonstration position (multiobject_geometry.ObjectSpec.initial_position)
HOME_ARM_DEG = [0.00, -89.95, 89.96, -89.96, -89.95, -44.98]
KP = [4500, 4500, 3500, 2000, 2000, 2000]+[3.0]*12
KV = [450, 450, 350, 200, 200, 200]+[.08]*12
CONTROL_SUBSTEPS = 5


def cpu(x):
    return x.detach().cpu().numpy() if hasattr(x, 'detach') else np.asarray(x)


def quat_to_mat(q):
    q = np.asarray(q, float)
    w, x, y, z = np.moveaxis(q/np.linalg.norm(q, axis=-1, keepdims=True), -1, 0)
    return np.stack([np.stack([1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)], -1),
                     np.stack([2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)], -1),
                     np.stack([2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)], -1)], -2)


class GraspScene:
    def __init__(self, oid: str, n_envs: int, mesh_path: Optional[str] = None, mass_kg: Optional[float] = None,
                 armature: float = .003, backend: str = 'gpu', seed: int = 0, cameras: Sequence[dict] = (),
                 cam_res=(640, 480)):
        """cameras: dicts with 'offset' (camera position relative to the object's start centre) and optional
        'lookat_offset', 'fov'; rendered for env 0 (use n_envs=1 for replays)."""
        import genesis as gs
        self.gs = gs
        if not gs._initialized:
            gs.init(backend=getattr(gs, backend), precision='32', seed=seed, logging_level='warning')
        self.E = n_envs
        self.info = O.load(oid)
        self.oid = oid
        model, self.layouts = build()
        scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=1/300, substeps=1),
            rigid_options=gs.options.RigidOptions(enable_self_collision=True, enable_joint_limit=True,
                                                  enable_neutral_collision=True, batch_links_info=True,
                                                  batch_dofs_info=True, noslip_iterations=5, noslip_tolerance=1e-6),
            show_viewer=False)
        scene.add_entity(gs.morphs.Plane(pos=(0, 0, FLOOR_Z)))
        self.table = scene.add_entity(gs.morphs.URDF(file=str(build_platform()[0]), pos=PLATFORM_POSITION, fixed=True),
                                      material=gs.materials.Rigid(friction=BASE_FRICTION))
        self.robot = scene.add_entity(gs.morphs.URDF(file=str(model), fixed=True, pos=(0, 0, .2),
                                                     merge_fixed_links=True, links_to_keep=['right_hand_link', 'tool0']),
                                      material=gs.materials.Rigid(friction=BASE_FRICTION))
        self.mass_kg = float(mass_kg or self.info['mass_kg'])
        rho = self.mass_kg/self.info['mesh'].convex_hull.volume
        self.center0 = np.array([*OBJECT_XY, TABLE_TOP_Z+self.info['extent_m'][2]/2+.001])
        self.offset = np.asarray(self.info['offset'])
        self.obj = scene.add_entity(gs.morphs.Mesh(file=mesh_path or self.info['path'],
                                                   pos=tuple(self.center0+self.offset), convexify=True),
                                    material=gs.materials.Rigid(rho=rho, friction=BASE_FRICTION))
        self.scene = scene
        self.cams = [scene.add_camera(res=cam_res, pos=tuple(self.center0+np.asarray(c['offset'])),
                                      lookat=tuple(self.center0+np.asarray(c.get('lookat_offset', (0, 0, .03)))),
                                      fov=c.get('fov', 40), GUI=False) for c in cameras]
        scene.build(n_envs=n_envs)

        r = self.robot
        self.names = ARM_JOINTS+HAND_JOINTS
        self.idx = [r.get_joint(n).dofs_idx_local[0] for n in self.names]
        self.arm_idx, self.hand_idx = self.idx[:6], self.idx[6:]
        self.kp, self.kv = np.array(KP, float), np.array(KV, float)
        self.set_gains()
        import xml.etree.ElementTree as ET
        xml = ET.parse(model).getroot()
        self.effort = np.array([float(xml.find(f"joint[@name='{n}']/limit").get('effort')) for n in self.names])
        r.set_dofs_force_range(-self.effort, self.effort, self.idx)   # as v1_env (shared across envs)
        r.set_dofs_armature(np.full((n_envs, 12), armature, np.float32), self.hand_idx)
        lo, hi = r.get_dofs_limit(self.idx)
        self.lower, self.upper = cpu(lo).reshape(-1, 18)[0], cpu(hi).reshape(-1, 18)[0]
        self.home = np.r_[np.deg2rad(HOME_ARM_DEG), np.zeros(12)]
        self.wrist = r.get_link('right_hand_link')
        self.finger_link = {f: r.get_link(l) for f, l in FINGER_LINKS.items()}
        self.finger_cols = {f: [self.names.index(j) for j in FINGER_JOINTS[f]] for f in FINGER_LINKS}
        self.obj_link = self.obj.base_link.idx
        self.robot_links = (r.link_start, r.link_end)
        self.table_links = (self.table.link_start, self.table.link_end)
        self.target = np.tile(self.home, (n_envs, 1))

    # ------------------------------------------------------------------ setup
    def set_gains(self, finger_gain: Optional[np.ndarray] = None, fingers: Sequence[str] = ('thumb', 'index', 'middle')):
        """kp of the listed fingers' flexion joints x gain (kv x sqrt(gain)), as v1_env grip_gain."""
        kp = np.tile(self.kp, (self.E, 1))
        kv = np.tile(self.kv, (self.E, 1))
        if finger_gain is not None:
            g = np.broadcast_to(np.asarray(finger_gain, float), (self.E,))
            cols = [6+self.names[6:].index(j) for f in fingers for j in FINGER_JOINTS[f]]
            kp[:, cols] *= g[:, None]
            kv[:, cols] *= np.sqrt(g)[:, None]
        self.kp_env = kp
        self.robot.set_dofs_kp(kp, self.idx)
        self.robot.set_dofs_kv(kv, self.idx)

    def set_friction(self, mu):
        """Per-env contact friction for object and hand (Genesis combines the pair)."""
        mu = np.broadcast_to(np.asarray(mu, float), (self.E,))
        ratio = (mu/BASE_FRICTION)[:, None]
        self.obj.set_friction_ratio(np.tile(ratio, (1, self.obj.n_links)), np.arange(self.obj.n_links))
        self.robot.set_friction_ratio(np.tile(ratio, (1, self.robot.n_links)), np.arange(self.robot.n_links))

    def set_mass(self, mass_kg):
        self.obj.set_mass(np.broadcast_to(np.asarray(mass_kg, float), (self.E,)).copy())

    def restore(self, snap: dict):
        """Put every env into a snapshot.py state (one grasp, already touching), zero velocities."""
        self.scene.reset()
        q = np.tile(np.asarray(snap['q'], float), (self.E, 1))
        self.robot.set_dofs_position(q, self.idx, zero_velocity=True)
        self.target = np.tile(np.asarray(snap['target'], float), (self.E, 1))
        self.robot.control_dofs_position(self.target, self.idx)
        self.robot.set_dofs_kp(np.tile(np.asarray(snap['kp'], float), (self.E, 1)), self.idx)
        self.kp_env = np.tile(np.asarray(snap['kp'], float), (self.E, 1))
        kv = np.tile(self.kv, (self.E, 1))
        kv[:, 6:] *= np.sqrt(self.kp_env[:, 6:]/self.kp[6:])
        self.robot.set_dofs_kv(kv, self.idx)
        self.obj.set_quat(np.tile(np.asarray(snap['obj_quat'], float), (self.E, 1)), zero_velocity=True)
        self.obj.set_pos(np.tile(np.asarray(snap['obj_pos_file'], float), (self.E, 1)), zero_velocity=True)

    def set_joint_friction(self, frictionloss_Nm):
        f = np.broadcast_to(np.asarray(frictionloss_Nm, float), (self.E,))
        self.robot.set_dofs_frictionloss(np.repeat(f[:, None], 12, 1), self.hand_idx)

    def reset(self, center=None, yaw=None):
        """Robot home, object at center (E,3) (default the demo position) with yaw about z (E,)."""
        self.scene.reset()
        self.robot.set_dofs_position(np.tile(self.home, (self.E, 1)), self.idx, zero_velocity=True)
        self.target = np.tile(self.home, (self.E, 1))
        self.robot.control_dofs_position(self.target, self.idx)
        c = np.tile(self.center0, (self.E, 1)) if center is None else np.asarray(center, float)
        yaw = np.zeros(self.E) if yaw is None else np.asarray(yaw, float)
        q = np.stack([np.cos(yaw/2), 0*yaw, 0*yaw, np.sin(yaw/2)], 1)
        R = quat_to_mat(q)
        self.obj.set_quat(q, zero_velocity=True)
        self.obj.set_pos(c+np.einsum('eij,j->ei', R, self.offset), zero_velocity=True)

    def render(self):
        """RGB images (H, W, 3) uint8 from every camera."""
        out = []
        for cam in self.cams:
            rgb = cam.render(rgb=True)[0]
            out.append(np.asarray(cpu(rgb))[..., :3].astype(np.uint8))
        return out

    # ------------------------------------------------------------------ state
    def object_pose(self):
        """(center (E,3), R (E,3,3)) of the object frame (bounding-box centre) in the world."""
        q = cpu(self.obj.get_quat()).reshape(self.E, 4)
        R = quat_to_mat(q)
        p = cpu(self.obj.get_pos()).reshape(self.E, 3)-np.einsum('eij,j->ei', R, self.offset)
        return p, R

    def wrist_pose(self):
        return cpu(self.wrist.get_pos()).reshape(self.E, 3), cpu(self.wrist.get_quat()).reshape(self.E, 4)

    def q(self):
        return cpu(self.robot.get_dofs_position(self.idx)).reshape(self.E, 18)

    def ik(self, pos, quat):
        qpos, err = self.robot.inverse_kinematics(self.wrist, pos=np.asarray(pos, np.float32),
                                                  quat=np.asarray(quat, np.float32), dofs_idx_local=self.arm_idx,
                                                  max_solver_iters=100, return_error=True, seed=0)
        return cpu(qpos).reshape(self.E, -1)[:, self.arm_idx], cpu(err).reshape(self.E, -1)

    def step(self, target: np.ndarray, ticks: int = 1):
        self.target = np.clip(np.asarray(target, float), self.lower, self.upper)
        self.robot.control_dofs_position(self.target, self.idx)
        for _ in range(ticks*CONTROL_SUBSTEPS):
            self.scene.step()

    def finger_contacts(self):
        """Per finger: normal force on the object (E,), force-weighted contact point (E,3, world),
        inward normal (E,3), has_contact (E,). Also table/self/other robot-object contact summaries."""
        c = self.robot.get_contacts(is_padded=True)
        a, b = cpu(c['link_a']).astype(int), cpu(c['link_b']).astype(int)
        valid = cpu(c['valid_mask']).astype(bool)
        fa = cpu(c['force_a'])
        pos = cpu(c['position'])
        nrm = cpu(c['normal'])
        center, _ = self.object_pose()
        obj = self.obj_link
        on_obj = np.where((a == obj)[..., None], fa, -fa)          # force on the object at each contact
        to_c = center[:, None, :]-pos
        nrm = nrm*np.sign((nrm*to_c).sum(-1, keepdims=True)+1e-12)   # oriented into the object
        fn_all = np.clip((on_obj*nrm).sum(-1), 0, None)
        out = {}
        for f, link in self.finger_link.items():
            m = valid & (((a == link.idx) & (b == obj)) | ((b == link.idx) & (a == obj)))
            w = np.where(m, fn_all, 0.)
            tot = w.sum(1)
            has = m.any(1)
            p = (pos*w[..., None]).sum(1)/np.maximum(tot, 1e-9)[:, None]
            n = (nrm*w[..., None]).sum(1)
            n = n/np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
            tang = np.linalg.norm((on_obj*m[..., None]).sum(1)-(n*tot[:, None]), axis=1)
            out[f] = dict(fn=tot, point=p, normal=n, has=has, tangential=tang)
        rob = (a >= self.robot_links[0]) & (a < self.robot_links[1]) | (b >= self.robot_links[0]) & (b < self.robot_links[1])
        tab = (a >= self.table_links[0]) & (a < self.table_links[1]) | (b >= self.table_links[0]) & (b < self.table_links[1])
        mag = np.linalg.norm(fa, axis=-1)
        out['table_robot_N'] = np.where(valid & rob & tab, mag, 0.).sum(1)
        robot_obj = valid & (((a == obj) & rob) | ((b == obj) & rob))
        fingers_any = np.zeros_like(valid)
        for link in self.finger_link.values():
            fingers_any |= (a == link.idx) | (b == link.idx)
        out['other_robot_obj_N'] = np.where(robot_obj & ~fingers_any, mag, 0.).sum(1)
        return out

    def finger_jacobians(self):
        """{finger: (J_origin (E,6,2) over its flexion joints, link origin (E,3))}."""
        out = {}
        for f, link in self.finger_link.items():
            J = cpu(self.robot.get_jacobian(link)).reshape(self.E, 6, self.robot.n_dofs)
            cols = [self.idx[i] for i in self.finger_cols[f]]
            out[f] = (J[:, :, cols], cpu(link.get_pos()).reshape(self.E, 3))
        return out
