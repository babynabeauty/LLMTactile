"""Collect fingertip slide trials over material plates, for material classification.

One XHand1 index fingertip presses onto a fixed plate and drags across it while a
per-link ``ContactForce`` sensor and a fingertip ``TemperatureGrid`` sensor are
logged. Every material is its own plate entity, so it carries its own friction and
its own thermal properties; the env index picks which plate the hand is placed
over, so a single batched build sweeps all of them at once. There is no floor and
the plates are narrow, so the three fingers next to the index one hang in open air
and never touch anything.

Press depth, drag speed, start offset, hand pitch and fingertip flexion are
randomized per env, so neither the absolute normal force nor the absolute
temperature drop is a giveaway on its own.

Writes an ``.npz`` consumed by ``scripts/material_id/fit_svm.py``.

Usage:
    python scripts/material_id/collect_slides.py --trials 32 --out data/material_id/slides.npz
    python scripts/material_id/collect_slides.py --probe          # pose/contact diagnostics
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import genesis as gs
import genesis.utils.geom as gu
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import features  # noqa: E402
from materials import (  # noqa: E402
    AMBIENT_TEMPERATURE,
    FINGERTIP_CONDUCTIVITY,
    FINGERTIP_DENSITY,
    FINGERTIP_EMISSIVITY,
    FINGERTIP_SPECIFIC_HEAT,
    FINGERTIP_TEMPERATURE,
    MATERIALS,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
HAND_URDF = str(REPO_ROOT / "src/assets/hands/xhand1/xhand_right_updated_effort.urdf")
TIP_LINK = "right_hand_index_rota_link2"

SIM_DT = 0.005

# --- scene layout (meters) ---
# Plates sit with their top face at z = 0. Each is narrower than the finger pitch
# and they are spaced far apart in y, so with the index finger centred over its own
# plate every other finger hangs over open air. There is no floor, so nothing else
# on the hand can touch anything.
PLATE_SIZE = (0.36, 0.025, 0.02)
PLATE_PITCH = 0.15

# Hand pose: pitched so the palm (and therefore the finger pads) faces down, plus
# a few degrees nose-down so the distal pad touches before the rest of the finger.
# The fingertip flexion cancels that extra pitch, putting the pad flat on the plate.
#
# The pitch is applied through the root DOF after the build, not on the morph: the
# TemperatureGrid sensor derives its voxel volume by transforming the link's world
# AABB corners into the link frame, which is only exact for an axis-aligned link.
# Building rotated collapses that volume and inflates every heat-transfer rate.
HAND_PITCH_DEG = 90.0 + 8.0
HAND_HOVER_Z = 0.004  # tip clearance above the plate before the press
MEASURE_Z = 0.15  # root height for the pre-trial fingertip-height measurement, clear of the plates
TIP_FLEX = np.deg2rad(8.0)  # index joint2, cancels the extra hand pitch
THUMB_BEND = 0.0  # bending swings the thumb down into the plate in this pose, so leave it out

# Fingertip friction is set low so the pairwise coefficient (Genesis takes the max
# of the two geoms) is the plate's alone.
FINGERTIP_FRICTION = 0.01

# Root PD. Gravity is compensated on the hand, so these gains only fight contact:
# the press force is roughly ROOT_KP_POS * commanded overlap.
ROOT_KP_POS = 2500.0
ROOT_KD_POS = 120.0
ROOT_Z_DAMPING = 150.0  # N*s/m on the press DOF: ~2 cm/s descent at a 2.5 N press
ROOT_KP_ROT = 300.0
ROOT_KD_ROT = 20.0
FINGER_KP = 60.0
FINGER_KD = 5.0
FINGER_DAMPING = 0.5  # N*m*s/rad of passive joint damping, settles the tip after touchdown
CONSTRAINT_TIMECONST = 0.02  # s, contact softness (see build_scene); sets ~0.33 mm penetration

# --- trial schedule (seconds) ---
T_WARMUP = 0.8  # unrecorded: lets the sensor's RC filter catch up to the tip temperature
T_SETTLE = 0.25
T_PRESS = 0.80
T_SLIDE = 1.60

# --- per-trial randomization ---
PRESS_FORCE_RANGE = (2.0, 3.0)  # N pressing the fingertip into the plate
SLIDE_SPEED_RANGE = (0.02, 0.06)  # m/s
START_X_RANGE = (-0.03, 0.03)
PITCH_JITTER_DEG = (-2.0, 2.0)
FLEX_JITTER_RAD = (-0.05, 0.05)
TEMP_OFFSET_RANGE = (-0.3, 0.3)  # per-sensor calibration offset, degC

# --- sensor noise, matching conf/sensor/tactile_params.yaml (link_force) and a
# thermistor-grade temperature channel ---
FORCE_NOISE = 0.08  # N per axis
FORCE_RANDOM_WALK = 0.01
FORCE_RESOLUTION = 0.005
TEMP_NOISE = 0.05  # degC
TEMP_RESOLUTION = 0.1
TEMP_TIME_CONSTANT = 0.15  # s, sensing element thermal lag

# The contact-heat model sizes the conducting patch as pi * weight * penetration,
# and penetration here is a third of a millimetre. This weight is the calibration
# that puts the fingertip's thermal response in the ~1 s range a real contact has:
# large enough that the plate's conductivity separates the classes over a 1.6 s
# drag, small enough that no material saturates at ambient.
CONTACT_DEPTH_WEIGHT = 12.0


def _sample(rng: np.random.Generator, lo: float, hi: float, n: int) -> np.ndarray:
    return rng.uniform(lo, hi, size=n).astype(np.float32)


def build_scene(show_viewer: bool, draw_debug: bool, contact_depth_weight: float = CONTACT_DEPTH_WEIGHT):
    """Build the hand + one plate per material, with the two fingertip sensors."""
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=SIM_DT),
        rigid_options=gs.options.RigidOptions(
            enable_self_collision=False,
            # An elliptic friction cone is essential here, not a preference. Under the
            # default pyramidal cone the tangential constraints bleed into the normal
            # one, so contact penetration comes out as a monotone function of the geom's
            # friction: 0.03 mm on steel against 1.8 mm on silicone rubber at the same
            # 2.5 N press. The contact-heat model keys off penetration, so that would
            # make the temperature channel a disguised friction sensor and the ablation
            # meaningless. With the elliptic cone every material penetrates identically.
            friction_cone=gs.friction_cone.elliptic,
            contact_resolution=gs.contact_resolution.signorini,
            # A light fingertip on a stiff plate chatters at this step size: the contact
            # collapses into a 9 N spike every third step instead of resting at the press
            # force. Softening the constraint holds contact for ~95% of the slide.
            constraint_timeconst=CONSTRAINT_TIMECONST,
        ),
        viewer_options=gs.options.ViewerOptions(camera_pos=(0.25, -0.25, 0.2), camera_lookat=(0.1, 0.0, 0.0)),
        profiling_options=gs.options.ProfilingOptions(show_FPS=False),
        show_viewer=show_viewer,
    )

    hand = scene.add_entity(
        gs.morphs.URDF(file=HAND_URDF, pos=(0.0, 0.0, 0.2), fixed=False),
        material=gs.materials.Rigid(friction=FINGERTIP_FRICTION, gravity_compensation=1.0),
    )
    plates = [
        scene.add_entity(
            gs.morphs.Box(size=PLATE_SIZE, pos=(0.0, i * PLATE_PITCH, -PLATE_SIZE[2] / 2), fixed=True),
            material=gs.materials.Rigid(friction=m.friction),
            surface=gs.surfaces.Default(color=(0.25 + 0.08 * i, 0.45, 0.75 - 0.06 * i, 1.0)),
        )
        for i, m in enumerate(MATERIALS)
    ]

    tip_idx_local = [link.name for link in hand.links].index(TIP_LINK)
    tip_link = hand.links[tip_idx_local]

    properties = {
        # Everything unlisted (palm, other fingers) is inert at ambient.
        -1: gs.sensors.TemperatureProperties(
            base_temperature=AMBIENT_TEMPERATURE,
            conductivity=FINGERTIP_CONDUCTIVITY,
            density=FINGERTIP_DENSITY,
            specific_heat=FINGERTIP_SPECIFIC_HEAT,
            emissivity=FINGERTIP_EMISSIVITY,
        ),
        tip_link.idx: gs.sensors.TemperatureProperties(
            base_temperature=FINGERTIP_TEMPERATURE,
            conductivity=FINGERTIP_CONDUCTIVITY,
            density=FINGERTIP_DENSITY,
            specific_heat=FINGERTIP_SPECIFIC_HEAT,
            emissivity=FINGERTIP_EMISSIVITY,
        ),
    }
    for plate, material in zip(plates, MATERIALS):
        properties[plate.base_link_idx] = gs.sensors.TemperatureProperties(
            base_temperature=AMBIENT_TEMPERATURE,
            conductivity=material.conductivity,
            density=material.density,
            specific_heat=material.specific_heat,
            emissivity=material.emissivity,
        )

    temp_sensor = scene.add_sensor(
        gs.sensors.TemperatureGrid(
            entity_idx=hand.idx,
            link_idx_local=tip_idx_local,
            grid_size=(1, 1, 1),
            properties_dict=properties,
            ambient_temperature=AMBIENT_TEMPERATURE,
            convection_coefficient=0.0,
            simulate_all_link_temperatures=False,
            sensor_time_constant=TEMP_TIME_CONSTANT,
            contact_depth_weight=contact_depth_weight,
            noise=TEMP_NOISE,
            resolution=TEMP_RESOLUTION,
            draw_debug=draw_debug,
            debug_temperature_range=(AMBIENT_TEMPERATURE, FINGERTIP_TEMPERATURE),
        )
    )
    force_sensor = scene.add_sensor(
        gs.sensors.ContactForce(
            entity_idx=hand.idx,
            link_idx_local=tip_idx_local,
            min_force=1e-4,
            max_force=50.0,
            noise=FORCE_NOISE,
            random_walk=FORCE_RANDOM_WALK,
            resolution=FORCE_RESOLUTION,
            draw_debug=draw_debug,
        )
    )
    return scene, hand, plates, tip_idx_local, temp_sensor, force_sensor


def _tip_aabb(hand, tip_idx_local: int) -> torch.Tensor:
    """Fingertip AABB, ``(n_envs, 2, 3)``."""
    aabb = hand.links[tip_idx_local].get_AABB()
    return aabb.unsqueeze(0) if aabb.ndim == 2 else aabb


def setup_trials(n_trials: int, seed: int) -> dict:
    """Per-env labels, randomized trial parameters, and the finger/root pose targets."""
    n_materials = len(MATERIALS)
    n_envs = n_materials * n_trials
    rng = np.random.default_rng(seed)

    labels = np.tile(np.arange(n_materials), n_trials).astype(np.int64)
    press_force = _sample(rng, *PRESS_FORCE_RANGE, n_envs)
    slide_speed = _sample(rng, *SLIDE_SPEED_RANGE, n_envs)
    start_x = _sample(rng, *START_X_RANGE, n_envs)
    pitch_jitter = np.deg2rad(_sample(rng, *PITCH_JITTER_DEG, n_envs))
    flex_jitter = _sample(rng, *FLEX_JITTER_RAD, n_envs)
    temp_offset = _sample(rng, *TEMP_OFFSET_RANGE, n_envs)
    slide_sign = rng.choice((-1.0, 1.0), size=n_envs).astype(np.float32)

    print(f"[collect] {n_envs} envs = {n_materials} materials x {n_trials} trials", flush=True)
    return dict(
        n_envs=n_envs,
        labels=labels,
        press_force=press_force,
        slide_speed=slide_speed,
        start_x=start_x,
        pitch_jitter=pitch_jitter,
        flex_jitter=flex_jitter,
        temp_offset=temp_offset,
        slide_sign=slide_sign,
    )


def run(args) -> dict:
    gs.init(backend=gs.cpu if args.cpu else gs.gpu, precision="32", logging_level="warning")
    scene, hand, _plates, tip_idx_local, temp_sensor, force_sensor = build_scene(args.vis, args.vis, args.heat_weight)

    trials = setup_trials(args.trials, args.seed)
    n_envs = trials["n_envs"]
    scene.build(n_envs=n_envs, env_spacing=(0.0, 0.0))

    device = gs.device
    n_dofs = hand.n_dofs
    q = torch.zeros((n_envs, n_dofs), dtype=torch.float32, device=device)

    tip_flex_dof = hand.get_joint("right_hand_index_joint2").dof_idx_local
    flex_jitter = torch.as_tensor(trials["flex_jitter"], device=device)
    q[:, tip_flex_dof] = TIP_FLEX + flex_jitter

    labels = torch.as_tensor(trials["labels"], device=device)
    plate_y = labels.to(torch.float32) * PLATE_PITCH
    start_x = torch.as_tensor(trials["start_x"], device=device)
    pitch = np.deg2rad(HAND_PITCH_DEG) + torch.as_tensor(trials["pitch_jitter"], device=device)
    q[:, 0] = start_x
    q[:, 1] = plate_y
    q[:, 2] = MEASURE_Z
    q[:, 4] = pitch
    q[:, hand.get_joint("right_hand_thumb_bend_joint").dof_idx_local] = THUMB_BEND

    hand.set_dofs_kp(torch.full((n_dofs,), FINGER_KP, device=device))
    hand.set_dofs_kv(torch.full((n_dofs,), FINGER_KD, device=device))
    hand.set_dofs_kp(torch.tensor([ROOT_KP_POS] * 3 + [ROOT_KP_ROT] * 3, device=device), dofs_idx_local=list(range(6)))
    hand.set_dofs_kv(torch.tensor([ROOT_KD_POS] * 3 + [ROOT_KD_ROT] * 3, device=device), dofs_idx_local=list(range(6)))

    # Place the fingertip from its measured pose rather than from URDF arithmetic:
    # the root DOF frame is offset from the palm origin, and the per-trial pitch and
    # flexion jitter move the tip by more than the press depth. Measured with the
    # hand held clear of the plates, so nothing is in contact.
    hand.set_dofs_position(q)
    hand.control_dofs_position(q)
    scene.step()
    aabb = _tip_aabb(hand, tip_idx_local)
    surface_z = MEASURE_Z - aabb[:, 0, 2]
    q[:, 0] += start_x - aabb[:, :, 0].mean(dim=1)
    q[:, 1] += plate_y - aabb[:, :, 1].mean(dim=1)
    q[:, 2] = surface_z + HAND_HOVER_Z
    hand.set_dofs_position(q)
    hand.zero_all_dofs_velocity()

    press_force = torch.as_tensor(trials["press_force"], device=device)
    slide_vel = torch.as_tensor(trials["slide_speed"], device=device) * torch.as_tensor(
        trials["slide_sign"], device=device
    )
    # The press is force-controlled, not position-controlled: a stiff servo commanded
    # below the surface hammers the plate at a few Hz instead of resting on it. Viscous
    # damping on the same DOF gives the descent a terminal velocity and a soft landing.
    hand.set_dofs_damping(torch.tensor([ROOT_Z_DAMPING], device=device), dofs_idx_local=[2])
    hand.set_dofs_damping(
        torch.full((n_dofs - 6,), FINGER_DAMPING, device=device), dofs_idx_local=list(range(6, n_dofs))
    )
    other_dofs = [0, 1, 3, 4, 5] + list(range(6, n_dofs))

    n_warmup = int(round(T_WARMUP / SIM_DT))
    n_settle = int(round(T_SETTLE / SIM_DT))
    n_press = int(round(T_PRESS / SIM_DT))
    n_slide = int(round(T_SLIDE / SIM_DT))
    n_steps = n_settle + n_press + n_slide

    target = q.clone()
    for _ in range(n_warmup):
        hand.control_dofs_position(target)
        scene.step()
        temp_sensor.read()

    temps, forces, link_forces = [], [], []
    for step in range(n_steps):
        if step >= n_settle + n_press:
            target[:, 0] += slide_vel * SIM_DT

        if step < n_settle:
            hand.control_dofs_position(target)
        else:
            hand.control_dofs_position(target[:, other_dofs], dofs_idx_local=other_dofs)
            hand.control_dofs_force(-press_force.unsqueeze(-1), dofs_idx_local=[2])
        scene.step()

        temps.append(temp_sensor.read().reshape(n_envs).clone())
        # Rotated into the world frame with the fingertip's own orientation, so the
        # plate normal is world +z for every trial: the same thing a real hand does
        # with forward kinematics before reasoning about a contact.
        tip_quat = hand.links[tip_idx_local].get_quat().reshape(n_envs, 4)
        forces.append(gu.transform_by_quat(force_sensor.read().reshape(n_envs, 3), tip_quat).clone())
        if args.probe:
            link_forces.append(hand.get_links_net_contact_force().clone())

    temperature = torch.stack(temps, dim=1).cpu().numpy().astype(np.float32)
    force = torch.stack(forces, dim=1).cpu().numpy().astype(np.float32)
    temperature += trials["temp_offset"][:, None]

    data = dict(
        temperature=temperature,
        force=force,
        labels=trials["labels"],
        material_names=np.array([m.name for m in MATERIALS]),
        friction=np.array([m.friction for m in MATERIALS], dtype=np.float32),
        conductivity=np.array([m.conductivity for m in MATERIALS], dtype=np.float32),
        fingertip_conductivity=np.float32(FINGERTIP_CONDUCTIVITY),
        press_force=trials["press_force"],
        slide_speed=trials["slide_speed"] * trials["slide_sign"],
        dt=np.float32(SIM_DT),
        n_settle=np.int32(n_settle),
        n_press=np.int32(n_press),
        n_slide=np.int32(n_slide),
    )
    if args.probe:
        report_probe(hand, tip_idx_local, trials, data, torch.stack(link_forces, dim=0))
    return data


def report_probe(hand, tip_idx_local: int, trials: dict, data: dict, link_force: torch.Tensor) -> None:
    """Print pose, contact and cooling diagnostics, using the same decomposition as the classifier."""
    labels = trials["labels"]
    n_settle, n_press = int(data["n_settle"]), int(data["n_press"])
    slide = slice(n_settle + n_press, None)

    print("\n[probe] link heights at rest (lowest AABB z, env 0):")
    for link in hand.links:
        aabb = link.get_AABB()
        if aabb.ndim == 3:
            aabb = aabb[0]
        z = float(aabb[0, 2])
        if z < 0.05:
            print(f"    {link.name:36s} {z * 1000:7.1f} mm")

    stray = link_force.norm(dim=-1).mean(dim=(0, 1))
    print("\n[probe] mean net contact force per hand link (N):")
    for link, value in zip(hand.links, stray.tolist()):
        if value > 0.05:
            print(f"    {link.name:36s} {value:7.2f}")

    force = np.asarray(data["force"], dtype=np.float64)
    temperature = np.asarray(data["temperature"], dtype=np.float64)
    f_n, f_t = features.split_normal_tangential(force)
    onset = features.contact_onset(f_n, default=n_settle + n_press)

    print("\n[probe] per material over the slide window:")
    header = f"{'material':10s} {'mu_set':>7s} {'mu_hat':>7s} {'f_n (N)':>8s} {'T_onset':>8s} {'T_end':>7s} {'dT':>6s}"
    print(f"    {header}")
    for i, m in enumerate(MATERIALS):
        sel = labels == i
        touching = f_n[sel][:, slide] > features.CONTACT_THRESHOLD
        mu_hat = f_t[sel][:, slide].sum(where=touching) / max(f_n[sel][:, slide].sum(where=touching), 1e-6)
        t_onset = temperature[sel][np.arange(sel.sum()), onset[sel]].mean()
        print(
            f"    {m.name:10s} {m.friction:7.2f} {mu_hat:7.3f} {f_n[sel][:, slide].mean():8.2f} "
            f"{t_onset:8.2f} {temperature[sel][:, -1].mean():7.2f} {temperature[sel][:, -1].mean() - t_onset:6.2f}"
        )
    print(f"\n[probe] contact fraction over the slide (f_n > 0.3 N): {(f_n[:, slide] > 0.3).mean():.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--trials", type=int, default=32, help="trials per material")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="data/material_id/slides.npz")
    parser.add_argument("--vis", action="store_true", help="show the viewer")
    parser.add_argument("--cpu", action="store_true", help="run on CPU")
    parser.add_argument("--probe", action="store_true", help="print pose/contact diagnostics")
    parser.add_argument(
        "--heat-weight", type=float, default=CONTACT_DEPTH_WEIGHT, help="contact_depth_weight, for recalibration"
    )
    args = parser.parse_args()

    data = run(args)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **data)
    print(f"[collect] wrote {out} ({data['temperature'].shape[0]} trials, {data['temperature'].shape[1]} steps)")


if __name__ == "__main__":
    main()
