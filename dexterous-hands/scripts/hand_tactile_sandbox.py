"""
Interactive tactile-sensor sandbox on a dexterous hand.

Adapted from Genesis ``examples/sensors/tactile_sandbox.py``: instead of a fixed taxel
pad (box / dome), this loads a hand from the project ``ROBOT_REGISTRY`` and places
probe-based tactile sensors at the link-local probe positions defined by the robot's
``tactile_probe_cfgs`` metadata. One object per env (duck / torus / sphere / dragon)
is dropped near the hand's grasp center, and the hand is PD-driven through a sequence
of DOF target poses read from a YAML file -- snapping to the next pose every
``--interval`` seconds and cycling back to the first pose after the last. The object
target pose is teleop-controllable from the viewer (UP/DOWN/LEFT/RIGHT/J/K, N/M,
COMMA/PERIOD, BACKSLASH to reset).

Per-env tactile-sensor plots are shown (borrowed from ``scripts/sensors_viewer.py``):
one combined plot per hand, covering every sensor on that hand. Each sensor's patch
is rotated about its centroid so its mean probe normal aligns with the plot view
normal -- this flattens off-axis patches (e.g. the thumb) against the viewing plane.
The whole plot is then rotated 90 degrees clockwise.

DOF-sequence YAML format (joints not listed in a step keep their previous value)::

    sequence:
      - right_hand_thumb_bend_joint: 0.3
        right_hand_mid_joint1: 0.5
      - right_hand_thumb_bend_joint: 0.8
        right_hand_mid_joint1: 0.2

Usage:
    python scripts/hand_tactile_sandbox.py --robot wuji --sensors low-tips/force \
        --interval 5 --num_envs 4
"""

import argparse
import os
import tempfile

import genesis as gs
import genesis.utils.geom as gu
import numpy as np
import torch
import trimesh
import yaml
from eden.utils.assets import get_asset_path
from genesis.recorders.plotters import IS_MATPLOTLIB_AVAILABLE
from genesis.utils.geom import euler_to_quat
from genesis.vis.keybindings import Key, KeyAction, Keybind
from imgui_panels import DofSequencePanel, DofSliderPanel, render_camera_pose_button

import entities.robots  # noqa: F401  -- import for the ROBOT_REGISTRY registration side effect
from registry import ROBOT_REGISTRY
from tactile_sensors import probe_imperfection_params, TACTILE_SENSORS
from task_mods import TactileSensorsMod

# Object shown per env; env 0 is the duck. One object morph is built per env.
OBJ_PER_ENV_LABELS = ("duck", "torus", "sphere", "dragon")
MAX_ENVS = len(OBJ_PER_ENV_LABELS)

# Rough object extent (m). Small enough that a hand can enclose the object; the
# per-object scales below are the Genesis example's, rescaled by OBJECT_SIZE / 0.08.
OBJECT_SIZE = 0.035

# Hand base orientation, applied as a quaternion on the URDF morph.
HAND_EULER_DEG = (0.0, -90.0, 0.0)

# Object teleop step sizes for the viewer keybinds (genesis tactile_sandbox values).
KEY_DPOS = 0.004  # meters per HOLD tick
KEY_DROT_RAD = float(np.deg2rad(5.0))  # radians per HOLD tick

# Per-type vector-field plot tuning (scale_factor, max_magnitude, vector description),
# borrowed from scripts/sensors_viewer.py.
_VECTOR_FIELD_PLOT_OPTS: dict[str, tuple[float, float, str]] = {
    "elastomer": (0.005, 1.0, "marker displacement"),
    "force_torque": (0.005, 1.0, "force"),
    "force": (0.005, 1.0, "force"),
    "proximity": (0.005, 1.0, "force"),
}

# Rotation applied to every vector-field plot's 3D inputs (positions + per-frame vectors)
# so the projected 2D plot ends up rotated 90 degrees clockwise. ``gu.orthogonals`` builds
# a right-handed ``(u, v, normal)`` basis for the projection, so a -pi/2 rotation about
# ``plot_normal`` is equivalent to a 90-degree CW rotation in the plotted axes. Kept in
# sync between this script and scripts/sensors_viewer.py.
PLOT_ROTATION_ANGLE = -np.pi / 2


def _plot_rotation_quat(plot_normal) -> torch.Tensor:
    """Quaternion (wxyz) that rotates 3D vectors by ``PLOT_ROTATION_ANGLE`` about ``plot_normal``."""
    n = np.asarray(plot_normal, dtype=np.float64).reshape(3)
    n = n / max(float(np.linalg.norm(n)), 1e-12)
    half = PLOT_ROTATION_ANGLE / 2.0
    s = float(np.sin(half))
    q = np.array([float(np.cos(half)), n[0] * s, n[1] * s, n[2] * s], dtype=np.float32)
    return torch.as_tensor(q, device=gs.device)


def _align_normal_quat(from_vec: torch.Tensor, to_vec: torch.Tensor) -> torch.Tensor:
    """Quaternion (wxyz) that rotates unit ``from_vec`` onto unit ``to_vec`` via the shortest arc.

    Used to flatten each sensor's probe patch against the viewing plane: rotates the
    sensor's mean probe normal onto the plot's view normal so probes that sit at an
    angle to the camera (e.g. the thumb) end up parallel to the viewing plane.
    """
    device, dtype = from_vec.device, from_vec.dtype
    f = from_vec / torch.linalg.norm(from_vec).clamp(min=1e-12)
    t = to_vec.to(device=device, dtype=dtype)
    t = t / torch.linalg.norm(t).clamp(min=1e-12)
    dot = float((f * t).sum())
    if dot > 1.0 - 1e-9:
        return torch.tensor([1.0, 0.0, 0.0, 0.0], device=device, dtype=dtype)
    if dot < -1.0 + 1e-9:
        # Antiparallel -- rotate 180 about any axis perpendicular to f.
        helper = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype)
        if abs(float((f * helper).sum())) > 0.9:
            helper = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=dtype)
        axis = torch.linalg.cross(f, helper)
        axis = axis / torch.linalg.norm(axis).clamp(min=1e-12)
        return torch.cat([torch.zeros(1, device=device, dtype=dtype), axis])
    half = f + t
    half = half / torch.linalg.norm(half).clamp(min=1e-12)
    w = (f * half).sum().unsqueeze(0)
    xyz = torch.linalg.cross(f, half)
    return torch.cat([w, xyz])


def _build_object_morphs(object_scale: float) -> dict:
    """Object morphs keyed by label (duck / torus / sphere / dragon)."""
    torus_path = os.path.join(tempfile.gettempdir(), "hand_tactile_torus.obj")
    if not os.path.exists(torus_path):
        trimesh.creation.torus(major_radius=1.0, minor_radius=0.5).export(torus_path)

    ratio = (OBJECT_SIZE / 0.08) * object_scale  # scale relative to the Genesis example
    # Torus is halved and the other objects doubled relative to the Genesis example's sizes.
    return {
        "duck": gs.morphs.Mesh(file="meshes/duck.obj", euler=(90.0, 0.0, 0.0), scale=0.06 * ratio),
        "torus": gs.morphs.Mesh(file=torus_path, scale=OBJECT_SIZE * object_scale, convexify=False),
        "sphere": gs.morphs.Sphere(radius=OBJECT_SIZE * object_scale),
        "dragon": gs.morphs.Mesh(file="meshes/dragon/dragon.obj", euler=(90.0, 0.0, 90.0), scale=0.002 * ratio),
    }


def _controllable_dofs(robot) -> tuple[list[str], list[int], np.ndarray, np.ndarray]:
    """Per-DOF ``(names, local_idx, lower, upper)`` for the hand's actuated 1-DOF joints.

    Fixed / free / spherical joints are skipped. Non-finite limits (continuous joints)
    are clamped to ``[-pi, pi]`` so the YAML targets land in a usable range.
    """
    names: list[str] = []
    idx: list[int] = []
    lower: list[float] = []
    upper: list[float] = []
    for joint in robot.joints:
        if joint.n_dofs != 1 or joint.type in (gs.JOINT_TYPE.FIXED, gs.JOINT_TYPE.FREE, gs.JOINT_TYPE.SPHERICAL):
            continue
        names.append(joint.name)
        idx.append(int(joint.dofs_idx_local[0]))
        lo = float(joint.desc.dofs_limit[0, 0])
        hi = float(joint.desc.dofs_limit[0, 1])
        lower.append(lo if np.isfinite(lo) else -np.pi)
        upper.append(hi if np.isfinite(hi) else np.pi)
    return names, idx, np.asarray(lower, dtype=np.float64), np.asarray(upper, dtype=np.float64)


def _load_dof_sequence(
    dofs_path: str | None,
    dof_names: list[str],
    initial: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    """Build an ``(n_steps, n_dofs)`` array of clamped DOF targets from the YAML file.

    Each YAML step is a ``{joint_name: value}`` mapping; joints absent from a step keep
    their value from the previous step (the pose is accumulated). Returns just the
    ``initial`` pose as a single step when no file is given.
    """
    if not dofs_path:
        return initial.reshape(1, -1)

    with open(dofs_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    raw = data.get("sequence", data) if isinstance(data, dict) else data
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"DOF sequence YAML {dofs_path!r} must hold a non-empty 'sequence' list.")

    name_to_col = {name: i for i, name in enumerate(dof_names)}
    pose = initial.astype(np.float64).copy()
    steps: list[np.ndarray] = []
    for step_idx, step in enumerate(raw):
        if not isinstance(step, dict):
            raise ValueError(f"DOF sequence step {step_idx} must be a joint_name->value mapping.")
        for joint_name, value in step.items():
            col = name_to_col.get(joint_name)
            if col is None:
                print(f"[hand_tactile_sandbox] step {step_idx}: unknown joint {joint_name!r} ignored.")
                continue
            pose[col] = float(value)
        steps.append(pose.copy())
    return np.clip(np.stack(steps, axis=0), lower, upper)


def _interp_target(seq: np.ndarray, t: float, interval: float) -> np.ndarray:
    """DOF target at time ``t``: snap to ``seq[k % n]`` and hold for ``interval`` seconds.

    No smoothing between poses -- the commanded target jumps to the next pose and the
    hand's PD controller drives toward it directly.
    """
    if seq.shape[0] == 1:
        return seq[0]
    return seq[int(t // interval) % seq.shape[0]]


def _add_tactile_sensors(scene, robot, robot_opts, obj, placement: str, sensor_type: str, noisy: bool) -> list:
    """Add probe-based tactile sensors to the hand from the robot's probe-config JSON.

    ``robot`` is the live Genesis entity; ``robot_opts`` is the registry ``RobotOptions``
    that carries the probe-config metadata. Reuses ``TACTILE_SENSORS`` (sensor class +
    base params) and ``TactileSensorsMod._parse_probe_config`` (probe JSON parsing) so
    this sandbox stays consistent with how training tasks build the same sensors.

    Returns a list of ``(link_name, sensor, probe_local_normal)`` triples;
    ``probe_local_normal`` is a ``(n_probes, 3)`` tensor of the link-local probe normals
    parsed from the probe JSON (carried separately because ContactDepthProbe / ContactProbe
    don't expose ``probe_local_normal`` as a constructor field).
    """
    spec = TACTILE_SENSORS.get(sensor_type)
    if spec is None:
        raise ValueError(f"Unknown sensor type {sensor_type!r}. Choices: {sorted(TACTILE_SENSORS)}")
    if spec.placement != "probes":
        raise ValueError(f"Sensor type {sensor_type!r} is link-attached, not probe-based; this sandbox needs probes.")

    probe_cfgs = robot_opts.metadata.tactile_probe_cfgs
    resolution, subset = TactileSensorsMod._split_placement_key(placement)
    if resolution not in probe_cfgs:
        raise ValueError(
            f"Unknown probe resolution {resolution!r} for placement {placement!r}. Choices: {sorted(probe_cfgs)}"
        )
    probe_data = TactileSensorsMod._parse_probe_config(probe_cfgs[resolution])
    probe_data = TactileSensorsMod._filter_probe_data_by_subset(probe_data, robot_opts.metadata, subset)
    if not probe_data:
        raise ValueError(
            f"No probes match placement {placement!r} on {robot_opts.__class__.__name__} "
            f"(subset {subset!r} produced empty probe set)."
        )

    sensor_kwargs = dict(spec.params)
    sensor_kwargs["draw_debug"] = True

    sensor_takes_normal = "probe_local_normal" in spec.sensor_cls.model_fields
    link_idx_by_name = {link.name: link.idx_local for link in robot.links}

    sensors: list[tuple[str, object, torch.Tensor]] = []
    for link_name, probe in probe_data.items():
        link_idx = link_idx_by_name.get(link_name)
        if link_idx is None:
            print(f"[hand_tactile_sandbox] probe link {link_name!r} not found on robot; skipping.")
            continue
        kwargs = dict(probe_local_pos=probe["local_pos"], probe_radius=probe["radius"])
        if sensor_takes_normal:
            kwargs["probe_local_normal"] = probe["local_normal"]
        if spec.needs_track_link:  # point-cloud sensors need a tracked link to query geometry
            kwargs["track_link_idx"] = (obj.base_link_idx,)
        kwargs.update(sensor_kwargs)
        kwargs["probe_radius"] = probe["radius"]
        kwargs.update(probe_imperfection_params(sensor_type, "strong" if noisy else "clean",
                                               np.asarray(probe["local_pos"]).size // 3))
        sensor = scene.add_sensor(spec.sensor_cls(entity_idx=robot.idx, link_idx_local=link_idx, **kwargs))
        local_normal = torch.as_tensor(
            np.asarray(probe["local_normal"], dtype=np.float32).reshape(-1, 3), device=gs.device
        )
        # ``valid_mask`` filters out radius-0 padding probes (grid cells whose original
        # ray missed geometry); they're kept in the sensor for FFT-grid alignment but
        # excluded from plotting / mean-normal computation downstream.
        radii = torch.as_tensor(np.asarray(probe["radius"], dtype=np.float32).reshape(-1), device=gs.device)
        valid_mask = radii > 0.0
        sensors.append((link_name, sensor, local_normal, valid_mask))
    return sensors


def _sensor_env_magnitudes(sensor, n_envs: int) -> torch.Tensor:
    """Per-env scalar magnitude summary of one sensor reading -> ``(n_envs,)`` tensor."""
    data = sensor.read()
    values = data.force if hasattr(data, "force") else data  # KinematicTaxel/Proximity -> .force
    if not torch.is_tensor(values):
        values = torch.as_tensor(values)
    if values.ndim >= 2 and values.shape[-1] == 3:  # per-probe 3D vectors
        per_probe = torch.linalg.norm(values, dim=-1)
    else:
        per_probe = values.abs()
    flat = per_probe.reshape(n_envs, -1) if per_probe.numel() % n_envs == 0 else per_probe.reshape(1, -1)
    return flat.max(dim=-1).values


def _read_sensor_vectors(sensor_type: str, sensor, env_idx: int, n_envs: int) -> torch.Tensor:
    """One sensor's per-probe 3D vectors for env ``env_idx`` as a ``(n_probes, 3)`` tensor.

    Borrowed from scripts/sensors_viewer.py, extended to select a specific env.
    """
    data = sensor.read()
    # ElastomerTaxel.read() returns the displacement tensor; KinematicTaxel / ProximityTaxel
    # return a NamedTuple whose `.force` field holds the per-probe force.
    vectors = data if sensor_type == "elastomer" else data.force
    if vectors.ndim >= 3 and vectors.shape[0] == n_envs:  # ([n_envs,] ..., 3) -> pick env
        vectors = vectors[env_idx]
    return vectors.reshape(-1, 3)


def _link_world_poses(robot) -> dict:
    """World ``(pos, quat)`` of every robot link at the zero joint configuration.

    Computed by walking the link tree pre-build: each link carries its initial
    parent-relative pose (``link.desc.pos`` / ``link.desc.quat``), and the base link's initial
    pose is already in world frame. Genesis only resolves live link poses after
    ``scene.build``, but ``start_recording`` must run before it -- so the plot probe
    positions are derived here instead.
    """
    by_idx = {link.idx: link for link in robot.links}
    poses: dict[int, tuple] = {}

    def resolve(link):
        if link.idx in poses:
            return poses[link.idx]
        local_pos = torch.as_tensor(np.asarray(link.desc.pos, dtype=np.float32).reshape(3), device=gs.device)
        local_quat = torch.as_tensor(np.asarray(link.desc.quat, dtype=np.float32).reshape(4), device=gs.device)
        if link.parent_idx < 0:  # base link -- initial pose is already in world frame
            world = (local_pos, local_quat)
        else:
            parent_pos, parent_quat = resolve(by_idx[link.parent_idx])
            world = (
                gu.transform_by_trans_quat(local_pos, parent_pos, parent_quat),
                gu.transform_quat_by_quat(local_quat, parent_quat),
            )
        poses[link.idx] = world
        return world

    return {link.name: resolve(link) for link in robot.links}


def _plot_sensors_per_env(
    scene, sensor_type: str, sensors: list, n_envs: int, plot_normal: tuple, link_world_poses: dict
) -> None:
    """One combined sensor plot per env -- every sensor on that hand in a single figure.

    Adapted from scripts/sensors_viewer.py ``_plot_all_sensors``. Each sensor's patch is
    rotated about its centroid so its mean probe normal aligns with ``plot_normal`` --
    that flattens off-axis patches (the thumb sits at an angle to the rest) against the
    viewing plane. The whole plot is then rotated by ``PLOT_ROTATION_ANGLE`` about
    ``plot_normal`` so the layout appears rotated 90 degrees CW in the matplotlib axes.

    Must be called before ``scene.build`` (Genesis rejects ``start_recording`` after), so
    probe positions are placed using ``link_world_poses`` (zero-config forward
    kinematics) rather than live link poses.
    """
    if not IS_MATPLOTLIB_AVAILABLE:
        print("[hand_tactile_sandbox] matplotlib unavailable; skipping sensor plots.")
        return
    if not sensors:
        return

    if sensor_type not in _VECTOR_FIELD_PLOT_OPTS:
        # Scalar sensors (depth / bool / agg_force): one max-reading line plot per env.
        # No 3D projection here, so neither per-sensor flattening nor plot rotation apply.
        for env_idx in range(n_envs):
            scene.start_recording(
                lambda i=env_idx: tuple(float(_sensor_env_magnitudes(s, n_envs)[i]) for _, s, _, _ in sensors),
                gs.recorders.MPLLinePlot(
                    title=f"env {env_idx} ({OBJ_PER_ENV_LABELS[env_idx]}) -- {sensor_type} max reading",
                    labels=[link for link, _, _, _ in sensors],
                    x_label="step",
                    y_label="value",
                    history_length=200,
                ),
            )
        return

    scale_factor, max_magnitude, desc = _VECTOR_FIELD_PLOT_OPTS[sensor_type]
    plot_rot_q = _plot_rotation_quat(plot_normal)
    plot_normal_t = torch.as_tensor(np.asarray(plot_normal, dtype=np.float32).reshape(3), device=gs.device)

    # World-frame probe positions, concatenated across all sensors. Hand pose is
    # identical across envs, so one set of positions serves every env's plot. For each
    # sensor we (1) drop radius-0 padding probes, (2) flatten the patch about its
    # centroid by rotating its mean probe normal onto plot_normal, then (3) apply the
    # global CW plot rotation.
    world_positions = []
    per_sensor: list[tuple[object, torch.Tensor, torch.Tensor]] = []  # (sensor, align_quat, valid_mask)
    for link_name, sensor, local_normal, valid_mask in sensors:
        local_pos = sensor.probe_local_pos.reshape(-1, 3)
        link_pos, link_quat = link_world_poses[link_name]
        world_pos = gu.transform_by_trans_quat(local_pos, link_pos, link_quat)
        # Drop radius-0 padding probes from the plot's positions and from the patch-
        # normal mean (they have no real geometry; they're just grid alignment).
        if valid_mask is not None and not bool(valid_mask.all()):
            world_pos = world_pos[valid_mask]
            valid_normals = local_normal[valid_mask] if local_normal is not None else None
        else:
            valid_normals = local_normal
        if world_pos.shape[0] == 0:
            continue  # nothing left to plot for this sensor
        if valid_normals is not None and valid_normals.shape[0] > 0:
            world_normals = gu.transform_by_quat(valid_normals, link_quat)
            mean_normal = world_normals.mean(dim=0)
            align_q = _align_normal_quat(mean_normal, plot_normal_t)
        else:
            align_q = torch.tensor([1.0, 0.0, 0.0, 0.0], device=gs.device, dtype=torch.float32)
        centroid = world_pos.mean(dim=0)
        aligned_pos = centroid + gu.transform_by_quat(world_pos - centroid, align_q)
        world_positions.append(gu.transform_by_quat(aligned_pos, plot_rot_q))
        per_sensor.append((sensor, align_q, valid_mask))
    positions = torch.cat(world_positions, dim=0).detach().cpu().numpy()

    def make_reader(env_idx: int):
        """Concatenate every sensor's per-probe vectors for ``env_idx``, in plot frame."""

        def read_all() -> torch.Tensor:
            chunks = []
            for sensor, align_q, valid_mask in per_sensor:
                vectors = _read_sensor_vectors(sensor_type, sensor, env_idx, n_envs)
                if valid_mask is not None and not bool(valid_mask.all()):
                    vectors = vectors[valid_mask]
                link_quat = sensor._link.get_quat().reshape(-1, 4)[env_idx]
                world_v = gu.transform_by_quat(vectors, link_quat)
                aligned_v = gu.transform_by_quat(world_v, align_q)
                chunks.append(gu.transform_by_quat(aligned_v, plot_rot_q))
            return torch.cat(chunks, dim=0)

        return read_all

    for env_idx in range(n_envs):
        scene.start_recording(
            make_reader(env_idx),
            gs.recorders.MPLVectorFieldPlot(
                title=f"env {env_idx} ({OBJ_PER_ENV_LABELS[env_idx]}) -- {len(sensors)} {sensor_type} sensors ({desc})",
                positions=positions,
                normal=plot_normal,
                scale_factor=scale_factor,
                max_magnitude=max_magnitude,
            ),
        )


class _SandboxPanel:
    """ImGui side-panel for hand_tactile_sandbox: YAML sequence controls + manual DOF sliders.

    Composes the shared ``DofSequencePanel`` (pause / file load / cycle status) and
    ``DofSliderPanel`` (per-DOF sliders + Print YAML pose). Sliders are read-only while
    the YAML cycle is actively driving; pausing the cycle (or having no sequence loaded)
    makes them editable so the user can pose the hand by hand. Also exposes a button
    to print the current viewer camera pose as a ``ViewerOptions`` snippet.
    """

    def __init__(
        self,
        viewer: object,
        dof_panel: DofSliderPanel,
        seq_panel: DofSequencePanel,
        manual_targets: np.ndarray,
    ) -> None:
        self._viewer = viewer
        self._dof_panel = dof_panel
        self._seq_panel = seq_panel
        self._manual_targets = manual_targets

    def __call__(self, imgui) -> None:
        imgui.separator()
        imgui.text("Hand Tactile Sandbox")
        self._seq_panel.render(imgui)
        imgui.separator()
        cycle_active = self._seq_panel.sequence is not None and not self._seq_panel.paused
        # While the cycle is driving the hand, mirror the current YAML pose into the
        # slider-backing array so the read-only sliders show the live pose. While paused
        # we DON'T touch the array -- it's the slider's authoritative target, edits stick.
        if cycle_active:
            cur = self._seq_panel.current_step_target()
            if cur is not None:
                self._manual_targets[:] = cur
        self._dof_panel.render(imgui, self._manual_targets, readonly=cycle_active)
        imgui.separator()
        render_camera_pose_button(imgui, self._viewer, id_suffix="hand_sandbox")


def main() -> None:
    parser = argparse.ArgumentParser(description="Tactile-sensor sandbox on a dexterous hand.")
    parser.add_argument("--robot", default="wuji", help=f"Robot from the registry: {sorted(ROBOT_REGISTRY.keys())}")
    parser.add_argument(
        "--sensors",
        default="low-tips/force",
        help="Probe placement/sensor_type[/noisy], e.g. 'low-tips/force' or 'low-hand/depth/noisy'.",
    )
    parser.add_argument("--dofs", default=None, help="YAML file with a 'sequence' of DOF target poses.")
    parser.add_argument(
        "--interval", type=float, default=1.0, help="Seconds to hold each DOF target pose before advancing."
    )
    parser.add_argument(
        "--num_envs", "-b", type=int, default=MAX_ENVS, help=f"Number of envs / objects to show (1..{MAX_ENVS})."
    )
    parser.add_argument("--object_scale", type=float, default=1.0, help="Uniform scale multiplier for the objects.")
    parser.add_argument("--kp", type=float, default=None, help="Override hand position gain; otherwise preserve model defaults.")
    parser.add_argument("--kv", type=float, default=None, help="Override hand velocity gain; otherwise preserve model defaults.")
    parser.add_argument(
        "--plot_normal",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 1.0),
        metavar=("X", "Y", "Z"),
        help="View axis for the per-env sensor plots; probes are projected onto the plane perpendicular to it.",
    )
    parser.add_argument("--gpu", action="store_true", help="Run on GPU instead of CPU.")
    parser.add_argument("--headless", action="store_true", help="Run without the viewer GUI (also disables plots).")
    parser.add_argument("--seconds", type=float, default=30.0, help="Headless run duration (seconds).")
    args = parser.parse_args()

    if args.robot not in ROBOT_REGISTRY:
        raise SystemExit(f"Unknown robot {args.robot!r}. Choices: {sorted(ROBOT_REGISTRY.keys())}")
    placement, sensor_type, noisy = TactileSensorsMod._parse_sensors_str(args.sensors)

    n_envs = max(1, min(args.num_envs, MAX_ENVS))
    if n_envs != args.num_envs:
        print(f"[hand_tactile_sandbox] clamped num_envs to {n_envs} (valid range 1..{MAX_ENVS}).")

    gs.init(backend=gs.gpu if args.gpu else gs.cpu, precision="32", logging_level="info")

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(gravity=(0.0, 0.0, -9.81), substeps=1),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.2620, -0.2138, 0.3846),
            camera_lookat=(-0.3746, 0.3410, -0.1511),
            camera_fov=40.0,
            refresh_rate=60,
            enable_default_keybinds=False,
        ),
        vis_options=gs.options.VisOptions(
            show_world_frame=False,
            background_color=(1.0, 1.0, 1.0),
            shadow=False,
        ),
        profiling_options=gs.options.ProfilingOptions(show_FPS=True),
        show_viewer=not args.headless,
    )

    # Resolve the hand through Eden and rotate it by HAND_EULER_DEG.
    # MJCF models such as Wuji encode fixed-base/link behavior in the XML.
    robot_opts = ROBOT_REGISTRY.get(args.robot)()
    robot_file = get_asset_path(
        file=robot_opts.file,
        registry=robot_opts.registry,
        dataset=robot_opts.dataset,
        local_dir=robot_opts.local_dir,
    )
    hand_quat = tuple(float(q) for q in np.asarray(euler_to_quat(HAND_EULER_DEG)).reshape(4))
    morph_kwargs = dict(file=robot_file, collision=True, pos=(0.1, 0.1, 0.1), quat=hand_quat)
    if robot_file.lower().endswith(".xml"):
        hand_morph = gs.morphs.MJCF(**morph_kwargs)
    else:
        hand_morph = gs.morphs.URDF(**morph_kwargs, fixed=True, links_to_keep=robot_opts.links_to_keep)
    robot = scene.add_entity(morph=hand_morph)

    obj_morphs = _build_object_morphs(args.object_scale)
    obj = scene.add_entity(
        morph=[obj_morphs[label] for label in OBJ_PER_ENV_LABELS[:n_envs]],
        surface=gs.surfaces.Default(color=(1.0, 0.85, 0.1, 1.0)),
        material=gs.materials.Rigid(friction=0.5),
    )

    sensors = _add_tactile_sensors(scene, robot, robot_opts, obj, placement, sensor_type, noisy)
    if not sensors:
        print("[hand_tactile_sandbox] WARNING: no tactile sensors were placed on the hand.")

    dof_names, dof_idx, dof_lower, dof_upper = _controllable_dofs(robot)
    dof_initial = np.clip(np.zeros(len(dof_names), dtype=np.float64), dof_lower, dof_upper)

    # Shared mutable target the slider panel edits; mirrors the active YAML pose while
    # the cycle is unpaused so the sliders display the current pose, and is the
    # authoritative target whenever the cycle is paused or no YAML is loaded.
    manual_targets = dof_initial.copy()

    def _load_seq(path: str):
        """YAML loader bound to the current robot's DOF metadata; ``None`` on parse error."""
        try:
            return _load_dof_sequence(path, dof_names, dof_initial, dof_lower, dof_upper)
        except Exception as exc:  # noqa: BLE001 -- surface parse failures, don't kill the viewer
            print(f"[hand_tactile_sandbox] Failed to load YAML {path!r}: {exc}")
            return None

    seq_panel = DofSequencePanel(_load_seq, interval=args.interval)
    if args.dofs:
        seq_panel.set_sequence(args.dofs, _load_seq(args.dofs))
    dof_slider_panel = DofSliderPanel(lambda: robot, header="Hand DOFs", id_suffix="hand_sandbox")

    # Per-env sensor plots must be registered before build (Genesis rejects later).
    if not args.headless:
        _plot_sensors_per_env(scene, sensor_type, sensors, n_envs, tuple(args.plot_normal), _link_world_poses(robot))

    scene.build(n_envs=n_envs, env_spacing=(0.4, 0.4))

    # Place each object at the hand's grasp center (xy), then shift z so its AABB
    # bottom touches the palm link. Gravity is on, so the object's free-joint DOFs are
    # PD-held to this pose to keep it pinned in place against gravity and finger contact.
    grasp_center = torch.as_tensor(robot_opts.metadata.grasp_center, dtype=torch.float32, device=gs.device)
    obj_init_pos = gu.transform_by_trans_quat(grasp_center.expand(n_envs, 3), robot.get_pos(), robot.get_quat())
    obj.set_pos(obj_init_pos)
    palm_link = next(link for link in robot.links if link.name == robot_opts.metadata.palm_link)
    obj_aabb = obj.get_vAABB()  # (n_envs, 2, 3) world-frame min/max at the current pose
    obj_init_pos[..., 2] += palm_link.get_pos()[..., 2] - obj_aabb[..., 0, 2]
    obj.set_pos(obj_init_pos)
    obj_init_euler = gu.quat_to_xyz(obj.get_quat()).clone()  # radians; teleop-edited copy below
    obj_target_pos = obj_init_pos.clone()  # mutable -- edited by viewer keybinds (translate)
    obj_target_euler = obj_init_euler.clone()  # mutable -- edited by viewer keybinds (rotate)

    # PD-hold the objects in place against gravity / finger contact (free-joint DOFs).
    obj.set_dofs_kp(100, dofs_idx_local=slice(0, 6))
    obj.set_dofs_kv(10, dofs_idx_local=slice(0, 6))

    # PD position control for the hand DOFs.
    if args.kp is not None:
        robot.set_dofs_kp(torch.full((len(dof_idx),), args.kp, device=gs.device), dofs_idx_local=dof_idx)
    if args.kv is not None:
        robot.set_dofs_kv(torch.full((len(dof_idx),), args.kv, device=gs.device), dofs_idx_local=dof_idx)
    robot.set_dofs_position(torch.as_tensor(dof_initial, dtype=torch.float32, device=gs.device), dofs_idx_local=dof_idx)

    is_running = True
    if not args.headless:

        def stop() -> None:
            nonlocal is_running
            is_running = False

        # In-place edits of the (n_envs, 3) target tensors -- applied uniformly to every env.
        def translate(axis_idx: int, is_negative: bool) -> None:
            delta = (-1.0 if is_negative else 1.0) * KEY_DPOS
            obj_target_pos[..., axis_idx] += delta

        def rotate(axis_idx: int, is_negative: bool) -> None:
            delta = (-1.0 if is_negative else 1.0) * KEY_DROT_RAD
            obj_target_euler[..., axis_idx] += delta

        def reset_obj_pose() -> None:
            obj_target_pos.copy_(obj_init_pos)
            obj_target_euler.copy_(obj_init_euler)

        scene.viewer.register_keybinds(
            Keybind("quit", Key.ESCAPE, KeyAction.RELEASE, callback=stop),
            Keybind("move_forward", Key.UP, KeyAction.HOLD, callback=translate, args=(0, True)),
            Keybind("move_backward", Key.DOWN, KeyAction.HOLD, callback=translate, args=(0, False)),
            Keybind("move_right", Key.RIGHT, KeyAction.HOLD, callback=translate, args=(1, False)),
            Keybind("move_left", Key.LEFT, KeyAction.HOLD, callback=translate, args=(1, True)),
            Keybind("move_down", Key.J, KeyAction.HOLD, callback=translate, args=(2, True)),
            Keybind("move_up", Key.K, KeyAction.HOLD, callback=translate, args=(2, False)),
            Keybind("rotate_yaw_cw", Key.M, KeyAction.HOLD, callback=rotate, args=(2, True)),
            Keybind("rotate_yaw_ccw", Key.N, KeyAction.HOLD, callback=rotate, args=(2, False)),
            Keybind("rotate_roll_neg", Key.COMMA, KeyAction.HOLD, callback=rotate, args=(0, True)),
            Keybind("rotate_roll_pos", Key.PERIOD, KeyAction.HOLD, callback=rotate, args=(0, False)),
            Keybind("reset_obj", Key.BACKSLASH, KeyAction.RELEASE, callback=reset_obj_pose),
        )

        # ImGui side panel: YAML pause/load + manual DOF sliders. Optional -- degrades
        # gracefully when imgui-bundle isn't installed (the keybinds + cycle still work).
        try:
            from genesis.ext.pyrender.overlay import ImGuiOverlayPlugin

            imgui_plugin = ImGuiOverlayPlugin(show_sim_controls=False)
            scene.viewer.add_plugin(imgui_plugin)
            imgui_plugin.register_panel(
                _SandboxPanel(scene.viewer, dof_slider_panel, seq_panel, manual_targets),
                section="side",
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[hand_tactile_sandbox] ImGui control panel unavailable: {exc}")

    n_poses = int(seq_panel.sequence.shape[0]) if seq_panel.sequence is not None else 0
    print("\n=== Hand Tactile Sandbox ===")
    print(f"robot={args.robot}  sensors={args.sensors}  taxel-sensors={len(sensors)}  noise={'on' if noisy else 'off'}")
    print(f"envs={n_envs}  objects: {', '.join(OBJ_PER_ENV_LABELS[:n_envs])}")
    print(f"DOF poses={n_poses}  interval={args.interval}s  kp={args.kp}  kv={args.kv}")
    if args.headless:
        print(f"Running headless for {args.seconds}s ...")
    else:
        print("Press [ESC] to quit. Object teleop:")
        print("  [UP/DOWN/LEFT/RIGHT] move XY    [J/K] move Z   [N/M] yaw    [, / .] roll    [\\] reset")
    print()

    real_t = 0.0
    last_pose_idx = -1
    try:
        while is_running:
            # YAML cycle drives directly from ``seq_panel`` when active; pause hands the
            # hand over to ``manual_targets`` (the slider-backing array). The loop never
            # writes ``manual_targets`` -- only the slider mutations or the panel's
            # cycle-active mirror touch it, so slider edits while paused are never
            # clobbered by the loop.
            cycle_active = seq_panel.sequence is not None and not seq_panel.paused
            if cycle_active:
                pose_idx = seq_panel.current_pose_index()
                new_pose = pose_idx != last_pose_idx
                last_pose_idx = pose_idx
                target_np = seq_panel.current_step_target()
            else:
                new_pose = False
                last_pose_idx = -1  # so the next unpause re-prints the first pose's readout
                target_np = manual_targets

            robot.control_dofs_position(
                torch.as_tensor(target_np, dtype=torch.float32, device=gs.device), dofs_idx_local=dof_idx
            )
            # Pin each object's free-joint DOFs to the current teleop target so gravity
            # and finger contact can't carry it away from the user-commanded pose.
            obj.control_dofs_position(obj_target_pos, dofs_idx_local=slice(0, 3))
            obj.control_dofs_position(obj_target_euler, dofs_idx_local=slice(3, 6))

            # Report the strongest sensor reading per env on each YAML pose transition.
            if cycle_active and new_pose and sensors:
                per_env = (
                    torch.stack([_sensor_env_magnitudes(s, n_envs) for _, s, _, _ in sensors], dim=0).max(dim=0).values
                )
                readout = "  ".join(f"{OBJ_PER_ENV_LABELS[i]}={per_env[i].item():.4f}" for i in range(n_envs))
                print(f"t={seq_panel.seq_t:6.2f}s  pose#{last_pose_idx}  max|reading|  {readout}")

            scene.step()
            seq_panel.advance(scene.dt)
            real_t += scene.dt
            if args.headless and real_t >= args.seconds:
                break
    except KeyboardInterrupt:
        print("\nSimulation interrupted.")
    finally:
        print("Simulation finished.")


if __name__ == "__main__":
    main()
