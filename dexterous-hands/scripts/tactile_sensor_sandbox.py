"""
Interactive demo of tactile sensors on a fixed taxel pad (box or dome) with controllable objects.
Sensor configurations are pulled from ``src/tactile_sensors.py`` so what you see here matches what
the training tasks actually use.

Note that the sensor readings here have not been calibrated to any units, and is purely for visualization purposes.
"""

import argparse
import os
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

sys.path.insert(0, str(Path(__file__).resolve().parent / "plot"))
import genesis as gs  # noqa: E402
import genesis.utils.geom as gu  # noqa: E402
import numpy as np  # noqa: E402
import style  # noqa: E402, F401 — registers Hanken Grotesk as the default font.
import torch  # noqa: E402
import trimesh  # noqa: E402
from genesis.recorders.plotters import IS_MATPLOTLIB_AVAILABLE  # noqa: E402
from genesis.utils.misc import tensor_to_array  # noqa: E402
from genesis.vis.keybindings import Key, KeyAction, Keybind  # noqa: E402
from imgui_panels import format_camera_pose  # noqa: E402

from tactile_sensors import probe_imperfection_params, TACTILE_SENSORS  # noqa: E402

if TYPE_CHECKING:
    from genesis.engine.entities.rigid_entity import RigidEntity
    from genesis.engine.sensors.base_sensor import Sensor

KEY_DPOS = 0.001
FORCE_SCALE = 100.0
ROT_FORCE_SCALE = 100.0

GRID_SIZE = 20  # 20x20 taxels for square
PROBE_RADIUS = 0.004
OBJECT_INITIAL_CLEARANCE = 0.01
OBJECT_MAX_PENETRATION = 1e-3

SENSOR_OBJ_SIZE = 0.15
SENSOR_OBJ_Z = 0.05
OBJECT_SIZE = 0.08

OBJ_PER_ENV_LABELS = ("cow", "torus", "duck", "dragon")

# Probe-placement sensor types from the registry are the ones that can be wired
# up against a fixed taxel pad here. Link-placement sensors (`link_bool`,
# `link_force`) attach to a link, not a probe array, so they are excluded.
SANDBOX_SENSORS = tuple(name for name, spec in TACTILE_SENSORS.items() if spec.placement == "probes")


def _add_tactile_sensor(
    scene: gs.Scene,
    entity: "RigidEntity",
    link_idx_local: int,
    sensor_type: str,
    probe_local_pos: np.ndarray,
    probe_normal: tuple[float, float, float] | np.ndarray,
    track_link_idx: tuple[int, ...],
    noise: bool,
) -> "Sensor":
    """Instantiate ``sensor_type`` using kwargs from ``TACTILE_SENSORS`` in src/tactile_sensors.py."""
    spec = TACTILE_SENSORS[sensor_type]

    kwargs: dict = {
        "entity_idx": entity.idx,
        "link_idx_local": link_idx_local,
        "draw_debug": True,
        "probe_radius": PROBE_RADIUS,
    }
    # Pass probe positions with their natural shape: 2D grid ``(ny, nx, 3)`` for the plane case,
    # flat ``(n_probes, 3)`` for the dome case. Grid-aware sensors (ElastomerTaxel marker grid,
    # KinematicTaxel FFT crosstalk) need the 2D layout; others flatten internally either way.
    kwargs["probe_local_pos"] = probe_local_pos

    # ContactDepthProbe / ContactProbe ignore probe_local_normal; the others need it.
    if spec.sensor_cls not in (gs.sensors.ContactDepthProbe, gs.sensors.ContactProbe):
        kwargs["probe_local_normal"] = probe_normal

    if spec.needs_track_link:
        kwargs["track_link_idx"] = track_link_idx

    # Registry params (e.g. proximity's probe_radius=0.005) override the sandbox defaults.
    kwargs.update(spec.params)
    if noise:
        kwargs.update(probe_imperfection_params(sensor_type, "strong", probe_local_pos.size // 3))

    kwargs["debug_probe_color"] = (0.0, 0.0, 1.0)
    kwargs["debug_contact_color"] = (1.0, 0.0, 0.0)

    return scene.add_sensor(spec.sensor_cls(**kwargs))


def _plot_tactile_sensor(
    scene: gs.Scene,
    sensor_type: str,
    sensors: "tuple[Sensor, ...]",
    n_envs: int = 1,
    plot_normal: tuple[float, float, float] = (0.0, 0.0, -1.0),
) -> list:
    if not IS_MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available; skipping plot setup.")
        return []

    spec = TACTILE_SENSORS[sensor_type]
    cls = spec.sensor_cls

    plot_window_size = (480, 360)
    recorders: list = []

    if cls is gs.sensors.ElastomerTaxel:
        for env_idx in range(n_envs):
            for sensor in sensors:
                recorders.append(
                    scene.start_recording(
                        lambda s=sensor, i=env_idx: s.read()[i].reshape(-1, 3),
                        gs.recorders.MPLVectorFieldPlot(
                            title=f"({OBJ_PER_ENV_LABELS[env_idx % len(OBJ_PER_ENV_LABELS)]}) ElastomerTaxel marker displacements",
                            positions=sensor.probe_local_pos.reshape(-1, 3),
                            normal=plot_normal,
                            scale_factor=5.0,
                            max_magnitude=0.1,
                            window_size=plot_window_size,
                        ),
                    )
                )
    elif cls is gs.sensors.KinematicTaxel:
        for env_idx in range(n_envs):
            for sensor in sensors:
                recorders.append(
                    scene.start_recording(
                        lambda s=sensor, i=env_idx: s.read().force[i].reshape(-1, 3),
                        gs.recorders.MPLVectorFieldPlot(
                            title=f"({OBJ_PER_ENV_LABELS[env_idx % len(OBJ_PER_ENV_LABELS)]}) {sensor_type} force",
                            positions=sensor.probe_local_pos.reshape(-1, 3),
                            normal=plot_normal,
                            scale_factor=0.01,
                            max_magnitude=1.0,
                            window_size=plot_window_size,
                        ),
                    )
                )
    elif cls is gs.sensors.ProximityTaxel:
        for env_idx in range(n_envs):
            for sensor in sensors:
                recorders.append(
                    scene.start_recording(
                        lambda s=sensor, i=env_idx: s.read().force[i].reshape(-1, 3),
                        gs.recorders.MPLVectorFieldPlot(
                            title=f"({OBJ_PER_ENV_LABELS[env_idx % len(OBJ_PER_ENV_LABELS)]}) ProximityTaxel force",
                            positions=sensor.probe_local_pos.reshape(-1, 3),
                            normal=plot_normal,
                            scale_factor=0.1,
                            max_magnitude=1.0,
                            window_size=plot_window_size,
                        ),
                    )
                )
    elif cls is gs.sensors.ContactDepthProbe:
        for env_idx in range(n_envs):
            recorders.append(
                scene.start_recording(
                    lambda i=env_idx: tuple(sensor.read()[i].max() for sensor in sensors),
                    gs.recorders.MPLLinePlot(
                        title=f"{sensor_type} max depth ({OBJ_PER_ENV_LABELS[env_idx % len(OBJ_PER_ENV_LABELS)]})",
                        x_label="step",
                        y_label="depth",
                        history_length=200,
                        window_size=plot_window_size,
                    ),
                )
            )
    elif cls is gs.sensors.ContactProbe:
        for env_idx in range(n_envs):
            recorders.append(
                scene.start_recording(
                    lambda i=env_idx: tuple(sensor.read()[i].sum() for sensor in sensors),
                    gs.recorders.MPLLinePlot(
                        title=f"{sensor_type} taxels in contact ({OBJ_PER_ENV_LABELS[env_idx % len(OBJ_PER_ENV_LABELS)]})",
                        x_label="step",
                        y_label="# taxels",
                        history_length=200,
                        window_size=plot_window_size,
                    ),
                )
            )

    return recorders


def _finalize_vector_field_plots(recorders: list) -> None:
    """Post-build visual tweaks to MPLVectorFieldPlotter recorders: thicker arrows + colorbar on the left."""
    from genesis.recorders.plotters import MPLVectorFieldPlotter

    for rec in recorders:
        if not isinstance(rec, MPLVectorFieldPlotter) or rec.fig is None:
            continue
        # Quiver.draw() rebuilds the arrow polygons each frame from self.width, so bumping it here
        # carries through to every subsequent process() update.
        rec._quiver.width *= 1.8
        for cax in list(rec.fig.axes):
            if cax is not rec.ax:
                cax.remove()
        rec.fig.colorbar(rec._quiver, ax=rec.ax, location="left", label="Force Magnitude")
        rec.fig.canvas.draw()
        rec._background = rec.fig.canvas.copy_from_bbox(rec.ax.bbox)


def _print_sensor_reading(sensor_type: str, sensor: "Sensor", t: float) -> None:
    data = sensor.read()
    cls = TACTILE_SENSORS[sensor_type].sensor_cls
    if cls is gs.sensors.ElastomerTaxel:
        magnitude = torch.linalg.norm(data, dim=-1)
        if magnitude.max() > gs.EPS:
            print(f"t={t:.2f}s  max|displacement|={magnitude.max():.5f}")
    elif cls is gs.sensors.ContactDepthProbe:
        max_depth = data.max()
        if max_depth > gs.EPS:
            print(f"t={t:.2f}s  max depth={max_depth:.4f}")
    elif cls is gs.sensors.ContactProbe:
        n_contact = int(data.sum())
        if n_contact > 0:
            print(f"t={t:.2f}s  taxels in contact={n_contact}")
    elif cls is gs.sensors.KinematicTaxel:
        magnitude = torch.linalg.norm(data.force, axis=-1).max()
        if magnitude > gs.EPS:
            print(f"t={t:.2f}s  max|F|={magnitude:.4f}")
    elif cls is gs.sensors.ProximityTaxel:
        magnitude = torch.linalg.norm(data.force, dim=-1)
        if magnitude.max() > gs.EPS:
            print(f"t={t:.2f}s  mean|F|={magnitude.mean():.5f}  max|F|={magnitude.max():.5f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive tactile sandbox with selectable sensor type")
    parser.add_argument("--vis", "-v", action="store_true", default=False, help="Show visualization GUI")
    parser.add_argument("--gpu", action="store_true", help="Run on GPU instead of CPU")
    parser.add_argument("--num-envs", "-b", type=int, default=4, help="Number of parallel envs.")
    parser.add_argument(
        "--set-pos", action="store_true", help="Set object position directly instead of using control force."
    )
    parser.add_argument("--seconds", "-t", type=float, default=3.0, help="Seconds to simulate (headless mode)")
    parser.add_argument("--dome", action="store_true", help="Change the sensor object to a dome instead of a box")
    parser.add_argument(
        "--sensor",
        choices=SANDBOX_SENSORS,
        default="elastomer",
        help="Tactile sensor type (key from TACTILE_SENSORS in src/tactile_sensors.py).",
    )
    parser.add_argument(
        "--noise",
        action="store_true",
        help="Apply the sensor type's strong imperfection profile.",
    )
    args = parser.parse_args()

    gs.init(
        backend=gs.gpu if args.gpu else gs.cpu,
        precision="32",
        logging_level="info",
    )

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            gravity=(0.0, 0.0, 0.0),
            substeps=4,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.2374, 0.2305, 0.3257),
            camera_lookat=(-0.3138, -0.3047, -0.3144),
            camera_fov=40.0,
        ),
        vis_options=gs.options.VisOptions(
            show_world_frame=False,
            background_color=(1.0, 1.0, 1.0),
            shadow=False,
        ),
        profiling_options=gs.options.ProfilingOptions(
            show_FPS=True,
        ),
        show_viewer=args.vis,
    )

    if args.dome:
        sensor_morph = gs.morphs.Sphere(
            radius=SENSOR_OBJ_SIZE / 2,
            pos=(0.0, 0.0, -SENSOR_OBJ_SIZE / 2 + SENSOR_OBJ_Z),
            fixed=True,
        )
    else:
        sensor_morph = gs.morphs.Box(
            size=(SENSOR_OBJ_SIZE, SENSOR_OBJ_SIZE, SENSOR_OBJ_Z),
            pos=(0.0, 0.0, SENSOR_OBJ_Z / 2),
            fixed=True,
        )
    sensor_obj = scene.add_entity(
        morph=sensor_morph,
        surface=gs.surfaces.Default(
            color=(0.9, 0.9, 0.9, 1.0),
        ),
        material=gs.materials.Rigid(
            friction=0.6,
        ),
    )
    probe_normal_axis = (0.0, 0.0, 1.0)
    if args.dome:
        sphere_radius = SENSOR_OBJ_SIZE / 2
        probe_local_pos, probe_normal = gu.generate_ring_points_on_sphere(
            radius=sphere_radius,
            cap_axis=probe_normal_axis,
            n_rings=GRID_SIZE,
            arc_spacing=2.0 * PROBE_RADIUS,
            return_normals=True,
        )
    else:
        probe_normal = probe_normal_axis
        probe_z = SENSOR_OBJ_Z / 2
        probe_local_pos = gu.generate_grid_points_on_plane(
            lo=(-SENSOR_OBJ_SIZE / 2, -SENSOR_OBJ_SIZE / 2, probe_z),
            hi=(SENSOR_OBJ_SIZE / 2, SENSOR_OBJ_SIZE / 2, probe_z),
            normal=probe_normal_axis,
            nx=GRID_SIZE,
            ny=GRID_SIZE,
        )

    torus_path = os.path.join(tempfile.gettempdir(), "tactile_sandbox_torus.obj")
    if not os.path.exists(torus_path):
        trimesh.creation.torus(major_radius=1.0, minor_radius=0.75).export(torus_path)

    obj = scene.add_entity(
        morph=[
            gs.morphs.Mesh(
                file="src/assets/objects/cow.obj",
                euler=(90.0, 0.0, 0.0),
                scale=0.08,
                pos=(0.01, 0.04, 0.0),
            ),
            gs.morphs.Mesh(
                file=torus_path,
                euler=(0.0, 0.0, 0.0),
                scale=OBJECT_SIZE / 2,
                convexify=False,
            ),
            gs.morphs.Mesh(
                file="meshes/duck.obj",
                euler=(90.0, 0.0, 180.0),
                scale=0.03,
            ),
            gs.morphs.Mesh(
                file="meshes/dragon/dragon.obj",
                euler=(90.0, 0.0, 180.0),
                scale=0.001,
            ),
        ],
        surface=gs.surfaces.Default(color=(1.0, 1.0, 1.0, 1.0)),
        material=gs.materials.Rigid(friction=0.5),
    )

    sensor = _add_tactile_sensor(
        scene,
        sensor_obj,
        0,
        args.sensor,
        probe_local_pos,
        probe_normal,
        track_link_idx=(obj.base_link_idx,),
        noise=args.noise,
    )
    plot_recorders: list = []
    if args.vis and "PYTEST_VERSION" not in os.environ:
        plot_recorders = _plot_tactile_sensor(
            scene, args.sensor, (sensor,), n_envs=args.num_envs, plot_normal=probe_normal_axis
        )
    scene.build(n_envs=args.num_envs, env_spacing=(SENSOR_OBJ_SIZE * 1.5, SENSOR_OBJ_SIZE * 1.5))
    _finalize_vector_field_plots(plot_recorders)

    obj_init_pos = tensor_to_array(obj.get_pos())
    obj_init_quat = tensor_to_array(obj.get_quat())
    obj_aabb = tensor_to_array(obj.get_vAABB())
    obj_init_pos[..., 2] += SENSOR_OBJ_Z + OBJECT_INITIAL_CLEARANCE - obj_aabb[..., 0, 2]
    obj.set_pos(obj_init_pos)

    obj_target_pos = obj_init_pos.copy()
    obj_target_euler = gu.quat_to_xyz(obj_init_quat, rpy=True, degrees=True)
    obj_target_quat = obj_init_quat.copy()
    obj_contact_pos = obj_init_pos.copy()
    obj_contact_pos[..., 2] -= OBJECT_INITIAL_CLEARANCE + OBJECT_MAX_PENETRATION

    is_running = True

    if args.vis:
        obj.set_dofs_kp(FORCE_SCALE / KEY_DPOS, dofs_idx_local=slice(0, 3))
        obj.set_dofs_kp(ROT_FORCE_SCALE / KEY_DPOS, dofs_idx_local=slice(3, 6))
        obj.set_dofs_kv(0.1 * FORCE_SCALE / KEY_DPOS, dofs_idx_local=slice(0, 6))

        def stop():
            nonlocal is_running
            is_running = False

        def reset_pose():
            nonlocal obj_target_pos, obj_target_euler, obj_target_quat
            obj_target_pos = obj_init_pos.copy()
            obj_target_euler = gu.quat_to_xyz(obj_init_quat, rpy=True, degrees=True)
            obj_target_quat = obj_init_quat.copy()
            obj.set_pos(obj_init_pos)
            obj.set_quat(obj_init_quat)

        def translate(index: int, is_negative: bool):
            nonlocal obj_target_pos
            delta = (-1 if is_negative else 1) * KEY_DPOS
            obj_target_pos[..., index] += delta
            # obj_target_pos[..., 2] = np.maximum(
            #     obj_target_pos[..., 2], obj_init_pos[..., 2] - OBJECT_INITIAL_CLEARANCE - OBJECT_MAX_PENETRATION
            # )

        def rotate(axis_idx: int, is_negative: bool):
            nonlocal obj_target_euler, obj_target_quat
            delta = -5.0 if is_negative else 5.0
            obj_target_euler[..., axis_idx] += delta
            obj_target_quat = gu.euler_to_quat(obj_target_euler)

        scene.viewer.register_keybinds(
            Keybind("move_forward", Key.UP, KeyAction.HOLD, callback=translate, args=(0, False)),
            Keybind("move_backward", Key.DOWN, KeyAction.HOLD, callback=translate, args=(0, True)),
            Keybind("move_right", Key.RIGHT, KeyAction.HOLD, callback=translate, args=(1, True)),
            Keybind("move_left", Key.LEFT, KeyAction.HOLD, callback=translate, args=(1, False)),
            Keybind("move_down", Key.J, KeyAction.HOLD, callback=translate, args=(2, True)),
            Keybind("move_up", Key.K, KeyAction.HOLD, callback=translate, args=(2, False)),
            Keybind("rotate_cw", Key.M, KeyAction.HOLD, callback=rotate, args=(2, True)),
            Keybind("rotate_ccw", Key.N, KeyAction.HOLD, callback=rotate, args=(2, False)),
            Keybind("rotate_roll_left", Key.COMMA, KeyAction.HOLD, callback=rotate, args=(0, True)),
            Keybind("rotate_roll_right", Key.PERIOD, KeyAction.HOLD, callback=rotate, args=(0, False)),
            Keybind("reset", Key.BACKSLASH, KeyAction.RELEASE, callback=reset_pose),
            Keybind(
                "print_camera_pose",
                Key.P,
                KeyAction.PRESS,
                callback=lambda: print(format_camera_pose(scene.viewer)),
            ),
            Keybind("quit", Key.ESCAPE, KeyAction.RELEASE, callback=stop),
        )

    print("\n=== Tactile Sensor Sandbox ===")
    n_taxels = probe_local_pos.reshape(-1, 3).shape[0]
    layout = f"dome ({GRID_SIZE} latitude rings)" if args.dome else f"plane grid {probe_local_pos.shape[:-1]}"
    print(f"sensor={args.sensor}; taxels={n_taxels}; {layout}; noise={'on' if args.noise else 'off'}")
    if args.vis and IS_MATPLOTLIB_AVAILABLE:
        print("Matplotlib live plot enabled when supported.")
    if args.vis:
        print()
        print("Keyboard Controls:")
        print("  [UP/DOWN/LEFT/RIGHT] Move selected object in XY")
        print("  [j / k]              Lower / raise selected object")
        print("  [n / m]              Rotate object around Z axis")
        print("  [SPACE]              Select next object")
        print("  [\\]                  Reset objects")
        print("  [p]                  Print camera pose (paste-ready ViewerOptions snippet)")
    else:
        obj.set_pos(obj_contact_pos)
        print(f"Running headless for {args.seconds}s ...")
    print()

    try:
        while is_running:
            t = scene.t * scene.dt
            if args.vis:
                if args.set_pos:
                    obj.set_pos(obj_target_pos)
                    obj.set_quat(obj_target_quat)
                else:
                    obj.control_dofs_position(gu.quat_to_xyz(obj_target_quat), dofs_idx_local=slice(3, 6))
                    obj.control_dofs_position(obj_target_pos, dofs_idx_local=slice(0, 3))
            _print_sensor_reading(args.sensor, sensor, t)

            scene.step()

            if "PYTEST_VERSION" in os.environ:
                break
            if not args.vis and t >= args.seconds:
                break
    except KeyboardInterrupt:
        gs.logger.info("Simulation interrupted.")
    finally:
        gs.logger.info("Simulation finished.")


if __name__ == "__main__":
    main()
