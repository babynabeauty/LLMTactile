"""Frame the complete environment grid for the original single-video recorder."""

import math
from eden.options import CameraOptions
from eden.options.camera import CamerasOptions


def frame_env_grid(config):
    """Camera pose enclosing the whole env grid: ``(cam_pos, cam_lookat, cam_fov_degrees)``."""
    n = config.env_options.num_envs
    columns = math.ceil(math.sqrt(n))
    sx, sy = config.env_options.env_spacing
    width = ((n - 1) // columns) * sx
    depth = (min(n, columns) - 1) * sy
    origin = config.scene_options.obj.default_root_pos
    centered = config.env_options.center_envs_at_origin
    center = (origin[0] + (0 if centered else width / 2), origin[1] + (0 if centered else depth / 2), origin[2])
    radius = math.sqrt((width + 0.35) ** 2 + (depth + 0.35) ** 2 + 0.4 ** 2) / 2
    distance = radius / math.sin(math.radians(22.5)) * 1.1
    direction = (0.4, -0.6, 0.7)
    norm = math.sqrt(sum(v * v for v in direction))
    cam_pos = tuple(center[i] + distance * direction[i] / norm for i in range(3))
    return cam_pos, center, 45.0


def configure_checkpoint_camera(config, *, fit_existing=False):
    if config.cameras_options is None:
        config.cameras_options = CamerasOptions()
    if getattr(config.cameras_options, "rec", None) is not None and not fit_existing:
        return config
    cam_pos, cam_lookat, cam_fov = frame_env_grid(config)
    config.cameras_options.rec = CameraOptions(
        cam_pos=cam_pos,
        cam_lookat=cam_lookat,
        cam_fov=cam_fov,
        debug=False,
    )
    return config
