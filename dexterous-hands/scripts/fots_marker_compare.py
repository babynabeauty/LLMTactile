"""Track A of the FOTS sim-vs-real comparison: marker motion.

Compares our ``ElastomerTaxel`` marker-displacement field against the *real*
tracked GelSight markers shipped in ``reference/FOTS/marker_calib/data/*.csv``.

See ``docs/fots_comparison.md`` for the full protocol. The real-data side
(loading, gridding, metrics, plotting) runs standalone without Genesis::

    python scripts/fots_marker_compare.py --no-sim --frame 0          # view real dilate
    python scripts/fots_marker_compare.py --no-sim --frame 1          # view real shear

The sim side reproduces the Ø4.70 mm ball press in a Genesis scene and overlays
its ``ElastomerTaxel`` field on the real one::

    python scripts/fots_marker_compare.py --frame 0 --motion dilate --depth 0.5
    python scripts/fots_marker_compare.py --frame 1 --motion shear --depth 0.5 --shear 0.5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# FOTS sensor constants (reference/FOTS/params.py) + measured csv geometry.
# ---------------------------------------------------------------------------
MM_TO_PIXEL = 19.58  # px per mm
PX_TO_MM = 1.0 / MM_TO_PIXEL
BALL_RADIUS_MM = 4.70 / 2  # 2.35 mm calibration ball
N_COLS = 11  # markers along image x (params.N)
N_ROWS = 9  # markers along image y (params.M)
N_MARKERS = N_COLS * N_ROWS  # 99

IMG_W, IMG_H = 320, 240  # FOTS marker-calib frame size (landscape)

REPO_ROOT = Path(__file__).resolve().parents[1]
FOTS_DATA = REPO_ROOT / "reference" / "FOTS" / "marker_calib" / "data"

# Which csv frame is which motion (measured in docs/fots_comparison.md):
# 0,2,3,5 = dilate (identical); 1,4 = shear (identical); no clean twist.
FRAME_MOTION = {0: "dilate", 1: "shear", 2: "dilate", 3: "dilate", 4: "shear", 5: "dilate"}


# ===========================================================================
# Real data
# ===========================================================================
def load_real_markers(frame: int, smooth_sigma: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(pos_px, flow_px)`` each ``(N_ROWS, N_COLS, 2)`` in image pixels.

    The csv lists 99 markers as ``Mx,My,dx,dy``. We sort them into a row-major
    grid: 9 rows by ascending ``My``, then 11 cols by ascending ``Mx`` per row.

    FOTS's tracking is integer-pixel NCC, so the flow has ~±0.5 px per-marker noise on a
    ~2 px signal. The true gel deformation is spatially smooth, so ``smooth_sigma`` (in
    marker units) Gaussian-low-passes the flow field to suppress that high-frequency
    tracking noise (mag-weighted radial fraction 0.81 -> 0.92 at sigma 0.8). Positions
    are not smoothed.
    """
    arr = np.loadtxt(FOTS_DATA / f"{frame}.csv", delimiter=",", skiprows=1)
    if arr.shape[0] != N_MARKERS:
        raise ValueError(f"frame {frame}: expected {N_MARKERS} markers, got {arr.shape[0]}")
    pos = arr[:, 0:2]
    flow = arr[:, 2:4]

    # group into rows by My, then order each row by Mx
    row_order = np.argsort(pos[:, 1])
    pos, flow = pos[row_order], flow[row_order]
    pos = pos.reshape(N_ROWS, N_COLS, 2)
    flow = flow.reshape(N_ROWS, N_COLS, 2)
    col_order = np.argsort(pos[:, :, 0], axis=1)
    pos = np.take_along_axis(pos, col_order[..., None], axis=1)
    flow = np.take_along_axis(flow, col_order[..., None], axis=1)
    if smooth_sigma > 0:
        from scipy.ndimage import gaussian_filter

        flow = np.stack([gaussian_filter(flow[..., k], smooth_sigma, mode="nearest") for k in range(2)], axis=-1)
    return pos, flow


def estimate_contact_center(pos_px: np.ndarray, flow_px: np.ndarray) -> np.ndarray:
    """Estimate the dilation contact center (px) as the point flow radiates from.

    Radial flow means ``flow`` is parallel to ``(pos - c)``, i.e. their 2D cross
    product is zero: ``fx*(py-cy) - fy*(px-cx) = 0``. We first remove the global drift
    (mean flow, a sensor/tracking artifact that otherwise biases the center) and weight
    by |flow| so far-field noise doesn't dominate, then solve the linear LSQ for c.
    """
    p = pos_px.reshape(-1, 2)
    f = flow_px.reshape(-1, 2)
    f = f - f.mean(axis=0)  # remove global drift
    w = np.linalg.norm(f, axis=1)
    A = np.stack([f[:, 1], -f[:, 0]], axis=1) * w[:, None]  # [fy, -fx]
    b = (f[:, 1] * p[:, 0] - f[:, 0] * p[:, 1]) * w  # fy*px - fx*py
    c, *_ = np.linalg.lstsq(A, b, rcond=None)
    return c  # (cx, cy) px


def estimate_slide_dir(pos_px: np.ndarray, flow_px: np.ndarray) -> np.ndarray:
    """Recover the dominant shear/slide direction (unit vector, image px coords).

    Average the flow of the most-displaced markers (the contact region), where the
    coherent slide dominates the radial dilation component.
    """
    f = flow_px.reshape(-1, 2)
    mag = np.linalg.norm(f, axis=1)
    mask = mag >= np.percentile(mag, 75)
    mean = f[mask].mean(axis=0)
    n = np.linalg.norm(mean)
    return mean / n if n > 1e-9 else np.array([1.0, 0.0])


def markers_params_sig(args, motion, model, center_px, stop_px) -> str:
    """Canonical JSON of every input that changes a ``(motion, model)`` marker result -- the cache
    key for ``--save-markers``. Geometry/motion is always included; only the active model's shape
    params are. Floats are rounded so trivial formatting noise doesn't force a rerun.
    """

    def r(v):
        return round(float(v), 6) if isinstance(v, (int, float)) else v

    sig = {
        "motion": motion,
        "model": model,
        "frame": int(args.frame),
        "ball_radius": r(args.ball_radius),
        "depth": r(args.depth),
        "depth_query": args.depth_query,
        "control_mode": args.control_mode,
        "center_px": [r(center_px[0]), r(center_px[1])],
    }
    if motion == "shear":
        sig["stop_px"] = [r(stop_px[0]), r(stop_px[1])] if stop_px is not None else None
        sig["shear"] = r(args.shear)
        sig["start_rot_deg"], sig["stop_rot_deg"] = r(args.start_rot_deg), r(args.stop_rot_deg)
    if model == "elastomer":
        for k in (
            "compressibility",
            "elastomer_thickness",
            "lambda_d",
            "lambda_s",
            "dilate_scale",
            "shear_scale",
        ):
            sig[k] = r(getattr(args, k)) if getattr(args, k) is not None else None
    elif model == "fots":
        sig["fots_lambda_d"], sig["fots_lambda_s"] = r(args.fots_lambda_d), r(args.fots_lambda_s)
    elif model == "hydroshear":
        sig["hydroshear_lambda_d"] = r(args.hydroshear_lambda_d)
        sig["hydroshear_lambda_s"] = r(args.hydroshear_lambda_s)
        sig["hydroshear_mu"] = r(args.hydroshear_mu)
    return json.dumps(sig, sort_keys=True)


def markers_sweep_sig(args, motion, model, center_px, stop_px, axes) -> str:
    """Cache key for a sweep that auto-saves its best fit: the geometry/motion/model signature plus
    the swept grid (axis names + value lists). Re-running the SAME sweep -> same key -> skipped."""
    sig = json.loads(markers_params_sig(args, motion, model, center_px, stop_px))
    sig["sweep"] = sorted([[n, [round(float(v), 6) for v in vals]] for n, _kind, vals in axes])
    return json.dumps(sig, sort_keys=True)


def markers_cached(path, motion, model, sig: str) -> bool:
    """True if ``path`` already has a ``(motion, model)`` result whose stored param signature
    matches ``sig`` (so the experiment can be skipped instead of rerun)."""
    p = Path(path)
    if not p.exists():
        return False
    z = np.load(p, allow_pickle=False)
    pk, sk = f"{motion}__{model}__params", f"{motion}__{model}__sim"
    return pk in z.files and sk in z.files and str(z[pk]) == sig


def markers_bestline(path, motion, model) -> str | None:
    """The cached sweep's ``BEST: ...`` line for ``(motion, model)``, so a skipped sweep can re-emit
    it and staged calibration (dilate -> shear) still reads the winning params from a cache hit."""
    p = Path(path)
    if not p.exists():
        return None
    z = np.load(p, allow_pickle=False)
    k = f"{motion}__{model}__bestline"
    return str(z[k]) if k in z.files else None


def append_markers_npz(
    path,
    motion,
    model,
    pos_px,
    real_flow,
    sim_flow,
    depth_map,
    center_px,
    stop_px,
    frame,
    center_rot_deg=0.0,
    stop_rot_deg=0.0,
    params_sig=None,
    bestline=None,
):
    """Append one run's marker data to ``path`` (load-update-save), for the 2x4 grid plotter.

    Per-motion shared keys (``{motion}__pos/real/center/stop/frame/center_rot/stop_rot``) +
    per-(motion,model) keys (``{motion}__{model}__sim/depth/params``). ``stop`` is NaN when not a
    shear; ``center_rot``/``stop_rot`` are the indenter z-rotation (degrees); ``params`` is the
    JSON param signature used to cache/skip reruns.
    """
    p = Path(path)
    store = {k: v for k, v in np.load(p).items()} if p.exists() else {}
    store[f"{motion}__pos"] = np.asarray(pos_px, np.float32)
    store[f"{motion}__real"] = np.asarray(real_flow, np.float32)
    store[f"{motion}__center"] = np.asarray(center_px, np.float32)
    store[f"{motion}__stop"] = np.asarray(stop_px if stop_px is not None else [np.nan, np.nan], np.float32)
    store[f"{motion}__center_rot"] = np.array([center_rot_deg], np.float32)
    store[f"{motion}__stop_rot"] = np.array([stop_rot_deg], np.float32)
    store[f"{motion}__frame"] = np.array([frame], np.int32)
    store[f"{motion}__{model}__sim"] = np.asarray(sim_flow, np.float32)
    if depth_map is not None:
        store[f"{motion}__{model}__depth"] = np.asarray(depth_map, np.float32)
    if params_sig is not None:
        store[f"{motion}__{model}__params"] = np.array(params_sig)
    if bestline is not None:
        store[f"{motion}__{model}__bestline"] = np.array(bestline)
    np.savez(p, **store)


def real_extent_mm() -> tuple[float, float]:
    """Physical span (x, y) covered by the real marker grid, in mm."""
    pos, _ = load_real_markers(0)
    span_x = (pos[:, :, 0].max() - pos[:, :, 0].min()) * PX_TO_MM
    span_y = (pos[:, :, 1].max() - pos[:, :, 1].min()) * PX_TO_MM
    return float(span_x), float(span_y)


# ===========================================================================
# Sim (Genesis) — imported lazily so --no-sim works without genesis installed.
# ===========================================================================
PAD_THICKNESS_M = 0.005
PAD_MARGIN_M = 0.002  # physical box extends this much past the clamped-wall loop
BOUNDARY_PAD_FRAC = 0.10  # clamped-wall loop = imaged window expanded by this fraction (a bit outside the image)
CLEARANCE_M = 0.001  # ball start gap above pad surface
CTRL_KP = 5.0e4  # PD position-control stiffness (high enough to press to target depth)
CTRL_KV_RATIO = 0.05  # damping = CTRL_KP * this
RAMP_STEPS = 40  # steps to ramp into each pose (gradual = shear anchoring works)
HOLD_STEPS = 25  # steps to hold before reading (lets PD position-control settle to target)


def px_to_pad_xy(px: np.ndarray, grid_center_px: np.ndarray) -> np.ndarray:
    """Map image-pixel coords ``(..., 2)`` to pad-frame x,y (meters).

    Pad origin = real grid center. Image x (Mx, right) → pad +x. Image y (My, down)
    → pad −y (pad +y is up). Same convention as ``sim_disp_to_px``'s ``flip_y``.
    """
    x = (px[..., 0] - grid_center_px[0]) * PX_TO_MM / 1000.0
    y = -(px[..., 1] - grid_center_px[1]) * PX_TO_MM / 1000.0
    return np.stack([x, y], axis=-1)


def marker_grid_from_real(pos_px: np.ndarray, z: float) -> tuple[np.ndarray, np.ndarray]:
    """``ElastomerTaxel`` probe grid ``(N_ROWS, N_COLS, 3)`` placed at the *real*
    marker locations (mapped into the pad frame), so ``sim[row,col]`` and
    ``real[row,col]`` are guaranteed to be the same physical point — no ordering or
    regular-spacing assumptions. Returns ``(grid, grid_center_px)``.
    """
    grid_center_px = pos_px.reshape(-1, 2).mean(axis=0)
    xy = px_to_pad_xy(pos_px, grid_center_px)  # (N_ROWS, N_COLS, 2)
    grid = np.concatenate([xy, np.full((*xy.shape[:-1], 1), z)], axis=-1)
    return grid.astype(np.float32), grid_center_px


def sensor_boundary_loop(grid_center_px: np.ndarray, z: float, n_per_edge: int = 12,
                         pad_frac: float = BOUNDARY_PAD_FRAC) -> np.ndarray:
    """Closed loop ``(m, 3)`` tracing the gel's clamped housing wall in the pad-local frame.

    Used as ``ElastomerTaxel.elastomer_boundary``: the real GelSight gel is confined by the sensor
    frame a bit OUTSIDE the imaged area, so the incompressible lateral flow can't cross it (no-flux)
    and the marker stretch piles up toward it. We take the 320x240 px image rectangle, expand it by
    ``pad_frac`` about its center (default +10% so the wall sits just outside the image), and map it
    to pad coords the same way as the probes. The physical box is sized to enclose this loop.
    """
    mx, my = IMG_W * pad_frac / 2.0, IMG_H * pad_frac / 2.0       # expand the rect about its center
    lo_x, hi_x, lo_y, hi_y = -mx, IMG_W + mx, -my, IMG_H + my
    e = np.linspace(0.0, 1.0, n_per_edge, endpoint=False)
    full = np.full_like(e, 1.0)
    perim_px = np.concatenate(
        [  # CCW closed loop of the (expanded) image rectangle
            np.stack([lo_x + e * (hi_x - lo_x), lo_y * full], -1),  # bottom
            np.stack([hi_x * full, lo_y + e * (hi_y - lo_y)], -1),  # right
            np.stack([hi_x - e * (hi_x - lo_x), hi_y * full], -1),  # top
            np.stack([lo_x * full, hi_y - e * (hi_y - lo_y)], -1),  # left
        ],
        axis=0,
    )
    xy = px_to_pad_xy(perim_px, grid_center_px)  # (m, 2) meters, pad-local
    return np.concatenate([xy, np.full((xy.shape[0], 1), z)], axis=-1).astype(np.float32)


# Gaussian falloff is exp(-lambda * r^2), r in meters -> spread sigma = 1/sqrt(lambda).
# lambda ~3e4 -> sigma ~5.8 mm; ~2.5e5 -> ~2 mm; ~1e6 -> ~1 mm. Empirically the real FOTS
# marker field is BROAD: sweeps over lambda up to 1e6 show cos_sim peaks at the LOW end
# (~3e4) and collapses (0.7 -> 0.2) as lambda rises, for both dilate and shear. So the
# default sits at the fitted spread sweet spot, not the local regime.
DEFAULT_SENSOR_PARAMS = dict(
    lambda_d=3.0e4,
    lambda_s=3.0e4,
    dilate_scale=200.0,
    shear_scale=300.0,
    normal_exponent=1.2,
    compressibility=1.0,
    elastomer_thickness_mm=0.0,  # gel layer thickness (mm); >0 => incompressible-layer kernel (vs free-space 1/r)
)


def depth_grid_local(grid_center_px: np.ndarray, z: float, downsample: int = 1) -> tuple[np.ndarray, tuple[int, int]]:
    """Flat ``(N, 3)`` ContactDepthProbe grid spanning the full image, mapped to the
    pad frame, plus the ``(h, w)`` image shape for reshaping the depth readout.

    One probe per (downsampled) image pixel, so the depth map overlays the marker
    quiver in the same pixel coordinates.
    """
    us = np.arange(0, IMG_W, downsample)
    vs = np.arange(0, IMG_H, downsample)
    uu, vv = np.meshgrid(us, vs)  # (h, w)
    px = np.stack([uu, vv], axis=-1).astype(np.float64)
    xy = px_to_pad_xy(px, grid_center_px)  # (h, w, 2) meters
    flat = xy.reshape(-1, 2)
    grid = np.concatenate([flat, np.full((flat.shape[0], 1), z)], axis=1)
    return grid.astype(np.float32), (len(vs), len(us))


def build_scene(
    gpu: bool,
    vis: bool,
    pos_px: np.ndarray,
    sensor_params: dict | None = None,
    with_depth: bool = False,
    depth_downsample: int = 1,
    ball_radius_mm: float = BALL_RADIUS_MM,
    depth_query: str = "sdf",
    use_boundary: bool = True,
):
    """Flat gel pad + Ø4.70 mm ball + ElastomerTaxel(11x9) at real mm scale.

    The probe grid is placed at the *real* marker locations so sim/real markers
    correspond exactly by ``[row, col]``. When ``with_depth``, also adds a dense
    ContactDepthProbe (one probe per image pixel) for the depth-image backdrop.
    Returns ``(scene, ball, sensor, depth_sensor, depth_shape, pad_top_z)``.
    """
    import os
    import sys

    sys.path.insert(0, str(REPO_ROOT / "src"))
    import genesis as gs

    from tactile_sensors import TACTILE_SENSORS  # noqa: F401  (registry import parity w/ sandbox)

    sp = {**DEFAULT_SENSOR_PARAMS, **(sensor_params or {})}
    if not getattr(gs, "_initialized", False):  # gs.init is once-per-process (sweep rebuilds scenes)
        gs.init(backend=gs.gpu if gpu else gs.cpu, precision="32", logging_level="warning")
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(gravity=(0.0, 0.0, 0.0), substeps=4),
        vis_options=gs.options.VisOptions(show_world_frame=False, background_color=(1.0, 1.0, 1.0)),
        show_viewer=vis,
    )

    # Clamped-wall loop = imaged window + BOUNDARY_PAD_FRAC; size the (centered) box to enclose it +
    # PAD_MARGIN_M, so the physical gel is a bit bigger than the image and the wall sits inside it.
    grid_center_px = pos_px.reshape(-1, 2).mean(axis=0)
    pad_top_z = 0.0
    boundary_loop = sensor_boundary_loop(grid_center_px, PAD_THICKNESS_M / 2)
    half = np.abs(boundary_loop[:, :2]).max(axis=0)   # box centered at pad origin must contain the loop
    pad_w = float(2 * half[0] + 2 * PAD_MARGIN_M)
    pad_h = float(2 * half[1] + 2 * PAD_MARGIN_M)
    pad = scene.add_entity(
        morph=gs.morphs.Box(
            size=(pad_w, pad_h, PAD_THICKNESS_M),
            pos=(0.0, 0.0, pad_top_z - PAD_THICKNESS_M / 2),
            fixed=True,
        ),
        surface=gs.surfaces.Default(color=(0.9, 0.9, 0.9, 1.0)),
        material=gs.materials.Rigid(friction=0.9),
    )

    r_m = ball_radius_mm / 1000.0
    ball_z = pad_top_z + CLEARANCE_M
    if depth_query == "raycast":
        # Raycast ContactDepthProbe traverses the collision BVH (triangle faces), which an
        # analytic Sphere primitive doesn't populate -> use a tessellated mesh sphere instead.
        import tempfile

        import trimesh

        mesh_path = os.path.join(tempfile.gettempdir(), f"fots_icosphere_{ball_radius_mm:.3f}.obj")
        if not os.path.exists(mesh_path):
            trimesh.creation.icosphere(subdivisions=4, radius=r_m).export(mesh_path)
        ball_morph = gs.morphs.Mesh(file=mesh_path, pos=(0.0, 0.0, ball_z), convexify=True)
    else:
        ball_morph = gs.morphs.Sphere(radius=r_m, pos=(0.0, 0.0, ball_z))
    ball = scene.add_entity(
        morph=ball_morph,
        surface=gs.surfaces.Default(color=(0.3, 0.5, 0.9, 1.0)),
        material=gs.materials.Rigid(friction=0.9),
    )

    # Probe grid is in the pad's link-local frame, whose origin is the box CENTER.
    # The top face (world pad_top_z) is at local z = +thickness/2. Placing probes at
    # the real marker locations guarantees sim[row,col] == real[row,col] physically.
    probe_local_pos, _ = marker_grid_from_real(pos_px, PAD_THICKNESS_M / 2)
    elast_thick_m = sp.get("elastomer_thickness_mm", 0.0) / 1000.0  # mm -> m for the option
    # No-flux clamped wall (boundary_loop = image + BOUNDARY_PAD_FRAC, computed above). The engine's
    # wall-layer width is physical (~thickness/sqrt(3)), so the wall REQUIRES a gel thickness; enable
    # it only when thickness>0 and compressibility<1.
    boundary = boundary_loop if (elast_thick_m > 0.0 and use_boundary and sp["compressibility"] < 1.0) else None
    sensor = scene.add_sensor(
        gs.sensors.ElastomerTaxel(
            entity_idx=pad.idx,
            link_idx_local=0,
            probe_local_pos=probe_local_pos,
            track_link_idx=(ball.base_link_idx,),
            probe_radius=0.0008,
            # --- calibration knobs (defaults from conf/sensor/tactile_params.yaml) ---
            lambda_d=sp["lambda_d"],
            lambda_s=sp["lambda_s"],
            dilate_scale=sp["dilate_scale"],
            shear_scale=sp["shear_scale"],
            normal_exponent=sp["normal_exponent"],
            compressibility=sp["compressibility"],
            elastomer_boundary=boundary,
            elastomer_thickness=elast_thick_m,
            n_sample_points=1000,
            draw_debug=vis,
        )
    )
    depth_sensor, depth_shape = None, None
    if with_depth:
        # Small radius => reading ≈ -sdf (true indentation depth) inside the contact
        # patch, 0 elsewhere. depth_grid spans the full image in pad-frame coords.
        depth_grid, depth_shape = depth_grid_local(
            pos_px.reshape(-1, 2).mean(axis=0), PAD_THICKNESS_M / 2, depth_downsample
        )
        # SDF reads -sdf (unbounded inside), so a tiny radius gives a true depth map.
        # The BVH/raycast path only sees geometry within probe_radius and returns
        # (radius - dist), so it needs a radius >= the indentation depth to register.
        depth_probe_radius = 2.0e-3 if depth_query == "raycast" else 1.0e-4
        depth_kwargs = dict(
            entity_idx=pad.idx,
            link_idx_local=0,
            probe_local_pos=depth_grid,
            probe_radius=depth_probe_radius,
            draw_debug=False,
        )
        # contact_depth_query / probe_local_normal only exist on newer genesis (local
        # ./Genesis checkout). The pip-installed fork is SDF-only — feature-detect.
        opt_fields = gs.sensors.ContactDepthProbe.model_fields
        if "contact_depth_query" in opt_fields:
            depth_kwargs["contact_depth_query"] = depth_query
            # SDF mode ignores the probe normal; raycast casts along it — point it up
            # (+z), out of the gel surface toward the indenting ball.
            if depth_query == "raycast":
                depth_kwargs["probe_local_normal"] = (0.0, 0.0, 1.0)
        elif depth_query == "raycast":
            raise RuntimeError(
                "raycast ContactDepthProbe is not in the installed genesis. Install the local "
                "checkout: `uv pip install --no-deps -e ./Genesis` (from the workspace root)."
            )
        depth_sensor = scene.add_sensor(gs.sensors.ContactDepthProbe(**depth_kwargs))

    scene.build(n_envs=1)
    return scene, ball, sensor, depth_sensor, depth_shape, pad_top_z


def run_motion(
    scene,
    ball,
    sensor,
    pad_top_z: float,
    motion: str,
    depth_mm: float,
    shear_mm: float,
    center_xy_mm: tuple[float, float] = (0.0, 0.0),
    slide_dir_pad: tuple[float, float] = (1.0, 0.0),
    depth_sensor=None,
    depth_shape=None,
    ball_radius_mm: float = BALL_RADIUS_MM,
    control_mode: str = "position",
    start_rot_rad: float = 0.0,
    stop_rot_rad: float = 0.0,
):
    """Drive the ball and return ``(disp, depth_map)``.

    ``disp`` is the marker displacement ``(N_ROWS, N_COLS, 3)`` in meters; ``depth_map``
    is the ``(h, w)`` ContactDepthProbe indentation map in meters (or None).
    ``center_xy_mm`` shifts the press location to the real contact center;
    ``slide_dir_pad`` is the recovered unit slide direction for shear.

    ``control_mode``: "position" drives the 6 free-body DOFs with a stiff PD
    (``control_dofs_position``) — physical force-based control (kp high enough to penetrate
    the rigid pad to the target depth). "teleport" hard-sets the pose each step (``set_pos``
    + ``set_quat``). ``start_rot_rad``/``stop_rot_rad`` (radians) set the indenter z-rotation:
    held at ``start_rot_rad`` through the press, then ramped start->stop over the shear slide
    (0 keeps rotation pinned, the previous behavior).
    """
    import numpy as _np

    r = ball_radius_mm / 1000.0
    d = depth_mm / 1000.0
    cx, cy = center_xy_mm[0] / 1000.0, center_xy_mm[1] / 1000.0
    sdx, sdy = slide_dir_pad
    # ball center height for a press of penetration ``d`` into the pad surface
    z_pressed = pad_top_z + r - d
    z_start = pad_top_z + r + CLEARANCE_M

    if control_mode == "position":
        ball.set_dofs_kp(_np.full(ball.n_dofs, CTRL_KP, dtype=_np.float32))
        ball.set_dofs_kv(_np.full(ball.n_dofs, CTRL_KP * CTRL_KV_RATIO, dtype=_np.float32))
        ball.set_pos(_np.array([[cx, cy, z_start]], dtype=_np.float32))
        ball.set_quat(_np.array([[1.0, 0.0, 0.0, 0.0]], dtype=_np.float32))

    def ramp(p0, p1, n, rot0=0.0, rot1=0.0):
        # "position": stiff PD toward the target (xyz + z-rotation) — penetrates the rigid
        # pad to the target depth. "teleport": hard-set pose each step. ``rot0``/``rot1`` ramp
        # the z-rotation (radians) over the segment; 0 keeps rotation pinned to identity.
        for i in range(n):
            a = (i + 1) / n
            pos = _np.asarray(p0, dtype=_np.float32) * (1 - a) + _np.asarray(p1, dtype=_np.float32) * a
            rz = rot0 * (1 - a) + rot1 * a
            if control_mode == "teleport":
                ball.set_pos(pos[None, :])
                ball.set_quat(_np.array([[_np.cos(rz / 2), 0.0, 0.0, _np.sin(rz / 2)]], dtype=_np.float32))
            else:
                target = _np.concatenate([pos, [0.0, 0.0, rz]]).astype(_np.float32)
                ball.control_dofs_position(target[None, :])
            scene.step()

    # press straight down at the contact center (holding start rotation), then hold (or slide for shear)
    ramp((cx, cy, z_start), (cx, cy, z_pressed), RAMP_STEPS, start_rot_rad, start_rot_rad)
    if motion == "shear":
        s = shear_mm / 1000.0
        end = (cx + sdx * s, cy + sdy * s, z_pressed)
        ramp((cx, cy, z_pressed), end, RAMP_STEPS, start_rot_rad, stop_rot_rad)
        hold, hold_rot = end, stop_rot_rad
    else:
        hold, hold_rot = (cx, cy, z_pressed), start_rot_rad
    ramp(hold, hold, HOLD_STEPS, hold_rot, hold_rot)

    if control_mode == "position":  # report tracking so kp/kv can be tuned
        bz = float(_np.asarray(ball.get_pos())[0, 2])
        bq = _np.asarray(ball.get_quat())[0]
        rot_deg = 2.0 * _np.degrees(_np.arccos(_np.clip(abs(bq[0]), 0, 1)))
        tgt_deg = _np.degrees(hold_rot)
        print(
            f"  [ctrl] penetration target={d * 1000:.3f}mm achieved={(pad_top_z + r - bz) * 1000:.3f}mm  "
            f"rotation={rot_deg:.2f}deg (target {tgt_deg:.2f})  (kp={CTRL_KP:g})"
        )

    disp = sensor.read()[0].detach().cpu().numpy()  # (N_ROWS, N_COLS, 3)
    depth_map = None
    if depth_sensor is not None and depth_shape is not None:
        depth_map = depth_sensor.read()[0].detach().cpu().numpy().reshape(depth_shape)
        vmax, umax = _np.unravel_index(int(_np.argmax(depth_map)), depth_map.shape)
        print(
            f"  [debug] z_pressed={z_pressed * 1000:.3f}mm  max depth={depth_map.max() * 1000:.4f}mm  "
            f"dome@px=({umax},{vmax})  ball@pad=({cx * 1000:.2f},{cy * 1000:.2f})mm"
        )
    return disp, depth_map


def sim_disp_to_px(disp_m: np.ndarray, flip_y: bool = True) -> np.ndarray:
    """In-plane sim displacement (m, pad frame) -> image-pixel flow ``(N_ROWS, N_COLS, 2)``.

    ``flip_y`` accounts for image y pointing down vs pad +y up (see open items).
    """
    flow = disp_m[..., :2] * 1000.0 * MM_TO_PIXEL
    if flip_y:
        flow[..., 1] *= -1.0
    return flow


# FOTS marker-motion calibration (reference/FOTS/fots_test.py): (dilate, shear, twist) Gaussian falloffs (1/px^2).
FOTS_LAMB = (0.00125, 0.00021, 0.00038)


def fots_marker_bases(
    pos_px: np.ndarray,
    depth_map: np.ndarray,
    motion: str,
    slide_px: np.ndarray | None,
    center_px: np.ndarray | None,
    downsample: int = 2,
    lamb_d: float = FOTS_LAMB[0],
    lamb_s: float = FOTS_LAMB[1],
) -> dict[str, np.ndarray]:
    """Marker-flow bases from FOTS's published model (reference/FOTS/utils/marker_motion.py) driven by
    OUR ContactDepth, in image-pixel coords (directly comparable to the real csv flow):

      dilation:  D(m) = sum_src  h_src * (m - src) * exp(-lamb_d * |m-src|^2)
      shear:     S(m) = slide * exp(-lamb_s * |m - center|^2)

    Returned as SEPARATE bases ``{"dilate": D[, "shear": S]}`` so ``fit_bases`` scales each
    independently (matching how the ElastomerTaxel comparison is fit) -- otherwise a near-uniform
    broad shear is absorbed by the drift basis and lamb_s has no effect. ``lamb_d``/``lamb_s`` are
    FOTS's Gaussian falloffs (1/px^2; sigma_px = 1/sqrt(lamb)). Sources are the in-contact pixels of
    ``depth_map`` (meters -> mm = FOTS's ``h``), subsampled by ``downsample``.
    """
    H, W = depth_map.shape
    ys, xs = np.mgrid[0:H:downsample, 0:W:downsample]
    dm = depth_map[::downsample, ::downsample]
    m = dm > 1e-5
    src = np.stack([xs[m], ys[m]], axis=-1).astype(float)  # (S, 2) px, image coords (Mx, My)
    h_mm = dm[m] * 1000.0  # FOTS source weight h
    p = pos_px.reshape(-1, 2)
    D = np.zeros_like(p)
    if src.shape[0] > 0:
        diff = p[:, None, :] - src[None, :, :]  # (N, S, 2)
        g = np.exp(-lamb_d * (diff**2).sum(-1))  # (N, S)
        D = (h_mm[None, :, None] * diff * g[..., None]).sum(1)  # dilation outflow
    bases = {"dilate": D.reshape(pos_px.shape)}
    if motion == "shear" and slide_px is not None and center_px is not None:
        gs = np.exp(-lamb_s * ((p - center_px) ** 2).sum(-1))  # localized shear falloff
        S = np.asarray(slide_px)[None, :] * gs[:, None]
        bases["shear"] = S.reshape(pos_px.shape)
    return bases


def hydroshear_marker_bases(
    pos_px,
    grid_center_px,
    motion,
    center_xy_mm,
    slide_dir_pad,
    shear_mm,
    ball_radius_mm,
    depth_mm,
    lamb_d=16000.0,
    lamb_s=10000.0,
    mu=0.6,
    n_indenter_pts=1500,
    press_steps=10,
    slide_steps=12,
):
    """HydroShear/HydroFOTS marker bases (image px) from the ported reference model
    (``scripts/hydroshear_ref.py``), on the SAME sphere-on-flat-pad geometry as our other
    comparisons. Returns separate ``{"dilate": D[, "shear": S]}`` so ``fit_bases`` scales each
    independently (scales left at 1 here). Pad frame: surface z=0, gel z<0, sphere above.
    """
    import sys

    import torch

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from hydroshear_ref import HydroFotsRef, PlaneSDF, SphereSDF, fibonacci_hemisphere

    R, depth = ball_radius_mm / 1000.0, depth_mm / 1000.0
    cx, cy = center_xy_mm[0] / 1000.0, center_xy_mm[1] / 1000.0
    grid_shape = pos_px.shape[:-1]
    # tactile (marker) points on the pad surface (z=0), pad frame, meters
    xy = px_to_pad_xy(pos_px.reshape(-1, 2), grid_center_px)  # (T, 2)
    tactile = np.concatenate([xy, np.zeros((xy.shape[0], 1))], axis=1).astype(np.float32)
    tactile_t = torch.from_numpy(tactile)[None]  # (1, T, 3)

    center_pressed = (cx, cy, R - depth)
    hf = HydroFotsRef(SphereSDF(center_pressed, R), PlaneSDF(), lambda_d=lamb_d, lambda_s=lamb_s, mu=mu)

    # dilation: marker penetration into the (pressed) sphere x pairwise-Gaussian outflow
    Mdilate = hf.get_marker_dilation(tactile_t, tactile_t)[0].numpy()  # (T, 3) meters
    bases = {"dilate": sim_disp_to_px(Mdilate.reshape(*grid_shape, 3))}

    if motion == "shear":
        # stateful hydrosoft shear: ramp touch->press (builds normal force), then press->slide;
        # the field after the final slide step is the shear response.
        dirs = fibonacci_hemisphere(n_indenter_pts)  # (P, 3) unit
        hf.initialize(1, dirs.shape[0])
        sx, sy = slide_dir_pad[0] * shear_mm / 1000.0, slide_dir_pad[1] * shear_mm / 1000.0
        centers = [(cx, cy, R - a / press_steps * depth) for a in range(press_steps + 1)]
        centers += [
            (cx + a / slide_steps * sx, cy + a / slide_steps * sy, R - depth) for a in range(1, slide_steps + 1)
        ]
        Mshear = None
        for c in centers:
            ind_pts = torch.from_numpy(np.asarray(c, np.float32)[None, :] + R * dirs)[None]  # (1, P, 3)
            Mshear = hf.get_marker_shear(tactile_t, ind_pts)
        bases["shear"] = sim_disp_to_px(Mshear[0].numpy().reshape(*grid_shape, 3))
    return bases


def eval_marker_bases(
    scene,
    ball,
    sensor,
    pad_top_z,
    motion,
    depth_mm,
    shear_mm,
    center_xy_mm,
    slide_dir_pad,
    ball_radius_mm,
    read_params=None,
    fixed_dilate_scale=None,
    depth_sensor=None,
    depth_shape=None,
    control_mode="position",
    start_rot_rad=0.0,
    stop_rot_rad=0.0,
):
    """Run the unit-scale basis fields for a fit; return ``(bases, fixed_offset, depth_map)``.

    Always runs the dilation basis (dilate_scale=1, shear=0). For shear motions also runs
    the shear basis (dilate=0, shear=1). If ``fixed_dilate_scale`` is given (staged calib:
    dilation already locked from the dilate sweep), the dilation is returned as a fixed
    offset and only "shear" is left to fit; otherwise both are fit.
    """

    def run_unit(dsc, ssc, want_depth):
        kw = {"dilate_scale": dsc, "shear_scale": ssc}
        if read_params:
            kw.update({k: v for k, v in read_params.items() if k in MUTABLE_PARAMS})
        set_sensor_scales(sensor, **kw)
        scene.reset()
        disp, dep = run_motion(
            scene,
            ball,
            sensor,
            pad_top_z,
            motion,
            depth_mm,
            shear_mm,
            center_xy_mm,
            slide_dir_pad,
            depth_sensor if want_depth else None,
            depth_shape if want_depth else None,
            ball_radius_mm,
            control_mode,
            start_rot_rad=start_rot_rad,
            stop_rot_rad=stop_rot_rad,
        )
        return sim_disp_to_px(disp), dep

    D, depth_map = run_unit(1.0, 0.0, depth_sensor is not None)
    if motion != "shear":
        return {"dilate": D}, None, depth_map
    S, _ = run_unit(0.0, 1.0, False)
    if fixed_dilate_scale is not None:
        return {"shear": S}, fixed_dilate_scale * D, depth_map
    return {"dilate": D, "shear": S}, None, depth_map


# ===========================================================================
# Metrics + plotting
# ===========================================================================
def fit_bases(
    real_flow: np.ndarray,
    bases: dict[str, np.ndarray],
    fixed: np.ndarray | None = None,
    weight_by_real: bool = True,
    fit_drift: bool = True,
    weight_power: float = 2.0,
) -> dict:
    """Best-fit the real 2D marker flow as a (fixed offset + uniform drift +) combo of sim bases.

    ``bases`` maps name -> per-marker 2D flow (px) at *unit* scale: "dilate" (dilate_scale=1,
    shear_scale=0) and/or "shear" (dilate=0, shear=1). Solve ``real ~= fixed + drift + sum c_i*B_i``
    (LSQ over in-plane components; sim z dropped upstream). Coefficients ARE the optimal scales
    (ratio kept, not normalized). ``fixed`` locks an already-calibrated part (dilation when
    fitting shear). Two key choices, both default-on:
    - ``weight_by_real``: weight each marker by |real|**weight_power so the fit matches where the
      signal IS (the contact bulge), not the noisy far-field. ``weight_power=2`` focuses on the
      strong bulge markers, which selects the lambda whose outflow PEAK matches the real bulge's
      (|real|**1 is dragged toward a too-spread lambda that peaks off-sensor).
    - ``fit_drift``: add a uniform (dx,dy) translation basis to absorb global sensor/tracking
      drift, so the dilation captures the radial bulge rather than the drift.
    """
    real2 = real_flow.reshape(-1, 2)
    real_full = real2.ravel()
    target = real_full - (fixed.reshape(-1, 2).ravel() if fixed is not None else 0.0)
    names = list(bases)
    cols = [bases[n].reshape(-1, 2).ravel() for n in names]
    n_basis = len(cols)
    if fit_drift:
        n = real2.shape[0]
        ex = np.zeros(2 * n)
        ex[0::2] = 1.0
        ey = np.zeros(2 * n)
        ey[1::2] = 1.0
        cols += [ex, ey]
    A = np.stack(cols, axis=1)  # (2N, k[+2])
    w = np.repeat(np.linalg.norm(real2, axis=1) ** weight_power, 2) if weight_by_real else np.ones(real_full.size)
    sw = np.sqrt(w)
    coef, *_ = np.linalg.lstsq(A * sw[:, None], target * sw, rcond=None)
    resid = target - A @ coef
    rmse = float(np.sqrt(np.sum(w * resid**2) / (np.sum(w) + 1e-12)))
    real_rms = float(np.sqrt(np.sum(w * real_full**2) / (np.sum(w) + 1e-12)))
    total = (A @ coef) + (fixed.reshape(-1, 2).ravel() if fixed is not None else 0.0)
    drift = coef[n_basis:].tolist() if fit_drift else [0.0, 0.0]
    return {
        "scaled_rmse_px": rmse,  # <-- objective (mag-weighted, scale-invariant)
        "rel_rmse": rmse / (real_rms + 1e-12),  # 0=perfect, ~1=no better than zero field
        "scales": {n: float(c) for n, c in zip(names, coef[:n_basis])},  # optimal scale(s)
        "drift_px": [float(d) for d in drift],  # absorbed global drift (px)
        "pred_flow": total.reshape(real_flow.shape),  # full best-fit sim flow for plotting
    }


def plot(
    pos_px: np.ndarray,
    real_flow: np.ndarray,
    sim_flow: np.ndarray | None,
    title: str,
    out: Path | None,
    frame: int,
    sim_depth: np.ndarray | None = None,
    center_px: np.ndarray | None = None,
    stop_px: np.ndarray | None = None,
    marker_label: str = "ElastomerTaxel",
    center_rot_deg: float = 0.0,
    stop_rot_deg: float = 0.0,
):
    """Side-by-side: real GelSight image + tracked flow, vs sim ElastomerTaxel flow.

    Both panels share image-pixel coords (origin top-left, y down) so orientation
    is directly comparable; quiver dy is used as-is (positive = down) on a y-down axis.
    When ``sim_depth`` is given, it is drawn as the backdrop of the sim panel (the
    ContactDepthProbe indentation map), mirroring the real image backdrop.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    p = pos_px.reshape(-1, 2)
    rf = real_flow.reshape(-1, 2)
    img_path = FOTS_DATA / f"{frame}.png"
    img = None
    try:
        from PIL import Image

        img = np.asarray(Image.open(img_path))
    except Exception as e:  # pragma: no cover - just skip the backdrop
        print(f"(could not load {img_path}: {e})")
    H, W = (img.shape[0], img.shape[1]) if img is not None else (240, 320)
    extent = [0, W, H, 0]  # y-down image convention

    ncol = 2 if sim_flow is not None else 1
    fig, axes = plt.subplots(1, ncol, figsize=(6 * ncol, 5), squeeze=False)
    axes = axes[0]

    if img is not None:
        axes[0].imshow(img, extent=extent)
    axes[0].quiver(p[:, 0], p[:, 1], rf[:, 0], rf[:, 1], color="cyan", angles="xy", scale_units="xy", scale=0.3)
    axes[0].set_title(f"REAL image + tracked markers — {title}")

    if sim_flow is not None:
        sf = sim_flow.reshape(-1, 2)
        im = None
        if sim_depth is not None:
            im = axes[1].imshow(sim_depth * 1000.0, extent=extent, cmap="magma", origin="upper")
        axes[1].quiver(p[:, 0], p[:, 1], sf[:, 0], sf[:, 1], color="cyan", angles="xy", scale_units="xy", scale=0.3)
        axes[1].set_title(f"SIM — depth (ContactDepthProbe) + markers ({marker_label})")
        # Reserve an equal-width colorbar slot on BOTH panels (hidden on the real one)
        # so set_aspect("equal") renders both images at the same size.
        if im is not None:
            cax_sim = make_axes_locatable(axes[1]).append_axes("right", size="5%", pad=0.05)
            fig.colorbar(im, cax=cax_sim, label="indentation depth (mm)")
            cax_real = make_axes_locatable(axes[0]).append_axes("right", size="5%", pad=0.05)
            cax_real.axis("off")

    # a "+" marker rotated by the indenter z-rotation; (4, 2, ang) = 4-spoke asterisk at `ang` deg.
    # negate so a +z (pad, y-up) rotation reads correctly on the y-down image axis.
    def rplus(deg):
        return (4, 2, -deg)

    for ax in axes:
        if center_px is not None:
            ax.plot(
                center_px[0],
                center_px[1],
                marker=rplus(center_rot_deg),
                color="lime",
                markersize=14,
                markeredgewidth=2,
                linestyle="None",
            )
        if stop_px is not None and center_px is not None:
            # shear start (+) -> stop (+ rotated by its z-rot) with a connecting arrow
            ax.annotate(
                "",
                xy=(stop_px[0], stop_px[1]),
                xytext=(center_px[0], center_px[1]),
                arrowprops=dict(arrowstyle="->", color="lime", lw=2),
            )
            ax.plot(
                stop_px[0],
                stop_px[1],
                marker=rplus(stop_rot_deg),
                color="lime",
                markersize=14,
                markeredgewidth=2,
                linestyle="None",
            )
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)  # y down to match the image
        ax.set_aspect("equal")
        ax.set_axis_off()  # hide spines, ticks, tick labels, and the px axis label (keep title)
    fig.tight_layout()
    if out:
        fig.savefig(out, dpi=120)
        print(f"saved plot -> {out}")
    else:
        plt.show()


# Params mutable at read-time (no rebuild). lambda_d is baked into the FFT dilation
# kernel at build, so it is NOT here — vary it via separate process invocations.
MUTABLE_PARAMS = ("dilate_scale", "shear_scale", "lambda_s", "normal_exponent")


def set_sensor_scales(sensor, **kw) -> None:
    """Mutate ElastomerTaxel read-time params in place on the shared-metadata tensors."""
    sm = sensor._shared_metadata
    i = sensor._idx
    table = {
        "dilate_scale": sm.dilate_scale,
        "shear_scale": sm.shear_scale,
        "lambda_s": sm.lambda_s,
        "normal_exponent": sm.normal_exponent,
    }
    for k, v in kw.items():
        if k not in table:
            raise ValueError(f"{k} is not mutable after build (mutable: {MUTABLE_PARAMS})")
        table[k][i] = float(v)


# Sweepable axes: name -> (kind, default values). kind drives how a value is applied:
#   "read"   sensor param mutated in place (no rebuild): scales, lambda_s
#   "build"  needs a scene rebuild per value: lambda_d (baked into FFT kernel), ball_radius
#   "motion" passed to run_motion, no rebuild: depth
#   "model"  fed to the FOTS/HydroShear marker model (not the ElastomerTaxel sensor); depth backdrop
#            is geometry-only so it's computed ONCE per geometry and these are swept cheaply on top
# Put a build axis on y (outer loop) so it rebuilds once per value, not every trial.
# dilate_scale is best-fit analytically (see fit_bases), so it's solved per trial, not swept.
# shear_scale CAN be swept (a "read" param): with linear shear the fit re-normalizes it (degenerate),
# but it becomes a real shape axis once shear_scale enters non-linearly (e.g. boundary clamping).
AXIS_REGISTRY = {
    "lambda_s": ("read", [3.0e3, 1.0e4, 3.0e4, 1.0e5, 3.0e5, 1.0e6]),
    "shear_scale": ("read", [0.5, 1.0, 2.0, 4.0, 8.0]),
    # Local (Gaussian) bulge width; the balanced 2-basis fit pins it at ~3e5 (sigma ~1.8mm), so the staged
    # pipeline FIXES it (--lambda-d) and sweeps the mix/thickness axes below instead.
    "lambda_d": ("build", [1.0e5, 2.0e5, 3.0e5, 5.0e5, 1.0e6]),
    "ball_radius": ("build", [2.0, 4.0, 6.0, 9.0, 14.0]),
    "depth": ("motion", [0.05, 0.1, 0.2, 0.4, 0.8]),
    # In-plane dilation mix (baked into the FFT kernel): 1=local Gaussian, 0=incompressible global stretch.
    # Optimum sits at ~0.6-0.7 on the FOTS dilate frame; range brackets it.
    "compressibility": ("build", [0.5, 0.6, 0.65, 0.7, 0.8]),
    # Bonded-layer thickness (mm): the global stretch peaks at r ~ h, so this sets where the far-field hump
    # lands. FOTS fit plateaus over ~12-18; treat as an effective shape scale, not literal gel thickness.
    # Needs compressibility < 1.
    "elastomer_thickness_mm": ("build", [6.0, 9.0, 12.0, 15.0, 18.0]),
    # FOTS Gaussian falloffs (1/px^2) and HydroShear ported-ref falloffs (1/m^2): swept on top of a
    # geometry-only depth backdrop (one Genesis run per geometry), so these trials are cheap.
    "fots_lambda_d": ("model", [0.0008, 0.00125, 0.002, 0.003]),
    "fots_lambda_s": ("model", [0.001, 0.003, 0.01, 0.03]),
    "hydroshear_lambda_d": ("model", [11000.0, 16000.0, 22000.0]),
    "hydroshear_lambda_s": ("model", [7000.0, 10000.0, 13000.0]),
}

# Default axes per motion (1 or 2; override with --sweep-x[/--sweep-y]). Staged calibration:
#   dilate (frame 0, pure dilation): sweep geometry x spread, fit dilate_scale.
#   shear  (frame 1): FIX the dilation (pass --lambda-d/--dilate-scale/--depth/--ball-radius
#           from the dilate result); sweep only lambda_s, fit shear_scale on top.
SWEEP_DEFAULTS = {
    # 2D: x=compressibility (local/global mix), y=elastomer_thickness_mm (far-field hump scale), with
    # lambda_d FIXED via --lambda-d (the balanced 2-basis fit pins the local width at ~3e5; sweeping it
    # here lets the count-weighted objective wash out the local bulge). depth/ball_radius are degenerate
    # with scale (fixed externally).
    "dilate": ["compressibility", "elastomer_thickness_mm"],
    "shear": ["lambda_s"],
}


def run_sweep(
    make_scene,
    motion,
    args,
    real_flow,
    center_xy_mm,
    slide_dir_pad,
    axes,
    fixed_dilate_scale=None,
    marker_model="elastomer",
    pos_px=None,
    grid_center_px=None,
    center_px=None,
    stop_px=None,
):
    """Grid-search 1 or 2 shape axes; the per-trial scale(s) are fit analytically (fit_bases).

    ``axes`` is a list of 1 or 2 ``(name, kind, values)``. For ``marker_model='elastomer'`` each trial
    (re)builds/mutates the ElastomerTaxel sensor (``make_scene``; build axes rebuild, read axes mutate)
    and runs the sim. For ``'fots'``/``'hydroshear'`` the swept ``"model"`` axes feed the published
    marker model; the ContactDepth backdrop is geometry-only so it's computed ONCE per geometry and the
    model trials are cheap. ``fixed_dilate_scale`` locks the dilation (staged shear, elastomer only).
    Returns ``(axes, obj, best)`` where ``best`` carries ``sim_flow``/``depth`` for the grid npz.
    """
    xname, xkind, xvals = axes[0]
    yaxis = axes[1] if len(axes) > 1 else None
    yname, ykind, yvals = yaxis if yaxis else (None, None, [None])
    obj = np.full((len(yvals), len(xvals)), np.nan)
    best, cur, cur_key, cur_depth = None, None, None, None
    print(
        f"\nsweep [{marker_model}/{motion}]: {xname}({xkind})"
        + (f" x {yname}({ykind})" if yaxis else "")
        + f"  ({len(xvals) * len(yvals)} trials)"
        + (" [dilation FIXED]" if fixed_dilate_scale is not None else "")
    )

    def settings(xv, yv):
        build, read, model_p, depth_v, ball_r = {}, {}, {}, args.depth, args.ball_radius
        for nm, kd, v in [(xname, xkind, xv)] + ([(yname, ykind, yv)] if yaxis else []):
            if nm == "depth":
                depth_v = v
            elif nm == "ball_radius":
                ball_r = v
                build[nm] = v
            elif kd == "build":
                build[nm] = v
            elif kd == "model":
                model_p[nm] = v
            else:
                read[nm] = v
        return build, read, model_p, depth_v, ball_r

    def geom_depth(build, depth_v, ball_r):
        # ContactDepth backdrop for FOTS/HydroShear: geometry-only, cached per (build, ball_r, depth).
        nonlocal cur_depth, cur_key
        key = (tuple(sorted(build.items())), ball_r, depth_v)
        if cur_depth is None or key != cur_key:
            scene, ball, sensor, ds, sh, pad = make_scene(build, True, ball_r)
            _, _, cur_depth = eval_marker_bases(
                scene,
                ball,
                sensor,
                pad,
                motion,
                depth_v,
                args.shear,
                center_xy_mm,
                slide_dir_pad,
                ball_r,
                depth_sensor=ds,
                depth_shape=sh,
                control_mode=args.control_mode,
                start_rot_rad=getattr(args, "start_rot_rad", 0.0),
                stop_rot_rad=getattr(args, "stop_rot_rad", 0.0),
            )
            cur_key = key
        return cur_depth

    def evaluate(build, read, model_p, depth_v, ball_r, with_depth):
        nonlocal cur, cur_key
        if marker_model == "elastomer":
            key = (tuple(sorted(build.items())), with_depth)
            if cur is None or key != cur_key:  # rebuild only when build params / depth-flag change
                cur = make_scene(build, with_depth, ball_r)
                cur_key = key
            scene, ball, sensor, ds, sh, pad = cur
            bases, fixed, depth_map = eval_marker_bases(
                scene,
                ball,
                sensor,
                pad,
                motion,
                depth_v,
                args.shear,
                center_xy_mm,
                slide_dir_pad,
                ball_r,
                read_params=read,
                fixed_dilate_scale=fixed_dilate_scale,
                depth_sensor=ds if with_depth else None,
                depth_shape=sh if with_depth else None,
                control_mode=args.control_mode,
                start_rot_rad=getattr(args, "start_rot_rad", 0.0),
                stop_rot_rad=getattr(args, "stop_rot_rad", 0.0),
            )
            return fit_bases(real_flow, bases, fixed), depth_map
        depth_map = geom_depth(build, depth_v, ball_r)  # fots / hydroshear
        if marker_model == "fots":
            slide_px = (stop_px - center_px) if (motion == "shear" and stop_px is not None) else None
            bases = fots_marker_bases(
                pos_px,
                depth_map,
                motion,
                slide_px,
                center_px,
                lamb_d=model_p.get("fots_lambda_d", args.fots_lambda_d),
                lamb_s=model_p.get("fots_lambda_s", args.fots_lambda_s),
            )
        else:
            bases = hydroshear_marker_bases(
                pos_px,
                grid_center_px,
                motion,
                center_xy_mm,
                slide_dir_pad,
                args.shear,
                ball_r,
                depth_v,
                lamb_d=model_p.get("hydroshear_lambda_d", args.hydroshear_lambda_d),
                lamb_s=model_p.get("hydroshear_lambda_s", args.hydroshear_lambda_s),
                mu=args.hydroshear_mu,
            )
        return fit_bases(real_flow, bases), depth_map

    for j, yv in enumerate(yvals):
        for i, xv in enumerate(xvals):
            settings_t = settings(xv, yv)
            m, _ = evaluate(*settings_t, False)
            obj[j, i] = m["scaled_rmse_px"]
            sc = " ".join(f"{k}={v:.4g}" for k, v in m["scales"].items())
            ytag = f"{yname}={yv:<10g} " if yaxis else ""
            print(f"  {xname}={xv:<9g} {ytag}rmse*={m['scaled_rmse_px']:.2f}px rel={m['rel_rmse']:.2f} [{sc}]")
            if best is None or m["scaled_rmse_px"] < best["rmse"]:
                best = {"rmse": m["scaled_rmse_px"], xname: xv, "metrics": m, "settings": settings_t}
                if yaxis:
                    best[yname] = yv

    bx = " ".join(f"{k}={best[k]:g}" for k in ([xname] + ([yname] if yaxis else [])))
    sc_str = ", ".join(
        f"'{k}': {v:.6g}" for k, v in best["metrics"]["scales"].items()
    )  # .6g keeps small scales (incompressible ~5e-4)
    best["bestline"] = (
        f"BEST: {bx}  rmse*={best['rmse']:.2f}px rel={best['metrics']['rel_rmse']:.2f} scales={{{sc_str}}}"
    )
    print(f"\n{best['bestline']}")

    m, depth_map = evaluate(*best["settings"], True)  # recompute best with depth backdrop
    best["metrics"] = m
    best["sim_flow"] = m["pred_flow"]
    best["depth"] = depth_map
    return axes, obj, best


def plot_sweep_heatmap(axes, obj, motion, out):
    """2-D heatmap (2 axes) or line plot (1 axis) of best-fit-scaled RMSE."""
    import matplotlib.pyplot as plt

    xname, _, xvals = axes[0]
    fig, ax = plt.subplots(figsize=(6, 5))
    if len(axes) > 1:
        yname, _, yvals = axes[1]
        im = ax.imshow(obj, origin="lower", aspect="auto", cmap="viridis_r")
        ax.set_xticks(range(len(xvals)), [f"{v:g}" for v in xvals])
        ax.set_yticks(range(len(yvals)), [f"{v:g}" for v in yvals])
        ax.set_xlabel(xname)
        ax.set_ylabel(yname)
        bj, bi = np.unravel_index(np.nanargmin(obj), obj.shape)
        ax.plot(bi, bj, "r*", markersize=18, markeredgecolor="white")
        fig.colorbar(im, ax=ax, label="best-fit-scaled RMSE px (lower=better)")
    else:
        y = obj[0]
        ax.plot(xvals, y, "o-")
        bi = int(np.nanargmin(y))
        ax.plot(xvals[bi], y[bi], "r*", markersize=18)
        ax.set_xscale("log")
        ax.set_xlabel(xname)
        ax.set_ylabel("best-fit-scaled RMSE (px)")
    ax.set_title(f"scale-invariant RMSE — {motion} sweep")
    fig.tight_layout()
    if out:
        fig.savefig(out, dpi=120)
        print(f"saved sweep plot -> {out}")
    else:
        plt.show()


def main() -> None:
    global CTRL_KP, CTRL_KV_RATIO  # set from CLI below; run_motion reads them module-level
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frame", type=int, default=0, help="FOTS csv frame (0=dilate, 1=shear)")
    ap.add_argument("--motion", choices=["dilate", "shear"], default=None, help="override motion (default: from frame)")
    ap.add_argument("--depth", type=float, default=0.5, help="press penetration depth (mm)")
    ap.add_argument("--shear", type=float, default=0.5, help="tangential slide for shear motion (mm)")
    ap.add_argument(
        "--center-px",
        type=float,
        nargs=2,
        default=None,
        metavar=("X", "Y"),
        help="real contact center in image px (overrides the radial-fit estimate)",
    )
    ap.add_argument(
        "--shear-start",
        type=float,
        nargs="+",
        default=None,
        metavar="V",
        help="shear press start: X Y [ZROT_DEG] (px, px, deg). Overrides --center-px for shear. "
        "ZROT_DEG (optional, default 0) is the indenter z-rotation at press (set in radians).",
    )
    ap.add_argument(
        "--shear-stop",
        type=float,
        nargs="+",
        default=None,
        metavar="V",
        help="shear slide end: X Y [ZROT_DEG] (px, px, deg). start->stop sets slide dir + distance; "
        "the z-rotation ramps ZROT_DEG(start)->ZROT_DEG(stop) over the slide (degrees in, radians applied).",
    )
    ap.add_argument(
        "--ball-radius",
        type=float,
        default=BALL_RADIUS_MM,
        help=f"indenter sphere radius in mm (default {BALL_RADIUS_MM}; bigger = flatter/less curved)",
    )
    ap.add_argument(
        "--depth-query",
        choices=["sdf", "raycast"],
        default="sdf",
        help="ContactDepthProbe backend: sdf (fast, primitives) or raycast (meshes, needs probe normal)",
    )
    ap.add_argument(
        "--fots-lambda-d",
        type=float,
        default=FOTS_LAMB[0],
        help=f"FOTS dilation Gaussian falloff (1/px^2; sigma_px=1/sqrt). Default {FOTS_LAMB[0]} (published).",
    )
    ap.add_argument(
        "--fots-lambda-s",
        type=float,
        default=FOTS_LAMB[1],
        help=f"FOTS shear Gaussian falloff (1/px^2). Default {FOTS_LAMB[1]} (published; "
        f"sigma~3.5mm, often too broad -- raise it to localize the shear).",
    )
    ap.add_argument(
        "--hydroshear-lambda-d",
        type=float,
        default=16000.0,
        help="HydroShear (ported ref) dilation Gaussian falloff (1/m^2; over marker spacing). "
        "Default 16000 (mid of the reference's trained 11000-22000 range).",
    )
    ap.add_argument(
        "--hydroshear-lambda-s",
        type=float,
        default=10000.0,
        help="HydroShear (ported ref) shear Gaussian falloff (1/m^2). Default 10000 "
        "(mid of the reference's trained 7000-13000 range).",
    )
    ap.add_argument(
        "--hydroshear-mu",
        type=float,
        default=0.6,
        help="HydroShear (ported ref) Coulomb friction coefficient for the hydrosoft "
        "tangential force (default 0.6; the reference trains over ~0.5-0.75).",
    )
    ap.add_argument(
        "--marker-model",
        choices=["elastomer", "fots", "hydroshear"],
        default="elastomer",
        help="which marker-motion model to compare: elastomer (ours, ElastomerTaxel), "
        "fots (FOTS's published model driven by our ContactDepth), or hydroshear "
        "(ours minus thickness/boundary: ElastomerTaxel compressibility=1).",
    )
    ap.add_argument(
        "--control-mode",
        choices=["position", "teleport"],
        default="position",
        help="position: stiff PD (control_dofs_position) — physical, pins rotation to 0; "
        "teleport: hard set_pos each step (ignores rotation)",
    )
    ap.add_argument(
        "--ctrl-kp",
        type=float,
        default=CTRL_KP,
        help=f"PD position-control stiffness (default {CTRL_KP:g}; raise if penetration < target)",
    )
    ap.add_argument(
        "--ctrl-kv-ratio",
        type=float,
        default=CTRL_KV_RATIO,
        help=f"PD damping as a fraction of kp (default {CTRL_KV_RATIO})",
    )
    ap.add_argument(
        "--smooth-real",
        type=float,
        default=0.5,
        help="Gaussian-smooth the real flow field (marker units) to suppress integer-px "
        "tracking noise; 0 disables. Default 0.5 (denoises without broadening the bulge; "
        "radial fraction 0.81->0.88). Higher over-smooths/broadens.",
    )
    ap.add_argument("--no-sim", action="store_true", help="only load & plot the real data (no Genesis)")
    ap.add_argument(
        "--sweep", action="store_true", help="grid-search two axes to best-fit the real flow (heatmap + best-fit plot)"
    )
    ap.add_argument(
        "--sweep-x",
        choices=list(AXIS_REGISTRY),
        default=None,
        help="sweep x-axis (default per motion). Axes: " + ", ".join(AXIS_REGISTRY),
    )
    ap.add_argument(
        "--sweep-y",
        choices=list(AXIS_REGISTRY),
        default=None,
        help="sweep y-axis (put build axes lambda_d/ball_radius here to rebuild once per value)",
    )
    ap.add_argument(
        "--sweep-x-values",
        default=None,
        metavar="v1,v2,...",
        help="comma-separated values for the x-axis (overrides the AXIS_REGISTRY default range)",
    )
    ap.add_argument(
        "--sweep-y-values",
        default=None,
        metavar="v1,v2,...",
        help="comma-separated values for the y-axis (overrides the AXIS_REGISTRY default range)",
    )
    ap.add_argument(
        "--compressibility",
        type=float,
        default=None,
        help="ElastomerTaxel in-plane dilation mix in [0,1]: 1=local Gaussian, 0=incompressible "
        "(r_hat/r global stretch), in-between blends both. Build-time (baked into the FFT kernel).",
    )
    ap.add_argument(
        "--no-boundary",
        action="store_true",
        help="disable the no-flux elastomer_boundary at the sensor edge (incompressible stretch "
        "then decays free-space instead of piling up at the edges). Boundary needs compressibility<1.",
    )
    ap.add_argument(
        "--elastomer-thickness",
        type=float,
        default=None,
        metavar="MM",
        help="gel layer thickness in mm; >0 uses the incompressible elastic-LAYER kernel (bonded to a "
        "rigid backing) instead of free-space 1/r. Needs compressibility<1; supersedes --no-boundary.",
    )
    ap.add_argument(
        "--dilation-reg",
        type=float,
        default=None,
        metavar="M",
        help="regularization epsilon (meters) for the incompressible 1/r in-plane kernel "
        "(r_hat*r/(r^2+eps^2)); 0 auto-sets it to the probe spacing. Build-time.",
    )
    ap.add_argument(
        "--lambda-d",
        type=float,
        default=None,
        help="override ElastomerTaxel lambda_d (dilate locality, 1/m^2; sigma=1/sqrt). Build-time.",
    )
    ap.add_argument("--lambda-s", type=float, default=None, help="override lambda_s (shear locality, 1/m^2)")
    ap.add_argument("--dilate-scale", type=float, default=None, help="override dilate_scale")
    ap.add_argument("--shear-scale", type=float, default=None, help="override shear_scale")
    ap.add_argument(
        "--no-depth-bg", action="store_true", help="skip the ContactDepthProbe depth-image backdrop in the sim panel"
    )
    ap.add_argument(
        "--save-depth",
        type=str,
        default=None,
        metavar="NPZ",
        help="precompute the ContactDepth map for these indenter params and append it to NPZ "
        "(keyed by motion/center/radius/depth), then exit. For offline RGB-render iteration.",
    )
    ap.add_argument(
        "--save-markers",
        type=str,
        default=None,
        metavar="NPZ",
        help="append this run's marker data (real + best-fit sim flow + depth + center + params) to NPZ, "
        "keyed by motion/marker-model. Read by scripts/fots_plot_grid.py for the 2x4 grid. "
        "If NPZ already holds a matching-parameter result, the run is SKIPPED (cached).",
    )
    ap.add_argument(
        "--force-markers",
        action="store_true",
        help="rerun and overwrite the --save-markers entry even if a matching-parameter result is cached.",
    )
    ap.add_argument(
        "--depth-downsample", type=int, default=1, help="depth grid stride in px (1 = full 320x240; raise to speed up)"
    )
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--vis", action="store_true", help="show Genesis viewer")
    ap.add_argument("--out", type=str, default=None, help="save plot to this path instead of showing")
    args = ap.parse_args()
    CTRL_KP, CTRL_KV_RATIO = args.ctrl_kp, args.ctrl_kv_ratio  # run_motion reads these module-level

    motion = args.motion or FRAME_MOTION.get(args.frame, "dilate")
    pos_px, real_flow = load_real_markers(args.frame, smooth_sigma=args.smooth_real)
    span = real_extent_mm()
    print(f"frame={args.frame} motion={motion}  grid={N_ROWS}x{N_COLS}  span={span[0]:.2f}x{span[1]:.2f} mm")
    print(
        f"real |flow|: mean={np.linalg.norm(real_flow.reshape(-1, 2), axis=1).mean():.2f}px "
        f"max={np.linalg.norm(real_flow.reshape(-1, 2), axis=1).max():.2f}px"
    )

    # Real contact center -> mm offset from the grid center (for placing the sim ball).
    # Image y points down, pad +y up, so the y offset is negated (matches sim_disp_to_px flip).
    # The radial-fit estimate is unreliable on noisy/asymmetric real fields; --center-px overrides it.
    if args.center_px is not None:
        center_px = np.asarray(args.center_px, dtype=float)
        src = "manual"
    else:
        center_px = estimate_contact_center(pos_px, real_flow)
        src = "radial-fit (may be off; pass --center-px)"
    grid_center_px = pos_px.reshape(-1, 2).mean(axis=0)
    off = (center_px - grid_center_px) * PX_TO_MM
    center_xy_mm = (float(off[0]), float(-off[1]))
    print(
        f"contact center [{src}] ~ ({center_px[0]:.0f},{center_px[1]:.0f})px  "
        f"-> sim offset ({center_xy_mm[0]:.2f},{center_xy_mm[1]:.2f})mm  ball_radius={args.ball_radius}mm"
    )

    # Slide direction recovered from the real field (image px) -> pad frame (y flips).
    slide_img = estimate_slide_dir(pos_px, real_flow)
    slide_dir_pad = (float(slide_img[0]), float(-slide_img[1]))
    stop_px = None
    start_rot_deg = stop_rot_deg = 0.0  # indenter z-rotation (deg; 0 = pinned, old behavior)
    if motion == "shear":
        # --shear-start/--shear-stop fully define the shear: press at start, slide to
        # stop. Together they set the press center, slide direction, and distance,
        # replacing --center-px and --shear for this motion. Each takes X Y [ZROT_DEG].
        if args.shear_start is not None and args.shear_stop is not None:
            center_px = np.asarray(args.shear_start[:2], dtype=float)
            stop_px = np.asarray(args.shear_stop[:2], dtype=float)
            start_rot_deg = float(args.shear_start[2]) if len(args.shear_start) > 2 else 0.0
            stop_rot_deg = float(args.shear_stop[2]) if len(args.shear_stop) > 2 else 0.0
            start_mm = px_to_pad_xy(center_px, grid_center_px) * 1000.0
            stop_mm = px_to_pad_xy(stop_px, grid_center_px) * 1000.0
            center_xy_mm = (float(start_mm[0]), float(start_mm[1]))
            vec = stop_mm - start_mm
            args.shear = float(np.linalg.norm(vec))  # effective slide distance (mm)
            slide_dir_pad = (
                (float(vec[0] / args.shear), float(vec[1] / args.shear)) if args.shear > 1e-9 else (1.0, 0.0)
            )
            print(
                f"shear start->stop px ({center_px[0]:.0f},{center_px[1]:.0f})->"
                f"({stop_px[0]:.0f},{stop_px[1]:.0f})  = {args.shear:.2f}mm "
                f"dir pad ({slide_dir_pad[0]:.2f},{slide_dir_pad[1]:.2f})  "
                f"z-rot {start_rot_deg:.1f}->{stop_rot_deg:.1f}deg"
            )
        else:
            print(
                f"real slide dir (image) ~ ({slide_img[0]:.2f},{slide_img[1]:.2f}) "
                f"-> pad ({slide_dir_pad[0]:.2f},{slide_dir_pad[1]:.2f})  (pass --shear-start/--shear-stop to set)"
            )

    # z-rotation: degrees on the CLI, radians applied to the indenter control.
    args.start_rot_deg, args.stop_rot_deg = start_rot_deg, stop_rot_deg
    args.start_rot_rad, args.stop_rot_rad = float(np.radians(start_rot_deg)), float(np.radians(stop_rot_deg))

    out = Path(args.out) if args.out else None

    if args.no_sim:
        plot(
            pos_px,
            real_flow,
            None,
            f"frame {args.frame} ({motion})",
            out,
            args.frame,
            center_px=center_px,
            stop_px=stop_px,
            center_rot_deg=args.start_rot_deg,
            stop_rot_deg=args.stop_rot_deg,
        )
        return

    base_params = {
        k: v
        for k, v in (
            ("lambda_d", args.lambda_d),
            ("lambda_s", args.lambda_s),
            ("dilate_scale", args.dilate_scale),
            ("shear_scale", args.shear_scale),
            ("compressibility", args.compressibility),
            ("elastomer_thickness_mm", args.elastomer_thickness),
        )
        if v is not None
    }
    marker_label = {"elastomer": "ElastomerTaxel (ours)", "fots": "FOTS", "hydroshear": "HydroShear"}[args.marker_model]
    # Depth backdrop on for all outputs. In sweep mode the trials skip the depth read
    # (it's param-independent); it's computed once for the best-fit plot at the end.
    with_depth_default = not args.no_depth_bg

    def make_scene(build_over, with_depth, ball_radius=None):
        # build_over may carry sensor build params (lambda_d) and/or ball_radius; the
        # latter is a morph arg, not a sensor param, so split it out.
        sp = {**base_params}
        br = ball_radius if ball_radius is not None else args.ball_radius
        for k, v in build_over.items():
            if k == "ball_radius":
                br = v
            else:
                sp[k] = v
        return build_scene(
            args.gpu,
            args.vis,
            pos_px,
            sp or None,
            with_depth,
            args.depth_downsample,
            br,
            args.depth_query,
            use_boundary=not args.no_boundary,
        )

    # Staged shear calibration: if dilation params are pinned (--dilate-scale), fix the
    # dilation contribution and fit only shear (see fit_bases / eval_marker_bases).
    fixed_dilate_scale = args.dilate_scale if motion == "shear" else None

    if args.save_depth:
        # Precompute the ContactDepth map for these indenter params -> append to NPZ, then
        # exit. Depth is geometry-only (independent of the swept scales), so this lets the
        # RGB render (scripts/fots_rgb_render.py) iterate offline without rebuilding Genesis.
        scene, ball, sensor, depth_sensor, depth_shape, pad_top_z = make_scene({}, True)
        _, _, depth_map = eval_marker_bases(
            scene,
            ball,
            sensor,
            pad_top_z,
            motion,
            args.depth,
            args.shear,
            center_xy_mm,
            slide_dir_pad,
            args.ball_radius,
            depth_sensor=depth_sensor,
            depth_shape=depth_shape,
            control_mode=args.control_mode,
            start_rot_rad=args.start_rot_rad,
            stop_rot_rad=args.stop_rot_rad,
        )
        key = f"{motion}_cx{center_px[0]:.0f}_cy{center_px[1]:.0f}_r{args.ball_radius:g}_d{args.depth:g}"
        path = Path(args.save_depth)
        store = {k: v for k, v in np.load(path).items()} if path.exists() else {}
        store[key] = depth_map.astype(np.float32)  # (H, W) meters
        store[key + "__meta"] = np.array(
            [center_px[0], center_px[1], args.ball_radius, args.depth, MM_TO_PIXEL], dtype=np.float32
        )
        np.savez(path, **store)
        print(f"saved depth '{key}'  shape={depth_map.shape}  max={depth_map.max() * 1000:.3f}mm  -> {path}")
        return

    if args.sweep:
        names = [args.sweep_x] if args.sweep_x else list(SWEEP_DEFAULTS[motion])
        if args.sweep_y:
            names = ([args.sweep_x] if args.sweep_x else [names[0]]) + [args.sweep_y]
        # values come from --sweep-x/y-values if given (lets run_fots_compare.sh own the ranges),
        # else the AXIS_REGISTRY default range.
        overrides = [args.sweep_x_values, args.sweep_y_values]
        axes = []
        for i, n in enumerate(names):
            ov = overrides[i] if i < len(overrides) else None
            vals = [float(v) for v in ov.split(",")] if ov else AXIS_REGISTRY[n][1]
            axes.append((n, AXIS_REGISTRY[n][0], vals))

        # Cache: skip the whole sweep if --save-markers already holds a result for this exact sweep
        # (same geometry/motion/model + same swept grid). --force-markers reruns.
        sweep_sig = (
            markers_sweep_sig(args, motion, args.marker_model, center_px, stop_px, axes) if args.save_markers else None
        )
        if (
            args.save_markers
            and not args.force_markers
            and markers_cached(args.save_markers, motion, args.marker_model, sweep_sig)
        ):
            bl = markers_bestline(args.save_markers, motion, args.marker_model)
            print(
                f"cached sweep '{motion}/{args.marker_model}' matches -> skipping "
                f"(use --force-markers to rerun) -> {args.save_markers}"
            )
            if bl:
                print(bl)  # re-emit so staged calibration (dilate -> shear) still reads the winners
            return

        axes_out, obj, best = run_sweep(
            make_scene,
            motion,
            args,
            real_flow,
            center_xy_mm,
            slide_dir_pad,
            axes,
            fixed_dilate_scale,
            marker_model=args.marker_model,
            pos_px=pos_px,
            grid_center_px=grid_center_px,
            center_px=center_px,
            stop_px=stop_px,
        )
        if args.save_markers:  # the sweep writes its own best fit -> grid npz (no separate render run)
            append_markers_npz(
                args.save_markers,
                motion,
                args.marker_model,
                pos_px,
                real_flow,
                best["sim_flow"],
                best.get("depth"),
                center_px,
                stop_px,
                args.frame,
                center_rot_deg=args.start_rot_deg,
                stop_rot_deg=args.stop_rot_deg,
                params_sig=sweep_sig,
                bestline=best["bestline"],
            )
            print(f"saved best-fit markers '{motion}/{args.marker_model}' -> {args.save_markers}")
        if out:
            base = out.with_suffix("")
            plot_sweep_heatmap(axes_out, obj, motion, Path(f"{base}_heatmap.png"))
            plot(
                pos_px,
                real_flow,
                best["sim_flow"],
                f"frame {args.frame} ({motion}) BEST-FIT",
                Path(f"{base}_bestfit.png"),
                args.frame,
                best.get("depth"),
                center_px,
                stop_px,
                marker_label=marker_label,
                center_rot_deg=args.start_rot_deg,
                stop_rot_deg=args.stop_rot_deg,
            )
        else:
            plot_sweep_heatmap(axes_out, obj, motion, None)
        return

    # Cache: if --save-markers already holds a matching-parameter result for this (motion, model),
    # skip the (expensive) Genesis run entirely. --force-markers reruns and overwrites.
    params_sig = markers_params_sig(args, motion, args.marker_model, center_px, stop_px) if args.save_markers else None
    if (
        args.save_markers
        and not args.force_markers
        and markers_cached(args.save_markers, motion, args.marker_model, params_sig)
    ):
        print(
            f"cached markers '{motion}/{args.marker_model}' match params -> skipping "
            f"(use --force-markers to rerun) -> {args.save_markers}"
        )
        return

    scene, ball, sensor, depth_sensor, depth_shape, pad_top_z = make_scene({}, with_depth_default)
    if args.marker_model in ("fots", "hydroshear"):
        # FOTS / HydroShear: drive a published marker model on OUR geometry (ElastomerTaxel output
        # unused). Run the sensor once only for the ContactDepth backdrop shared by the plot.
        _, _, depth_map = eval_marker_bases(
            scene,
            ball,
            sensor,
            pad_top_z,
            motion,
            args.depth,
            args.shear,
            center_xy_mm,
            slide_dir_pad,
            args.ball_radius,
            depth_sensor=depth_sensor,
            depth_shape=depth_shape,
            control_mode=args.control_mode,
            start_rot_rad=args.start_rot_rad,
            stop_rot_rad=args.stop_rot_rad,
        )
        if args.marker_model == "fots":
            slide_px = (stop_px - center_px) if (motion == "shear" and stop_px is not None) else None
            model_bases = fots_marker_bases(
                pos_px, depth_map, motion, slide_px, center_px, lamb_d=args.fots_lambda_d, lamb_s=args.fots_lambda_s
            )
        else:  # HydroShear / HydroFOTS reference model, ported (scripts/hydroshear_ref.py)
            model_bases = hydroshear_marker_bases(
                pos_px,
                grid_center_px,
                motion,
                center_xy_mm,
                slide_dir_pad,
                args.shear,
                args.ball_radius,
                args.depth,
                lamb_d=args.hydroshear_lambda_d,
                lamb_s=args.hydroshear_lambda_s,
                mu=args.hydroshear_mu,
            )
        m = fit_bases(real_flow, model_bases)
    else:
        bases, fixed, depth_map = eval_marker_bases(
            scene,
            ball,
            sensor,
            pad_top_z,
            motion,
            args.depth,
            args.shear,
            center_xy_mm,
            slide_dir_pad,
            args.ball_radius,
            fixed_dilate_scale=fixed_dilate_scale,
            depth_sensor=depth_sensor,
            depth_shape=depth_shape,
            control_mode=args.control_mode,
            start_rot_rad=args.start_rot_rad,
            stop_rot_rad=args.stop_rot_rad,
        )
        m = fit_bases(real_flow, bases, fixed)
    _sc = ", ".join(f"'{k}': {v:.6g}" for k, v in m["scales"].items())  # .6g keeps small (incompressible) scales
    print(
        f"[{marker_label}] fit: rmse*={m['scaled_rmse_px']:.3f}px  rel={m['rel_rmse']:.3f}  scales={{{_sc}}}  "
        f"drift=({m['drift_px'][0]:.2f},{m['drift_px'][1]:.2f})px"
    )
    if args.save_markers:
        append_markers_npz(
            args.save_markers,
            motion,
            args.marker_model,
            pos_px,
            real_flow,
            m["pred_flow"],
            depth_map,
            center_px,
            stop_px,
            args.frame,
            center_rot_deg=args.start_rot_deg,
            stop_rot_deg=args.stop_rot_deg,
            params_sig=params_sig,
        )
        print(f"appended markers '{motion}/{args.marker_model}' -> {args.save_markers}")
    if out:
        plot(
            pos_px,
            real_flow,
            m["pred_flow"],
            f"frame {args.frame} ({motion})",
            out,
            args.frame,
            depth_map,
            center_px,
            stop_px,
            marker_label=marker_label,
            center_rot_deg=args.start_rot_deg,
            stop_rot_deg=args.stop_rot_deg,
        )


if __name__ == "__main__":
    main()
