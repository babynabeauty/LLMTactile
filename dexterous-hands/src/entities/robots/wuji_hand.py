from __future__ import annotations

from typing import ClassVar

from eden.options.entities import MetadataOptions, RobotOptions
from eden.options.materials import MaterialLike, RigidMaterialOptions

from registry import ROBOT_REGISTRY
from utils import get_asset_path


@ROBOT_REGISTRY.register(name="wuji")
class WujiHand(RobotOptions):
    # Wuji Hand2 beta2, right. The vendor's MJCF rather than their URDF: URDF carries no
    # actuator description, so Genesis falls back to its own generic PD gains and a task then
    # has to restate them. The MJCF declares a position actuator per joint with a measured
    # `kp`/`kv` pair and a `forcerange` matching the real motor (0.2-2 Nm depending on the
    # joint), plus per-joint armature. Leaving every `default_dofs_*` field empty here is what
    # lets those reach the sim.
    file: str = get_asset_path("hands/wuji/mjcf/right.xml")
    # 0.8 uniformly, and uniform is a modelling compromise: the real hand is gloved and grippier than its palm shell, but Genesis
    # takes friction per geom and the task has no per-link friction support, so palm and fingertips
    # share one value. The number matters because Genesis resolves a contact as
    # `max(hand, object * ratio)` (matching MuJoCo), so the hand sets the floor on contact friction:
    # at the ~1.0 default every object friction draw below 1.0 was inert. 0.8 puts the floor at the
    # bottom of the object range the tasks sample, so the whole range reaches the dynamics.
    material: MaterialLike = RigidMaterialOptions(gravity_compensation=0.0, friction=0.8)

    up: tuple[int, int, int] = (0, 0, 1)
    front: tuple[int, int, int] = (1, 0, 0)

    # Grouped by finger, thumb first, in the order the MJCF declares them.
    dofs_name: list[str] = [
        "r_thumb_cmc_flex",
        "r_thumb_cmc_abd",
        "r_thumb_mcp",
        "r_thumb_ip",
        "r_index_finger_mcp_flex",
        "r_index_finger_mcp_abd",
        "r_index_finger_pip",
        "r_index_finger_dip",
        "r_middle_finger_mcp_flex",
        "r_middle_finger_mcp_abd",
        "r_middle_finger_pip",
        "r_middle_finger_dip",
        "r_ring_finger_mcp_flex",
        "r_ring_finger_mcp_abd",
        "r_ring_finger_pip",
        "r_ring_finger_dip",
        "r_pinky_mcp_flex",
        "r_pinky_mcp_abd",
        "r_pinky_pip",
        "r_pinky_dip",
    ]

    # Enumerate each joint explicitly rather than using a `{"*": 0.0}` glob.
    # Eden's config `_deep_merge` joins dicts, so a glob key would survive
    # alongside per-joint overrides from a sample-grasp YAML and Eden's
    # `resolve_matching_names_values` rejects a joint matching both `*` and
    # its explicit key. Per-joint keys merge cleanly with one match each.
    # A non-empty dict is also required so Eden broadcasts `default_dofs_pos`
    # to `[num_envs, n_dofs]`; an empty dict leaves it `[1, n_dofs]` and the
    # `RandomizeStartupDofsPosBias` event term fails with a shape mismatch.
    default_dofs_pos: dict[str, float] = dict.fromkeys(dofs_name, 0.0)

    metadata: ClassVar[MetadataOptions] = MetadataOptions(
        side="right",
        # The pad, not the bone: each `*_tip_sensor_frame` is a fixed child of its distal link
        # holding the tactile sensor's own mesh, so it is where contact is read.
        fingertip_links=[
            "r_thumb_tip_sensor_frame",
            "r_index_finger_tip_sensor_frame",
            "r_middle_finger_tip_sensor_frame",
            "r_ring_finger_tip_sensor_frame",
            "r_pinky_tip_sensor_frame",
        ],
        finger_links=[
            "r_thumb_proximal",
            "r_thumb_proximal_abd",
            "r_thumb_middle",
            "r_thumb_distal",
            "r_thumb_tip_sensor_frame",
            "r_index_finger_proximal",
            "r_index_finger_proximal_abd",
            "r_index_finger_middle",
            "r_index_finger_distal",
            "r_index_finger_tip_sensor_frame",
            "r_middle_finger_proximal",
            "r_middle_finger_proximal_abd",
            "r_middle_finger_middle",
            "r_middle_finger_distal",
            "r_middle_finger_tip_sensor_frame",
            "r_ring_finger_proximal",
            "r_ring_finger_proximal_abd",
            "r_ring_finger_middle",
            "r_ring_finger_distal",
            "r_ring_finger_tip_sensor_frame",
            "r_pinky_proximal",
            "r_pinky_proximal_abd",
            "r_pinky_middle",
            "r_pinky_distal",
            "r_pinky_tip_sensor_frame",
        ],
        palm_link="r_wrist",
        palm_normal=(0.0, 1.0, 0.0),
        grasp_center=(0.0, 0.02, -0.06),
        # Glove-resolution layout fitted to this robot's meshes: 526 active probes,
        # stored as per-link grids. "actual" refers to taxel counts, not calibrated
        # robot hardware positions. Regenerate with scripts/generate_wuji_tactile.py.
        tactile_probe_cfgs={
            "low": get_asset_path("sensors/wuji/low/probes_hand.json"),
            "high": get_asset_path("sensors/wuji/high/probes_hand.json"),
            "actual": get_asset_path("sensors/wuji/actual/probes_526_hand_wuji.json"),
        },
        # Offsets are the MJCF's own `*_tip` sites, which the vendor places at the centre of
        # each contact pad. A sensor frame shares its distal link's origin (`pos="0 0 0"`,
        # no joint), so the site offset applies unchanged from either link.
        priv_sensor_cfgs={
            "fingertips": {
                "r_thumb_tip_sensor_frame": (0.0, 0.0, -0.02978),
                "r_index_finger_tip_sensor_frame": (0.0, 0.0, -0.02475),
                "r_middle_finger_tip_sensor_frame": (0.0, 0.0, -0.02475),
                "r_ring_finger_tip_sensor_frame": (0.0, 0.0, -0.02475),
                "r_pinky_tip_sensor_frame": (0.0, 0.0, -0.02475),
            }
        },
    )

    def model_post_init(self, context):
        super().model_post_init(context)
        # Deferred so `gs.surfaces.Metal()` is not built at module import,
        # which would violate the Eden-before-genesis import ordering.
        import genesis as gs

        self.surface = gs.surfaces.Metal(
            color=(0.72, 0.74, 0.76, 1.0),
            roughness=0.12,
            metallic=1.0,
        )
