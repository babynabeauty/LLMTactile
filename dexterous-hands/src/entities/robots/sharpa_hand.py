from __future__ import annotations

from typing import ClassVar

from eden.options.entities import MetadataOptions, RobotOptions
from eden.options.materials import MaterialLike, RigidMaterialOptions

from registry import ROBOT_REGISTRY
from utils import get_asset_path


@ROBOT_REGISTRY.register(name="sharpa")
class SharpaHand(RobotOptions):
    # The vendor's MJCF, not their URDF: URDF carries no actuator description, so Genesis
    # falls back to its own generic PD gains and a task then has to restate them. The MJCF
    # declares a position actuator per joint (`kp` with `dampratio="0.9"`), per-joint armature,
    # frictionloss and damping through its joint classes, and elastomer pads on the distal
    # links. Leaving every `default_dofs_*` field empty here is what lets those reach the sim.
    file: str = get_asset_path("hands/sharpa/right_sharpa_wave.xml")
    # 0.8 uniformly, and uniform is a modelling compromise: the real hand is metallic with grippier fingertips, but Genesis
    # takes friction per geom and the task has no per-link friction support, so palm and fingertips
    # share one value. The number matters because Genesis resolves a contact as
    # `max(hand, object * ratio)` (matching MuJoCo), so the hand sets the floor on contact friction:
    # at the ~1.0 default every object friction draw below 1.0 was inert. 0.8 puts the floor at the
    # bottom of the object range the tasks sample, so the whole range reaches the dynamics.
    material: MaterialLike = RigidMaterialOptions(gravity_compensation=0.0, friction=0.8)

    up: tuple[int, int, int] = (0, 0, 1)
    front: tuple[int, int, int] = (1, 0, 0)

    dofs_name: list[str] = [
        "right_thumb_CMC_FE",
        "right_thumb_CMC_AA",
        "right_thumb_MCP_FE",
        "right_thumb_MCP_AA",
        "right_thumb_IP",
        "right_index_MCP_FE",
        "right_index_MCP_AA",
        "right_index_PIP",
        "right_index_DIP",
        "right_middle_MCP_FE",
        "right_middle_MCP_AA",
        "right_middle_PIP",
        "right_middle_DIP",
        "right_ring_MCP_FE",
        "right_ring_MCP_AA",
        "right_ring_PIP",
        "right_ring_DIP",
        "right_pinky_CMC",
        "right_pinky_MCP_FE",
        "right_pinky_MCP_AA",
        "right_pinky_PIP",
        "right_pinky_DIP",
    ]

    # Enumerate each joint explicitly rather than using a `{"*": 0.0}` glob.
    # Eden's config `_deep_merge` joins dicts, so a glob key would survive
    # alongside per-joint overrides from a sample-grasp YAML and Eden's
    # `resolve_matching_names_values` rejects a joint matching both `*` and
    # its explicit key. Per-joint keys merge cleanly with one match each.
    # A non-empty dict is also required so Eden broadcasts `default_dofs_pos`
    # to `[num_envs, n_dofs]`; an empty dict leaves it `[1, n_dofs]` and the
    # `RandomizeDefaultDofsPos` event term fails with a shape mismatch.
    default_dofs_pos: dict[str, float] = dict.fromkeys(dofs_name, 0.0)

    metadata: ClassVar[MetadataOptions] = MetadataOptions(
        side="right",
        fingertip_links=[
            "right_thumb_DP",
            "right_index_DP",
            "right_middle_DP",
            "right_ring_DP",
            "right_pinky_DP",
        ],
        finger_links=[
            "right_thumb_MC",
            "right_thumb_PP",
            "right_thumb_DP",
            "right_index_PP",
            "right_index_MP",
            "right_index_DP",
            "right_middle_PP",
            "right_middle_MP",
            "right_middle_DP",
            "right_ring_PP",
            "right_ring_MP",
            "right_ring_DP",
            "right_pinky_MC",
            "right_pinky_PP",
            "right_pinky_MP",
            "right_pinky_DP",
        ],
        palm_link="right_hand_C_MC",
        palm_normal=(1.0, 0.0, 0.0),
        grasp_center=(0.0, 0.015, 0.07),
        # Keyed by resolution. actual covers the fingertip pads at nominal
        # 1 mm spacing; med provides synthetic whole-hand coverage with
        # Wuji-matched regional counts. TactileSensorsMod filters subsets.
        tactile_probe_cfgs={
            "actual": get_asset_path("sensors/sharpa/actual/probes_1206_tips_sharpa.json"),
            "low": get_asset_path("sensors/sharpa/low/probes_hand.json"),
            "med": get_asset_path("sensors/sharpa/med/probes_526_hand_sharpa.json"),
            "high": get_asset_path("sensors/sharpa/high/probes_hand.json"),
        },
        priv_sensor_cfgs={
            "fingertips": {
                "right_index_DP": (0.018, 0.005, 0.0001),
                "right_middle_DP": (0.018, 0.005, 0.0001),
                "right_ring_DP": (0.018, 0.005, 0.0001),
                "right_pinky_DP": (0.018, 0.005, 0.0001),
                "right_thumb_DP": (0.020, 0.005, 0.0005),
            }
        },
    )

    def model_post_init(self, context):
        super().model_post_init(context)
        # Deferred so `gs.surfaces.Metal()` is not built at module import,
        # which would violate the Eden-before-genesis import ordering.
        import genesis as gs

        self.surface = gs.surfaces.Metal(
            color=(0.82, 0.84, 0.86, 1.0),
            roughness=0.12,
            metallic=1.0,
        )
