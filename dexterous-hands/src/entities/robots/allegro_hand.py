from __future__ import annotations

from typing import ClassVar

from eden.options.entities import MetadataOptions, RobotOptions

from registry import ROBOT_REGISTRY
from utils import get_asset_path


@ROBOT_REGISTRY.register(name="allegro")
class AllegroHand(RobotOptions):
    file: str = get_asset_path("hands/allegro/xela_v4/right_hand.xml")
    up: tuple[int, int, int] = (0, 0, 1)
    front: tuple[int, int, int] = (1, 0, 0)

    dofs_name: list[str] = [
        "index_roll",
        "middle_roll",
        "ring_roll",
        "thumb_bend0",
        "index_bend0",
        "middle_bend0",
        "ring_bend0",
        "thumb_roll",
        "index_bend1",
        "middle_bend1",
        "ring_bend1",
        "thumb_bend1",
        "index_bend2",
        "middle_bend2",
        "ring_bend2",
        "thumb_bend2",
    ]

    # A non-empty per-joint dict is required so Eden broadcasts
    # `default_dofs_pos` to `[num_envs, n_dofs]`; an absent/empty dict leaves
    # it `[1, n_dofs]` and the `RandomizeStartupDofsPosBias` event term fails
    # with a shape mismatch.
    default_dofs_pos: dict[str, float] = {
        **dict.fromkeys(dofs_name, 0.0),
        "thumb_bend0": 0.263,
    }

    # MJCF owns actuator gains, damping, armature and effort limits.
    # Keep sensor pads distinct so each retains its own accelerated grid.
    links_to_keep: list[str] = [
        *(f"{finger}_3_tip" for finger in ("index", "middle", "ring", "thumb")),
        *(
            f"{finger}_{part}_4x4_palm_link"
            for finger in ("index", "middle", "ring")
            for part in ("1A", "1B", "2")
        ),
        "thumb_2_4x4_palm_link",
        "thumb_3_4x4_palm_link",
        *(f"ahr_palm_{i}_4x6_palm_link" for i in (1, 2, 3)),
    ]

    metadata: ClassVar[MetadataOptions] = MetadataOptions(
        side="right",
        fingertip_links=[
            "index_3_tip",
            "middle_3_tip",
            "ring_3_tip",
            "thumb_3_tip",
        ],
        finger_links=[
            "index_0",
            "middle_0",
            "ring_0",
            "thumb_0",
            "index_1",
            "middle_1",
            "ring_1",
            "thumb_1",
            "index_2",
            "middle_2",
            "ring_2",
            "thumb_2",
            "index_3",
            "middle_3",
            "ring_3",
            "thumb_3",
            *(name for name in links_to_keep if not name.startswith("ahr_palm_")),
        ],
        palm_link="base_link",
        palm_normal=(1.0, 0.0, 0.0),
        grasp_center=(0.04, 0.0, 0.07),
        palm_links=["base_link", *(f"ahr_palm_{i}_4x6_palm_link" for i in (1, 2, 3))],
        tactile_probe_cfgs={
            "low": get_asset_path("sensors/allegro/low/probes_hand.json"),
            "high": get_asset_path("sensors/allegro/high/probes_hand.json"),
            "actual": get_asset_path(
                "sensors/allegro/actual/probes_368_hand_allegro.json"
            )
        },
        priv_sensor_cfgs={
            "fingertips": {
                "index_3_tip": (3.205e-05, 0.01704531, 0.02846424),
                "middle_3_tip": (3.205e-05, 0.01704531, 0.02846424),
                "ring_3_tip": (3.205e-05, 0.01704531, 0.02846424),
                "thumb_3_tip": (3.205e-05, 0.01704531, 0.02846424),
            }
        },
    )
