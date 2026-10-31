"""Configuration for the grasp-and-lift task.

Reach for an object resting on a table, close on it, lift it clear, and carry it to a sampled
goal position. Modelled on https://github.com/ruoyiqiao/mjlab_hand.

The reward is staged rather than sparse: an approach term pays per-fingertip progress toward the
object, a lift term pays height above the resting pose plus a one-shot bonus for clearing
threshold, and a goal term pays progress toward the carry target once the object is up. Each is
best-so-far, so a partial improvement is banked and cannot be farmed by oscillating. That staging
is what lets the hand start open: unlike ``in_fingers_rotate``, there is a gradient from the first
step, so no precomputed grasp cache is needed to bootstrap.

mjlab_hand pairs these rewards with ContactExplorer, a contact-novelty bonus over per-object point
clouds. This task uses RND instead (``--rnd=100``), which the runner mod already supports.
"""

import eden as en
import genesis as gs
from eden.managers.modifiers.observations import GaussianNoise
from eden.options import (
    ActionManagerOptions,
    CommandManagerOptions,
    EnvOptions,
    EventManagerOptions,
    MetricManagerOptions,
    ObservationGroupOptions,
    ObservationManagerOptions,
    RewardManagerOptions,
    SceneOptions,
    SensorOptions,
    SensorsOptions,
    TerminationManagerOptions,
)
from eden.options.entities import BoxOptions
from eden.options.learning.rsl_rl import RslRlBaseRunnerOptions, RslRlPpoAlgorithmOptions
from eden.tasks import TASK_REGISTRY
from eden.terms import DoneTerm, MetricTerm, ObsTerm, RewardTerm
from eden.utils.configs import EdenRLConfig
from genesis.utils.geom import euler_to_quat

import lift_teacher  # noqa: F401 — register versioned teacher controller and observations
from entities.objects import OBJECTS_4CM_16
from models import RslRlDistillationAlgorithmOptions
from optimization import DEFAULT_HYPERPARAM_RANGES
from registry import HYPERPARAMS_REGISTRY
from shared_terms import (
    CONTROLLER_RANDOMIZATIONS,
    DECIMATION,
    HAND_CONTROLLER,
    PI,
    SIM_DT,
    CachedObs,
    ObjectSizeObs,
    PalmApproachReward,
    RandomizeFrictionRatioWithObs,
    RandomizeMassScaleWithObs,
    RootPoseController,
    SetSampledBottomAlignedPos,
    TactileSensorRead,
    base_rot6d,
    dofs_pos_limits_penalty,
    episode_reward_metric,
    joint_vel_penalty,
    obj_below_height,
    termination_penalty,
    torque_penalty,
    work_penalty,
)
from task_mods import (
    DexHandRslRlRunnerMod,
    ManipulationObjectMod,
    NoTactileDebugDrawMod,
    RobotHandWithPrivSensorsMod,
    TactileSensorsMod,
    set_base_condition,
)
from utils import get_asset_path

from .custom_terms import ApproachProgressReward, GoalProgressReward, HandCentricSurfaceVectors, LiftProgressReward

TABLE_Z = 0.5
TABLE_SIZE = (0.6, 0.6, 0.02)
# Object rests on the table surface; the hand starts above it, open.
OBJECT_POS = (0.0, 0.0, TABLE_Z)
LIFT_THRESHOLD = 0.05

RUNNER_CFG = {
    "num_steps_per_env": 24,
    "max_iterations": 6000,
    "save_interval": 500,
    "clip_actions": 10.0,
    "obs_groups": {
        # actor is rewritten per stage by DexHandRslRlRunnerMod
        "critic": ["goal", "priv_proprio", "priv_obj_state", "priv_obj_props"],
        "student": ["proprio", "goal", "tactile_sensors"],
        "teacher": ["goal", "priv_proprio", "priv_obj_state", "priv_obj_props"],
        "rnd_state": ["goal", "priv_proprio", "priv_obj_state", "priv_obj_props"],
    },
}
PPO_OPTIONS = RslRlPpoAlgorithmOptions(
    value_loss_coef=1.0,
    use_clipped_value_loss=True,
    clip_param=0.2,
    entropy_coef=0.002,
    num_learning_epochs=5,
    num_mini_batches=4,
    learning_rate=0.001,
    schedule="adaptive",
    gamma=0.998,
    lam=0.95,
    desired_kl=0.01,
    max_grad_norm=1.0,
)
DISTILL_OPTIONS = RslRlDistillationAlgorithmOptions(
    num_learning_epochs=5,
    normalize_action_targets=True,
    learning_rate=1.0e-4,
    gradient_length=2,
    max_grad_norm=None,
    optimizer="adam",
    loss_type="mse",
)
# Sharpa and Wuji ship welded to the world -- Genesis takes a link's fixed/free state from the
# model file and ignores `is_fixed_base` for MJCF -- so a floating hand needs its own copy of the
# vendor model with a `<freejoint/>` on the root body. The originals stay untouched for every task
# that pins the wrist.
FLOATING_MODELS = {
    "allegro": get_asset_path("hands/allegro/xela_v4/right_hand_floating.xml"),
    "sharpa": get_asset_path("hands/sharpa/right_sharpa_wave_floating.xml"),
    "wuji": get_asset_path("hands/wuji/mjcf/right_floating.xml"),
}
# Only a hover to start from: the policy flies the hand, so this is not the hand-tuned constant it
# has to be in the fixed-base tasks. Measured fixed-base placement was a dead end -- across eight
# orientations at the best height, wuji got one fingertip within 9 mm of the object and sharpa
# never came within 95 mm, because a pinned wrist cannot descend onto a table.
# The wrist starts this far above the tallest object of the set (mjlab_hand starts 2 cm above an
# 8 cm object); the runtime preflight checks that no fingertip touches at reset.
HAND_START_CLEARANCE = 0.08
HAND_EULERS = {
    # Palm normals in the vendor root frame: Wuji +Y, Sharpa/Allegro +X.
    # Map these to world -Z so fingers close toward the tabletop object.
    "wuji": (-90.0, 0.0, 0.0),
    "default": (0.0, 90.0, 0.0),
}


def object_height(options) -> float:
    """Vertical extent of a primitive variant at its default orientation."""
    if hasattr(options, "size"):
        return float(options.size[2])
    if hasattr(options, "height"):
        return float(options.height)
    if hasattr(options, "radius"):
        return 2.0 * float(options.radius)
    return 0.10


def hand_start_height(config: EdenRLConfig) -> float:
    obj = getattr(config.scene_options, "obj", None)
    variants = getattr(obj, "grouped_entities", None) or [obj]
    return max(object_height(v) for v in variants) + HAND_START_CLEARANCE


class GraspLiftRobotMod(RobotHandWithPrivSensorsMod):
    """Hover the hand over the table on a free-floating wrist the policy drives."""

    def _configure_robot_and_actions(self, config: EdenRLConfig, robot_instance):
        floating = FLOATING_MODELS.get(self.robot)
        return robot_instance if floating is None else robot_instance.model_copy(update={"file": floating})

    def apply(self, config: EdenRLConfig) -> EdenRLConfig:
        config = super().apply(config)
        robot = config.scene_options.robot
        metadata = robot.metadata
        # The standard private probes cover the fingertip pad centres. Add one
        # object-only surface probe at the actual palm grasp centre so the
        # proximity reward measures the contact geometry requested by the task,
        # rather than palm-link-to-object-centre distance.
        config.sensors_options.priv_surface_distance_palm_center = SensorOptions(
            sensor=gs.sensors.SurfaceDistanceProbe(
                probe_local_pos=(metadata.grasp_center,),
                probe_radius=0.5,
                draw_debug=False,
                track_link_idx=(0,),
            ),
            attach_entity_name="robot",
            attach_link_name=metadata.palm_link,
            track_link_names=["obj"],
        )
        euler = HAND_EULERS.get(self.robot, HAND_EULERS["default"])
        robot.default_root_pos = (OBJECT_POS[0], OBJECT_POS[1], OBJECT_POS[2] + hand_start_height(config))
        robot.default_root_quat = euler_to_quat(euler)
        set_base_condition(robot, fixed=False)
        return config


HYPERPARAMS_REGISTRY.register(
    DEFAULT_HYPERPARAM_RANGES
    | {
        "rew_approach": (0.001, 0.5),
        "rew_lift": (0.1, 5.0),
        "rew_goal_progress": (0.001, 0.5),
        "rew_held": (0.5, 10.0),
        "rew_dropped": (-50.0, -2.0),
        "rew_action_rate_penalty": (-0.1, -0.001),
        "rew_work_penalty": (-10.0, -0.1),
    },
    name="grasp_lift",
)


@TASK_REGISTRY.register(
    name="grasp_lift",
    modifiers=[
        ManipulationObjectMod(obj="primitives", entity_name="obj", add_vis_entity=True, primitives=OBJECTS_4CM_16),
        GraspLiftRobotMod(
            robot="sharpa",
            priv_sensor_cfg_name="fingertips",
            add_shape_sensors=True,
            track_link_idx="obj",
        ),
        TactileSensorsMod(sensors="none", track_link_idx="obj"),
        NoTactileDebugDrawMod(),
        DexHandRslRlRunnerMod(runner_cfg=RUNNER_CFG, ppo_options=PPO_OPTIONS, distill_options=DISTILL_OPTIONS),
    ],
)
class GraspLiftConfig(EdenRLConfig):
    """Reach for an object on a table, grasp it, lift it clear, and carry it to a goal."""

    env_options: EnvOptions = EnvOptions(
        # Genesis keeps link mass, COM and inertia once for the whole batch unless this is set,
        # and rejects the per-env writes the inertial randomization below performs.
        batch_links_info=True,
        num_envs=4096,
        env_spacing=(0.40, 0.40),
        episode_length_s=10.0,
        sim_dt=SIM_DT,
        sim_substeps=1,
        decimation=DECIMATION,
        use_gjk_collision=True,
        max_collision_pairs=48,
        multiplier_collision_broad_phase=16,
        enable_multi_contact=True,
    )

    scene_options: SceneOptions = SceneOptions(
        # obj / vis_obj are set by ManipulationObjectMod, robot by GraspLiftRobotMod.
        table=BoxOptions(
            size=TABLE_SIZE,
            default_root_pos=(0.0, 0.0, TABLE_Z - TABLE_SIZE[2] / 2.0),
            fixed=True,
        ),
    )

    sensors_options: SensorsOptions = SensorsOptions(
        # tactile_sensors set by TactileSensorsMod; priv_* by GraspLiftRobotMod.
    )

    observation_options: ObservationManagerOptions = ObservationManagerOptions(
        tactile_sensors=ObservationGroupOptions(
            tactile_sensors=TactileSensorRead.configure(sensor_names=["tactile_*"])
        ),
        proprio=ObservationGroupOptions(
            dofs_pos=ObsTerm.configure(
                func=en.observations.dofs_pos,
                params={"entity_name": "robot", "offset_from_default": False},
                noise=GaussianNoise.configure(std=0.005),
            ),
            dofs_vel=ObsTerm.configure(
                func=en.observations.dofs_vel,
                params={"entity_name": "robot"},
                scale=0.2,
                noise=GaussianNoise.configure(std=0.01),
            ),
            last_action=ObsTerm.configure(func=en.observations.last_action),
        ),
        goal=ObservationGroupOptions(
            goal_pos=ObsTerm.configure(func=en.observations.generated_commands, params={"command_name": "goal_pos"}),
        ),
        priv_proprio=ObservationGroupOptions(
            dofs_pos=ObsTerm.configure(
                func=en.observations.dofs_pos, params={"entity_name": "robot", "offset_from_default": False}
            ),
            dofs_vel=ObsTerm.configure(func=en.observations.dofs_vel, params={"entity_name": "robot"}, scale=0.2),
            last_action=ObsTerm.configure(func=en.observations.last_action),
            contact=en.observations.SensorRead.configure(sensor_names=["priv_contact_*"], scale=0.2),
            surface_distance=en.observations.SensorRead.configure(sensor_names=["priv_surface_distance_*"]),
        ),
        priv_obj_state=ObservationGroupOptions(
            obj_pos=ObsTerm.configure(func=en.observations.base_pos, params={"entity_name": "obj"}),
            obj_rot=ObsTerm.configure(func=base_rot6d, params={"entity_name": "obj"}),
        ),
        priv_obj_props=ObservationGroupOptions(
            obj_size=ObjectSizeObs.configure(entity_name="obj", scale=20.0),
            obj_friction=CachedObs.configure(width=1),
            obj_mass=CachedObs.configure(width=1, scale=10.0),
        ),
        priv_shape=ObservationGroupOptions(
            surface_vectors=HandCentricSurfaceVectors.configure(),
        ),
    )

    action_options: ActionManagerOptions = ActionManagerOptions(
        dofs_pos_controller=HAND_CONTROLLER,
        # The wrist is the reach: the fingers alone cannot get to an object on a table.
        ee_controller=RootPoseController.configure(
            entity_name="robot",
            pos_x_range=(-0.15, 0.15),
            pos_y_range=(-0.15, 0.15),
            pos_z_range=(TABLE_Z - 0.02, TABLE_Z + 0.50),
            euler_x_range=(-PI / 4, PI / 4),
            euler_y_range=(-PI / 4, PI / 4),
            euler_z_range=(-PI / 4, PI / 4),
        ),
    )

    reward_options: RewardManagerOptions = RewardManagerOptions(
        # Staged: approach pays until the object is up, lift pays the climb and the one-shot
        # bonus, goal_progress takes over for the carry. Scales are mjlab_hand's, which keeps
        # the three stages roughly comparable in magnitude despite the different units.
        approach=ApproachProgressReward.configure(
            weight=0.01,
            object_name="obj",
            robot_name="robot",
            lift_threshold=LIFT_THRESHOLD,
        ),
        lift=LiftProgressReward.configure(
            weight=1.0,
            object_name="obj",
            lift_threshold=LIFT_THRESHOLD,
            lift_bonus=10.0,
        ),
        goal_progress=GoalProgressReward.configure(
            weight=0.01,
            object_name="obj",
            command_name="goal_pos",
            success_tolerance=0.03,
            success_bonus=10.0,
        ),
        held=PalmApproachReward.configure(
            weight=1.0,
            object_name="obj",
            sigma=0.15,
        ),
        dropped=RewardTerm.configure(
            func=termination_penalty,
            weight=-30.0,
            params={"termination_names": ("obj_fallen",)},
        ),
        # Effort is charged per step while `dropped` fires once, so it has to stay near a tenth
        # of the positive reward or the policy learns to shed the object and stop paying. Same
        # reasoning as docs/grasp_catch.md.
        action_rate_penalty=RewardTerm.configure(
            func=en.rewards.action_rate_l2,
            weight=-1.0,
        ),
        joint_vel_penalty=RewardTerm.configure(
            func=joint_vel_penalty,
            weight=-1e-3,
            params={"entity_name": "robot"},
        ),
        joint_limits_penalty=RewardTerm.configure(
            func=dofs_pos_limits_penalty,
            weight=-1.0,
            params={"entity_name": "robot"},
        ),
        torque_penalty=RewardTerm.configure(
            func=torque_penalty,
            weight=-0.01,
            params={"entity_name": "robot"},
        ),
        work_penalty=RewardTerm.configure(
            func=work_penalty,
            weight=-0.05,
            params={"entity_name": "robot"},
        ),
    )

    metric_options: MetricManagerOptions = MetricManagerOptions(
        objective=MetricTerm.configure(
            func=episode_reward_metric,
            params={"reward_names": ["lift", "dropped"], "weights": [1.0, -10.0]},
        ),
    )

    termination_options: TerminationManagerOptions = TerminationManagerOptions(
        time_out=DoneTerm.configure(func=en.terminations.time_out, time_out=True),
        obj_fallen=DoneTerm.configure(
            func=obj_below_height,
            params={"entity_name": "obj", "threshold": TABLE_Z - 0.1},
        ),
    )

    command_options: CommandManagerOptions = CommandManagerOptions(
        goal_pos=en.commands.UniformT3Command.configure(
            x_range=(-0.05, 0.05),
            y_range=(-0.05, 0.05),
            # High enough that a 10 cm object has to leave the table by 10-25 cm.
            z_range=(TABLE_Z + 0.20, TABLE_Z + 0.35),
            resampling_time_range=(1e9, 1e9),
            marker_entity_name="vis_obj",
        ),
    )

    event_options: EventManagerOptions = EventManagerOptions(
        place_obj=SetSampledBottomAlignedPos.configure(
            mode=en.EventMode.RESET,
            entity_name="obj",
            pos_z=TABLE_Z,
            range_x=(-0.03, 0.03),
            range_y=(-0.03, 0.03),
            range_yaw=(-PI, PI),
        ),
        obj_friction=RandomizeFrictionRatioWithObs.configure(
            mode=en.EventMode.RESET,
            obs_name="obj_friction",
            entity_name="obj",
            links_name="*",
            # Starts at the hands' own 0.8: Genesis resolves a contact as `max(hand, object)`,
            # so draws below the hand's friction are inert.
            friction_range=(0.8, 2.0),
        ),
        obj_mass=RandomizeMassScaleWithObs.configure(
            mode=en.EventMode.RESET,
            obs_name="obj_mass",
            entity_name="obj",
            links_name="*",
            mass_scale_range=(0.5, 1.5),
        ),
        **CONTROLLER_RANDOMIZATIONS,
    )

    runner_options: RslRlBaseRunnerOptions = RslRlBaseRunnerOptions()
