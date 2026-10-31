"""Configuration for in-fingers rotation task.

HORA-style continuous in-hand rotation around a fixed target axis (world -Z).
The reward is the angular velocity component along the target axis, clipped so it
saturates: past ``angvel_clip_max`` a faster spin earns nothing more, and the
penalties below make it cost more. The policy therefore settles into a slow steady
gait rather than flicking the object.

Reference: HORA (In-Hand Object Rotation via Rapid Motor Adaptation).
https://arxiv.org/abs/2210.04887

The Allegro setup is a direct port of HORA's ``AllegroHandHora``: palm up with the
fingers horizontal, 20 Hz control integrating 1/24 rad joint increments, compliant
PD gains with the torque clipped at the real motor's 0.5 Nm, and HORA's five reward
terms. See ``docs/in_fingers_rotate_hora.md`` for the numbers and where each came from.
"""

import object_wrench  # noqa: F401 — register opt-in smooth teacher terms
import smooth_control  # noqa: F401

import eden as en
from eden.constants import ReferenceSource
from eden.managers.modifiers.observations import GaussianNoise
from eden.options import (
    ActionManagerOptions,
    CameraOptions,
    CamerasOptions,
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
from eden.options.learning.rsl_rl import RslRlBaseRunnerOptions, RslRlPpoAlgorithmOptions
from eden.tasks import TASK_REGISTRY
from eden.terms import DoneTerm, MetricTerm, ObsTerm, RewardTerm
from eden.utils.configs import EdenRLConfig
from genesis import gs
from genesis.utils.geom import euler_to_quat

from entities.objects import OBJECTS_4CM_16, OBJECTS_HORA_MIXED_11
from models import RslRlDistillationAlgorithmOptions
from shared_terms import (
    CONTROLLER_RANDOMIZATIONS,
    HAND_CONTROLLER,
    SIM_DT,
    CachedObs,
    LoadGraspPose,
    ObjectSizeObs,
    RandomizeFrictionRatioWithObs,
    RandomizeMassShiftWithObs,
    RotationAxisCommand,
    TactileSensorRead,
    base_rot6d,
    dofs_pos_limits_penalty,
    episode_reward_metric,
    object_linvel_penalty,
    rotation_reward,
    termination_penalty,
    torque_penalty,
    work_penalty,
)
from task_mods import (
    DexHandRslRlRunnerMod,
    ManipulationObjectMod,
    RobotHandWithPrivSensorsMod,
    TactileSensorsMod,
    set_base_condition,
)

from .custom_terms import AxisTurnsMetric, DroppedBelowGraspStart, GraspPoseDiffPenalty

OBJECT_POS = (0.0, 0.0, 0.5)

# HORA runs 20 Hz control and integrates a 1/24 rad joint increment per step, so a joint
# travels at most 0.83 rad/s. That rate limit, not a reward term, is what keeps the gait
# smooth, and the increment only means the same thing at a matching control rate.
DECIMATION = 10
ACTION_INCREMENT_RAD = 1.0 / 24.0

# `RandomizeStartupDofsPosBias` biases the controller by writing `_offset`, and rejects an
# accumulating controller at build time. HORA puts its joint noise on the observation instead
# (`jointNoiseScale: 0.02`), which the `proprio` group's GaussianNoise already covers.
CONTROLLER_RANDOMIZATIONS_NO_BIAS = {
    name: term for name, term in CONTROLLER_RANDOMIZATIONS.items() if name != "dofs_randomize_bias"
}

RUNNER_CFG = {
    "num_steps_per_env": 24,
    # HORA's `max_agent_steps: 1500000000` is about 7600 iterations at 8192 envs x 24 steps.
    # The teacher clears three turns an episode inside 400.
    "max_iterations": 5000,
    "save_interval": 500,
    "obs_groups": {
        # actor will be set to student/teacher based on stage
        # The teacher is privileged but sensorless, as in HORA: no tactile group, so its only
        # proprioception is joint position/velocity and its own last action. Fingertip contact
        # force is a tactile signal in all but name, so it stays on the critic.
        "critic": ["goal", "priv_proprio", "priv_contact", "priv_obj_state", "priv_obj_props"],
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
    num_learning_epochs=1,
    normalize_action_targets=True,
    learning_rate=1.0e-4,
    gradient_length=2,
    max_grad_norm=None,
    optimizer="adam",
    loss_type="mse",
)
RND_CFG = {
    "learning_rate": 1e-3,
    "num_outputs": 8,
    "state_normalization": True,
    "reward_normalization": True,
}
HAND_OFFSETS = {
    # HORA puts the Allegro base 0.15 m directly below the object: palm up, fingers
    # horizontal, object cradled on the fingers just above the palm.
    "allegro": (0.0, 0.0, -0.15),
    # Set by eye in the viewer, together with the 30-degree tilt in HAND_EULERS that brings
    # the thumb round to oppose the fingers.


    "sharpa": (0.07, -0.015, -0.02),
    # Measured against the object rather than set by eye: at the -0.02 this entry shared with the
    # other hands, all five wuji fingertips sit ~3 cm above a 4 cm object and no sampled grasp ever
    # gets one within 2 cm. Matches conf/sample_grasps/in_fingers_rotate_wuji.yaml.
    "wuji": (0.0, -0.07, -0.08),
    "default": (0.06, 0.0, -0.02),
}
# Extrinsic XYZ euler, degrees. The Allegro entry reproduces HORA's
# ``Quat.from_axis_angle(y, -pi/2) * Quat.from_axis_angle(x, pi/2)``: base-link +z
# (the finger direction) maps to world -y and +x (the palm normal) to world +z.
HAND_EULERS = {
    "allegro": (90.0, -90.0, 0.0),
    # XHand1's thumb swings in the plane a 30-degree tilt off the generic -90 opens up. At
    # -90 it cannot reach round to meet the fingertips at all.

    "wuji": (90.0, 0.0, 0.0),
    "default": (0.0, -90.0, 0.0),
}
# Object size is a property of the hand: HORA's canonical grasp closes the Allegro fingertips
# around a 6.4 cm block, and on the 4 cm set the tips finish 2-3 cm clear of every face so no
# sampled grasp passes. The smaller hands keep the 4 cm set their grasp files were built
# against. See "Object size" in docs/in_fingers_rotate_hora.md.
OBJECT_SETS = {"allegro": OBJECTS_HORA_MIXED_11,  "default": OBJECTS_4CM_16}
# All active hands use MJCF actuator defaults except the explicit Wuji override below.
# Wuji is the exception that needs an entry despite being MJCF. Its vendor gains (kp 0.18-0.69)
# are soft enough that Genesis's implicitly-solved `control_dofs_pos`, which the grasp sampler
# validates with, and the `ExplicitPDController` that trains here, which integrates torque
# itself, stop agreeing: of the grasps the sampler accepted, only 8% survived a 400-step
# zero-action episode under the training controller, against 70% for sharpa, and the teacher
# dropped the object within 7 of 400 steps on every episode. See "Wuji" in
# docs/in_fingers_rotate_hora.md.
#
# Allegro XELA V4 uses the MJCF controller defaults.
#
HAND_GAINS = {
    # Gains only. Force limits and armature stay the vendor's: wuji's force range is per-joint
    # (0.2 Nm on the abduction joints, 2.0 on MCP flexion), and flattening it to a single value
    # cuts the main gripping torque -- measured, a uniform 1.0 Nm limit at these gains drops
    # zero-action grasp survival from 75% to 32%. kp 5.0 restores survival to 75% from the
    # vendor's 8%; kd is critical damping for it at the vendor armature, `2*sqrt(kp*I)` = 0.1
    # at I = 5e-4.
    "wuji": {
        "default_dofs_kp": {"*": 5.0},
        "default_dofs_kd": {"*": 0.10},
    },
}


class InFingersRotateRobotMod(RobotHandWithPrivSensorsMod):
    """Task-specific robot placement, fixed base, and fingertip proximity sensors."""

    def _configure_robot_and_actions(self, config: EdenRLConfig, robot_instance):
        gains = HAND_GAINS.get(self.robot)
        if gains is not None:
            robot_instance = robot_instance.model_copy(update=gains)
        return robot_instance

    def apply(self, config: EdenRLConfig) -> EdenRLConfig:
        config = super().apply(config)
        primitives = OBJECT_SETS.get(self.robot)
        if primitives is not None:
            config.scene_options.obj = config.scene_options.obj.model_copy(update={"grouped_entities": primitives})
        offset = HAND_OFFSETS.get(self.robot, HAND_OFFSETS["default"])
        euler = HAND_EULERS.get(self.robot, HAND_EULERS["default"])
        config.scene_options.robot.default_root_pos = tuple(OBJECT_POS[i] + offset[i] for i in range(3))
        config.scene_options.robot.default_root_quat = euler_to_quat(euler)
        set_base_condition(config.scene_options.robot, fixed=True)
        config.scene_options.robot.batch_fixed_verts = True
        return config


@TASK_REGISTRY.register(
    name="in_fingers_rotate",
    modifiers=[
        # Default set; InFingersRotateRobotMod swaps in a per-hand size (see OBJECT_SETS).
        ManipulationObjectMod(obj="primitives", entity_name="obj", primitives=OBJECTS_4CM_16),
        InFingersRotateRobotMod(robot="sharpa", priv_sensor_cfg_name="fingertips", track_link_idx="obj"),
        TactileSensorsMod(sensors="none", track_link_idx="obj"),
        DexHandRslRlRunnerMod(
            runner_cfg=RUNNER_CFG,
            ppo_options=PPO_OPTIONS,
            distill_options=DISTILL_OPTIONS,
            rnd_cfg=RND_CFG,
        ),
    ],
)
class InFingersRotateConfig(EdenRLConfig):
    """
    Continuously rotate an object in the fingertips around a target axis (default world Z).
    """

    # Framed on env 0's hand. The default recording camera in `position_camera_config`
    # aims at z = 0.1, which suits a table-top task; here the palm is at 0.35 and the
    # object at 0.5, so that view cuts off everything worth watching.
    cameras_options: CamerasOptions = CamerasOptions(
        rec=CameraOptions(cam_pos=(0.42, -0.40, 0.70), cam_lookat=(0.03, -0.02, 0.44), cam_fov=30.0)
    )

    env_options: EnvOptions = EnvOptions(
        # Genesis keeps link mass, COM and inertia once for the whole batch unless this is
        # set, and rejects the per-env writes the inertial randomization below performs.
        batch_links_info=True,
        num_envs=8192,
        env_spacing=(0.30, 0.25),
        episode_length_s=20.0,
        sim_dt=SIM_DT,
        sim_substeps=1,
        decimation=DECIMATION,
        use_gjk_collision=True,
        max_collision_pairs=30,
        multiplier_collision_broad_phase=12,
        enable_multi_contact=True,
    )

    scene_options: SceneOptions = SceneOptions(
        # obj is set by ManipulationObjectMod
        # robot is set by InFingersRotateRobotMod
    )
    sensors_options: SensorsOptions = SensorsOptions(
        # tactile_sensors is set by TactileSensorsMod
        # priv_surface_distance_* is set by InFingersRotateRobotMod
        # priv_contact_* is set by InFingersRotateRobotMod
        obj_force=SensorOptions(
            sensor=gs.sensors.ContactForce(
                draw_debug=True,
            ),
            attach_entity_name="obj",
        ),
    )

    action_options: ActionManagerOptions = ActionManagerOptions(
        # HORA integrates the action onto a persistent target rather than commanding an
        # absolute pose: target_t = clamp(target_{t-1} + a * 1/24, joint limits), restarting
        # from the sampled grasp each episode. The per-step increment caps joint speed at
        # 0.83 rad/s, so a gait has to be built out of small strokes however large the action.
        dofs_pos_controller=HAND_CONTROLLER.model_copy(
            update={
                "reference_source": ReferenceSource.ACCUMULATE,
                "scale_ratio": None,
                "scale": ACTION_INCREMENT_RAD,
            }
        ),
    )

    observation_options: ObservationManagerOptions = ObservationManagerOptions(
        tactile_sensors=ObservationGroupOptions(
            tactile_sensors=TactileSensorRead.configure(
                sensor_names=["tactile_*"],
            ),
        ),
        goal=ObservationGroupOptions(
            rotation_axis=ObsTerm.configure(
                func=en.observations.generated_commands,
                params={"command_name": "rotation_axis"},
            ),
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
            last_action=ObsTerm.configure(
                func=en.observations.last_action,
            ),
        ),
        priv_proprio=ObservationGroupOptions(
            dofs_pos=ObsTerm.configure(
                func=en.observations.dofs_pos,
                params={"entity_name": "robot", "offset_from_default": False},
            ),
            dofs_vel=ObsTerm.configure(
                func=en.observations.dofs_vel,
                params={"entity_name": "robot"},
                scale=0.2,
            ),
            last_action=ObsTerm.configure(
                func=en.observations.last_action,
            ),
        ),
        # Critic only. Fingertip contact force is the signal the student is meant to learn
        # to do without, so the teacher never sees it.
        priv_contact=ObservationGroupOptions(
            contact=en.observations.SensorRead.configure(
                sensor_names=["priv_contact_*"],
            ),
        ),
        priv_obj_state=ObservationGroupOptions(
            obj_pos=ObsTerm.configure(
                func=en.observations.base_pos,
                params={"entity_name": "obj"},
            ),
            obj_rot6d=ObsTerm.configure(
                func=base_rot6d,
                params={"entity_name": "obj"},
            ),
            obj_lin_vel=ObsTerm.configure(
                func=en.observations.base_lin_vel,
                params={"entity_name": "obj"},
            ),
            obj_ang_vel=ObsTerm.configure(
                func=en.observations.base_ang_vel,
                params={"entity_name": "obj"},
                scale=0.2,
            ),
        ),
        # Everything the object varies by that the policy cannot see directly. The
        # teacher reads this group; the student has to recover its effects from how the
        # object responds. Genesis does not support per-env COM writes on primitive
        # morphs, so this keeps the supported HORA properties: scale, mass and friction.
        priv_obj_props=ObservationGroupOptions(
            obj_friction=CachedObs.configure(
                default_value=1.0,
            ),
            obj_mass=CachedObs.configure(
                default_value=0.1,
            ),
            obj_size=ObjectSizeObs.configure(
                entity_name="obj",
            ),
            surface_distance=en.observations.SensorRead.configure(
                sensor_names=["priv_surface_distance_*"],
            ),
        ),
    )

    command_options: CommandManagerOptions = CommandManagerOptions(
        rotation_axis=RotationAxisCommand.configure(
            resampling_time_range=(999.0, 999.0),  # Fixed per episode
            axis_mode="z_only",
        ),
    )

    # HORA's five terms at their published weights. The rotation reward is the only
    # positive one and it saturates at 0.5 rad/s, so once the object is turning at
    # roughly a twelfth of a revolution per second the policy's only remaining
    # gradient is to spend less effort and hold the object more steadily.
    reward_options: RewardManagerOptions = RewardManagerOptions(
        rotation_reward=RewardTerm.configure(
            func=rotation_reward,
            weight=1.0,
            params={
                "entity_name": "obj",
                "command_name": "rotation_axis",
                "angvel_clip_min": -0.5,
                "angvel_clip_max": 0.5,
            },
        ),
        # Rotating in place is the task; translating the object is how a policy flicks or
        # throws it, so the L1 speed is charged directly.
        obj_linvel_penalty=RewardTerm.configure(
            func=object_linvel_penalty,
            weight=-0.3,
            params={"entity_name": "obj"},
        ),
        # Anchors the fingers on the grasp they started from, making the motion a repeating
        # gait rather than a one-way scramble to the joint limits.
        pose_diff_penalty=GraspPoseDiffPenalty.configure(
            weight=-0.3,
            entity_name="robot",
        ),
        torque_penalty=RewardTerm.configure(
            func=torque_penalty,
            weight=-0.1,
            params={"entity_name": "robot"},
        ),
        work_penalty=RewardTerm.configure(
            func=work_penalty,
            weight=-2.0,
            params={"entity_name": "robot"},
        ),
        # Not in HORA, which relies on the drop termination alone. Small, because the joint
        # limits here are the URDF's rather than HORA's tightened ones, so the integrating
        # target can park a joint against a limit and sit there.
        joint_limits_penalty=RewardTerm.configure(
            func=dofs_pos_limits_penalty,
            weight=-1.0,
            params={"entity_name": "robot"},
        ),
        # HORA has no drop penalty: its 5 mm termination fires before a falling object can
        # earn anything. At the 2 cm threshold used here a spin-and-drop banks a few steps
        # of rotation reward first. Deliberately small, since a large drop penalty buys the
        # opposite failure, a policy that holds the object still because moving it risks it.
        is_dropped=RewardTerm.configure(
            func=termination_penalty,
            weight=-10.0,
            params={
                "termination_names": ("obj_dropped",),
            },
        ),
    )

    metric_options: MetricManagerOptions = MetricManagerOptions(
        # Net turns about the target axis while the object is still in the fingers. Reward
        # can be collected by a half-turn wobble repeated forever; turns cannot.
        turns=AxisTurnsMetric.configure(
            entity_name="obj",
            command_name="rotation_axis",
            hold_radius=0.06,
        ),
        objective=MetricTerm.configure(
            func=episode_reward_metric,
            params={
                "reward_names": ["rotation_reward"],
                "weights": [1.0],
            },
        ),
    )

    termination_options: TerminationManagerOptions = TerminationManagerOptions(
        time_out=DoneTerm.configure(func=en.terminations.time_out, time_out=True),
        # HORA terminates 5 mm below the start height, so an object that has left the
        # fingers cannot keep earning rotation reward on the way down. 2 cm keeps that
        # property with room for Genesis's contact resolution.
        obj_dropped=DroppedBelowGraspStart.configure(
            entity_name="obj",
            margin=0.02,
            fallback_height=OBJECT_POS[2],
        ),
    )

    event_options: EventManagerOptions = EventManagerOptions(
        load_grasp=LoadGraspPose.configure(
            mode=en.EventMode.RESET,
            entity_name="robot",
            obj_name="obj",
            grasps_paths={
                "allegro": "src/assets/grasps/in_fingers_rotate_allegro_mixed_grasps_128.pt",

                "sharpa": "src/assets/grasps/in_fingers_rotate_sharpa_grasps_64.pt",
                "wuji": "src/assets/grasps/in_fingers_rotate_wuji_grasps_64.pt",
            },
        ),
        # Bounded below by the hand's own friction, because Genesis resolves a contact as
        # `max(hand, object * ratio)` (matching MuJoCo) rather than combining the pair. The hands
        # sit at 0.8, so a range starting there is entirely live: the old (0.15, 1.2) spent about
        # 81% of its draws below the then ~1.0 hand floor, producing identical physics while
        # `obs_name` faithfully reported the sampled value to the critic. Slip is where tactile
        # sensing has its clearest advantage, since shear at the contact patch leads gross motion
        # and proprioception cannot register a slip until the object has moved -- which is exactly
        # the regime the inert lower range was failing to produce.
        obj_friction=RandomizeFrictionRatioWithObs.configure(
            mode=en.EventMode.RESET,
            obs_name="obj_friction",
            entity_name="obj",
            links_name="*",
            friction_range=(0.8, 2.0),
        ),
        # HORA's full shift. The primitives declare no material, so they build at Eden's 200 kg/m^3
        # default: measured, `OBJECTS_4CM_16` comes out at 8.6-18.2 g. A 0-150 g shift on top of
        # that is up to twelve times the object's own weight, and it dominates the base mass
        # rather than perturbing it -- the sampled object masses span 19-110 g over eight envs.
        obj_mass=RandomizeMassShiftWithObs.configure(
            mode=en.EventMode.RESET,
            obs_name="obj_mass",
            entity_name="obj",
            links_name="*",
            mass_shift_range=(0.0, 0.15),
        ),
        # Off, as in HORA (`forceScale: 0.0`), which adds external pushes for deployment
        # robustness only once the gait is stable.
        ext_force=en.events.ApplyExternalForce.configure(
            mode=en.EventMode.INTERVAL,
            interval_range_s=(2.0, 4.0),
            entity_name="obj",
            links_name="*",
            force_x_range=(0.0, 0.0),
            force_y_range=(0.0, 0.0),
            force_z_range=(0.0, 0.0),
        ),
        **CONTROLLER_RANDOMIZATIONS_NO_BIAS,
    )

    runner_options: RslRlBaseRunnerOptions = RslRlBaseRunnerOptions(
        # runner is set by DexHandRslRlRunnerMod
    )
