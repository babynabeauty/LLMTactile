"""Evaluate complete deterministic first episodes and write auditable traces.

Example: python scripts/eval_smooth_teacher.py --task=in_fingers_rotate --robot=wuji
 --stage=1 --config=conf/experiments/in_fingers_rotate_wuji_smooth_v1.yaml
 --checkpoint=... --output=logs/eval-smooth
Use --fixed_target to validate cached grasps with the training controller.
"""

import json
import random
import sys
from pathlib import Path
import numpy as np
import torch
import eden as en
import genesis as gs
from eden.envs.wrappers.rsl_rl_env import RslRlVecEnvWrapper

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from registry import get_argparser, get_task_config_from_args, set_eval_num_envs, make_runner
from teacher_evaluation import EpisodeCollector, snapshot_run, preserve_evaluation_rng


def main():
    parser = get_argparser(description=__doc__)
    parser.add_argument("--eval_envs", type=int, default=272)
    parser.add_argument("--eval_seed", type=int, default=42)
    parser.add_argument("--output", required=True)
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--fixed_target", action="store_true")
    parser.add_argument("--pulse_strength", type=float, choices=[0, 0.25, 0.5, 1], default=None)
    parser.add_argument("--probe_wrench_scale", type=float, default=None)
    parser.add_argument("--pulse_profile", choices=["probe", "weight_scaled"], default=None)
    parser.add_argument("--no_traces", action="store_true", help="Write metrics only; skip the 200 MB of per-step traces")
    args = parser.parse_args()
    if not args.fixed_target and not args.checkpoint:
        parser.error("Provide --checkpoint or explicitly request --fixed_target")
    if args.fixed_target and args.checkpoint:
        parser.error("--fixed_target does not evaluate a checkpoint; omit --checkpoint")
    random.seed(args.eval_seed)
    np.random.seed(args.eval_seed)
    torch.manual_seed(args.eval_seed)
    en.init(backend=gs.cpu if args.cpu else gs.gpu, log_root_path=str(Path(args.output) / "runtime"))
    random.seed(args.eval_seed)
    np.random.seed(args.eval_seed)
    torch.manual_seed(args.eval_seed)
    config = get_task_config_from_args(args, run_name="smooth-eval", upload_logs=False)
    set_eval_num_envs(config, args.eval_envs)
    config.runner_options.seed = args.eval_seed
    config.runner_options.device = str(gs.device)
    variants = len(config.scene_options.obj.grouped_entities or [config.scene_options.obj])
    if args.eval_envs % variants:
        raise ValueError(f"Use a multiple of {variants} for balanced object coverage")
    if args.pulse_strength is not None:
        config.event_options.object_pulse.strength = args.pulse_strength
    if args.probe_wrench_scale is not None:
        config.event_options.object_pulse.probe_wrench_scale = args.probe_wrench_scale
    if args.pulse_profile is not None:
        config.event_options.object_pulse.profile = args.pulse_profile
    # Evaluation conditions are separated: every probe episode is eligible;
    # the training mixture remains 25% pulse-free.
    config.event_options.object_pulse.clean_fraction = 0.0
    if args.video:
        if args.eval_envs != variants:
            raise ValueError("Videos require one environment per variant")
        from eden.options import CameraOptions

        config.env_options.env_separate_rigid = True
        config.env_options.env_spacing = (0.0, 0.0)
        config.cameras_options.rec = CameraOptions(
            cam_pos=(0.42, -0.40, 0.70),
            cam_lookat=(0.03, -0.02, 0.44),
            cam_fov=30.0,
            cam_resolution=(640, 480),
            debug=True,
        )
        config.event_options.object_pulse.draw_envs = list(range(variants))
        if hasattr(config.sensors_options, "obj_force"):
            config.sensors_options.obj_force.sensor.draw_debug = False
        if args.task == "grasp_lift":
            config.cameras_options.rec.cam_pos = (0.65, -0.65, 1.05)
            config.cameras_options.rec.cam_lookat = (0.0, 0.0, 0.62)
            config.cameras_options.rec.cam_fov = 40.0
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = snapshot_run(config, output, args.checkpoint or None)
    env = RslRlVecEnvWrapper.from_config(config, show_viewer=False, eval_mode=True)
    with preserve_evaluation_rng():
        runner = (
            None if args.fixed_target else make_runner(env, args.checkpoint, args.actor_checkpoint, args.critic_checkpoint)
        )
    if runner is None:
        policy = lambda obs: torch.zeros(env.num_envs, env.num_actions, device=gs.device)
    else:
        policy = runner.get_inference_policy(device=gs.device)
    obs, *_ = env.reset()
    if runner:
        # PPO's policy is the actor and distillation's is the student; reset the one
        # `policy` above is about to roll out.
        runner.alg.get_policy().reset()
    raw = env.unwrapped
    control = raw.action_manager.get_term("dofs_pos_controller")
    torch.save(
        {
            "observations": {name: value.detach().cpu() for name, value in obs.items()},
            "object_position": raw.entities["obj"].get_pos().clone().cpu(),
            "object_quaternion": raw.entities["obj"].get_quat().clone().cpu(),
            "joint_position": raw.entities["robot"].get_dofs_pos().clone().cpu(),
            "target": control.processed_action.detach().clone().cpu(),
            **{name: getattr(control, "_" + name).clone().cpu()
               for name in ("kp", "kd") if hasattr(control, "_" + name)},
        },
        output / "initial_state.pt",
    )
    collector = EpisodeCollector(env.unwrapped)
    env.unwrapped.physics_state_callback = collector.capture_physics
    video = None
    if args.video:
        from teacher_video import TeacherVideo

        video = TeacherVideo(env.unwrapped, collector, output)
    env.unwrapped.terminal_state_callback = video or collector
    with torch.no_grad():
        for _ in range(int(env.unwrapped.max_episode_length)):
            obs, *_ = env.step(policy(obs))
            if bool(collector.finished.all()) and not video:
                break
    if video:
        video.close()
    report = collector.summary()
    from fresh_campaign import runtime_identity
    report.update(
        runtime=runtime_identity(),
        task_protocol={"in_fingers_rotate":"quaternion_increment_drop_excluded_v1", "grasp_lift":"lowest_point_rise_v1", "in_hand_repose":"live_orientation_goal_v1"}.get(args.task),
        seed=args.eval_seed,
        reset_rng_protocol="runner_rng_isolated_v1",
        fixed_target=args.fixed_target,
        checkpoint_sha256=None if args.fixed_target else manifest["checkpoint"]["sha256"],
        training_env_steps=None if runner is None else getattr(runner, "smooth_training_env_steps", None),
        task=args.task,
        robot=args.robot,
        pulse_strength=config.event_options.object_pulse.strength,
        pulse_free_fraction=config.event_options.object_pulse.clean_fraction,
        evaluation_protocol="separate_clean_probe_16_per_variant_v2",
        pulse_profile=env.unwrapped.object_wrench.state.profile,
        probe_wrench_scale=env.unwrapped.object_wrench.state.probe_wrench_scale,
        goal_curriculum=(
            args.task == "in_hand_repose"
            and config.command_options.goal_rot.name == "reachable_settled_target_rotation_command"
        ),
    )
    (output / "metrics.json").write_text(json.dumps(report, indent=2))
    if not args.no_traces:
        torch.save(collector.trace, output / "trace.pt")
        torch.save(collector.physics_trace, output / "physics_trace.pt")
    print(json.dumps({k: v for k, v in report.items() if k != "per_episode"}, indent=2))
    env.close()


if __name__ == "__main__":
    main()
