"""
Export a policy rollout as a standalone, interactive Viser replay.

The output is a single self-contained `.html` file: open it directly in a browser, or
embed it with `<iframe src="rollout.html">`. It carries the scene geometry and the
per-step link poses, so the viewer can orbit the camera and scrub the trajectory with
no server and no simulator. Pass a `.viser` output instead for the compact binary form.

Usage:
    python scripts/record_viser_rollout.py --task in_fingers_rotate --robot wuji \
        --checkpoint logs/<run>/train/model_1500.pt -o rollouts/wuji_teacher.html
"""

import argparse
import copy
import math
from pathlib import Path

import eden as en
import genesis as gs
import numpy as np
import torch
from eden.envs.wrappers.rsl_rl_env import RslRlVecEnvWrapper
from eden.extensions.visualization.viser import ViserViewer

from checkpoint_recording import frame_env_grid
from registry import get_argparser, get_task_config_from_args, make_runner, set_eval_num_envs


def resolve_output(output: str | None, checkpoint: str | None) -> Path:
    if output:
        return Path(output)
    if checkpoint:
        return Path(checkpoint).with_suffix(".html")
    return Path("rollout.html")


def select_variant(config, index: int) -> None:
    """Replace the object set with one of its variants, so a replay shows a single object.

    A one-element group is not the same thing: Genesis stops treating the entity as
    heterogeneous while the options stay grouped, and Eden's viewer then asserts. Promoting
    the variant to the scene's object keeps it an ordinary entity. ``vis_obj`` holds a ghost
    copy of the same set and moves in step, or the goal marker stops matching the object.
    """
    for name in ("obj", "vis_obj"):
        entity = getattr(config.scene_options, name, None)
        grouped = getattr(entity, "grouped_entities", None)
        if not grouped:
            continue
        variant = copy.deepcopy(grouped[index])
        # The group carries the placement the task gave it -- ``vis_obj`` in particular is
        # lifted above the hand -- while a variant only describes its own geometry. Keep the
        # group's placement and appearance, or the goal marker drops to the floor and the
        # camera, which frames on this field, follows it down.
        for field in ("default_root_pos", "default_root_quat", "surface"):
            value = getattr(entity, field, None)
            if value is not None:
                setattr(variant, field, value)
        config.scene_options._locked = False
        setattr(config.scene_options, name, variant)


def place_camera(viewer: ViserViewer, config, cam_pos, cam_lookat) -> None:
    """Point the replay's initial camera at the env grid, or at an explicit pose.

    Note that an exported replay currently ignores this: the pose is serialized into the
    recording (verified on viser 1.0.30 and 1.1.0, as both ``initial=True`` hints and
    direct sets) but the standalone player does not apply it, so a replay opens on viser's
    own default view. It still works for a live viewer.
    """
    auto_pos, auto_lookat, fov_degrees = frame_env_grid(config)
    viewer.server.initial_camera.position = tuple(cam_pos) if cam_pos else auto_pos
    viewer.server.initial_camera.look_at = tuple(cam_lookat) if cam_lookat else auto_lookat
    viewer.server.initial_camera.fov = math.radians(fov_degrees)


def main():
    parser = get_argparser(description="Record a policy rollout to an interactive Viser HTML replay.")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Replay path ending in .html or .viser (default: <checkpoint>.html).")
    parser.add_argument("-t", "--seconds", type=float, default=None,
                        help="Seconds of policy to record (default: one full episode).")
    parser.add_argument("--every", type=int, default=1,
                        help="Record one frame every N control steps. Playback stays real-time; "
                             "raising it shrinks the file.")
    parser.add_argument("--stochastic", action="store_true", help="Sample from the policy distribution.")
    parser.add_argument("--cam_pos", type=float, nargs=3, default=None, help="Override the initial camera position.")
    parser.add_argument("--cam_lookat", type=float, nargs=3, default=None, help="Override the initial camera target.")
    parser.add_argument("--port", type=int, default=8080, help="Port the recording Viser server binds to.")
    parser.add_argument("--variant", type=int, default=None,
                        help="Record a single object variant by index into the task's grouped entities.")
    args = parser.parse_args()

    if args.every < 1:
        parser.error("--every must be at least 1.")
    output_path = resolve_output(args.output, args.checkpoint)
    if output_path.suffix.lower() not in {".html", ".viser"}:
        parser.error("--output must end in .html or .viser.")

    en.init(backend=gs.gpu if not args.cpu else gs.cpu, log_root_path="logs/temp")

    config = get_task_config_from_args(args, upload_logs=False)
    if args.variant is not None:
        grouped = getattr(getattr(config.scene_options, "obj", None), "grouped_entities", None)
        if not grouped:
            parser.error("--variant needs a task whose object is a set of grouped entities.")
        if not 0 <= args.variant < len(grouped):
            parser.error(f"--variant must be in [0, {len(grouped)}), got {args.variant}.")
        select_variant(config, args.variant)
    config = set_eval_num_envs(config, args.num_eval_envs)

    env = RslRlVecEnvWrapper.from_config(config, show_viewer=False, eval_mode=True)

    policy = None
    if args.checkpoint or args.actor_checkpoint:
        runner = make_runner(
            env,
            checkpoint=args.checkpoint,
            actor_checkpoint=args.actor_checkpoint,
            critic_checkpoint=args.critic_checkpoint,
        )
        policy = runner.get_inference_policy(device=gs.device)
    else:
        print("No checkpoint given; recording the zero-action rollout.")

    # `update()` advances playback by one frame duration, so a decimated recording has to
    # declare the wall-clock time those frames actually span or the replay runs fast.
    frame_dt = env.dt * args.every
    steps = int(args.seconds / env.dt) if args.seconds else int(env.unwrapped.max_episode_length)

    viewer = ViserViewer(env.unwrapped, port=args.port, enable_gui=False)
    viewer.build()
    place_camera(viewer, config, args.cam_pos, args.cam_lookat)

    obs, *_ = env.reset()
    if policy is not None and hasattr(policy, "reset"):
        policy.reset()

    zero_actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)
    print(f"Recording {steps} steps ({steps * env.dt:.1f}s) of {args.num_eval_envs} env(s) to {output_path} ...")
    try:
        viewer.start_recording(output_path, fps=1.0 / frame_dt)
        with torch.no_grad():
            for step in range(steps):
                if policy is not None:
                    actions = policy(obs, stochastic_output=True) if args.stochastic else policy(obs)
                else:
                    actions = zero_actions
                obs, _, dones, *_ = env.step(actions)
                if policy is not None and hasattr(policy, "reset"):
                    policy.reset(dones)
                if (step + 1) % args.every == 0:
                    viewer.update()
        saved = viewer.stop_recording()
        print(f"Saved interactive replay to {saved} ({saved.stat().st_size / 1e6:.1f} MB)")
    finally:
        viewer.close()
        if hasattr(env, "close"):
            env.close()


if __name__ == "__main__":
    main()
