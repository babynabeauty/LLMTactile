"""Main training script for dexterous manipulation tasks using Eden + RSL-RL."""

import argparse
import os

import eden as en
import genesis as gs
import torch
from eden.envs.wrappers.rsl_rl_env import RslRlVecEnvWrapper

from distill_manifest import now_iso, setup_wandb_resume, write_run_status
from optimization import apply_params_to_config, run_evaluation, sample_hyperparams
from registry import (
    build_run_name,
    find_latest_checkpoint,
    get_argparser,
    get_hyperparam_ranges,
    get_task_config_from_args,
    make_runner,
    patch_on_policy_runner_to_save_video,
    save_video_from_checkpoint,
    save_video_from_runner,
)


def train(config, args):
    """Train the policy."""
    is_distill = "distill" in str(config.runner_options.class_name).lower()
    if is_distill and not args.checkpoint:
        raise ValueError("Teacher checkpoint is required for distillation.")
    from experiment_runs import validate_run_contract

    from training_preferences import validate_training_objects

    validate_training_objects(config)
    validate_run_contract(config, args, en.log_dir)

    controller = getattr(config.action_options, "dofs_pos_controller", None)
    smooth_teacher = controller is not None and controller.name == "smooth_pd_controller"
    if not args.no_checkpoint_video and not smooth_teacher:
        patch_on_policy_runner_to_save_video()

    # Resume support: with --resume, load the latest checkpoint already present
    # in the (fixed) log dir instead of the teacher. A distillation model_*.pt
    # embeds the teacher and restores the optimizer + iteration counter, so
    # `make_runner` picks up exactly where the preempted job left off.
    checkpoint_arg = args.checkpoint
    resume_checkpoint = None
    if getattr(args, "resume", False):
        resume_checkpoint = find_latest_checkpoint(en.log_dir)
        if resume_checkpoint:
            checkpoint_arg = resume_checkpoint
            print(f"Resuming from checkpoint: {resume_checkpoint}")
        else:
            print("--resume set but no prior checkpoint found in log dir; starting fresh.")

    # Pin the W&B run id so a resumed job continues the same W&B run.
    wandb_run_id = None
    if getattr(config.runner_options, "logger", "") == "wandb":
        wandb_run_id = setup_wandb_resume(en.log_dir, resuming=resume_checkpoint is not None)

    write_run_status(
        en.log_dir,
        status="running",
        task=getattr(args, "task", ""),
        robot=getattr(args, "robot", ""),
        sensors=getattr(args, "sensors", "none"),
        tactile_imperfection=getattr(args, "tactile_imperfection", None)
        or ("strong" if getattr(args, "sensors", "").endswith("/noisy") else "clean"),
        seed=config.runner_options.seed,
        model=getattr(args, "model", None),
        tactile_encoder=getattr(args, "tactile_encoder", None),
        teacher_checkpoint=args.checkpoint if is_distill else None,
        max_iterations=config.runner_options.max_iterations,
        wandb_run_id=wandb_run_id,
        wandb_project=getattr(config.runner_options, "wandb_project", None),
        started=now_iso(),
    )

    controller = getattr(config.action_options, "dofs_pos_controller", None)
    if controller is not None and controller.name == "smooth_pd_controller":
        from eden.utils.random import set_random_seed

        set_random_seed(config.runner_options.seed)
    config.env_options.background_color = (1.0, 1.0, 1.0)
    env = RslRlVecEnvWrapper.from_config(config, show_viewer=args.show_viewer)
    # Eden replay config (``--full_config``); keep separate from W&B ``config.yaml`` in the log dir.
    config.save_as_yaml(path=f"{en.log_dir}/eden_config.yaml", lock_config=False)
    try:
        runner = make_runner(
            env,
            checkpoint_arg,
            args.actor_checkpoint,
            args.critic_checkpoint,
            load_actor_distribution=getattr(args, "stage", None) != 3,
        )
    except (RuntimeError, ValueError) as exc:
        if resume_checkpoint is not None:
            raise RuntimeError(
                f"Cannot resume {resume_checkpoint}; checkpoints and logs have been preserved. "
                "Use the original configuration or choose a new run name."
            ) from exc
        raise
    if (checkpoint_arg or args.actor_checkpoint or args.critic_checkpoint) and not args.no_loaded_video:
        save_video_from_runner(
            config, runner, f"{en.log_dir}/loaded_policy.mp4", seconds=args.record_seconds,
            num_envs=len(config.scene_options.obj.grouped_entities or [config.scene_options.obj]),
        )
    if getattr(runner, "smooth_teacher_contract", None) is not None:
        from teacher_evaluation import attach_training_evaluation

        attach_training_evaluation(
            runner, config, en.log_dir, args.task, args.robot, record_video=not args.no_checkpoint_video
        )
    runner.add_git_repo_to_log(__file__)

    # `learn` runs `current_learning_iteration + num_learning_iterations` total,
    # so on resume we ask only for the remaining iterations.
    max_iterations = config.runner_options.max_iterations
    remaining = max(0, max_iterations - runner.current_learning_iteration)
    if remaining == 0:
        print(f"\nAlready at {runner.current_learning_iteration}/{max_iterations} iterations; skipping training.")
    else:
        print(
            f"\nStarting training: {remaining} iterations ({runner.current_learning_iteration} -> {max_iterations})..."
        )
        runner.learn(num_learning_iterations=remaining, init_at_random_ep_len=not args.sequential_episodes)

    print("\nTraining complete!")
    print(f"Logs saved to: {en.log_dir}")

    write_run_status(
        en.log_dir,
        status="completed",
        last_iteration=runner.current_learning_iteration,
        finished=now_iso(),
    )

    if args.no_checkpoint_video or smooth_teacher:
        return
    args.checkpoint = f"{en.log_dir}/model_{max_iterations - 1}.pt"
    play_config = get_task_config_from_args(args, upload_logs=False)
    play(play_config, args)


def play(config, args):
    """Run inference with trained policy."""
    tactile_npz_path = None
    if getattr(args, "save_tactile", False):
        tactile_npz_path = args.tactile_output or os.path.join(en.log_dir, "tactile_readings.npz")
    save_video_from_checkpoint(
        config,
        args.checkpoint,
        seconds=args.record_seconds,
        video_path=args.video_output,
        tactile_npz_path=tactile_npz_path,
        num_envs=args.num_eval_envs,
    )


def rollout_benchmark(config, args):
    """Run one episode of the policy at several num_envs values and report per-step timing."""
    import copy
    import time

    import numpy as np

    num_envs_list = args.benchmark_num_envs
    report_fastest_k = 100

    print(f"\nRollout benchmark: num_envs sweep = {num_envs_list}")
    print(f"Final stats computed on the fastest {report_fastest_k} steps of each rollout.")
    results: list[dict] = []
    all_times_path = os.path.join(en.log_dir, "rollout_benchmark_times.npz")
    raw_times: dict[str, np.ndarray] = {}

    for n_envs in num_envs_list:
        print(f"\n=== num_envs = {n_envs} ===")
        cfg = copy.deepcopy(config)
        cfg.env_options._locked = False
        cfg.env_options.num_envs = n_envs

        env = RslRlVecEnvWrapper.from_config(cfg, show_viewer=False, eval_mode=True)
        runner = make_runner(env, args.checkpoint, args.actor_checkpoint, args.critic_checkpoint)
        policy = runner.get_inference_policy(device=gs.device)

        obs, *_ = env.reset()
        steps = max(1, int(env.unwrapped.max_episode_length))
        print(f"Running {steps} steps with {n_envs} envs...")

        step_times_ms: list[float] = []
        with torch.no_grad():
            wall_t0 = time.perf_counter()
            for _ in range(steps):
                t0 = time.perf_counter()
                actions = policy(obs)
                obs, *_ = env.step(actions)
                step_times_ms.append((time.perf_counter() - t0) * 1000.0)
            wall_ms = (time.perf_counter() - wall_t0) * 1000.0

        arr = np.asarray(step_times_ms, dtype=np.float64)
        raw_times[f"n{n_envs}"] = arr
        k = min(report_fastest_k, arr.size)
        fastest = np.sort(arr)[:k]
        mean_ms = float(fastest.mean())
        std_ms = float(fastest.std())
        median_ms = float(np.median(fastest))
        p95_ms = float(np.percentile(fastest, 95))
        min_ms = float(fastest.min())
        max_ms = float(fastest.max())
        full_max_ms = float(arr.max())
        full_argmax_step = int(arr.argmax())
        wall_per_step_ms = wall_ms / steps
        print(
            f"  fastest-{k}: mean = {mean_ms:.3f}  std = {std_ms:.3f}  median = {median_ms:.3f}  "
            f"p95 = {p95_ms:.3f}  min = {min_ms:.3f}  max = {max_ms:.3f}"
        )
        print(f"  (full {steps} steps: max = {full_max_ms:.3f} at step {full_argmax_step}, wall {wall_ms:.1f} ms)")
        results.append(
            dict(
                n_envs=n_envs,
                mean=mean_ms,
                std=std_ms,
                median=median_ms,
                p95=p95_ms,
                min=min_ms,
                max=max_ms,
                full_max=full_max_ms,
                wall_per_step=wall_per_step_ms,
                steps=steps,
                k=k,
            )
        )

        if hasattr(env, "close"):
            env.close()

    np.savez(all_times_path, **raw_times)
    print(f"\nRaw per-step times saved to {all_times_path}")

    print(f"\nBenchmark summary (stats over fastest {report_fastest_k} steps per rollout):")
    header = f"{'num_envs':>10}  {'mean':>10}  {'std':>10}  {'median':>10}  {'p95':>10}  {'max':>10}  {'full_max':>10}"
    print(header)
    for r in results:
        print(
            f"{r['n_envs']:>10}  {r['mean']:>10.3f}  {r['std']:>10.3f}  {r['median']:>10.3f}  "
            f"{r['p95']:>10.3f}  {r['max']:>10.3f}  {r['full_max']:>10.3f}"
        )


def optimize(config, args):
    """Run Optuna hyperparameter optimization over reward weights and PPO policy/algorithm."""
    import optuna

    from optimization import resolve_storage

    hyperparam_ranges = get_hyperparam_ranges(config.runner_options.experiment_name)
    storage = resolve_storage(args.opt_storage)

    # Workers sharing a study must not share a sampler seed: TPE's startup phase draws from the
    # sampler's own RNG, so identically seeded workers propose the same configs. A single local
    # run keeps the config seed and stays reproducible.
    sampler_seed = config.runner_options.seed
    if args.opt_storage:
        slurm_id = os.environ.get("SLURM_JOB_ID")
        sampler_seed = int(slurm_id) % (2**31) if slurm_id else None

    MIN_STARTUP_TRIALS = 10
    n_trials = args.opt_trials
    n_iterations = args.opt_iterations
    n_eval_episodes = args.opt_eval_episodes

    study = optuna.create_study(
        study_name=args.study_name,
        sampler=optuna.samplers.TPESampler(n_startup_trials=MIN_STARTUP_TRIALS, seed=sampler_seed),
        pruner=optuna.pruners.NopPruner(),
        storage=storage,
        load_if_exists=bool(storage),
        direction="maximize",
    )

    config.runner_options.max_iterations = n_iterations

    def objective(trial: optuna.Trial) -> float:
        params = sample_hyperparams(trial, hyperparam_ranges=hyperparam_ranges)
        trial_config = apply_params_to_config(params, config)

        # Train
        env = RslRlVecEnvWrapper.from_config(trial_config, show_viewer=False)

        trial_log_dir = f"{en.log_dir}/trial_{trial.number}"
        runner = make_runner(
            env,
            args.checkpoint,
            args.actor_checkpoint,
            args.critic_checkpoint,
            log_dir=trial_log_dir,
            load_actor_distribution=getattr(args, "stage", None) != 3,
        )
        runner.learn(num_learning_iterations=n_iterations, init_at_random_ep_len=True)

        # Evaluate
        trial_config.env_options._locked = False
        trial_config.env_options.num_envs = n_eval_episodes
        return run_evaluation(trial_config, runner, n_eval_episodes, gs.device, args.opt_save_video)

    study.optimize(objective, n_trials=n_trials)

    print("Number of finished trials:", len(study.trials))
    if study.best_trial is None:
        print("No completed trials.")
        return None

    print("Best trial:")
    print("  Value (mean return):", study.best_trial.value)
    print("  Params:")
    for k, v in study.best_trial.params.items():
        print(f"    {k}: {v}")
    return study


def main():
    parser = get_argparser(description="Train or evaluate RL agents for dexterous manipulation tasks.")
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        help="Mode to run: train, play/inference, optimize, or rollout-benchmark",
    )
    parser.add_argument("--show_viewer", action="store_true", help="Show the viewer during run")
    # Training (mode=train)
    parser.add_argument("--load_best_hyperparams", action="store_true", help="Load best hyperparameters from study")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the latest checkpoint in the (fixed) log dir, if one exists.",
    )
    parser.add_argument(
        "--sequential_episodes",
        action="store_true",
        help="Start every environment at episode step 0 instead of rsl_rl's random episode phases.",
    )
    parser.add_argument(
        "--no_loaded_video",
        action="store_true",
        help="Skip rendering loaded_policy.mp4 of the loaded checkpoint at the start of training. "
        "Useful for distillation sweeps where the teacher checkpoint is always loaded and the "
        "render adds latency to every job.",
    )
    parser.add_argument(
        "--no_checkpoint_video",
        action="store_true",
        help="Skip per-checkpoint async video saves (model_*.mp4 / model_*.video.log) and the "
        "final play() video at the end of training. Used by distillation sweeps to avoid the "
        "genesis probe-debug IndexError that fires when tactile sensors have history_length > 0.",
    )
    # Inference (mode=play)
    parser.add_argument("--record_seconds", "-t", type=float, default=10.0, help="Seconds to run inference for")
    parser.add_argument(
        "--save_tactile",
        action="store_true",
        help="Save per-step tactile sensor readings of the play episode to an .npz "
        "(animate with scripts/animate_tactile.py).",
    )
    parser.add_argument("--tactile_output", type=str, default=None, help=argparse.SUPPRESS)
    # Hyperparameter optimization (mode=optimize)
    parser.add_argument("--study_name", type=str, default="", help="The study name")
    parser.add_argument("--opt_save_video", action="store_true", help="Save videos of evaluations")
    parser.add_argument("--opt_trials", type=int, default=100, help="Number of trials for hyperparameter optimization")
    parser.add_argument("--opt_iterations", type=int, default=1000, help="Training iterations per trial")
    parser.add_argument("--opt_eval_episodes", type=int, default=10, help="Evaluation episodes per trial")
    parser.add_argument("--opt_storage", type=str, default=None, help="Database URL for Optuna study")
    parser.add_argument("--video_output", type=str, default=None, help=argparse.SUPPRESS)
    # Rollout benchmark (mode=rollout-benchmark)
    parser.add_argument(
        "--benchmark_num_envs",
        type=int,
        nargs="+",
        default=[256, 512, 1024],
        help="num_envs values to sweep over for rollout-benchmark mode.",
    )
    args = parser.parse_args()
    from training_preferences import apply_training_preferences

    apply_training_preferences(args)

    run_name = build_run_name(
        run_name=args.run_name,
        task=args.task,
        robot=getattr(args, "robot", ""),
        sensors=getattr(args, "sensors", "none"),
        config=args.config,
    )

    if args.mode == "train":
        from experiment_runs import cli_run_identity

        level = getattr(args, "tactile_imperfection", None) or (
            "strong" if getattr(args, "sensors", "").endswith("/noisy") else "clean"
        )
        run_name += f"-{level}-" + cli_run_identity(args)

    # Initialize Eden
    en.init(
        backend=gs.gpu if not args.cpu else gs.cpu,
        log_root_path=f"logs/{run_name}/{args.mode}",
        performance_mode=args.mode not in {"play"},  # Disable performance mode for viewer/video modes
    )
    # Training uses a fixed (timestamp-free) log dir so a preempted job that is
    # resubmitted with --resume writes to the same place and finds its
    # checkpoints. Other modes keep Eden's timestamped dirs.
    if args.mode == "train":
        eden_timestamped_dir = en.log_dir
        en.log_dir = os.path.abspath(f"logs/{run_name}/train")
        os.makedirs(en.log_dir, exist_ok=True)
        # en.init() already created an empty timestamped dir; drop it.
        if os.path.isdir(eden_timestamped_dir) and not os.listdir(eden_timestamped_dir):
            os.rmdir(eden_timestamped_dir)
    print("Logging to", en.log_dir)

    config = get_task_config_from_args(args, run_name=run_name, upload_logs=args.mode == "train")

    if args.load_best_hyperparams:
        from optimization import apply_params_to_config, optuna, resolve_storage

        study = optuna.load_study(study_name=args.study_name, storage=resolve_storage(args.opt_storage))
        config = apply_params_to_config(study.best_params, config)

    if args.mode == "train":
        train(config, args)
    elif args.mode in ["play", "inference"]:
        play(config, args)
    elif args.mode in ["optimize", "opt"]:
        optimize(config, args)
    elif args.mode == "rollout-benchmark":
        rollout_benchmark(config, args)
    else:
        raise ValueError(f"Invalid mode: {args.mode}")


if __name__ == "__main__":
    main()
