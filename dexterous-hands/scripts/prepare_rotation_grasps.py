"""Build a fresh all-object cache with distinct settled poses and holding targets.

Candidates settle around a temporarily supported object, then undergo a cold
reset and a full 20-second unsupported replay with randomized training physics.
Existing caches are never overwritten. Passing candidates are initialization
data, not evidence that a teacher policy passes task acceptance.
"""

import argparse
import json
from pathlib import Path

import torch
import eden as en
import genesis as gs
from eden.envs.wrappers.rsl_rl_env import RslRlVecEnvWrapper
from eden.utils.configs import EdenRLConfig
from eden.utils.random import set_random_seed
from registry import _load_tasks, set_eval_num_envs, get_task_config
from teacher_evaluation import snapshot_run, sha256
from grasps_generator import check_penetration, check_finger_self_penetration, build_link_index_tensor


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--resolve_config", action="store_true", help="Resolve a task override instead of loading a full saved config")
    p.add_argument("--robot", choices=["sharpa", "wuji", "allegro"], default="sharpa")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--offsets", nargs="+", type=float, default=[0.15, 0.20, 0.25, 0.10])
    p.add_argument("--grasps_per_variant", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--replay_trials", type=int, default=1)
    p.add_argument("--source_cache", help="Override setup grasp data without changing the task recipe")
    args = p.parse_args()
    output_file = args.output / "grasps.pt"
    if output_file.exists():
        p.error("Output cache already exists; use a fresh output directory")
    en.init(backend=gs.gpu, log_root_path=str(args.output / "runtime"), logging_level="WARNING")
    _load_tasks()
    config = (get_task_config(task_name="in_fingers_rotate", modifiers={"robot": args.robot, "obj": "primitives", "sensors": "none", "stage": 1}, config_override_path=args.config)
              if args.resolve_config else EdenRLConfig.load_from_file(args.config))
    config.runner_options.logger = "tensorboard"
    config.runner_options.device = str(gs.device)
    config.env_options.episode_length_s = 20.0
    config.event_options.object_pulse.strength = 0.0
    config.action_options.dofs_pos_controller.accumulator_hard_limits = True
    config.action_options.dofs_pos_controller.reset_target_source = "cached_grasp_target"
    if args.source_cache:
        config.event_options.load_grasp.grasps_paths[args.robot] = args.source_cache
    source_path = config.event_options.load_grasp.grasps_paths[args.robot]
    source = torch.load(source_path, weights_only=False, map_location="cpu")
    variants, counts = source["geom_idx"].unique(sorted=True, return_counts=True)
    if len(variants) != len(config.scene_options.obj.grouped_entities) or len(variants) < 2:
        raise ValueError("Source cache must cover the complete multiple-object set")
    set_eval_num_envs(config, len(variants) * int(counts.max()))
    snapshot_run(config, args.output / "provenance")
    env = RslRlVecEnvWrapper.from_config(config, show_viewer=False, eval_mode=True)
    raw = env.unwrapped
    hand, obj = raw.entities["robot"], raw.entities["obj"]
    control = raw.action_manager.get_term("dofs_pos_controller")
    loader = next(term for _, name, term in raw.event_manager.iter_terms() if name == "load_grasp")
    mapping = torch.empty(env.num_envs, dtype=torch.long, device=gs.device)
    variant_index = torch.empty_like(mapping)
    for i, geom in enumerate(obj.links[0].geoms):
        ids = geom.active_envs_idx
        cache_ids = loader.grasps_per_geom[loader.env_to_grasp_geom_idx[geom.idx]]
        mapping[ids] = cache_ids[torch.arange(len(ids), device=gs.device) % len(cache_ids)]
        variant_index[ids] = i
    flexion = torch.tensor(
        [name.endswith(("_FE", "_PIP", "_DIP")) or "bend" in name or (args.robot == "wuji" and not name.endswith("_abd")) for name in control.dofs_name], device=gs.device
    )
    finger_indices = build_link_index_tensor(hand.options.metadata.finger_links, hand, gs.device)
    selected = [[] for _ in variants]
    reports = []

    def apply_state(state):
        hand.set_pos(state["robot_pos"])
        hand.set_quat(state["robot_quat"])
        hand.set_dofs_pos(state["joint_pos"])
        hand.zero_all_dofs_velocity()
        obj.set_pos(state["obj_pos"])
        obj.set_quat(state["obj_quat"])
        obj.zero_all_dofs_velocity()
        control.target.copy_(state["joint_target"])
        control.filtered_target.copy_(control.target)
        control._processed_action.copy_(control.target)
        control._accumulated_target.copy_(control.target)
        control.target_velocity.zero_()
        raw._sampled_grasp_joint_pos.copy_(state["joint_pos"])
        raw._sampled_grasp_obj_pos.copy_(state["obj_pos"])

    for round_id, offset in enumerate(args.offsets):
        set_random_seed(args.seed + round_id)
        env.reset()
        q = loader.grasp_joint_pos[mapping].clone()
        target = loader.grasp_joint_target[mapping].clone()
        target[:, flexion] += offset
        target.clamp_(control.lower, control.upper)
        state = dict(
            joint_pos=q, joint_target=target,
            robot_pos=loader.grasp_robot_pos[mapping].clone(),
            robot_quat=loader.grasp_robot_quat[mapping].clone(),
            obj_pos=loader.grasp_obj_pos[mapping].clone(),
            obj_quat=loader.grasp_obj_quat[mapping].clone(),
        )
        apply_state(state)
        for _ in range(round(1 / raw.sim_dt)):
            control.apply_actions()
            raw.scene.step()
            obj.set_pos(state["obj_pos"])
            obj.set_quat(state["obj_quat"])
            obj.zero_all_dofs_velocity()
        state["joint_pos"] = hand.get_dofs_pos().clone()
        candidate_file = args.output / f"candidates-{round_id}.pt"
        torch.save({k: v.cpu() for k, v in state.items()}, candidate_file)
        # A real reset clears velocities and contact history and draws fresh mass,
        # friction and gains. Replay never supports or teleports the object.
        valid = torch.ones(env.num_envs, dtype=torch.bool, device=gs.device)
        trial_retention = []
        for trial in range(args.replay_trials):
            set_random_seed(args.seed + 100 * round_id + trial + 1)
            raw.terminal_state_callback = None
            env.reset()
            apply_state(torch.load(candidate_file, map_location=gs.device, weights_only=False))
            finished = torch.zeros(env.num_envs, dtype=torch.bool, device=gs.device)
            dropped = torch.zeros_like(finished)
            finite = torch.ones_like(finished)
            max_penetration = torch.zeros(env.num_envs, device=gs.device)

            def capture(e):
                live = ~finished
                _, penetration, available = check_penetration(env.num_envs, hand, obj, 0.003, gs.device)
                _, self_penetration, self_available = check_finger_self_penetration(
                    env.num_envs, hand, finger_indices, 0.003, gs.device
                )
                if not available or not self_available:
                    raise RuntimeError("Penetration diagnostics unavailable; refusing to approve a cache")
                max_penetration.copy_(torch.maximum(max_penetration, torch.maximum(penetration, self_penetration) * live))
                good = torch.isfinite(hand.get_dofs_pos()).all(-1) & torch.isfinite(hand.get_dofs_vel()).all(-1)
                good &= torch.isfinite(obj.get_pos()).all(-1) & torch.isfinite(obj.get_quat()).all(-1)
                finite.logical_and_(good | ~live)
                dropped.logical_or_(e.reset_terminated & live)
                finished.logical_or_(e.reset_buf)

            raw.terminal_state_callback = capture
            with torch.no_grad():
                for _ in range(round(20 / raw.dt)):
                    env.step(torch.zeros(env.num_envs, env.num_actions, device=gs.device))
            valid &= finished & ~dropped & finite & (max_penetration <= 0.003)
            trial_retention.append(float((~dropped).float().mean()))
            print("CACHE_TRIAL", round_id, trial, "valid_across_trials", int(valid.sum()), flush=True)
        for i in range(len(variants)):
            ids = torch.where(valid & (variant_index == i))[0]
            for idx in ids[:max(0, args.grasps_per_variant - len(selected[i]))]:
                selected[i].append({k: v[idx].cpu() for k, v in state.items()})
        report = dict(round=round_id, offset=offset, retained=float((~dropped).float().mean()),
                      valid=int(valid.sum()), trial_retention=trial_retention, finite=bool(finite.all()), counts=[len(s) for s in selected])
        reports.append(report)
        # Preserve valid candidates even if a later round cannot fill every quota.
        torch.save({
            "selected": selected, "reports": reports, "source_sha256": sha256(source_path),
            "config_sha256": sha256(args.config), "seed": args.seed,
        }, args.output / "selection-progress.pt")
        (args.output / "progress.json").write_text(json.dumps(reports, indent=2))
        print("CACHE_ROUND", json.dumps(report), flush=True)
        if all(len(s) == args.grasps_per_variant for s in selected):
            break
    if any(len(s) < args.grasps_per_variant for s in selected):
        raise RuntimeError("Cache quota incomplete; no training cache published. See progress.json")
    rows = [row for group in selected for row in group]
    result = {key: torch.stack([row[key] for row in rows]) for key in rows[0]}
    result.update(
        geom_idx=torch.arange(len(variants)).repeat_interleave(args.grasps_per_variant),
        joint_names=list(hand.dofs_name), obj_pose_frame="root", task="in_fingers_rotate",
        save_object_pose=True, cache_version=2, source_sha256=sha256(source_path),
        preparation=dict(seed=args.seed, offsets=args.offsets, replay_seconds=20, replay_trials=args.replay_trials, max_penetration=0.003),
    )
    torch.save(result, output_file)
    print("CACHE_READY", str(output_file), sha256(output_file), flush=True)
    env.close()


if __name__ == "__main__":
    main()
