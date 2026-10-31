"""Short opt-in simulator smoke check; does not train or require teacher checkpoints."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import eden as en
import genesis as gs
import torch
from eden.envs.wrappers.rsl_rl_env import RslRlVecEnvWrapper
from registry import get_task_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--robot", required=True)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--sensors", default="low-hand/force")
    args = parser.parse_args()
    en.init(backend=gs.cpu if args.cpu else gs.gpu, log_root_path=".tmp/runtime", performance_mode=True)
    cfg = get_task_config(
        task_name=args.task,
        modifiers=dict(
            robot=args.robot,
            stage=2,
            sensors=args.sensors,
            tactile_imperfection="strong",
            model="tac_mlp",
            tactile_encoder="tactile_convrnn",
        ),
    )
    cfg.env_options.num_envs = 2
    cfg.runner_options.device = "cpu" if args.cpu else "cuda:0"
    cfg.cameras_options = None
    for name in cfg.sensors_options.model_fields_set:
        option = getattr(cfg.sensors_options, name, None)
        if getattr(option, "sensor", None) is not None:
            option.sensor.draw_debug = False
    env = RslRlVecEnvWrapper.from_config(cfg, show_viewer=False)
    try:
        obs, _ = env.reset()
        for _ in range(3):
            obs, reward, done, info = env.step(torch.zeros((2, env.num_actions), device=gs.device))
            assert torch.isfinite(reward).all()
            assert all(torch.isfinite(v).all() for v in obs.values())
        print(f"PASS {args.task}/{args.robot}: {env.num_actions} actions, observations {list(obs.keys())}")
    finally:
        if hasattr(env, "close"):
            env.close()


if __name__ == "__main__":
    main()
