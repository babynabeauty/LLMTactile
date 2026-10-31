"""Inspect any registered hand asset without a task, policy, or calibration preset."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import genesis as gs
from registry import ROBOT_REGISTRY, _load_tasks


def main():
    _load_tasks()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", required=True, choices=sorted(ROBOT_REGISTRY.keys()))
    parser.add_argument("--asset", type=Path, help="Override an external hand definition with a local URDF/MJCF")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--steps", type=int, default=1000)
    args = parser.parse_args()
    robot = ROBOT_REGISTRY.get(args.robot)()
    from genesis.utils.misc import get_assets_dir

    path = args.asset or Path(robot.file)
    if not path.is_file():
        path = Path(get_assets_dir()) / path
    if not path.is_file():
        parser.error(f"Hand asset is not installed: {robot.file}. Supply its URDF/MJCF using --asset.")
    robot.file = str(path.resolve())
    gs.init(backend=gs.cpu if args.cpu else gs.gpu, logging_level="warning")
    scene = gs.Scene(sim_options=gs.options.SimOptions(gravity=(0, 0, 0)), show_viewer=not args.headless)
    morph = gs.morphs.MJCF if robot.file.endswith(".xml") else gs.morphs.URDF
    hand = scene.add_entity(morph(file=robot.file))
    scene.build(n_envs=1)
    print(f"{args.robot}: {hand.n_dofs} DOFs, {hand.n_links} links")
    for _ in range(args.steps):
        scene.step()


if __name__ == "__main__":
    main()
