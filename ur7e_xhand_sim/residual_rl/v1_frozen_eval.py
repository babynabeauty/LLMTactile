"""Unchanged V1 evaluation, reading byte-identical private scene assets."""
import argparse
from pathlib import Path

from .frozen_scene_assets import frozen_scene_assets
from . import v1_eval


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--frozen-assets', type=Path, required=True)
    args, remaining = parser.parse_known_args(argv)
    with frozen_scene_assets(args.frozen_assets):
        return v1_eval.main(remaining)


if __name__ == '__main__':
    main()
