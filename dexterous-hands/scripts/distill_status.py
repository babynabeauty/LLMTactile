#!/usr/bin/env python
"""Read local run statuses without reconstructing names or querying external services.

A recorded 'running' state is not proof that the process is still alive. This
command reports it literally and does not resubmit or modify runs.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from distill_manifest import read_run_status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("logs"))
    parser.add_argument("--task")
    parser.add_argument("--robot")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    rows = []
    for path in sorted(args.root.rglob("run_status.yaml")):
        status = read_run_status(path.parent)
        if args.task and status.get("task") != args.task:
            continue
        if args.robot and status.get("robot") != args.robot:
            continue
        rows.append({"log_dir": str(path.parent), **status})
    if args.json:
        print(json.dumps(rows, indent=2))
    else:
        for row in rows:
            print(f"{row.get('status', 'unknown'):12} {row['log_dir']} iteration={row.get('last_iteration', '?')}")
        print(f"{len(rows)} recorded runs; running states may be stale")


if __name__ == "__main__":
    main()
