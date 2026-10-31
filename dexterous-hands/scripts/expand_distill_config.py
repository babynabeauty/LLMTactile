#!/usr/bin/env python
"""Validate and preview tactile experiments, or submit the validated matrix to SLURM."""

import argparse
import contextlib
import json
import shlex
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from distill_experiments import ROOT, command, expand, preflight


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="*", help="Named hand/task entries; omitted selects all nine")
    parser.add_argument("--config", default=str(ROOT / "conf/experiments/tactile_distillation.yaml"))
    parser.add_argument("--submit", action="store_true", help="Submit only if every job passes preflight")
    parser.add_argument("--json", action="store_true", help="Emit resolved configs, provenance, and errors")
    parser.add_argument("--partition", default=None, help="SLURM partition for --submit")
    args = parser.parse_args()
    cache = {}
    try:
        with contextlib.redirect_stdout(sys.stderr):
            jobs = [preflight(job, cache) for job in expand(args.config, args.names)]
    except (ValueError, KeyError, OSError) as exc:
        parser.error(str(exc))
    if args.json:
        print(json.dumps(jobs, indent=2))
    else:
        for job in jobs:
            print(shlex.join([sys.executable, "main.py", *command(job)]))
            for error in job["errors"]:
                print("  NOT READY: " + error)
        print(f"{len(jobs)} jobs; {sum(bool(j['errors']) for j in jobs)} not ready")
    if any(j["errors"] for j in jobs):
        return 2
    if args.submit:
        (ROOT / "logs/slurm").mkdir(parents=True, exist_ok=True)
        for job in jobs:
            subprocess.run(
                [
                    "sbatch",
                    "--job-name=" + job["run_name"],
                    *(["--partition=" + args.partition] if args.partition else []),
                    "scripts/slurm/run.sh",
                    "train",
                    *command(job),
                ],
                cwd=ROOT,
                check=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
