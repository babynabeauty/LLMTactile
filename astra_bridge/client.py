"""Small, dependency-free client for the persistent Tactile Genesis bridge."""
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request


def call(command: str, payload: dict | None = None, url: str = "http://127.0.0.1:8765") -> dict:
    request = urllib.request.Request(
        f"{url}/{command}",
        data=json.dumps(payload or {}, allow_nan=False).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            result = json.load(response)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(exc.read().decode()) from exc
    if not result.get("ok"):
        raise RuntimeError(result)
    return result["result"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["describe", "reset", "observe", "step", "status", "close"])
    parser.add_argument("--url", default="http://127.0.0.1:8765")
    parser.add_argument("--json", default="{}", help="JSON object; use '-' to read stdin")
    parser.add_argument("--output", help="Save the exact response as JSON")
    parser.add_argument("--compact", action="store_true", help="Print compact observations; --output still saves the full response")
    args = parser.parse_args()
    try:
        payload = json.load(sys.stdin) if args.json == "-" else json.loads(args.json)
        result = call(args.command, payload, args.url)
        encoded = json.dumps(result, ensure_ascii=False, allow_nan=False)
        if args.output:
            from pathlib import Path
            Path(args.output).write_text(encoded + "\n")
        if args.compact and "joint_positions_rad" in result:
            compact = {k: result[k] for k in ["episode", "episode_step", "sim_time_s", "total_decisions", "terminated", "rgb_path", "goal_quaternion_wxyz"]}
            compact["q_rad_in_described_joint_order"] = [round(v, 4) for v in result["joint_positions_rad"].values()]
            compact["active_tactile_regions"] = {k: v for k, v in result["tactile"].items() if v["active_taxels"]}
            encoded = json.dumps(compact, ensure_ascii=False, allow_nan=False)
        print(encoded)
    except (ValueError, RuntimeError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
