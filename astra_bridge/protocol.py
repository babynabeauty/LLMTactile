"""Validation shared by the server and fast unit tests (no GPU imports)."""
import math


def validate_step(payload: dict, joint_names: list[str], previous: list[float]) -> tuple[list[float], int]:
    if not isinstance(payload, dict):
        raise ValueError("Payload must be an object")
    unknown = set(payload) - {"actions", "steps", "note", "expected_episode", "expected_step"}
    if unknown:
        raise ValueError(f"Unknown step fields: {sorted(unknown)}")
    steps = payload.get("steps", 4)
    if type(steps) is not int or not 1 <= steps <= 10:
        raise ValueError("steps must be an integer in [1, 10]")
    actions = payload.get("actions", {})
    if not isinstance(actions, dict):
        raise ValueError("actions must map joint names to normalized position targets")
    result = list(previous)
    for name, value in actions.items():
        if name not in joint_names:
            raise ValueError(f"Unknown joint: {name}")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"Action for {name} must be a finite number")
        if not -1 <= value <= 1:
            raise ValueError(f"Action for {name} must be in [-1, 1]")
        result[joint_names.index(name)] = float(value)
    return result, steps
