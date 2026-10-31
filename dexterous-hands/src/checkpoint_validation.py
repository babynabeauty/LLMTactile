"""Fail on incompatible teacher weights before RSL-RL's permissive load."""


def load_checkpoint(path):
    import torch

    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError(f"Cannot read tensor checkpoint {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Checkpoint must be a mapping: {path}")
    return value


def read_teacher_checkpoint(path):
    value = load_checkpoint(path)
    actor = value.get("actor_state_dict")
    if not isinstance(actor, dict) or not actor:
        raise ValueError(f"Teacher checkpoint requires a nonempty actor_state_dict: {path}")
    return value


def validate_distillation_weights(runner, path):
    value = load_checkpoint(path)
    from_teacher = "actor_state_dict" in value
    pairs = [("teacher", "actor_state_dict" if from_teacher else "teacher_state_dict")]
    if not from_teacher:
        pairs.append(("student", "student_state_dict"))
    for model_name, key in pairs:
        saved = value.get(key)
        if not isinstance(saved, dict):
            raise ValueError(f"Missing {key} in {path}")
        expected = getattr(runner.alg, model_name).state_dict()
        unexpected = set(saved) - set(expected)
        if from_teacher:
            unexpected = {name for name in unexpected if not name.startswith("distribution.")}
        missing = set(expected) - set(saved)
        shapes = [
            name
            for name in expected.keys() & saved.keys()
            if getattr(saved[name], "shape", None) != expected[name].shape
        ]
        if missing or unexpected or shapes:
            raise ValueError(
                f"Incompatible {model_name} checkpoint {path}: missing={sorted(missing)}, "
                f"unexpected={sorted(unexpected)}, shape mismatch={shapes}"
            )
