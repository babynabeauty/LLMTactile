"""User launch defaults: checkpoint videos and the multiple-object set."""


def apply_training_preferences(args):
    if args.mode not in {"train", "opt", "optimize"}:
        return
    if getattr(args, "obj", None) == "cube":
        print("Training preference: replacing cube-only with the primitives object set.")
        args.obj = "primitives"
    if getattr(args, "no_checkpoint_video", False):
        print("Training preference: checkpoint videos are always enabled.")
    args.no_checkpoint_video = False


def validate_training_objects(config):
    obj = getattr(config.scene_options, "obj", None)
    if len(getattr(obj, "grouped_entities", None) or []) < 2:
        raise ValueError("Training requires the multiple-object set, including with --full_config.")
