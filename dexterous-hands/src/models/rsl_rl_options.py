"""Option classes for the rsl_rl knobs the Milotrince/rsl_rl fork reads but Eden does not declare."""

from typing import Any

from eden.options.learning.rsl_rl import RslRlDistillationAlgorithmOptions as EdenDistillationAlgorithmOptions


class RslRlDistillationAlgorithmOptions(EdenDistillationAlgorithmOptions):
    """Distillation options.

    Parameters
    ----------
    normalize_action_targets: bool
        Standardize teacher action targets per-dim before the behavior loss.
    auxiliary_losses: list[dict[str, Any]] | None
        Auxiliary loss specs (``class_name`` plus its kwargs), resolved by rsl_rl.
    """

    normalize_action_targets: bool = False
    auxiliary_losses: list[dict[str, Any]] | None = None
