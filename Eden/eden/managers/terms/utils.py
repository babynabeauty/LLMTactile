"""Joint-limit utility missing from the vendored tactile-genesis snapshot."""

import torch


def soft_dof_pos_violation(dofs_pos: torch.Tensor, limits: torch.Tensor) -> torch.Tensor:
    """Return per-DOF distance outside the supplied [lower, upper] band.

    ``limits`` has trailing shape ``(n_dofs, 2)``; leading batch dimensions
    broadcast against ``dofs_pos``. Values inside the band have zero penalty.
    """
    return torch.clamp(limits[..., 0] - dofs_pos, min=0.0) + torch.clamp(
        dofs_pos - limits[..., 1], min=0.0
    )
