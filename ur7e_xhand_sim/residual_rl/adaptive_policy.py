"""Compact causal-window encoder with raw and summarized tactile inputs.

Observation contract:
  tactile: (12, 5, 120, 3), chronological measured forces in finger-link XYZ.
  state:   (12, D), synchronized measured robot/nominal-plan state, or (D,).

The flat-state compatibility path marks only the latest state as valid; it does
not repeat today's state into the past. Use the SAME full robot-state history
for tactile and no-tactile training comparisons. No simulator labels, object
pose, mass, contact velocity, or friction are read here.

Summary "force" quantities are sums of sensor estimates, not independent
measurements of the solver contact load. All transforms use fixed scales and
backward differences. There is no across-time normalization or future padding.
"""
from __future__ import annotations

import math

import gymnasium as gym
import torch
from torch import nn
from torch.nn import functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class _CausalConv(nn.Module):
    """Preserve sequence length with left-only zero padding."""
    def __init__(self, channels: int, dilation: int):
        super().__init__()
        self.left_padding = 2 * dilation
        self.conv = nn.Conv1d(channels, channels, kernel_size=3, dilation=dilation)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(values, (self.left_padding, 0)))


class AdaptiveTactileExtractor(BaseFeaturesExtractor):
    """Ordered taxel MLP + sensor summaries + shared robot-state history.

    ``use_tactile=False`` replaces raw forces with zero BEFORE every raw,
    summary, difference, and projection operation. It therefore also removes
    tactile information from the auxiliary feature path.

    ``use_shear=False`` keeps only force projected onto supplied static taxel
    normals, in the same link frame as ``tactile``. It requires calibration
    normals when tactile is enabled. This is a sensor-normal projection
    ablation, not a claim to remove the simulator's velocity-derived force
    term exactly: its contact normal can differ from the sensor's normal.
    To isolate that synthetic physics term, use a separate sensor-generation
    experiment with its shear coefficient disabled; never pass dynamic solver
    contact normals to this actor.
    """
    history_frames = 12
    finger_count = 5
    taxel_count = 120
    summary_names = (
        "sum_fx", "sum_fy", "sum_fz", "sum_taxel_force_norms",
        "peak_taxel_force_norm", "active_taxel_fraction",
    )

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        features_dim: int = 128,
        use_tactile: bool = True,
        use_shear: bool = True,
        normal_directions_link=None,
        force_scale_N: float = .1,
        sum_force_scale_N: float = 1.,
        active_threshold_N: float = 1e-5,
    ):
        super().__init__(observation_space, features_dim)
        if observation_space.spaces["tactile"].shape != (12, 5, 120, 3):
            raise ValueError("tactile observation must have shape (12,5,120,3)")
        state_shape = observation_space.spaces["state"].shape
        if not (len(state_shape) == 1 or (len(state_shape) == 2 and state_shape[0] == 12)):
            raise ValueError("state observation must have shape (D,) or (12,D)")
        self.state_dim = state_shape[-1]
        if self.state_dim < 1:
            raise ValueError("state must contain at least one field")
        for name, value in (("force_scale_N", force_scale_N),
                            ("sum_force_scale_N", sum_force_scale_N),
                            ("active_threshold_N", active_threshold_N)):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        self.force_scale_N = float(force_scale_N)
        self.sum_force_scale_N = float(sum_force_scale_N)
        self.active_threshold_N = float(active_threshold_N)
        self.use_tactile = bool(use_tactile)
        self.use_shear = bool(use_shear)

        if normal_directions_link is None:
            if self.use_tactile and not self.use_shear:
                raise ValueError("use_shear=False requires calibrated static normal_directions_link (5,120,3)")
            normals = torch.zeros(5, 120, 3, dtype=torch.float32)
        else:
            normals = torch.as_tensor(normal_directions_link, dtype=torch.float32).detach().clone()
            if normals.shape != (5, 120, 3) or not torch.isfinite(normals).all():
                raise ValueError("normal_directions_link must be finite with shape (5,120,3)")
            lengths = torch.linalg.vector_norm(normals, dim=-1, keepdim=True)
            if (lengths <= 1e-8).any():
                raise ValueError("Every calibrated taxel normal must have nonzero length")
            normals = normals / lengths
        self.register_buffer("normal_directions_link", normals)

        # Flattening preserves each ordered taxel's identity before aggregation;
        # no averaging of the raw spatial pattern precedes this learned mapping.
        self.raw_encoder = nn.Sequential(
            nn.Linear(120 * 3, 48), nn.SiLU(), nn.Linear(48, 24), nn.SiLU(),
        )
        self.state_encoder = nn.Sequential(nn.Linear(2 * self.state_dim + 1, 48), nn.SiLU())
        self.frame_encoder = nn.Sequential(nn.Linear(5 * (24 + 6 + 6) + 48, 64), nn.SiLU())
        self.temporal_encoder = nn.Sequential(
            _CausalConv(64, dilation=1), nn.SiLU(),
            _CausalConv(64, dilation=2), nn.SiLU(),
        )
        self.fusion = nn.Sequential(nn.Linear(64 + 64 + 48, features_dim), nn.SiLU())

    @staticmethod
    def _backward_difference(sequence: torch.Tensor) -> torch.Tensor:
        # Unknown pre-window value is not assumed zero; its difference is zero.
        return torch.cat((torch.zeros_like(sequence[:, :1]), sequence[:, 1:] - sequence[:, :-1]), dim=1)

    def _prepare_tactile(self, raw: torch.Tensor) -> torch.Tensor:
        if raw.ndim != 5 or tuple(raw.shape[1:]) != (12, 5, 120, 3):
            raise ValueError("Batched tactile tensor must have shape (B,12,5,120,3)")
        force = raw.float()
        if not self.use_tactile:
            # Do not derive magnitudes, contact flags, or statistics beforehand.
            return torch.zeros_like(force)
        if not self.use_shear:
            normals = self.normal_directions_link[None, None]
            force = (force * normals).sum(dim=-1, keepdim=True) * normals
        return force

    def _summaries_from_prepared(self, force: torch.Tensor) -> torch.Tensor:
        magnitude = torch.linalg.vector_norm(force, dim=-1)
        vector_sum = force.sum(dim=-2)
        total = magnitude.sum(dim=-1, keepdim=True)
        peak = magnitude.amax(dim=-1, keepdim=True)
        fraction = (magnitude > self.active_threshold_N).to(force.dtype).mean(dim=-1, keepdim=True)
        base = torch.cat((
            torch.asinh(vector_sum / self.sum_force_scale_N),
            torch.asinh(total / self.sum_force_scale_N),
            torch.asinh(peak / self.force_scale_N), fraction,
        ), dim=-1)
        return torch.cat((base, self._backward_difference(base)), dim=-1)

    def tactile_summaries(self, raw: torch.Tensor) -> torch.Tensor:
        """Return (B,12,5,12) transformed summaries, respecting both ablations."""
        return self._summaries_from_prepared(self._prepare_tactile(raw))

    def _prepare_state(self, raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        state = raw.float()
        if state.ndim == 2 and state.shape[-1] == self.state_dim:
            history = torch.cat((
                state.new_zeros(state.shape[0], 11, self.state_dim), state[:, None],
            ), dim=1)
            valid = state.new_zeros(state.shape[0], 12, 1)
            valid[:, -1] = 1.
        elif state.ndim == 3 and tuple(state.shape[1:]) == (12, self.state_dim):
            history = state
            valid = state.new_ones(state.shape[0], 12, 1)
        else:
            raise ValueError("Batched state tensor must have shape (B,D) or (B,12,D)")
        both_valid = valid * torch.cat((torch.zeros_like(valid[:, :1]), valid[:, :-1]), dim=1)
        delta = self._backward_difference(history) * both_valid
        return torch.cat((history, delta, valid), dim=-1), valid

    def _encoded_sequence(self, observations: dict[str, torch.Tensor]):
        force = self._prepare_tactile(observations["tactile"])
        batch = force.shape[0]
        summaries = self._summaries_from_prepared(force)
        raw = self.raw_encoder(torch.asinh(force / self.force_scale_N).reshape(batch * 12 * 5, 360))
        raw = raw.reshape(batch, 12, 5, 24)
        tactile = torch.cat((raw, summaries), dim=-1).reshape(batch, 12, -1)
        state_input, valid = self._prepare_state(observations["state"])
        if state_input.shape[0] != batch:
            raise ValueError("Tactile and state batch sizes differ")
        state = self.state_encoder(state_input) * valid
        frames = self.frame_encoder(torch.cat((tactile, state), dim=-1))
        temporal = self.temporal_encoder(frames.transpose(1, 2)).transpose(1, 2)
        return temporal, state

    def encode_sequence(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        """Return (B,12,64); position t depends only on samples through t.

        Exposed for the causal-prefix contract test and diagnostics. Final
        actions use forward(), which pools only this already-observed window.
        """
        return self._encoded_sequence(observations)[0]

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        temporal, state = self._encoded_sequence(observations)
        return self.fusion(torch.cat((temporal[:, -1], temporal.mean(dim=1), state[:, -1]), dim=-1))


def policy_kwargs(
    use_tactile: bool = True,
    use_shear: bool = True,
    normal_directions_link=None,
    action_std_init: float = .1,
) -> dict:
    """SB3 MultiInputPolicy kwargs; no environment or training side effects."""
    if not math.isfinite(action_std_init) or action_std_init <= 0:
        raise ValueError("action_std_init must be finite and positive")
    return {
        "features_extractor_class": AdaptiveTactileExtractor,
        "features_extractor_kwargs": {
            "features_dim": 128, "use_tactile": use_tactile, "use_shear": use_shear,
            "normal_directions_link": normal_directions_link,
        },
        "net_arch": {"pi": [64, 64], "vf": [64, 64]},
        "activation_fn": nn.SiLU,
        "normalize_images": False,
        "log_std_init": math.log(action_std_init),
    }
