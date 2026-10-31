"""Integration tests for TactilePreEncodeMLPModel / TactilePreEncodeRecurrentModel
with the IntersectionRNN-based ConvRNN encoder. These exercise the full
rsl_rl-facing surface: model build, ``is_recurrent`` toggling, hidden-state
round-trip, reset propagation, and batched (T, B) training-mode forward.

Skipped entirely if ``pt_tnn`` is not installed.

Run with::

    cd dexterous_hands
    uv run pytest tests/models/test_tactile_pre_encode.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from tensordict import TensorDict

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

pytest.importorskip("pt_tnn", reason="pt_tnn not installed in this venv")

from models.tactile_pre_encode import (  # noqa: E402
    TactilePreEncodeMLPModel,
    TactilePreEncodeRecurrentModel,
)


# ---- fixtures ----------------------------------------------------------


LAYOUT = {
    "num_sensors": 5,
    "grid_h": 3,
    "grid_w": 3,
    "features_per_probe": 3,
    "history_length": 1,
}
TACTILE_FLAT = (
    LAYOUT["history_length"] * LAYOUT["num_sensors"]
    * LAYOUT["features_per_probe"] * LAYOUT["grid_h"] * LAYOUT["grid_w"]
)
PROPRIO_DIM = 20
GOAL_DIM = 10
ACTION_DIM = 8


def _obs(batch: int) -> TensorDict:
    return TensorDict(
        {
            "tactile_sensors": torch.randn(batch, TACTILE_FLAT),
            "proprio": torch.randn(batch, PROPRIO_DIM),
            "goal": torch.randn(batch, GOAL_DIM),
        },
        batch_size=[batch],
    )


def _obs_seq(time: int, batch: int) -> TensorDict:
    return TensorDict(
        {
            "tactile_sensors": torch.randn(time, batch, TACTILE_FLAT),
            "proprio": torch.randn(time, batch, PROPRIO_DIM),
            "goal": torch.randn(time, batch, GOAL_DIM),
        },
        batch_size=[time, batch],
    )


def _obs_groups() -> dict[str, list[str]]:
    return {"student": ["proprio", "goal", "tactile_sensors"]}


def _convrnn_encoder_cfg() -> dict:
    return {
        "tactile_sensors": {
            "kind": "tactile_convrnn",
            "output_dim": 32,
            "shared_per_sensor": False,
            "tactile_layout": dict(LAYOUT),
            "convrnn": {"out_channels": 6, "ksize": 3, "layernorm": True},
        },
        "proprio": {"kind": "mlp", "hidden_dims": [32], "output_dim": 16},
    }


def _cnn_encoder_cfg() -> dict:
    return {
        "tactile_sensors": {
            "kind": "tactile_cnn",
            "output_dim": 32,
            "shared_per_sensor": False,
            "tactile_layout": dict(LAYOUT),
            "cnn": {"output_channels": [8, 16], "kernel_size": 3, "padding": "zeros"},
        },
        "proprio": {"kind": "mlp", "hidden_dims": [32], "output_dim": 16},
    }


def _build_mlp_model(encoder_cfg: dict) -> TactilePreEncodeMLPModel:
    obs = _obs(batch=4)
    return TactilePreEncodeMLPModel(
        obs=obs, obs_groups=_obs_groups(), obs_set="student",
        output_dim=ACTION_DIM, hidden_dims=[64, 64], activation="elu",
        obs_normalization=False, distribution_cfg=None,
        encoder_cfg=encoder_cfg,
    )


def _build_rnn_model(encoder_cfg: dict, rnn_type: str = "lstm") -> TactilePreEncodeRecurrentModel:
    obs = _obs(batch=4)
    return TactilePreEncodeRecurrentModel(
        obs=obs, obs_groups=_obs_groups(), obs_set="student",
        output_dim=ACTION_DIM, hidden_dims=[64, 64], activation="elu",
        obs_normalization=False, distribution_cfg=None,
        encoder_cfg=encoder_cfg,
        rnn_type=rnn_type, rnn_hidden_dim=32, rnn_num_layers=1,
    )


# ---- MLP head + (tactile_cnn | tactile_convrnn) ------------------------


def test_mlp_model_with_cnn_is_not_recurrent() -> None:
    model = _build_mlp_model(_cnn_encoder_cfg())
    assert model.is_recurrent is False
    assert model.get_hidden_state() is None
    # reset is a no-op for non-recurrent models — must not raise.
    model.reset(torch.tensor([1, 0, 0, 1]))


def test_mlp_model_with_convrnn_is_recurrent() -> None:
    model = _build_mlp_model(_convrnn_encoder_cfg())
    assert model.is_recurrent is True
    # Before any forward, state is None.
    assert model.get_hidden_state() is None
    obs = _obs(batch=4)
    out = model(obs)
    assert out.shape == (4, ACTION_DIM)
    # After forward, encoder state populated; exposed as 3D (1, B, D).
    state = model.get_hidden_state()
    assert isinstance(state, torch.Tensor)
    assert state.dim() == 3 and state.shape[0] == 1 and state.shape[1] == 4


def test_mlp_model_hidden_state_roundtrip() -> None:
    model = _build_mlp_model(_convrnn_encoder_cfg())
    obs = _obs(batch=3)
    model(obs)
    saved = model.get_hidden_state().clone()

    # Wipe internal state, then restore via the public API.
    model.reset(None)
    assert model.get_hidden_state() is None
    model.set_hidden_state(saved)
    restored = model.get_hidden_state()
    assert torch.allclose(restored, saved)


def test_mlp_model_reset_dones_propagates_to_encoder() -> None:
    model = _build_mlp_model(_convrnn_encoder_cfg())
    obs = _obs(batch=4)
    model(obs)
    state = model.get_hidden_state().clone()
    assert state[0, 0].abs().sum() > 0  # warmed up

    dones = torch.tensor([1, 0, 1, 0])
    model.reset(dones)
    after = model.get_hidden_state()
    assert after[0, 0].abs().sum() == 0
    assert after[0, 2].abs().sum() == 0
    assert torch.allclose(after[0, 1], state[0, 1])
    assert torch.allclose(after[0, 3], state[0, 3])


def test_mlp_model_inference_state_persists_across_calls() -> None:
    model = _build_mlp_model(_convrnn_encoder_cfg())
    obs = _obs(batch=2)
    out0 = model(obs).detach().clone()
    state0 = model.get_hidden_state().clone()
    out1 = model(obs).detach().clone()
    state1 = model.get_hidden_state().clone()
    # State must evolve; outputs must differ on the same input (because the
    # encoder's recurrent state is non-zero on the second call).
    assert not torch.allclose(state0, state1)
    assert not torch.allclose(out0, out1)


# ---- RNN head + tactile_convrnn ---------------------------------------


@pytest.mark.parametrize("rnn_type", ["lstm", "gru"])
def test_rnn_model_combines_policy_and_encoder_state(rnn_type: str) -> None:
    model = _build_rnn_model(_convrnn_encoder_cfg(), rnn_type=rnn_type)
    assert model.is_recurrent is True
    obs = _obs(batch=3)
    out = model(obs)
    assert out.shape == (3, ACTION_DIM)

    state = model.get_hidden_state()
    if rnn_type == "lstm":
        # LSTM contributes (h, c); encoder appends a 3rd tensor.
        assert isinstance(state, tuple) and len(state) == 3
        h_rnn, c_rnn, enc = state
        assert h_rnn.shape[1] == 3 and c_rnn.shape[1] == 3
        assert enc.shape[0] == 1 and enc.shape[1] == 3
    else:
        # GRU contributes a single hidden tensor; encoder is the second.
        assert isinstance(state, tuple) and len(state) == 2
        h_rnn, enc = state
        assert h_rnn.shape[1] == 3
        assert enc.shape[0] == 1 and enc.shape[1] == 3


def test_rnn_model_hidden_state_roundtrip() -> None:
    model = _build_rnn_model(_convrnn_encoder_cfg(), rnn_type="lstm")
    obs = _obs(batch=4)
    model(obs)
    saved = tuple(s.clone() for s in model.get_hidden_state())

    model.reset(None)
    after_reset = model.get_hidden_state()
    # After full reset, both policy and encoder states should be cleared.
    assert after_reset is None or all(s is None or s.abs().sum() == 0 for s in (after_reset if isinstance(after_reset, tuple) else (after_reset,)))

    model.set_hidden_state(saved)
    restored = model.get_hidden_state()
    assert isinstance(restored, tuple) and len(restored) == len(saved)
    for r, s in zip(restored, saved):
        assert torch.allclose(r, s)


def test_rnn_model_reset_dones_zeros_both_states() -> None:
    model = _build_rnn_model(_convrnn_encoder_cfg(), rnn_type="lstm")
    obs = _obs(batch=4)
    model(obs)

    dones = torch.tensor([1, 0, 1, 0])
    model.reset(dones)
    state = model.get_hidden_state()
    h_rnn, c_rnn, enc = state
    # Policy RNN state has zeros on done rows.
    assert h_rnn[:, 0, :].abs().sum() == 0
    assert h_rnn[:, 2, :].abs().sum() == 0
    assert c_rnn[:, 0, :].abs().sum() == 0
    # Encoder state has zeros on done rows.
    assert enc[0, 0, :].abs().sum() == 0
    assert enc[0, 2, :].abs().sum() == 0


# ---- batched (training-mode) forward through TactilePreEncodeMLPModel --


def test_mlp_model_batched_forward_with_full_masks() -> None:
    """Simulate the training path PPO uses: padded (T, B) obs + all-valid mask."""
    model = _build_mlp_model(_convrnn_encoder_cfg())
    T_pad, B = 5, 3
    seq_obs = _obs_seq(T_pad, B)
    masks = torch.ones(T_pad, B, dtype=torch.bool)
    # Provide an initial hidden state, shaped on the wire as (1, B, D).
    enc_state_dim = LAYOUT["num_sensors"] * 6 * LAYOUT["grid_h"] * LAYOUT["grid_w"]
    init_state = torch.zeros(1, B, enc_state_dim)

    out = model(seq_obs, masks=masks, hidden_state=init_state)
    # Output is (T_pad, B, action_dim) — same leading-T shape after unpad with a
    # full-valid mask, since no rows were dropped.
    assert out.dim() == 3
    assert out.shape[-1] == ACTION_DIM
    assert out.shape[1] == B


def test_mlp_model_batched_forward_gradients_flow_to_encoder() -> None:
    model = _build_mlp_model(_convrnn_encoder_cfg())
    T_pad, B = 4, 2
    seq_obs = _obs_seq(T_pad, B)
    masks = torch.ones(T_pad, B, dtype=torch.bool)

    out = model(seq_obs, masks=masks, hidden_state=None)
    out.sum().backward()

    # Both per-finger input projections and IntersectionRNN conv weights must
    # accumulate gradient (random inputs, non-trivial loss).
    enc = model.encoders["tactile_sensors"]
    proj0 = enc.input_projs[0]
    cell0 = enc.cells[0]
    assert proj0.weight.grad is not None and proj0.weight.grad.abs().sum() > 0
    cell_conv = cell0.conv
    assert cell_conv.weight.grad is not None and cell_conv.weight.grad.abs().sum() > 0


# ---- task_mods option resolution ---------------------------------------


def test_task_mods_resolves_tac_options() -> None:
    """Smoke-check that the runner mod can produce the right options class for tac_*."""
    import task_mods  # noqa: F401  (registers _MODEL_CLASSES)
    from models import (
        RslRlActorTactilePreEncodeMLPOptions,
        RslRlActorTactilePreEncodeRNNOptions,
        RslRlTactilePreEncodeMLPOptions,
        RslRlTactilePreEncodeRNNOptions,
    )

    classes = task_mods.RslRlRunnerMod._MODEL_CLASSES
    assert classes["actor"]["tac_mlp"] is RslRlActorTactilePreEncodeMLPOptions
    assert classes["actor"]["tac_rnn"] is RslRlActorTactilePreEncodeRNNOptions
    assert classes["critic"]["tac_mlp"] is RslRlTactilePreEncodeMLPOptions
    assert classes["critic"]["tac_rnn"] is RslRlTactilePreEncodeRNNOptions

    # Verify the class_name field resolves through rsl_rl.utils.resolve_callable
    # — this is what the runner will do at policy build time.
    from rsl_rl.utils import resolve_callable
    actor_cls = resolve_callable(RslRlActorTactilePreEncodeMLPOptions().class_name)
    assert actor_cls is TactilePreEncodeMLPModel
    rnn_cls = resolve_callable(RslRlActorTactilePreEncodeRNNOptions().class_name)
    assert rnn_cls is TactilePreEncodeRecurrentModel


# ---- two recurrent encoders: tactile ConvRNN + proprio RNN -------------


def _two_recurrent_encoder_cfg(proprio_rnn_type: str = "lstm") -> dict:
    return {
        "tactile_sensors": {
            "kind": "tactile_convrnn",
            "output_dim": 32,
            "shared_per_sensor": False,
            "tactile_layout": dict(LAYOUT),
            "convrnn": {"out_channels": 6, "ksize": 3, "layernorm": True},
        },
        "proprio": {
            "kind": "rnn",
            "rnn_type": proprio_rnn_type,
            "hidden_dim": 24,
            "num_layers": 1,
            "output_dim": 16,
        },
    }


@pytest.mark.parametrize("proprio_rnn_type", ["lstm", "gru"])
def test_two_recurrent_encoders_build_and_inference(proprio_rnn_type: str) -> None:
    """tac_mlp head + tactile_convrnn + proprio rnn — full bundle round-trips."""
    model = _build_mlp_model(_two_recurrent_encoder_cfg(proprio_rnn_type))
    assert model.is_recurrent is True

    # Pre-forward: no state populated yet.
    assert model.get_hidden_state() is None

    obs = _obs(batch=3)
    out = model(obs)
    assert out.shape == (3, ACTION_DIM)

    # Bundle = (conv_state, h, c) for LSTM proprio, (conv_state, h) for GRU proprio.
    state = model.get_hidden_state()
    assert isinstance(state, tuple)
    expected_slots = 1 + (2 if proprio_rnn_type == "lstm" else 1)
    assert len(state) == expected_slots
    # Encoder ordering: by _recurrent_encoder_groups (insertion order in obs_groups_encode).
    # Both tactile_sensors and proprio are recurrent; check at least one tensor per slot.
    for t in state:
        assert isinstance(t, torch.Tensor)
        assert t.dim() == 3  # storage expects 3D (num_layers, num_envs, hidden_dim)
        assert t.shape[1] == 3  # batch (num_envs)


@pytest.mark.parametrize("proprio_rnn_type", ["lstm", "gru"])
def test_two_recurrent_encoders_hidden_state_roundtrip(proprio_rnn_type: str) -> None:
    model = _build_mlp_model(_two_recurrent_encoder_cfg(proprio_rnn_type))
    obs = _obs(batch=4)
    model(obs)
    saved = tuple(t.clone() for t in model.get_hidden_state())

    # Reset everything, then restore.
    model.reset(None)
    assert model.get_hidden_state() is None

    model.set_hidden_state(saved)
    restored = model.get_hidden_state()
    assert isinstance(restored, tuple) and len(restored) == len(saved)
    for r, s in zip(restored, saved):
        assert torch.allclose(r, s)


def test_two_recurrent_encoders_reset_dones_propagates_to_both() -> None:
    model = _build_mlp_model(_two_recurrent_encoder_cfg("lstm"))
    obs = _obs(batch=4)
    model(obs)
    before = tuple(t.clone() for t in model.get_hidden_state())

    dones = torch.tensor([1, 0, 1, 0])
    model.reset(dones)
    after = model.get_hidden_state()
    assert isinstance(after, tuple) and len(after) == 3

    # Every tensor in the bundle has zeros on the done envs.
    for t in after:
        assert t[..., 0, :].abs().sum() == 0
        assert t[..., 2, :].abs().sum() == 0
    # Non-done rows are unchanged.
    for a, b in zip(after, before):
        assert torch.allclose(a[..., 1, :], b[..., 1, :])
        assert torch.allclose(a[..., 3, :], b[..., 3, :])


def test_two_recurrent_encoders_inference_state_persists_across_calls() -> None:
    model = _build_mlp_model(_two_recurrent_encoder_cfg("lstm"))
    obs = _obs(batch=2)
    out0 = model(obs).detach().clone()
    state0 = tuple(t.clone() for t in model.get_hidden_state())
    out1 = model(obs).detach().clone()
    state1 = tuple(t.clone() for t in model.get_hidden_state())

    # Same input twice → both encoder states evolved, outputs differ.
    for a, b in zip(state0, state1):
        assert not torch.allclose(a, b)
    assert not torch.allclose(out0, out1)


def test_two_recurrent_encoders_batched_forward_with_full_masks() -> None:
    """Training-mode forward: padded (T, B) seq + per-encoder init state bundle.

    Bundle order follows ``_recurrent_encoder_groups``, which is the order groups
    appear in ``obs_groups[obs_set]``. Here proprio comes before tactile_sensors,
    so the bundle is ``(h_lstm, c_lstm, conv_state)``.
    """
    model = _build_mlp_model(_two_recurrent_encoder_cfg("lstm"))
    assert model._recurrent_encoder_groups == ["proprio", "tactile_sensors"]
    T_pad, B = 4, 2
    seq_obs = _obs_seq(T_pad, B)
    masks = torch.ones(T_pad, B, dtype=torch.bool)

    conv_state_dim = LAYOUT["num_sensors"] * 6 * LAYOUT["grid_h"] * LAYOUT["grid_w"]
    bundle = (
        torch.zeros(1, B, 24),          # proprio LSTM h (hidden_dim=24)
        torch.zeros(1, B, 24),          # proprio LSTM c
        torch.zeros(1, B, conv_state_dim),  # tactile convrnn state
    )
    out = model(seq_obs, masks=masks, hidden_state=bundle)
    assert out.dim() == 3
    assert out.shape[-1] == ACTION_DIM
    assert out.shape[1] == B


def test_two_recurrent_encoders_batched_forward_gradients_flow() -> None:
    model = _build_mlp_model(_two_recurrent_encoder_cfg("lstm"))
    T_pad, B = 3, 2
    seq_obs = _obs_seq(T_pad, B)
    masks = torch.ones(T_pad, B, dtype=torch.bool)

    out = model(seq_obs, masks=masks, hidden_state=None)
    out.sum().backward()

    # ConvRNN cell weights see gradient.
    conv_cell = model.encoders["tactile_sensors"].cells[0].conv
    assert conv_cell.weight.grad is not None and conv_cell.weight.grad.abs().sum() > 0
    # Proprio LSTM weights see gradient.
    proprio_rnn = model.encoders["proprio"]._rnn.rnn
    for name, p in proprio_rnn.named_parameters():
        assert p.grad is not None, f"proprio rnn param {name} has no grad"
        assert p.grad.abs().sum() > 0, f"proprio rnn param {name} grad is zero"


def test_three_recurrent_encoders_bundle_size_matches_storage_layout() -> None:
    """Two recurrent encoders side-by-side with an MLP encoder in between.
    Ensures _recurrent_encoder_groups ordering + state_slots accounting are correct."""
    # Add a goal encoder (MLP, non-recurrent) between tactile and proprio.
    enc_cfg = {
        "tactile_sensors": {
            "kind": "tactile_convrnn",
            "output_dim": 32,
            "shared_per_sensor": True,
            "tactile_layout": dict(LAYOUT),
            "convrnn": {"out_channels": 4, "ksize": 3, "layernorm": True},
        },
        "goal": {"kind": "mlp", "hidden_dims": [16], "output_dim": 8},
        "proprio": {
            "kind": "rnn", "rnn_type": "lstm", "hidden_dim": 16,
            "num_layers": 1, "output_dim": 12,
        },
    }
    obs = _obs(batch=4)
    model = TactilePreEncodeMLPModel(
        obs=obs, obs_groups={"student": ["tactile_sensors", "goal", "proprio"]},
        obs_set="student", output_dim=ACTION_DIM, hidden_dims=[32],
        activation="elu", obs_normalization=False, distribution_cfg=None,
        encoder_cfg=enc_cfg,
    )
    assert model.is_recurrent is True
    assert model._recurrent_encoder_groups == ["tactile_sensors", "proprio"]
    model(obs)
    state = model.get_hidden_state()
    assert isinstance(state, tuple) and len(state) == 3  # conv + h + c
    # Round-trip the bundle.
    saved = tuple(t.clone() for t in state)
    model.reset(None)
    model.set_hidden_state(saved)
    for r, s in zip(model.get_hidden_state(), saved):
        assert torch.allclose(r, s)


# ---- tac_rnn + tactile_convrnn + proprio rnn ---------------------------


def test_rnn_head_with_two_recurrent_encoders() -> None:
    """tac_rnn (top-level LSTM head) + tactile_convrnn + proprio rnn.

    Bundle layout: (h_top, c_top, conv_state, h_proprio, c_proprio) for LSTM-on-LSTM.
    """
    encoder_cfg = _two_recurrent_encoder_cfg("lstm")
    obs = _obs(batch=3)
    model = TactilePreEncodeRecurrentModel(
        obs=obs, obs_groups=_obs_groups(), obs_set="student",
        output_dim=ACTION_DIM, hidden_dims=[64, 64], activation="elu",
        obs_normalization=False, distribution_cfg=None,
        encoder_cfg=encoder_cfg,
        rnn_type="lstm", rnn_hidden_dim=32, rnn_num_layers=1,
    )
    out = model(obs)
    assert out.shape == (3, ACTION_DIM)

    state = model.get_hidden_state()
    assert isinstance(state, tuple)
    # Top LSTM: 2 slots; tactile_convrnn: 1 slot; proprio LSTM: 2 slots = 5 total.
    assert len(state) == 5

    # Round-trip.
    saved = tuple(t.clone() for t in state)
    model.reset(None)
    model.set_hidden_state(saved)
    restored = model.get_hidden_state()
    assert len(restored) == len(saved)
    for r, s in zip(restored, saved):
        assert torch.allclose(r, s)
