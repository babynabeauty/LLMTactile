"""Unit tests for the per-sensor tactile encoders.

Run with::

    cd dexterous_hands
    uv run pytest tests/models/test_tactile_encoders.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest
import torch

# Make ``src/`` importable when running tests directly via uv run pytest.
_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from models.tactile_encoders import (  # noqa: E402
    CanvasTactileLayout,
    TactileCanvasCNNEncoder,
    TactileCNNEncoder,
    TactileConvGRUEncoder,
    TactileLayout,
)

try:  # pt_tnn is an optional dep for ConvRNN tests only.
    import pt_tnn  # noqa: F401
    from models.tactile_encoders import (
        TactileCanvasConvRNNEncoder,
        TactileConvRNNEncoder,
    )

    _HAVE_PT_TNN = True
except ImportError:
    _HAVE_PT_TNN = False

requires_pt_tnn = pytest.mark.skipif(not _HAVE_PT_TNN, reason="pt_tnn not installed in this venv")


LAYOUT = TactileLayout(
    num_sensors=5,
    grid_hw=((3, 3), (3, 3), (3, 3), (3, 3), (3, 3)),
    features_per_probe=3,
    history_length=1,
)
OUTPUT_DIM = 32


def _rand_obs(batch: int) -> torch.Tensor:
    return torch.randn(batch, LAYOUT.flat_dim)


# ---- canvas-layout fixture --------------------------------------------

# 3 sensors with heterogeneous shapes packed into a 7x9 canvas with gaps,
# matching the structure of the real med-hand layouts (just smaller).
CANVAS_BASE = TactileLayout(
    num_sensors=3,
    grid_hw=((3, 4), (4, 5), (2, 3)),
    features_per_probe=3,
    history_length=1,
)
CANVAS_LAYOUT = CanvasTactileLayout(
    base=CANVAS_BASE,
    canvas_hw=(7, 9),
    placements=((0, 0, 3, 4), (0, 4, 4, 5), (4, 0, 2, 3)),
    canvas_name="test-3sensor",
)


def _canvas_obs(batch: int) -> torch.Tensor:
    return torch.randn(batch, CANVAS_LAYOUT.flat_dim)


# ---- shape tests --------------------------------------------------------


@pytest.mark.parametrize("shared", [False, True])
def test_cnn_encoder_output_shape(shared: bool) -> None:
    enc = TactileCNNEncoder(
        LAYOUT,
        cnn_cfg={"output_channels": [8, 16], "kernel_size": 3, "padding": "zeros"},
        output_dim=OUTPUT_DIM,
        shared_per_sensor=shared,
    )
    x = _rand_obs(batch=4)
    y = enc(x)
    assert y.shape == (4, OUTPUT_DIM)


@requires_pt_tnn
@pytest.mark.parametrize("shared", [False, True])
def test_convrnn_encoder_output_shape(shared: bool) -> None:
    enc = TactileConvRNNEncoder(
        LAYOUT,
        convrnn_cfg={"out_channels": 6, "ksize": 3, "layernorm": True},
        output_dim=OUTPUT_DIM,
        shared_per_sensor=shared,
    )
    x = _rand_obs(batch=4)
    y = enc(x)
    assert y.shape == (4, OUTPUT_DIM)


# ---- per-finger isolation: zero one finger, gradients only flow there ---


def test_cnn_per_finger_weight_isolation() -> None:
    """With per-finger weights and a finger-i-only input, only finger i's CNN weights see gradient.

    This catches sensor-order drift: if the slicer accidentally routed finger j's
    input to CNN j' for some j != j', the wrong CNN's weights would receive
    gradient.
    """
    enc = TactileCNNEncoder(
        LAYOUT,
        cnn_cfg={"output_channels": [4], "kernel_size": 3, "padding": "zeros"},
        output_dim=OUTPUT_DIM,
        shared_per_sensor=False,
    )
    finger_idx = 2
    # Build an input that is nonzero only at finger index 2.
    grid_h, grid_w = LAYOUT.grid_hw[0]  # uniform across sensors in this fixture
    x = torch.zeros(2, LAYOUT.flat_dim)
    grids = x.reshape(2, 1, LAYOUT.num_sensors, LAYOUT.features_per_probe,
                       grid_h, grid_w).clone()
    grids[:, :, finger_idx] = torch.randn_like(grids[:, :, finger_idx])
    x = grids.reshape(2, LAYOUT.flat_dim)

    enc.zero_grad()
    enc(x).sum().backward()

    for s, cnn in enumerate(enc.cnns):
        conv = next(m for m in cnn.modules() if isinstance(m, torch.nn.Conv2d))
        if s == finger_idx:
            assert conv.weight.grad is not None and conv.weight.grad.abs().sum() > 0
        else:
            # Bias may still receive gradient through the projection head; weight
            # gradient w.r.t. a Conv2d with zero input is exactly zero.
            assert conv.weight.grad is None or conv.weight.grad.abs().sum() == 0


# ---- ConvRNN state lifecycle -------------------------------------------


@requires_pt_tnn
def test_convrnn_state_updates_across_steps() -> None:
    enc = TactileConvRNNEncoder(
        LAYOUT,
        convrnn_cfg={"out_channels": 4, "ksize": 3, "layernorm": True},
        output_dim=OUTPUT_DIM,
        shared_per_sensor=True,
    )
    x = _rand_obs(batch=3)
    y0 = enc(x).detach().clone()
    s0 = enc.get_hidden_state().detach().clone()
    # Same input twice -> state evolves, output should differ from first call.
    y1 = enc(x).detach().clone()
    s1 = enc.get_hidden_state().detach().clone()
    assert not torch.allclose(s0, s1)
    assert not torch.allclose(y0, y1)


@requires_pt_tnn
def test_convrnn_reset_zeros_done_rows() -> None:
    enc = TactileConvRNNEncoder(
        LAYOUT,
        convrnn_cfg={"out_channels": 4, "ksize": 3, "layernorm": True},
        output_dim=OUTPUT_DIM,
        shared_per_sensor=True,
    )
    x = _rand_obs(batch=4)
    enc(x)  # warm up state
    # Reset envs 0 and 2 only.
    dones = torch.tensor([1, 0, 1, 0])
    enc.reset(dones)
    state = enc.get_hidden_state()  # shape (1, 4, per_env_state_dim)
    # Rows 0 and 2 are zero.
    assert state[0, 0].abs().sum() == 0
    assert state[0, 2].abs().sum() == 0
    assert state[0, 1].abs().sum() > 0
    assert state[0, 3].abs().sum() > 0

    # Full reset.
    enc.reset(None)
    assert enc.get_hidden_state() is None


@requires_pt_tnn
def test_convrnn_set_hidden_state_roundtrip() -> None:
    enc = TactileConvRNNEncoder(
        LAYOUT,
        convrnn_cfg={"out_channels": 4, "ksize": 3, "layernorm": True},
        output_dim=OUTPUT_DIM,
        shared_per_sensor=True,
    )
    x = _rand_obs(batch=5)
    enc(x)
    state = enc.get_hidden_state().clone()
    # Wipe and restore.
    enc.reset(None)
    enc.set_hidden_state(state)
    assert torch.allclose(enc.get_hidden_state(), state)


# ---- ConvLSTM cell variant --------------------------------------------


@requires_pt_tnn
def test_convrnn_lstm_variant_packed_state_is_2x_wider() -> None:
    """LSTM cell packs (c, h) along the channel axis -> per-env state has 2*C_h channels.

    Same fixture as the Intersection tests so we can check the channel doubling
    directly: per-sensor state shape must be ``(N, 2*C_h, H_s, W_s)`` for LSTM
    and ``(N, C_h, H_s, W_s)`` for Intersection, with packed wire-state sized
    accordingly.
    """
    C_H = 4
    enc_int = TactileConvRNNEncoder(
        LAYOUT,
        convrnn_cfg={"out_channels": C_H, "ksize": 3, "layernorm": True},
        output_dim=OUTPUT_DIM,
    )
    enc_lstm = TactileConvRNNEncoder(
        LAYOUT,
        convrnn_cfg={"out_channels": C_H, "ksize": 3, "layernorm": True, "cell_type": "lstm"},
        output_dim=OUTPUT_DIM,
    )
    x = _rand_obs(batch=3)
    y_int = enc_int(x)
    y_lstm = enc_lstm(x)
    # Output shape is invariant to cell type -- the head reads the cell's visible output.
    assert y_int.shape == y_lstm.shape == (3, OUTPUT_DIM)

    # Per-sensor state channels: 1x for Intersection, 2x for LSTM.
    h, w = LAYOUT.grid_hw[0]
    assert enc_int._state[0].shape == (3, C_H, h, w)
    assert enc_lstm._state[0].shape == (3, 2 * C_H, h, w)

    # Packed wire-state must reflect the doubled per-sensor dim.
    packed_int = enc_int.get_hidden_state()
    packed_lstm = enc_lstm.get_hidden_state()
    assert packed_lstm.shape[-1] == 2 * packed_int.shape[-1]


@requires_pt_tnn
def test_convrnn_lstm_state_lifecycle_roundtrip() -> None:
    """Reset / set_hidden_state / detach all behave for the LSTM cell variant."""
    enc = TactileConvRNNEncoder(
        LAYOUT,
        convrnn_cfg={"out_channels": 4, "ksize": 3, "layernorm": True, "cell_type": "lstm"},
        output_dim=OUTPUT_DIM,
        shared_per_sensor=True,
    )
    x = _rand_obs(batch=4)
    y0 = enc(x).detach().clone()
    s0 = enc.get_hidden_state().detach().clone()
    y1 = enc(x).detach().clone()
    s1 = enc.get_hidden_state().detach().clone()
    assert not torch.allclose(s0, s1), "LSTM state should evolve across steps"
    assert not torch.allclose(y0, y1), "Same input twice should produce different outputs once state moves"

    # Partial reset on envs 0 and 2 only.
    enc.reset(torch.tensor([1, 0, 1, 0]))
    state = enc.get_hidden_state()
    assert state[0, 0].abs().sum() == 0
    assert state[0, 2].abs().sum() == 0
    assert state[0, 1].abs().sum() > 0

    # Roundtrip via the wire-state API.
    snap = enc.get_hidden_state().clone()
    enc.reset(None)
    enc.set_hidden_state(snap)
    assert torch.allclose(enc.get_hidden_state(), snap)


@requires_pt_tnn
@pytest.mark.parametrize("cell_type", ["intersection", "lstm"])
def test_convrnn_batch_mode_leaves_encoder_state_untouched(cell_type: str) -> None:
    """A minibatch replay must not overwrite the rollout state.

    ``set_hidden_state`` unpacks into ``self._state``, so calling it from the batch
    path left the last minibatch's state in the buffer; the next rollout step then
    resumed from an unrelated minibatch (or, on a batch-size mismatch, silently
    restarted from zeros).
    """
    enc = TactileConvRNNEncoder(
        LAYOUT,
        convrnn_cfg={"out_channels": 4, "ksize": 3, "layernorm": True, "cell_type": cell_type},
        output_dim=OUTPUT_DIM,
    )
    enc(_rand_obs(batch=3))
    before = enc.get_hidden_state().clone()

    seq, batch = 4, 6  # deliberately a different batch size than the live state
    enc(
        torch.randn(seq, batch, LAYOUT.flat_dim),
        masks=torch.ones(seq, batch, dtype=torch.bool),
        hidden_state=torch.randn(1, batch, before.shape[-1]),
    )
    assert torch.allclose(enc.get_hidden_state(), before)


@requires_pt_tnn
def test_convrnn_lstm_rejects_unknown_cell_type() -> None:
    with pytest.raises(ValueError, match="Unknown convrnn cell_type"):
        TactileConvRNNEncoder(
            LAYOUT,
            convrnn_cfg={"out_channels": 4, "ksize": 3, "layernorm": True, "cell_type": "not-a-cell"},
            output_dim=OUTPUT_DIM,
        )


# ---- ConvGRU ----------------------------------------------------------

# Heterogeneous grid shapes with two sensors sharing 3x3, so the grouped-conv
# path runs with a group of 2 and two groups of 1.
GRU_LAYOUT = TactileLayout(
    num_sensors=4, grid_hw=((3, 3), (3, 3), (2, 4), (5, 4)), features_per_probe=3, history_length=1
)
GRU_CFG = {"out_channels": 4, "ksize": 3, "layernorm": True}


def _gru_enc(**overrides: Any) -> TactileConvGRUEncoder:
    cfg = GRU_CFG | overrides
    return TactileConvGRUEncoder(GRU_LAYOUT, convrnn_cfg=cfg, output_dim=OUTPUT_DIM)


def _gru_obs(batch: int) -> torch.Tensor:
    return torch.randn(batch, GRU_LAYOUT.flat_dim)


def test_convgru_output_shape_and_packed_state_width() -> None:
    enc = _gru_enc()
    y = enc(_gru_obs(batch=4))
    assert y.shape == (4, OUTPUT_DIM)
    expected = sum(GRU_CFG["out_channels"] * h * w for h, w in GRU_LAYOUT.grid_hw)
    assert enc.get_hidden_state().shape == (1, 4, expected)


def test_convgru_state_updates_across_steps() -> None:
    enc = _gru_enc()
    x = _gru_obs(batch=3)
    y0 = enc(x).detach().clone()
    s0 = enc.get_hidden_state().detach().clone()
    y1 = enc(x).detach().clone()
    assert not torch.allclose(s0, enc.get_hidden_state())
    assert not torch.allclose(y0, y1)


def test_convgru_output_depends_on_history() -> None:
    """The output must carry state, not just re-encode the current frame.

    ``IntersectionRNNCell`` returns ``gy * input + (1 - gy) * y``, so its output is
    dominated by the current frame; a ConvGRU returns the hidden state itself. This
    pins that difference: the same frame must encode differently depending on what
    preceded it.
    """
    enc = _gru_enc().eval()
    a, b = _gru_obs(batch=8), _gru_obs(batch=8)
    with torch.no_grad():
        enc.reset(None)
        enc(a)
        after_a = enc(b).clone()
        enc.reset(None)
        fresh = enc(b).clone()
    assert (after_a - fresh).norm() / fresh.norm() > 0.1


def test_convgru_reset_zeros_done_rows() -> None:
    enc = _gru_enc()
    enc(_gru_obs(batch=4))
    enc.reset(torch.tensor([1, 0, 1, 0]))
    state = enc.get_hidden_state()
    assert state[0, 0].abs().sum() == 0
    assert state[0, 2].abs().sum() == 0
    assert state[0, 1].abs().sum() > 0
    assert state[0, 3].abs().sum() > 0

    enc.reset(None)
    assert enc.get_hidden_state() is None


def test_convgru_set_hidden_state_roundtrip() -> None:
    enc = _gru_enc()
    enc(_gru_obs(batch=5))
    state = enc.get_hidden_state().clone()
    enc.reset(None)
    enc.set_hidden_state(state)
    assert torch.allclose(enc.get_hidden_state(), state)


def test_convgru_batch_mode_matches_stepwise_rollout() -> None:
    """The padded (T, N, D) training path must reproduce the per-step inference path.

    The recurrent PPO/distillation update replays trajectories through the batch path
    while rollout collection uses the step path, so a mismatch trains the encoder on
    states it never sees at inference.
    """
    enc = _gru_enc().eval()
    seq, batch = 5, 3
    x = torch.randn(seq, batch, GRU_LAYOUT.flat_dim)
    masks = torch.ones(seq, batch, dtype=torch.bool)
    hidden = torch.zeros(1, batch, sum(4 * h * w for h, w in GRU_LAYOUT.grid_hw))

    with torch.no_grad():
        batched = enc(x, masks=masks, hidden_state=hidden)
        enc.set_hidden_state(hidden)
        stepwise = torch.stack([enc(x[t]) for t in range(seq)])

    assert batched.shape == stepwise.shape
    assert torch.allclose(batched, stepwise, atol=1e-5)


def test_convgru_batch_mode_leaves_encoder_state_untouched() -> None:
    """A minibatch replay must not clobber the rollout state.

    The update path replays arbitrary minibatches; if their hidden state leaked into
    the encoder buffers, the next rollout step would resume from an unrelated env's
    state. This was a real crash in the recurrent PPO path.
    """
    enc = _gru_enc()
    enc(_gru_obs(batch=3))
    before = enc.get_hidden_state().clone()

    seq, batch = 4, 6  # deliberately a different batch size than the live state
    enc(
        torch.randn(seq, batch, GRU_LAYOUT.flat_dim),
        masks=torch.ones(seq, batch, dtype=torch.bool),
        hidden_state=torch.randn(1, batch, sum(4 * h * w for h, w in GRU_LAYOUT.grid_hw)),
    )
    assert torch.allclose(enc.get_hidden_state(), before)


def test_convgru_per_sensor_weight_isolation() -> None:
    """Sensors sharing a grid shape share one grouped conv but must not share weights.

    Two sensors land in the same 3x3 group; driving only one of them must leave the
    other's slice of the grouped kernel at zero gradient. A wrong `groups=` or a
    channel interleave that crosses sensors would show up here.
    """
    enc = _gru_enc()
    group_idx = next(i for i, m in enumerate(enc._group_members) if len(m) == 2)
    members = enc._group_members[group_idx]

    x = torch.zeros(2, GRU_LAYOUT.flat_dim)
    per_sensor = list(torch.split(x, [3 * h * w for h, w in GRU_LAYOUT.grid_hw], dim=1))
    per_sensor[members[0]] = torch.randn_like(per_sensor[members[0]])
    enc.zero_grad()
    enc(torch.cat(per_sensor, dim=1)).sum().backward()

    grad = enc.cand_convs[group_idx].weight.grad
    out_per_sensor = grad.shape[0] // len(members)
    driven, quiet = grad[:out_per_sensor], grad[out_per_sensor:]
    assert driven.abs().sum() > 0
    assert quiet.abs().sum() == 0


def test_convgru_backward_through_rollout_with_resets() -> None:
    """Distillation backprops through a rollout that calls reset/detach mid-sequence.

    Because the hidden state *is* the output, ``_project`` holds a view of the state
    while it is still in the graph. Zeroing done rows in place bumped its version and
    made ``loss.backward()`` raise "modified by an inplace operation".
    """
    enc = _gru_enc()
    dones = torch.tensor([0, 1, 0])
    loss = torch.zeros((), requires_grad=True)
    for step in range(4):
        loss = loss + enc(_gru_obs(batch=3)).square().sum()
        if step == 1:
            enc.reset(dones)
            enc.detach_hidden_state(dones)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in enc.parameters())


def test_convgru_reset_survives_backward_and_still_zeroes() -> None:
    """The out-of-place reset must still actually zero the done rows."""
    enc = _gru_enc()
    enc(_gru_obs(batch=4))
    enc.reset(torch.tensor([1, 0, 1, 0]))
    state = enc.get_hidden_state()
    assert state[0, 0].abs().sum() == 0
    assert state[0, 2].abs().sum() == 0
    assert state[0, 1].abs().sum() > 0


def test_convgru_rejects_stacked_history() -> None:
    stacked = TactileLayout(num_sensors=2, grid_hw=((3, 3), (3, 3)), features_per_probe=3, history_length=4)
    with pytest.raises(ValueError, match="history_length == 1"):
        TactileConvGRUEncoder(stacked, convrnn_cfg=dict(GRU_CFG), output_dim=OUTPUT_DIM)


def test_convgru_rejects_unknown_cfg_keys() -> None:
    with pytest.raises(ValueError, match="Unknown keys"):
        _gru_enc(nonsense=1)


# ---- canvas encoders --------------------------------------------------


def _scatter_reference(canvas_layout: CanvasTactileLayout, per_sensor_frames: list[torch.Tensor]) -> torch.Tensor:
    """Reference scatter used for the round-trip exactness test."""
    N = per_sensor_frames[0].shape[0]
    F = per_sensor_frames[0].shape[1]
    H, W = canvas_layout.canvas_hw
    canvas = torch.zeros(N, F, H, W)
    for frame, (r, c, h, w) in zip(per_sensor_frames, canvas_layout.placements, strict=True):
        canvas[:, :, r:r + h, c:c + w] = frame
    return canvas


def test_canvas_layout_rejects_overlap() -> None:
    with pytest.raises(ValueError, match="overlaps"):
        CanvasTactileLayout(
            base=CANVAS_BASE,
            canvas_hw=(5, 9),
            placements=((0, 0, 3, 4), (0, 3, 4, 5), (3, 0, 2, 3)),  # 2nd overlaps 1st
        )


def test_canvas_layout_rejects_out_of_bounds() -> None:
    with pytest.raises(ValueError, match="out of canvas"):
        CanvasTactileLayout(
            base=CANVAS_BASE,
            canvas_hw=(5, 5),
            placements=((0, 0, 3, 4), (0, 4, 4, 5), (4, 0, 2, 3)),
        )


def test_canvas_layout_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="does not match base grid"):
        CanvasTactileLayout(
            base=CANVAS_BASE,
            canvas_hw=(7, 9),
            placements=((0, 0, 3, 4), (0, 4, 4, 5), (4, 0, 3, 3)),  # 3rd should be 2x3 or 3x2
        )


def test_canvas_layout_accepts_transposed_placements() -> None:
    """Placements may declare the patch transposed (e.g. to orient fingers vertically)."""
    layout = CanvasTactileLayout(
        base=CANVAS_BASE,
        canvas_hw=(9, 8),
        # All three sensors transposed: probe (3,4)->(4,3), (4,5)->(5,4), (2,3)->(3,2).
        placements=((0, 0, 4, 3), (0, 3, 5, 4), (5, 0, 3, 2)),
    )
    assert layout.transposed == (True, True, True)


def test_repack_active_placements_drops_unused_rows_and_cols() -> None:
    """Repacking should remove rows/cols that no active sensor occupies, shifting
    the remaining sensors into the dense layout while preserving their sizes."""
    sys_path_backup = sys.path[:]
    try:
        if str(_SRC) not in sys.path:
            sys.path.insert(0, str(_SRC))
        from task_mods import _repack_active_placements  # noqa: WPS433

        # Med-hand palm-rotate active subset, exact placements from canvas_xhand1.json.
        placements = (
            (0, 0, 5, 4),    # thumb_link2
            (0, 8, 5, 4),    # mid_link2
            (5, 0, 4, 4),    # thumb_link1
            (5, 8, 4, 3),    # mid_link1
            (9, 6, 5, 7),    # palm
        )
        new_p, new_hw = _repack_active_placements(placements, (14, 20))
        # Live rows = 0..13 (palm reaches row 13). Live cols = 0..3, 6..12.
        # Total live: 14 rows, 11 cols.
        assert new_hw == (14, 11)
        # Sensor sizes preserved.
        for old, new in zip(placements, new_p, strict=True):
            assert old[2:] == new[2:]
        # No-op rows (palm reaches the bottom), col mapping drops cols 4,5,13..19.
        # Col map: 0..3 stay; cols 4,5 dead; cols 6..12 -> 4..10.
        assert new_p[0] == (0, 0, 5, 4)   # thumb_link2 unchanged
        assert new_p[1] == (0, 6, 5, 4)   # mid_link2: col 8 -> col 6
        assert new_p[4] == (9, 4, 5, 7)   # palm: col 6 -> col 4
    finally:
        sys.path[:] = sys_path_backup


def test_canvas_matcher_accepts_subset_of_active_sensors() -> None:
    """A canvas with extra (frozen) sensors should still match when the active
    subset is contained in it. Placements come out in active-iteration order."""
    sys_path_backup = sys.path[:]
    try:
        if str(_SRC) not in sys.path:
            sys.path.insert(0, str(_SRC))
        from task_mods import _canvas_matches  # noqa: WPS433 - local import inside test

        canvas = {
            "name": "test-fivesensor",
            "canvas_hw": [10, 10],
            "sensors": [
                {"link": "sA", "row": 0, "col": 0, "h": 3, "w": 4},
                {"link": "sB", "row": 0, "col": 5, "h": 4, "w": 5},
                {"link": "sC", "row": 5, "col": 0, "h": 2, "w": 3},
                {"link": "sD", "row": 5, "col": 5, "h": 3, "w": 3},  # extra / frozen
                {"link": "sE", "row": 8, "col": 0, "h": 2, "w": 2},  # extra / frozen
            ],
        }
        active = ["sB", "sA", "sC"]
        grid_hw = ((4, 5), (3, 4), (2, 3))
        assert _canvas_matches(canvas, active, grid_hw)

        # Active link not on the canvas -> no match.
        assert not _canvas_matches(canvas, ["sA", "sZ"], ((3, 4), (1, 1)))

        # Active grid that doesn't match the canvas patch (nor its transpose) -> no match.
        assert not _canvas_matches(canvas, ["sA"], ((5, 5),))
    finally:
        sys.path[:] = sys_path_backup


def test_canvas_scatter_applies_transpose() -> None:
    """When a placement is transposed, the scatter rotates the frame to match."""
    layout = CanvasTactileLayout(
        base=CANVAS_BASE,
        canvas_hw=(9, 8),
        placements=((0, 0, 4, 3), (0, 3, 5, 4), (5, 0, 3, 2)),
    )
    # Build per-sensor frames with row/col indices baked in as features so we can
    # check exact placement after transpose.
    frames = []
    for i, (h, w) in enumerate(CANVAS_BASE.grid_hw):
        F_ = CANVAS_BASE.features_per_probe
        f = torch.arange(h * w, dtype=torch.float32).reshape(1, 1, h, w).expand(1, F_, h, w).contiguous()
        frames.append(f)
    canvas = layout.scatter_frames(frames)
    # Sensor 0: probe (3,4) -> canvas (4,3) at (0,0). Patch on canvas is
    # the input transposed on the last two dims.
    expected0 = frames[0].transpose(-1, -2)
    assert torch.equal(canvas[:, :, 0:4, 0:3], expected0)
    # Padding stays zero.
    mask = layout.build_presence_mask()
    pad = (mask == 0).expand_as(canvas[0])
    assert canvas[0][pad].abs().sum().item() == 0


def test_canvas_presence_mask_counts_real_cells() -> None:
    mask = CANVAS_LAYOUT.build_presence_mask()
    expected = sum(h * w for _, _, h, w in CANVAS_LAYOUT.placements)
    assert mask.shape == (1, 7, 9)
    assert mask.sum().item() == float(expected)


def test_canvas_cnn_output_shape() -> None:
    enc = TactileCanvasCNNEncoder(
        CANVAS_LAYOUT,
        cnn_cfg={"output_channels": [8, 16], "kernel_size": 3, "padding": "zeros"},
        output_dim=OUTPUT_DIM,
    )
    y = enc(_canvas_obs(batch=4))
    assert y.shape == (4, OUTPUT_DIM)


def test_canvas_scatter_is_exact_round_trip() -> None:
    """Per-sensor patches arrive at their declared canvas placements unchanged.

    Builds a flat obs whose per-sensor blocks are filled with sensor-specific
    constants, then asserts the encoder's internal scatter places those constants
    at the right canvas cells (and zeros everywhere else).
    """
    from models.tactile_encoders import _scatter_to_canvas

    per_sensor_grids = []
    for i, (h, w) in enumerate(CANVAS_BASE.grid_hw):
        F = CANVAS_BASE.features_per_probe
        per_sensor_grids.append(torch.full((2, F, h, w), float(i + 1)))

    canvas = _scatter_to_canvas(per_sensor_grids, CANVAS_LAYOUT.placements, CANVAS_LAYOUT.canvas_hw)
    ref = _scatter_reference(CANVAS_LAYOUT, per_sensor_grids)
    assert torch.equal(canvas, ref)
    # And padding cells stay zero everywhere.
    mask = CANVAS_LAYOUT.build_presence_mask()  # (1, H, W)
    padding_mask = (mask == 0).expand_as(canvas[0])  # (F, H, W)
    assert canvas[0][padding_mask].abs().sum().item() == 0
    assert canvas[1][padding_mask].abs().sum().item() == 0


@requires_pt_tnn
def test_canvas_convrnn_output_shape() -> None:
    enc = TactileCanvasConvRNNEncoder(
        CANVAS_LAYOUT,
        convrnn_cfg={"out_channels": 6, "ksize": 3, "layernorm": True},
        output_dim=OUTPUT_DIM,
    )
    y = enc(_canvas_obs(batch=4))
    assert y.shape == (4, OUTPUT_DIM)


@requires_pt_tnn
def test_canvas_convrnn_state_updates_across_steps() -> None:
    enc = TactileCanvasConvRNNEncoder(
        CANVAS_LAYOUT,
        convrnn_cfg={"out_channels": 6, "ksize": 3, "layernorm": True},
        output_dim=OUTPUT_DIM,
    )
    x = _canvas_obs(batch=3)
    y0 = enc(x).detach().clone()
    s0 = enc.get_hidden_state().detach().clone()
    y1 = enc(x).detach().clone()
    s1 = enc.get_hidden_state().detach().clone()
    assert not torch.allclose(s0, s1)
    assert not torch.allclose(y0, y1)


@requires_pt_tnn
def test_canvas_convrnn_reset_zeros_done_rows() -> None:
    enc = TactileCanvasConvRNNEncoder(
        CANVAS_LAYOUT,
        convrnn_cfg={"out_channels": 6, "ksize": 3, "layernorm": True},
        output_dim=OUTPUT_DIM,
    )
    enc(_canvas_obs(batch=4))
    dones = torch.tensor([1, 0, 1, 0])
    enc.reset(dones)
    state = enc.get_hidden_state()
    assert state[0, 0].abs().sum() == 0
    assert state[0, 2].abs().sum() == 0
    assert state[0, 1].abs().sum() > 0
    assert state[0, 3].abs().sum() > 0
    enc.reset(None)
    assert enc.get_hidden_state() is None


@requires_pt_tnn
def test_canvas_convrnn_set_hidden_state_roundtrip() -> None:
    enc = TactileCanvasConvRNNEncoder(
        CANVAS_LAYOUT,
        convrnn_cfg={"out_channels": 6, "ksize": 3, "layernorm": True},
        output_dim=OUTPUT_DIM,
    )
    enc(_canvas_obs(batch=5))
    state = enc.get_hidden_state().clone()
    enc.reset(None)
    enc.set_hidden_state(state)
    assert torch.allclose(enc.get_hidden_state(), state)


@requires_pt_tnn
def test_canvas_convrnn_batch_mode_shape_and_unpad() -> None:
    enc = TactileCanvasConvRNNEncoder(
        CANVAS_LAYOUT,
        convrnn_cfg={"out_channels": 6, "ksize": 3, "layernorm": True},
        output_dim=OUTPUT_DIM,
    )
    T, B = 5, 3
    x = torch.randn(T, B, CANVAS_LAYOUT.flat_dim)
    masks = torch.ones(T, B, dtype=torch.bool)
    y = enc(x, masks=masks)
    # All steps valid -> output retains full T x B shape.
    assert y.shape == (T, B, OUTPUT_DIM)


@requires_pt_tnn
def test_canvas_convrnn_batch_mode_leaves_encoder_state_untouched() -> None:
    """Same rollout-state guard as the per-sensor ConvRNN; see that test for why."""
    enc = TactileCanvasConvRNNEncoder(
        CANVAS_LAYOUT, convrnn_cfg={"out_channels": 6, "ksize": 3, "layernorm": True}, output_dim=OUTPUT_DIM
    )
    enc(_canvas_obs(batch=3))
    before = enc.get_hidden_state().clone()

    seq, batch = 4, 6
    enc(
        torch.randn(seq, batch, CANVAS_LAYOUT.flat_dim),
        masks=torch.ones(seq, batch, dtype=torch.bool),
        hidden_state=torch.randn(1, batch, before.shape[-1]),
    )
    assert torch.allclose(enc.get_hidden_state(), before)
