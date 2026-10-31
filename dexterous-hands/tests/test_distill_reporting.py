"""Regression tests for the static and interactive distillation learning curves."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_ablation_widget import payload, require_complete_matrix  # noqa: E402
from plot_distill_ablations import TYPE_LABEL, draw_types, require_complete_matrix as require_complete_plot  # noqa: E402


def run(modality, series, *, last_iteration=2999, imperfection="clean",
        task="in_hand_repose", robot="wuji"):
    return {
        "task": task,
        "robot": robot,
        "modality": modality,
        "imperfection": imperfection,
        "last_iteration": last_iteration,
        "series": {"clean": series},
    }


def test_widget_payload_filters_partial_retries_and_keeps_curves():
    results = {
        "runs": [
            run("force_torque", [[500, 0.2], [2999, 0.8]]),
            run("none", [[500, 0.1], [2999, 0.3]]),
            run("elastomer", [[500, 0.4]], last_iteration=None),
        ]
    }
    data = payload(results, {"in_hand_repose-wuji": 0.9}, final_iteration=2999)
    assert len(data["pairs"]) == 1
    pair = data["pairs"][0]
    assert pair["types"] == [
        {"name": "force_torque", "final": 0.8, "series": [[500, 0.2], [2999, 0.8]]}
    ]
    assert pair["baselineSeries"] == [[500, 0.1], [2999, 0.3]]
    assert pair["teacher"] == 0.9


def test_static_type_view_draws_iteration_series_not_final_value_dots():
    fig, ax = plt.subplots()
    try:
        subset = [
            run("force_torque", [[500, 0.2], [1000, 0.5], [2999, 0.8]]),
            run("none", [[500, 0.1], [1000, 0.2], [2999, 0.3]]),
        ]
        draw_types(ax, subset, teacher=0.9, task="in_hand_repose", first=True)
        series = {line.get_label(): list(line.get_xdata()) for line in ax.lines}
        assert series[TYPE_LABEL["force_torque"]] == [500, 1000, 2999]
        assert series["no tactile"] == [500, 1000, 2999]
        rule = [line for line in ax.lines if list(line.get_xdata()) == [0, 1]]
        assert [line.get_ydata()[0] for line in rule] == [0.9]
        assert [text.get_text() for text in ax.texts] == ["teacher"]
        assert ax.get_xlabel() == "Distillation iteration"
    finally:
        plt.close(fig)


def test_publication_widget_rejects_incomplete_matrix():
    results = {
        "runs": [
            run("force_torque", [[500, 0.2], [2999, 0.8]]),
            run("none", [[500, 0.1], [2999, 0.3]]),
        ]
    }
    try:
        require_complete_matrix(results, 2999)
    except ValueError as exc:
        assert "publication matrix is incomplete" in str(exc)
        assert "in_hand_repose-wuji/elastomer/clean" in str(exc)
    else:
        raise AssertionError("incomplete publication matrix was accepted")


def test_publication_plot_rejects_incomplete_matrix():
    runs = [
        run("force_torque", [[500, 0.2], [2999, 0.8]]),
        run("none", [[500, 0.1], [2999, 0.3]]),
    ]
    try:
        require_complete_plot(runs, 2999)
    except ValueError as exc:
        assert "publication matrix is incomplete" in str(exc)
        assert "in_hand_repose-wuji/elastomer/clean" in str(exc)
    else:
        raise AssertionError("incomplete publication matrix was accepted")


def test_publication_gates_require_all_iteration_points():
    results = {
        "runs": [
            run("force_torque", [[500, 0.2], [2999, 0.8]]),
            run("none", [[500, 0.1], [2999, 0.3]]),
        ]
    }
    for gate, value in (
        (require_complete_matrix, results),
        (require_complete_plot, results["runs"]),
    ):
        try:
            gate(value, 2999)
        except ValueError as exc:
            assert "bad-series=" in str(exc)
            assert "[500, 2999]" in str(exc)
        else:
            raise AssertionError("publication matrix with truncated curves was accepted")


def test_publication_gates_accept_the_exact_cell_matrix():
    points = [[iteration, 0.5] for iteration in [500, 1000, 1500, 2000, 2500, 2999]]
    pairs = [
        ("in_hand_repose", "wuji"),
        ("in_fingers_rotate", "sharpa"),
        ("grasp_lift", "allegro"),
    ]
    from plot_distill_ablations import TYPE_ORDER

    modalities = ["none", *TYPE_ORDER]
    runs = [
        run(modality, points, task=task, robot=robot)
        for task, robot in pairs
        for modality in modalities
    ]
    runs.extend(
        run("force_torque", points, imperfection=level, task=task, robot=robot)
        for task, robot in pairs
        for level in ("mild", "strong")
    )
    assert len(runs) == len(pairs) * (len(modalities) + 2)
    require_complete_matrix({"runs": runs}, 2999)
    require_complete_plot(runs, 2999)


def complete_matrix(final_iteration):
    """One run per publication cell, scored on the full ladder up to ``final_iteration``."""
    from plot_distill_ablations import checkpoints_through

    ladder = checkpoints_through(final_iteration)
    pairs = [("in_hand_repose", "wuji"), ("in_fingers_rotate", "sharpa"), ("grasp_lift", "allegro")]
    from plot_distill_ablations import TYPE_ORDER

    types = list(TYPE_ORDER)
    runs = []
    for task, robot in pairs:
        for modality in ["none", *types]:
            runs.append(run(modality, [[i, 0.5] for i in ladder], task=task, robot=robot))
        for level in ("mild", "strong"):
            runs.append(run("force_torque", [[i, 0.5] for i in ladder],
                            imperfection=level, task=task, robot=robot))
    return {"runs": runs}


def test_gate_accepts_a_complete_matrix_at_2500():
    # Every cell reaches 2500 well before the 3000-update budget closes, so the ladder that
    # ends there is a publishable matrix in its own right.
    require_complete_matrix(complete_matrix(2500), 2500)
    require_complete_plot(complete_matrix(2500)["runs"], 2500)


def test_gate_at_2500_ignores_points_beyond_it():
    # A cell that ran on to 2999 is still judged at 2500 rather than rejected for having more.
    data = complete_matrix(2500)
    data["runs"][0]["series"]["clean"].append([2999, 0.7])
    require_complete_matrix(data, 2500)


def test_gate_at_2500_still_rejects_a_missing_cell():
    data = complete_matrix(2500)
    data["runs"] = [r for r in data["runs"] if r["modality"] != "elastomer"]
    try:
        require_complete_matrix(data, 2500)
    except ValueError as exc:
        assert "elastomer" in str(exc)
    else:
        raise AssertionError("expected the incomplete 2500 matrix to be rejected")


def test_widget_truncates_curves_to_the_judged_checkpoint():
    from build_ablation_widget import payload

    data = complete_matrix(2500)
    data["runs"][0]["series"]["clean"].append([2999, 0.9])
    out = payload(data, {}, final_iteration=2500)
    for pair in out["pairs"]:
        for series in pair["types"] + pair["levels"]:
            assert max(point[0] for point in series["series"]) == 2500
