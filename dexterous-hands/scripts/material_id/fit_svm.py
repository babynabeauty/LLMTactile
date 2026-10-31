"""Classify the plate material from one fingertip slide, force against temperature.

Fits an RBF SVM on three feature sets -- force only, temperature only, and both --
and reports cross-validated accuracy plus a confusion matrix for each. The point of
the ablation is the confusion structure: friction cannot separate materials that grip
alike, conduction cannot separate materials that a silicone fingertip feels as equally
cold, and the pairs the two channels confuse are not the same pairs.

Usage:
    python scripts/material_id/fit_svm.py --data data/material_id/slides.npz
    python scripts/material_id/fit_svm.py --data data/material_id/slides.npz --figure figures/material_id.pdf
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parent))
import features  # noqa: E402

N_FOLDS = 5
SVM_C = 10.0

CHANNELS = ("force", "temperature", "force + temperature")


def build_classifier() -> object:
    return make_pipeline(StandardScaler(), SVC(kernel="rbf", C=SVM_C, gamma="scale"))


def evaluate(x: np.ndarray, y: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Cross-validated fold accuracies and out-of-fold predictions."""
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    scores = cross_val_score(build_classifier(), x, y, cv=cv)
    predictions = cross_val_predict(build_classifier(), x, y, cv=cv)
    return scores, predictions


def thermal_groups(data: dict, tolerance: float = 0.15) -> list[list[int]]:
    """Materials a silicone fingertip cannot tell apart thermally, grouped.

    Heat crosses the contact at the series conductivity of skin and plate, so the
    skin caps how different two plates can feel: every metal saturates at twice the
    skin's conductivity. Materials within ``tolerance`` of each other in that series
    value are grouped.
    """
    k_skin = float(data["fingertip_conductivity"])
    k_plate = np.asarray(data["conductivity"], dtype=np.float64)
    k_series = 2 * k_skin * k_plate / (k_skin + k_plate)
    order = np.argsort(k_series)
    groups: list[list[int]] = []
    for idx in order:
        if groups and k_series[idx] <= k_series[groups[-1][-1]] * (1 + tolerance):
            groups[-1].append(int(idx))
        else:
            groups.append([int(idx)])
    return [g for g in groups if len(g) > 1]


def report_cue_independence(x_temp: np.ndarray, y: np.ndarray, names, data: dict, seed: int) -> None:
    """Check the temperature channel carries no friction information.

    Within a thermally indistinguishable group the members still differ in friction,
    often widely, so a temperature-only classifier that beats chance there would be
    reading friction through some back door (contact penetration, say) rather than
    conduction. Chance is what we want to see.
    """
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    predictions = cross_val_predict(build_classifier(), x_temp, y, cv=cv)
    print("\n[svm] temperature-only accuracy inside thermally matched groups (chance is the target):")
    for group in thermal_groups(data):
        selected = np.isin(y, group)
        accuracy = float((predictions[selected] == y[selected]).mean())
        members = ", ".join(f"{names[i]} (mu={data['friction'][i]:.2f})" for i in group)
        print(f"    {accuracy:.2f} vs {1 / len(group):.2f}   {members}")


def plot(matrices: dict[str, np.ndarray], names, x_force, x_temp, y, path: Path) -> None:
    """Confusion matrices per channel, plus the two-cue feature space they come from."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "plot"))
    try:
        import style  # noqa: F401  -- registers the paper font
    except ImportError:
        pass

    # The matrices are square (imshow keeps its aspect), so their slots are sized to
    # the box rather than left to spread; the scatter takes the width they give up.
    fig, axes = plt.subplots(1, 4, figsize=(11.5, 3.5), gridspec_kw={"width_ratios": [1, 1, 1, 1.55]})
    for ax, (channel, matrix) in zip(axes, matrices.items()):
        normalized = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
        ax.imshow(normalized, cmap="viridis", vmin=0.0, vmax=1.0)
        # Material names stay lowercase: they are data values, not headings. Only the
        # leftmost panel carries the row names; the three matrices share an axis.
        ax.set_xticks(range(len(names)), names, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(names)), names if ax is axes[0] else [""] * len(names), fontsize=8)
        ax.tick_params(pad=1.5)
        ax.set_title(f"{channel.title()}\n{np.trace(matrix) / matrix.sum():.1%} Accuracy", fontsize=10)
        ax.set_xlabel("Predicted", fontsize=9, labelpad=2)
        if ax is axes[0]:
            ax.set_ylabel("True", fontsize=9, labelpad=2)
        for i in range(len(names)):
            for j in range(len(names)):
                if normalized[i, j] > 0.01:
                    ax.text(
                        j,
                        i,
                        f"{normalized[i, j]:.2f}".lstrip("0"),
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="white" if normalized[i, j] < 0.6 else "black",
                    )

    ax = axes[3]
    mu = x_force[:, features.FORCE_FEATURE_NAMES.index("mu_mean")]
    # The feature is a drop; negate it so the axis reads as a signed change.
    change = -x_temp[:, features.TEMP_FEATURE_NAMES.index("dT_total")]
    colors = plt.get_cmap("tab10")(np.linspace(0, 1, 10))
    for i, name in enumerate(names):
        sel = y == i
        ax.scatter(mu[sel], change[sel], s=12, color=colors[i % 10], label=name, alpha=0.75, linewidths=0)
    ax.set_xlabel(r"Friction Estimate $\hat{\mu} = f_t / f_n$", fontsize=9, labelpad=2)
    ax.set_ylabel("Fingertip Temperature Change Over Trial (Δ°C)", fontsize=9, labelpad=2)
    ax.set_title("Material Signatures in Friction and Temperature", fontsize=10)
    ax.legend(
        fontsize=7,
        ncols=2,
        loc="lower right",
        borderpad=0.3,
        labelspacing=0.25,
        handletextpad=0.2,
        handlelength=1.0,
        columnspacing=0.6,
        framealpha=0.95,
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.margins(0.02)
    ax.tick_params(labelsize=8, pad=1.5)

    fig.tight_layout(pad=0.3, w_pad=0.2)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"[svm] wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=str, default="data/material_id/slides.npz")
    parser.add_argument("--figure", type=str, default="figures/material_id/material_id.pdf")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    data = dict(np.load(args.data, allow_pickle=True))
    x_force, x_temp, y = features.extract(data)
    names = [str(n) for n in data["material_names"]]
    x_both = np.concatenate([x_force, x_temp], axis=1)

    print(f"[svm] {len(y)} trials, {len(names)} materials, chance = {1 / len(names):.1%}")
    matrices = {}
    for channel, x in zip(CHANNELS, (x_force, x_temp, x_both)):
        scores, predictions = evaluate(x, y, args.seed)
        matrices[channel] = confusion_matrix(y, predictions, labels=range(len(names)))
        print(f"  {channel:22s} {x.shape[1]:2d} features   {scores.mean():.3f} +/- {scores.std():.3f}")

    report_cue_independence(x_temp, y, names, data, args.seed)

    print("\n[svm] worst confusions with both channels:")
    matrix = matrices[CHANNELS[2]]
    off_diagonal = matrix - np.diag(np.diag(matrix))
    for i, j in zip(*np.unravel_index(np.argsort(off_diagonal, axis=None)[::-1][:5], matrix.shape)):
        if off_diagonal[i, j]:
            print(f"    {names[i]:10s} -> {names[j]:10s} {off_diagonal[i, j]:3d} / {matrix[i].sum()}")

    plot(matrices, names, x_force, x_temp, y, Path(args.figure))


if __name__ == "__main__":
    main()
