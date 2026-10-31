"""Per-trial features from a fingertip slide: friction from force, effusivity from temperature.

The two channels are kept strictly separate so the force-only / temperature-only /
combined ablation means what it says. Both sets are invariant to the nuisance
parameters the collection script randomizes: the force features work on the ratio
of tangential to normal force rather than either magnitude, and the temperature
features are all relative to the temperature at contact onset.
"""

from __future__ import annotations

import numpy as np

AMBIENT = 22.0  # asymptote of the cooling curve, used for the decay-rate fit
CONTACT_THRESHOLD = 0.3  # N of normal force that counts as touching

FORCE_FEATURE_NAMES = ("f_n_mean", "f_n_std", "f_t_mean", "f_t_std", "mu_mean", "mu_std", "mu_p90_p10", "f_t_rate")
TEMP_FEATURE_NAMES = (
    "dT_total",
    "dT_first_300ms",
    "dT_at_500ms",
    "dT_slide",
    "cool_rate_max",
    "T_std_slide",
    "decay_rate",
)


def split_normal_tangential(force: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split ``(n_trials, n_steps, 3)`` world-frame forces into normal and tangential parts.

    The collection script rotates the sensor's link-frame reading into the world frame
    with the fingertip's own orientation, so the plate normal is +z for every trial.
    """
    f_n = force[:, :, 2]
    f_t = np.linalg.norm(force[:, :, :2], axis=-1)
    return f_n, f_t


def contact_onset(f_n: np.ndarray, default: int) -> np.ndarray:
    """First step where the normal force crosses the contact threshold, per trial."""
    touching = f_n > CONTACT_THRESHOLD
    first = np.argmax(touching, axis=1)
    return np.where(touching.any(axis=1), first, default)


def force_features(f_n: np.ndarray, f_t: np.ndarray, slide: slice, dt: float) -> np.ndarray:
    """Friction-flavoured features over the slide window.

    The friction estimate is a ratio of sums over the steps that are actually in
    contact, not a mean of per-step ratios: a single step where the normal force
    dips to the noise floor would otherwise dominate the average.
    """
    n, t = f_n[:, slide], f_t[:, slide]
    touching = n > CONTACT_THRESHOLD
    weight = touching.astype(np.float64)
    count = np.maximum(weight.sum(axis=1), 1.0)
    mu = np.where(touching, t / np.maximum(n, CONTACT_THRESHOLD), np.nan)
    mu_mean = t.sum(axis=1, where=touching) / np.maximum(n.sum(axis=1, where=touching), 1e-6)
    return np.stack(
        [
            (weight * n).sum(axis=1) / count,
            n.std(axis=1),
            (weight * t).sum(axis=1) / count,
            t.std(axis=1),
            mu_mean,
            np.nan_to_num(np.nanstd(mu, axis=1)),
            np.nan_to_num(np.nanpercentile(mu, 90, axis=1) - np.nanpercentile(mu, 10, axis=1)),
            np.abs(np.diff(t, axis=1)).mean(axis=1) / dt,
        ],
        axis=1,
    )


def temp_features(temperature: np.ndarray, onset: np.ndarray, slide: slice, dt: float) -> np.ndarray:
    """Cooling-curve features, all measured relative to the temperature at contact onset."""
    n_trials, n_steps = temperature.shape
    idx = np.arange(n_steps)[None, :]
    t_onset = temperature[np.arange(n_trials), onset]

    def drop_at(seconds: float) -> np.ndarray:
        step = np.minimum(onset + int(round(seconds / dt)), n_steps - 1)
        return t_onset - temperature[np.arange(n_trials), step]

    after_contact = idx >= onset[:, None]
    smoothed = np.apply_along_axis(lambda row: np.convolve(row, np.ones(9) / 9, mode="same"), 1, temperature)
    rate = np.gradient(smoothed, dt, axis=1)
    cool_rate_max = np.where(after_contact, rate, 0.0).min(axis=1)

    # Rate of the exponential relaxation toward the plate, fitted on the contact
    # window: log((T - ambient) / (T_onset - ambient)) is linear in time with slope
    # -1/tau, and tau is what the plate's conductivity sets.
    excess = np.maximum(temperature - AMBIENT, 1e-3)
    log_excess = np.log(excess / np.maximum(t_onset - AMBIENT, 1e-3)[:, None])
    weight = after_contact.astype(np.float64)
    time = (idx - onset[:, None]) * dt
    w_sum = np.maximum(weight.sum(axis=1), 1.0)
    t_mean = (weight * time).sum(axis=1) / w_sum
    y_mean = (weight * log_excess).sum(axis=1) / w_sum
    cov = (weight * (time - t_mean[:, None]) * (log_excess - y_mean[:, None])).sum(axis=1)
    var = np.maximum((weight * (time - t_mean[:, None]) ** 2).sum(axis=1), 1e-9)
    decay_rate = -cov / var

    return np.stack(
        [
            t_onset - temperature[:, -1],
            drop_at(0.3),
            drop_at(0.5),
            temperature[:, slide][:, 0] - temperature[:, slide][:, -1],
            cool_rate_max,
            temperature[:, slide].std(axis=1),
            decay_rate,
        ],
        axis=1,
    )


def extract(data: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(force_features, temp_features, labels)`` for a collected dataset."""
    force = np.asarray(data["force"], dtype=np.float64)
    temperature = np.asarray(data["temperature"], dtype=np.float64)
    dt = float(data["dt"])
    n_settle, n_press = int(data["n_settle"]), int(data["n_press"])
    slide = slice(n_settle + n_press, None)

    # Tare against the pre-touch window, as any real tactile pipeline does: the
    # sensor's standing offset and its random-walk drift are what limit the friction
    # estimate, and both are visible while the fingertip is still in free air.
    force = force - force[:, :n_settle, :].mean(axis=1, keepdims=True)
    f_n, f_t = split_normal_tangential(force)
    onset = contact_onset(f_n, default=n_settle + n_press)
    return (
        force_features(f_n, f_t, slide, dt),
        temp_features(temperature, onset, slide, dt),
        np.asarray(data["labels"]),
    )
