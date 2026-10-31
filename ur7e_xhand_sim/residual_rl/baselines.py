"""Causal tactile-only baseline, sampled at 60 Hz and acted on at 15 Hz."""
from collections import deque
import numpy as np


class ReactiveGripController:
    """Adapt the earlier centroid/drop heuristic to the actual 60/15 Hz rates.

    Contact coordinates and the nominal phase clock are fixed configuration,
    not object-state inputs. A latched alarm requests 2x gain; the environment's
    SAME action-rate limiter applies to this and every other controller.
    """
    config = {
        "tactile_hz": 60, "action_hz": 15, "window_s": .05, "window_samples": 4,
        "centroid_speed_threshold_m_s": .020, "signal_drop_threshold": .25,
        "min_signal_N": .01, "debounce_samples": 3, "requested_gain_after_alarm": 2.,
        "arming": "nominal preload completed", "latching": True,
        "note": "A 60 Hz adaptation, not the old 200 Hz controller; actions only change at 15 Hz.",
    }

    def __init__(self, positions_link_m, n_envs, start_time_s, arm_time_s):
        self.positions = np.asarray(positions_link_m, dtype=np.float64)
        if self.positions.shape != (2, 120, 3):
            raise ValueError("Need fixed thumb/index 120-taxel link-local coordinates")
        self.n_envs = n_envs
        self.start_time_s = float(start_time_s)
        self.arm_time_s = float(arm_time_s)
        self.history = [deque(maxlen=4) for _ in range(n_envs)]
        self.decision = np.zeros(n_envs, dtype=int)
        self.count = np.zeros(n_envs, dtype=int)
        self.latched = np.zeros(n_envs, dtype=bool)
        self.trigger_sample_time_s = np.full(n_envs, np.nan)
        self.trigger_action_time_s = np.full(n_envs, np.nan)

    def reset(self, mask):
        for i in np.flatnonzero(mask):
            self.history[i].clear()
            self.decision[i] = 0
            self.count[i] = 0
            self.latched[i] = False
            self.trigger_sample_time_s[i] = np.nan
            self.trigger_action_time_s[i] = np.nan

    def predict(self, observations):
        tactile = np.asarray(observations["tactile"])
        if tactile.shape != (self.n_envs, 12, 5, 120, 3):
            raise ValueError("Unexpected tactile window shape")
        events = []
        for i in range(self.n_envs):
            now = self.start_time_s + self.decision[i] / 15
            # Warm the detector with the available reset history once; thereafter
            # process only the four NEW samples, not overlapping history again.
            first = 0 if self.decision[i] == 0 else 8
            for j in range(first, 12):
                timestamp = now - (11 - j) / 60
                weight = np.linalg.norm(tactile[i, j, :2], axis=-1)
                total = weight.sum(-1)
                center = (weight[:, :, None] * self.positions).sum(1) / np.maximum(total[:, None], 1e-10)
                self.history[i].append((total, center))
                candidate = False
                speed, drop = np.zeros(2), np.zeros(2)
                if len(self.history[i]) == 4 and timestamp >= self.arm_time_s - 1e-9:
                    old, old_center = self.history[i][0]
                    speed = np.linalg.norm(center - old_center, axis=-1) / .05
                    drop = 1 - total / np.maximum(old, 1e-10)
                    valid = (old >= .01) & (total >= .01)
                    candidate = bool(np.any(valid & ((speed > .020) | (drop > .25))))
                self.count[i] = self.count[i] + 1 if candidate else 0
                if not self.latched[i] and self.count[i] >= 3:
                    self.latched[i] = True
                    self.trigger_sample_time_s[i] = timestamp
                    self.trigger_action_time_s[i] = now
                    events.append({"env_index": i, "sample_time_s": timestamp, "action_time_s": now,
                                   "observation_to_action_delay_s": now - timestamp,
                                   "centroid_speed_m_s": speed.tolist(), "drop_fraction": drop.tolist()})
            self.decision[i] += 1
        actions = np.repeat(np.where(self.latched, 1., -1.)[:, None], 2, axis=1).astype(np.float32)
        return actions, events
