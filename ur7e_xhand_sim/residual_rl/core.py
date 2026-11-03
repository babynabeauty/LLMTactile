"""Pure timing, trajectory and task contracts, independent of Genesis."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from numbers import Integral
import numpy as np

ROOT = Path(__file__).resolve().parents[2]

@dataclass(frozen=True)
class RateConfig:
    physics_hz: int = 300
    tactile_hz: int = 60
    action_hz: int = 15

    def __post_init__(self):
        if any(isinstance(rate, (bool, np.bool_)) or not isinstance(rate, Integral)
               for rate in (self.physics_hz, self.tactile_hz, self.action_hz)):
            raise ValueError('Rates must be integer tick frequencies')
        if min(self.physics_hz, self.tactile_hz, self.action_hz) <= 0:
            raise ValueError('Rates must be positive')
        if self.physics_hz % self.tactile_hz or self.tactile_hz % self.action_hz:
            raise ValueError('Rates must divide exactly')

    @property
    def physics_dt(self): return 1 / self.physics_hz
    @property
    def action_dt(self): return 1 / self.action_hz
    @property
    def tactile_stride(self): return self.physics_hz // self.tactile_hz
    @property
    def action_stride(self): return self.physics_hz // self.action_hz
    @property
    def samples_per_action(self): return self.tactile_hz // self.action_hz

class Trajectory:
    def __init__(self, path):
        path = Path(path).expanduser().resolve()
        self.path = path
        self.sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        # Stream rows: each contains large tactile arrays which are not needed here.
        times, ends, targets, phases, grip_gains, explicit_gains = [], [], [], [], [], []
        with path.open() as stream:
            for line in stream:
                row = json.loads(line)
                times.append(row['before']['sim_time_s'])
                ends.append(row['after']['sim_time_s'])
                targets.append(row['sent_target_rad'])
                phases.append(row['phase'])
                gain=row.get('grip_gain',1.)
                if isinstance(gain,bool) or not isinstance(gain,(float,int)) or not np.isfinite(gain) or not 0 < gain <= 3:
                    raise ValueError('Recorded grip_gain must be one finite absolute scalar in (0, 3]')
                grip_gains.append(gain)
                explicit_gains.append('grip_gain' in row)
        self.times = np.asarray(times, dtype=float)
        ends = np.asarray(ends, dtype=float)
        self.targets = np.asarray(targets, dtype=float)
        self.phases = np.asarray(phases)
        self.command_grip_gains = np.asarray(grip_gains,dtype=float)
        self.explicit_grip_gain = np.asarray(explicit_gains,dtype=bool)
        if not len(times) or self.targets.ndim != 2 or self.targets.shape[1] == 0:
            raise ValueError('Trajectory must contain a nonempty rectangular target array')
        if not (np.isfinite(self.times).all() and np.isfinite(ends).all()
                and np.isfinite(self.targets).all()):
            raise ValueError('Trajectory times and targets must be finite')
        if (ends <= self.times).any() or not np.allclose(ends[:-1], self.times[1:], rtol=0, atol=1e-8):
            raise ValueError('Trajectory command intervals must be positive and contiguous')
        self.duration_s = float(ends[-1])
        if not np.all(np.diff(self.times) > 0):
            raise ValueError('Non-monotonic demonstration timestamps')
        self.phase_starts = {p: float(self.times[np.flatnonzero(self.phases == p)[0]])
                             for p in dict.fromkeys(phases)}
        reference = path.parent / 'reference.json'
        if reference.exists():
            expected = json.loads(reference.read_text())['files'].get(path.name)
            if expected and expected != self.sha256:
                raise ValueError('Demonstration hash differs from successful reference')

    def target(self, time_s):
        t = np.asarray(time_s)
        return np.stack([np.interp(t, self.times, self.targets[:, i])
                         for i in range(self.targets.shape[1])], axis=-1)

    def phase(self, time_s):
        index = np.clip(np.searchsorted(self.times, time_s, side='right') - 1,
                        0, len(self.times) - 1)
        return self.phases[index]

    def grip_gain_at(self,time_s):
        """Actual per-command absolute gain, held over its interval, never interpolated."""
        time_s=np.asarray(time_s)
        if not np.isfinite(time_s).all():raise ValueError('Gain query times must be finite')
        index=np.clip(np.searchsorted(self.times,time_s,side='right')-1,0,len(self.times)-1)
        return self.command_grip_gains[index]

def interval_overlap(t0, t1, start, end):
    """Elapsed time shared by a physics interval and a designated task interval.

    This excludes the fraction of a final physics/action step beyond the
    demonstration end instead of extending the required hold duration.
    """
    t0, t1, start, end = np.broadcast_arrays(
        *[np.asarray(value, dtype=float) for value in (t0, t1, start, end)])
    if not all(np.isfinite(value).all() for value in (t0, t1, start, end)):
        raise ValueError('Interval endpoints must be finite')
    if np.any(t1 < t0) or np.any(end < start):
        raise ValueError('Interval ends must not precede starts')
    return np.maximum(0., np.minimum(t1, end) - np.maximum(t0, start))

def hold_success(hold_duration_s, hold_min_clearance_m, fault=False):
    """All of the designated final hold, not an accumulated arbitrary interval."""
    return ((np.asarray(hold_duration_s) >= 3. - 1e-6)
            & (np.asarray(hold_min_clearance_m) >= .05) & ~np.asarray(fault, dtype=bool))
