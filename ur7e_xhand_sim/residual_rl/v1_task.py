"""v1 task contracts: episode sampling, residual limits, breakage, slip and success.

CPU-only; never imports Genesis. Units are kg, N, m, rad and s. Mass, friction
and F_break are reward/termination labels only; the actor sees at most the
Astra fragility level, which is sampled with errors.
"""
from dataclasses import dataclass, asdict
import numpy as np

GRAVITY = 9.81
LEVEL_BOUNDS_N = (2., 5., 12.)
LEVEL_RANGES_N = {1: (.8, 2.), 2: (2., 5.), 3: (5., 12.), 4: (12., np.inf)}
# Force-bearing regions; F_break applies to the normal resultant of each one.
REGIONS = ('thumb', 'index', 'middle', 'ring', 'pinky', 'palm', 'arm')
FINGERS = REGIONS[:5]
# Hand-local joints per finger: thumb bend/rota1/rota2, index bend/joint1/joint2, then two each.
FINGER_JOINTS = {'thumb': (0, 1, 2), 'index': (3, 4, 5), 'middle': (6, 7), 'ring': (8, 9), 'pinky': (10, 11)}
# Fingers the small model may correct; a wrap does not require every finger to touch.
HAND_SHAPES = {'pinch': ('thumb', 'index'), 'tripod': ('thumb', 'index', 'middle'), 'wrap': FINGERS}
PHASES = ('close', 'preload', 'test_lift', 'lift', 'hold')
# BasePolicyV1 motion per 60 Hz tick; v1_demo closes and lifts with the same timing.
BASE_MOTION = dict(position_m_per_tick=.0015, hand_rad_per_tick=.012, turn_rad_per_tick=.01,
                   min_ticks=20, settle_ticks=12)
# Yaw symmetry period of each shape; cylinders and spheres look the same at every yaw.
YAW_PERIOD_DEG = {'cube': 90., 'cuboid': 180., 'hex_prism': 60.}


REGION_KEYS = (('thumb', 0), ('index', 1), ('mid', 2), ('ring', 3), ('pinky', 4), ('little', 4))


def link_region(name):
    """REGIONS index of a robot link: finger by name, other hand links are palm, the rest arm."""
    name = name.lower()
    for key, region in REGION_KEYS:
        if key in name:
            return region
    return 5 if 'hand' in name else 6


def hand_shape_joints(shape):
    return tuple(j for finger in HAND_SHAPES[shape] for j in FINGER_JOINTS[finger])


def action_joints(shapes):
    """Union of corrected joints over the hand shapes of one experiment, in joint order."""
    return tuple(sorted({j for shape in shapes for j in hand_shape_joints(shape)}))


@dataclass(frozen=True)
class V1Config:
    residual_bound_rad: float = .15
    residual_rate_rad: float = .05
    progress_weight: float = 3.
    force_weight: float = .02
    break_soft_weight: float = 5.
    break_soft_start: float = .8
    break_consecutive_steps: int = 2
    slip_weight_per_mm: float = 2.
    slip_fail_m: float = .005
    residual_change_weight: float = .005
    success_bonus: float = 25.
    failure_penalty: float = 200.
    table_fault_N: float = 5.
    support_threshold_N: float = .05

    def __post_init__(self):
        values = asdict(self)
        if not all(np.isfinite(v) and v >= 0 for v in values.values()):
            raise ValueError('V1Config values must be finite and nonnegative')
        if not 0 < self.break_soft_start < 1:
            raise ValueError('break_soft_start must lie in (0, 1)')
        if type(self.break_consecutive_steps) is not int or self.break_consecutive_steps < 1:
            raise ValueError('break_consecutive_steps must be a positive integer')
        if self.residual_bound_rad <= 0 or self.residual_rate_rad <= 0:
            raise ValueError('Residual bound and rate must be positive')


@dataclass(frozen=True)
class SamplerConfig:
    levels: tuple = (1, 2, 3, 4)
    mass_range_kg: tuple = (.05, .2)
    friction_range: tuple = (.2, .8)
    narrow_fraction: float = .5
    narrow_ratio: tuple = (1.2, 2.)
    feasibility_margin: float = 1.2
    astra_error_rate: float = .15
    astra_lower_share: float = .7
    max_tries: int = 100000
    # PLAN_v1 section 5/6: R bases get at most this share of training episodes; this share of S
    # episodes starts from a shifted object (composite deviation within offset_max_*).
    r_max_fraction: float = .5
    s_offset_fraction: float = .5
    s_offset_variants: int = 4
    offset_max_m: float = .01
    offset_max_yaw_deg: float = 5.

    def __post_init__(self):
        levels = tuple(int(v) for v in self.levels)
        if not levels or len(set(levels)) != len(levels) or any(v not in LEVEL_RANGES_N for v in levels):
            raise ValueError('levels must be distinct values from 1-4')
        object.__setattr__(self, 'levels', levels)
        for name in ('mass_range_kg', 'friction_range', 'narrow_ratio'):
            lo, hi = getattr(self, name)
            if not (np.isfinite([lo, hi]).all() and 0 < lo <= hi):
                raise ValueError(f'{name} must be a positive ordered pair')
        if self.narrow_ratio[0] < self.feasibility_margin:
            raise ValueError('Narrow windows must remain feasible')
        for name in ('narrow_fraction', 'astra_error_rate', 'astra_lower_share', 'r_max_fraction', 's_offset_fraction'):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f'{name} must lie in [0, 1]')
        if type(self.s_offset_variants) is not int or self.s_offset_variants < 0:
            raise ValueError('s_offset_variants must be a nonnegative integer')
        if not (self.offset_max_m > 0 and self.offset_max_yaw_deg > 0):
            raise ValueError('Offset bounds must be positive')


def minimum_grip_force(mass_kg, friction):
    """Per-finger normal force an opposed two-finger pinch needs: 2*mu*F >= m*g."""
    return np.asarray(mass_kg, float)*GRAVITY/(2*np.asarray(friction, float))


def force_to_level(f_break_N):
    return int(np.searchsorted(LEVEL_BOUNDS_N, f_break_N)) + 1


def astra_level(rng, level, cfg):
    """Simulated Astra judgment: wrong with a fixed rate, biased to the safer side."""
    if rng.random() >= cfg.astra_error_rate:
        return level
    wrong = level-1 if rng.random() < cfg.astra_lower_share else level+1
    return int(np.clip(wrong, 1, 4))


def sample_episode(rng, cfg=SamplerConfig()):
    """Level first, then a feasible material, so rejection never thins level 1."""
    level = int(rng.choice(cfg.levels))
    lo, hi = LEVEL_RANGES_N[level]
    m_lo, m_hi = cfg.mass_range_kg
    narrow = level < 4 and rng.random() < cfg.narrow_fraction
    f_break = np.inf if level == 4 else float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
    for _ in range(cfg.max_tries):
        mu = rng.uniform(*cfg.friction_range)
        if level == 4:
            mass = rng.uniform(m_lo, m_hi)
            break
        if narrow:
            # Set F_break/F_min directly. Near the top of level 3 even 200 g
            # cannot need that much force, so F_break is redrawn as well.
            mass = 2*mu*(f_break/rng.uniform(*cfg.narrow_ratio))/GRAVITY
            if m_lo <= mass <= m_hi:
                break
            f_break = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
        else:
            mass = rng.uniform(m_lo, m_hi)
            if f_break > cfg.feasibility_margin*minimum_grip_force(mass, mu):
                break
    else:
        raise RuntimeError(f'No feasible material for fragility level {level}')
    f_min = float(minimum_grip_force(mass, mu))
    return dict(mass_kg=float(mass), friction=float(mu), level=level, f_break_N=float(f_break),
                astra_level=astra_level(rng, level, cfg), f_min_N=f_min,
                window_ratio=float(f_break/f_min), narrow_window=bool(narrow))


def limit_residual(action, previous, cfg):
    """Absolute residual request in [-1, 1] * bound, moved at most rate per action."""
    action = np.asarray(action, dtype=np.float64)
    if not np.isfinite(action).all():
        raise ValueError('Non-finite policy action')
    requested = np.clip(action, -1, 1)*cfg.residual_bound_rad
    step = np.clip(requested-previous, -cfg.residual_rate_rad, cfg.residual_rate_rad)
    return np.clip(previous+step, -cfg.residual_bound_rad, cfg.residual_bound_rad)


def break_soft_rate(region_normal_N, f_break_N, cfg):
    """Per-second penalty rising linearly from break_soft_start*F_break to F_break."""
    peak = np.asarray(region_normal_N, float).max(-1)
    ratio = np.where(np.isfinite(f_break_N), peak/np.asarray(f_break_N, float), 0.)
    return cfg.break_soft_weight*np.clip((ratio-cfg.break_soft_start)/(1-cfg.break_soft_start), 0, 1)


class BreakMonitor:
    """Broken after F_break is exceeded on consecutive physics steps (any region)."""
    def __init__(self, n, consecutive):
        self.consecutive = consecutive
        self.count = np.zeros(n, dtype=int)
        self.broken = np.zeros(n, dtype=bool)

    def reset(self, lanes):
        self.count[lanes] = 0
        self.broken[lanes] = False

    def update(self, region_normal_N, f_break_N, active):
        over = active & (np.asarray(region_normal_N).max(-1) > f_break_N)
        self.count = np.where(over, self.count+1, 0)
        newly = (self.count >= self.consecutive) & ~self.broken & active
        self.broken |= newly
        return newly


class SlipTracker:
    """Largest object displacement in the palm frame since lifting began."""
    def __init__(self, n):
        self.reference = np.full((n, 3), np.nan)
        self.distance = np.zeros(n)

    def reset(self, lanes):
        self.reference[lanes] = np.nan
        self.distance[lanes] = 0.

    def update(self, object_in_palm_m, active):
        start = active & np.isnan(self.reference[:, 0])
        self.reference[start] = object_in_palm_m[start]
        moved = np.linalg.norm(object_in_palm_m-np.nan_to_num(self.reference), axis=-1)
        increase = np.where(active, np.maximum(moved-self.distance, 0.), 0.)
        self.distance += increase
        return increase


def opposing_support(region_normal_N, threshold_N):
    """Thumb or palm on one side and any other finger on the other; columns follow REGIONS."""
    touch = np.asarray(region_normal_N) > threshold_N
    return (touch[:, 0] | touch[:, 5]) & touch[:, 1:5].any(-1)


def takeover_tick(phase_starts, physics_hz=300, action_hz=15):
    """First 'close' command, floored to the action grid, in physics ticks."""
    if 'close' not in phase_starts:
        raise ValueError('v1 bases require a close phase')
    return int(np.floor(phase_starts['close']*action_hz+1e-9))*(physics_hz//action_hz)


def wrap_yaw(yaw_deg, period_deg):
    """Signed yaw error modulo the shape's symmetry; 0 for shapes without a yaw period."""
    return 0. if period_deg is None else float((yaw_deg+period_deg/2) % period_deg-period_deg/2)


def base_draw_weights(labels, r_max_fraction):
    """Uniform over bases, except that R bases share at most r_max_fraction of the draws."""
    r = np.asarray([label == 'R' for label in labels])
    weights = np.ones(len(r))
    if r.any() and (~r).any() and r.mean() > r_max_fraction:
        weights[r] = r_max_fraction/r.sum()
        weights[~r] = (1-r_max_fraction)/(~r).sum()
    return weights/weights.sum()


def object_offsets(rng, pool, own, yaw_now, period_deg, palm_rotation, count, cfg):
    """Object shifts giving an S base a composite deviation drawn from the scene's S/R deviations.

    pool: (palm-frame deviation m, unsigned yaw error deg) pairs within the offset bounds; own and
    yaw_now: this base's palm-frame deviation and signed yaw error. The object rests on the table,
    so a shift is horizontal; a draw whose composite then leaves the bound is redrawn. Returns
    (world shift m, yaw turn deg) pairs.
    """
    out = []
    for _ in range(100*count):
        if len(out) == count:
            break
        target, yaw = pool[rng.integers(len(pool))]
        shift = palm_rotation@(np.asarray(target)-own)
        shift[2] = 0.
        if np.linalg.norm(own+palm_rotation.T@shift) > cfg.offset_max_m:
            continue
        turn = 0. if period_deg is None else float(rng.choice((-1., 1.))*yaw-yaw_now)
        out.append((shift, turn))
    return out


def episode_specs(base_indices, materials, levels, seed, cfg=SamplerConfig()):
    """Deterministic evaluation grid: every base x material x feasible level, one F_break each.

    F_break is drawn inside the level range above the feasibility margin; a
    level whose whole range is infeasible for that material is skipped.
    """
    rng = np.random.default_rng(seed)
    specs = []
    for base in base_indices:
        for mass, mu in materials:
            f_min = float(minimum_grip_force(mass, mu))
            for level in levels:
                lo, hi = LEVEL_RANGES_N[level]
                lo = max(lo, cfg.feasibility_margin*f_min)
                if level < 4 and lo >= hi:
                    continue
                f_break = np.inf if level == 4 else float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
                specs.append(dict(base_index=int(base), mass_kg=float(mass), friction=float(mu),
                    level=int(level), f_break_N=float(f_break), astra_level=astra_level(rng, level, cfg),
                    f_min_N=f_min, window_ratio=float(f_break/f_min),
                    narrow_window=bool(f_break/f_min <= cfg.narrow_ratio[1])))
    return specs
