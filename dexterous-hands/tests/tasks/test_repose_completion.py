from types import SimpleNamespace as NS

import pytest
import torch

from tasks.in_hand_repose.custom_terms import SettledTargetRotationCommand, settled_goal_completion


@pytest.mark.parametrize('dt', [.025, .05])
def test_settled_completion_pays_twenty_once_at_both_control_rates(dt):
    c = object.__new__(SettledTargetRotationCommand)
    c._env = NS(dt=dt, device='cpu', num_envs=2)
    c.quat = torch.tensor([[1., 0., 0., 0.]]).repeat(2, 1)
    omega = torch.zeros(2, 3)
    omega[1, 0] = .3
    c.object = NS(get_quat=lambda: c.quat, get_ang=lambda: omega)
    c.stats = {'orientation_error':torch.zeros(2), 'consecutive_success':torch.zeros(2)}
    c._dwell = torch.zeros(2)
    c._awarded = torch.zeros(2,dtype=torch.bool)
    c.time_left = torch.full((2,), 8.)
    c.orientation_success_threshold = .1
    c.settle_time = .25
    c.max_success_angular_speed = .25
    env = NS(dt=dt,command_manager=NS(get_term=lambda name:c))
    total = torch.zeros(2)
    for _ in range(round(2/dt)):
        c.post_physics_refresh()
        total += settled_goal_completion(env)*20*dt
    torch.testing.assert_close(total,torch.tensor([20.,0.]))


@pytest.mark.parametrize('hand, stem', [(h, s) for h in ('wuji', 'sharpa', 'allegro') for s in ('completion_p2', 'finish_p3')]
                         + [('wuji', 'centered_p4'), ('allegro', 'centered_p4')]
                         + [(h, s) for h in ('wuji', 'sharpa', 'allegro') for s in ('reach_r6', 'accumulate_r6b', 'wide_r7', 'pulses_p6', 'pulses_p6b', 'wide_r7b', 'minimal_r8', 'minimal_wide_r8w', 'minimal_pulses_p8', 'minimal_wide_pulses_p8w')]
                         + [(h, 'wide_pulses_p7w') for h in ('wuji', 'sharpa')])
def test_configured_completion_exceeds_tracking_credit_over_goal_timeout(hand, stem):
    from pathlib import Path
    import yaml
    from tasks.in_hand_repose.config import InHandReposeConfig

    root = Path(__file__).resolve().parents[2]
    config = yaml.safe_load((root / f'conf/experiments/in_hand_repose_{hand}_{stem}.yaml').read_text())
    rewards = config['reward_options']
    assert rewards['success_bonus']['name'] == 'settled_goal_completion'
    defaults = InHandReposeConfig().reward_options
    tracking_weight = rewards['track_orientation_gaussian'].get('weight', defaults.track_orientation_gaussian.weight)
    inv_l2 = rewards['track_orientation_inv_l2']
    tracking_weight += inv_l2['weight'] / inv_l2.get('params', {}).get('rot_eps', defaults.track_orientation_inv_l2.params['rot_eps'])
    # Near-goal shaping (P3) is bounded too; lingering at the goal must still pay less than completing it.
    tracking_weight += rewards.get('track_orientation_fine', {}).get('weight', 0.0)
    timeout = max(config['command_options']['goal_rot']['resampling_time_range'])
    assert rewards['success_bonus']['weight'] > tracking_weight * timeout

