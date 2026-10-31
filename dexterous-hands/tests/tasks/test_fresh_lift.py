"""Fixed lift geometry protocol and yaw-invariant palm cost."""
from types import SimpleNamespace
import math
import torch
from shared_terms import surface_distance_reward
from tasks.grasp_lift.custom_terms import PalmDownPenalty


def test_palm_cost_is_yaw_invariant():
    term = object.__new__(PalmDownPenalty)
    term._normal = torch.tensor([0., 0., -1.])
    # World yaw leaves a downward normal invariant; pi around X points upward.
    q = torch.tensor([[1.,0.,0.,0.], [math.cos(.7),0.,0.,math.sin(.7)], [0.,1.,0.,0.]])
    term._palm = SimpleNamespace(get_quat=lambda:q)
    torch.testing.assert_close(term.compute(),torch.tensor([0.,0.,2.]))


def test_fresh_lift_resolves_sizes_goals_and_noncolliding_markers():
    import genesis as gs
    from registry import _load_tasks, get_task_config
    if not gs._initialized:
        gs.init(backend=gs.cpu, logging_level='ERROR')
    _load_tasks()
    for hand, size, band in [('wuji', .08, (.65,.75)), ('sharpa', .08, (.65,.75)), ('allegro', .10, (.70,.85))]:
        c=get_task_config(task_name='grasp_lift',modifiers={'robot':hand,'obj':'primitives','sensors':'none','stage':1},config_override_path=f'conf/experiments/grasp_lift_{hand}_fresh_v141.yaml')
        assert len(c.scene_options.obj.grouped_entities)==17
        assert tuple(c.scene_options.obj.grouped_entities[3].size)==(size,)*3
        assert tuple(c.command_options.goal_pos.z_range)==band
        assert c.runner_options.max_iterations==5000
        assert c.event_options.object_pulse.clean_fraction==.25
        assert all(not v.collision and v.fixed for v in c.scene_options.vis_obj.grouped_entities)
        assert c.reward_options.natural_motion.weight==-.5


def test_plane_clearance_successors_remove_tilt_reward():
    import genesis as gs
    from registry import _load_tasks, get_task_config
    if not gs._initialized:
        gs.init(backend=gs.cpu, logging_level='ERROR')
    _load_tasks()
    for hand in ('wuji', 'sharpa', 'allegro'):
        c = get_task_config(
            task_name='grasp_lift',
            modifiers={'robot': hand, 'obj': 'primitives', 'sensors': 'none', 'stage': 1},
            config_override_path=f'conf/experiments/grasp_lift_{hand}_fresh_v141_plane_clearance.yaml',
        )
        assert not hasattr(c.reward_options, 'centre_rise')
        assert c.reward_options.plane_clearance.name == 'plane_clearance_reward'
        assert c.reward_options.plane_clearance.plane_z == .5
        assert c.reward_options.plane_clearance.threshold == .05


def test_surface_proximity_average_rewards_every_close_probe_without_hand_count_bias():
    readings = torch.tensor([[0.0, 0.03, 0.06], [0.03, 0.03, 0.03]])
    obs = SimpleNamespace(compute=lambda: readings)
    env = SimpleNamespace(observation_manager=SimpleNamespace(get_term=lambda name: obs))
    reward = surface_distance_reward(env, sigma=0.03, average=True)
    expected = torch.exp(-(readings / 0.03).square()).mean(dim=-1)
    torch.testing.assert_close(reward, expected)


def test_v5_contact_recipe_adds_object_only_palm_center_and_fingertip_probes():
    import genesis as gs
    from registry import _load_tasks, get_task_config
    if not gs._initialized:
        gs.init(backend=gs.cpu, logging_level='ERROR')
    _load_tasks()
    for hand in ('wuji', 'sharpa', 'allegro'):
        c = get_task_config(
            task_name='grasp_lift',
            modifiers={'robot': hand, 'obj': 'primitives', 'sensors': 'none', 'stage': 1},
            config_override_path=f'conf/experiments/grasp_lift_{hand}_fresh_v141_plane_lift_contact_fast_v5.yaml',
        )
        names = [name for name, _ in c.sensors_options if name.startswith('priv_surface_distance_')]
        assert 'priv_surface_distance_palm_center' in names
        assert len(names) == len(c.scene_options.robot.metadata.fingertip_links) + 1
        palm = c.sensors_options.priv_surface_distance_palm_center
        assert palm.attach_link_name == c.scene_options.robot.metadata.palm_link
        assert tuple(palm.sensor.probe_local_pos[0]) == tuple(c.scene_options.robot.metadata.grasp_center)
        assert palm.track_link_names == ['obj']
        proximity = c.reward_options.contact_proximity
        assert proximity.name == 'surface_distance_reward'
        assert proximity.weight == 1.0
        assert proximity.params == {
            'obs_name': 'surface_distance', 'nearest_k': -1, 'sigma': 0.03, 'average': True
        }


def test_small_lift_recipe_scales_wuji_and_sharpa_objects_to_six_cm():
    import genesis as gs
    from registry import _load_tasks, get_task_config
    if not gs._initialized:
        gs.init(backend=gs.cpu, logging_level='ERROR')
    _load_tasks()
    for hand in ('wuji', 'sharpa'):
        base = get_task_config(
            task_name='grasp_lift', modifiers={'robot': hand, 'obj': 'primitives', 'sensors': 'none', 'stage': 1},
            config_override_path=f'conf/experiments/grasp_lift_{hand}_fresh_v141_simple_contact_fast_v7.yaml')
        small = get_task_config(
            task_name='grasp_lift', modifiers={'robot': hand, 'obj': 'primitives', 'sensors': 'none', 'stage': 1},
            config_override_path=f'conf/experiments/grasp_lift_{hand}_fresh_v141_simple_contact_small6cm_fast_v8.yaml')
        assert len(base.scene_options.obj.grouped_entities) == len(small.scene_options.obj.grouped_entities) == 17
        for old, new, ghost in zip(base.scene_options.obj.grouped_entities, small.scene_options.obj.grouped_entities,
                                   small.scene_options.vis_obj.grouped_entities):
            for key in ('size', 'radius', 'height', 'scale'):
                if hasattr(old, key):
                    old_v, new_v, ghost_v = getattr(old, key), getattr(new, key), getattr(ghost, key)
                    old_t, new_t, ghost_t = torch.as_tensor(old_v), torch.as_tensor(new_v), torch.as_tensor(ghost_v)
                    torch.testing.assert_close(new_t, old_t * 0.75)
                    torch.testing.assert_close(ghost_t, new_t)
        assert tuple(small.command_options.goal_pos.z_range) == (0.65, 0.75)
