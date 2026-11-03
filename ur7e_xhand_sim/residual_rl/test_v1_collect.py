"""Collection planning: scaled and unseen objects, tabletop bounds. CPU only, no bridge or Genesis."""
import json
from pathlib import Path
import tempfile
import unittest

import v1_collect


def demo_file(directory, object_id, hand_shape, spec, xy=(-.50, -.1333)):
    path = Path(directory)/f'{object_id}_{hand_shape}.json'
    path.write_text(json.dumps(dict(object_id=object_id, object_spec=spec, hand_shape=hand_shape,
                                    object_initial_position_m=[*xy, .2225])))
    return str(path)


def config(directory):
    demos = [demo_file(directory, 'cuboid', 'pinch', {'shape': 'cuboid'}),
             demo_file(directory, 'cuboid', 'wrap', {'shape': 'cuboid'}),
             demo_file(directory, 'sphere', 'pinch', {'shape': 'sphere'})]
    return dict(demos=demos, scales=[.85, 1.15], train_placements=10, test_placements=4, edge_placements=3,
                unseen=[dict(object_id='cube45', object_spec={'shape': 'cube'}, near='cuboid')],
                unseen_placements=2, xy_range_m=.10, edge_range_m=.15, yaw_range_deg=10, seed=1)


class PlanJobsTest(unittest.TestCase):
    def test_scaled_objects_get_train_and_test_but_no_edge_placements(self):
        with tempfile.TemporaryDirectory() as d:
            jobs = v1_collect.plan_jobs(config(d))

        def count(object_id, split, edge=False):
            return sum(j['object_id'] == object_id and j['split'] == split and j['placement'].get('edge', False) == edge
                       for j in jobs)
        self.assertEqual([count('cuboid', 'train'), count('cuboid', 'test'), count('cuboid', 'test', True)], [20, 8, 6])
        self.assertEqual([count('cuboid_x085', 'train'), count('cuboid_x085', 'test'),
                          count('cuboid_x085', 'test', True)], [20, 8, 0])
        sphere = next(j for j in jobs if j['object_id'] == 'sphere_x115')
        self.assertAlmostEqual(sphere['object_spec']['radius_m'], .025875)
        self.assertEqual(len({j['run_id'] for j in jobs}), len(jobs))

    def test_skip_drops_one_scene_only(self):
        with tempfile.TemporaryDirectory() as d:
            jobs = v1_collect.plan_jobs(dict(config(d), skip=['cuboid_x115_wrap']))
        scenes = {(j['object_id'], j['hand_shape']) for j in jobs}
        self.assertNotIn(('cuboid_x115', 'wrap'), scenes)
        self.assertIn(('cuboid_x115', 'pinch'), scenes)
        self.assertIn(('cuboid_x085', 'wrap'), scenes)

    def test_unseen_object_runs_every_hand_shape_of_its_near_object(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = config(d)
            unseen = [j for j in v1_collect.plan_jobs(cfg) if j['object_id'] == 'cube45']
            self.assertEqual(sorted(j['hand_shape'] for j in unseen), ['pinch', 'pinch', 'wrap', 'wrap'])
            self.assertTrue(all(j['split'] == 'test' and j['placement']['unseen_object'] for j in unseen))
            cfg['unseen'][0]['near'] = 'cylinder'
            with self.assertRaises(ValueError):
                v1_collect.plan_jobs(cfg)

    def test_object_near_the_tabletop_edge_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            # From the old demonstration position (x = -.60), -10 cm already reaches the edge at x = -.72.
            cfg = dict(config(d), demos=[demo_file(d, 'cuboid', 'pinch', {'shape': 'cuboid'}, xy=(-.60, -.1333))],
                       unseen=[])
            with self.assertRaises(ValueError):
                v1_collect.plan_jobs(cfg)


if __name__ == '__main__':
    unittest.main()
