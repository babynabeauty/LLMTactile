"""Read-only contact coverage logging during frozen validation policy replays.

This does not change the environment, reward, policy inputs, or checkpoint.
Only a simulation is run. Force magnitudes are not calibrated normal loads.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time
import traceback

import numpy as np
import torch
from stable_baselines3 import PPO

from .adaptive_env import AdaptiveConfig, AdaptiveVecEnv
from .core import interval_overlap
from .env import cpu


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


class CoverageAuditEnv(AdaptiveVecEnv):
    """Adds observation-only contact logging after the unchanged metric call."""

    audit_active = False

    def prepare_audit(self):
        self.audit_records = []
        self.audit_contacts = []
        self.audit_links = list(self.robot.links)
        by_id = {link.idx: link for link in self.audit_links}
        hand_root = self.robot.get_link('right_hand_link').idx

        def is_hand(link):
            seen = set()
            while link.idx not in seen:
                if link.idx == hand_root:
                    return True
                seen.add(link.idx)
                if link.parent_idx not in by_id:
                    return False
                link = by_id[link.parent_idx]
            raise AssertionError('Cyclic rigid link tree')

        self.audit_hand_mask = np.asarray([is_hand(link) for link in self.audit_links])
        self.audit_sensor_mask = np.asarray([link.idx in self.finger_links for link in self.audit_links])
        assert self.audit_sensor_mask.sum() == 5
        assert np.all(self.audit_hand_mask[self.audit_sensor_mask])
        self.audit_link_to_column = {link.idx: i for i, link in enumerate(self.audit_links)}
        self.audit_link_table = [dict(column=i, global_link_id=int(link.idx),
            name=link.name, parent_global_link_id=int(link.parent_idx),
            hand_subtree=bool(self.audit_hand_mask[i]),
            counted_by_existing_metric=bool(self.audit_sensor_mask[i]))
            for i, link in enumerate(self.audit_links)]

    def _metrics(self):
        result = super()._metrics()
        if not self.audit_active:
            return result
        contacts = self.robot.get_contacts(is_padded=True)
        a, b = cpu(contacts['link_a']), cpu(contacts['link_b'])
        valid = cpu(contacts['valid_mask']).astype(bool)
        cube_id = self.cube.base_link.idx
        pair = valid & ((a == cube_id) | (b == cube_id))
        lanes, slots = np.nonzero(pair)
        other = np.where(a[pair] == cube_id, b[pair], a[pair]).astype(np.int64)
        assert all(int(link) in self.audit_link_to_column for link in other)
        columns = np.asarray([self.audit_link_to_column[int(link)] for link in other], dtype=np.int64)
        force_a = cpu(contacts['force_a'])[pair]
        force_cube = np.where((a[pair] == cube_id)[:, None], force_a, -force_a)
        magnitudes = np.linalg.norm(force_a, axis=-1)
        shape = (self.num_envs, len(self.audit_links))
        load = np.zeros(shape, np.float64)
        vector = np.zeros((*shape, 3), np.float64)
        count = np.zeros(shape, np.int32)
        np.add.at(load, (lanes, columns), magnitudes)
        np.add.at(vector, (lanes, columns), force_cube)
        np.add.at(count, (lanes, columns), 1)
        sensor_columns = [self.audit_link_to_column[link] for link in self.finger_links]
        np.testing.assert_allclose(load[:, sensor_columns], result[1], rtol=1e-6, atol=1e-5)
        sample = len(self.audit_records)
        self.audit_records.append(dict(tick=self.tick, load=load, vector=vector,
            count=count, sensor_load=result[1].copy(), clearance=result[0].copy(),
            slip=result[2].copy(), fault=result[3].copy()))
        self.audit_contacts.append(dict(sample=np.full(len(lanes), sample, np.int32),
            lane=lanes.astype(np.int32), column=columns.astype(np.int32),
            position=cpu(contacts['position'])[pair], force_cube=force_cube,
            normal=cpu(contacts['normal'])[pair], magnitude=magnitudes))
        return result


def summarize(env, trace, episodes):
    load = trace['per_link_load_N']
    hand = load[:, :, env.audit_hand_mask].sum(-1)
    tracked = load[:, :, env.audit_sensor_mask].sum(-1)
    omitted = hand - tracked
    robot = load.sum(-1)
    omitted_mask = env.audit_hand_mask & ~env.audit_sensor_mask
    dt = 1 / 60
    times = trace['sample_time_s']
    phase_starts = env.trajectory.phase_starts
    windows = {'entire_control_window': np.full(len(times), dt)}
    for phase in ('preload', 'test_lift', 'lift', 'hold'):
        start = phase_starts[phase]
        later = [t for t in phase_starts.values() if t > start]
        end = min(later) if later else env.trajectory.duration_s
        windows[phase] = interval_overlap(times - dt, times, start, end)

    def weighted_mean(values, weights):
        return (values * weights[:, None]).sum(0) / weights.sum()

    phase_summary = {}
    for name, weights in windows.items():
        duration = float(weights.sum())
        integrals = {key: (values * weights[:, None]).sum(0)
            for key, values in [('all_hand', hand), ('tracked_five', tracked), ('omitted_hand', omitted), ('all_robot', robot)]}
        total_hand = float(integrals['all_hand'].sum())
        phase_summary[name] = dict(duration_s=duration,
            mean_all_hand_load_N=float(integrals['all_hand'].mean() / duration),
            mean_tracked_five_load_N=float(integrals['tracked_five'].mean() / duration),
            mean_omitted_hand_load_N=float(integrals['omitted_hand'].mean() / duration),
            mean_non_hand_robot_load_N=float((integrals['all_robot']-integrals['all_hand']).mean() / duration),
            omitted_fraction_of_hand_load_integral=float(integrals['omitted_hand'].sum() / max(total_hand, 1e-12)),
            per_lane_mean_all_hand_load_N=weighted_mean(hand, weights).tolist(),
            per_lane_mean_tracked_five_load_N=weighted_mean(tracked, weights).tolist(),
            per_lane_omitted_fraction=(integrals['omitted_hand']/np.maximum(integrals['all_hand'], 1e-12)).tolist())

    rows = []
    for column, meta in enumerate(env.audit_link_table):
        values = load[:, :, column]
        raw = trace['contact_magnitude_N'][trace['contact_link_column'] == column]
        rows.append(dict(**meta, mean_contact_magnitude_sum_N=float(values.mean()),
            max_contact_magnitude_sum_N=float(values.max()),
            integral_over_all_lanes_N_s=float(values.sum() * dt),
            fraction_of_all_hand_integral=float(values.sum()/max(hand.sum(), 1e-12)) if meta['hand_subtree'] else None,
            lane_sample_count_over_001N=int((values > .01).sum()),
            lane_sample_count_over_005N=int((values > .05).sum()),
            contact_count=int(len(raw)),
            individual_contact_force_quantiles_N=dict(zip(('min','p50','p90','p99','max'),
                np.quantile(raw,[0,.5,.9,.99,1]).tolist())) if len(raw) else None,
            per_lane_mean_load_N=values.mean(0).tolist()))
    mean_reported = np.asarray([e['mean_contact_force_N'] for e in episodes])
    np.testing.assert_allclose(tracked.mean(0), mean_reported, rtol=1e-6, atol=1e-5)
    return dict(task_successes=sum(e['task_success'] for e in episodes),
        strict_successes=sum(e['strict_success'] for e in episodes),
        phase_summary=phase_summary, per_link=rows,
        omitted_active_links=[row['name'] for row in rows if row['hand_subtree'] and
            not row['counted_by_existing_metric'] and row['max_contact_magnitude_sum_N'] > .01],
        non_hand_active_links=[row['name'] for row in rows if not row['hand_subtree'] and
            row['max_contact_magnitude_sum_N'] > .01],
        max_omitted_hand_load_N=float(omitted.max()),
        omitted_hand_active_lane_samples_over_001N=int((omitted > .01).sum()),
        all_sample_count=int(omitted.size),
        tracked_metric_max_abs_error_N=float(abs(tracked.mean(0)-mean_reported).max()),
        peak_hand_load_per_lane_N=hand.max(0).tolist(),
        peak_tracked_load_per_lane_N=tracked.max(0).tolist(),
        net_cube_force_from_hand_mean_N=trace['per_link_force_on_cube_N'][:, :, env.audit_hand_mask].sum(2).mean((0,1)).tolist(),
        episodes=episodes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest_path = args.reference / 'manifest.json'
    reference = json.loads(manifest_path.read_text())
    if reference['split'] != 'validation':
        raise ValueError('This audit is frozen to validation, not heldout')
    args.output.mkdir(parents=True, exist_ok=False)
    snapshot = args.output / 'source_snapshot'
    snapshot.mkdir()
    critical = ('env.py','adaptive_env.py','core.py','policy.py','adaptive_policy.py','normalized_sde_policy.py')
    sources = {}
    for name in critical:
        path = Path(__file__).parent / name
        digest = sha256(path)
        if digest != reference['source_hashes'][name]:
            raise AssertionError(f'Frozen source changed: {name}')
        shutil.copy2(path, snapshot / name)
        sources[name] = digest
    shutil.copy2(__file__, snapshot / Path(__file__).name)
    model_path = Path(reference['models']['tactile']['path'])
    assert sha256(model_path) == reference['models']['tactile']['sha256']
    torch.set_num_threads(4)
    model = PPO.load(model_path, device='cpu')
    write_json(args.output / 'manifest.json', dict(reference_manifest=str(manifest_path.resolve()),
        reference_manifest_sha256=sha256(manifest_path), conditions_kg_mu=reference['conditions_kg_mu'],
        config=reference['config'], seed=reference['seed'], model=reference['models']['tactile'],
        source_sha256=sources, audit_source_sha256=sha256(__file__),
        scope='simulation only; frozen validation conditions; no training or observation/reward modification',
        timing='Read-only logging immediately after unchanged _metrics at 60Hz, after current physics step; actions 15Hz',
        force_definition='Sum of each physical contact force vector magnitude, separately by effective rigid link. Not a normal-force estimate.',
        hand_definition='Runtime right_hand_link and descendants; fixed-link-merged collision geometries belong to their effective rigid link',
        existing_trace_limitation='Existing trace only contains five sensor-bearing link loads; it cannot recover other contact links'))
    env = None
    try:
        env = CoverageAuditEnv(n_envs=len(reference['conditions_kg_mu']), seed=reference['seed'],
            material_conditions=np.asarray(reference['conditions_kg_mu'], np.float32),
            config=AdaptiveConfig(**reference['config']), output_dir=args.output/'environment', mode='eval')
        env.prepare_audit()
        write_json(args.output/'link_table.json', env.audit_link_table)
        results = {}
        for controller in ('fixed_2', 'tactile'):
            destination = args.output/controller
            destination.mkdir()
            env.audit_active = False
            obs = env.reset()
            env.prepare_audit()
            env.audit_active = True
            actions, gains = [], []
            began = time.monotonic()
            while True:
                if controller == 'fixed_2':
                    action = np.full((env.num_envs,1), env.action_for_gain(2.), np.float32)
                else:
                    action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, infos = env.step(action)
                actions.append(action.copy())
                gains.append(np.asarray([info['applied_gains'] for info in infos]))
                if done.all():
                    break
            env.audit_active = False
            records = env.audit_records
            contact_rows = env.audit_contacts
            trace = dict(sample_time_s=np.asarray([r['tick']/300 for r in records]),
                per_link_load_N=np.stack([r['load'] for r in records]),
                per_link_force_on_cube_N=np.stack([r['vector'] for r in records]),
                per_link_contact_count=np.stack([r['count'] for r in records]),
                tracked_five_load_N=np.stack([r['sensor_load'] for r in records]),
                clearance_m=np.stack([r['clearance'] for r in records]),
                slip_speed_m_s=np.stack([r['slip'] for r in records]),
                fault=np.stack([r['fault'] for r in records]),
                actions=np.stack(actions), applied_gains=np.stack(gains))
            for source, dest in [('sample','contact_sample'),('lane','contact_lane'),
                ('column','contact_link_column'),('position','contact_position_world_m'),
                ('force_cube','contact_force_on_cube_N'),('normal','contact_normal_world'),('magnitude','contact_magnitude_N')]:
                trace[dest] = np.concatenate([r[source] for r in contact_rows])
            assert len(records) == 4*len(actions) == 724
            reference_trace_path = args.reference/controller/'trace.npz'
            with np.load(reference_trace_path) as original:
                compare = {'tracked_five_load_N':'contact_load_N','clearance_m':'clearance_m',
                    'slip_speed_m_s':'slip_speed_m_s','fault':'fault','actions':'actions','applied_gains':'applied_gains'}
                differences = {key:float(np.max(abs(trace[key].astype(float)-original[other].astype(float))))
                    for key,other in compare.items()}
            if max(differences.values()) > 1e-5:
                raise AssertionError(f'Frozen replay differs: {controller}, {differences}')
            episodes = [info['episode_metrics'] for info in infos]
            result = summarize(env, trace, episodes)
            result.update(reference_trace_sha256=sha256(reference_trace_path),
                reference_max_abs_differences=differences, wall_seconds=time.monotonic()-began)
            np.savez_compressed(destination/'trace.npz', **trace)
            write_json(destination/'summary.json', result)
            results[controller] = result
            print(json.dumps(dict(controller=controller, task_successes=result['task_successes'],
                entire_window=result['phase_summary']['entire_control_window'],
                omitted_active_links=result['omitted_active_links'], reference_differences=differences)), flush=True)
        baseline = results['fixed_2']['phase_summary']['entire_control_window']
        tactile = results['tactile']['phase_summary']['entire_control_window']
        comparison = {metric:dict(fixed_2=baseline[metric], tactile=tactile[metric],
            relative_reduction=(baseline[metric]-tactile[metric])/baseline[metric] if baseline[metric] else None)
            for metric in ('mean_tracked_five_load_N','mean_all_hand_load_N','mean_omitted_hand_load_N')}
        write_json(args.output/'comparison.json', comparison)
        write_json(args.output/'status.json', dict(status='completed', controllers=list(results)))
    except BaseException as exc:
        write_json(args.output/'status.json', dict(status='failed', error=repr(exc), traceback=traceback.format_exc()))
        raise
    finally:
        if env is not None:
            env.close()


if __name__ == '__main__':
    main()
