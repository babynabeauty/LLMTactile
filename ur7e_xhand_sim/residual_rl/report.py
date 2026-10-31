"""Refresh a read-only analysis of residual-PPO training and matched evaluation.

This module never imports Genesis or the training/environment modules.
Run: .venv/bin/python -m ur7e_xhand_sim.residual_rl.report --root outputs/residual_rl_20261025
Missing or still-running suites are explicitly reported; rerun after completion.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'residual_rl_matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


TRAIN_RUNS = ('ppo_tactile_seed0', 'ppo_no_tactile_seed0')
SUITES = ('suite_mixed', 'suite_200g')
CONTROLLERS = ('nominal', 'mid', 'strong', 'reactive', 'initial',
               'ppo_tactile', 'ppo_no_tactile', 'nominal_reset_check')
LABELS = {'nominal': '固定 1.0', 'mid': '固定 1.5', 'strong': '逐步达到 2.0',
          'reactive': '触觉报警后增力', 'initial': '未训练初始策略',
          'ppo_tactile': '触觉 PPO', 'ppo_no_tactile': '无触觉 PPO',
          'nominal_reset_check': '固定 1.0 重置复核'}
PLOT_LABELS = {'nominal': 'Fixed 1.0', 'mid': 'Fixed 1.5', 'strong': 'Fixed 2.0',
               'reactive': 'Reactive', 'initial': 'Initial policy',
               'ppo_tactile': 'Tactile PPO', 'ppo_no_tactile': 'No-tactile PPO'}
COLORS = dict(zip(PLOT_LABELS, plt.get_cmap('tab10').colors))
RESET_TOLERANCES = {'clearance_m': 1e-5, 'contact_load_N': 1e-5,
                    'force_link_N_before': 1e-5, 'state_before': 1e-5,
                    'prefix_tactile': 0., 'prefix_state': 1e-7}


def finite_mean(values):
    values = np.asarray([v for v in values if v is not None], dtype=float)
    values = values[np.isfinite(values)]
    return float(values.mean()) if values.size else None


def number(value, decimals=3):
    return '缺失' if value is None else f'{value:.{decimals}f}'


class Reader:
    def __init__(self, root):
        self.root = root
        self.missing = []
        self.warnings = []
        self.sources = {}

    def record(self, path, data):
        self.sources[self.display(path)] = {
            'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}

    def display(self, path):
        try:
            return str(path.relative_to(self.root))
        except ValueError:
            return str(path.resolve())

    def json(self, path, required=True):
        if not path.is_file():
            if required:
                self.missing.append(self.display(path))
            return None
        data = path.read_bytes()
        self.record(path, data)
        try:
            return json.loads(data)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            self.warnings.append(f'{self.display(path)}: unreadable JSON: {exc}')
            return None

    def jsonl(self, path):
        if not path.is_file():
            self.missing.append(str(path.relative_to(self.root)))
            return []
        data = path.read_bytes()
        self.record(path, data)
        rows = []
        for index, line in enumerate(data.splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError('Expected an object')
                rows.append(row)
            except (ValueError, UnicodeDecodeError) as exc:
                self.warnings.append(f'{path.relative_to(self.root)}:{index}: skipped incomplete/invalid row: {exc}')
        return rows


def summarize_training(reader, name):
    directory = reader.root / name
    status = reader.json(directory / 'status.json') or {}
    config = reader.json(directory / 'model_config.json') or {}
    protocol = reader.json(directory / 'environment/environment_protocol.json') or {}
    episodes = reader.jsonl(directory / 'episodes.jsonl')
    logs = reader.jsonl(directory / 'training_metrics.jsonl')
    by_step = defaultdict(list)
    for episode in episodes:
        if 'training_timesteps' in episode and episode.get('return') is not None:
            by_step[int(episode['training_timesteps'])].append(float(episode['return']))
    curve = [{'timesteps': step, 'mean_return': float(np.mean(values)),
              'min_return': float(np.min(values)), 'max_return': float(np.max(values)),
              'episodes': len(values)} for step, values in sorted(by_step.items())]
    known = [e for e in episodes if e.get('success') is not None]
    tail = episodes[-32:]
    return {
        'kind': 'stochastic_exploratory_training_not_held_out_evaluation',
        'status': status.get('status', 'missing'),
        'timesteps': status.get('timesteps'), 'completed_episode_records': len(episodes),
        'training_successes': sum(bool(e['success']) for e in known),
        'episodes_with_success_label': len(known),
        'mean_return_all_episodes': finite_mean([e.get('return') for e in episodes]),
        'mean_return_last_32_episodes': finite_mean([e.get('return') for e in tail]),
        'last_32_episode_count': len(tail), 'episode_batch_curve': curve,
        'rollout_metrics': logs, 'model_config': config,
        'mass_kg_per_lane': protocol.get('mass_kg_per_lane', []),
        'friction_per_lane': protocol.get('friction_per_lane', []),
        'final_checkpoint_sha256': status.get('final_checkpoint_sha256'),
        'deterministic_algorithms': protocol.get('deterministic_algorithms'),
        'reset_mode': protocol.get('reset'),
    }


def summarize_revision(reader):
    """Preserve simultaneous changes and independently read the restoration probe."""
    revision = reader.json(reader.root / 'experiment_revision.json', required=False)
    if revision is None:
        return None
    previous = revision.get('previous_diagnostic_run')
    evidence = {}
    if previous:
        previous = Path(previous)
        if not previous.is_absolute():
            previous = reader.root / previous
        for name in ('determinism_probe', 'determinism_full_checkpoint_probe'):
            path = previous / name / 'result.json'
            result = reader.json(path, required=False)
            if result is None:
                continue
            evidence[name] = {
                'source': str(path.resolve()),
                'deterministic_algorithms': result.get('deterministic_algorithms'),
                'max_abs_diff': result.get('max_abs_diff'),
                'max_height_diff_m': result.get('max_height_diff_m'),
                'run_successes': [sum(bool(e.get('success')) for e in run)
                                  for run in result.get('runs', [])],
                'run_lanes': [len(run) for run in result.get('runs', [])],
            }
            trace_path = path.with_name('traces.npz')
            if trace_path.is_file():
                reader.record(trace_path, trace_path.read_bytes())
                with np.load(trace_path, allow_pickle=False) as data:
                    traces = data['traces']
                    if traces.ndim != 4 or traces.shape[0] != 3:
                        reader.warnings.append(f'{trace_path}: unexpected probe trace shape {traces.shape}')
                        continue
                    difference = np.abs(traces[0].astype(float) - traces[2].astype(float))
                    entry = evidence[name]
                    entry['trace_values_finite'] = bool(np.isfinite(traces).all())
                    entry['first_last_nominal_array_equal'] = bool(np.array_equal(traces[0], traces[2]))
                    entry['verified_max_abs_diff'] = float(difference.max())
                    entry['verified_max_height_diff_m'] = float(difference[..., 0].max())
                    entry['trace_shape'] = list(traces.shape)
                    if entry['verified_max_abs_diff'] != entry['max_abs_diff']:
                        reader.warnings.append(f'{path}: recorded max difference disagrees with trace')
    return {'record': revision, 'restoration_probe_evidence': evidence,
            'cross_version_causal_attribution_allowed': False,
            'interpretation': 'Deterministic solver, full checkpoint reset, and exploration standard deviation changed together; not a one-factor standard-deviation ablation.'}


def summarize_latency(reader, training):
    final_path = reader.root / 'cpu_latency_final_model.json'
    legacy_path = reader.root / 'cpu_latency.json'
    selected = final_path if final_path.is_file() else legacy_path
    data = reader.json(selected, required=False)
    if data is None:
        return None
    is_final = selected == final_path
    expected = training['ppo_tactile_seed0'].get('final_checkpoint_sha256')
    measured = data.get('model_sha256', data.get('checkpoint_sha256'))
    matches = bool(expected and measured == expected)
    if is_final and expected and not matches:
        reader.warnings.append('cpu_latency_final_model.json checkpoint does not match current tactile final model')
    return {'source': str(selected.resolve()), 'is_final_model_measurement': is_final,
            'checkpoint_matches_current_tactile_final': matches, 'data': data}


def summarize_contact_diagnostic(reader, evaluation, traces):
    path = reader.root / 'contact_continuity_200g/diagnostic.json'
    data = reader.json(path, required=False)
    if data is None:
        return None
    entry = {'source': str(path.resolve()), 'data': data, 'lane0_event_neighbours': [],
             'interpretation': 'One sampled loss of two-finger support is a strict failure; '
             'it does not establish a dropped object or distinguish solver transients from physical slip.'}
    trace_path = path.with_name('physical_trace_60hz.npz')
    if not trace_path.is_file():
        reader.warnings.append('Contact diagnostic exists but its 60 Hz physical trace is missing')
        return entry
    reader.record(trace_path, trace_path.read_bytes())
    try:
        with np.load(trace_path, allow_pickle=False) as raw:
            times, height, load = [raw[k] for k in
                ('absolute_time_s', 'clearance_m', 'contact_load_N')]
        if (times.ndim != 1 or height.shape[0] != len(times)
                or load.shape != (*height.shape, 5)
                or not all(np.isfinite(x).all() for x in (times, height, load))):
            raise ValueError('Invalid contact diagnostic trace shapes/values')
        interval = evaluation['suite_200g'].get('hold_interval_s')
        if interval is not None:
            hold = (times > interval[0]) & (times - 1 / data['sample_hz'] < interval[1])
            entry['verified_hold_samples'] = int(hold.sum())
            entry['verified_hold_min_height_per_lane_m'] = height[hold].min(0).tolist()
            for threshold in data.get('threshold_diagnostic', []):
                counts = ((load[hold] > threshold['contact_load_threshold_N']).sum(-1) < 2).sum(0).tolist()
                if counts != threshold['hold_samples_with_fewer_than_two_sensor_links_per_lane']:
                    reader.warnings.append('Contact diagnostic threshold counts disagree with its 60 Hz trace')
        benchmark = traces.get('suite_200g', {}).get('ppo_tactile')
        if benchmark is not None:
            sampled = height[3::4]
            if sampled.shape != benchmark['clearance_m'].shape:
                raise ValueError('Contact diagnostic does not align with benchmark decision cadence')
            diff = float(np.abs(sampled - benchmark['clearance_m']).max())
            entry['verified_height_trace_max_difference_from_benchmark_m'] = diff
            if diff != data.get('height_trace_max_difference_from_benchmark_m'):
                reader.warnings.append('Contact diagnostic height difference disagrees with original benchmark')
        for event in data.get('lane0_contact_events', []):
            matches = np.flatnonzero(np.isclose(times, event['absolute_time_s'], atol=1e-9, rtol=0))
            if len(matches) != 1:
                raise ValueError('Contact event cannot be uniquely located in 60 Hz trace')
            i = int(matches[0])
            entry['lane0_event_neighbours'].append([
                {'absolute_time_s': float(times[j]), 'clearance_m': float(height[j, 0]),
                 'contact_load_N': load[j, 0].tolist()}
                for j in range(max(0, i - 1), min(len(times), i + 2))])
    except (OSError, ValueError, KeyError) as exc:
        reader.warnings.append(f'Contact diagnostic trace verification failed: {exc}')
    return entry


def read_trace(reader, path):
    if not path.is_file():
        reader.missing.append(str(path.relative_to(reader.root)))
        return None
    try:
        # Avoid loading the much larger tactile arrays solely to draw a figure.
        with np.load(path, allow_pickle=False) as data:
            result = {key: data[key].copy() for key in
                      ('elapsed_sim_s', 'clearance_m', 'applied_gains')}
        time, height, gain = (result[k] for k in
                              ('elapsed_sim_s', 'clearance_m', 'applied_gains'))
        if time.ndim != 2 or height.shape != time.shape or gain.shape != (*time.shape, 2):
            raise ValueError(f'Unexpected time/height/gain shapes: {time.shape}, {height.shape}, {gain.shape}')
        if not all(np.isfinite(v).all() for v in result.values()):
            raise ValueError('Nonfinite trace values')
        if not np.all(np.diff(time, axis=0) > 0):
            raise ValueError('Trace times are not strictly increasing within each lane')
        return result
    except (OSError, ValueError, KeyError) as exc:
        reader.warnings.append(f'{path.relative_to(reader.root)}: {exc}')
        return None


def summarize_suite(reader, name, training):
    directory = reader.root / name
    raw = reader.json(directory / 'results.json')
    if raw is None:
        return {'status': 'missing', 'controllers': [],
                'missing_controllers': list(CONTROLLERS)}, {}
    if not isinstance(raw, list):
        reader.warnings.append(f'{name}/results.json must contain a list')
        return {'status': 'invalid', 'controllers': [],
                'missing_controllers': list(CONTROLLERS)}, {}
    protocol = reader.json(directory / 'environment/environment_protocol.json') or {}
    manifest = reader.json(directory / 'manifest.json') or {}
    status = reader.json(directory / 'status.json', required=False) or {}
    reset = reader.json(directory / 'reset_consistency.json', required=False)
    rows, traces = [], {}
    seen = set()
    for row in raw:
        controller = row.get('controller')
        if not controller or controller in seen:
            reader.warnings.append(f'{name}: missing/duplicate controller {controller!r}')
            continue
        seen.add(controller)
        episodes = row.get('episodes', [])
        known = [e for e in episodes if e.get('success') is not None]
        successes = sum(bool(e['success']) for e in known)
        if row.get('successes') != successes or row.get('conditions') != len(episodes):
            reader.warnings.append(f'{name}/{controller}: aggregate counts disagree with episode records')
        gains = [e['mean_gains'] for e in episodes if e.get('mean_gains') is not None]
        entry = {
            'controller': controller, 'label': LABELS.get(controller, controller),
            'episodes': len(episodes), 'known_success_labels': len(known),
            'successes': successes, 'success_fraction_descriptive_only': successes / len(known) if known else None,
            'height_only_completions': sum(e.get('hold_min_clearance_m') is not None
                and e['hold_min_clearance_m'] >= .05 and e.get('hold_duration_s', 0) >= 3 - 1e-6
                for e in episodes),
            'mean_return': finite_mean([e.get('return') for e in episodes]),
            'mean_contact_load_N': finite_mean([e.get('mean_contact_force_N') for e in episodes]),
            'mean_hold_min_clearance_m': finite_mean([e.get('hold_min_clearance_m') for e in episodes]),
            'minimum_hold_clearance_m_over_lanes': min(
                [e['hold_min_clearance_m'] for e in episodes if e.get('hold_min_clearance_m') is not None], default=None),
            'mean_gains_thumb_index': np.asarray(gains).mean(axis=0).tolist() if gains else None,
            'mean_slip_distance_proxy_m': finite_mean([e.get('slip_distance_proxy_m') for e in episodes]),
            'failure_criteria_counts_can_overlap': {
                'height_or_duration': sum(e.get('hold_min_clearance_m') is not None
                    and (e['hold_min_clearance_m'] < .05 or e.get('hold_duration_s', 0) < 3 - 1e-6)
                    for e in episodes),
                'contact_support': sum(e.get('hold_contact_ok') is False for e in episodes),
                'collision': sum(e.get('termination_reason') == 'collision' for e in episodes),
            },
            'per_lane': episodes,
        }
        trace = read_trace(reader, directory / controller / 'trace.npz')
        if trace is not None:
            traces[controller] = trace
            entry['trace'] = {'path': str((directory / controller / 'trace.npz').relative_to(reader.root)),
                              'steps': int(trace['elapsed_sim_s'].shape[0]),
                              'lanes': int(trace['elapsed_sim_s'].shape[1]),
                              'max_lane_clearance_spread_m': float(np.ptp(trace['clearance_m'], axis=1).max())}
        rows.append(entry)
    masses = protocol.get('mass_kg_per_lane', [])
    friction = protocol.get('friction_per_lane', [])
    unique_materials = set((round(float(m), 8), round(float(mu), 8))
                           for m, mu in zip(masses, friction))
    train_masses = [m for run in training.values() for m in run['mass_kg_per_lane']]
    overlap = [float(m) for m in masses if any(abs(m - train_m) <= 1e-7 for train_m in train_masses)]
    missing_controllers = [c for c in CONTROLLERS if c not in seen]
    reset_missing_fields = [] if reset is None else [k for k in RESET_TOLERANCES if k not in reset]
    reset_violations = {} if reset is None else {
        key: {'difference': reset[key], 'tolerance': tolerance}
        for key, tolerance in RESET_TOLERANCES.items() if key in reset
        and (not np.isfinite(reset[key]) or reset[key] > tolerance)}
    reset_failed = bool(reset_violations)
    reset_all_zero = (bool(all(reset[key] == 0 for key in RESET_TOLERANCES))
                      if reset is not None and not reset_missing_fields else None)
    if reset_missing_fields:
        reader.warnings.append(f'{name}: reset check fields missing: {reset_missing_fields}')
    if reset_failed:
        reader.warnings.append(f'{name}: repeatability_failed; repeated nominal differences exceed '
                               f"benchmark tolerances: {reset_violations}. "
                               'Treat controller comparisons as diagnostic, not validated matched evidence.')
    reset_outcomes = None
    row_by_name = {row['controller']: row for row in rows}
    if 'nominal' in row_by_name and 'nominal_reset_check' in row_by_name:
        first = row_by_name['nominal']['per_lane']
        last = row_by_name['nominal_reset_check']['per_lane']
        reset_outcomes = [{'lane': i, 'mass_kg': before.get('mass_kg'),
                           'success_before': before.get('success'), 'success_after': after.get('success'),
                           'hold_contact_before': before.get('hold_contact_ok'),
                           'hold_contact_after': after.get('hold_contact_ok')}
                          for i, (before, after) in enumerate(zip(first, last))
                          if before.get('success') != after.get('success')]
    scope = ('Newly sampled fixed mass lanes, one attempt per controller per lane; '
             'same object, friction and nominal initial geometry, not eight training seeds.'
             if name == 'suite_mixed' else
             'Repeated lanes of the same 200 g condition; not independent diverse conditions or a generalization rate.')
    return {
        'kind': 'deterministic_frozen_policy_matched_evaluation',
        'status': 'repeatability_failed' if reset_failed else status.get(
            'status', 'partial' if missing_controllers else 'awaiting_reset_check'),
        'matched_comparison_validated': bool(not reset_failed and reset is not None
                                             and not reset_missing_fields
                                             and status.get('status') == 'completed'),
        'scope': scope, 'seed': manifest.get('seed'),
        'lanes': len(masses), 'unique_material_conditions': len(unique_materials),
        'mass_kg_per_lane': masses, 'friction_per_lane': friction,
        'evaluation_masses_also_seen_in_training_kg': overlap,
        'start_time_s': protocol.get('start_time_s'),
        'hold_interval_s': protocol.get('hold_interval_s'),
        'missing_controllers': missing_controllers, 'controllers': rows,
        'reset_consistency': reset, 'reset_clearance_threshold_m': 1e-5,
        'reset_tolerances': RESET_TOLERANCES, 'reset_violations': reset_violations,
        'reset_all_recorded_differences_exactly_zero': reset_all_zero,
        'reset_success_label_changes': reset_outcomes,
        'checkpoints': manifest.get('checkpoints', {}),
        'success_contract': protocol.get('success'),
    }, traces


def build_plot(root, report, traces):
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    figure, axes = plt.subplots(2, 2, figsize=(15, 9.2), constrained_layout=True)
    ax = axes[0, 0]
    for name, label, color in [('ppo_tactile_seed0', 'Tactile PPO', '#1764ab'),
                                ('ppo_no_tactile_seed0', 'No-tactile PPO', '#d87820')]:
        curve = report['training'][name]['episode_batch_curve']
        if not curve:
            continue
        x = np.array([row['timesteps'] for row in curve])
        y = np.array([row['mean_return'] for row in curve])
        ax.plot(x, y, label=label, color=color, linewidth=2)
        ax.fill_between(x, [r['min_return'] for r in curve], [r['max_return'] for r in curve],
                        color=color, alpha=.10)
    ax.set(title='Exploratory training: episode return', xlabel='Training transitions', ylabel='Return')
    ax.text(.02, .02, 'Bands: lane range, not confidence intervals', transform=ax.transAxes,
            fontsize=9, color='#555555')
    if ax.lines:
        ax.legend(loc='best')
    else:
        ax.text(.5, .5, 'Training records missing', ha='center', transform=ax.transAxes)
    ax.grid(alpha=.2)

    ax = axes[0, 1]
    mixed = report['evaluation']['suite_mixed']
    rows = [r for r in mixed['controllers'] if r['controller'] in PLOT_LABELS]
    if rows:
        names = [r['controller'] for r in rows]
        values = [r['success_fraction_descriptive_only'] or 0 for r in rows]
        heights = [r['height_only_completions'] / r['episodes'] if r['episodes'] else 0 for r in rows]
        positions = np.arange(len(rows))
        ax.bar(positions - .18, values, color=[COLORS[n] for n in names], width=.34)
        ax.bar(positions + .18, heights, facecolor='white', edgecolor=[COLORS[n] for n in names],
               hatch='//', width=.34, linewidth=1.2)
        ax.set_xticks(range(len(rows)), [PLOT_LABELS[n] for n in names], rotation=25, ha='right')
        for i, row in enumerate(rows):
            ax.text(i - .18, values[i] + .025, f"{row['successes']}/{row['known_success_labels']}", ha='center', fontsize=8)
            ax.text(i + .18, heights[i] + .025, f"{row['height_only_completions']}/{row['episodes']}", ha='center', fontsize=8)
        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(facecolor='#777777', label='Strict success'),
                           Patch(facecolor='white', edgecolor='#777777', hatch='//', label='Height only')],
                   loc='upper left', fontsize=8, ncol=2)
        ax.set_ylim(0, 1.2)
    else:
        ax.text(.5, .5, 'suite_mixed results missing', ha='center', transform=ax.transAxes)
    title = ('Mixed-mass diagnostics: reset repeatability FAILED'
             if mixed['status'] == 'repeatability_failed'
             else 'Mixed masses: strict success vs height-only completion')
    ax.set(title=title, ylabel='Observed success fraction')
    ax.grid(axis='y', alpha=.2)

    height_ax, gain_ax = axes[1]
    selected = traces.get('suite_200g', {})
    for name in PLOT_LABELS:
        trace = selected.get(name)
        if trace is None:
            continue
        time = trace['elapsed_sim_s'][:, 0]
        height_ax.plot(time, trace['clearance_m'][:, 0] * 100,
                       color=COLORS[name], label=PLOT_LABELS[name], linewidth=1.8)
        gain_ax.plot(time, trace['applied_gains'][:, 0].mean(axis=-1),
                     color=COLORS[name], label=PLOT_LABELS[name], linewidth=1.8)
    suite = report['evaluation']['suite_200g']
    start, interval = suite.get('start_time_s'), suite.get('hold_interval_s')
    if start is not None and interval is not None:
        for ax in (height_ax, gain_ax):
            ax.axvspan(interval[0] - start, interval[1] - start, color='#a5aaaf', alpha=.13)
    height_ax.axhline(5, color='black', linestyle='--', linewidth=.9, label='Height only: 5 cm')
    height_ax.set(title='200 g: cube bottom clearance (lane 0)', xlabel='Local episode time (s)', ylabel='Clearance (cm)')
    gain_ax.set(title='200 g: mean thumb/index gain (lane 0)', xlabel='Local episode time (s)', ylabel='Applied gain', ylim=(.95, 2.05))
    tactile_row = next((row for row in suite['controllers'] if row['controller'] == 'ppo_tactile'), None)
    if tactile_row:
        height_ax.text(.98, .03, f"Tactile PPO: height {tactile_row['height_only_completions']}/{tactile_row['episodes']}; "
                       f"strict {tactile_row['successes']}/{tactile_row['known_success_labels']}",
                       transform=height_ax.transAxes, ha='right', fontsize=9,
                       bbox={'facecolor': 'white', 'alpha': .85, 'edgecolor': 'none'})
    for ax in (height_ax, gain_ax):
        ax.grid(alpha=.2)
        if selected:
            ax.legend(loc='best', ncol=2, fontsize=8)
        else:
            ax.text(.5, .5, 'suite_200g traces missing', ha='center', transform=ax.transAxes)
    title = 'Residual tactile PPO pilot | training and evaluation are different evidence'
    if report['status'] == 'repeatability_failed':
        title += '\nRESET REPEATABILITY FAILED: evaluation curves are diagnostic records'
    else:
        title += '\nStrict success also requires contact support at every hold sample; height alone is insufficient'
    figure.suptitle(title, fontsize=14)
    figure.savefig(root / 'overview.png', dpi=160)
    plt.close(figure)


def markdown_report(report):
    lines = ['# 高频触觉夹持策略：训练与独立评估记录', '',
             f"生成时间：{report['generated_utc']}。报告状态：`{report['status']}`。", '',
             '本报告读取已有实验文件，不运行训练或物理仿真。训练阶段带探索；suite 中的模型冻结并使用确定性动作，两类数据分开解释。', '',
             '## 训练记录（不能代替独立评估）', '',
             '| 模型 | 状态 | transition 数 | 完成 episode | 训练中成功记录 | 最近最多32条平均回报 |',
             '| --- | --- | ---: | ---: | ---: | ---: |']
    if report['status'] == 'repeatability_failed':
        lines[4:4] = ['**固定控制器的重复运行未通过物理轨迹一致性检查。以下评估数字保留为调试记录，不能据此作正式配对归因或宣称触觉模型优势。**', '']
    completed_suites = [s for s in report['evaluation'].values() if s.get('matched_comparison_validated')]
    if len(completed_suites) == len(SUITES):
        highlights = ['## 本轮关键结果', '']
        for suite_name, label in [('suite_mixed', '8种混合质量条件'), ('suite_200g', '200 g同条件重复')]:
            suite = report['evaluation'][suite_name]
            rows_by_name = {row['controller']: row for row in suite['controllers']}
            if not all(name in rows_by_name for name in ('ppo_tactile', 'ppo_no_tactile', 'initial')):
                continue
            tactile, blind, initial = [rows_by_name[name] for name in ('ppo_tactile', 'ppo_no_tactile', 'initial')]
            highlights += [f"- **{label}**：触觉PPO严格成功 {tactile['successes']}/{tactile['episodes']}；"
                           f"未训练初始策略 {initial['successes']}/{initial['episodes']}；"
                           f"无触觉PPO {blind['successes']}/{blind['episodes']}。"]
            if tactile['height_only_completions'] == tactile['episodes'] and tactile['successes'] < tactile['episodes']:
                highlights += [f"  触觉PPO仅按高度/时长完成为 {tactile['height_only_completions']}/{tactile['episodes']}，"
                               f"全部lane中最低保持高度为 {tactile['minimum_hold_clearance_m_over_lanes'] * 100:.3f} cm。"
                               '严格失败来自完整保持区间内的接触支持条件，不能描述为这些物体全部掉落。']
        highlights += ['', '严格判据保持不变：高度达标之外，还要求每个60 Hz保持采样都有至少两个带触觉指节提供物理接触支持，且无规定碰撞故障。', '']
        position = lines.index('## 训练记录（不能代替独立评估）')
        lines[position:position] = highlights
    revision = report.get('experiment_revision')
    if revision is not None:
        record = revision['record']
        revision_lines = ['## 本轮版本变化与恢复验证', '',
                          '依据 `experiment_revision.json`，本轮同时改变了以下设置：', '']
        translations = {
            'Genesis deterministic solver selection': '开启 Genesis 确定性求解器选择。',
            'full SceneCheckpoint reset (rather than SimState)': '将精简 SimState 恢复换成完整 SceneCheckpoint 恢复。',
            'action standard deviation initialized at 0.1 instead of exp(-1)': '动作探索标准差的初始值由 exp(-1)≈0.368 改为0.1。'}
        revision_lines += [f'- {translations.get(change, change)}' for change in record.get('changes', [])]
        revision_lines += ['', '**这不是单因素探索标准差消融。与旧版相比的任何性能变化，都不能单独归因于降低动作噪声。**', '',
                           '触觉与无触觉模型仍应在本轮相同恢复方式、任务、奖励、质量范围、seed和训练预算下比较；旧版未通过重复性检查的suite仍作为调试记录保留。']
        evidence = revision['restoration_probe_evidence']
        deterministic_only = evidence.get('determinism_probe')
        full = evidence.get('determinism_full_checkpoint_probe')
        if deterministic_only:
            revision_lines += ['', '此前仅打开确定性算法的恢复探针，前后nominal最大高度差为 '
                               f"{number(deterministic_only['max_height_diff_m'], 9)} m。"]
        if full:
            if full.get('first_last_nominal_array_equal') and full.get('trace_values_finite'):
                shape = full['trace_shape']
                revision_lines += ['同开关加完整检查点的nominal→strong→nominal探针中，'
                                   f'前后nominal的{shape[1]}个时刻、{shape[2]}个lane、木块高度及逐指接触载荷**逐点完全相同**；'
                                   '本报告直接读取探针trace复核，最大绝对差为0。']
            else:
                revision_lines += ['完整检查点探针的原始逐点复核结果：`'
                                   + json.dumps(full, ensure_ascii=False) + '`。']
            revision_lines += ['这个探针验证恢复方式；本轮最终suite仍须各自通过末尾nominal复跑检查，不能用探针替代suite验收。']
        revision_lines += ['', '**当前仅为gain-only试验**：名义关节轨迹固定，策略只调整拇指/食指的两个夹持增益。'
                           '它不能证明主动改变手形、重新选择接触位置、多指规划或暂停抬升能力。', '']
        position = lines.index('## 训练记录（不能代替独立评估）')
        lines[position:position] = revision_lines
    for name in TRAIN_RUNS:
        run = report['training'][name]
        label = '触觉 PPO' if name == TRAIN_RUNS[0] else '无触觉 PPO（独立训练）'
        lines.append(f"| {label} | {run['status']} | {run['timesteps']} | {run['completed_episode_records']} | "
                     f"{run['training_successes']}/{run['episodes_with_success_label']} | {number(run['mean_return_last_32_episodes'])} |")
    lines += ['', '图中训练曲线按同一训练时刻完成的 episode 求均值，阴影为这些 lane 的最小/最大回报，不是置信区间。每个 lane 的质量在初始化时确定，reset 后不会获得一种新质量。', '',
              '## 冻结模型的确定性评估', '']
    for name, title in [('suite_mixed', '混合质量条件'), ('suite_200g', '200 g 条件')]:
        suite = report['evaluation'][name]
        lines += [f'### {title}', '', f"状态：`{suite['status']}`。"]
        if suite['status'] in ('missing', 'invalid'):
            lines += ['', f'`{name}/results.json` 尚不可用，不能由训练回报推断此评估结果。', '']
            continue
        lines += ['', f"记录包含 {suite['lanes']} 个 lane、{suite['unique_material_conditions']} 种质量/摩擦组合，评估 seed 为 {suite['seed']}。"]
        if suite['status'] == 'repeatability_failed':
            lines += ['**本组未通过 reset 重复性验收。相同名义动作重放产生不同物理结果；初始触觉/机器人状态数组相同，不能证明所有底层求解器状态相同。**']
        if name == 'suite_mixed':
            masses = '、'.join(f'{m * 1000:.2f}' for m in suite['mass_kg_per_lane'])
            lines += [f'质量分别为 {masses} g。每个固定条件、每个方法仅一次；物体形状、摩擦和名义初始几何相同。',
                      f"与训练质量在 1e-7 kg 容差内重叠的 lane 数：{len(suite['evaluation_masses_also_seen_in_training_kg'])}。新质量条件不等于新形状或新的训练种子。"]
        else:
            lines += ['这些 lane 是同一个200 g场景的并行重复，不能当作八种独立多样条件，或据此宣称100%的泛化成功率。曲线选择 lane 0 展示，完整逐lane结果保留在 report.json。']
        lines += ['', '| 方法 | 严格成功 | 仅高度/时长完成 | 平均局部回报 | 平均总接触载荷（N） | 各lane保持最低高度均值（cm） | 平均拇指/食指增益 |',
                  '| --- | ---: | ---: | ---: | ---: | ---: | --- |']
        for row in suite['controllers']:
            gain = row['mean_gains_thumb_index']
            gain_text = '/'.join(f'{x:.3f}' for x in gain) if gain else '缺失'
            height = row['mean_hold_min_clearance_m']
            lines.append(f"| {row['label']} | {row['successes']}/{row['known_success_labels']} | "
                         f"{row['height_only_completions']}/{row['episodes']} | {number(row['mean_return'])} | "
                         f"{number(row['mean_contact_load_N'])} | {number(None if height is None else height * 100)} | {gain_text} |")
        for row in suite['controllers']:
            failures = row['failure_criteria_counts_can_overlap']
            if failures['contact_support'] and not failures['height_or_duration']:
                lines += ['', f"{row['label']}：所有记录均达到高度/时长要求，但其中 "
                          f"{failures['contact_support']} 条未满足完整保持区间的双指接触支持。高度曲线高于5 cm不能单独解释为成功。"]
        if suite['missing_controllers']:
            lines += ['', '尚未完成的方法：' + '、'.join(suite['missing_controllers']) + '。']
        reset = suite.get('reset_consistency')
        if reset is not None:
            lines += ['', '固定1.0方法在全部方法之后再次运行，与首次运行的最大绝对差：`'
                      + json.dumps(reset, ensure_ascii=False) + '`。这是同控制器 reset 的一致性检查。']
            lines += [f"木块高度差验收阈值为 {suite['reset_clearance_threshold_m']} m；"
                      + ('检查失败。' if suite['status'] == 'repeatability_failed' else '请结合 suite 状态判断是否完整通过。')]
            if suite.get('reset_all_recorded_differences_exactly_zero'):
                lines += ['本组记录的高度、逐指接触载荷、触觉历史、机器人观测及前缀观测最大差均为0；前后nominal在这些记录上逐点复现。']
            for change in suite.get('reset_success_label_changes') or []:
                lines += [f"lane {change['lane']}（{change['mass_kg'] * 1000:.3f} g）的成功标签由 "
                          f"{change['success_before']} 变为 {change['success_after']}；保持接触支持由 "
                          f"{change['hold_contact_before']} 变为 {change['hold_contact_after']}。"]
        else:
            lines += ['', 'reset 的物理重复性检查尚未产出，暂不能声称检查通过。']
        lines.append('')
    diagnostic = report.get('contact_continuity_diagnostic')
    if diagnostic is not None:
        data = diagnostic['data']
        lines += ['## 200 g接触连续性诊断', '',
                  '额外重放冻结的触觉策略，读取60 Hz物理接触，不修改策略、奖励或本轮成功判据。'
                  f"与原benchmark在决策时刻的木块高度逐点最大差为 {number(diagnostic.get('verified_height_trace_max_difference_from_benchmark_m'), 9)} m。", '',
                  f"保持区间共有 {data['hold_samples']} 个与保持时间有交集的60 Hz检查样本。"
                  '原始逐采样记录与阈值诊断保存在 `contact_continuity_200g`；JSON及物理trace的SHA256记录在本报告JSON的 `source_files`。', '',
                  '| 接触载荷阈值 | 各lane未满足双指支持的采样数 |', '| ---: | --- |']
        for row in data.get('threshold_diagnostic', []):
            lines.append(f"| {row['contact_load_threshold_N']:g} N | "
                         + ' / '.join(str(x) for x in row['hold_samples_with_fewer_than_two_sensor_links_per_lane']) + ' |')
        for neighbours in diagnostic.get('lane0_event_neighbours', []):
            lines += ['', 'lane 0中该时刻及相邻采样：', '',
                      '| 示范绝对时间 | 木块最低点高度 | 拇指接触载荷 | 食指接触载荷 |', '| ---: | ---: | ---: | ---: |']
            for sample in neighbours:
                lines.append(f"| {sample['absolute_time_s']:.6f} s | {sample['clearance_m'] * 100:.4f} cm | "
                             f"{sample['contact_load_N'][0]:.4f} N | {sample['contact_load_N'][1]:.4f} N |")
        lines += ['', '这一检查中，31.666667 s的拇指载荷为0，仅食指记录到约1.6697 N，其他机器人link没有木块接触载荷；'
                  '前后相邻采样均恢复双指支持。四档阈值都在同一帧失败，不能简单归因于0.05 N阈值过高。', '',
                  '**当前严格判据将一次采样的短暂单指承载记为整回合失败；木块没有掉落，本轮仍保留0/8的严格成功结果。**'
                  '60 Hz记录只能证明这一采样时刻的物理接触支持中断，不能确定真实中断持续时间，'
                  '也尚不能区分求解器数值跳变与真实滑动；需要进一步检查更细时间分辨率的接触、相对运动及求解器状态。', '']
    latency = report.get('cpu_inference_latency')
    if latency is not None:
        data = latency['data']
        lines += ['## CPU策略推理耗时', '']
        if latency['is_final_model_measurement'] and latency['checkpoint_matches_current_tactile_final']:
            measurement = data['measurement']
            values = data['latency_ms']
            lines += ['优先采用 `cpu_latency_final_model.json`，测量checkpoint的SHA256与本轮触觉最终模型一致。', '',
                      f"单环境CPU推理，{measurement['torch_intraop_threads']}个PyTorch计算线程，"
                      f"排除{measurement['warmup_predictions']}次热身，计时{measurement['measured_predictions']}次。", '',
                      '| p50 | p95 | p99 | 最大观测值 |', '| ---: | ---: | ---: | ---: |',
                      f"| {values['p50']:.3f} ms | {values['p95']:.3f} ms | {values['p99']:.3f} ms | {values['max']:.3f} ms |", '',
                      '输入是此前保存的真实仿真观测，重复使用同一个输入测试计算路径；来源为v1的前缀触觉和机器人状态。'
                      'v1物理重复性失败不妨碍用该固定数组测计算耗时，但该观测不作为本轮抓取性能证据。',
                      '时间包含NumPy输入转换、策略编码与动作返回，不包含传感器采集、历史构建、通信、驱动或物理仿真。'
                      '共享主机未采用实时调度，因此这不是端到端闭环时延或最坏时延保证。', '']
        else:
            lines += ['当前只有历史测量，或测量checkpoint与本轮最终模型不匹配；不把该数字报告为本轮最终模型延迟。',
                      f"原始记录：`{latency['source']}`。", '']
    lines += ['## 指标解释与结论边界', '',
              '- 成功判据按60 Hz采样检查指定最终3秒：木块旋转后的最低点离桌至少5 cm、至少两个带触觉指节有物理接触，并且没有机器人桌面/自碰撞故障。',
              '- 接触载荷是物理接触力模长之和；不是标定后的指腹法向力，也不是触觉taxel数值之和。表格载荷均值覆盖整个局部episode，不能当作仅保持阶段的夹持力。',
              '- 增益为PD刚度倍数；高于1的增益不保证接触力按相同比例增加。2.0基线开始时也受到每次决策最多0.25的变化率限制。',
              '- 这只是固定名义轨迹下的两个夹持增益策略，尚未检验手形调整、主动接触探索或重抓能力。',
              '- 训练回报提高、价值函数拟合较好或某次抓取成功，都不能单独证明模型利用触觉取得了泛化收益。应与无触觉重训练、固定高增益及规则触觉基线比较。',
              '- 当前只有单个训练seed、同形状刚体和固定摩擦，触觉剪切又包含仿真相对速度项。结果用于最小流程验证，不代表真实XHand或更广材料范围。', '',
              '当前结果尚未建立触觉反馈的优势。下一步先核验接触连续性：定位支持中断的手指、持续时间、载荷阈值及其与高度保持的关系；保留当前严格判据，同时分开记录高度完成与接触连续性。'
              '随后设计需要反馈的任务，例如未预告的负载/摩擦变化、接触扰动或更宽的初始接触差异，再与固定增益和无触觉策略比较。不能先改成功定义再宣称策略改善。', '',
              f"![训练与评估概览]({Path(report['root']) / 'overview.png'})", '', '## 缺失项与读取告警', '']
    if not report['missing'] and not report['warnings']:
        lines += ['本次读取未发现必需文件缺失或解析告警。']
    else:
        lines += [f'- 缺失：`{item}`' for item in report['missing']]
        lines += [f'- 告警：{item}' for item in report['warnings']]
    lines += ['', '刷新命令：', '', '```bash',
              '.venv/bin/python -m ur7e_xhand_sim.residual_rl.report --root ' + report['root'],
              '```', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    reader = Reader(root)
    revision = summarize_revision(reader)
    training = {name: summarize_training(reader, name) for name in TRAIN_RUNS}
    latency = summarize_latency(reader, training)
    evaluation, traces = {}, {}
    for name in SUITES:
        evaluation[name], traces[name] = summarize_suite(reader, name, training)
    contact_diagnostic = summarize_contact_diagnostic(reader, evaluation, traces)
    completed = all(run['status'] == 'completed' for run in training.values()) and all(
        suite['status'] == 'completed' and not suite['missing_controllers'] for suite in evaluation.values())
    failed_repeatability = any(suite['status'] == 'repeatability_failed' for suite in evaluation.values())
    report = {'generated_utc': datetime.now(timezone.utc).isoformat(), 'root': str(root),
              'status': 'repeatability_failed' if failed_repeatability else (
                  'complete' if completed and not reader.missing and not reader.warnings else 'partial'),
              'experiment_revision': revision, 'cpu_inference_latency': latency,
              'contact_continuity_diagnostic': contact_diagnostic,
              'training': training, 'evaluation': evaluation,
              'missing': sorted(set(reader.missing)), 'warnings': reader.warnings,
              'source_files': reader.sources,
              'notes': ['Training uses exploration; evaluation uses frozen deterministic policies.',
                        'Repeated 200 g lanes are not independent diverse conditions.',
                        'Contact load is summed force magnitude, not calibrated normal grip force.']}
    build_plot(root, report, traces)
    for path, content in [(root / 'report.json', json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n'),
                          (root / 'REPORT.zh-CN.md', markdown_report(report))]:
        temporary = path.with_suffix(path.suffix + '.tmp')
        temporary.write_text(content)
        temporary.replace(path)
    print(json.dumps({'status': report['status'], 'report': str(root / 'report.json'),
                      'missing': report['missing'], 'warnings': report['warnings']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
