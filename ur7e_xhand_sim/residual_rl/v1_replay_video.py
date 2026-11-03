"""Record one collected base's actual V1 zero-residual evaluation, including its prefix.

Use one process per run so renderer and Genesis resources are released between
videos. The MP4 is rendered from the evaluator scene, before its automatic reset.
The prefix is followed by the same zero-velocity takeover reset used in training.
"""
import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

from .v1_bases import Base, sha256
from .v1_env import V1VecEnv, cpu, quaternion_matrix
from .v1_label import COLLECTION, record_check
from .v1_task import SamplerConfig, V1Config, hand_shape_joints
from .control_protocol import V1_RATES, current_control_protocol
from cameras import camera_kwargs, load_camera_views


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f'Not JSON serializable: {type(value).__name__}')


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=json_default, allow_nan=False)+'\n')


class VideoRecorder:
    """Own output streams independently of construction/destruction of Genesis."""
    def __init__(self, output, run):
        self.output, self.run = Path(output), run
        self.frames = 0
        self.prefix_frames = 0
        self.episode_frames = 0
        self.terminal_metrics = None
        self.writer = None
        self.metrics = (self.output/'frames.jsonl').open('w')

    def append(self, canvas, row):
        if self.writer is None:
            self.writer = imageio.get_writer(self.output/'replay.mp4', fps=15,
                codec='libx264', quality=8, macro_block_size=1)
        self.writer.append_data(np.asarray(canvas))
        row['frame_index'] = self.frames
        self.metrics.write(json.dumps(row, default=json_default, allow_nan=False)+'\n')
        self.frames += 1
        if row['segment'] == 'prefix':
            self.prefix_frames += 1
        else:
            self.episode_frames += 1
        if self.frames == 1:
            canvas.save(self.output/'first.png')
        if row['phase'] == 'hold':
            canvas.save(self.output/'held.png')
        if row.get('episode_metrics') is not None:
            self.terminal_metrics = row['episode_metrics']
            canvas.save(self.output/'last.png')

    def close(self):
        try:
            if self.writer is not None:
                self.writer.close()
                self.writer = None
        finally:
            self.metrics.close()


class RecordedV1Env(V1VecEnv):
    def __init__(self, *args, recorder, **kwargs):
        self.recorder = recorder
        super().__init__(*args, **kwargs)
        if self.num_envs != 1:
            raise ValueError('Video evaluation requires exactly one environment')

    def _configure_rendering(self):
        self.camera_views, self.camera_source = load_camera_views()
        self.video_cameras = {name: self.scene.add_camera(**camera_kwargs(self.camera_views[name]), env_idx=0)
                              for name in ('scene_front', 'scene_left', 'wrist')}
        self.video_cameras['detail'] = self.scene.add_camera(res=(480, 480),
            pos=(1, 1, 1), lookat=(0, 0, 0), fov=50, GUI=False, env_idx=0)

    def _on_prefix_step(self, lane_bases, tick):
        if tick % (self.rates.physics_hz//15) == 0:
            self._record_frame(lane_bases[0], tick, 'prefix')

    def _on_episode_step(self, done, infos):
        if self.episode[0]['discard']:
            return
        base = self.bases[self.lane_base[0]]
        result = infos[0].get('episode_metrics')
        if result is None and self.lane_tick[0] % (self.rates.physics_hz//15):
            return
        if result is not None:
            # U is only needed to construct an unlabelled Base for evaluation.
            # Do not expose this internal placeholder as an assigned data label.
            result = dict(result, base_label=None, internal_base_label=result['base_label'],
                          base_label_status='unlabelled; zero-residual evaluation only')
        self._record_frame(base, int(self.lane_tick[0]), 'zero_residual', result)

    def _record_frame(self, base, tick, segment, result=None):
        pos = cpu(self.palm.get_pos()).reshape(-1, 3)[0]
        quat = cpu(self.palm.get_quat()).reshape(-1, 4)[0]
        rotation = quaternion_matrix(quat)
        mount = self.camera_views['wrist']
        self.video_cameras['wrist'].set_pose(pos=pos+rotation@np.asarray(mount['position_m']),
            lookat=pos+rotation@np.asarray(mount['lookat_m']), up=rotation@np.asarray(mount['up']))
        center = pos+rotation@np.array([0., 0., .09])
        self.video_cameras['detail'].set_pose(pos=center+rotation@np.array([.14, .42, .10]),
            lookat=center, up=rotation@np.array([-1., 0., 0.]))
        t = tick/self.rates.physics_hz
        # Display the phase commanded at the beginning of this control interval.
        phase = str(base.trajectory.phase(max(0, tick-self.rates.action_stride)/self.rates.physics_hz))
        metrics = {key: value[0] for key, value in self._metrics().items()}
        slip = None if segment == 'prefix' else float(self.slip.distance[0])
        reset_notice = segment == 'zero_residual' and self.recorder.episode_frames < 15
        canvas = Image.new('RGB', (1280, 960), '#121922')
        draw = ImageDraw.Draw(canvas)
        for name, label, xy in [('scene_front', 'FRONT', (0, 40)),
                               ('scene_left', 'LEFT', (640, 40)),
                               ('wrist', 'WRIST', (0, 500)),
                               ('detail', 'HAND DETAIL', (640, 500))]:
            rgb = cpu(self.video_cameras[name].render(rgb=True)[0]).squeeze().astype('uint8')
            im = Image.fromarray(rgb)
            im.thumbnail((632, 414))
            canvas.paste(im, (xy[0]+(640-im.width)//2, xy[1]+26))
            draw.text((xy[0]+12, xy[1]+5), label, fill='white')
        title = f'{base.base_id} | {phase} | t={t:.2f}s | residual=0 | 1x simulation time'
        draw.text((16, 10), title, fill='white')
        if reset_notice:
            draw.text((16, 27), 'V1 takeover: replayed pose restored, velocities reset to zero', fill='#ffda78')
        elif segment == 'prefix':
            draw.text((16, 27), 'PREFIX: approach before residual-control takeover', fill='#bed8ff')
        else:
            draw.text((16, 27), 'OFFICIAL V1 ZERO-RESIDUAL EVALUATION', fill='#bed8ff')
        slip_text = 'not started' if slip is None else f'{1000*slip:.2f} mm'
        status = '' if result is None else f" | Result: {result['termination_reason']}"
        draw.text((16, 932), f"Clearance: {1000*metrics['clearance_m']:.1f} mm | Slip: {slip_text}{status}", fill='white')
        row = dict(segment=segment, physics_tick=tick, trajectory_time_s=t, phase=phase,
            base_id=base.base_id, grip_gain=float(self.gains[0]), metrics=metrics,
            slip_distance_m=slip, takeover_reset_notice=reset_notice,
            joint_position_rad=cpu(self.robot.get_dofs_position(self.indices)).reshape(-1, 18)[0],
            residual_rad=None if segment == 'prefix' else self.residual[0].copy(),
            episode_metrics=result)
        self.recorder.append(canvas, row)
        if self.recorder.frames % 90 == 0 or result is not None:
            print(f'{base.base_id}: {self.recorder.frames} frames, {phase}, t={t:.2f}s'+status, flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--collection', type=Path, required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args(argv)
    collection, output = args.collection.expanduser().resolve(), args.output.expanduser().resolve()
    runs_path = collection/'runs.jsonl'
    runs = [json.loads(line) for line in runs_path.open()]
    matches = [run for run in runs if run['run_id'] == args.run_id]
    if len(matches) != 1:
        raise ValueError(f'Expected exactly one run named {args.run_id}, found {len(matches)}')
    run = matches[0]
    demo_path = Path(run['demo'])
    check = record_check(run, json.loads(demo_path.read_text()))
    if not check['usable']:
        raise ValueError(f'Collected run is unusable: {check}')
    base = Base(dict(base_id=run['run_id'], trajectory_path=run['trajectory_path'],
        trajectory_sha256=run['trajectory_sha256'], object_initial_position_m=run['object_initial_position_m'],
        object_initial_quaternion_wxyz=run['object_initial_quaternion_wxyz'], label='U', split='test'), collection)
    output.mkdir(parents=True, exist_ok=False)
    source_files = [runs_path, demo_path, base.path, base.path.with_name('trajectory_metadata.json'),
                    Path(__file__), Path(__file__).with_name('v1_env.py'),
                    Path(__file__).with_name('v1_label.py'), Path(__file__).with_name('v1_task.py'),
                    Path(__file__).with_name('core.py')]
    provenance = dict(run_id=base.base_id, original_split=run['split'], hand_shape=run['hand_shape'],
        collection=str(collection), trajectory_sha256=base.trajectory.sha256, record_check=check,
        collected_success=bool((run.get('score') or {}).get('success')),
        internal_base_label='U', internal_base_split='test',
        internal_base_note='Temporary evaluator eligibility only; original split is preserved and no U label is assigned.',
        source_hashes={str(path.resolve()): sha256(path) for path in source_files if path.is_file()},
        replay_protocol='V1VecEnv, original initial pose, physical prefix, zero-velocity takeover reset, then zero residual.',
        physics_hz=V1_RATES.physics_hz, tactile_hz=V1_RATES.tactile_hz,
        action_hz=V1_RATES.action_hz, control_protocol=current_control_protocol(), video_fps=15,
        mass_kg=.05, friction=.8, unbreakable=True, recorded_grip_gain=True,
        joint_friction_Nm=0., tactile_noise_N=0., augment=False, seed=args.seed)
    write_json(output/'provenance.json', provenance)
    recorder = VideoRecorder(output, run)
    env = None
    try:
        env = RecordedV1Env(run['object_spec'], [base], n_envs=1, seed=args.seed,
            output_dir=output/'environment', object_id=run['object_id'], hand_shape=run['hand_shape'],
            action_joints=hand_shape_joints(run['hand_shape']), config=V1Config(),
            sampler=SamplerConfig(levels=(4,)), fixed_gain=None, augment=False, recorder=recorder)
        env.set_episode_specs([dict(COLLECTION, base_id=base.base_id)])
        env.reset()
        maximum_steps = (base.end_tick-base.takeover_tick)//env.rates.action_stride+2
        for _ in range(maximum_steps):
            _, _, done, infos = env.step(np.zeros((1, len(env.action_joints)), np.float32))
            if done[0]:
                if recorder.terminal_metrics is None:
                    raise RuntimeError('Terminal evaluation was not captured before reset')
                break
        else:
            raise RuntimeError('Evaluation did not terminate by its recorded horizon')
        result = recorder.terminal_metrics
        if result['mean_abs_residual_rad'] != 0.:
            raise RuntimeError('Evaluation unexpectedly contains a nonzero residual')
        summary = dict(provenance, status='completed', episode_metrics=result,
            success=bool(result['success']), label=None, label_status='zero-residual replay complete; labelling separate',
            frames=recorder.frames, prefix_frames=recorder.prefix_frames, episode_frames=recorder.episode_frames,
            video_duration_s=recorder.frames/15, video=str(output/'replay.mp4'),
            camera_source=env.camera_source, solver_profile=env.solver_profile,
            solver_readback=env.solver_readback, prefix_contact_N=env.takeover_states[0]['prefix_contact_N'],
            replay_collection_agree=bool(result['success']) == provenance['collected_success'])
        recorder.close()
        summary['video_sha256'] = sha256(output/'replay.mp4')
        summary['frames_sha256'] = sha256(output/'frames.jsonl')
        write_json(output/'validation.json', summary)
        print(json.dumps(summary, default=json_default, allow_nan=False), flush=True)
    except Exception as exc:
        write_json(output/'error.json', dict(error=repr(exc), run_id=base.base_id, captured_frames=recorder.frames))
        raise
    finally:
        recorder.close()
        if env is not None:
            env.close()


if __name__ == '__main__':
    main()
