"""Offline plots and scene-synchronized tactile video from a recorded run."""
import argparse
import json
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    episode = run / 'episode_000'
    out = run / 'tactile_results'
    out.mkdir(exist_ok=True)
    font = FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    plt.rcParams.update({'font.family': font.get_name(), 'axes.unicode_minus': False,
                         'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    data = np.load(episode / 'tactile.npz', allow_pickle=False)
    description = json.loads((run / 'description.json').read_text())
    layout = description['tactile_layout']
    dt = description['physics_dt_s']
    control_dt = description['control_dt_s']
    groups = [('thumb', '拇指'), ('index', '食指'), ('middle_finger', '中指'),
              ('ring', '无名指'), ('pinky', '小指'), ('wrist', '掌部')]
    names = [str(n) for key, _ in groups for n in data['sensor_names'] if key in str(n)]
    assert len(names) == len(set(names)) == 16
    histories = [data[n] for n in names]
    # Genesis stores newest-first. Reverse each completed 5-substep block.
    raw = np.concatenate(histories, axis=2)
    signal = np.concatenate([raw[:1, 0], raw[1:, ::-1].reshape(-1, raw.shape[2], 6)], axis=0)
    times = np.arange(len(signal)) * dt
    np.testing.assert_allclose(times[::5], data['sim_time_s'], atol=1e-10)
    assert signal.shape == (661, 140, 6) and np.isfinite(signal).all()
    force = np.linalg.norm(signal[..., :3], axis=-1)
    torque = np.linalg.norm(signal[..., 3:], axis=-1)
    slices, cursor = {}, 0
    for name in names:
        count = data[name].shape[2]
        slices[name] = slice(cursor, cursor + count)
        cursor += count
    indices = {label: np.concatenate([np.arange(slices[n].start, slices[n].stop)
                                     for n in names if key in n]) for key, label in groups}
    colors = plt.get_cmap('tab10').colors
    stats = []
    for i, (label, ids) in enumerate(indices.items()):
        active = (force[:, ids] > 1e-5).sum(axis=1)
        contact = np.flatnonzero(active)
        stats.append({'region': label, 'taxels': len(ids), 'ever_active_taxels': int((force[:, ids] > 1e-5).any(axis=0).sum()),
                      'max_active_taxels': int(active.max()), 'first_active_time_s': float(times[contact[0]]) if len(contact) else None,
                      'peak_single_taxel_force_N': float(force[:, ids].max()),
                      'peak_single_taxel_torque_Nm': float(torque[:, ids].max())})
    report = {'source': str(episode / 'tactile.npz'), 'sample_rate_Hz': 1/dt, 'samples': len(times),
              'duration_s': float(times[-1]), 'components': data['components'].tolist(), 'regions': stats,
              'interpretation': 'KinematicTaxel estimates; may include self-contact. Not solver net forces or calibrated hardware forces.',
              'history_order': 'source newest-first; plotted chronologically at 200 Hz; original Astra summaries used index -1 (20 ms older).'}
    (out / 'tactile_summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    np.savez_compressed(out / 'tactile_200hz.npz', time_s=times, force_torque=signal,
                        sensor_names=np.array(names), sensor_offsets=np.array([slices[n].start for n in names]),
                        components=data['components'])

    fig, axes = plt.subplots(4, 1, figsize=(14, 13), sharex=True,
                             gridspec_kw={'height_ratios': [2, 1, 1, 1]}, layout='constrained')
    fig.suptitle('Astra 演示对应的触觉仿真结果 · 种子 42 · 200 Hz 完整记录', fontsize=18, weight='bold')
    im = axes[0].imshow(force.T, origin='lower', aspect='auto', extent=[0, times[-1], -.5, 139.5],
                        interpolation='nearest', cmap='magma', vmin=0, vmax=2)
    mids = []
    for label, ids in indices.items():
        mids.append((ids[0] + ids[-1])/2)
        axes[0].axhline(ids[-1]+.5, color='white', lw=.8, alpha=.7)
    axes[0].set_yticks(mids, list(indices)); axes[0].set_title('每个触觉点的力模长 |F|（140 个点，按传感器索引排列）', loc='left')
    fig.colorbar(im, ax=axes[0], label='N', pad=.01)
    for i, (label, ids) in enumerate(indices.items()):
        axes[1].plot(times, (force[:, ids] > 1e-5).sum(axis=1), color=colors[i], label=label, lw=1.2)
        axes[2].plot(times, force[:, ids].max(axis=1), color=colors[i], lw=1)
        axes[3].plot(times, torque[:, ids].max(axis=1), color=colors[i], lw=1)
    axes[1].set_ylabel('激活触觉点数量'); axes[1].legend(ncol=6, loc='upper left')
    axes[2].set_ylabel('区域内最大 |F| / N'); axes[3].set_ylabel('区域内最大 |T| / N·m')
    axes[3].set_xlabel('仿真时间 / 秒（不包含 Astra 决策等待）')
    for ax in axes[1:]:
        ax.grid(alpha=.2); ax.set_ylim(bottom=0)
    axes[0].set_xlim(0, times[-1])
    fig.supxlabel('力/力矩为上游弹簧阻尼估计；可能包含自接触。区域最大值不是区域合力；力矩含扭转阻尼项。', fontsize=10)
    fig.savefig(out / 'tactile_overview.png', dpi=150)
    fig.savefig(out / 'tactile_overview.pdf')
    plt.close(fig)

    chosen = [n for n in names if ('tip_sensor_frame' in n and any(k in n for k in ['thumb','index','middle_finger'])) or n.endswith('r_wrist')]
    fig, axes = plt.subplots(len(chosen), 2, figsize=(14, 11), sharex=True, layout='constrained')
    fig.suptitle('六维原始信号示例 · 每个区域固定一个触觉点，不逐帧切换', fontsize=17, weight='bold')
    selected = []
    for row, name in enumerate(chosen):
        region = force[:, slices[name]]
        index = int(np.unravel_index(region.argmax(), region.shape)[1])
        values = signal[:, slices[name].start + index]
        label = next(label for key, label in groups if key in name) + ('指尖' if 'tip' in name else '')
        for col in range(2):
            for c, component in enumerate(['x', 'y', 'z']):
                axes[row,col].plot(times, values[:, 3*col+c], label=('F' if col==0 else 'T')+component, lw=.9)
            axes[row,col].set_title(f'{label} · taxel {index}（从 0 开始）', loc='left')
            axes[row,col].set_ylabel('N' if col==0 else 'N·m'); axes[row,col].grid(alpha=.2)
        selected.append({'sensor': name, 'taxel_index': index})
    axes[0,0].legend(ncol=3); axes[0,1].legend(ncol=3)
    axes[-1,0].set_xlabel('仿真时间 / 秒'); axes[-1,1].set_xlabel('仿真时间 / 秒')
    fig.supxlabel('所有分量均在各自附着 link 的局部坐标系中；不同 link 的 XYZ 方向不能直接相加。', fontsize=10)
    fig.savefig(out / 'tactile_six_components.png', dpi=150)
    plt.close(fig)
    (out / 'selected_taxels.json').write_text(json.dumps(selected, ensure_ascii=False, indent=2)+'\n')

    # All 16 sensor grids at native control timestamps, next to the exact scene frame.
    fig = plt.figure(figsize=(16, 9), dpi=100, layout='constrained')
    grid = fig.add_gridspec(1, 2, width_ratios=[1, 1.3])
    scene_ax = fig.add_subplot(grid[0,0]); scene_ax.axis('off')
    scene = scene_ax.imshow(Image.open(episode/'frames/000000.png'))
    tilegrid = grid[0,1].subgridspec(4,4)
    tile_images = []
    for i, name in enumerate(names):
        ax = fig.add_subplot(tilegrid[i//4,i%4])
        shape = np.asarray(layout[name]['normals_link']).shape[:-1]
        assert np.prod(shape) == data[name].shape[2]
        tile = ax.imshow(np.zeros(shape), vmin=0, vmax=2, cmap='magma', interpolation='nearest', aspect='equal')
        label = next(label for key,label in groups if key in name)
        part = '指尖' if 'tip_sensor' in name else '近节' if 'proximal' in name else '中节' if not name.endswith('wrist') else ''
        ax.set_title(label + part, fontsize=11); ax.set_xticks([]); ax.set_yticks([])
        tile_images.append((tile,name,shape))
    fig.colorbar(tile_images[0][0], ax=fig.axes[1:], label='单个触觉点 |F| / N', shrink=.75, pad=.02)
    title = fig.suptitle('', fontsize=17, weight='bold')
    fig.supxlabel('传感器原始索引网格，非手部空间投影 · 140 taxels · 8 倍慢放 · 仿真时间不含决策等待', fontsize=11)
    width,height=fig.canvas.get_width_height()
    command=['ffmpeg','-y','-loglevel','error','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24',
             '-s',f'{width}x{height}','-r',str(1/control_dt/8),'-i','-','-an','-c:v','libx264',
             '-pix_fmt','yuv420p','-r','40','-crf','20',str(out/'scene_tactile_8x_slow.mp4')]
    proc=subprocess.Popen(command,stdin=subprocess.PIPE)
    for step,t in enumerate(data['sim_time_s']):
        scene.set_data(Image.open(episode/'frames'/f'{step:06d}.png'))
        for tile,name,shape in tile_images:
            tile.set_data(np.linalg.norm(data[name][step,0,:,:3],axis=-1).reshape(shape))
        title.set_text(f'场景与触觉同步回放  |  t = {t:.3f} s  |  第 {step} 控制步')
        fig.canvas.draw()
        proc.stdin.write(np.asarray(fig.canvas.buffer_rgba())[...,:3].tobytes())
        if step in [0,64,132]:
            fig.savefig(out/f'synchronized_{step:03d}.png', dpi=100)
    proc.stdin.close()
    assert proc.wait()==0
    plt.close(fig)
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
