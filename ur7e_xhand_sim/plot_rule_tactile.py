"""Plot recorded tactile observations without advancing the simulation."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run_dir', type=Path)
    args = parser.parse_args()
    out = args.run_dir / 'tactile_analysis'
    out.mkdir(exist_ok=True)
    fingers = ['thumb', 'index', 'middle', 'ring', 'pinky']
    labels = ['拇指', '食指', '中指', '无名指', '小指']
    rows = [json.loads(line) for line in (args.run_dir / 'control.jsonl').open()]
    evaluations = [json.loads(line) for line in (args.run_dir / 'evaluation.jsonl').open()]
    assert [r['step'] for r in rows] == [r['step'] for r in evaluations]
    t = np.array([r['after']['sim_time_s'] for r in rows])
    raw = np.array([[r['after']['tactile'][f]['force_sensor_N'] for f in fingers] for r in rows])
    assert raw.shape == (len(rows), 5, 120, 3)
    magnitude = np.linalg.norm(raw, axis=-1)
    peak = magnitude.max(axis=-1)
    active = (magnitude > 1e-5).sum(axis=-1)
    logged = np.array([[r['after']['tactile'][f]['peak_force_N'] for f in fingers] for r in rows])
    assert np.allclose(peak, logged, atol=1e-7)
    summed = raw.sum(axis=2)
    clearance = np.array([r['bottom_clearance_m'] for r in evaluations])
    phases = np.array([r['phase'] for r in rows])
    np.savez_compressed(out/'tactile_series.npz', time_s=t, fingers=fingers,
                        force_sensor_N=raw, peak_taxel_N=peak, active_taxels=active,
                        sum_sensor_xyz_N=summed, phase=phases, bottom_clearance_m=clearance)
    stats = {'samples':len(t), 'sample_interval_s':float(np.median(np.diff(t))),
             'units':'uncalibrated simulation nominal N; peaks are single-taxel magnitudes, not grip force',
             'first_threshold_crossing_s':{}, 'hold':{}, 'stage_snapshots':[]}
    hold = phases == 'hold'
    for i, f in enumerate(fingers):
        hits = np.where(peak[:,i] >= .01)[0]
        stats['first_threshold_crossing_s'][f] = float(t[hits[0]]) if len(hits) else None
        stats['hold'][f] = {'peak_taxel_min_N':float(peak[hold,i].min()),
                            'peak_taxel_max_N':float(peak[hold,i].max()),
                            'active_taxels_min':int(active[hold,i].min()),
                            'active_taxels_max':int(active[hold,i].max()),
                            'samples_above_contact_threshold':int((peak[hold,i]>=.01).sum())}
    for stage in json.loads((args.run_dir/'stages.json').read_text()):
        stats['stage_snapshots'].append({'name':stage['name'], 'time_s':stage['step']*.025,
            'peak_taxel_N':{f:stage['observation']['tactile'][f]['peak_force_N'] for f in fingers}})
    (out/'summary.json').write_text(json.dumps(stats, indent=2))
    font = FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    plt.rcParams.update({'font.family':font.get_name(), 'axes.unicode_minus':False,
                         'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False})
    colors=['#d16a16','#1b75b4','#26946a','#9755a2','#737d87']
    stage_names={'close':'闭合','preload':'预紧','test_lift':'试抬','lift':'抬升','hold':'保持'}
    spans=[]
    for name, label in stage_names.items():
        ids=np.where(phases==name)[0]
        spans.append((t[ids[0]]-.025,t[ids[-1]],label))
    def decorate(ax, label=False):
        for i,(a,b,text) in enumerate(spans):
            ax.axvspan(a,b,color=['#e5ecf2','#fae7cf','#d9eee5','#dee8fa','#e9e0f4'][i],alpha=.45,zorder=0)
            if label:ax.text((a+b)/2,1.02,text,ha='center',transform=ax.get_xaxis_transform(),fontsize=10)
        ax.grid(alpha=.2); ax.set_xlim(16.5,t[-1])
    fig,axs=plt.subplots(4,1,figsize=(12,10),gridspec_kw={'height_ratios':[1,1.3,1,1]},layout='constrained')
    for i in range(5):
        axs[0].plot(t,peak[:,i],color=colors[i],label=labels[i],lw=1.3,linestyle='-' if i<2 else ['--',':','-.'][i-2])
    axs[0].set(xlim=(0,t[-1]),ylabel='单测点峰值（名义 N）',title='全过程：接近前触觉为零，接触后以拇指与食指为主')
    axs[0].legend(ncol=5,loc='upper left');axs[0].grid(alpha=.2)
    for i in range(2):
        axs[1].plot(t,peak[:,i],color=colors[i],label=labels[i],lw=1.3)
        axs[2].step(t,active[:,i],where='post',color=colors[i],label=labels[i],lw=1.3)
    axs[1].axhline(.01,color='#737d87',ls='--',lw=1,label='接触阈值 0.01')
    axs[1].set_ylabel('单测点峰值（名义 N）');axs[1].legend(loc='upper right',ncol=3)
    axs[2].set_ylabel('激活测点数 / 120')
    axs[3].plot(t,clearance*100,color='#33485f',lw=1.5)
    axs[3].axhline(5,color='#737d87',ls='--',lw=1)
    axs[3].set(ylabel='木块底面离桌（cm）',xlabel='仿真时间（秒）')
    for ax in axs[1:]:decorate(ax,ax is axs[1])
    fig.suptitle('成功示范的触觉变化 · 40 Hz 原始记录，无平滑\n传感器几何近似，未实机标定；曲线不是整指总夹持力',fontsize=14)
    fig.savefig(out/'tactile_timeline.png',dpi=160);fig.savefig(out/'tactile_timeline.pdf');plt.close(fig)
    fig,axs=plt.subplots(2,2,figsize=(12,7),layout='constrained')
    for i in range(2):
        im=axs[i,0].imshow(magnitude[:,i].T,origin='lower',aspect='auto',interpolation='nearest',
                           extent=[t[0]-.0125,t[-1]+.0125,.5,120.5],cmap='magma',vmin=0)
        axs[i,0].set(title=labels[i]+'：120 个测点幅值',ylabel='测点编号（非空间排列）',xlim=(16.5,t[-1]))
        fig.colorbar(im,ax=axs[i,0],label='名义 N')
        for j,axis in enumerate('XYZ'):axs[i,1].plot(t,summed[:,i,j],label=axis,lw=1.1)
        axs[i,1].set(title=labels[i]+'：传感器本体坐标三轴求和',ylabel='Σ 测点分量（名义 N）')
        decorate(axs[i,1]);axs[i,1].legend(ncol=3)
    for ax in axs[1]:ax.set_xlabel('仿真时间（秒）')
    fig.suptitle('触觉细节：接触分布与三轴变化\n各指传感器坐标轴不同；不可将 XYZ 直接视为统一世界坐标的法向／切向',fontsize=13)
    fig.savefig(out/'tactile_details.png',dpi=160);plt.close(fig)
    print(json.dumps(stats,indent=2))


if __name__=='__main__':main()
