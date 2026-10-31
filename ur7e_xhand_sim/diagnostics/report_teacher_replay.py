"""CPU-only evidence report for freshly sensed, frozen-command teacher replays.

Writes only DATASET/report/. Source runs and old raw tactile records are read
only. Metrics come from evaluation contacts and raw taxels, not score summaries.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import html
import json
import os
from pathlib import Path
import re
from urllib.parse import quote

os.environ.setdefault('MPLCONFIGDIR', '/tmp/tactile-replay-report-matplotlib')
import numpy as np


SHAPES = ('cuboid', 'hex_prism', 'sphere', 'cylinder')
LABELS = dict(cuboid='长方体', hex_prism='六棱柱', sphere='球体', cylinder='直立圆柱')
FINGERS = ('thumb', 'index', 'middle', 'ring', 'pinky')
FINGER_LABELS = dict(zip(FINGERS, ('拇指', '食指', '中指', '无名指', '小指')))
LINKS = dict(zip(FINGERS, ('right_hand_thumb_rota_link2', 'right_hand_index_rota_link2',
                         'right_hand_mid_link2', 'right_hand_ring_link2', 'right_hand_pinky_link2')))
ACTIVE_THRESHOLD = 1e-6
LOAD_THRESHOLD = 1e-8


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def canonical_finger(name):
    return 'pinky' if name in ('little', 'pinky') else name


def raw_key(raw, *names):
    for name in names:
        if name in raw:
            return np.asarray(raw[name])
    raise ValueError('Raw NPZ missing '+ '/'.join(names))


def parse_series(source, manifest, scene):
    evaluation=[json.loads(line) for line in (source/'evaluation.jsonl').read_text().splitlines() if line.strip()]
    if not evaluation:
        raise ValueError('No evaluation samples')
    time=np.asarray([row['sample_time_s'] for row in evaluation], float)
    phases=np.asarray([row.get('phase', '') for row in evaluation])
    clearance=np.asarray([row['bottom_clearance_m'] for row in evaluation], float)
    with np.load(source/'raw_tactile.npz', allow_pickle=False) as raw:
        tactile=raw_key(raw, 'tactile')
        raw_time=raw_key(raw, 'time', 'sample_time_s', 'time_s')
        gain=raw_key(raw, 'gain', 'grip_gain')
        q=raw_key(raw, 'q', 'joint_positions_rad', 'actual_joint_rad')
        target=raw_key(raw, 'target', 'sent_target_rad', 'target_rad')
    if tactile.shape!=(len(time), 5, 120, 3):
        raise ValueError(f'Unexpected tactile shape {tactile.shape}')
    if any(len(value)!=len(time) for value in (gain,q,target)):
        raise ValueError('Raw arrays do not share the evaluation clock')
    np.testing.assert_allclose(raw_time.reshape(-1), time, rtol=0, atol=1e-8)
    if len(time)<180 or not np.allclose(np.diff(time), 1/60, rtol=0, atol=1e-8):
        raise ValueError('Require contiguous 60 Hz records and at least a 3 s window')
    if not all(np.isfinite(value).all() for value in (time,clearance,tactile,gain,q,target)):
        raise ValueError('Nonfinite recorded signal')
    order=manifest.get('sensor_finger_order', scene.get('tactile_finger_order', list(FINGERS)))
    order=tuple(canonical_finger(name) for name in order)
    if set(order)!=set(FINGERS) or len(order)!=5:
        raise ValueError('Unexpected tactile finger order')
    tactile=tactile[:, [order.index(finger) for finger in FINGERS]]
    mapping=LINKS.copy()
    for layout in scene.get('tactile_layout', []):
        mapping[canonical_finger(layout['finger'])]=layout['link_name']
    norm=np.linalg.norm(tactile, axis=-1)
    active=(norm>ACTIVE_THRESHOLD).sum(axis=-1)
    sensor_force=np.zeros((len(time),5))
    finger_force=np.zeros_like(sensor_force)
    sensor_records=np.zeros((len(time),5),int)
    physical=np.zeros(len(time))
    physical_normal=np.zeros(len(time))
    excluded_collisions=[]
    token={'thumb':'thumb', 'index':'index', 'middle':'_mid_', 'ring':'ring', 'pinky':'pinky'}
    for sample,row in enumerate(evaluation):
        for contact in row.get('solver_contacts', []):
            entity_a,entity_b=contact.get('entity_a'),contact.get('entity_b')
            link_a,link_b=contact.get('link_a',''),contact.get('link_b','')
            robot_a=(entity_a in ('ur7e_xhand','robot') or link_a.startswith('right_hand'))
            robot_b=(entity_b in ('ur7e_xhand','robot') or link_b.startswith('right_hand'))
            object_a=entity_a in ('cube','object','target')
            object_b=entity_b in ('cube','object','target')
            if not ((robot_a and object_b) or (robot_b and object_a)):
                if (robot_a and robot_b) or ((robot_a or robot_b) and 'table' in (entity_a,entity_b)):
                    excluded_collisions.append(dict(sample=sample, contact=contact))
                continue
            robot_link=link_a if robot_a else link_b
            vector=np.asarray(contact['force_on_a_world_N'],float)
            if not np.isfinite(vector).all():
                raise ValueError('Nonfinite solver contact force')
            load=float(np.linalg.norm(vector))
            physical[sample]+=load
            normal=np.asarray(contact.get('normal_world',[0,0,0]),float)
            physical_normal[sample]+=abs(float(vector@normal/max(np.linalg.norm(normal),1e-15)))
            for finger_index,finger in enumerate(FINGERS):
                if token[finger] in robot_link:
                    finger_force[sample,finger_index]+=load
                if robot_link==mapping[finger]:
                    sensor_force[sample,finger_index]+=load
                    sensor_records[sample,finger_index]+=1
    if all('all_robot_contact_load_N' in row for row in evaluation):
        np.testing.assert_allclose(physical,[row['all_robot_contact_load_N'] for row in evaluation],rtol=1e-5,atol=1e-6,
                                   err_msg='Recomputed object contact force differs from logged total')
    hold=time>time[-1]-3+1e-8
    if int(hold.sum())!=180:
        raise ValueError('Final 3 s window must contain 180 records')
    return dict(time=time,phases=phases,clearance=clearance,norm=norm,active=active,
        physical=physical,physical_normal=physical_normal,sensor_force=sensor_force,
        finger_force=finger_force,sensor_records=sensor_records,hold=hold,gain=gain,
        excluded_collisions=excluded_collisions,evaluation=evaluation,sensor_link_mapping=mapping)


def window_stats(series, mask, finger):
    force=series['sensor_force'][mask,finger]
    active=series['active'][mask,finger]
    records=series['sensor_records'][mask,finger]
    loaded=force>LOAD_THRESHOLD
    zero=active==0
    event_mask=loaded&zero
    event_times=series['time'][mask][event_mask]
    event_phases=series['phases'][mask][event_mask]
    phase_names,phase_counts=np.unique(event_phases,return_counts=True)
    return dict(samples=int(mask.sum()),physical_sensor_link_mean_force_N=float(force.mean()),
        physical_whole_finger_mean_force_N=float(series['finger_force'][mask,finger].mean()),
        active_taxels_mean=float(active.mean()), active_taxels_min=int(active.min()), active_taxels_max=int(active.max()),
        taxel_norm_sum_mean_uncalibrated_nominal_N=float(series['norm'][mask,finger].sum(-1).mean()),
        positive_load_samples=int(loaded.sum()),positive_load_but_zero_response_samples=int((loaded&zero).sum()),
        positive_load_but_zero_response_max_force_N=float(force[event_mask].max()) if event_mask.any() else None,
        positive_load_but_zero_response_min_force_N=float(force[event_mask].min()) if event_mask.any() else None,
        positive_load_but_zero_response_first_time_s=float(event_times.min()) if len(event_times) else None,
        positive_load_but_zero_response_last_time_s=float(event_times.max()) if len(event_times) else None,
        positive_load_but_zero_response_phase_counts={str(name):int(count) for name,count in zip(phase_names,phase_counts)},
        zero_load_contact_record_and_zero_response_samples=int(((records>0)&~loaded&zero).sum()),
        no_contact_record_and_zero_response_samples=int(((records==0)&zero).sum()))


def coverage_events(series):
    events=[]
    for sample,finger_index in np.argwhere((series['sensor_force']>LOAD_THRESHOLD)&(series['active']==0)):
        finger=FINGERS[finger_index]
        events.append(dict(finger=finger,sensor_link=series['sensor_link_mapping'][finger],
            sample_index_0based=int(sample),step=int(series['evaluation'][sample]['step']),
            sample_time_s=float(series['time'][sample]),phase=str(series['phases'][sample]),
            in_final_3s_hold=bool(series['hold'][sample]),
            physical_sensor_link_contact_force_N=float(series['sensor_force'][sample,finger_index]),
            active_taxels=0,peak_taxel_norm_uncalibrated_nominal_N=float(series['norm'][sample,finger_index].max())))
    return dict(schema='loaded_zero_tactile_quality_events_v1',positive_load_threshold_N=LOAD_THRESHOLD,
        active_threshold_nominal_N=ACTIVE_THRESHOLD,events_count=len(events),
        max_contact_force_N=max((event['physical_sensor_link_contact_force_N'] for event in events),default=None),
        phase_counts={phase:sum(event['phase']==phase for event in events) for phase in sorted({event['phase'] for event in events})},
        interpretation='Loaded zero-response data-quality events. These records alone do not identify skin, geometry or algorithm causes. Raw taxels are retained unchanged, without imputation.',
        events=events)


def summarize(series):
    hold=series['hold']
    all_samples=np.ones(len(hold),bool)
    return dict(samples=len(hold),duration_s=float(series['time'][-1]),hold_samples=int(hold.sum()),
        hold_start_exclusive_s=float(series['time'][-1]-3),hold_end_inclusive_s=float(series['time'][-1]),
        final_window_all_hold_phase=bool(np.all(series['phases'][hold]=='hold')),
        hold_min_bottom_clearance_m=float(series['clearance'][hold].min()),
        hold_mean_all_robot_object_contact_force_N=float(series['physical'][hold].mean()),
        hold_mean_all_robot_object_normal_force_N=float(series['physical_normal'][hold].mean()),
        hold_peak_all_robot_object_contact_force_N=float(series['physical'][hold].max()),
        robot_self_or_table_contact_records=len(series['excluded_collisions']),
        height_condition_passed=bool(np.all(series['clearance'][hold]>=.05)),
        active_threshold_nominal_N=ACTIVE_THRESHOLD,positive_load_threshold_N=LOAD_THRESHOLD,
        physical_force_definition='Per-sample sum of all robot-object solver contact vector norms, including every robot link.',
        zero_response_definition='A sensor-bearing rigid link has force >1e-8 N and all 120 taxel norms are <=1e-6 nominal N. Zero-load contact records are counted separately.',
        fingers={finger:dict(sensor_link=series['sensor_link_mapping'][finger],
            hold=window_stats(series,hold,i),all=window_stats(series,all_samples,i)) for i,finger in enumerate(FINGERS)})


def plot(series, output, shape):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(4,1,figsize=(11,10),sharex=True,constrained_layout=True)
    time=series['time']
    for ax in axes:
        ax.axvspan(time[-1]-3,time[-1],color='#e1f1e2',alpha=.8,zorder=0)
        ax.grid(alpha=.2)
    axes[0].plot(time,series['clearance']*1000,color='#1864ab')
    axes[0].axhline(50,linestyle='--',color='#c92a2a',label='50 mm success threshold')
    axes[0].set_ylabel('Lowest surface\nclearance (mm)');axes[0].legend(loc='upper left')
    axes[1].plot(time,series['physical'],color='black',label='All robot-object contacts',linewidth=1.5)
    for i,finger in enumerate(FINGERS):
        axes[1].plot(time,series['sensor_force'][:,i],label=finger,alpha=.75)
        axes[2].plot(time,series['active'][:,i],label=finger)
        axes[3].plot(time,series['norm'][:,i].sum(-1),label=finger)
    axes[1].set_ylabel('Physical force (N)');axes[1].legend(loc='upper left',ncol=3,fontsize=8)
    axes[2].set_ylabel('Active taxels / 120');axes[2].legend(loc='upper left',ncol=5,fontsize=8)
    axes[3].set_ylabel('Sum of taxel norms\nuncalibrated nominal N');axes[3].set_xlabel('Replay simulation time (s)')
    fig.suptitle(f'{shape}: fresh tactile replay v3\nGreen = final 3 seconds; tactile proxy magnitude is not physical contact force',fontsize=12)
    fig.savefig(output/f'{shape}_signals.png',dpi=180)
    fig.savefig(output/f'{shape}_signals.pdf')
    plt.close(fig)


def gallery(source, series):
    views={}
    for path in (source/'frames').glob('*'):
        match=re.match(r'^(\d+)_(.+)\.(jpg|jpeg|png)$',path.name,re.IGNORECASE)
        if match and path.is_file():
            step,view=int(match[1]),match[2]
            if view not in views or step>views[view][0]:
                views[view]=(step,path)
    preferred=('scene_front','scene_left','wrist','hand_thumb','hand_palm','hand_back','hand_little')
    order=[view for view in preferred if view in views]+[view for view in sorted(views) if view not in preferred]
    lookup={int(row['step']):float(row['sample_time_s']) for row in series['evaluation']}
    return [dict(view=view,path=str(views[view][1]),step=views[view][0],
                 time_s=lookup.get(views[view][0]),sha256=sha(views[view][1])) for view in order[:5]]


def json_link(path, output):
    return quote(os.path.relpath(path,output),safe='/')


def render(report, output):
    esc=html.escape
    sections=[]
    for shape in SHAPES:
        card=report['objects'][shape]
        blocks=[f'<section><h2>{LABELS[shape]}</h2>']
        if card['status']=='pending':
            blocks.append('<p>本条回放尚未完成，暂不计算结果。</p>')
        elif card['status']=='error':
            blocks.append('<p class="error">证据核查失败：'+esc(card['error'])+'</p>')
        else:
            metrics=card['metrics']
            status='本次回放通过' if card['accepted'] else '本次回放尚未通过全部核查'
            blocks.append(f'<p class="status">{status}</p><p>最终 3 秒最低离桌 <b>{metrics["hold_min_bottom_clearance_m"]*1000:.2f} mm</b>；'
                          f'所有机器人—物体接触力均值 <b>{metrics["hold_mean_all_robot_object_contact_force_N"]:.3f} N</b>。</p>')
            comparison=card['source_comparison']
            position_error=comparison.get('max_position_component_error_m')
            if position_error is not None:
                blocks.append(f'<p>与旧源逐条核对相同的是关节目标、增益、指令时间与阶段；状态轨迹仍由物理仿真生成。'
                              f'本次物体位置相对旧源的最大坐标分量差为 <b>{position_error*1000:.3f} mm</b>，没有宣称状态完全相同。</p>')
            if card['audit_errors']:
                blocks.append('<p class="error">'+'；'.join(esc(error) for error in card['audit_errors'])+'</p>')
            blocks.append('<table><tr><th>手指</th><th>保持期传感器所在刚体物理力 N</th><th>保持期整根手指物理力 N</th><th>保持期活跃测点均值 / 120</th><th>带载但零响应：保持 / 全过程</th><th>带载零响应最大力 N · 阶段</th><th>零载荷接触记录且零响应：保持 / 全过程</th></tr>')
            for finger in FINGERS:
                data=metrics['fingers'][finger];hold=data['hold'];all_=data['all']
                maximum=all_['positive_load_but_zero_response_max_force_N']
                event_text='—' if maximum is None else f'{maximum:.5f} · '+', '.join(
                    f'{name} ({count})' for name,count in all_['positive_load_but_zero_response_phase_counts'].items())
                blocks.append(f'<tr><td>{FINGER_LABELS[finger]}</td><td>{hold["physical_sensor_link_mean_force_N"]:.3f}</td>'
                    f'<td>{hold["physical_whole_finger_mean_force_N"]:.3f}</td><td>{hold["active_taxels_mean"]:.2f}</td>'
                    f'<td>{hold["positive_load_but_zero_response_samples"]} / {all_["positive_load_but_zero_response_samples"]}</td>'
                    f'<td>{esc(event_text)}</td>'
                    f'<td>{hold["zero_load_contact_record_and_zero_response_samples"]} / {all_["zero_load_contact_record_and_zero_response_samples"]}</td></tr>')
            blocks.append('</table><p class="note">带载定义：传感器所在刚体与物体的接触力模之和 &gt;10⁻⁸ N；活跃测点定义：三轴信号模 &gt;10⁻⁶ nominal N。零载荷 contact 记录单独列出，不算带载漏响应。统计样本是连续时间点，并非相互独立的抓取试验。</p>')
            events=card['coverage_events_summary']
            blocks.append(f'<p class="note">本条共有 <b>{events["events_count"]}</b> 个手指—时间点带载零响应质量事件。'
                '这些事件需要单独核查原因，不能仅凭零读数归因于蒙皮或某个算法；原始信号保留为零，没有填补。'
                f'<a href="{shape}_coverage_events.json">逐事件时间、阶段与实际接触力</a>。</p>')
            if card['images']:
                blocks.append('<div class="gallery">')
                for image in card['images']:
                    href=json_link(image['path'],output)
                    when=f'{image["time_s"]:.2f} s' if image['time_s'] is not None else f'记录步 {image["step"]}'
                    blocks.append(f'<figure><a href="{href}"><img src="{href}" loading="lazy" alt="{esc(image["view"])}"></a><figcaption>{esc(image["view"])} · {when} · 实际保存帧</figcaption></figure>')
                blocks.append('</div>')
            else:
                blocks.append('<p>本次回放未保存展示帧，没有使用旧源帧替代。</p>')
            blocks.append(f'<a href="{shape}_signals.png"><img class="plot" src="{shape}_signals.png" alt="本次回放的离桌、物理力和新触觉记录"></a>')
            blocks.append('<p>'+' · '.join(f'<a href="{json_link(Path(card["source"])/name,output)}">{label}</a>'
                for name,label in [('manifest.json','本次来源与触觉版本'),('validation.json','回放验证'),('score.json','任务判据'),
                                   ('raw_tactile.npz','新采集五指原始触觉'),('evaluation.jsonl','物理接触记录')])+'</p>')
            blocks.append('<details><summary>来源与版本声明</summary><pre>'+esc(json.dumps(card['provenance'],indent=2,ensure_ascii=False))+'</pre></details>')
        blocks.append('</section>')
        sections.append(''.join(blocks))
    evidence_links=' · '.join(f'<a href="{json_link(path,output)}">{label}</a>' for path,label in
        ((Path(report['dataset_root'])/'dataset_manifest.json','数据集清单'),
         (Path(report['dataset_root'])/'independent_audit.json','独立审查')) if path.is_file())
    return '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>v3 示范回放 · 新触觉采集</title>
<style>body{font:16px/1.65 system-ui,sans-serif;max-width:1440px;margin:30px auto;padding:0 22px;background:#f3f6fa;color:#203040}section{background:white;padding:24px;border:1px solid #dbe4ec;border-radius:10px;margin:24px 0}.note{background:#eef4ff;border-left:4px solid #527fb9;padding:14px}.error{color:#a52828}.status{font-weight:700;color:#246043}table{width:100%;border-collapse:collapse;font-size:14px}th,td{border-bottom:1px solid #dde5ec;padding:8px;text-align:left}.gallery{display:grid;grid-template-columns:repeat(5,1fr);gap:10px;margin:22px 0}figure{margin:0}img{width:100%}figcaption{font-size:12px}.plot{display:block;max-width:1100px;margin:24px auto}a{color:#245fa0}pre{white-space:pre-wrap;font-size:12px}details{margin-top:15px}@media(max-width:900px){.gallery{grid-template-columns:repeat(2,1fr)}section{padding:12px}body{padding:0 8px}table{font-size:12px}}</style>
<h1>v3 示范回放：重新采集触觉</h1><p class="note">回放旧成功示范的固定关节指令，在当前仿真器中重新计算并记录触觉。旧源 raw 记录不用于本报告曲线或统计；本页只读取本次回放目录。没有重新调用 Astra，也不是新策略的泛化成功率测试。</p>
<p>物理 300 Hz · 触觉 60 Hz · 动作 15 Hz。任务条件：最后 3 秒真实最低表面持续离桌 ≥50 mm，并满足本次验证记录。</p>
<p class="note"><b>触觉仍未做真实传感器标定。</b>切向分量包含相对速度代理，不能作为真实切向力或摩擦余量。拇指末端固定组件与测点覆盖之间的盲区未改；修正读取算法不等于消除了几何盲区。抓取成功也不证明触觉覆盖完整。</p>'''+''.join(sections)+f'<p>生成时间 {esc(report["generated_utc"])} · <a href="report.json">全部统计、阈值及输入文件哈希</a> · {evidence_links}</p></html>'


def build(dataset):
    dataset=Path(dataset).resolve()
    output=dataset/'report'
    output.mkdir(parents=True,exist_ok=True)
    report=dict(schema='fresh_teacher_replay_report_v1',generated_utc=datetime.now(timezone.utc).isoformat(),
        dataset_root=str(dataset),generator_sha256=sha(__file__),hardware_connected=False,
        source_raw_used_for_statistics=False, objects={})
    for shape in SHAPES:
        source=dataset/shape/'attempt01'
        required=('manifest.json','score.json','validation.json','control.jsonl','evaluation.jsonl','raw_tactile.npz')
        missing=[name for name in required if not (source/name).is_file()]
        if missing:
            report['objects'][shape]=dict(status='pending',missing=missing)
            continue
        try:
            manifest=read(source/'manifest.json');score=read(source/'score.json');validation=read(source/'validation.json')
            scene=read(source/'object_scene.json') if (source/'object_scene.json').exists() else {}
            series=parse_series(source,manifest,scene)
            metrics=summarize(series)
            errors=[]
            if manifest.get('driver')!='direct_teacher_frozen_replay':
                errors.append('driver 不符合直接固定指令回放')
            if manifest.get('raw_tactile_reused') is not False:
                errors.append('manifest 未明确声明 raw_tactile_reused=false')
            if not manifest.get('tactile_model'):
                errors.append('缺少触觉模型版本声明')
            if score.get('task_success',score.get('success')) is not True:
                errors.append('任务得分未通过')
            if not (validation.get('accepted') is True or validation.get('status')=='passed'):
                errors.append('回放验证未通过')
            if validation.get('exact_source_commands_replayed') is not True:
                errors.append('未验证旧源指令逐条一致')
            if not metrics['height_condition_passed'] or not metrics['final_window_all_hold_phase']:
                errors.append('最终3秒高度或保持阶段核查未通过')
            if metrics['robot_self_or_table_contact_records']:
                errors.append('存在机器人自身或桌面接触记录')
            plot(series,output,shape)
            images=gallery(source,series)
            files=list(required)+(['object_scene.json'] if scene else [])
            events=coverage_events(series)
            events['source_evidence_sha256']={name:sha(source/name) for name in ('evaluation.jsonl','raw_tactile.npz')}
            events_path=output/f'{shape}_coverage_events.json'
            events_path.write_text(json.dumps(events,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
            report['objects'][shape]=dict(status='complete',accepted=not errors,audit_errors=errors,
                source=str(source),metrics=metrics,images=images,
                source_comparison=validation.get('source_comparison',{}),
                exact_source_commands_replayed=validation.get('exact_source_commands_replayed'),
                exact_replay_scope='Frozen joint targets, gain schedule, command times and phase labels; not exact physical states.',
                coverage_events_summary={key:value for key,value in events.items() if key!='events'},
                coverage_events_path=str(events_path),coverage_events_sha256=sha(events_path),
                input_sha256={name:sha(source/name) for name in files},
                provenance={key:manifest.get(key) for key in ('driver','source_run','source_sha256','tactile_model',
                    'raw_tactile_reused','sensor_finger_order','solver_profile','physics_profile')})
        except (OSError,ValueError,KeyError,TypeError,AssertionError) as error:
            report['objects'][shape]=dict(status='error',error=str(error),source=str(source))
    (output/'report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    (output/'index.html').write_text(render(report,output))
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root',type=Path,required=True)
    args=parser.parse_args()
    report=build(args.dataset_root)
    print(json.dumps({shape:{key:card.get(key) for key in ('status','accepted','audit_errors','error')}
                      for shape,card in report['objects'].items()},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
