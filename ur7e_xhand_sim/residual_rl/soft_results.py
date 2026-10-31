"""Render offline, incomplete-safe reports from the elastic evaluation summary.

No physics, inference, training, suite execution or trace rescoring is performed.
Only protocol-declared rows from experiment_summary.json enter the report; a
missing row is pending, never a failed grasp. Existing suite_status.json can
identify currently running rows, but cannot promote an unverified score.
"""
from __future__ import annotations

import argparse
import base64
from collections import Counter
from datetime import datetime, timezone
import hashlib
import html
import io
import json
import math
import os
from pathlib import Path
import re
from urllib.parse import quote

import numpy as np


LABELS = {'fixed_2': '固定增益 2', 'fixed_3': '固定增益 3'}
STATES = {
    'pending': ('待测 / 未提供', '#e5e9ed'),
    'running': ('运行中', '#c5d9f4'),
    'awaiting_audit': ('已输出，待汇总审计', '#d9ccef'),
    'invalid': ('执行或完整性异常', '#f4bf8b'),
    'task_failed': ('任务未成功', '#eab8b5'),
    'task_only': ('任务成功，严格条件未满足', '#f3dfa0'),
    'strict': ('任务及严格条件均成功', '#a7d9c8'),
}


def read(path, fallback=None):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else fallback


def label(controller):
    if controller in LABELS:
        return LABELS[controller]
    match = re.fullmatch(r'(no_tactile|tactile)_seed(\d+)', controller)
    return f"种子 {match[2]} · {'无触觉' if match[1] == 'no_tactile' else '触觉'}" if match else controller


def finite(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


def fmt(value, digits=3):
    return f'{value:.{digits}f}' if finite(value) else '—'


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalize(root, synthetic=False):
    protocol = read(root/'protocol.json')
    if not protocol:
        raise ValueError('protocol.json is required')
    summary = read(root/'experiment_summary.json', {})
    progress = read(root/'suite_status.json', {})
    controllers = protocol['controllers']
    conditions = protocol['conditions_kg_mu']
    if len(set(controllers)) != len(controllers) or not controllers or not conditions:
        raise ValueError('Protocol has missing or duplicate controllers/conditions')
    if any(len(c) != 2 or not all(finite(v) for v in c) for c in conditions):
        raise ValueError('Protocol material conditions must be finite mass/friction pairs')
    indexed = {}
    excluded = []
    for row in summary.get('episodes', []):
        key = (row.get('condition_index'), row.get('controller'))
        if key[1] not in controllers:
            excluded.append(row.get('id', str(key)))
            continue
        if key[0] not in range(len(conditions)):
            raise ValueError(f'Unexpected non-protocol material condition: {key}')
        if key in indexed:
            raise ValueError(f'Duplicate summary row: {key}')
        indexed[key] = row
    progress_rows = {(r.get('condition_index'), r.get('controller')): r for r in progress.get('episodes', [])}
    rows = []
    suite_running = progress.get('status') in ('running', 'preparing', 'resuming')
    for ci, (mass, mu) in enumerate(conditions):
        condition = f'condition_{ci:02d}_{round(mass*1000):03d}g_mu{mu:g}'
        for controller in controllers:
            key = (ci, controller)
            source = indexed.get(key, {})
            row = dict(source)
            row.update(condition_index=ci, controller=controller, mass_kg=mass, friction=mu)
            row.setdefault('id', f'{condition}__{controller}')
            row.setdefault('output', str(root/'official'/condition/controller))
            row.setdefault('log', str(root/'official_logs'/f"{row['id']}.log"))
            valid = source.get('valid_completed') is True
            if valid:
                if type(source.get('task_success')) is not bool or type(source.get('strict_success')) is not bool:
                    raise ValueError(f"Verified row lacks boolean labels: {row['id']}")
                if source['strict_success'] and not source['task_success']:
                    raise ValueError(f"Strict success without task success: {row['id']}")
                state = 'strict' if source['strict_success'] else 'task_only' if source['task_success'] else 'task_failed'
            else:
                pr = progress_rows.get(key, {})
                if suite_running and pr.get('active'):
                    state = 'running'
                elif pr.get('episode_status', {}).get('status') == 'completed':
                    state = 'awaiting_audit'
                elif source.get('status') in ('running', 'initializing'):
                    state = 'running'
                elif source.get('status') in ('completed', 'awaiting_audit'):
                    state = 'awaiting_audit'
                elif source.get('status') not in (None, 'pending', 'not_attempted') or source.get('attempted'):
                    state = 'invalid'
                else:
                    state = 'pending'
            row.update(valid_completed=valid, report_state=state)
            rows.append(row)
    planned = len(controllers)*len(conditions)
    declared = [protocol.get(k) for k in ('official_model_episodes','official_fixed_baseline_episodes')]
    warnings, notes = [], []
    if all(isinstance(n,int) for n in declared) and sum(declared) != planned:
        warnings.append(f'协议官方回合数 {sum(declared)} 与当前 controllers × conditions={planned} 不一致。')
    if summary.get('official_planned') not in (None, planned):
        warnings.append(f"汇总计划数 {summary['official_planned']} 与本轮协议 {planned} 不一致；仅统计协议声明的控制器。")
    if excluded:
        notes.append(f'原始汇总中有 {len(excluded)} 条不属于本轮 controllers 的记录，已排除；它们不是本轮待测或失败项。')
    if summary.get('protocol_sha256') and summary['protocol_sha256'] != file_hash(root/'protocol.json'):
        warnings.append('汇总记录的协议哈希与当前 protocol.json 不一致；仅展示结果，不能作最终结论。')
    if summary.get('input_integrity_passed') is False:
        warnings.append('输入完整性核查未通过：'+str(summary.get('input_integrity_error') or '原因见原始汇总'))
    bounded = False
    if summary and summary.get('all_prefixes_exact') is not True:
        budget = protocol.get('prefix_tolerance_abs')
        bounded = bool(summary.get('all_prefixes_within_tolerance') is True
            and isinstance(budget, dict) and budget
            and budget == summary.get('prefix_tolerance_abs')
            and all(finite(v) and v >= 0 for v in budget.values()))
        if bounded:
            notes.append('共同前缀并非逐值完全相同，但全部差异通过协议预先声明的逐字段绝对容差；属于已接受的有限数值差异，不是精确一致。')
        else:
            warnings.append('各材料条件的共同前缀尚未全部通过协议的一致性验收；可能仍有待测项，也可能存在超限差异。')
    seed_ids = sorted({int(m[1]) for c in controllers if (m := re.fullmatch(r'(?:no_tactile|tactile)_seed(\d+)', c))})
    if set(controllers) == {'tactile_seed1','no_tactile_seed1'}:
        scope = f'本轮仅种子 1：触觉与无触觉 × {len(conditions)} 个材料条件 = {planned} 回合；不包含种子 2/3 或固定增益组。'
    else:
        scope = f"本轮计划：{len(controllers)} 个控制器 × {len(conditions)} 个材料条件 = {planned} 回合；训练种子 {', '.join(map(str,seed_ids)) or '无'}。"

    counts = Counter(r['report_state'] for r in rows)
    unfinished = sum(counts[k] for k in ('pending', 'running', 'awaiting_audit'))
    status = '运行中（部分结果）' if counts['running'] or suite_running else '尚无完整结果' if not any(r['valid_completed'] for r in rows) else '尚未完成（部分结果）' if unfinished else '已结束，含执行 / 审计异常' if counts['invalid'] or warnings else '已完成'
    if status == '已完成' and bounded:
        status = '已完成（接受了有限数值差异）'
    return {'root': root, 'protocol': protocol, 'summary': summary, 'rows': rows,
            'controllers': controllers, 'conditions': conditions, 'counts': counts,
            'status': status, 'warnings': warnings, 'notes': notes, 'scope': scope,
            'excluded_nonplanned_rows': excluded, 'prefix_bounded_accepted': bounded, 'seed_ids': seed_ids,
            'synthetic': bool(synthetic or summary.get('synthetic') or protocol.get('synthetic')),
            'updated': summary.get('updated_utc', progress.get('updated_utc', '尚无汇总时间'))}


def aggregate(rows):
    valid = [r for r in rows if r['valid_completed']]
    return {'planned': len(rows), 'completed': len(valid),
            'task': sum(r['task_success'] for r in valid),
            'strict': sum(r['strict_success'] for r in valid),
            'pending': sum(r['report_state'] == 'pending' for r in rows),
            'running': sum(r['report_state'] == 'running' for r in rows),
            'awaiting_audit': sum(r['report_state'] == 'awaiting_audit' for r in rows),
            'invalid': sum(r['report_state'] == 'invalid' for r in rows)}


def paired(report):
    rows = report['rows']
    indexed = {(r['controller'], r['condition_index']): r for r in rows}
    seeds = sorted({int(m[1]) for c in report['controllers'] if (m := re.fullmatch(r'tactile_seed(\d+)', c))})
    result = []
    for seed in seeds:
        common, loads = [], []
        for ci in range(len(report['conditions'])):
            t, b = indexed.get((f'tactile_seed{seed}', ci)), indexed.get((f'no_tactile_seed{seed}', ci))
            if t and b and all(r['valid_completed'] and r['task_success'] for r in (t, b)):
                common.append(ci)
                if all(finite(r.get('mean_normal_load_N')) for r in (t, b)):
                    loads.append((t['mean_normal_load_N'], b['mean_normal_load_N']))
        t_mean, b_mean = np.mean(loads, axis=0) if loads else (None, None)
        result.append({'seed': seed, 'common': common, 'load_pairs': len(loads), 'touch_N': t_mean, 'blind_N': b_mean,
                       'reduction': 1-t_mean/b_mean if b_mean and t_mean is not None else None})
    return result


def chart(report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    from matplotlib import font_manager
    for font in ('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
                 '/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf'):
        if Path(font).exists():
            font_manager.fontManager.addfont(font)
            plt.rcParams['font.family'] = font_manager.FontProperties(fname=font).get_name()
            break
    plt.rcParams.update({'font.size': 10, 'axes.unicode_minus': False, 'axes.spines.top': False,
                         'axes.spines.right': False})
    controllers, conditions = report['controllers'], report['conditions']
    indexed = {(r['controller'], r['condition_index']): r for r in report['rows']}
    state_keys = list(STATES)
    data = np.array([[state_keys.index(indexed[c, ci]['report_state']) for ci in range(len(conditions))] for c in controllers])
    compact = len(controllers) <= 2
    fig = plt.figure(figsize=(16, 8.5 if compact else 10), facecolor='#f6f8fb')
    grid = fig.add_gridspec(2, 1, height_ratios=(1.1, 1) if compact else (1.35, 1),
                          hspace=.90 if compact else .60,
                          left=.16, right=.97, top=.77 if compact else .81, bottom=.17 if compact else .10)
    ax = fig.add_subplot(grid[0])
    ax.imshow(data, cmap=ListedColormap([STATES[k][1] for k in state_keys]), vmin=-.5, vmax=len(state_keys)-.5, aspect='auto')
    ax.set_yticks(range(len(controllers)), [label(c) for c in controllers])
    ax.set_xticks(range(len(conditions)), [f'{m*1000:g} g\nμ = {f:g}' for m, f in conditions])
    for i,c in enumerate(controllers):
        for j in range(len(conditions)):
            r=indexed[c,j]
            txt=f"{int(r['task_success'])} / {int(r['strict_success'])}" if r['valid_completed'] else {'pending':'待测','running':'运行中','invalid':'异常','awaiting_audit':'待审计'}[r['report_state']]
            ax.text(j,i,txt,ha='center',va='center',fontsize=9,color='#15283a')
    ax.set_xticks(np.arange(-.5,len(conditions),1),minor=True)
    ax.set_yticks(np.arange(-.5,len(controllers),1),minor=True)
    ax.grid(which='minor',color='white',linewidth=2);ax.tick_params(which='minor',length=0)
    ax.set_title('每个材料条件的单次结果：任务 / 严格成功（1 或 0）',loc='left',pad=15,fontweight='bold')
    ax.legend(handles=[Patch(facecolor=color,label=name) for name,color in STATES.values()],
              loc='upper center',bbox_to_anchor=(.48,-.32 if compact else -.18),ncol=4,frameon=False,fontsize=8)
    ax2 = fig.add_subplot(grid[1]);ax2.set_facecolor('#f6f8fb')
    policy_controllers=[c for c in controllers if re.fullmatch(r'(no_tactile|tactile)_seed\d+',c)]
    stats=[aggregate([r for r in report['rows'] if r['controller']==c]) for c in policy_controllers]
    y=np.arange(len(stats));ax2.barh(y-.16,[s['task'] for s in stats],.28,color='#3478a5',label='任务成功数')
    ax2.barh(y+.16,[s['strict'] for s in stats],.28,color='#29866b',label='严格成功数')
    ax2.set_yticks(y,[label(c) for c in policy_controllers]);ax2.invert_yaxis()
    planned=max((s['planned'] for s in stats),default=len(conditions));ax2.set_xlim(0,planned+4.1)
    ax2.set_xticks(range(planned+1));ax2.grid(axis='x',alpha=.16);ax2.set_axisbelow(True)
    for i,s in enumerate(stats):
        ax2.text(planned+.25,i,f"任务 {s['task']} · 严格 {s['strict']}   已验证 {s['completed']}/{s['planned']}",va='center',fontsize=8)
    ax2.set_title('各训练种子的触觉 / 无触觉对照 · 计数而非未完成样本的失败率',loc='left',pad=12,fontweight='bold')
    ax2.legend(loc='lower left',bbox_to_anchor=(0,-.37 if compact else -.23),ncol=2,frameon=False,fontsize=9)
    total=aggregate(report['rows'])
    title=('SYNTHETIC · 合成渲染测试，不是实验结果\n' if report['synthetic'] else '')+'弹性方块 × Astra 大平移轨迹 · 冻结 PPO 零样本迁移'
    fig.suptitle(title,x=.04,y=.97,ha='left',fontsize=17,fontweight='bold',color='#18364a')
    fig.text(.04,.885 if compact else .895,f"{report['status']}  |  有效完成 {total['completed']}/{total['planned']}  |  执行/审计异常 {total['invalid']}  |  待测及待审计 {total['pending']+total['awaiting_audit']}",fontsize=11,color='#42596b')
    fig.text(.04,.845 if compact else .855,report['scope'],fontsize=10,color='#42596b')
    fig.text(.04,.028,'缺失、运行中和异常均不计为抓取失败。严格条件额外使用独立 IPC 法向载荷；虚拟 taxel 为几何估计，未作真实硬件标定。',fontsize=9,color='#536879')
    buffer=io.BytesIO();fig.savefig(buffer,format='png',dpi=150);plt.close(fig)
    return buffer.getvalue()


def make_html(report, png, output_dir):
    esc=lambda x:html.escape(str(x),quote=True)
    def link(path,text):
        path=Path(path)
        if not path.is_absolute():path=report['root']/path
        if not path.exists():return f'<span class="muted">{esc(text)}（未提供）</span>'
        href=quote(os.path.relpath(path.resolve(),output_dir.resolve()),safe='/')
        return f'<a href="{esc(href)}">{esc(text)}</a>'
    def pill(text,state):return f'<span class="pill" style="background:{STATES[state][1]}">{esc(text)}</span>'
    total=aggregate(report['rows']);p=report['protocol'];s=report['summary']
    warnings=''.join(f'<p class="warning">{esc(w)}</p>' for w in report['warnings'])
    notes=''.join(f'<p class="note">{esc(n)}</p>' for n in report.get('notes',[]))
    synthetic='<div class="synthetic">SYNTHETIC · 合成数据，仅用于报告渲染测试，不是实验结果</div>' if report['synthetic'] else ''
    stats=[]
    for c in report['controllers']:
        a=aggregate([r for r in report['rows'] if r['controller']==c])
        task=f"{a['task']}/{a['completed']}" if a['completed'] else '—（无有效完成）'
        strict=f"{a['strict']}/{a['completed']}" if a['completed'] else '—（无有效完成）'
        stats.append(f"<tr><td>{esc(label(c))}</td><td>{a['completed']}/{a['planned']}</td><td>{task}</td><td>{strict}</td><td>{a['pending']}</td><td>{a['running']}</td><td>{a['awaiting_audit']}</td><td>{a['invalid']}</td></tr>")
    pair_rows=[]
    for r in paired(report):
        conds='；'.join(f"{p['conditions_kg_mu'][i][0]*1000:g} g / μ={p['conditions_kg_mu'][i][1]:g}" for i in r['common']) or '尚无共同成功条件'
        reduction=f"{r['reduction']*100:.1f}%" if finite(r['reduction']) else '—'
        pair_rows.append(f"<tr><td>{r['seed']}</td><td>{len(r['common'])}</td><td>{r['load_pairs']}</td><td>{fmt(r['touch_N'])}</td><td>{fmt(r['blind_N'])}</td><td>{reduction}</td><td>{esc(conds)}</td></tr>")
    failure_names={'hold_failed':'任务条件未满足（保持高度、碰撞或网格有效性；见 score/trace）','contact_discontinuity':'任务成功，但保持期间独立法向支撑不连续','not_attempted':'尚未尝试'}
    episodes=[]
    for r in report['rows']:
        state=r['report_state'];valid=r['valid_completed']
        task='成功' if valid and r['task_success'] else '未成功' if valid else '—'
        strict='成功' if valid and r['strict_success'] else '未满足' if valid else '—'
        reason=r.get('failure_reason')
        if state in ('pending','running','awaiting_audit'):reason=STATES[state][0]
        elif state=='strict':reason='—'
        else:
            reason=failure_names.get(reason,reason or '未提供具体原因')
            if valid and not r['task_success']:
                details=[]
                if r.get('collision'):details.append('出现碰撞标记')
                clearance=r.get('hold_min_clearance_m')
                if finite(clearance) and clearance < .05:
                    details.append(f'保持最低净空 {clearance*1000:.1f} mm < 50 mm')
                min_j=r.get('deformation',{}).get('min_J')
                if finite(min_j) and min_j <= 0:details.append(f'最小单元 J={min_j:g}，不为正')
                if details:reason='；'.join(details)
        out=Path(r['output']);out=out if out.is_absolute() else report['root']/out
        artifacts=' · '.join([link(out/'score.json','score'),link(out/'manifest.json','manifest'),link(out/'trace.npz','trace'),link(out/'partial_trace.npz','partial'),link(out/'environment'/'soft_scene_protocol.json','物理参数'),link(r['log'],'日志')])
        coverage=r.get('sensor_coverage',{});positive=coverage.get('positive_samples_per_finger');missed=coverage.get('missed_samples_per_finger')
        thumb=f"{missed[0]}/{positive[0]}" if positive and missed else '—'
        index=f"{missed[1]}/{positive[1]}" if positive and missed else '—'
        episodes.append(f"<tr><td>{r['condition_index']+1}</td><td>{r['mass_kg']*1000:g}</td><td>{r['friction']:g}</td><td>{esc(label(r['controller']))}</td><td>{pill(STATES[state][0],state)}</td><td>{task}</td><td>{strict}</td><td>{fmt(r.get('mean_normal_load_N'))}</td><td>{fmt(r.get('hold_min_clearance_m')*1000 if finite(r.get('hold_min_clearance_m')) else None,1)}</td><td>{thumb} / {index}</td><td>{esc(reason)}</td><td class='artifacts'>{artifacts}</td></tr>")
    material=p.get('material',{});rates=p.get('rates_hz',{});physics=p.get('physics',{})
    parameters=f"E={material.get('E_Pa','—')} Pa；ν={material.get('nu','—')}；尺寸={material.get('size_m','—')} m；物理/触觉/动作={rates.get('physics','—')}/{rates.get('touch','—')}/{rates.get('action','—')} Hz；substeps={physics.get('substeps',p.get('physics_substeps','—'))}；Newton tolerance={physics.get('newton_tolerance',p.get('physics_newton_tolerance','—'))}"
    main_links=' · '.join([link(report['root']/'protocol.json','协议'),link(report['root']/'experiment_summary.json','原始汇总'),link(report['root']/'prefix_audit.json','前缀审计'),link(report['root']/'suite_status.json','执行状态')])
    model_rows=''.join(f"<tr><td>{esc(label(name))}</td><td>{link(info['path'],'模型文件')}</td><td class='hash'>{esc(info.get('sha256','—'))}</td></tr>" for name,info in p.get('models',{}).items() if name in report['controllers'])
    return f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>弹性方块 · 触觉 PPO 迁移结果</title>
<style>*{{box-sizing:border-box}}body{{margin:0;background:#f5f7fa;color:#20374a;font:15px/1.65 system-ui,"Noto Sans CJK SC",sans-serif}}main{{max-width:1480px;margin:auto;padding:32px}}h1{{font-size:29px;line-height:1.35;margin:0 0 12px}}h2{{font-size:21px;margin:0 0 14px}}section{{background:white;border:1px solid #dce4eb;border-radius:12px;padding:24px;margin:22px 0}}p{{margin:10px 0}}.eyebrow,.muted{{color:#71808c}}.cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:20px 0}}.card{{background:white;padding:16px 20px;border-radius:10px;border:1px solid #dce4eb}}.number{{font-size:30px;font-weight:700;display:block}}.warning{{padding:14px;background:#fff0d6;border-left:4px solid #cb7c12}}.note{{padding:14px;background:#e9f2fa;border-left:4px solid #397ba9}}.synthetic{{padding:20px;background:#99294b;color:white;font-size:22px;font-weight:bold;margin-bottom:20px}}.scroll{{overflow-x:auto}}table{{width:100%;border-collapse:collapse;font-size:13px}}th,td{{padding:10px 11px;text-align:left;border-bottom:1px solid #e1e7ed;vertical-align:top}}th{{background:#eef3f7;white-space:nowrap}}td.hash{{font-family:monospace;font-size:11px;word-break:break-all}}.pill{{display:inline-block;padding:3px 8px;border-radius:5px;white-space:nowrap;font-size:12px}}a{{color:#17649b;text-decoration:none}}a:hover{{text-decoration:underline}}img{{width:100%;height:auto}}.artifacts{{min-width:230px;font-size:12px}}.foot{{font-size:12px;color:#667b8d}}@media(max-width:700px){{main{{padding:15px}}section{{padding:16px}}.cards{{grid-template-columns:repeat(2,1fr)}}h1{{font-size:24px}}}}</style>
<main>{synthetic}<p class="eyebrow">UR7e × XHand · 弹性 FEM 方块 · 已归档 Astra 大平移命令轨迹</p><h1>冻结触觉 PPO 的软体零样本迁移</h1><p><strong>{esc(report['status'])}</strong> · 汇总快照：{esc(report['updated'])}</p><p>{esc(report['scope'])}</p><p>{main_links}</p>{notes}{warnings}
<div class="cards"><div class="card">有效完成 / 计划<span class="number">{total['completed']} / {total['planned']}</span></div><div class="card">任务成功 / 有效完成<span class="number">{str(total['task'])+' / '+str(total['completed']) if total['completed'] else '—'}</span></div><div class="card">严格成功 / 有效完成<span class="number">{str(total['strict'])+' / '+str(total['completed']) if total['completed'] else '—'}</span></div><div class="card">执行或审计异常<span class="number">{total['invalid']}</span></div></div>
<p>待测 {total['pending']}；运行中 {total['running']}；已输出待审计 {total['awaiting_audit']}。这些状态均不会被当作抓取失败，也不进入成功率分母。图表只反映本次快照，不自动刷新。</p>
<section><h2>材料与控制器结果</h2><img alt="按控制器和九个材料条件排列的状态图，以及各训练种子成功计数" src="data:image/png;base64,{base64.b64encode(png).decode()}"><div class="scroll"><table><thead><tr><th>控制器</th><th>有效 / 计划</th><th>任务成功 / 有效</th><th>严格成功 / 有效</th><th>待测</th><th>运行</th><th>待审计</th><th>异常</th></tr></thead><tbody>{''.join(stats)}</tbody></table></div></section>
<section><h2>共同成功条件下的配对法向载荷</h2><p>同一训练种子、同一材料条件，触觉和无触觉回合都任务成功时才组成一对。下表比较各回合策略阶段的五指 IPC 法向 primitive 载荷之和的时间均值，再对共同条件取均值；不把失败掉落后的低载荷当作节力。该条件化比较存在成功子集选择效应，不能单独证明整体优越性。</p><div class="scroll"><table><thead><tr><th>种子</th><th>共同成功条件数</th><th>有效载荷对数</th><th>触觉 N</th><th>无触觉 N</th><th>相对减少</th><th>共同条件</th></tr></thead><tbody>{''.join(pair_rows)}</tbody></table></div><p class="foot">相对减少 = 1 − 触觉均值 / 无触觉均值；负数表示触觉载荷更高。无共同成功条件时显示“—”，不补零。</p></section>
<section><h2>每回合状态、失败原因与原始证据</h2><p>任务成功要求最终保持窗口中，变形后物体最低表面相对桌面的净空持续 ≥ 50 mm，且网格有效、无已检出的机械臂自身或桌面碰撞。严格成功还要求每个保持采样中至少两个传感器所在指节的独立法向载荷 > 0.05 N。保持时间和最终定义以链接协议为准。</p><div class="scroll"><table><thead><tr><th>条件</th><th>质量 g</th><th>摩擦 μ</th><th>控制器</th><th>状态</th><th>任务</th><th>严格</th><th>平均法向载荷 N</th><th>保持最小净空 mm</th><th>拇 / 食指漏响应：漏检 / 接触样本</th><th>原因</th><th>原始文件</th></tr></thead><tbody>{''.join(episodes)}</tbody></table></div></section>
<section><h2>如何解释本实验</h2><p><strong>冻结模型迁移。</strong>使用在刚体任务上训练完成的 PPO 权重，在这批软体条件上不重训、不按结果挑选 checkpoint。各控制器使用同一条归档 Astra 命令轨迹；局部策略根据自己的观测调整动作。本评估不进行实时 Astra 调用或规则抓取补救。仿真接触模型和虚拟触觉生成方式也发生变化，因此结果属于零样本迁移，不能当成只改变“材质”一个变量的实验。</p><p><strong>虚拟 taxel 与 IPC 法向载荷是两个量。</strong>模型输入是原厂家测点上的几何弹簧/阻尼估计，采用真实接触存在性筛选；它不是 FEM 牵引力，也没有将真实接触力大小注入 actor。独立评估量来自 IPC 接触势梯度的法向 primitive 载荷，表示法向接触强度，不能解释为完整合力或切向摩擦力；默认较松 Newton 容差下的切向梯度导出不可靠。</p><p><strong>真实背盖盲区。</strong>碰撞装配保留厂家固定指尖及背盖。主轨迹可能由拇指背盖提供物理支撑，背盖接触不保证厂家测点有响应；不能通过扩大探针半径或删除背盖强行消除盲区。厂家正面传感区域的独立接触正对照已验证通道可响应，但实际覆盖率仍应逐回合检查。漏响应统计是存在性对应关系，不是测力标定。</p><p><strong>证据范围。</strong>每个控制器每个材料条件是一次仿真回合；seed 表示训练随机种子；本轮只包含协议列出的种子和控制器，未计划的组不进入统计，也不是待测项。单种子结果不能代表跨训练种子的稳定性。尚不能仅据少量成功数宣称统计显著优越、真实硅胶标定或实机可靠性。执行异常、未完成项及审计发现完整保留。预检、smoke 和参考重复回合不进入正式统计。</p><p class="foot">{esc(parameters)}</p><p class="foot">前缀绝对容差：{esc(p.get("prefix_tolerance_abs", "未声明"))}。逐数组最大差异、精确一致与容差验收分别见前缀审计文件。</p></section>
<section><h2>冻结模型来源</h2><div class="scroll"><table><thead><tr><th>模型</th><th>文件</th><th>SHA-256</th></tr></thead><tbody>{model_rows}</tbody></table></div><p>轨迹：{link(p.get('trajectory','missing'),'归档命令文件')}；SHA-256：<span class="hash">{esc(p.get('trajectory_sha256','—'))}</span></p><p class="foot">本页面仅离线读取汇总与协议，不重新执行评分、不运行物理。原始 artifact 链接依赖本地目录结构；页面图表和样式完全内嵌。生成时间：{datetime.now(timezone.utc).isoformat()}</p></section></main></html>'''


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,help='Defaults to --root; does not alter input artifacts')
    parser.add_argument('--synthetic',action='store_true',help='Prominent synthetic QA watermark; never use for real results')
    args=parser.parse_args()
    root=args.root.expanduser().resolve();out=(args.output_dir or root).expanduser().resolve()
    report=normalize(root,args.synthetic)
    out.mkdir(parents=True,exist_ok=True)
    png=chart(report);page=make_html(report,png,out)
    for name,payload in [('results.png',png),('results.html',page.encode())]:
        temp=out/f'.{name}.{os.getpid()}.tmp';temp.write_bytes(payload);temp.replace(out/name)
    print(json.dumps({'status':report['status'],'synthetic':report['synthetic'],
        'counts':aggregate(report['rows']),'results_png':str(out/'results.png'),
        'results_html':str(out/'results.html')},ensure_ascii=False))


if __name__=='__main__':main()
