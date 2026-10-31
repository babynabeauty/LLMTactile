"""Read-only evidence gallery for direct simulation demonstrations (CPU only).

Only this report directory is written. Original runs, images, demonstrations,
verification records and the dataset manifest are never edited or inferred as
successful from their filenames. Optional video uses already saved frames.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import html
import json
import os
from pathlib import Path
import shutil
import subprocess
from urllib.parse import quote

import numpy as np

from .residual_rl.solver_profile import normalize_solver_profile,validate_solver_readback


LABELS={'cuboid':'长方体','hex_prism':'正六棱柱','sphere':'球体','cylinder':'直立瓶状圆柱'}
FINGERS=('thumb','index','middle','ring','little')
VIEWS={'scene_front':'前方整体','scene_left':'左侧整体','hand_thumb':'拇指侧手部'}


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


def read(path):return json.loads(Path(path).read_text())


def resolve(root,value):
    path=Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root/path).resolve()


def relative_link(path,output):
    return quote(os.path.relpath(Path(path),Path(output)),safe='/')


def checked_json(path,expected):
    if not expected or sha(path)!=expected:raise ValueError(f'Evidence hash mismatch: {path}')
    return read(path)


def declared_solver_profile(record):
    """Preserve absent legacy provenance instead of relabeling an old run."""
    value=record.get('solver_profile',record.get('physics_profile'))
    return None if value is None else normalize_solver_profile(value)


def audit_report_solver_profile(dataset,metadata,verification,source_manifest,scene):
    expected=declared_solver_profile(dataset)
    declarations={name:declared_solver_profile(record) for name,record in (
        ('trajectory_metadata',metadata),('independent_verification',verification),
        ('source_manifest',source_manifest),('source_scene',scene))}
    actual=[profile for profile in declarations.values() if profile is not None]
    if expected is not None:
        for name,profile in declarations.items():
            if profile!=expected:
                raise ValueError(f'Solver profile mismatch or absent explicit provenance: {name}')
    elif actual:
        raise ValueError('Explicit run solver profile requires an explicit dataset solver profile')
    source_readback=replay_readback=None
    if expected is not None:
        source_readback=validate_solver_readback(expected,scene.get('solver_readback'))
        validate_solver_readback(expected,metadata.get('source_solver_readback'))
        replay_readback=validate_solver_readback(expected,verification.get('solver_readback'))
    return dict(dataset_declared=expected is not None,solver_profile=expected,
        declarations=declarations,source_solver_readback=source_readback,
        independent_replay_solver_readback=replay_readback,
        legacy_note=None if expected is not None else
        'This legacy dataset does not explicitly declare a solver profile. No new profile has been assigned to its original evidence.')


def verification_record(root,entry):
    """A manifest declaration alone never certifies a successful replay."""
    bind_path=resolve(root,entry['bind_record']);bind=read(bind_path)
    evidence=bind['demonstration_evidence']
    if evidence.get('demonstration_source')!='direct_teacher_search' or evidence.get('privileged_teacher') is not True:
        raise ValueError('Expected honest privileged direct-teacher evidence')
    verification_path=resolve(root,evidence['verification_path'])
    verification=checked_json(verification_path,evidence['verification_sha256'])
    metadata_path=resolve(root,evidence['trajectory_metadata_path'])
    metadata=checked_json(metadata_path,evidence['trajectory_metadata_sha256'])
    trajectory=resolve(root,bind['trajectory_path'])
    if sha(trajectory)!=bind['trajectory_sha256'] or metadata['trajectory_sha256']!=bind['trajectory_sha256']:
        raise ValueError('Demonstration command trajectory hash mismatch')
    if (verification.get('accepted') is not True or verification.get('audit_errors')
            or verification.get('trace_recomputed',{}).get('task_success') is not True
            or evidence.get('task_success') is not True):
        raise ValueError('Independent replay did not pass the original success criterion')
    if metadata.get('demonstration_source')!='direct_teacher_search':
        raise ValueError('Trajectory metadata is not a direct teacher demonstration')
    if metadata['object_id']!=bind['object_id']:
        raise ValueError('Demonstration shape identity mismatch')
    source=resolve(root,metadata['source_run'])
    if 'source_run' in entry and resolve(root,entry['source_run'])!=source:
        raise ValueError('Dataset source run differs from verified trajectory provenance')
    return dict(bind=bind,metadata=metadata,verification=verification,source=source,
        bundle=metadata_path.parent,bind_path=bind_path,verification_path=verification_path,
        metadata_path=metadata_path,trajectory_path=trajectory)


def attempt_records(root,shape,source=None,collection_roots=()):
    rows=[]
    # All named searches in this collection, including reference_attempt and
    # tripod_attempt, are part of the retained search cost.
    paths=set((root/shape).glob('*attempt*/manifest.json'))
    for collection in collection_roots:
        paths.update(Path(collection).glob(f'{shape}_*attempt*/manifest.json'))
    if source is not None:
        source=Path(source)
        paths.add(source/'manifest.json')
        # Curved-object collections use a separate root and shape-prefixed
        # directories. Read only siblings of the explicitly declared source.
        paths.update(source.parent.glob(f'{shape}_*attempt*/manifest.json'))
    for path in sorted(paths):
        if not path.is_file():continue
        manifest=read(path)
        if manifest.get('driver')!='direct_teacher_search':continue
        if manifest.get('object_spec',{}).get('shape',shape)!=shape:continue
        score_path=path.parent/'score.json'
        score=read(score_path) if score_path.is_file() else None
        rows.append(dict(path=str(path.parent),name=path.parent.name,
            completed=score is not None,source_success=(score or {}).get('success') is True,
            fault=(score or {}).get('fault'),score_path=str(score_path) if score else None,
            solver_profile=declared_solver_profile(manifest),
            saved_file_mtime_ns=max(path.stat().st_mtime_ns,score_path.stat().st_mtime_ns if score else 0)))
    return sorted(rows,key=lambda row:(row['saved_file_mtime_ns'],row['path']))


def source_series(evidence_dir,manifest):
    values=[]
    with (evidence_dir/'evaluation.jsonl').open() as stream:
        for line in stream:
            if line.strip():values.append(json.loads(line))
    if not values:raise ValueError('No physical source samples')
    times=np.asarray([v['sample_time_s'] for v in values],float)
    clearance=np.asarray([v['bottom_clearance_m'] for v in values],float)
    contact=np.asarray([v['all_robot_contact_load_N'] for v in values],float)
    normal=np.asarray([v['all_robot_normal_load_N'] for v in values],float)
    phases=np.asarray([v['phase'] for v in values])
    with np.load(evidence_dir/'raw_tactile.npz',allow_pickle=False) as raw:
        tactile=np.asarray(raw['tactile'])
        raw_times=np.asarray(raw['sample_time_s'])
        gains=np.asarray(raw['grip_gain'])
    if tactile.shape!=(len(times),5,120,3):raise ValueError('Raw tactile sample/shape mismatch')
    if raw_times.shape!=times.shape or not np.allclose(raw_times,times,rtol=0,atol=1e-8):
        raise ValueError('Physical force and raw tactile clocks differ')
    if any(not np.isfinite(x).all() for x in (times,clearance,contact,normal,tactile,gains)):
        raise ValueError('Nonfinite saved source signals')
    if len(times)>1 and not np.allclose(np.diff(times),1/60,rtol=0,atol=1e-8):
        raise ValueError('Source samples are not contiguous at60Hz')
    order=tuple('little' if name=='pinky' else name for name in manifest.get('sensor_finger_order',FINGERS))
    if order!=FINGERS:
        raise ValueError('Unexpected tactile finger order')
    hold=(phases=='hold') & (times>times[-1]-3+1e-8)
    taxel_norm=np.linalg.norm(tactile,axis=-1)
    return dict(time=times,clearance=clearance,contact=contact,normal=normal,phases=phases,
        tactile_peak=taxel_norm.max(-1),taxel_norm=taxel_norm,
        evaluation_records=values,gains=gains,hold=hold)


def tactile_coverage(series,scene):
    """Separate physical contact evidence from uncalibrated taxel responses."""
    layouts={('little' if row['finger']=='pinky' else row['finger']):row['link_name']
             for row in scene['tactile_layout']}
    if set(layouts)!=set(FINGERS):raise ValueError('Incomplete frozen finger/link mapping')
    hold=series['hold'];count=int(hold.sum())
    if count==0:raise ValueError('No final hold samples for tactile coverage statistics')
    values=series['taxel_norm'][hold];records=[row for row,take in zip(series['evaluation_records'],hold) if take]
    physical=np.zeros((count,len(FINGERS)))
    for sample,row in enumerate(records):
        for contact in row.get('solver_contacts',[]):
            if 'cube' not in (contact.get('entity_a'),contact.get('entity_b')):continue
            force=float(np.linalg.norm(contact['force_on_a_world_N']))
            for finger,name in enumerate(FINGERS):
                if layouts[name] in (contact.get('link_a'),contact.get('link_b')):
                    physical[sample,finger]+=force
    active=(values>1e-6).sum(-1);summed=values.sum(-1);rows=[]
    for finger,name in enumerate(FINGERS):
        contacted=physical[:,finger]>.05
        rows.append(dict(finger=name,sensor_link=layouts[name],taxels_per_finger=int(values.shape[-1]),
            physical_sensor_link_mean_contact_force_N=float(physical[:,finger].mean()),
            physical_contact_samples=int(contacted.sum()),
            active_taxels_mean=float(active[:,finger].mean()),active_taxels_min=int(active[:,finger].min()),
            active_taxels_max=int(active[:,finger].max()),
            taxel_norm_sum_mean_uncalibrated_nominal_N=float(summed[:,finger].mean()),
            contact_samples_with_zero_active_taxels=int((contacted & (active[:,finger]==0)).sum()),
            sparse_response_flag=bool(physical[:,finger].mean()>.05 and active[:,finger].mean()<5)))
    return dict(window='Final 3 seconds of source hold, same samples as source plots',
        start_time_s_exclusive=float(series['time'][-1]-3),end_time_s_inclusive=float(series['time'][-1]),
        sample_count=count,active_threshold_uncalibrated_nominal_N=1e-6,
        physical_contact_threshold_N=.05,
        physical_measurement='Per-sample sum of solver contact-force magnitudes between the object and the sensor-bearing rigid link; averaged over the hold window.',
        signal_measurement='Per-taxel three-axis vector norm, then summed across 120 taxels and averaged over the hold window. This is an uncalibrated signal statistic, not physical force.',
        sparse_flag_definition='Descriptive flag only: physical sensor-link mean contact >0.05N and mean active taxels <5/120. It does not measure contact area or a missed-force percentage.',
        interpretation='A small number of responding taxels can reflect localized contact and does not by itself prove missed contact or sensor failure. Physical grasp success does not establish complete tactile coverage; a separate surface/measurement-layout audit is needed for stronger coverage claims.',
        fingers=rows)


def plot_signals(series,destination,title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'figure.dpi':140,'savefig.dpi':180})
    fig,axes=plt.subplots(4,1,figsize=(10,9.2),sharex=True,layout='constrained',
                          gridspec_kw={'height_ratios':[1,1.2,1.2,.65]})
    t=series['time'];hold=series['hold']
    for ax in axes:
        ax.grid(alpha=.2)
        if hold.any():ax.axvspan(t[-1]-3,t[-1],color='#e4f4df',zorder=0)
    axes[0].plot(t,series['clearance']*1000,color='#235ca6',label='Actual lowest surface')
    axes[0].axhline(50,color='#ad3434',linestyle='--',label='Success threshold: 50 mm')
    axes[0].set_ylabel('Table clearance (mm)');axes[0].legend(loc='upper left',ncol=2)
    axes[1].plot(t,series['contact'],label='Sum of contact-force magnitudes',color='#196f59')
    axes[1].plot(t,series['normal'],label='Sum of normal-force magnitudes',color='#d08525',alpha=.85)
    axes[1].set_ylabel('Physical solver force (N)');axes[1].legend(loc='upper left')
    for i,finger in enumerate(FINGERS):axes[2].plot(t,series['tactile_peak'][:,i],label=finger,linewidth=1.1)
    axes[2].set_ylabel('Peak taxel norm\n(uncalibrated nominal N)');axes[2].legend(ncol=5,loc='upper left')
    axes[3].step(t,series['gains'],where='post',color='#654f95')
    axes[3].set_ylabel('Thumb/index\nabsolute PD gain');axes[3].set_xlabel('Source simulation time (s)')
    last=None
    for time,phase in zip(t,series['phases']):
        if phase!=last:
            for ax in axes:ax.axvline(time,color='gray',alpha=.2,linewidth=.6)
            axes[0].text(time,1.02,phase,transform=axes[0].get_xaxis_transform(),rotation=55,
                         va='bottom',ha='left',fontsize=8,color='#56616d')
            last=phase
    fig.suptitle(title+' — saved source trajectory\nGreen shading: final 3 s; tactile estimates and physical forces are different measurements',fontsize=11)
    fig.savefig(destination.with_suffix('.png'));fig.savefig(destination.with_suffix('.pdf'));plt.close(fig)


def held_frames(source):
    stages=read(source/'stages.json') if (source/'stages.json').is_file() else []
    held=next((s for s in reversed(stages) if s.get('name')=='held'),None)
    step=int(held['step']) if held else None
    rows=[]
    for view,label in VIEWS.items():
        candidates=sorted((source/'frames').glob(f'*_{view}.jpg'))
        exact=source/'frames'/f'{step:06d}_{view}.jpg' if step is not None else None
        image=exact if exact is not None and exact.is_file() else (candidates[-1] if candidates else None)
        if image:
            actual_step=int(image.name.split('_',1)[0])
            rows.append(dict(view=view,label=label,path=str(image),sha256=sha(image),
                sample_time_s=actual_step/60,is_exact_held_frame=bool(held and actual_step==step)))
    return rows


def make_video(source,destination):
    executable=shutil.which('ffmpeg')
    if not executable:return {'status':'unavailable','reason':'ffmpeg is not installed'}
    frames=sorted((source/'frames').glob('*_scene_front.jpg'))
    if not frames:return {'status':'unavailable','reason':'No saved front-view frames'}
    script=destination.with_suffix('.concat.txt');lines=[]
    for i,path in enumerate(frames):
        escaped=str(path.resolve()).replace("'","'\\''")
        lines.append(f"file '{escaped}'")
        if i+1<len(frames):
            duration=(int(frames[i+1].name.split('_')[0])-int(path.name.split('_')[0]))/60
            lines.append(f'duration {duration:.9f}')
    lines.append(lines[-1] if lines[-1].startswith('file ') else f"file '{str(frames[-1].resolve())}'")
    script.write_text('\n'.join(lines)+'\n')
    result=subprocess.run([executable,'-v','error','-y','-f','concat','-safe','0','-i',str(script),
        '-vf','scale=960:-2,fps=20','-c:v','libx264','-pix_fmt','yuv420p','-movflags','+faststart',str(destination)],
        capture_output=True,text=True,timeout=180)
    if result.returncode:return {'status':'failed','reason':result.stderr[-1500:]}
    return dict(status='created',path=str(destination),sha256=sha(destination),
        source_frames=len(frames),note='Saved source front-view frames at their recorded simulation-time spacing; no synthesized viewpoints.')


def dimensions(record):
    spec=record.get('object_spec',record);shape=record.get('object_id',spec.get('shape'))
    if shape=='cuboid':return ' × '.join(f'{1000*v:g}' for v in spec.get('size_m',spec.get('dimensions_m',[.045,.060,.045])))+' mm'
    if shape=='hex_prism':return f"X 对边距 {1000*spec.get('across_flats_m',spec.get('across_x_flats_m',.045)):g} mm · 高 {1000*spec.get('height_m',.045):g} mm"
    diameter=spec.get('diameter_m',2*spec.get('radius_m',.0225))
    return f'直径 {1000*diameter:g} mm'+(f" · 高 {1000*spec.get('height_m',.09):g} mm" if shape=='cylinder' else '')


def build_report(dataset_root,output=None,manifest_path=None,video=False):
    root=Path(dataset_root).expanduser().resolve();output=Path(output).resolve() if output else root/'report'
    manifest_path=Path(manifest_path).resolve() if manifest_path else root/'dataset_manifest.json'
    dataset=read(manifest_path);output.mkdir(parents=True,exist_ok=True);(output/'plots').mkdir(exist_ok=True)
    # Canonically declared source collections may contain pending shapes too.
    # This keeps unsuccessful sphere attempts visible before sphere is bound.
    collection_roots={resolve(root,path) for path in dataset.get('collection_roots',[])} | {resolve(root,entry['source_run']).parent
                      for entry in dataset.get('verified',{}).values() if entry.get('source_run')}
    cards=[]
    for declaration in dataset['objects']:
        shape=declaration['object_id'];attempts=attempt_records(root,shape,collection_roots=collection_roots)
        card=dict(object_id=shape,label=LABELS.get(shape,shape),dimensions=dimensions(declaration),
            status='pending',attempts=attempts,verified_replays=0,verification_trials=0,
            source_successes=sum(r['source_success'] for r in attempts),
            completed_source_attempts=sum(r['completed'] for r in attempts),warnings=[],images=[])
        entry=dataset.get('verified',{}).get(shape)
        if entry:
            try:
                record=verification_record(root,entry)
                if record['bind']['object_id']!=shape:raise ValueError('Dataset object differs from bind identity')
                source=record['source'];evidence_dir=record['bundle']/'source_evidence'
                attempts=attempt_records(root,shape,source,collection_roots)
                card.update(attempts=attempts,
                    source_successes=sum(r['source_success'] for r in attempts),
                    completed_source_attempts=sum(r['completed'] for r in attempts))
                metadata=record['metadata'];source_manifest=read(evidence_dir/'manifest.json')
                # Revalidate the two immutable data files that generate figures.
                for filename in ('manifest.json','evaluation.jsonl','raw_tactile.npz','object_scene.json'):
                    original=str(source/filename);frozen=evidence_dir/filename
                    if sha(frozen)!=metadata['source_sha256'][original]:raise ValueError('Frozen source data hash mismatch: '+filename)
                scene=read(evidence_dir/'object_scene.json')
                solver_audit=audit_report_solver_profile(dataset,metadata,record['verification'],source_manifest,scene)
                signals=source_series(evidence_dir,source_manifest)
                coverage=tactile_coverage(signals,scene)
                coverage['source_evidence']={filename:{'path':str(evidence_dir/filename),
                    'sha256':sha(evidence_dir/filename)} for filename in ('evaluation.jsonl','raw_tactile.npz','object_scene.json')}
                coverage_path=output/f'{shape}_tactile_coverage.json'
                coverage_path.write_text(json.dumps(coverage,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
                plot=output/'plots'/shape;plot_signals(signals,plot,shape)
                card.update(status='verified',verified_replays=1,verification_trials=1,
                    source_run=str(source),source_score=read(evidence_dir/'score.json'),
                    source_metrics=metadata['source_physics_audit'],
                    replay_metrics=record['verification']['trace_recomputed'],
                    dimensions=dimensions(record['bind']),material=metadata['material'],
                    hand_shape=entry.get('hand_shape',source_manifest.get('parameters',{}).get('name','未命名手形')),
                    parameters=source_manifest.get('parameters',{}),images=held_frames(source),
                    plot_png=str(plot.with_suffix('.png')),plot_pdf=str(plot.with_suffix('.pdf')),
                    verification_path=str(record['verification_path']),bind_record=str(record['bind_path']),
                    trajectory_metadata_path=str(record['metadata_path']),trajectory_path=str(record['trajectory_path']),
                    raw_tactile_path=str(evidence_dir/'raw_tactile.npz'),evaluation_path=str(evidence_dir/'evaluation.jsonl'),
                    tactile_coverage=coverage,tactile_coverage_path=str(coverage_path),
                    solver_profile_audit=solver_audit,
                    verification_sha256=sha(record['verification_path']))
                if video:card['video']=make_video(source,output/f'{shape}_source.mp4')
                if not card['images']:card['warnings'].append('源运行没有保存展示帧；未生成替代图。')
            except (OSError,ValueError,KeyError,TypeError) as exc:
                card.update(status='evidence_error',verification_trials=1)
                card['warnings'].append(str(exc))
        elif attempts:
            latest=attempts[-1]
            card.update(status='source_success_pending_verification' if latest['source_success'] else
                        'failed_attempt_pending' if latest['completed'] else 'collecting',
                        latest_attempt=latest)
        cards.append(card)
    report=dict(schema='direct_teacher_gallery_v2',created_utc=datetime.now(timezone.utc).isoformat(),
        dataset_manifest_path=str(manifest_path),dataset_manifest_sha256=sha(manifest_path),
        dataset_root=str(root),dataset_name=root.name,solver_profile=declared_solver_profile(dataset),
        profile_reason=dataset.get('profile_reason'),previous_dataset=dataset.get('previous_dataset'),
        generator_sha256=sha(Path(__file__)),dataset_status=dataset.get('status'),
        search_collection_roots=[str(path) for path in sorted(collection_roots)],
        driver='direct_teacher_search',privileged_teacher=True,actual_astra_api=False,
        formal_ppo_training_started=dataset.get('formal_ppo_training_started',False),hardware_connected=False,
        interpretation='Each verified entry is one selected successful source trajectory plus one fresh replay acceptance. This is not a statistical policy success-rate estimate. Search attempts are retained separately.',cards=cards)
    (output/'report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    (output/'index.html').write_text(render_html(report,output))
    return report


def render_html(report,output):
    esc=html.escape
    def link(path,label):return f'<a href="{relative_link(path,output)}">{esc(label)}</a>'
    def number(value,scale=1):return '—' if value is None else f'{value*scale:.2f}'
    profile=report.get('solver_profile')
    if profile is None:
        profile_banner='<p class="note"><b>求解器版本：旧清单未显式声明。</b>本报告保留原始证据，没有给旧运行重新指定新求解器设置；不能与新版本直接合并统计。</p>'
    else:
        profile_banner=(f'<p class="note"><b>本数据集统一求解器：noslip_iterations={profile["noslip_iterations"]} · '
            f'noslip_tolerance={profile["noslip_tolerance"]:g}。</b>这是求解器的无滑移后处理配置，物体接触摩擦仍单独记录。'
            '每条已验收示范核对源采集、冻结轨迹、独立回放以及两次实际求解器读回；只统计本版本数据。</p>')
    if report.get('profile_reason'):
        profile_banner+='<details><summary>版本变更说明</summary><p>'+esc(report['profile_reason'])+'</p></details>'
    if report.get('previous_dataset'):
        old=Path(report['previous_dataset'])/'report'/'index.html'
        profile_banner+='<p>'+link(old,'查看保留的上一版报告')+'；上一版数据没有迁入或重新标记为本版本。</p>'
    sections=[]
    statuses={'verified':'源示范与独立回放均通过','pending':'尚未采集完成','collecting':'采集中',
        'failed_attempt_pending':'最近一次源采集失败，仍待成功示范',
        'source_success_pending_verification':'源采集成功，独立回放尚未验收','evidence_error':'证据核验异常，不能标记通过'}
    for card in report['cards']:
        block=[f'<section id="{esc(card["object_id"])}"><h2>{esc(card["label"])}</h2>',
            f'<p class="badge {esc(card["status"])}">{esc(statuses[card["status"]])}</p>',
            f'<p>{esc(card["dimensions"])} · 源搜索已完成 {card["completed_source_attempts"]} 次，其中 {card["source_successes"]} 次源成功。</p>']
        if card['status']=='verified':
            source,replay=card['source_metrics'],card['replay_metrics'];material=card['material']
            block.append(f'<p>独立回放验收：<b>1 / 1</b>（单条示范验收，不能视为统计成功率）。质量 {1000*material["mass_kg"]:.2f} g · 接触摩擦 μ={material["friction"]:g} · 手形：{esc(card["hand_shape"])}。</p>')
            block.append('<table><tr><th>实际测量</th><th>源采集</th><th>独立回放</th></tr>')
            for label,key,scale in [('最后3秒最低离桌高度 mm','hold_min_clearance_m',1000),
                ('保持阶段全接触力均值 N','hold_mean_all_robot_contact_force_N',1),
                ('保持阶段法向力均值 N','hold_mean_all_robot_normal_force_N',1),
                ('保持阶段全接触力峰值 N','hold_peak_all_robot_contact_force_N',1)]:
                block.append(f'<tr><td>{label}</td><td>{number(source.get(key),scale)}</td><td>{number(replay.get(key),scale)}</td></tr>')
            block.append('</table><div class="gallery">')
            for image in card['images']:
                label=image['label']+(' · held实际帧' if image['is_exact_held_frame'] else ' · 最后保存帧')
                block.append(f'<figure><a href="{relative_link(image["path"],output)}"><img loading="lazy" src="{relative_link(image["path"],output)}" alt="{esc(label)}"></a><figcaption>{esc(label)} · {image["sample_time_s"]:.2f} s</figcaption></figure>')
            block.append('</div>')
            block.append(f'<a href="{relative_link(card["plot_png"],output)}"><img class="plot" src="{relative_link(card["plot_png"],output)}" alt="实际源轨迹的离桌高度、物理接触力、五指触觉和PD增益时间曲线"></a>')
            block.append('<p class="note">触觉图显示每指 120 个测点中最大的三轴信号模，单位为未标定 nominal N；物理力图来自求解器接触力，两者不可直接当作同一种力比较。PD 曲线是拇指／食指共同增益；三指手形的中指保持原始增益 1。曲线和照片来自源采集；上表单列独立回放结果。</p>')
            coverage=card['tactile_coverage']
            block.append(f'<h3>触觉覆盖核查：源保持阶段 {coverage["sample_count"]} 帧</h3><table><tr><th>手指</th><th>传感器所在刚体的物理接触力均值 N</th><th>响应测点均值 / 120</th><th>测点信号模之和均值<br>未标定 nominal N，非物理力</th></tr>')
            finger_names={'thumb':'拇指','index':'食指','middle':'中指','ring':'无名指','little':'小指'}
            for row in coverage['fingers']:
                block.append(f'<tr><td>{finger_names[row["finger"]]}</td><td>{row["physical_sensor_link_mean_contact_force_N"]:.3f}</td><td>{row["active_taxels_mean"]:.2f}</td><td>{row["taxel_norm_sum_mean_uncalibrated_nominal_N"]:.3f}</td></tr>')
            block.append('</table><p class="note">响应测点定义为三轴信号模大于 10⁻⁶ nominal N；这些统计使用与曲线相同的最后3秒。测点信号之和是未标定信号统计，不能用作实际抓握力，也不能据此计算漏测力比例或接触面积。'+link(card['tactile_coverage_path'],'查看完整统计、阈值与原始证据哈希')+'。</p>')
            sparse=[finger_names[row['finger']] for row in coverage['fingers'] if row['sparse_response_flag']]
            if sparse:
                block.append('<p class="note"><b>触觉响应集中于少数测点：</b>'+esc('、'.join(sparse))+'存在真实物理接触，平均响应不足5个测点。这可以来自局部小区域接触，本身不能证明缺测或传感器故障；也不能用信号模之和与物理力的比例判断漏测程度。完整覆盖判断仍需结合接触位置、表面法向与测点重叠审计。报告没有改动触觉算法或模型输入。</p>')
            references=[('verification_path','独立回放验收'),('bind_record','验收绑定记录'),
                ('trajectory_metadata_path','轨迹来源与参数'),('trajectory_path','实际关节指令'),
                ('raw_tactile_path','原始60Hz触觉 NPZ'),('evaluation_path','物理接触记录'),('plot_pdf','导出曲线 PDF')]
            block.append('<p>'+' · '.join(link(card[key],label) for key,label in references)+'</p>')
            if card.get('video',{}).get('status')=='created':
                block.append(f'<video controls preload="metadata" src="{relative_link(card["video"]["path"],output)}"></video><p>视频仅按真实时间间隔串联已保存的源前方帧。</p>')
            block.append('<details><summary>本条手形／控制参数</summary><pre>'+esc(json.dumps(card['parameters'],indent=2,ensure_ascii=False))+'</pre></details>')
        elif card.get('latest_attempt',{}).get('fault'):
            block.append('<p>最近失败原因：'+esc(card['latest_attempt']['fault'])+'</p>')
        if card['warnings']:block.append('<p class="warning">'+'<br>'.join(esc(x) for x in card['warnings'])+'</p>')
        if card['attempts']:
            block.append('<details><summary>保留的搜索尝试（不是独立策略评测）</summary><ul>')
            for attempt in card['attempts']:
                state='源成功' if attempt['source_success'] else ('源失败' if attempt['completed'] else '进行中')
                label=attempt['name']+' · '+state
                attempt_profile=attempt.get('solver_profile')
                label+=(' · noslip='+str(attempt_profile['noslip_iterations'])) if attempt_profile else ' · 旧记录未显式声明求解器配置'
                block.append('<li>'+(link(attempt['score_path'],label) if attempt['score_path'] else esc(label))+'</li>')
            block.append('</ul></details>')
        block.append('</section>');sections.append(''.join(block))
    return '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>直接仿真示范 · 抓取与触觉记录</title>
<style>body{font:16px/1.65 system-ui,sans-serif;max-width:1320px;margin:32px auto;padding:0 24px;background:#f2f5fa;color:#192a3b}h1{font-size:30px}h2{margin:0;font-size:25px}section{background:white;padding:26px;margin:26px 0;border:1px solid #dde5ee;border-radius:12px}.note{padding:14px;background:#edf3ff;border-left:4px solid #6287ba}.badge{display:inline-block;padding:5px 10px;background:#fff1d8;border-radius:6px}.badge.verified{background:#def2e4}.warning,.evidence_error{color:#9c302b}.gallery{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin-top:22px}figure{margin:0}figure img{width:100%;border-radius:6px}figcaption{font-size:14px;color:#536475}table{border-collapse:collapse;width:100%}td,th{text-align:left;padding:8px;border-bottom:1px solid #dde5ee}.plot{display:block;width:100%;max-width:1100px;margin:24px auto}a{color:#225ca6}video{width:100%;max-width:960px}pre{white-space:pre-wrap;font-size:13px}details{margin-top:15px}@media(max-width:800px){.gallery{grid-template-columns:1fr}body{padding:0 10px}section{padding:16px}}</style>
<h1>直接仿真示范：抓取与触觉记录</h1><p class="note">这些示范由可读取物体几何、位姿及接触信息的规则教师直接在仿真中采集。没有调用 Astra，没有进行 PPO 训练，没有连接或控制真实硬件。每个物体单独展示源示范和独立回放；不能用少量选出的成功轨迹推断泛化成功率。</p>
<p>物理 300 Hz · 触觉记录 60 Hz · 动作更新 15 Hz。成功判据：最终 3 秒内真实最低表面始终离桌 ≥5 cm，且无机器人桌面／自身碰撞。</p>'''+f'<p>数据集：<code>{esc(report.get("dataset_name","未命名"))}</code></p>'+profile_banner+''.join(sections)+f'<p>生成时间：{esc(report["created_utc"])} · {link(report["dataset_manifest_path"],"数据集清单")} · <a href="report.json">报告与证据哈希</a></p></html>'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root',type=Path,required=True)
    parser.add_argument('--manifest',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--video',action='store_true',help='Optional source-frame MP4, only when ffmpeg is installed')
    args=parser.parse_args();report=build_report(args.dataset_root,args.output,args.manifest,args.video)
    output=args.output.resolve() if args.output else args.dataset_root.resolve()/'report'
    print(json.dumps({'index':str(output/'index.html'),'objects':{c['object_id']:c['status'] for c in report['cards']}},ensure_ascii=False))


if __name__=='__main__':main()
