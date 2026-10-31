"""Audit causal 200 Hz slip experiments; export separate training targets and review media."""
import argparse
import hashlib
import html
import json
from pathlib import Path
import subprocess
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from slip_recovery_experiment import TactileTrigger, CONFIG

ROOT=Path(__file__).resolve().parents[1]
NAMES={'mass200_monitor':'200 g · 仅监测','lowmu_monitor':'μ=0.01 · 仅监测',
       'mass100_monitor':'100 g · 成功对照','mass200_reactive':'200 g · 触觉触发增力',
       'lowmu_reactive':'μ=0.01 · 触觉触发增力'}
load=lambda p:json.loads(p.read_text())
lines=lambda p:[json.loads(s) for s in p.read_text().splitlines()]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def first_persistent(mask,n=3):
    hits=np.flatnonzero(np.convolve(mask.astype(int),np.ones(n,dtype=int),'valid')==n)
    return None if not len(hits) else int(hits[0])
def resultants(rows):
    result=[]
    for row in rows:
        vectors=np.zeros((2,3))
        for c in row['solver_contacts']:
            if {c['entity_a'],c['entity_b']}!={'cube','ur7e_xhand'}:continue
            link=c['link_b'] if c['entity_a']=='cube' else c['link_a']
            force=np.array(c['force_on_a_world_N'])*(1 if c['entity_a']=='cube' else -1)
            for j,finger in enumerate(('thumb','index')):
                if finger in link:vectors[j]+=force
        result.append(np.linalg.norm(vectors,axis=1))
    return np.array(result)

def main():
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);p.add_argument('--skip-video',action='store_true');a=p.parse_args();out=a.directory.resolve()
    protocol=load(out/'protocol.json')
    drift=[]
    for source,h in protocol['source_sha256'].items():
        if sha(Path(source))==h:continue
        review=load(out/'source_drift_review.json')
        assert source==review['file'] and sha(Path(source))==review['reviewed_current_sha256'],'Unreviewed source change'
        drift.append(review)
    demo=Path(protocol['demonstration']);assert sha(demo/'control.jsonl')==protocol['trajectory_sha256']
    assert sha(demo/'reference.json')==protocol['reference_sha256']
    ref=lines(demo/'control.jsonl');targets=np.array([x['sent_target_rad'] for x in ref]);phase40=np.array([x['phase'] for x in ref])
    layout_path=ROOT/'ur7e_xhand_sim/generated/tactile_layout.json';layouts=load(layout_path)
    (out/'tactile_layout.json').write_bytes(layout_path.read_bytes())
    positions=np.array([x['positions_link_m'] for x in layouts[:2]])
    reports=[];data={}
    for case in protocol['cases']:
        folder=out/case;s=load(folder/'score.json');material=load(folder/'material_verification.json')
        obs=lines(folder/'control.jsonl');ev=lines(folder/'evaluation.jsonl');events=lines(folder/'highfreq_events.jsonl')
        inp=np.load(folder/'inputs_200hz.npz');lab=np.load(folder/'labels_200hz.npz');old=np.load(folder/'trajectory_tactile.npz')
        t=inp['time_s'];n=len(t);assert n==6375 and len(obs)==len(ev)==1275 and len(events)==n
        np.testing.assert_allclose(np.diff(t),.005,atol=1e-13,rtol=0)
        for arr in (inp['force_link_N'],inp['q_rad'],inp['target_q_rad'],lab['contact_tangent_speed_m_s']):assert np.isfinite(arr).all()
        np.testing.assert_allclose(np.array([x['sent_target_rad'] for x in obs]),targets,atol=1e-12,rtol=0)
        np.testing.assert_allclose(inp['target_q_rad'],np.repeat(targets,5,axis=0),atol=1.3e-7,rtol=0)
        np.testing.assert_allclose(inp['force_link_N'][4::5],old['force_link_N'],atol=1e-7,rtol=0)
        assert material['all_checks_passed']
        np.testing.assert_allclose(material['engine_mass_kg'],material['resolved']['cube_mass_kg'],rtol=1e-5)
        np.testing.assert_allclose(material['engine_mass_divided_by_volume_kg_m3'],material['material_density_kg_m3'],rtol=1e-5)
        for pair,mu in [('robot_cube',s['contact_mu']),('table_cube',.8)]:
            m=material['observed_pair_friction'][pair];assert m['contact_samples']>0
            np.testing.assert_allclose([m['min'],m['max']],mu,atol=1e-6,rtol=0)
        armed=np.arange(n)>=855*5
        detector=TactileTrigger(positions);features=[detector.update(f,bool(on)) for f,on in zip(inp['force_link_N'],armed)]
        assert [f['event'] for f in features]==[e['features']['event'] for e in events]
        assert [f['candidate'] for f in features]==[e['features']['candidate'] for e in events]
        eventids=np.flatnonzero([f['event'] for f in features]);assert len(eventids)<=1
        k=int(eventids[0]) if len(eventids) else None;trigger=None if k is None else float(t[k]);assert trigger==s['trigger_time_s']
        expected_gain=np.ones(n)
        if k is not None and s['reactive']:
            u=np.clip((np.arange(n)-k)/10.,0,1);expected_gain=1+u*u*(3-2*u)
        np.testing.assert_allclose(inp['gain_applied'],expected_gain,atol=1e-7,rtol=0)
        gainids=np.flatnonzero(inp['gain_applied']>1);gain_k=int(gainids[0]) if len(gainids) else None
        h=lab['cube_position_m'][:,2]-.0225*np.abs(lab['cube_rotation'][:,2,:]).sum(axis=1)-.2
        np.testing.assert_allclose(h,lab['bottom_clearance_m'],atol=1e-12,rtol=0)
        np.testing.assert_allclose(h[4::5],[x['bottom_clearance_m'] for x in ev],atol=1e-9,rtol=0)
        hold=np.repeat(phase40=='hold',5);success=bool(h[hold].min()>=.05 and not s['fault']);assert success==s['success']
        speed=lab['contact_tangent_speed_m_s'][:,:2];loadN=lab['contact_force_norm_sum_N'][:,:2]
        contact=lab['contact_count'].sum(axis=1)>0
        slip=np.any((speed>.005)&(loadN>.05),axis=1)&armed
        onset=first_persistent(slip)
        # Onset labels refer to the start of a sustained run; certification is two samples later.
        sustained=np.zeros(n,dtype=bool)
        for idx in np.flatnonzero(np.convolve(slip.astype(int),np.ones(3,dtype=int),'valid')==3):sustained[idx:idx+3]=True
        contactids=np.flatnonzero(contact);loss_k=int(contactids[-1]+1) if len(contactids) and contactids[-1]<n-1 else None
        loss=None if loss_k is None else float(t[loss_k])
        labels={'time_s':t,'armed':armed,'has_hand_contact':contact,'slip_proxy_instantaneous':slip,
                'slip_proxy_sustained_15ms':sustained,'kp_multiplier':inp['gain_applied']}
        for ms in (250,500,1000):
            steps=ms//5;valid=armed&contact&(np.arange(n)+steps<n)
            if gain_k is not None:valid&=np.arange(n)+steps<gain_k
            future=np.zeros(n,dtype=bool)
            if loss_k is not None:future=(np.arange(n)<loss_k)&(np.arange(n)+steps>=loss_k)
            labels[f'final_contact_loss_within_{ms}ms']=future
            labels[f'valid_loss_target_{ms}ms']=valid
        np.savez_compressed(folder/'training_labels.npz',**labels)
        forbidden=[c for v in ev for c in v['solver_contacts'] if c['entity_a']==c['entity_b']=='ur7e_xhand' or 'table' in (c['entity_a'],c['entity_b'])]
        forces=resultants(ev);t40=t[4::5]
        before=(t40>=trigger-.5)&(t40<trigger) if trigger else np.zeros(1275,dtype=bool)
        after=(t40>=trigger+.05)&(t40<trigger+.55) if trigger else np.zeros(1275,dtype=bool)
        report={**s,'checks_passed':True,'samples_200hz':n,'independent_success_200hz':success,
            'hold_min_bottom_clearance_200hz_m':float(h[hold].min()),'forbidden_contact_count':len(forbidden),
            'first_sustained_slip_proxy_onset_s':None if onset is None else float(t[onset]),
            'alarm_delay_from_first_slip_proxy_s':None if trigger is None or onset is None else trigger-float(t[onset]),
            'final_hand_contact_loss_s':loss,'alarm_lead_to_final_loss_s':None if loss is None or trigger is None else loss-trigger,
            'first_gain_change_s':None if gain_k is None else float(t[gain_k]),
            'first_gain_change_after_alarm_ms':None if gain_k is None else round((t[gain_k]-trigger)*1000,6),
            'force_resultant_before_alarm_mean_N':forces[before].mean(axis=0).tolist() if before.any() else None,
            'force_resultant_after_alarm_mean_N':forces[after].mean(axis=0).tolist() if after.any() else None,
            'hold_force_resultant_mean_N':forces[phase40=='hold'].mean(axis=0).tolist(),
            'force_order':['thumb','index'],'force_definition':'40 Hz solver contacts vector-summed on cube per finger, then norm; not normal force or taxel signal',
            'slip_onsets_threshold_sensitivity':{str(threshold):None if (ix:=first_persistent(np.any((speed>threshold)&(loadN>.05),axis=1)&armed)) is None else float(t[ix]) for threshold in (.005,.01,.02)}}
        reports.append(report)
        manifest=load(folder/'manifest.json');manifest.pop('no_VLM_IK_support_controller_or_tactile_gate',None)
        manifest.update(uses_VLM=False,uses_IK=False,uses_tactile_lift_gate=False,uses_tactile_reactive_gains=s['reactive'])
        write(folder/'manifest.json',manifest)
        data[case]={'t':t,'h':h,'speed':speed,'features':features,'gain':inp['gain_applied'],'forces':forces,'t40':t40,'report':report}
        if not a.skip_video and not (folder/'replay_three_views.mp4').exists():
            cmd=['ffmpeg','-hide_banner','-loglevel','error','-y']
            for view in ('scene_front','scene_left','wrist'):cmd+=['-framerate','3.333333333','-pattern_type','glob','-i',str(folder/'frames'/f'*_{view}.jpg')]
            cmd+=['-filter_complex','[0:v]scale=640:480,setsar=1[a];[1:v]scale=640:480,setsar=1[b];[2:v]scale=640:362,pad=640:480:0:59,setsar=1[c];[a][b][c]hstack=inputs=3[v]',
                  '-map','[v]','-c:v','libx264','-crf','20','-pix_fmt','yuv420p',str(folder/'replay_three_views.mp4')]
            subprocess.run(cmd,check=True)
    write(out/'audit.json',{'checks':'Source provenance with explicitly reviewed optional-FEM drift; exact 1275 targets, 6375 samples at 5 ms, 40/200 Hz consistency, replayed tactile-only causal events, gain latency, engine mass/density/contact friction, independently recomputed clearance', 'source_drift':drift, 'results':reports})
    write(out/'dataset_manifest.json',{'episodes':list(protocol['cases']),'total_samples':6375*len(reports),'sensor_hz':200,
        'inputs':'Per episode inputs_200hz.npz: force_link_N [6375,5,120,3], q_rad, target_q_rad [6375,18], gain_applied, time_s. No object truth. Static coordinates in tactile_layout.json.',
        'labels':'labels_200hz.npz contains simulator-only finite-difference contact speed, physical load, object pose and clearance. training_labels.npz provides masks and targets; never concatenate these to model inputs.',
        'contact_loss_target':'Predict final fingertip-contact loss within 250/500/1000 ms, only with corresponding valid_loss_target mask. This is contact loss, not necessarily airborne drop: low-mu cube stays on table.',
        'intervention_censoring':'Future-loss targets invalid when their horizon reaches/passes the first gain change; reactive rescue is not a negative example of pre-intervention risk.',
        'slip_proxy':'Contact-force-weighted tangential relative material-point speed >5 mm/s, resultant contact-weight sum >0.05 N, at least3 consecutive5ms samples. Offline sustained label uses up to10ms future; use past-only input windows.',
        'limitations':['Only five exploratory episodes, one per configuration; no statistical success rate or held-out accuracy.',
          'Thresholds informed by historical40Hz runs. Successful grasp may contain physical micro-slip; slip is not identical to imminent drop.',
          'KinematicTaxel shear is constructed from simulated relative velocity. Synthetic shortcut; hardware bandwidth/noise/shear calibration and real validation required.',
          'Split datasets by full episode/material/object/trajectory, not random adjacent windows.',
          'Earlier contact-sort-index sensor bugs must be tagged separately, not labelled as genuine slip.'],
        'provenance':{'protocol_sha256':sha(out/'protocol.json'),'layout_sha256':sha(out/'tactile_layout.json'),
                      'inputs_sha256':{case:sha(out/case/'inputs_200hz.npz') for case in protocol['cases']}}})
    make_plots(out,data)
    make_gallery(out,reports)
    print(json.dumps(reports,ensure_ascii=False,indent=2))

def make_plots(out,data):
    font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    plt.rcParams.update({'font.family':font.get_name(),'axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False})
    fig,axs=plt.subplots(2,3,figsize=(16,8),layout='constrained')
    for row,stem in enumerate(('mass200','lowmu')):
        for suffix,color,label in [('monitor','#b44939','仅监测'),('reactive','#12807c','触觉触发增力')]:
            d=data[stem+'_'+suffix];t=d['t'];event=d['report']['trigger_time_s']
            axs[row,0].plot(t,d['h']*100,color=color,label=label)
            axs[row,1].plot(d['t40'],d['forces'][:,0],color=color,label=label+'·拇指')
            axs[row,1].plot(d['t40'],d['forces'][:,1],color=color,ls='--',label=label+'·食指')
            axs[row,2].plot(t,d['gain'],color=color,label=label)
            for ax in axs[row]:
                if event is not None:ax.axvline(event,color=color,alpha=.4,ls=':')
        axs[row,0].axhline(5,color='grey',ls='--');axs[row,0].set(ylabel='木块最低点离桌（cm）',title='200 g / μ=0.8' if row==0 else '45.56 g / μ=0.01')
        axs[row,1].set(ylabel='物理接触合力（N）',title='增力效果：逐指接触力矢量求和后取模')
        axs[row,2].set(ylabel='拇指 / 食指 kp 倍率',title='报警后 5 ms 开始、50 ms 内完成增益爬升')
        for ax in axs[row]:ax.set_xlim(20.5,32);ax.set_xlabel('仿真时间（秒）');ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('相同示范目标轨迹：200 Hz 触觉闭环与被动监测配对测试\n100 g 对照成功且未报警；每组仅一次，成功判据为最低点离桌 ≥5 cm 保持3秒',fontsize=14)
    fig.savefig(out/'recovery_comparison.png',dpi=145);plt.close(fig)
    fig,axs=plt.subplots(3,3,figsize=(16,10),layout='constrained')
    for row,case in enumerate(('mass200_monitor','lowmu_monitor','mass100_monitor')):
        d=data[case];t=d['t'];fs=d['features'];r=d['report']
        for j,finger in enumerate(('拇指','食指')):
            axs[row,0].plot(t,d['speed'][:,j]*1000,lw=.8,label=finger)
            axs[row,1].plot(t,[f['centroid_speed_m_s'][j]*1000 for f in fs],lw=.8,label=finger)
            axs[row,2].plot(t,[f['total_N'][j] for f in fs],lw=.9,label=finger)
        axs[row,0].set(title=NAMES[case]+' · 物理滑移代理',ylabel='接触切向相对速度（mm/s）',ylim=(0,40))
        axs[row,0].axhline(5,color='grey',ls='--')
        axs[row,1].set(title='触觉加权中心移动速度',ylabel='mm/s',ylim=(0,70));axs[row,1].axhline(20,color='grey',ls='--')
        axs[row,2].set(title='触觉测点信号总量',ylabel='各点三轴模长之和（名义 N）')
        for ax in axs[row]:
            for key,color,ls in [('first_sustained_slip_proxy_onset_s','#8b63a7','--'),('trigger_time_s','#d67517','-'),('final_hand_contact_loss_s','#b13e45',':')]:
                if r[key] is not None:ax.axvline(r[key],color=color,ls=ls,alpha=.8)
            ax.set(xlim=(21.3,31.9),xlabel='仿真时间（秒）');ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('紫虚线：首次持续微滑代理；橙线：触觉报警；红点线：最终失去手指接触\n200 Hz 采样不等于 5 ms 起滑检测；成功组也可能出现微滑，剪切信号尚未实机标定',fontsize=14)
    fig.savefig(out/'slip_timing.png',dpi=145);plt.close(fig)

def make_gallery(out,rows):
    cards=[]
    for r in rows:
        case=r['case'];trigger='无' if r['trigger_time_s'] is None else f"{r['trigger_time_s']:.3f} s"
        result='成功保持3秒' if r['success'] else '失败'
        cards.append(f'''<article><h2>{NAMES[case]} · {result}</h2><p>报警 {trigger}；质量 {r['actual_mass_g']:.2f} g；μ={r['contact_mu']}；保持最低离桌 {r['hold_min_bottom_clearance_200hz_m']*100:.2f} cm</p><video controls preload="metadata" src="{case}/replay_three_views.mp4"></video><p>录像为稀疏渲染，不能据此确定5ms事件时刻；请以200Hz日志为准。<a href="{case}/score.json">结果</a> · <a href="{case}/highfreq_events.jsonl">高频事件</a> · <a href="{case}/inputs_200hz.npz">模型输入</a> · <a href="{case}/training_labels.npz">训练标签</a></p></article>''')
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>触觉滑移与增力恢复实验</title><style>body{max-width:1400px;margin:auto;padding:28px;background:#111923;color:#e2eaf5;font:16px/1.6 sans-serif}a{color:#7bcef2}article{padding:20px;background:#1b2938;margin:24px 0;border-radius:14px}video,img{width:100%;border-radius:8px}h1{font-size:30px}p{color:#c9d5e2}</style><h1>200 Hz 触觉监测与增力恢复</h1><p>同一示范目标轨迹，40 Hz 发指令，200 Hz 读触觉。报警只用当前与历史触觉；物体状态仅作独立评价。</p><p>200 g 组报警后增力成功保持3秒；μ=0.01组同等增力结果见下方。每组仅一次，不代表成功率。触觉剪切由仿真相对速度构造，仍需实机验证。</p><p><a href="audit.json">完整核验</a> · <a href="dataset_manifest.json">训练数据说明</a> · <a href="protocol.json">冻结实验协议</a></p><img src="recovery_comparison.png"><img src="slip_timing.png">'''+''.join(cards)+'</html>'
    (out/'replays.html').write_text(page)
if __name__=='__main__':main()
