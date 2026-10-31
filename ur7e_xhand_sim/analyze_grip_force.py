"""Compare solver contact resultants with original replay baselines."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

def main():
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);a=p.parse_args();out=a.directory.resolve()
    root=Path(__file__).resolve().parents[1];old=root/'outputs/material_replay_20261024_v1'
    sources={'friction_original':old/'friction_080','friction_low':out/'friction_001',
        'mass_original':old/'mass_200g','mass_stronger':out/'mass_200g_stronger'}
    series={};report={}
    for name,folder in sources.items():
        rows=[json.loads(s) for s in (folder/'evaluation.jsonl').read_text().splitlines()]
        force=[]
        for row in rows:
            vecs={f:np.zeros(3) for f in ('thumb','index')}
            for c in row['solver_contacts']:
                if {c['entity_a'],c['entity_b']}!={'cube','ur7e_xhand'}:continue
                link=c['link_b'] if c['entity_a']=='cube' else c['link_a']
                on_cube=np.array(c['force_on_a_world_N'])*(1 if c['entity_a']=='cube' else -1)
                for f in vecs:
                    if f in link:vecs[f]+=on_cube
            force.append([float(np.linalg.norm(vecs[f])) for f in ('thumb','index')])
        force=np.array(force);phase=np.array([r['phase'] for r in rows]);pre=np.where(phase=='preload')[0][-20:];hold=phase=='hold'
        height=np.array([r['bottom_clearance_m'] for r in rows]);t=np.array([r['step']*.025 for r in rows])
        report[name]={'run_dir':str(folder),'preload_last20_mean_contact_resultant_N':force[pre].mean(axis=0).tolist(),
            'hold_mean_contact_resultant_N':force[hold].mean(axis=0).tolist(),
            'finger_order':['thumb','index'],'force_source':'Solver world-space contact forces on cube, vector-summed per finger then norm; not taxel estimates'}
        series[name]=(t,height,force)
    original=np.array(report['mass_original']['preload_last20_mean_contact_resultant_N'])
    stronger=np.array(report['mass_stronger']['preload_last20_mean_contact_resultant_N'])
    report['stronger_preload_force_ratio']=(stronger/original).tolist()
    report['force_increase_verified_both_fingers']=bool(np.all(stronger>original))
    (out/'force_audit.json').write_text(json.dumps(report,indent=2))
    font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    plt.rcParams.update({'font.family':font.get_name(),'axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,3,figsize=(15,8),layout='constrained')
    for i,pairs in enumerate([ [('friction_original','μ=0.8，原夹持'),('friction_low','μ=0.01，原夹持')],
                               [('mass_original','200 g，原夹持'),('mass_stronger','200 g，加强夹持')] ]):
        for name,label in pairs:
            t,h,f=series[name];axes[i,0].plot(t,h*100,label=label)
            for j in range(2):axes[i,j+1].plot(t,f[:,j],label=label)
        axes[i,0].set(title='木块实际离桌高度',ylabel='木块最低点离桌（cm）');axes[i,0].axhline(5,color='grey',ls='--')
        for j,f in enumerate(('拇指','食指')):axes[i,j+1].set(title=f+'物理接触合力',ylabel='求解器接触合力（N）')
        for ax in axes[i]:ax.set_xlabel('仿真时间（秒）');ax.grid(alpha=.2);ax.legend();ax.axvspan(28.9,31.875,color='#d7eadb',alpha=.35)
    fig.suptitle('原轨迹低摩擦测试 / 200 g 加强夹持测试\n接触合力来自物理求解器，与未标定触觉测点读数分别记录',fontsize=14)
    fig.savefig(out/'force_comparison.png',dpi=150);plt.close(fig)
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
