"""Independent material-sweep audit, plots, and three-view replay videos."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

def main():
    parser=argparse.ArgumentParser();parser.add_argument('directory',type=Path);args=parser.parse_args();out=args.directory.resolve()
    load=lambda p:json.loads(p.read_text())
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    protocol=load(out/'protocol.json')
    assert all(sha(Path(p))==h for p,h in protocol['source_sha256'].items())
    demo=Path(protocol['demonstration'])
    assert sha(demo/'control.jsonl')==protocol['trajectory_sha256']
    reference=np.array([json.loads(s)['sent_target_rad'] for s in (demo/'control.jsonl').read_text().splitlines()])
    results=[];series=[]
    for case in protocol['cases']:
        folder=out/case;score=load(folder/'score.json');material=load(folder/'material_verification.json')
        c=[json.loads(s) for s in (folder/'control.jsonl').read_text().splitlines()]
        e=[json.loads(s) for s in (folder/'evaluation.jsonl').read_text().splitlines()]
        assert len(c)==len(e)==score['control_steps']<=1275
        assert score['fault'] or len(c)==1275
        assert [x['step'] for x in c]==list(range(1,len(c)+1))
        recorded=np.load(folder/'trajectory_tactile.npz')
        np.testing.assert_allclose(recorded['sent_targets_rad'],reference[:len(c)],atol=1e-12,rtol=0)
        assert recorded['force_link_N'].shape==(len(c),5,120,3)
        assert np.isfinite(recorded['force_link_N']).all()
        np.testing.assert_allclose(material['engine_mass_kg'],material['resolved']['cube_mass_kg'],atol=1e-8,rtol=1e-5)
        np.testing.assert_allclose(material['engine_mass_divided_by_volume_kg_m3'],material['material_density_kg_m3'],rtol=1e-5)
        assert material['all_checks_passed']
        for pair,expected in [('robot_cube',score['contact_mu']),('table_cube',.8)]:
            v=material['observed_pair_friction'][pair];assert v['contact_samples']>0
            np.testing.assert_allclose([v['min'],v['max']],expected,atol=1e-6,rtol=0)
        clearance=[]
        for v in e:
            r=Rotation.from_quat(np.array(v['cube_quaternion_wxyz'])[[1,2,3,0]]).as_matrix()
            h=v['cube_position_m'][2]-.0225*np.abs(r[2]).sum()-.20
            assert abs(h-v['bottom_clearance_m'])<1e-9;clearance.append(h)
        phase=np.array([v['phase'] for v in c]);hold=phase=='hold';heights=np.array(clearance)
        success=bool(hold.sum()==120 and heights[hold].min()>=.05 and not score['fault'])
        assert success==score['success']
        peaks=np.array([[v['observation']['tactile'][f]['peak_force_N'] for f in ('thumb','index')] for v in c])
        np.testing.assert_allclose(np.linalg.norm(recorded['force_link_N'][:,:2],axis=-1).max(axis=-1),peaks,atol=1e-7)
        bad=[contact for v in e for contact in v['solver_contacts'] if contact['entity_a']==contact['entity_b']=='ur7e_xhand' or 'table' in (contact['entity_a'],contact['entity_b'])]
        row={**score,'independent_score_verified':True,'trajectory_verified':True,'parameter_readback_verified':True,
            'forbidden_contact_count':len(bad),'hold_both_tactile_samples':int(np.all(peaks[hold]>=.01,axis=1).sum()),
            'hold_mean_tactile_peak_N':peaks[hold].mean(axis=0).tolist() if hold.any() else None,
            'hold_min_tactile_peak_N':peaks[hold].min(axis=0).tolist() if hold.any() else None,
            'finger_order':['thumb','index']}
        results.append(row);series.append((case,np.arange(1,len(c)+1)*.025,phase,heights,peaks))
        video=folder/'replay_three_views.mp4'
        if not video.exists():
            command=['ffmpeg','-hide_banner','-loglevel','error','-y']
            for view in ('scene_front','scene_left','wrist'):
                command+=['-framerate','3.333333333','-pattern_type','glob','-i',str(folder/'frames'/f'*_{view}.jpg')]
            command+=['-filter_complex','[0:v]scale=640:480,setsar=1[a];[1:v]scale=640:480,setsar=1[b];[2:v]scale=640:362,pad=640:480:0:59,setsar=1[c];[a][b][c]hstack=inputs=3[v]',
                '-map','[v]','-c:v','libx264','-crf','20','-pix_fmt','yuv420p',str(video)]
            subprocess.run(command,check=True)
    (out/'audit.json').write_text(json.dumps({'checks':'source hashes, exact targets, engine density/mass, actual pair friction, full tactile arrays, rotated-box physical clearance','results':results},indent=2))
    font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    plt.rcParams.update({'font.family':font.get_name(),'axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,3,figsize=(15,8),layout='constrained')
    for i,(case,t,phase,h,p) in enumerate(series):
        row=0 if case.startswith('friction') else 1
        result=results[i];label=f'μ={result["contact_mu"]}' if row==0 else f'{result["actual_mass_g"]:.0f} g'
        axes[row,0].plot(t,h*100,label=label,lw=1.4)
        for j in range(2):axes[row,j+1].plot(t,p[:,j],label=label,lw=1.2)
    for row in range(2):
        axes[row,0].axhline(5,color='grey',ls='--');axes[row,0].set(ylabel='木块最低点离桌（cm）',title=('摩擦组：质量 45.56 g' if row==0 else '质量组：摩擦 0.8'))
        for j,name in enumerate(('拇指','食指')):axes[row,j+1].set(title=name+'触觉峰值',ylabel='名义 N（未实测标定）')
        for ax in axes[row]:
            ax.axvspan(28.9,31.875,color='#d7eadb',alpha=.35);ax.set_xlabel('仿真时间（秒）');ax.grid(alpha=.2);ax.legend()
    fig.suptitle(f'同一成功示范的关节目标回放：{len(series)} 组参数测试\n未使用触觉反馈修正动作；绿色区为 3 秒保持阶段',fontsize=15)
    fig.savefig(out/'comparison.png',dpi=150);plt.close(fig)
    print(json.dumps(results,indent=2))

if __name__=='__main__':main()
