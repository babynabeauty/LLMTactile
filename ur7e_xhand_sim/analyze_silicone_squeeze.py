"""Render recorded FEM squeeze diagnostics and camera replay; never drives physics."""
import argparse,json,subprocess
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args();root=a.run
rows=[json.loads(s) for s in (root/'deformation.jsonl').read_text().splitlines()]
control=[json.loads(s) for s in (root/'control.jsonl').read_text().splitlines()]
recovery=[json.loads(s) for s in (root/'recovery.jsonl').read_text().splitlines()] if (root/'recovery.jsonl').exists() else []
for item in recovery:
 r=item['deformation'].copy();r['phase']={'recovery_stopped':'release','recovery_released':'retreat','recovery_retreated':'recovery'}.get(r['phase'],r['phase']);rows.append(r)
t=np.array([r['step']*.025 for r in rows]);ph=np.array([r['phase'] for r in rows])
strain=np.array([r['principal_compression_p05'] for r in rows]);ext=np.array([r['extent_m'] for r in rows])
base=strain[np.isin(ph,['preshape','approach'])];baseline=float(np.median(base[-40:])) if len(base) else float(strain[0])
load=np.isin(ph,['squeeze','hold']);peak=float(strain[load].max()) if load.any() else None
final=float(np.median(strain[-20:]));report={'baseline_p05_compression':baseline,'loaded_peak_p05_compression':peak,
    'final_p05_compression':final,'overall_peak_p05_compression':float(strain.max()),'deformation_definition':'1 minus fifth percentile of per-tetrahedron minimum principal stretch relative to undeformed mesh; not overall cube shortening',
    'minimum_volume_ratio':min(r['volume_ratio_min'] for r in rows),
    'recovered_fraction_of_excess_compression':None if peak is None or peak<=baseline else float((peak-final)/(peak-baseline)),
    'tactile_is_not_FEM_force':True,'recorded_frames':len(list((root/'frames').glob('*_scene_front.jpg')))}
(root/'analysis.json').write_text(json.dumps(report,indent=2))
fig,ax=plt.subplots(3,1,figsize=(11,9),sharex=True,layout='constrained')
ax[0].plot(t,strain*100);ax[0].axhline(baseline*100,color='gray',ls='--');ax[0].set_ylabel('FEM compression p05 (%)')
for i,n in enumerate('XYZ'):ax[1].plot(t,ext[:,i]*1000,label=n)
ax[1].set_ylabel('World AABB size (mm)');ax[1].legend()
name='right_hand_index_joint1'
# Use the same joint order saved by the live observation rather than an assumed name.
keys=list(control[0]['after']['joint_positions_rad']);name=keys[10]
tc=np.array([r['step']*.025 for r in control])
ax[2].plot(tc,[r['sent_target_rad'][10] for r in control],label='Index command')
ax[2].plot(tc,[r['after']['joint_positions_rad'][name] for r in control],label='Index actual')
if recovery:ax[2].plot([r['observation']['sim_time_s'] for r in recovery],[r['observation']['joint_positions_rad'][name] for r in recovery],label='Release / retreat actual')
ax[2].set(ylabel='Joint angle (rad)',xlabel='Simulated time (s)');ax[2].legend()
for x in ax:
 x.grid(alpha=.2)
 for phase,color in [('squeeze','#e6a0c4'),('hold','#e6c78f'),('release','#9ab7e0'),('recovery','#9ad4ae')]:
  ii=np.flatnonzero(ph==phase)
  if len(ii):x.axvspan(t[ii[0]],t[ii[-1]],color=color,alpha=.22)
fig.suptitle('Pink FEM block: recorded squeeze / release / recovery\nMesh strain diagnostic; soft tactile channel not connected')
fig.savefig(root/'deformation.png',dpi=150);plt.close(fig)
cmd=['ffmpeg','-hide_banner','-loglevel','error','-y']
for view in ('scene_front','scene_left','wrist'):
 files=sorted((root/'frames').glob(f'*_{view}.jpg'));manifest=root/(view+'_video.txt')
 lines=[]
 for i,f in enumerate(files):
  duration=(int(files[i+1].name.split('_')[0])-int(f.name.split('_')[0]))*.025 if i+1<len(files) else .3
  lines += ["file '"+str(f.resolve())+"'",'duration '+str(duration)]
 lines += ["file '"+str(files[-1].resolve())+"'"];manifest.write_text('\n'.join(lines)+'\n')
 cmd+=['-f','concat','-safe','0','-i',str(manifest)]
cmd+=['-filter_complex','[0:v]scale=640:480,setsar=1[a];[1:v]scale=640:480,setsar=1[b];[2:v]scale=640:362,pad=640:480:0:59,setsar=1[c];[a][b][c]hstack=inputs=3[v]',
 '-map','[v]','-r','30','-c:v','libx264','-crf','20','-pix_fmt','yuv420p',str(root/'replay_three_views.mp4')]
subprocess.run(cmd,check=True);print(json.dumps(report,indent=2))
