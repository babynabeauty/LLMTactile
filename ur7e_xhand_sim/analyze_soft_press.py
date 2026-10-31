"""Compare independent IPC press resolutions; reads recorded results only."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image,ImageOps,ImageDraw
p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args()
names=['press_5ms_coarse_v2','press_2p5ms_coarse','press_2p5ms_fine']
cases=[];report={'scope':'numerical consistency for a specified elastic material; no measured silicone calibration','cases':[],'comparisons':[]}
fig,axes=plt.subplots(3,1,figsize=(10,9),sharex=True,layout='constrained')
for name in names:
 d=a.root/name;c=json.loads((d/'config.json').read_text());s=json.loads((d/'summary.json').read_text());r=[json.loads(x) for x in (d/'metrics.jsonl').read_text().splitlines()]
 t=np.array([x['t'] for x in r]);h=np.array([x['height_m'] for x in r]);j=np.array([x['min_J'] for x in r]);v=np.array([x['volume_ratio'] for x in r]);cases.append((t,h,j,v))
 label=f"{c['dt']*1000:g} ms, {c['mesh']['tetrahedra']} tets"
 for ax,y in zip(axes,[h*1000,j,v]):ax.plot(t,y,label=label,linewidth=1.3)
 report['cases'].append({'name':name,'dt_s':c['dt'],'tets':c['mesh']['tetrahedra'],'all_samples_finite':all(x['finite'] for x in r),'samples':len(r),'min_J':s['min_J'],'max_inverted':s['max_inverted'],'max_Genesis_plane_penetration_mm':s['max_plate_penetration_m']*1000,'hold_end_height_mm':s['stages']['2.0s']['height_m']*1000,'hold_last_0p1s_mean_height_mm':float(h[(t>1.9)&(t<=2)].mean()*1000),'final_height_mm':h[-1]*1000,'minimum_height_mm':h.min()*1000,'maximum_height_compression_fraction':float(1-h.min()/.045),'final_volume_ratio':v[-1]})
for before,after,label in [(0,1,'halve timestep'),(1,2,'refine mesh')]:
 t,h,_,_=cases[before];tt,hh,_,_=cases[after];delta=(h-np.interp(t,tt,hh))*1000
 report['comparisons'].append({'change':label,'height_time_series_rms_difference_mm':float(np.sqrt(np.mean(delta**2))),'height_time_series_max_difference_mm':float(np.abs(delta).max()),'hold_end_height_difference_mm':abs(report['cases'][before]['hold_end_height_mm']-report['cases'][after]['hold_end_height_mm'])})
for ax,ylabel in zip(axes,['Block height (mm)','Minimum element J','Total volume / initial volume']):
 ax.set_ylabel(ylabel);ax.grid(alpha=.2);ax.axvspan(.5,1.5,color='pink',alpha=.2);ax.axvspan(1.5,2,color='orange',alpha=.1)
axes[0].legend();axes[-1].set_xlabel('Simulated time (s)');fig.suptitle('IPC soft block: timestep and mesh sensitivity\nPD press, 500 N actuator limit; material E=100 kPa, nu=0.45')
fig.savefig(a.root/'press_convergence.png',dpi=150);plt.close(fig)
report['limitations']=['Servo force is a clipped PD estimate, not measured contact force.','Penetration is relative to Genesis platen planes, not a complete IPC surface-distance audit.','A two-level sensitivity comparison is not an asymptotic convergence proof.','Contact regularization d_hat=0.5 mm; this sets a contact length scale.']
(a.root/'press_comparison.json').write_text(json.dumps(report,indent=2))
canvas=Image.new('RGB',(1280,510),'white');draw=ImageDraw.Draw(canvas)
for i,(stage,label) in enumerate([('2.0s','Compressed / hold'),('4.0s','Released / recovered')]):
 im=Image.open(a.root/names[1]/f'{stage}.png');canvas.paste(im,(640*i,30));draw.text((640*i+20,8),label,fill='black')
canvas.save(a.root/'press_before_after.png');print(json.dumps(report,indent=2))
