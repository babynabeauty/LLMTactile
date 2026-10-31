"""Summarize saved soft-body trial; never changes simulation state."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args();d=a.run
rows=[json.loads(s) for s in (d/'deformation.jsonl').read_text().splitlines()]
stages=json.loads((d/'stages.json').read_text());coupling=[]
for s in stages:
 f=Path(s['image_dir'])/'ipc_coupling.json'
 if f.exists():
  r=json.loads(f.read_text());coupling.append({'stage':s['name'],'max_translation_difference_mm':r['max_translation_difference_m']*1000,'max_affine_stretch_error':max(x['max_affine_stretch_error'] for x in r['robot_links'])})
release=[r for r in rows if r['phase'] in ('release','retreat','recovery')]
frames=sorted((d/'frames').glob('*_fem.npz'));speeds=[]
for f,g in zip(frames[:-1],frames[1:]):
 dt=(int(g.name.split('_')[0])-int(f.name.split('_')[0]))*.025
 speed=np.linalg.norm(np.load(g)['positions']-np.load(f)['positions'],axis=1)/dt
 speeds.append({'step':int(g.name.split('_')[0]),'max_average_vertex_speed_m_s':float(speed.max()),'interval_s':dt})
report={'score':json.loads((d/'score.json').read_text()),'samples':len(rows),'sampling_interval_s':.025,
 'min_J':min(r['volume_ratio_min'] for r in rows),'max_inverted':max(r['inverted_element_count'] for r in rows),
 'peak_principal_compression_p05':max(r['principal_compression_p05'] for r in rows),
 'peak_principal_compression_max':max(r['principal_compression_max'] for r in rows),
 'max_release_retreat_bottom_above_table_mm':max(r['bounds_min_m'][2]-.2 for r in release)*1000,
 'final':rows[-1],'ipc_coupling_stages':coupling,
 'velocity_note':'Current Genesis IPC retrieves FEM positions only; recorded get_state().vel zeros are unavailable velocity data, NOT evidence of zero motion. Saved-node finite differences below are interval averages.',
 'max_saved_frame_average_vertex_speed_m_s':max(x['max_average_vertex_speed_m_s'] for x in speeds),
 'sampling_note':'Hand diagnostics sampled each 25 ms control tick, not all IPC substeps. Virtual press diagnostics sampled every physics step.',
 'material_scope':'specified elastic model; soft FEM is not connected to XHand taxels'}
(d/'audit_summary.json').write_text(json.dumps(report,indent=2));(d/'finite_difference_speed.json').write_text(json.dumps(speeds,indent=2));print(json.dumps(report,indent=2))
