"""Mesh, renderer, freefall and table-contact audit; robot remains at its home command."""
import argparse,json
from pathlib import Path
import numpy as np
from environment import Sandbox
from fem_diagnostics import mesh_quality,contact_diagnostics
from genesis.utils.misc import qd_to_torch
p=argparse.ArgumentParser();p.add_argument('--substeps',type=int,default=5);p.add_argument('--softness',type=float,default=.002)
p.add_argument('--skip-idle-pcg',action='store_true');p.add_argument('--solver',choices=['legacy','ipc'],default='legacy')
p.add_argument('--maxvolume',type=float,default=1e-7);p.add_argument('--output',type=Path);a=p.parse_args()
out=a.output or Path('outputs/silicone_audit_20261024')/(f'substeps_{a.substeps}_softness_{a.softness}'+('_skip_idle' if a.skip_idle_pcg else ''));out.mkdir(parents=True,exist_ok=True)
sim=Sandbox(output_dir=out,render=False,scenario='small_clutter',silicone=True,silicone_substeps=a.substeps,silicone_contact_softness=a.softness,silicone_skip_idle_pcg=a.skip_idle_pcg,silicone_solver=a.solver,silicone_maxvolume=a.maxvolume)
pcg_stats=sim.fem_pcg_stats
e=sim.silicone;r=sim.array(e.init_positions).reshape(-1,3).astype(float);t=sim.array(e.get_el2v()).astype(int)
x=r[t];D=np.stack([x[:,i]-x[:,3] for i in range(3)],axis=-1);inv=np.linalg.inv(D);V=np.abs(np.linalg.det(D))/6
quality=mesh_quality(r,t);quality['rendering']=sim.silicone_parameters['surface_rendering']
(out/'mesh.json').write_text(json.dumps(quality,indent=2));print('MESH',json.dumps(quality),flush=True)
np.savez_compressed(out/'mesh.npz',rest=r,tets=t,boundary=e._surface_tri_np)
rows=[]
for i in range(121):
 if i in (0,1,4,8,10,12,20,40,80,120):
  pos=sim.array(e.get_state().pos).reshape(-1,3);x=pos[t];F=np.stack([x[:,j]-x[:,3] for j in range(3)],axis=-1)@inv;J=np.linalg.det(F);k=int(np.argmin(J))
  rendered,_,_=sim.scene.fem_solver.get_state_render(sim.scene.sim.cur_substep_local)
  actual=qd_to_torch(rendered,copy=True).cpu().numpy().reshape(-1,3)
  ids=np.concatenate([g.sim_verts_idx for g in e.vgeoms]);render_error=float(np.abs(actual-pos[ids]).max())
  row={'time_s':i*.005,'min_J':float(J.min()),'volume_ratio':float((J*V).sum()/V.sum()),'worst_tet':k,'worst_rest_center':r[t[k]].mean(0).tolist(),
       'worst_current_center':pos[t[k]].mean(0).tolist(),'worst_rest_volume_m3':float(V[k]),'renderer_max_error_m':render_error,
       'contacts':contact_diagnostics(sim,pos)}
  rows.append(row);print(json.dumps(row),flush=True)
  np.savez_compressed(out/f'state_{i:03d}.npz',positions=pos,J=J)
 if i<120:sim.scene.step()
sim.render_enabled=True;sim.render()
(out/'result.json').write_text(json.dumps({'mesh':quality,'samples':rows,'pcg_stats':pcg_stats},indent=2))
sim.scene.destroy()
