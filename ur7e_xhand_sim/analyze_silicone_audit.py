"""Compare independent drop-contact diagnostics; no physics actions."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
out=Path('outputs/silicone_audit_20261024')
cases=[('substeps_5','1 ms / softness 2 mm'),('substeps_10_softness_0.002','0.5 ms / softness 2 mm'),('substeps_20_softness_0.002','0.25 ms / softness 2 mm'),('substeps_5_softness_0.0002','1 ms / softness 0.2 mm')]
fig,ax=plt.subplots(1,2,figsize=(12,4),layout='constrained');report=[]
for folder,label in cases:
 d=json.loads((out/folder/'result.json').read_text());r=d['samples'];t=[x['time_s'] for x in r]
 ax[0].plot(t,[x['min_J'] for x in r],'.-',label=label)
 ax[1].plot(t,[100*x['volume_ratio'] for x in r],'.-',label=label)
 report.append({'case':label,'sampled_minimum_J':min(x['min_J'] for x in r),'end_minimum_J':r[-1]['min_J'],'end_total_volume_ratio':r[-1]['volume_ratio'],'max_renderer_position_error_m':max(x['renderer_max_error_m'] for x in r)})
ax[0].axhline(0,color='black',ls='--');ax[0].set(ylabel='Minimum local signed volume ratio J',xlabel='Simulated time (s)')
ax[1].set(ylabel='Total signed volume (% of rest)',xlabel='Simulated time (s)')
for x in ax:x.grid(alpha=.2);x.legend(fontsize=8)
fig.suptitle('Same 45 mm FEM block: free fall and table contact (sampled states)')
fig.savefig(out/'contact_comparison.png',dpi=150);plt.close(fig)
mesh=np.load(out/'substeps_5/mesh.npz');state=np.load(out/'substeps_5/state_012.npz');rest=mesh['rest'];pos=state['positions'];tets=mesh['tets'];boundary=mesh['boundary'];J=state['J'];k=int(J.argmin())
fig=plt.figure(figsize=(11,5));left=fig.add_subplot(121,projection='3d');right=fig.add_subplot(122,projection='3d')
# Original primitive render vertices are the first 8 FEM points. Draw their convex hull.
from scipy.spatial import ConvexHull
for a,verts,faces,title in [(left,pos[:8],ConvexHull(rest[:8]).simplices,'Old display: 8 corner nodes'),(right,pos,boundary,'Actual physics boundary: 304 nodes')]:
 verts=(verts-pos.mean(0))*1000
 a.add_collection3d(Poly3DCollection(verts[faces],facecolor='#e37aa8',edgecolor='#67314b',linewidth=.25,alpha=.65))
 a.set(xlim=(-27,27),ylim=(-27,27),zlim=(-27,27),xlabel='X (mm)',ylabel='Y (mm)',zlabel='Z (mm)',title=title);a.set_box_aspect((1,1,1));a.view_init(elev=16,azim=-110)
right.scatter(*((pos[tets[k]]-pos.mean(0))*1000).T,color='red',s=20)
fig.suptitle('Drop at 0.06 s: corner-only rendering hides local boundary distortion')
fig.savefig(out/'render_mesh_comparison.png',dpi=150,bbox_inches='tight');plt.close(fig)
(out/'comparison.json').write_text(json.dumps({'cases':report,'physical_mesh':{'vertices':len(rest),'tetrahedra':len(tets),'boundary_vertices':len(np.unique(boundary)),'boundary_triangles':len(boundary)},'note':'Minima over recorded samples only; not an all-substep convergence proof.'},indent=2))
print(json.dumps(report,indent=2))
