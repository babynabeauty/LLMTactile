"""Unoccluded, true-scale deformation comparison from saved FEM nodes only."""
import argparse,json,itertools
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.animation import FuncAnimation,PillowWriter
from analyze_silicone_mesh import aligned_displacement
p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args();root=a.run;out=root/'closeup';out.mkdir(exist_ok=True)
font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams.update({'font.family':font.get_name(),'axes.unicode_minus':False,'font.size':12})
m=np.load(root/'mesh.npz');rest=m['rest'];faces=m['boundary'];x=(rest-rest.mean(0))*1000
names=['initial','held','released'];folders=['view_000','view_005','view_006'];st=[]
for f in folders:
 y,d=aligned_displacement(rest,np.load(root/f/'fem_positions.npz')['positions']);st.append((y*1000,d*1000))
held,d=st[1];idx=int(d.argmax());plane=float(held[idx,1])+1e-3
# Intersect the physical triangular boundary with a fixed plane (no silhouette approximation).
def section(y):
 segments=[]
 for tri in y[faces]:
  pts=[]
  for i,j in [(0,1),(1,2),(2,0)]:
   u,v=tri[i],tri[j]
   if (u[1]-plane)*(v[1]-plane)<0:pts.append(u+(v-u)*(plane-u[1])/(v[1]-u[1]))
  if len(pts)==2:segments.append(np.array(pts)[:,[0,2]])
 return segments
lo=x.min(0);hi=x.max(0);corners=np.array(list(itertools.product(*zip(lo,hi))))
edges=[(i,j) for i in range(8) for j in range(i+1,8) if np.sum(corners[i]!=corners[j])==1]
norm=plt.Normalize(0,7)
def setup3d(ax,y,disp,title):
 coll=Poly3DCollection(y[faces],facecolors=plt.cm.YlOrRd(norm(disp[faces].mean(1))),edgecolors=(.35,.35,.4,.35),linewidths=.25)
 ax.add_collection3d(coll)
 for i,j in edges:ax.plot(*corners[[i,j]].T,color='#8c94a0',ls='--',lw=.8,alpha=.65)
 ax.view_init(elev=16,azim=-25);ax.set_proj_type('ortho');ax.set_box_aspect((1,1,1))
 ax.set(xlim=(lo[0]-2,hi[0]+2),ylim=(lo[1]-2,hi[1]+2),zlim=(lo[2]-2,hi[2]+2),title=title)
 ax.set_axis_off();return coll
fig=plt.figure(figsize=(13,6),layout='constrained');ax=fig.add_subplot(121,projection='3d')
setup3d(ax,held,d,'捏压保持：正对主要凹陷面')
ax.scatter(*held[idx],color='#00bde3',s=35,depthshade=False)
ax.text2D(.12,.03,f'亮色接触区：最大节点位移 {d[idx]:.2f} mm',transform=ax.transAxes,color='#aa2244',fontsize=13)
b=fig.add_subplot(122)
for (y,dd),label,color,style in zip(st,['捏压前','捏压保持','松开后'],['#9098a1','#d61b64','#009d98'],['--','-','dotted']):
 b.add_collection(LineCollection(section(y),colors=color,linewidths=2.5,linestyles=style,label=label))
b.set(xlim=(lo[0]-3,hi[0]+3),ylim=(lo[2]-3,hi[2]+3),xlabel='X（mm）',ylabel='Z（mm）',title='同一剖面的轮廓对比');b.set_aspect('equal');b.grid(alpha=.15);b.legend(loc='lower center',bbox_to_anchor=(.5,.08),fontsize=11)
b.text(.15,.77,'右侧轮廓向内凹陷\n松开后与原轮廓基本重合',transform=b.transAxes,va='top',fontsize=11)
fig.suptitle('真实物理网格近景 · 无机械手遮挡 · 形变倍率 1×',fontsize=18)
fig.supxlabel('已去除整体平移、旋转；灰色虚线为未变形参考，颜色表示节点位移。',fontsize=11)
fig.savefig(out/'indentation_detail.png',dpi=160);plt.close(fig)
fig=plt.figure(figsize=(15,5),layout='constrained')
for i,((y,dd),title) in enumerate(zip(st,['捏压前','捏压保持','松开后'])):
 ax=fig.add_subplot(1,3,i+1,projection='3d');setup3d(ax,y,dd,title)
fig.suptitle('同一方向、同一尺寸 · 45 mm 软块 · 形变未放大',fontsize=18)
fig.savefig(out/'three_stages.png',dpi=160);plt.close(fig)
# Animate only actually recorded states, without interpolating or magnifying deformation.
paths=sorted((root/'frames').glob('*_fem.npz'));paths=[f for f in paths if 804<=int(f.name.split('_')[0])<=1404]
fig=plt.figure(figsize=(6,6));ax=fig.add_subplot(111,projection='3d');fig.subplots_adjust(0,0,1,.89)
coll=setup3d(ax,*st[0],'');label=fig.text(.5,.94,'',ha='center',fontsize=15);fig.text(.5,.03,'保存的物理节点回放 · 形变 1× · 非重新仿真',ha='center',fontsize=10)
cache=[]
for f in paths:
 y,dd=aligned_displacement(rest,np.load(f)['positions']);cache.append((int(f.name.split('_')[0]),y*1000,dd*1000))
def update(i):
 step,y,dd=cache[i];coll.set_verts(y[faces]);coll.set_facecolor(plt.cm.YlOrRd(norm(dd[faces].mean(1))))
 phase='接近' if step<815 else '捏压' if step<=1005 else '保持' if step<=1045 else '松开' if step<=1386 else '退回'
 label.set_text(f'{phase}  |  t={step*.025:.2f} s\n最大节点位移 {dd.max():.2f} mm');return coll,label
ani=FuncAnimation(fig,update,frames=len(cache),interval=150,blit=False);ani.save(out/'deformation_closeup.gif',writer=PillowWriter(fps=7),dpi=100);plt.close(fig)
(out/'metadata.json').write_text(json.dumps({'source':str(root),'physical_shape_scale':1,'rigid_motion_removed':True,'plane_y_mm':plane,'held_max_node_displacement_mm':float(d[idx]),'tracked_node_displacement_xyz_mm':(held[idx]-x[idx]).tolist(),'animation':'recorded states only; 0.30 s sample spacing played at 7 fps (~2.1x simulation speed)'},indent=2))
print(out)
