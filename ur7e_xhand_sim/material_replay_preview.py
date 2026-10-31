"""Read-only live preview of replay artifacts; no simulation control endpoints."""
import argparse
import json
from functools import partial
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path

PAGE='''<!doctype html><meta charset="utf-8"><title>材料测试 · 实时回放</title>
<style>body{margin:24px;background:#101722;color:#e9f0f8;font:16px system-ui}h1{font-size:24px}#cases{display:flex;gap:10px;flex-wrap:wrap;margin:20px 0}.case{background:#243346;border:2px solid transparent;border-radius:8px;padding:12px;cursor:pointer}.selected{border-color:#62c7ec}#views{display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px}img{width:100%;height:29vw;object-fit:contain;background:#080e16;border-radius:8px}small{color:#afc1d4}#state{margin:18px 0;line-height:1.8}button{padding:8px 16px;border:0;border-radius:6px;background:#62c7ec;cursor:pointer}label{display:block;margin:8px 0}</style>
<h1>摩擦 / 质量测试：成功示范轨迹回放</h1><small>同一条 1275 步关节目标轨迹 · 前方 / 左侧 / 腕部 · 画面每 12 个控制步更新</small>
<div id="cases"></div><button onclick="selected=null;update()">跟随当前测试组</button><div id="state">连接中…</div>
<div id="views"><section><label>前方整体</label><img id="scene_front"></section><section><label>左侧整体</label><img id="scene_left"></section><section><label>腕部</label><img id="wrist"></section></div>
<p><small>关节目标按示范回放；是否启用触觉触发增力由实验组配置。成功标准：木块底面离桌 ≥ 5 cm，并保持 3 秒。</small></p>
<script>let selected=null;const names={friction_020:'摩擦 0.2 · 45.56 g',friction_050:'摩擦 0.5 · 45.56 g',friction_080:'摩擦 0.8 · 45.56 g',mass_050g:'质量 50 g · μ 0.8',mass_100g:'质量 100 g · μ 0.8',mass_200g:'质量 200 g · μ 0.8',friction_001:'摩擦 0.01 · 原始轨迹',mass_200g_stronger:'200 g · 加强夹持',mass200_monitor:'200 g · 只监测',lowmu_monitor:'μ 0.01 · 只监测',mass100_monitor:'100 g · 监测对照',mass200_reactive:'200 g · 触觉触发增力',lowmu_reactive:'μ 0.01 · 触觉触发增力'};
async function update(){try{let d=await(await fetch('/status?t='+Date.now())).json();let active=selected||d.current;document.getElementById('cases').innerHTML=d.cases.map(c=>`<div class="case ${c.name===active?'selected':''}" onclick="selected='${c.name}';update()">${names[c.name]}<br><small>${c.score?(c.score.success?'已完成 · 成功':'已完成 · 未通过'):c.step?'回放中 · '+c.step+'/1275':c.started?'初始化':'等待'}</small></div>`).join('');let c=d.cases.find(c=>c.name===active);if(!c)return;document.getElementById('state').textContent=names[active]+' ｜ '+(c.phase||'初始化')+' ｜ '+(c.step||0)+'/1275 步'+(c.score?' ｜ 最高离桌 '+(c.score.max_bottom_clearance_m*100).toFixed(2)+' cm':'')+(c.material?' ｜ 引擎质量 '+(c.material.engine_mass_kg*1000).toFixed(3)+' g，密度 '+c.material.material_density_kg_m3.toFixed(2)+' kg/m³':'');for(let view of ['scene_front','scene_left','wrist']){let im=document.getElementById(view);if(c.started)im.src='/'+active+'/runtime/'+view+'.png?t='+Date.now();}}catch(e){document.getElementById('state').textContent='正在等待回放记录…';}}update();setInterval(update,1500);</script>'''

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--port',type=int,default=8768);a=p.parse_args();root=a.output.resolve()
    class Handler(SimpleHTTPRequestHandler):
        def do_GET(self):
            path=self.path.split('?')[0]
            if path not in ('/','/status'):return super().do_GET()
            if path=='/':body=PAGE.encode();mime='text/html; charset=utf-8'
            else:
                cases=[]
                for name in json.loads((root/'protocol.json').read_text())['cases']:
                    folder=root/name;row={'name':name,'started':folder.exists(),'step':0}
                    for filename,key in [('score.json','score'),('material_verification.json','material')]:
                        try:row[key]=json.loads((folder/filename).read_text())
                        except (FileNotFoundError,json.JSONDecodeError):pass
                    try:
                        with (folder/'control.jsonl').open('rb') as f:
                            f.seek(0,2);n=f.tell();f.seek(max(0,n-40000));lines=f.read().splitlines()
                        for line in reversed(lines):
                            try:record=json.loads(line);row.update(step=record['step'],phase=record['phase']);break
                            except (json.JSONDecodeError,KeyError):continue
                    except FileNotFoundError:pass
                    cases.append(row)
                running=[x for x in cases if x['started'] and 'score' not in x]
                started=[x for x in cases if x['started']]
                current=(running or started or cases)[-1]['name']
                body=json.dumps({'cases':cases,'current':current}).encode();mime='application/json'
            self.send_response(200);self.send_header('Content-Type',mime);self.send_header('Cache-Control','no-store');self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
        def log_message(self,*args):pass
    http=ThreadingHTTPServer(('127.0.0.1',a.port),partial(Handler,directory=str(root)))
    print(f'PREVIEW_READY http://127.0.0.1:{a.port}',flush=True);http.serve_forever()
if __name__=='__main__':main()
