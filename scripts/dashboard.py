#!/usr/bin/env python
"""Teleop dashboard — one browser page showing every piece of the puzzle, live.

Consumes the same newline-delimited TCP JSON `render.state` stream Unity uses
(no new schema, no extra deps) and serves a local page with:

  - stream/Quest status: connected, state age, loop Hz, per-side TRACKED/ENGAGED,
    calibration banner;
  - a drag-to-rotate 3D view: both arm link chains, achieved EE (dot) vs
    commanded target (ring), operator torso→wrist vectors;
  - numbers: per-joint angles (deg) with limit-margin highlighting, torso→wrist
    body vectors [right, up, forward], commanded-vs-achieved EE error.

    uv run python scripts/dashboard.py                 # http://127.0.0.1:8180
    uv run python scripts/dashboard.py --port 8181 --endpoint tcp://127.0.0.1:8102

Run alongside ANY teleop/jog process that has the Unity JSON bridge enabled
(run_teleop, run_hw with a render tee, scripts/jog_arms.py).
"""
from __future__ import annotations

import argparse
import json
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from bimanual_teleop.config import load_rig             # noqa: E402


class StateFeed:
    """Reconnecting reader for the newline-JSON render stream; keeps the latest."""

    def __init__(self, endpoint: str):
        host, port_s = endpoint.removeprefix("tcp://").rsplit(":", 1)
        self.addr = (host, int(port_s))
        self.latest: dict | None = None
        self.rx_time: float = 0.0
        self.connected = False
        self._stop = False
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop = True

    def snapshot(self) -> dict:
        age = (time.monotonic() - self.rx_time) if self.rx_time else None
        return {"connected": self.connected,
                "age": round(age, 3) if age is not None else None,
                "state": self.latest}

    def _run(self) -> None:
        while not self._stop:
            try:
                with socket.create_connection(self.addr, timeout=2.0) as sock:
                    self.connected = True
                    f = sock.makefile("r", encoding="utf-8")
                    while not self._stop:
                        line = f.readline()
                        if not line:
                            break
                        try:
                            self.latest = json.loads(line)
                            self.rx_time = time.monotonic()
                        except json.JSONDecodeError:
                            continue
            except OSError:
                pass
            self.connected = False
            time.sleep(0.5)


PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>bimanual-teleop dashboard</title>
<style>
 body{background:#101216;color:#dde3ea;font:13px/1.45 -apple-system,Segoe UI,sans-serif;margin:0}
 header{display:flex;gap:10px;align-items:center;padding:10px 14px;background:#181b21;flex-wrap:wrap}
 .chip{padding:3px 10px;border-radius:12px;background:#2a2f38;font-weight:600}
 .ok{background:#1e5d3a}.bad{background:#7c2d2d}.warn{background:#7a6020}
 main{display:flex;gap:14px;padding:14px;flex-wrap:wrap}
 canvas{background:#14171c;border-radius:10px}
 table{border-collapse:collapse;margin:6px 0 14px}
 td,th{padding:2px 9px;text-align:right;font-variant-numeric:tabular-nums}
 th{color:#8b95a3;font-weight:600}
 .lim{background:#7c2d2d;border-radius:4px}
 h3{margin:8px 0 2px;color:#9fb2c8} .sub{color:#76808d}
</style></head><body>
<header>
 <b>bimanual-teleop</b>
 <span id=conn class=chip>stream: …</span><span id=age class=chip>age: …</span>
 <span id=hz class=chip>… Hz</span>
 <span id=trkL class=chip>L track</span><span id=engL class=chip>L engage</span>
 <span id=trkR class=chip>R track</span><span id=engR class=chip>R engage</span>
 <span id=calib class=chip style="display:none"></span>
</header>
<main>
 <div><canvas id=c3d width=620 height=560></canvas>
  <div class=sub>drag to rotate &nbsp;|&nbsp; orange = right arm, blue = left, dot = achieved EE, ring = commanded, gold = your torso→wrist</div></div>
 <div id=tables></div>
</main>
<script>
let yaw=-0.7, pitch=0.32, drag=null;
const cv=document.getElementById('c3d'), cx=cv.getContext('2d');
cv.onmousedown=e=>drag=[e.clientX,e.clientY];
window.onmouseup=()=>drag=null;
window.onmousemove=e=>{if(!drag)return; yaw+=(e.clientX-drag[0])*0.008; pitch+=(e.clientY-drag[1])*0.006; pitch=Math.max(-1.4,Math.min(1.4,pitch)); drag=[e.clientX,e.clientY];};
const CTR=[ -0.05, 0, 0.95 ], SCALE=300;
function proj(p){ // robot world: +Z up, +Y right, -X forward
  let x=p[0]-CTR[0], y=p[1]-CTR[1], z=p[2]-CTR[2];
  let X= x*Math.cos(yaw)+y*Math.sin(yaw), Y=-x*Math.sin(yaw)+y*Math.cos(yaw);
  let Z= z*Math.cos(pitch)-X*Math.sin(pitch); X= X*Math.cos(pitch)+z*Math.sin(pitch);
  return [cv.width/2+Y*SCALE, cv.height/2-Z*SCALE];
}
function line(a,b,c,w){cx.strokeStyle=c;cx.lineWidth=w;cx.beginPath();cx.moveTo(...proj(a));cx.lineTo(...proj(b));cx.stroke();}
function dot(p,c,r){const q=proj(p);cx.fillStyle=c;cx.beginPath();cx.arc(q[0],q[1],r,0,7);cx.fill();}
function ring(p,c,r){const q=proj(p);cx.strokeStyle=c;cx.lineWidth=2;cx.beginPath();cx.arc(q[0],q[1],r,0,7);cx.stroke();}
function chip(id,on,txt,cls){const e=document.getElementById(id); e.textContent=txt; e.className='chip '+(on?cls||'ok':'bad');}
function qangle(a,b){if(!a||!b)return null;let d=Math.abs(a[0]*b[0]+a[1]*b[1]+a[2]*b[2]+a[3]*b[3]);return 2*Math.acos(Math.min(1,d))*57.2958;}
function draw(s){
  cx.clearRect(0,0,cv.width,cv.height);
  cx.globalAlpha=0.5;
  for(let i=-4;i<=4;i++){line([-1.0,i*0.25,0],[0.6,i*0.25,0],'#222831',1);line([i*0.25-0.2,-1,0],[i*0.25-0.2,1,0],'#222831',1);}
  cx.globalAlpha=1;
  const colors={left:'#6f9fe8',right:'#e8854a'};
  let bases=[];
  for(const side of ['left','right']){
    const a=s.arms[side]; if(!a||!a.link_pos) continue;
    const P=[]; for(let i=0;i<a.link_pos.length;i+=3)P.push(a.link_pos.slice(i,i+3));
    bases.push(P[0]);
    for(let i=1;i<P.length;i++) line(P[i-1],P[i],colors[side],4);
    P.forEach(p=>dot(p,colors[side],3.4));
    if(a.ee_pos) dot(a.ee_pos,'#fff',4);
    if(a.cmd_pos) ring(a.cmd_pos,'#41d98d',7);
    if(a.cmd_pos&&a.ee_pos) line(a.ee_pos,a.cmd_pos,'#41d98d',1.4);
  }
  if(bases.length==2&&s.op&&s.op.hands){
    const anchor=[(bases[0][0]+bases[1][0])/2,(bases[0][1]+bases[1][1])/2,(bases[0][2]+bases[1][2])/2-0.15];
    dot(anchor,'#caa520',5);
    for(const side of ['left','right']){
      const h=s.op.hands[side]; if(!h||!h.tracked||!h.wrist_body)continue;
      const w=h.wrist_body; // [right,up,fwd] -> world [-f, r, u]
      const p=[anchor[0]-w[2],anchor[1]+w[0],anchor[2]+w[1]];
      line(anchor,p,'#caa520',2); dot(p,'#caa520',4.5);
    }
  }
}
function tables(s){
  let h='';
  for(const side of ['left','right']){
    const a=s.arms[side]; if(!a)continue;
    h+=`<h3>${side.toUpperCase()} arm</h3><table><tr><th></th>`+[1,2,3,4,5,6].map(i=>`<th>j${i}</th>`).join('')+'</tr>';
    h+='<tr><td>deg</td>'+a.q.map((v,i)=>{const m=a.margins?a.margins[i]:1;return `<td class="${m<0.15?'lim':''}">${(v*57.2958).toFixed(1)}</td>`}).join('')+'</tr>';
    if(a.margins)h+='<tr><td>margin</td>'+a.margins.map(v=>`<td>${v.toFixed(2)}</td>`).join('')+'</tr>';
    h+='</table>';
    const op=s.op&&s.op.hands?s.op.hands[side]:null;
    if(op&&op.wrist_body)h+=`<div class=sub>torso→wrist [r,u,f]: ${op.wrist_body.map(v=>v.toFixed(3)).join(', ')}</div>`;
    if(a.cmd_pos&&a.ee_pos){
      const e=Math.hypot(a.cmd_pos[0]-a.ee_pos[0],a.cmd_pos[1]-a.ee_pos[1],a.cmd_pos[2]-a.ee_pos[2]);
      const oa=qangle(a.cmd_quat,a.ee_quat);
      h+=`<div class=sub>cmd−ee: ${(e*100).toFixed(1)} cm${oa!=null?', '+oa.toFixed(0)+'°':''}</div>`;
    }
  }
  document.getElementById('tables').innerHTML=h;
}
async function tick(){
  try{
    const r=await fetch('/state'); const d=await r.json();
    chip('conn',d.connected,d.connected?'stream: connected':'stream: OFFLINE');
    chip('age',d.age!=null&&d.age<0.5,'age: '+(d.age==null?'—':d.age.toFixed(2)+'s'),d.age<0.2?'ok':'warn');
    const s=d.state;
    if(s){
      chip('hz',s.status.hz>30,(s.status.hz||0).toFixed(0)+' Hz');
      chip('trkL',s.status.tracked.left,'L '+(s.status.tracked.left?'TRACKED':'no track'));
      chip('trkR',s.status.tracked.right,'R '+(s.status.tracked.right?'TRACKED':'no track'));
      chip('engL',s.status.engaged.left,'L '+(s.status.engaged.left?'ENGAGED':'idle'),'ok');
      chip('engR',s.status.engaged.right,'R '+(s.status.engaged.right?'ENGAGED':'idle'),'ok');
      const c=document.getElementById('calib');
      if(s.status.calib&&s.status.calib.msg){c.style.display='';c.textContent='calib: '+s.status.calib.msg;c.className='chip warn';}
      else c.style.display='none';
      draw(s); tables(s);
    }
  }catch(e){chip('conn',false,'dashboard error');}
  setTimeout(tick,66);
}
tick();
</script></body></html>"""


def make_server(feed: StateFeed, host: str, port: int) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):                                  # noqa: N802 (stdlib API)
            if self.path.startswith("/state"):
                body = json.dumps(feed.snapshot()).encode()
                ctype = "application/json"
            else:
                body = PAGE.encode()
                ctype = "text/html; charset=utf-8"
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):                         # quiet
            pass

    return ThreadingHTTPServer((host, port), Handler)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--endpoint", default=None,
                    help="render JSON stream (default: rig vr.unity_json_endpoint)")
    ap.add_argument("--port", type=int, default=8180, help="dashboard HTTP port")
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()

    endpoint = args.endpoint or load_rig().get("vr", {}).get("unity_json_endpoint", "tcp://127.0.0.1:8102")
    feed = StateFeed(endpoint)
    feed.start()
    srv = make_server(feed, args.host, args.port)
    print(f"[dashboard] http://{args.host}:{args.port}  ←  {endpoint}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        feed.stop()
        srv.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
