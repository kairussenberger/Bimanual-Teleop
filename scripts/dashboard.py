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
 :root{--bg:#0e1116;--panel:#161a21;--ink:#dde3ea;--dim:#76808d;--blue:#6f9fe8;--orange:#e8854a;--gold:#d4af37;--green:#41d98d}
 body{background:var(--bg);color:var(--ink);font:14px/1.5 -apple-system,Segoe UI,sans-serif;margin:0}
 header{display:flex;gap:8px;align-items:center;padding:10px 16px;background:#11141a;border-bottom:1px solid #232936;flex-wrap:wrap}
 header b{font-size:15px;margin-right:8px}
 .chip{padding:4px 12px;border-radius:13px;background:#2a2f38;font-weight:600;font-size:13px}
 .ok{background:#1e5d3a}.bad{background:#7c2d2d}.warn{background:#7a6020}
 main{display:grid;grid-template-columns:minmax(560px,1fr) 360px;gap:14px;padding:14px;max-width:1250px}
 .panel{background:var(--panel);border:1px solid #232936;border-radius:12px;padding:12px 14px}
 canvas{display:block;border-radius:8px;background:#0b0e13;cursor:grab}
 .legend{color:var(--dim);font-size:12px;margin-top:8px}
 .legend i{display:inline-block;width:10px;height:10px;border-radius:5px;margin:0 4px -1px 10px}
 h3{margin:2px 0 8px;font-size:14px} .armttl{display:flex;justify-content:space-between;align-items:baseline}
 .gauge{display:grid;grid-template-columns:24px 1fr 62px;gap:8px;align-items:center;margin:3px 0;font-variant-numeric:tabular-nums}
 .bar{position:relative;height:12px;background:#222833;border-radius:6px;overflow:hidden}
 .bar .zone{position:absolute;top:0;bottom:0;background:#3a2026}
 .bar .tick{position:absolute;top:-1px;bottom:-1px;width:3px;background:#aab7c9;border-radius:2px}
 .bar .tick.alert{background:#ff6b6b}
 .kv{display:flex;justify-content:space-between;color:var(--dim);font-size:12.5px;margin:2px 0}
 .kv b{color:var(--ink);font-weight:600;font-variant-numeric:tabular-nums}
 .err-ok{color:var(--green)} .err-bad{color:#ff8a8a}
</style></head><body>
<header><b>bimanual-teleop</b>
 <span id=conn class="chip bad">stream …</span><span id=hz class=chip>— Hz</span>
 <span id=L class="chip bad">LEFT —</span><span id=R class="chip bad">RIGHT —</span>
 <span id=calib class=chip style="display:none"></span>
 <span style="flex:1"></span><span class=chip id=age>age —</span>
</header>
<main>
 <div class=panel>
  <canvas id=c3d width=620 height=560></canvas>
  <div class=legend>drag = orbit, scroll = zoom
   <i style="background:var(--orange)"></i>right arm <i style="background:var(--blue)"></i>left arm
   <i style="background:#fff"></i>EE <i style="background:var(--green)"></i>command target
   <i style="background:var(--gold)"></i>your hands (torso→wrist)</div>
 </div>
 <div>
  <div class=panel id=cardR style="margin-bottom:14px"></div>
  <div class=panel id=cardL></div>
 </div>
</main>
<script>
const $=id=>document.getElementById(id);
let yaw=1.05,pitch=0.26,scale=330,drag=null,RIG=null,CTR=[-0.1,-0.05,0.82];
const cv=$('c3d'),cx=cv.getContext('2d');
cv.onmousedown=e=>{drag=[e.clientX,e.clientY];cv.style.cursor='grabbing'};
window.onmouseup=()=>{drag=null;cv.style.cursor='grab'};
window.onmousemove=e=>{if(!drag)return;yaw-=(e.clientX-drag[0])*0.008;pitch+=(e.clientY-drag[1])*0.006;pitch=Math.max(-1.35,Math.min(1.35,pitch));drag=[e.clientX,e.clientY]};
cv.onwheel=e=>{e.preventDefault();scale*=e.deltaY<0?1.1:0.9;scale=Math.max(140,Math.min(900,scale))};
const sub=(a,b)=>[a[0]-b[0],a[1]-b[1],a[2]-b[2]], dotp=(a,b)=>a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
const cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
const norm=a=>{const n=Math.hypot(...a)||1;return[a[0]/n,a[1]/n,a[2]/n]};
function cam(){const cy=Math.cos(yaw),sy=Math.sin(yaw),cp=Math.cos(pitch),sp=Math.sin(pitch);
 const fwd=[cp*cy,cp*sy,sp];const right=norm(cross([0,0,1],fwd));const up=cross(fwd,right);return{right,up,fwd}}
let CAM=null;
function P(p){const q=sub(p,CTR);return{x:cv.width/2+dotp(q,CAM.right)*scale,y:cv.height/2-dotp(q,CAM.up)*scale,d:dotp(q,CAM.fwd)}}
let prims=[];
const shade=(c,d)=>{const f=Math.max(0.55,Math.min(1.15,1-d*0.35));return c.map(v=>Math.round(v*f))};
const rgb=c=>`rgb(${c[0]},${c[1]},${c[2]})`;
function seg(a,b,c,w){const A=P(a),B=P(b);prims.push({d:(A.d+B.d)/2,f(){cx.strokeStyle=rgb(shade(c,this.d));cx.lineWidth=w;cx.lineCap='round';cx.beginPath();cx.moveTo(A.x,A.y);cx.lineTo(B.x,B.y);cx.stroke()}})}
function dotPrim(p,c,r){const A=P(p);prims.push({d:A.d,f(){cx.fillStyle=rgb(shade(c,this.d));cx.beginPath();cx.arc(A.x,A.y,r,0,7);cx.fill()}})}
function ringPrim(p,c,r){const A=P(p);prims.push({d:A.d-0.001,f(){cx.strokeStyle=rgb(c);cx.lineWidth=2.5;cx.beginPath();cx.arc(A.x,A.y,r,0,7);cx.stroke()}})}
function flush(){prims.sort((a,b)=>b.d-a.d);for(const p of prims)p.f();prims=[]}
function grid(){cx.globalAlpha=0.6;for(let i=-4;i<=4;i++){seg([-1.1,i*0.25,0],[0.7,i*0.25,0],[34,40,52],1);seg([i*0.25-0.2,-1.05,0],[i*0.25-0.2,1.05,0],[34,40,52],1)}flush();cx.globalAlpha=1}
function axesHud(){const o=[46,cv.height-46],L=30;const ax=[[[-1,0,0],'FWD','#9fe89f'],[[0,1,0],'RIGHT','#e89f9f'],[[0,0,1],'UP','#9fb6e8']];
 cx.font='10px sans-serif';for(const[a,t,c]of ax){const u=dotp(a,CAM.right)*L,v=dotp(a,CAM.up)*L;cx.strokeStyle=c;cx.fillStyle=c;cx.lineWidth=2;
 cx.beginPath();cx.moveTo(o[0],o[1]);cx.lineTo(o[0]+u,o[1]-v);cx.stroke();cx.fillText(t,o[0]+u*1.35-9,o[1]-v*1.35+3)}}
const QCOL={left:[111,159,232],right:[232,133,74]};
function quat2cols(q){const[w,x,y,z]=q;return[[1-2*(y*y+z*z),2*(x*y+w*z),2*(x*z-w*y)],[2*(x*y-w*z),1-2*(x*x+z*z),2*(y*z+w*x)],[2*(x*z+w*y),2*(y*z-w*x),1-2*(x*x+y*y)]]}
function draw(s){
 CAM=cam();cx.clearRect(0,0,cv.width,cv.height);grid();
 let bases=[];
 const bl=s.arms.left,br=s.arms.right;
 if(bl&&bl.link_pos&&br&&br.link_pos){const a=bl.link_pos.slice(0,3),b=br.link_pos.slice(0,3);seg(a,b,[44,52,66],10)}
 for(const side of['left','right']){const a=s.arms[side];if(!a||!a.link_pos)continue;
  const Pn=[];for(let i=0;i<a.link_pos.length;i+=3)Pn.push(a.link_pos.slice(i,i+3));
  bases.push(Pn[0]);
  seg([Pn[0][0],Pn[0][1],0],Pn[0],[44,52,66],10);                      // stand column
  for(let i=1;i<Pn.length;i++)seg(Pn[i-1],Pn[i],QCOL[side],i<4?9:6.5); // the arm
  for(const p of Pn)dotPrim(p,[24,28,36],2.6);
  if(a.ee_pos){dotPrim(a.ee_pos,[255,255,255],4.5);
   if(a.ee_quat){const C=quat2cols(a.ee_quat),L=0.09,tri=[[224,72,72],[60,190,80],[80,110,250]];
    for(let k=0;k<3;k++)seg(a.ee_pos,[a.ee_pos[0]+C[k][0]*L,a.ee_pos[1]+C[k][1]*L,a.ee_pos[2]+C[k][2]*L],tri[k],2.5)}}
  if(a.cmd_pos){ringPrim(a.cmd_pos,[65,217,141],8);if(a.ee_pos)seg(a.ee_pos,a.cmd_pos,[65,217,141],1.5)}}
 if(bases.length===2&&s.op&&s.op.hands){
  const an=[(bases[0][0]+bases[1][0])/2,(bases[0][1]+bases[1][1])/2,(bases[0][2]+bases[1][2])/2-0.15];
  dotPrim(an,[212,175,55],5);
  for(const side of['left','right']){const h=s.op.hands[side];if(!h||!h.tracked||!h.wrist_body)continue;
   const w=h.wrist_body,p=[an[0]-w[2],an[1]+w[0],an[2]+w[1]];
   seg(an,p,[212,175,55],2.5);dotPrim(p,[212,175,55],5)}}
 flush();axesHud();
}
function gaugeRow(i,q,lo,hi,margin){
 const span=hi-lo||1,pos=Math.max(0,Math.min(1,(q-lo)/span));
 const alert=margin<0.12,warnZ=0.25/span*100;
 return `<div class=gauge><span style="color:var(--dim)">j${i+1}</span>
  <span class=bar><span class=zone style="left:0;width:${warnZ}%"></span><span class=zone style="right:0;width:${warnZ}%"></span>
  <span class="tick ${alert?'alert':''}" style="left:calc(${(pos*100).toFixed(1)}% - 1px)"></span></span>
  <span style="text-align:right;${alert?'color:#ff8a8a':''}">${(q*57.2958).toFixed(1)}&deg;</span></div>`}
function card(side,s){
 const a=s.arms[side];if(!a)return'';
 const lim=RIG?RIG[side]:null;
 let h=`<div class=armttl><h3 style="color:${side==='left'?'var(--blue)':'var(--orange)'}">${side.toUpperCase()} ARM</h3>`;
 const st=s.status;h+=`<span style="font-size:12px;color:${st.engaged[side]?'var(--green)':'var(--dim)'}">${st.engaged[side]?'ENGAGED':'idle'}${st.tracked[side]?'':' · NOT TRACKED'}</span></div>`;
 for(let i=0;i<6;i++){const lo=lim?lim.lo[i]:-3.14,hi=lim?lim.hi[i]:3.14;h+=gaugeRow(i,a.q[i],lo,hi,a.margins?a.margins[i]:1)}
 const op=s.op&&s.op.hands?s.op.hands[side]:null;
 if(op&&op.wrist_body){const w=op.wrist_body;h+=`<div class=kv><span>your hand (torso→wrist)</span><b>R ${w[0].toFixed(2)} · U ${w[1].toFixed(2)} · F ${w[2].toFixed(2)} m</b></div>`}
 if(a.ee_pos)h+=`<div class=kv><span>EE world [x,y,z]</span><b>${a.ee_pos.map(v=>v.toFixed(2)).join(', ')} m</b></div>`;
 if(a.cmd_pos&&a.ee_pos){const e=Math.hypot(...[0,1,2].map(k=>a.cmd_pos[k]-a.ee_pos[k]));
  h+=`<div class=kv><span>target gap (cmd−ee)</span><b class="${e<0.06?'err-ok':'err-bad'}">${(e*100).toFixed(1)} cm</b></div>`}
 return h}
function chip(id,cls,txt){const e=$(id);e.className='chip '+cls;e.textContent=txt}
async function tick(){
 try{
  if(!RIG){try{RIG=await(await fetch('/rig')).json()}catch(e){}}
  const d=await(await fetch('/state')).json();
  chip('conn',d.connected?'ok':'bad',d.connected?'stream connected':'STREAM OFFLINE');
  chip('age',d.age!=null&&d.age<0.3?'ok':'warn','age '+(d.age==null?'—':d.age.toFixed(2)+'s'));
  const s=d.state;
  if(s&&d.connected){
   chip('hz',s.status.hz>30?'ok':'warn',(s.status.hz||0).toFixed(0)+' Hz');
   for(const[side,id]of[['left','L'],['right','R']]){
    const tr=s.status.tracked[side],en=s.status.engaged[side];
    chip(id,tr?(en?'ok':'warn'):'bad',`${id==='L'?'LEFT':'RIGHT'} ${tr?(en?'tracked + engaged':'tracked'):'NO TRACKING'}`)}
   const c=$('calib');
   if(s.status.calib&&s.status.calib.msg){c.style.display='';c.className='chip warn';c.textContent='calib: '+s.status.calib.msg}else c.style.display='none';
   draw(s);
   $('cardL').innerHTML=card('left',s);$('cardR').innerHTML=card('right',s);
  }
 }catch(e){chip('conn','bad','dashboard error')}
 requestAnimationFrame(()=>setTimeout(tick,50));
}
tick();
</script></body></html>"""


def rig_info() -> dict:
    """Static per-side joint SOFT ranges for the dashboard gauges (falls back to
    hard limits if the IK model cannot be built)."""
    rig = load_rig()
    try:
        from bimanual_teleop.arms.ik import ArmIK
        out = {}
        for side in ("left", "right"):
            ik = ArmIK(rig, side)
            out[side] = {"lo": ik.soft_lo.tolist(), "hi": ik.soft_hi.tolist()}
        return out
    except Exception:
        lim = rig["arms"]["joint_limits"]
        return {side: {"lo": list(lim["lower"]), "hi": list(lim["upper"])}
                for side in ("left", "right")}


def make_server(feed: StateFeed, host: str, port: int, rig: dict | None = None) -> ThreadingHTTPServer:
    rig_body = json.dumps(rig or {}).encode()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):                                  # noqa: N802 (stdlib API)
            if self.path.startswith("/state"):
                body = json.dumps(feed.snapshot()).encode()
                ctype = "application/json"
            elif self.path.startswith("/rig"):
                body = rig_body
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
    srv = make_server(feed, args.host, args.port, rig=rig_info())
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
