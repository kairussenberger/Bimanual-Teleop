"""Browser hand-tracking visualizer, served FROM the live ORBIT source process.

The teleop process binds the ORBIT NetMQ ports, so a separate web_viz can't bind
them too. Instead this serves a tiny three.js page off the source's already-parsed
frames: you get the live hand skeletons + wrist triads + head FOV in a browser tab,
so you can always see whether your hands are tracked while Unity or hardware runs.

`start_viz(get_snapshot, port)` runs an HTTP server in a daemon thread. get_snapshot
returns a JSON-able dict (see OrbitVRSource.viz_snapshot): {hands:{right,left}, head}.
Data is in the WebXR frame (x right, y up, -z forward) which matches three.js, so it
renders directly. Front-locked camera (rides head pose); press O to free-orbit, F to
re-lock.
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# 25 W3C XRHand joints (wrist, then thumb/index/middle/ring/pinky chains).
BONES = [
    [0, 1], [1, 2], [2, 3], [3, 4],
    [0, 5], [5, 6], [6, 7], [7, 8], [8, 9],
    [0, 10], [10, 11], [11, 12], [12, 13], [13, 14],
    [0, 15], [15, 16], [16, 17], [17, 18], [18, 19],
    [0, 20], [20, 21], [21, 22], [22, 23], [23, 24],
]

HTML = """<!doctype html><html><head><meta charset=utf-8><title>ORBIT hands</title>
<style>body{margin:0;background:#0b0e14;color:#cdd6f4;font:13px ui-monospace,monospace;overflow:hidden}
#hud{position:fixed;top:10px;left:12px;z-index:10;line-height:1.6;text-shadow:0 1px 2px #000;pointer-events:none}
#hud b{color:#89b4fa}.r{color:#74c0fc}.l{color:#ff8787}.stale{color:#6c7086}</style></head>
<body><div id=hud></div>
<div id=err style="position:fixed;bottom:8px;left:12px;z-index:40;color:#f38ba8;font:12px ui-monospace,monospace;white-space:pre-wrap;pointer-events:none"></div>
<script>addEventListener('error',e=>{var d=document.getElementById('err');if(d)d.textContent='JS ERROR: '+e.message+'  ('+e.lineno+':'+e.colno+')';});</script>
<div id=wiz style="position:fixed;top:12px;left:50%;transform:translateX(-50%);z-index:30;display:none;background:rgba(11,14,20,.85);border:1px solid #45475a;border-radius:8px;padding:10px 18px;text-align:center;text-shadow:0 1px 2px #000;pointer-events:none"></div>
<div id=cal style="position:fixed;inset:0;z-index:20;display:none;align-items:center;justify-content:center;flex-direction:column;pointer-events:none;text-align:center;text-shadow:0 2px 8px #000">
  <div id=calmsg style="font-size:34px;font-weight:700"></div>
  <div id=calnum style="font-size:140px;font-weight:800;line-height:1"></div>
</div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
<script>
const BONES=__BONES__;
const scene=new THREE.Scene();
const cam=new THREE.PerspectiveCamera(55,innerWidth/innerHeight,0.01,100);
cam.position.set(0,1.0,0.35);
const rnd=new THREE.WebGLRenderer({antialias:true});rnd.setSize(innerWidth,innerHeight);
rnd.setPixelRatio(devicePixelRatio);document.body.appendChild(rnd.domElement);
const ctrl=new THREE.OrbitControls(cam,rnd.domElement);ctrl.target.set(0,0.85,-0.5);ctrl.enabled=true;
scene.add(new THREE.GridHelper(2,20,0x313244,0x1e1e2e));
scene.add(new THREE.HemisphereLight(0xffffff,0x222233,1.3));
addEventListener('resize',()=>{cam.aspect=innerWidth/innerHeight;cam.updateProjectionMatrix();rnd.setSize(innerWidth,innerHeight);});

function makeHand(color){
  const g=new THREE.Group();const sph=[];const sg=new THREE.SphereGeometry(0.006,8,8);
  const sm=new THREE.MeshStandardMaterial({color,emissive:color,emissiveIntensity:0.4});
  for(let i=0;i<25;i++){const m=new THREE.Mesh(sg,sm);g.add(m);sph.push(m);}
  const bg=new THREE.BufferGeometry();
  bg.setAttribute('position',new THREE.BufferAttribute(new Float32Array(BONES.length*2*3),3));
  const lines=new THREE.LineSegments(bg,new THREE.LineBasicMaterial({color}));lines.frustumCulled=false;
  g.add(lines);scene.add(g);
  const triad=new THREE.Group();
  for(const[ax,c]of[[[1,0,0],0xff5555],[[0,1,0],0x55ff55],[[0,0,1],0x5599ff]]){
    const tg=new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(),new THREE.Vector3(...ax).multiplyScalar(0.06)]);
    triad.add(new THREE.Line(tg,new THREE.LineBasicMaterial({color:c})));}
  scene.add(triad);
  return {g,sph,lines,triad};
}
const hands={right:makeHand(0x74c0fc),left:makeHand(0xff8787)};
const head=new THREE.Mesh(new THREE.BoxGeometry(0.07,0.045,0.05),new THREE.MeshStandardMaterial({color:0x9399b2}));scene.add(head);
const HFOV=110*Math.PI/180,VFOV=96*Math.PI/180,FD=0.6;
const fov=new THREE.LineSegments(new THREE.BufferGeometry(),new THREE.LineBasicMaterial({color:0xf9e2af,transparent:true,opacity:0.75}));
fov.geometry.setAttribute('position',new THREE.BufferAttribute(new Float32Array(8*2*3),3));fov.frustumCulled=false;scene.add(fov);

let mode='orbit';   // DRAGGABLE by default; 'f' locks to the saved view / head-chase
// ===== two overlays for understanding the teleop mapping =====
//  [1] PSEUDO-SKELETON: torso (head->hip, down), shoulders (out from torso), then per
//      hand a POSITION vector (shoulder->wrist) and a WRIST vector (wrist->fingertip,
//      orientation). Torso/shoulders are inferred from the head; wrist/tip are tracked.
//  [2] YAM-LINK chain: the SAME shoulder->wrist->tip path but built from the i2rt YAM's
//      ACTUAL link lengths laid end-to-end, so you can see the robot's reach budget vs
//      where your hand is (the red stub = robot wrist-reach end -> your actual wrist).
const V=a=>new THREE.Vector3(a[0],a[1],a[2]);
const NECK=0.20,SHW=0.20;                     // head-relative shoulder FALLBACK (m) before calibration
const YAM=[0.0914,0.0584,0.2678,0.2489,0.0877,0.0,0.1121];  // i2rt YAM link lengths (m)
const YAM_REACH=YAM[0]+YAM[1]+YAM[2]+YAM[3];  // base->j4 reach (~0.67m): how far behind the wrist the calib base sits
function mkArrow(c){const a=new THREE.ArrowHelper(new THREE.Vector3(0,0,1),new THREE.Vector3(),0.1,c,0.05,0.03);a.visible=false;scene.add(a);return a;}
function setArrow(a,from,to){const d=to.clone().sub(from),L=d.length();if(L<1e-4){a.visible=false;return;}
  a.position.copy(from);a.setDirection(d.normalize());a.setLength(L,Math.min(0.06,L*0.3),Math.min(0.04,L*0.18));a.visible=true;}
const posA={right:mkArrow(0xfab387),left:mkArrow(0xfab387)};   // orange: vector TO the wrist (arm base -> wrist)
const wrA ={right:mkArrow(0xa6e3a1),left:mkArrow(0xa6e3a1)};   // green:  vector OF the wrist (wrist -> fingertip)
let showVec=true;                             // toggle the two vectors (key 1)
let lenScale=1.0;
// CALIBRATION (frozen once set, persisted to disk). Each entry holds WORLD-frame pose so
// nothing drifts: arm = {shoulder, wristRef, quatRef}; cam = {pos, target}.
let calib={right:null,left:null,cam:null};
let wiz=null;const WIZ=['right','left','cam'];     // wizard step order
// live refs the wizard captures from (updated every frame in loop)
let curHead=null,curFwd=null,curB=null;const curHand={right:null,left:null};
fetch('/calib').then(r=>r.json()).then(d=>{if(d&&d.calib)calib=d.calib;if(d&&d.lenScale)lenScale=d.lenScale;}).catch(()=>{});
function saveCalib(){fetch('/calib',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({calib,lenScale})}).catch(()=>{});}
function captureStep(s){
  if(s==='right'||s==='left'){const h=curHand[s];if(!h||!curHead||!curFwd)return;
    const sh=h.wrist.clone().addScaledVector(curFwd,-lenScale*YAM_REACH);    // arm base one reach behind the wrist
    calib[s]={shoulder:sh.toArray(),wristRef:h.wrist.toArray(),quatRef:h.quat.slice()};}
  else if(s==='cam'){calib.cam={pos:cam.position.toArray(),target:ctrl.target.toArray()};}
}
function updateWiz(){const el=document.getElementById('wiz');if(!el)return;
  if(wiz===null){el.style.display='none';return;}el.style.display='block';
  const tag=s=>calib[s]?'<span style="color:#a6e3a1">✓</span>':'<span style="color:#6c7086">·</span>';
  const msg={right:'Hold your RIGHT hand flat & horizontal, in front of your torso',
             left:'Hold your LEFT hand flat & horizontal, in front of your torso',
             cam:'Drag with the mouse to the camera view you want'}[wiz];
  el.innerHTML=`<div style="font-size:21px;font-weight:700;color:#f9e2af">CALIBRATE · ${wiz.toUpperCase()}</div>`+
    `<div style="font-size:15px;margin:5px 0">${msg}</div>`+
    `<div style="opacity:.85"><b>c</b> capture &amp; next · <b>b</b> back · <b>x</b> cancel &nbsp;&nbsp; right ${tag('right')} left ${tag('left')} cam ${tag('cam')}</div>`;
}
addEventListener('keydown',e=>{const k=e.key.toLowerCase();
  if(k==='o'){mode='orbit';if(curHead)ctrl.target.copy(curHead);ctrl.enabled=true;}   // drag to reposition the view
  if(k==='f'){mode='front';}                                                          // back to auto (frozen / chase)
  if(k==='v'){calib.cam={pos:cam.position.toArray(),target:ctrl.target.toArray()};mode='front';saveCalib();} // save current view
  if(k==='1')showVec=!showVec;
  if(k==='-'||k==='_'){lenScale=Math.max(0.2,lenScale*0.91);saveCalib();}
  if(k==='='||k==='+'){lenScale=Math.min(5.0,lenScale*1.10);saveCalib();}
  if(k==='c'){ if(wiz===null){wiz='right';}
    else{captureStep(wiz);const i=WIZ.indexOf(wiz);wiz=(i+1<WIZ.length)?WIZ[i+1]:null;saveCalib();if(wiz===null)mode='front';}
    if(wiz==='cam'){mode='orbit';if(curHead)ctrl.target.copy(curHead);ctrl.enabled=true;}   // drag to set the view
    updateWiz(); }
  if(k==='x'){calib={right:null,left:null,cam:null};wiz=null;mode='orbit';saveCalib();updateWiz();}
  if(k==='b'){const i=wiz?WIZ.indexOf(wiz):WIZ.length;wiz=WIZ[Math.max(0,i-1)];updateWiz();}
});
function basis(q){const Q=new THREE.Quaternion(q[0],q[1],q[2],q[3]);
  return {f:new THREE.Vector3(0,0,-1).applyQuaternion(Q).normalize(),
          u:new THREE.Vector3(0,1,0).applyQuaternion(Q).normalize(),
          r:new THREE.Vector3(1,0,0).applyQuaternion(Q).normalize()};}
function updHand(h,d){
  const on=d&&d.pts&&(d.age==null||d.age<0.5);h.g.visible=on;h.triad.visible=on;if(!on)return;
  const p=d.pts;for(let i=0;i<25;i++)h.sph[i].position.set(p[i][0],p[i][1],p[i][2]);
  const a=h.lines.geometry.attributes.position.array;let k=0;
  for(const[x,y]of BONES){a[k++]=p[x][0];a[k++]=p[x][1];a[k++]=p[x][2];a[k++]=p[y][0];a[k++]=p[y][1];a[k++]=p[y][2];}
  h.lines.geometry.attributes.position.needsUpdate=true;
  if(d.wrist){h.triad.position.set(...d.wrist.pos);h.triad.quaternion.set(...d.wrist.quat);}
}
let latest=null;
async function poll(){try{latest=await (await fetch('/data')).json();}catch(e){}}
setInterval(poll,40);
function fmt(d,c){return (d&&(d.age==null||d.age<0.5))?`<span class=${c}>TRACKING</span>`:`<span class=stale>-- not tracked</span>`;}
function loop(){requestAnimationFrame(loop);
  if(latest){
    updHand(hands.right,latest.hands.right);updHand(hands.left,latest.hands.left);
    const hp=(latest.head&&(latest.head.age==null||latest.head.age<0.5))?latest.head:null;
    if(hp){const b=basis(hp.quat);const hpos=new THREE.Vector3(...hp.pos);
      // Head + hands live in ONE world frame (wrist.pos == hand keypoint[0]); the
      // head MUST be drawn at its real tracked position, else the hands (rendered at
      // their world coords) float ~1m off from the head box. Then the chase camera
      // rides BEHIND + ABOVE the head looking forward, so the hands sit in front of it.
      head.visible=true;head.position.copy(hpos);head.lookAt(hpos.clone().add(b.f));
      const apex=hpos.clone(),fc=apex.clone().addScaledVector(b.f,FD);
      const hw=FD*Math.tan(HFOV/2),hh=FD*Math.tan(VFOV/2);
      const c=[[1,1],[-1,1],[-1,-1],[1,-1]].map(([sx,sy])=>fc.clone().addScaledVector(b.r,sx*hw).addScaledVector(b.u,sy*hh));
      const segs=[];for(const cc of c){segs.push(apex,cc);}for(let i=0;i<4;i++)segs.push(c[i],c[(i+1)%4]);
      const fa=fov.geometry.attributes.position.array;let j=0;for(const v of segs){fa[j++]=v.x;fa[j++]=v.y;fa[j++]=v.z;}
      fov.geometry.attributes.position.needsUpdate=true;fov.visible=true;
      // ---- overlays: TWO vectors per hand ----
      //   posA (orange) = vector TO the wrist: calibrated arm base -> live wrist
      //   wrA  (green)  = vector OF the wrist: wrist -> middle fingertip (hand pointing/orientation)
      const wd=new THREE.Vector3(0,-1,0);
      const fH=new THREE.Vector3(b.f.x,0,b.f.z);if(fH.lengthSq()<1e-6)fH.set(0,0,-1);fH.normalize();
      const rH=new THREE.Vector3(b.r.x,0,b.r.z);if(rH.lengthSq()<1e-6)rH.set(1,0,0);rH.normalize();
      curHead=hpos.clone();curFwd=fH.clone();curB=b;
      const liveHand=side=>{const hd=latest.hands[side];if(!hd||!hd.pts||(hd.age!=null&&hd.age>=0.5))return null;
        return {wrist:hd.wrist?V(hd.wrist.pos):V(hd.pts[0]),tip:V(hd.pts[14]),quat:hd.wrist?hd.wrist.quat:[0,0,0,1]};};
      const live={right:liveHand('right'),left:liveHand('left')};
      curHand.right=live.right;curHand.left=live.left;
      const baseOf=sd=>calib[sd]?new THREE.Vector3().fromArray(calib[sd].shoulder)
        :hpos.clone().addScaledVector(wd,NECK).addScaledVector(rH,sd==='right'?SHW:-SHW);   // calibrated base, else fallback
      for(const sd of ['right','left']){const s=live[sd];
        if(s&&showVec){setArrow(posA[sd],baseOf(sd),s.wrist);setArrow(wrA[sd],s.wrist,s.tip);}
        else{posA[sd].visible=false;wrA[sd].visible=false;}}
    }else{head.visible=false;fov.visible=false;
      [posA.right,posA.left,wrA.right,wrA.left].forEach(a=>a.visible=false);}
    // ---- camera: drag (orbit) wins so you can ALWAYS reposition; else frozen calib; else chase ----
    if(mode==='orbit'||wiz==='cam'){ctrl.enabled=true;}
    else if(calib.cam){ctrl.enabled=false;const tgt=new THREE.Vector3().fromArray(calib.cam.target);
      cam.position.fromArray(calib.cam.pos);cam.up.set(0,1,0);ctrl.target.copy(tgt);cam.lookAt(tgt);}
    else if(curHead&&curB){ctrl.enabled=false;const BACK=0.70,UP=0.12,FWD=0.45,DOWN=0.15;
      cam.position.copy(curHead).addScaledVector(curB.f,-BACK).addScaledVector(curB.u,UP);cam.up.set(0,1,0);
      cam.lookAt(curHead.clone().addScaledVector(curB.f,FWD).addScaledVector(curB.u,-DOWN));}
    const cs=s=>calib[s]?'<span style="color:#a6e3a1">set</span>':'<span style="color:#6c7086">—</span>';
    const cm=(mode==='orbit'||wiz==='cam')?'<span style="color:#f9e2af">DRAG (mouse)</span>':(calib.cam?'frozen':'chase');
    document.getElementById('hud').innerHTML=`<b>ORBIT hands</b> &nbsp; R: ${fmt(latest.hands.right,'r')} &nbsp; L: ${fmt(latest.hands.left,'l')}`+
      `<br><span style="color:#fab387">&rarr; to wrist</span> &nbsp; <span style="color:#a6e3a1">&rarr; of wrist</span> &nbsp; <b>1</b> vectors:${showVec?'on':'off'}`+
      `<br>cam: ${cm} &nbsp; <b>o</b> drag · <b>v</b> save view · <b>f</b> lock · <b>c</b> calibrate · <b>x</b> clear &nbsp; [R:${cs('right')} L:${cs('left')} cam:${cs('cam')}]`;
    const ov=latest.overlay||{};const cal=document.getElementById('cal');
    if(ov.text){cal.style.display='flex';
      const col=ov.color==='green'?'#a6e3a1':(ov.color==='red'?'#f38ba8':'#f9e2af');
      const m=document.getElementById('calmsg');m.textContent=ov.text;m.style.color=col;
      const n=document.getElementById('calnum');n.textContent=(ov.count!=null?ov.count:'');n.style.color=col;
    }else{cal.style.display='none';}
  }
  if(ctrl.enabled)ctrl.update();rnd.render(scene,cam);
}
loop();
</script></body></html>"""


# Calibration persists here so a captured rig (arm-base anchors + camera) survives sim
# restarts — "once it's set it doesn't change". Repo-root config/viz_calib.json.
CALIB_PATH = Path(__file__).resolve().parents[3] / "config" / "viz_calib.json"


def start_viz(get_snapshot, port: int = 8099, calib_path: Path = CALIB_PATH):
    """Serve the hand viz; returns (server, url). get_snapshot() -> JSON-able dict.
    GET /calib -> saved calibration JSON; POST /calib -> persist it to calib_path."""
    page = HTML.replace("__BONES__", json.dumps(BONES)).encode()

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, body, ctype, code=200):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path.startswith("/data"):
                self._send(json.dumps(get_snapshot()).encode(), "application/json")
            elif self.path.startswith("/calib"):
                try:
                    body = calib_path.read_bytes()
                except OSError:
                    body = b"{}"
                self._send(body, "application/json")
            else:
                self._send(page, "text/html")

        def do_POST(self):
            if not self.path.startswith("/calib"):
                self._send(b"not found", "text/plain", 404)
                return
            n = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(n) if n else b"{}"
            try:
                json.loads(raw)                      # validate before persisting
                calib_path.parent.mkdir(parents=True, exist_ok=True)
                calib_path.write_bytes(raw)
                self._send(b'{"ok":true}', "application/json")
            except (ValueError, OSError) as e:
                self._send(json.dumps({"ok": False, "error": str(e)}).encode(), "application/json", 400)

    srv = ThreadingHTTPServer(("127.0.0.1", port), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{port}"
