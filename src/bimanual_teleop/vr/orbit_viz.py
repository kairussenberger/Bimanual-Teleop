"""Browser hand-tracking visualizer, served FROM the live ORBIT source process.

The teleop process binds the ORBIT NetMQ ports, so a separate web_viz can't bind
them too. Instead this serves a tiny three.js page off the source's already-parsed
frames: you get the live hand skeletons + wrist triads + head FOV in a browser tab
next to the MuJoCo window, so you can always see whether your hands are tracked.

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
#hud{position:fixed;top:10px;left:12px;z-index:10;line-height:1.6;text-shadow:0 1px 2px #000}
#hud b{color:#89b4fa}.r{color:#74c0fc}.l{color:#ff8787}.stale{color:#6c7086}</style></head>
<body><div id=hud></div>
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
cam.position.set(0,0.1,0.6);
const rnd=new THREE.WebGLRenderer({antialias:true});rnd.setSize(innerWidth,innerHeight);
rnd.setPixelRatio(devicePixelRatio);document.body.appendChild(rnd.domElement);
const ctrl=new THREE.OrbitControls(cam,rnd.domElement);ctrl.enabled=false;
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

let mode='front';
addEventListener('keydown',e=>{const k=e.key.toLowerCase();if(k==='o'){mode='orbit';ctrl.enabled=true;}if(k==='f'){mode='front';ctrl.enabled=false;}});
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
      if(mode==='front'){cam.position.copy(hpos).addScaledVector(b.f,-0.55).addScaledVector(b.u,0.18);cam.up.set(0,1,0);cam.lookAt(hpos.clone().addScaledVector(b.f,0.45));}
    }else{head.visible=false;fov.visible=false;}
    document.getElementById('hud').innerHTML=`<b>ORBIT hands</b> &nbsp; R: ${fmt(latest.hands.right,'r')} &nbsp; L: ${fmt(latest.hands.left,'l')}`+
      `<br><span class=r>blue=right</span> <span class=l>red=left</span> · triad=wrist · yellow=FOV · <b>F</b> front-lock <b>O</b> orbit`;
    const ov=latest.overlay||{};const cal=document.getElementById('cal');
    if(ov.text){cal.style.display='flex';
      const col=ov.color==='green'?'#a6e3a1':(ov.color==='red'?'#f38ba8':'#f9e2af');
      const m=document.getElementById('calmsg');m.textContent=ov.text;m.style.color=col;
      const n=document.getElementById('calnum');n.textContent=(ov.count!=null?ov.count:'');n.style.color=col;
    }else{cal.style.display='none';}
  }
  ctrl.update();rnd.render(scene,cam);
}
loop();
</script></body></html>"""


def start_viz(get_snapshot, port: int = 8099):
    """Serve the hand viz; returns (server, url). get_snapshot() -> JSON-able dict."""
    page = HTML.replace("__BONES__", json.dumps(BONES)).encode()

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, body, ctype):
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path.startswith("/data"):
                self._send(json.dumps(get_snapshot()).encode(), "application/json")
            else:
                self._send(page, "text/html")

    srv = ThreadingHTTPServer(("127.0.0.1", port), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{port}"
