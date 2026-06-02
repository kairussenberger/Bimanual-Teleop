"""Run the full teleop pipeline against the MuJoCo sim, in ONE process.

Single-process is the right shape for the sim demo (the MuJoCo passive viewer must
own the main thread on macOS): a background VR source feeds latest poses; the main
loop runs IK + finger retarget for both sides and steps the viewer. The
hardware-grade multi-process/ZMQ split (for the 250 Hz CAN loops on Linux) reuses
the same TeleopEngine + controllers — see launch/run_hw.py.

    uv run mjpython -m bimanual_teleop.launch.run_sim                 # viewer, fake VR
    uv run mjpython -m bimanual_teleop.launch.run_sim --vr vuer       # viewer, real Quest
    uv run python    -m bimanual_teleop.launch.run_sim --gif out.gif  # headless GIF (no window)
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import threading
import time

from ..config import SIDES, load_rig
from ..engine import TeleopEngine
from ..safety.clutch import AlwaysOn
from ..safety.supervisor import Supervisor
from ..sim.sim_world import SimWorld
from ..vr.ingest import make_source


def run_gif(args) -> int:
    import imageio.v3 as iio
    rig = load_rig()
    if args.vr:
        rig["vr"]["transport"] = args.vr
    world = SimWorld(rig)
    engine = TeleopEngine(rig, world)
    supervisor = Supervisor(rig, AlwaysOn())
    src = make_source(rig)
    frames = []
    n = int(args.seconds * args.fps)
    for i in range(n):
        t = i / args.fps
        frame = src.frame_at(t) if hasattr(src, "frame_at") else src.latest()
        engine.tick(frame, supervisor.update(frame, t), t)
        world.step(4)
        if i % max(1, int(args.fps / 20)) == 0:
            frames.append(world.render_rgb(azimuth=args.azimuth, elevation=args.elevation))
    iio.imwrite(args.gif, frames, duration=1000 / 20, loop=0)
    print(f"wrote {args.gif} ({len(frames)} frames)")
    return 0


def _start_tunnel() -> subprocess.Popen | None:
    """Spawn a cloudflared quick tunnel to the local HTTP Vuer server and print
    the public https URL to paste on the Quest. No account needed."""
    if not shutil.which("cloudflared"):
        print("!! cloudflared not found — run: brew install cloudflared")
        return None
    proc = subprocess.Popen(["cloudflared", "tunnel", "--url", "http://localhost:8012"],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

    def watch():
        for line in proc.stdout:
            m = re.search(r"https://[-\w.]+\.trycloudflare\.com", line)
            if m:
                print("\n" + "=" * 64 + f"\n  OPEN THIS ON THE QUEST 3 BROWSER:\n    {m.group(0)}\n"
                      + "  (then Enter VR and raise your hands)\n" + "=" * 64 + "\n", flush=True)
    threading.Thread(target=watch, daemon=True).start()
    return proc


def _lan_ip() -> str:
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except OSError:
        return "<your-LAN-IP>"
    finally:
        s.close()


def _print_lan_url() -> None:
    ip = _lan_ip()
    print("\n" + "=" * 64 + f"\n  OPEN THIS EXACT URL ON THE QUEST 3 BROWSER:\n    https://{ip}:8012\n"
          + "  (NOT the vuer.ai?ws=... line above — that one 502s / won't connect.)\n"
          + "  Accept the cert warning, Enter VR, raise your hands.\n" + "=" * 64 + "\n", flush=True)


def run_viewer(args) -> int:
    import mujoco.viewer
    rig = load_rig()
    if args.vr:
        rig["vr"]["transport"] = args.vr
    tunnel = None
    if args.tunnel:
        rig["vr"]["transport"] = "vuer"
        rig["vr"]["tunnel"] = True
        tunnel = _start_tunnel()
    elif (args.vr == "vuer") or rig["vr"].get("transport") == "vuer":
        _print_lan_url()
    world = SimWorld(rig)
    engine = TeleopEngine(rig, world)
    supervisor = Supervisor(rig, AlwaysOn())
    src = make_source(rig)
    src.start()
    try:
        with mujoco.viewer.launch_passive(world.model, world.data) as v:
            while v.is_running():
                t = time.monotonic()   # one clock shared with the source stamps + supervisor
                frame = src.latest()
                engine.tick(frame, supervisor.update(frame, t), t)
                world.step(2)
                v.sync()
                time.sleep(1 / 120)
    finally:
        src.stop()
        if tunnel is not None:
            tunnel.terminate()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vr", choices=["fake", "vuer"], help="override vr.transport")
    ap.add_argument("--tunnel", action="store_true",
                    help="serve over a public cloudflared HTTPS URL (works on isolated/campus Wi-Fi)")
    ap.add_argument("--gif", metavar="PATH", help="headless: render a GIF and exit")
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--fps", type=float, default=60.0)
    ap.add_argument("--azimuth", type=float, default=70)
    ap.add_argument("--elevation", type=float, default=-18)
    args = ap.parse_args()
    return run_gif(args) if args.gif else run_viewer(args)


if __name__ == "__main__":
    raise SystemExit(main())
