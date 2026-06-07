"""Run the full teleop pipeline against the MuJoCo sim, in ONE process.

Single-process is the right shape for the sim demo (the MuJoCo passive viewer must
own the main thread on macOS): a background VR source feeds latest poses; the main
loop runs IK + finger retarget for both sides and steps the viewer. The
hardware-grade multi-process/ZMQ split (for the 250 Hz CAN loops on Linux) reuses
the same TeleopEngine + controllers — see launch/run_hw.py.

    uv run mjpython -m bimanual_teleop.launch.run_sim                 # viewer, fake VR
    uv run mjpython -m bimanual_teleop.launch.run_sim --vr vuer       # viewer, real Quest (WebXR/browser)
    uv run mjpython -m bimanual_teleop.launch.run_sim --vr orbit      # viewer, real Quest (ORBIT app, NetMQ/adb)
    uv run python    -m bimanual_teleop.launch.run_sim --gif out.gif  # headless GIF (no window)
"""
from __future__ import annotations

import argparse
import os
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
    # --protocol http2: campus Wi-Fi (eduroam) throttles/blocks QUIC (UDP 7844),
    # which shows up as endless "context canceled / control stream failure" drops.
    # HTTP/2 runs over plain TCP 443 and stays up. --edge-ip-version 4 dodges flaky IPv6.
    proc = subprocess.Popen(
        ["cloudflared", "tunnel", "--url", "http://localhost:8012",
         "--protocol", "http2", "--edge-ip-version", "4"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

    def watch():
        for line in proc.stdout:
            m = re.search(r"https://[-\w.]+\.trycloudflare\.com", line)
            if m:
                print("\n" + "=" * 64 + f"\n  OPEN THIS ON THE QUEST 3 BROWSER:\n    {m.group(0)}\n"
                      + "  (then Enter VR and raise your hands)\n" + "=" * 64 + "\n", flush=True)
    threading.Thread(target=watch, daemon=True).start()
    return proc


def _free_port(port: int = 8012) -> None:
    """Kill any leftover process holding the Vuer port — a stuck server from a
    previous run / Ctrl+C / headset hang is the usual 'doesn't connect' cause."""
    try:
        pids = subprocess.run(["lsof", "-ti", f"tcp:{port}"], capture_output=True, text=True).stdout.split()
    except FileNotFoundError:
        return
    for pid in pids:
        try:
            os.kill(int(pid), 9)
            print(f"(freed port {port}: killed stale pid {pid})", flush=True)
        except (ProcessLookupError, ValueError, PermissionError):
            pass


def _wait_port(port: int = 8012, timeout: float = 15.0) -> None:
    import socket
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with socket.socket() as s:
            s.settimeout(0.5)
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.3)


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


def _push_robot_frames(src, engine, frame) -> None:
    """Send each robot hand ORIENTATION, expressed in the headset frame, to the
    Vuer viz (it pins position itself). robot_R_webxr = R_base_from_vrᵀ · ee_R_base."""
    from ..config import SIDES
    for s in SIDES:
        hs = frame.hands.get(s)
        ac = engine.arm[s]
        if hs is None or not hs.tracked or ac.mapper.R is None:
            continue
        ee_R = ac.ik.fk_ee().rotation().as_matrix()
        R_webxr = ac.mapper.R.T @ ee_R                       # robot hand orientation in headset frame
        src.set_robot_frame(s, R_webxr)


def _draw_frames(scn, engine, frame=None, engaged=None, calibrating=False) -> None:
    """Overlay, at each robot hand: SOLID triad = where the robot's EE actually is;
    FAINT longer triad = the orientation the operator is commanding. If wrist ROLL
    is mapped right, twisting your hand spins the faint triad about its blue axis and
    the solid one follows; if instead the arm arcs, the two diverge — that's the bug,
    now visible.

    Plus a STATUS DOT above each hand so you never need the terminal:
      GREEN  = that hand is tracked AND driving the arm,
      YELLOW = calibrating — hold the rest pose (arms down, palms inward),
      RED    = that hand is NOT tracked (no data / out of camera view)."""
    import numpy as np
    from ..viz import overlay
    scn.ngeom = 0
    engaged = engaged or {}
    for s in SIDES:
        ac = engine.arm[s]
        ee = ac.ik.fk_ee()
        pos = ac.base_R @ ee.translation() + ac.base_pos
        overlay.triad(scn, pos, ac.base_R @ ee.rotation().as_matrix(), 0.12, 0.007, 1.0)  # actual
        if ac.cmd_R is not None:
            overlay.triad(scn, pos, ac.base_R @ ac.cmd_R, 0.17, 0.004, 0.4)               # commanded
        h = frame.hands.get(s) if frame else None
        tracked = bool(h and getattr(h, "tracked", False))
        if calibrating:
            col = (1.0, 0.85, 0.2, 1.0)            # yellow: calibrating
        elif tracked and engaged.get(s):
            col = (0.15, 0.9, 0.3, 1.0)            # green: tracked + driving
        elif tracked:
            col = (0.3, 0.6, 1.0, 1.0)             # blue: tracked but not engaged
        else:
            col = (0.95, 0.2, 0.2, 1.0)            # red: not tracked
        overlay.sphere(scn, pos + np.array([0.0, 0.0, 0.22]), 0.035, col)


def _rpy_deg(R) -> tuple[float, float, float]:
    """ZYX roll/pitch/yaw (deg) of a 3x3 rotation — roll is the forearm twist."""
    import math
    roll = math.degrees(math.atan2(R[2, 1], R[2, 2]))
    pitch = math.degrees(math.atan2(-R[2, 0], (R[2, 1] ** 2 + R[2, 2] ** 2) ** 0.5))
    yaw = math.degrees(math.atan2(R[1, 0], R[0, 0]))
    return roll, pitch, yaw


def _hud_lines(engine, frame, engaged, hz) -> list[str]:
    """The in-headset status/log lines (Section 6 HUD, headset edition): calib/teleop
    state, loop rate, and per-hand tracked + live wrist roll + clutch state."""
    import numpy as np
    cal = engine.calib_status
    if cal and cal.get("active") and cal.get("phase") != "done":
        lines = [f"CALIB {cal.get('phase', ''):>4} {cal.get('remaining', 0):.0f}s",
                 str(cal.get("msg", ""))[:24]]
    else:
        lines = [f"TELEOP   {hz:4.0f} Hz"]
    for s in SIDES:
        h = frame.hands.get(s) if frame else None
        tag = s[0].upper()
        if h is not None and h.tracked:
            roll, _, _ = _rpy_deg(np.asarray(h.wrist, float)[:3, :3])
            eng = "ENG" if engaged.get(s) else "off"
            lines.append(f"{tag} TRK roll{roll:+4.0f}  {eng}")
        else:
            lines.append(f"{tag} LOST")
    return lines


# Starting view: stand BEHIND the robot (camera on -y) and ABOVE it, looking
# forward (+y) and down over its shoulders — the natural teleoperator vantage, and
# +x (robot's right arm) lands on screen-right so left/right match the operator.
_CAM_BEHIND_ABOVE = dict(azimuth=180, elevation=-40, distance=2.0,
                         lookat=(0.0, 0.05, 0.95))


def _set_start_camera(cam) -> None:
    cam.azimuth = _CAM_BEHIND_ABOVE["azimuth"]
    cam.elevation = _CAM_BEHIND_ABOVE["elevation"]
    cam.distance = _CAM_BEHIND_ABOVE["distance"]
    cam.lookat[:] = _CAM_BEHIND_ABOVE["lookat"]


def _place_window_left(width_frac: float = 0.5) -> None:
    """macOS: dock the MuJoCo viewer window to the LEFT of the current display,
    instead of letting it open centered / on a separate fullscreen Space ("home").
    Best-effort and async: the window doesn't exist yet when the viewer launches,
    so a daemon thread polls for it via System Events. Needs Accessibility
    permission for the terminal/IDE running this; if that's not granted it just
    leaves the window where it is and prints a one-line hint."""
    import platform
    if platform.system() != "Darwin":
        return
    # Find the window titled "MuJoCo : ..." across all processes and pin it to the
    # left column of the main display. Returns "ok" once moved, "none" until then.
    script = """
    tell application "Finder" to set b to bounds of window of desktop
    set winW to ((item 3 of b) * %f) as integer
    set winH to (item 4 of b) - 25
    set moved to "none"
    tell application "System Events"
        repeat with proc in (every application process)
            try
                repeat with w in (every window of proc)
                    if name of w contains "MuJoCo" then
                        set position of w to {0, 25}
                        set size of w to {winW, winH}
                        set moved to "ok"
                    end if
                end repeat
            end try
        end repeat
    end tell
    return moved
    """ % width_frac

    def worker():
        for _ in range(16):
            time.sleep(0.5)
            try:
                out = subprocess.run(["osascript", "-e", script],
                                     capture_output=True, text=True, timeout=5)
            except (FileNotFoundError, subprocess.TimeoutExpired):
                return
            if out.stdout.strip() == "ok":
                return
            err = (out.stderr or "").lower()
            if "not allowed" in err or "-1743" in err or "assistive" in err:
                print("(left-dock skipped: grant Accessibility to your terminal/IDE in "
                      "System Settings > Privacy & Security > Accessibility)", flush=True)
                return
    threading.Thread(target=worker, daemon=True).start()


def _make_key_cb(engine):
    """Live tuning while teleoperating (no terminal needed for the feel — you adjust
    by hand and watch the robot). Keys in the MuJoCo window:
      ] / [   reach scale up / down   (robot arms bigger -> scale your motion up)
      p / o   elbow floor up / down   (p = keep the elbow more BENT / kill overextension;
                                       o = allow it straighter / more reach)
    Re-anchors each mapper so scale changes don't jump."""
    def cb(keycode):
        ch = chr(keycode) if 0 <= keycode < 0x110000 else ""
        if ch in ("o", "p"):                  # live ELBOW floor
            de = 0.1 if ch == "p" else -0.1
            v = 0.0
            for s in SIDES:
                ik = engine.arm[s].ik
                v = ik.set_elbow_min(ik.soft_lo[ik.ELBOW] + de)
            print(f"[tune] elbow floor (j3 min) -> {v:.2f} rad "
                  f"(higher = more bent, less overextension)", flush=True)
            return
        if ch in ("]", "=", "+"):
            d = 0.1
        elif ch in ("[", "-", "_"):
            d = -0.1
        else:
            return
        for s in SIDES:
            m = engine.arm[s].mapper
            m.scale = float(min(3.0, max(0.2, m.scale + d)))
            m.release()                       # re-anchor on next tick -> no jump
        print(f"[tune] reach scale -> {engine.arm['left'].mapper.scale:.2f}", flush=True)
    return cb


def run_viewer(args) -> int:
    import mujoco.viewer
    rig = load_rig()
    if args.vr:
        rig["vr"]["transport"] = args.vr
    if args.debug:
        rig["vr"]["debug"] = True
    if (args.vr == "vuer") or args.tunnel or rig["vr"].get("transport") == "vuer":
        _free_port(8012)                       # clear any stuck server from a prior run
    if args.tunnel or args.http:
        rig["vr"]["transport"] = "vuer"
        rig["vr"]["tunnel"] = True   # serve plain HTTP; an https tunnel fronts it
    world = SimWorld(rig)
    engine = TeleopEngine(rig, world)
    supervisor = Supervisor(rig, AlwaysOn())
    src = make_source(rig)
    src.start()
    tunnel = None
    if args.http:
        print("\n" + "=" * 64 + "\n  Serving HTTP on :8012. Open your BOOKMARKED cloudflared URL on\n"
              "  the Quest (start it once with ./scripts/vr_tunnel.sh and keep it\n"
              "  running — its URL stays fixed across sim restarts).\n" + "=" * 64 + "\n", flush=True)
    elif args.tunnel:
        _wait_port(8012)            # server must be listening BEFORE cloudflared (else 502)
        tunnel = _start_tunnel()
    elif (args.vr == "vuer") or rig["vr"].get("transport") == "vuer":
        _print_lan_url()
    push_viz = hasattr(src, "set_robot_frame")    # in-headset frame visualization (Vuer)
    push_calib = hasattr(src, "set_calib")        # in-headset calibration countdown (Vuer)
    push_hud = hasattr(src, "set_hud")            # in-headset live status/log panel (Vuer)
    from ..logging_utils import RateMeter
    rate = RateMeter()
    _t_prev = None
    try:
        _place_window_left()                       # dock the window to the left of the screen (macOS)
        with mujoco.viewer.launch_passive(world.model, world.data,
                                          key_callback=_make_key_cb(engine)) as v:
            _set_start_camera(v.cam)               # start behind + above the robot, looking forward
            print("\n  ON-SCREEN MAPPING VIZ: at each robot hand, the SOLID RGB triad is\n"
                  "  the robot's actual hand frame, the FAINT longer triad is what your\n"
                  "  hand is COMMANDING. Twist your wrist — they should spin together about\n"
                  "  the blue axis. If they diverge, that's the mapping bug, now on screen.\n", flush=True)
            while v.is_running():
                t = time.monotonic()   # one clock shared with the source stamps + supervisor
                if _t_prev is not None:
                    rate.update(t - _t_prev)
                _t_prev = t
                frame = src.latest()
                eng = supervisor.update(frame, t)
                engine.tick(frame, eng, t)
                if push_calib:
                    src.set_calib(engine.calib_status)
                if push_hud:
                    src.set_hud(_hud_lines(engine, frame, eng, rate.hz))
                if push_viz and frame is not None:
                    _push_robot_frames(src, engine, frame)
                if getattr(v, "user_scn", None) is not None:
                    _draw_frames(v.user_scn, engine, frame, eng, not engine.calibrated)
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
    ap.add_argument("--vr", choices=["fake", "vuer", "orbit"], help="override vr.transport")
    ap.add_argument("--tunnel", action="store_true",
                    help="serve over a public cloudflared HTTPS URL (spawns a NEW random URL each run)")
    ap.add_argument("--http", action="store_true",
                    help="serve plain HTTP for a PERSISTENT external tunnel (stable bookmarked URL) — "
                         "run ./scripts/vr_tunnel.sh once and keep it open")
    ap.add_argument("--debug", action="store_true", help="print incoming Vuer HAND_MOVE events")
    ap.add_argument("--gif", metavar="PATH", help="headless: render a GIF and exit")
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--fps", type=float, default=60.0)
    ap.add_argument("--azimuth", type=float, default=70)
    ap.add_argument("--elevation", type=float, default=-18)
    args = ap.parse_args()
    return run_gif(args) if args.gif else run_viewer(args)


if __name__ == "__main__":
    raise SystemExit(main())
