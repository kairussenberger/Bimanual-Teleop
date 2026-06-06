"""Mapping studio — SEE your hand and the robot arm side by side, live, and TUNE
the operator→robot mapping until they agree.

Why this exists: the teleop mapping is NOT joint-to-joint. The Quest gives only
your WRIST pose (+ 25 hand landmarks); ClutchMapper turns your wrist motion into a
Cartesian EE target and the two-stage IK reaches it. So the thing that's actually
hard to get right — and impossible to eyeball-guess — is FRAME ALIGNMENT: which
way "forward/up/right" point in the headset frame vs each arm's base frame
(`R_base_from_vr`), and how a wrist twist maps onto the wrist-roll joint. This tool
makes both visible.

In one MuJoCo scene:
  OPERATOR (floating panel, per hand)
    - 25-joint hand SKELETON (exactly what the Quest streams, world-aligned)
    - wrist TRIAD = the frame the mapping consumes  (X=red Y=green Z=blue)
    - YELLOW arrow = your wrist displacement since the clutch engaged
  ROBOT (at each hand)
    - SOLID triad   = the EE's ACTUAL orientation
    - FAINT triad   = the orientation you're COMMANDING
    - SOLID arrow   = the EE's ACTUAL displacement since engage
    - FAINT arrow   = the displacement your hand is ASKING for (nominal headset→world)
When the mapping is right, faint and solid agree on both sides; when you tweak the
frame they diverge — that divergence is the mapping bug, now on screen.

LIVE TUNING (keys, applied to both arms; nudges `mapping.r_base_from_vr_euler`):
  I / K  pitch (about headset X)      J / L  yaw (about Y)      U / O  roll (about Z)
  - / =  position scale               0  reset tweaks            P  print status

    uv run mjpython -m bimanual_teleop.tools.mapping_studio              # synthetic operator, no headset
    uv run mjpython -m bimanual_teleop.tools.mapping_studio --vr vuer    # your real Quest 3
    uv run python    -m bimanual_teleop.tools.mapping_studio --gif out.gif   # headless clip
"""
from __future__ import annotations

import argparse
import math
import time

import numpy as np

from ..config import SIDES, load_rig
from ..engine import TeleopEngine
from ..sim.sim_world import SimWorld
from ..vr.frames import (HandSample, VRFrame, WEBXR_TO_WORLD, euler_to_R,
                         r_base_from_vr)
from ..viz import overlay

# Where the operator hands float in the world (+X side, head height) so they sit
# beside the robot without overlapping it. Fingers point −X = toward the robot.
OP_ANCHOR = {"left": np.array([0.45, -0.42, 1.2]), "right": np.array([0.45, 0.42, 1.2])}
_OP_RGBA = {"left": (0.95, 0.85, 0.2, 1.0), "right": (0.3, 0.8, 0.95, 1.0)}
ARROW_GAIN = 1.6          # visual length per metre of displacement
ARROW_MAX = 0.40          # cap arrow length (m)
ARROW_MIN_SHOW = 0.012    # don't draw sub-cm jitter


# --------------------------------------------------------------------------- #
# Synthetic operator (no headset): a hand whose wrist pose AND landmarks are
# consistent, sweeping position + pitch/yaw/roll + a finger curl, so the mapping
# (incl. orientation) is exercised and legible on the desktop.
# --------------------------------------------------------------------------- #
def _canonical_hand() -> np.ndarray:
    """A flat right hand in WebXR world axes (x=right, y=up, −z=forward): wrist at
    origin, fingers forward (−z), palm down. 25 W3C joints."""
    w = np.zeros((25, 3))
    seg = 0.035
    for base, x in ((5, 0.02), (10, 0.0), (15, -0.02), (20, -0.04)):   # index/middle/ring/pinky
        w[base] = [x, 0.0, -0.02]
        for k in range(1, 5):
            w[base + k] = [x, 0.0, -0.02 - seg * k]
    w[1] = [0.035, 0.0, 0.0]                       # thumb metacarpal, off to +x
    for k in range(1, 4):
        w[1 + k] = [0.035 + 0.016 * k, 0.0, -0.016 * k]
    return w


_CANON = _canonical_hand()


def _synthetic_frame(t: float) -> VRFrame:
    """Both wrists sweep position + orientation; landmarks ride the wrist pose."""
    rx, ry, rz = 0.5 * math.sin(0.6 * t), 0.4 * math.sin(0.37 * t), 0.7 * math.sin(0.5 * t)
    R = euler_to_R([rx, ry, rz])
    pos = np.array([0.10 * math.sin(0.55 * t), 0.06 * math.sin(0.8 * t),
                    -0.12 + 0.07 * math.sin(0.45 * t)])
    curl = 0.5 - 0.5 * math.cos(0.9 * t)
    rel = (_CANON - _CANON[0]) * (1.0 - 0.5 * curl)   # crude curl = pull fingers in
    lm = pos + rel @ R.T                              # landmarks in WebXR world axes
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = pos
    hands = {s: HandSample(tracked=True, wrist=T.copy(), landmarks=lm.copy(), pinch=curl)
             for s in SIDES}
    return VRFrame(stamp=t, head=np.eye(4), hands=hands)


class ManualOperator:
    """Drive the operator wrist from the keyboard (no headset). Pose is in the WebXR
    frame (x=right, y=up, −z=forward); both hands share it, so both arms follow."""
    T_STEP = 0.02                  # metres per keypress
    R_STEP = math.radians(6.0)

    def __init__(self):
        self.pos = np.array([0.0, 0.0, -0.10])   # start a little forward
        self.rpy = np.zeros(3)
        self.curl = 0.0

    def key(self, ch: str) -> None:
        moves = {"W": (0, 2, -1), "S": (0, 2, +1),    # forward/back (−z/+z)
                 "A": (0, 0, -1), "D": (0, 0, +1),    # left/right
                 "Q": (0, 1, +1), "Z": (0, 1, -1),    # up/down
                 "T": (1, 0, +1), "G": (1, 0, -1),    # pitch
                 "F": (1, 1, +1), "H": (1, 1, -1),    # yaw
                 "R": (1, 2, +1), "Y": (1, 2, -1)}    # roll
        if ch in moves:
            which, i, sgn = moves[ch]
            (self.pos if which == 0 else self.rpy)[i] += sgn * (self.T_STEP if which == 0 else self.R_STEP)
        elif ch == "X":
            self.curl = 0.0 if self.curl > 0.5 else 1.0
        elif ch == "B":
            self.pos[:] = [0.0, 0.0, -0.10]; self.rpy[:] = 0.0

    def frame(self, t: float) -> VRFrame:
        R = euler_to_R(self.rpy)
        lm = self.pos + (_CANON - _CANON[0]) * (1.0 - 0.5 * self.curl) @ R.T
        T = np.eye(4); T[:3, :3] = R; T[:3, 3] = self.pos
        hands = {s: HandSample(tracked=True, wrist=T.copy(), landmarks=lm.copy(), pinch=self.curl)
                 for s in SIDES}
        return VRFrame(stamp=t, head=np.eye(4), hands=hands)


# --------------------------------------------------------------------------- #
# Drawing: append the operator panel + robot comparison into a scene.
# --------------------------------------------------------------------------- #
def _arrow_len(v: np.ndarray) -> float:
    return float(min(ARROW_MAX, np.linalg.norm(v) * ARROW_GAIN))


def _draw(scn, engine: TeleopEngine, frame: VRFrame | None) -> None:
    """Append operator + robot overlay geoms. Does NOT reset scn.ngeom (the caller
    does for the viewer; the offscreen renderer keeps the model geoms)."""
    if frame is None:
        return
    for s in SIDES:
        ac = engine.arm[s]
        hand = frame.hands.get(s)

        # ---- operator panel: skeleton + wrist frame + your displacement ----
        if hand is not None and hand.tracked and hand.landmarks is not None:
            anchor = OP_ANCHOR[s]
            lm = np.asarray(hand.landmarks, float).reshape(25, 3)
            pts = anchor + (lm - lm[0]) @ WEBXR_TO_WORLD.T        # wrist-centered, world-aligned
            overlay.skeleton(scn, pts, rgba=_OP_RGBA[s])
            wR = np.asarray(hand.wrist, float).reshape(4, 4)[:3, :3]
            overlay.triad(scn, anchor, WEBXR_TO_WORLD @ wR, length=0.13, width=0.008)
            if ac.mapper.anchor_ctrl is not None:
                dp_vr = np.asarray(hand.wrist, float).reshape(4, 4)[:3, 3] \
                    - ac.mapper.anchor_ctrl.translation()
                world_dp = WEBXR_TO_WORLD @ dp_vr
                if np.linalg.norm(world_dp) > ARROW_MIN_SHOW:
                    overlay.arrow(scn, anchor, world_dp, _arrow_len(world_dp), 0.012,
                                  (0.95, 0.85, 0.2, 1.0))

        # ---- robot: actual vs commanded orientation (EE site) + displacement ----
        # Orientation is read at the EE site; POSITION is driven at the WRIST site
        # (ik.py targets `{side}_wrist`), and the clutch position anchor IS the wrist
        # site (arm_control.py), so the displacement arrows live at the wrist — never
        # mix the two sites (they differ by the ~11 cm hand offset).
        ee = ac.ik.fk_ee()
        ee_w = ac.base_R @ ee.translation() + ac.base_pos
        overlay.triad(scn, ee_w, ac.base_R @ ee.rotation().as_matrix(), 0.12, 0.007, 1.0)
        if ac.cmd_R is not None:
            overlay.triad(scn, ee_w, ac.base_R @ ac.cmd_R, 0.17, 0.004, 0.4)
        if ac.mapper.anchor_ee is not None:
            anchor_w = ac.base_R @ ac.mapper.anchor_ee.translation() + ac.base_pos   # wrist anchor
            wrist_w = ac.base_R @ ac.ik.fk_wrist().translation() + ac.base_pos
            actual_dp = wrist_w - anchor_w                                # where the wrist went
            if np.linalg.norm(actual_dp) > ARROW_MIN_SHOW:
                overlay.arrow(scn, anchor_w, actual_dp, _arrow_len(actual_dp), 0.011,
                              (0.2, 0.9, 0.9, 1.0))                        # SOLID = actual
            if hand is not None and hand.tracked and ac.mapper.anchor_ctrl is not None:
                dp_vr = np.asarray(hand.wrist, float).reshape(4, 4)[:3, 3] \
                    - ac.mapper.anchor_ctrl.translation()
                asked = WEBXR_TO_WORLD @ dp_vr                            # nominal hand→world
                if np.linalg.norm(asked) > ARROW_MIN_SHOW:
                    overlay.arrow(scn, anchor_w, asked, _arrow_len(asked), 0.006,
                                  (0.95, 0.85, 0.2, 0.45))                # FAINT = asked-for


# --------------------------------------------------------------------------- #
# Live tuning: a tweak euler on R_base_from_vr + pos_scale, applied to both arms.
# --------------------------------------------------------------------------- #
class Tuner:
    STEP = math.radians(5.0)
    SCALE_STEP = 0.05

    def __init__(self, rig: dict, engine: TeleopEngine):
        self.rig = rig
        self.engine = engine
        self.tweak = np.zeros(3)                       # rx, ry, rz (headset frame, rad)
        self.scale = float(rig["mapping"]["pos_scale"])
        self._dirty = True

    def apply(self) -> None:
        if not self._dirty:
            return
        self._dirty = False
        for s in SIDES:
            ac = self.engine.arm[s]
            ac.mapper.set_R(r_base_from_vr(self.rig["arms"][s]["base_quat"], self.tweak))
            ac.mapper.scale = self.scale
        self.status()

    def status(self) -> None:
        d = np.degrees(self.tweak)
        print(f"[studio] tweak(deg) pitch={d[0]:+5.1f} yaw={d[1]:+5.1f} roll={d[2]:+5.1f} "
              f"| scale={self.scale:.2f}", flush=True)

    def key(self, code: int) -> None:
        self.handle(chr(code) if 0 <= code < 0x110000 else "")

    def handle(self, ch: str) -> None:
        bumps = {"I": (0, +1), "K": (0, -1), "J": (1, +1), "L": (1, -1),
                 "U": (2, +1), "O": (2, -1)}
        if ch in bumps:
            i, sgn = bumps[ch]; self.tweak[i] += sgn * self.STEP; self._dirty = True
        elif ch == "=":
            self.scale = min(2.0, self.scale + self.SCALE_STEP); self._dirty = True
        elif ch == "-":
            self.scale = max(0.1, self.scale - self.SCALE_STEP); self._dirty = True
        elif ch == "0":
            self.tweak[:] = 0.0; self.scale = float(self.rig["mapping"]["pos_scale"])
            self._dirty = True
        elif ch == "P":
            self.status()


_KEYS_HELP = ("  TUNE:  I/K pitch   J/L yaw   U/O roll   -/= scale   0 reset   P status\n"
              "  Operator panel = YOUR hand (skeleton + wrist triad + yellow motion arrow).\n"
              "  Robot: SOLID triad/arrow = actual, FAINT = commanded/asked-for. Tune until\n"
              "  faint and solid agree on both sides.")
_MANUAL_HELP = ("  MOVE HAND (no headset):  W/S fwd/back   A/D left/right   Q/Z up/down\n"
                "                           T/G pitch   F/H yaw   R/Y roll   X fist   B recenter")


def run_gif(args, rig: dict) -> int:
    import imageio.v3 as iio
    import mujoco
    world = SimWorld(rig)
    engine = TeleopEngine(rig, world)
    w, h = min(args.width, 1280), min(args.height, 960)   # model offscreen buffer cap
    cam = mujoco.MjvCamera(); mujoco.mjv_defaultCamera(cam)
    cam.lookat[:] = [-0.1, 0.0, 0.85]; cam.distance = 2.6
    cam.azimuth = args.azimuth; cam.elevation = args.elevation
    frames, n = [], int(args.seconds * args.fps)
    with mujoco.Renderer(world.model, height=h, width=w) as renderer:
        for i in range(n):
            t = i / args.fps
            frame = _synthetic_frame(t)
            engine.tick(frame, {s: True for s in SIDES}, t)
            world.step(2)
            mujoco.mj_forward(world.model, world.data)
            renderer.update_scene(world.data, cam)
            _draw(renderer.scene, engine, frame)
            if i % max(1, int(args.fps / 20)) == 0:
                frames.append(renderer.render())
    iio.imwrite(args.gif, frames, duration=1000 / 20, loop=0)
    print(f"wrote {args.gif} ({len(frames)} frames)")
    return 0


def _open_and_tile(url: str) -> None:
    """Open the hand viz in a browser and tile it LEFT, MuJoCo window RIGHT (macOS,
    best-effort). The MuJoCo window opens shortly after, so we tile in a delayed
    thread; failures are silently ignored (just leaves both windows openable)."""
    import platform
    import subprocess
    import threading
    import time as _t
    if platform.system() != "Darwin":
        subprocess.Popen(["open" if False else "xdg-open", url],
                         stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
        return
    browser = None
    for b in ("Google Chrome", "Safari"):
        if subprocess.run(["osascript", "-e", f'id of application "{b}"'],
                          capture_output=True).returncode == 0:
            browser = b
            break
    subprocess.Popen(["open", "-a", browser, url] if browser else ["open", url])

    def tile():
        _t.sleep(2.5)                                   # let both windows exist
        bproc = browser or "Safari"
        script = f'''
        tell application "Finder" to set sb to bounds of window of desktop
        set sw to item 3 of sb
        set sh to item 4 of sb
        try
          tell application "System Events" to tell process "{bproc}"
            set position of front window to {{0, 0}}
            set size of front window to {{sw / 2, sh}}
          end tell
        end try
        repeat with pn in {{"mjpython", "Python", "python3.12", "python3", "python"}}
          try
            tell application "System Events" to tell process (pn as string)
              set position of front window to {{sw / 2, 0}}
              set size of front window to {{sw / 2, sh}}
            end tell
          end try
        end repeat
        '''
        subprocess.run(["osascript", "-e", script], capture_output=True)
    threading.Thread(target=tile, daemon=True).start()


def run_viewer(args, rig: dict) -> int:
    import mujoco.viewer
    from ..vr.ingest import make_source
    world = SimWorld(rig)
    engine = TeleopEngine(rig, world)
    if getattr(args, "pos_only", False):           # debug position axes in isolation
        for s in SIDES:
            engine.arm[s].mapper.freeze_ori = True
        print("[studio] POSITION-ONLY mode: EE holds rest orientation; only translation maps.", flush=True)
    tuner = Tuner(rig, engine)
    manual = ManualOperator() if args.manual else None
    use_real = manual is None and (args.vr in ("vuer", "orbit") or args.tunnel or args.http
                                   or rig["vr"].get("transport") in ("vuer", "orbit"))
    src, tunnel = None, None
    if use_real:
        # Reuse run_sim's Quest-connection plumbing (cloudflared tunnel / LAN URL).
        from ..launch.run_sim import _free_port, _wait_port, _start_tunnel, _print_lan_url
        _free_port(8012)                       # clear any stuck server from a prior run
        src = make_source(rig)
        src.start()
        if args.http:
            print("\n" + "=" * 70 + "\n  Serving HTTP on :8012 — open your BOOKMARKED cloudflared URL\n"
                  "  on the Quest (start it once with ./scripts/vr_tunnel.sh).\n" + "=" * 70, flush=True)
        elif args.tunnel:
            _wait_port(8012)                   # server must listen BEFORE cloudflared (else 502)
            tunnel = _start_tunnel()
        elif rig["vr"].get("transport") == "vuer":
            _print_lan_url()                   # ORBIT prints its own banner in start()
        # ORBIT: open the live hand viz in the browser, tiled beside the MuJoCo window
        if getattr(src, "viz_url", None):
            _open_and_tile(src.viz_url)
    def on_key(code):
        ch = chr(code) if 0 <= code < 0x110000 else ""
        if manual is not None:
            manual.key(ch)
        tuner.handle(ch)

    header = (_MANUAL_HELP + "\n" + _KEYS_HELP) if manual is not None else _KEYS_HELP
    print("\n" + "=" * 70 + "\n  MAPPING STUDIO\n" + header + "\n" + "=" * 70 + "\n", flush=True)
    tuner.status()
    try:
        with mujoco.viewer.launch_passive(world.model, world.data,
                                          key_callback=on_key) as v:
            v.cam.lookat[:] = [-0.1, 0.0, 0.85]; v.cam.distance = 2.6
            v.cam.azimuth = args.azimuth; v.cam.elevation = args.elevation
            t0 = time.monotonic()
            while v.is_running():
                t = time.monotonic()
                frame = (manual.frame(t) if manual is not None
                         else src.latest() if src is not None
                         else _synthetic_frame(t - t0))
                tuner.apply()
                engine.tick(frame, {s: True for s in SIDES}, t)
                if getattr(v, "user_scn", None) is not None:
                    v.user_scn.ngeom = 0
                    _draw(v.user_scn, engine, frame)
                world.step(2)
                v.sync()
                time.sleep(1 / 120)
    finally:
        if src is not None:
            src.stop()
        if tunnel is not None:
            tunnel.terminate()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vr", choices=["fake", "vuer", "orbit"], help="real Quest: vuer (browser) or orbit (app); else synthetic")
    ap.add_argument("--tunnel", action="store_true",
                    help="connect a real Quest over a public cloudflared HTTPS URL (random each run)")
    ap.add_argument("--http", action="store_true",
                    help="serve plain HTTP for a PERSISTENT external tunnel (scripts/vr_tunnel.sh)")
    ap.add_argument("--debug", action="store_true", help="print incoming Vuer HAND_MOVE events")
    ap.add_argument("--manual", action="store_true",
                    help="drive the operator hand from the keyboard (no headset)")
    ap.add_argument("--calib", action="store_true", help="run the hold-stance calibration first")
    ap.add_argument("--pos-only", action="store_true",
                    help="freeze EE orientation; only wrist POSITION maps (debug the 3 translation axes first)")
    ap.add_argument("--gif", metavar="PATH", help="headless: render a synthetic clip and exit")
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--fps", type=float, default=60.0)
    ap.add_argument("--width", type=int, default=1100)
    ap.add_argument("--height", type=int, default=800)
    ap.add_argument("--azimuth", type=float, default=140.0)
    ap.add_argument("--elevation", type=float, default=-12.0)
    args = ap.parse_args()

    rig = load_rig()
    if args.vr:
        rig["vr"]["transport"] = args.vr
    if args.tunnel or args.http:
        rig["vr"]["transport"] = "vuer"
        rig["vr"]["tunnel"] = True             # serve HTTP; the external tunnel fronts HTTPS
    if args.debug:
        rig["vr"]["debug"] = True
    if not args.calib:
        rig["vr"]["calib_seconds"] = 0     # knobs own the frame unless you calibrate
    return run_gif(args, rig) if args.gif else run_viewer(args, rig)


if __name__ == "__main__":
    raise SystemExit(main())
