"""Head-anchored ABSOLUTE pose teleop (position + orientation) with a VISUAL
calibration routine.

Mapping (you track the WRIST: position + orientation, so we drive the EE to both):
  POSITION   p_ee_base = neck + scale * ( A · (p_wrist_op - p_head_op) )   (absolute, no drift)
  ORIENTATION R_ee_base = A · (flip?) R_wrist_op (flip?) · B                (hand follows wrist)
  A = r_base_from_vr(base_quat, tweak);  B fixed at calibration.

CALIBRATION ROUTINE (visual — countdown shows BIG in the browser viz):
  Put both hands DOWN AT YOUR SIDES (matching the robot's resting pose) and hold
  still. A 5s countdown runs in the viz; at 0 that stance becomes the robot's rest.
  Trigger: auto on first tracking · PINCH BOTH HANDS (thumb+index) · or press C.
  The robot holds its rest pose during the countdown so you can match it.

  uv run mjpython -m bimanual_teleop.tools.head_map --vr orbit

Keys: C recalibrate · V flip wrist-turn · -/= scale · I/K neck · J/L yaw · U/O roll · N/M pitch · 0 reset
"""
from __future__ import annotations

import argparse
import math
import time

import numpy as np

from ..config import load_rig, SIDES
from ..sim.sim_world import SimWorld
from ..arms.ik import ArmIK
from ..vr.frames import r_base_from_vr, quat_to_R
import mink

CAL_DURATION = 5.0          # countdown seconds
CAL_CAPTURE = 1.5           # average the pose over the last N seconds
PINCH_ON = 0.7              # both-hand pinch strength to arm the trigger
PINCH_HOLD = 0.5            # seconds of held pinch to fire
FZ = np.diag([1.0, 1.0, -1.0])   # forward-axis chirality flip for the wrist-turn toggle


def _ortho(R: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(R)
    O = U @ Vt
    if np.linalg.det(O) < 0:
        U[:, -1] *= -1
        O = U @ Vt
    return O


def _avg_R(mats) -> np.ndarray:
    return _ortho(np.mean(np.stack(mats), axis=0))


def _open_and_tile(url: str) -> None:
    """Open the hand viz in a browser, tiled LEFT, MuJoCo RIGHT (macOS best-effort).
    Inlined here so head_map doesn't import mapping_studio (which pulls in engine.py)."""
    import platform
    import subprocess
    import threading
    import time as _t
    if platform.system() != "Darwin":
        subprocess.Popen(["xdg-open", url], stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
        return
    browser = None
    for b in ("Google Chrome", "Safari"):
        if subprocess.run(["osascript", "-e", f'id of application "{b}"'],
                          capture_output=True).returncode == 0:
            browser = b
            break
    subprocess.Popen(["open", "-a", browser, url] if browser else ["open", url])

    def tile():
        _t.sleep(2.5)
        script = f'''
        tell application "Finder" to set sb to bounds of window of desktop
        set sw to item 3 of sb
        set sh to item 4 of sb
        try
          tell application "System Events" to tell process "{browser or 'Safari'}"
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vr", choices=["orbit", "fake"], default="orbit")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--j2-margin", type=float, default=0.5,
                    help="soft limit for j2 = home ± this (rad); smaller = less overextension")
    args = ap.parse_args()

    import mujoco
    import mujoco.viewer
    from ..vr.ingest import make_source

    rig = load_rig()
    rig["vr"]["transport"] = args.vr
    # Tighten j2's soft IK limit so the shoulder can't overextend.
    m = list(rig["ik"].get("soft_margin", [1.4, 1.0, 1.4, 1.4, 1.5, 1.7]))
    m[1] = float(args.j2_margin)
    rig["ik"]["soft_margin"] = m

    world = SimWorld(rig)
    ik = {s: ArmIK(rig, s) for s in SIDES}
    for s in SIDES:
        ik[s].reset()

    base_R = {s: quat_to_R(rig["arms"][s]["base_quat"]) for s in SIDES}
    base_pos = {s: np.asarray(rig["arms"][s]["base_pos"], float) for s in SIDES}
    neutral = {s: np.asarray(rig["arms"][s]["neutral_q"], float) for s in SIDES}
    rest_wrist = {s: ik[s].fk_wrist().translation().copy() for s in SIDES}
    rest_ee_R = {s: ik[s].fk_ee().rotation().as_matrix().copy() for s in SIDES}
    ws_min = np.asarray(rig["safety"]["workspace"]["min"], float)
    ws_max = np.asarray(rig["safety"]["workspace"]["max"], float)
    gap = float(rig.get("vr", {}).get("cross_gap", 0.05))

    st = {"scale": float(args.scale), "tweak": np.zeros(3), "neck_dz": 0.0, "oflip": False}
    st["A"] = {s: _ortho(r_base_from_vr(rig["arms"][s]["base_quat"], (0, 0, 0))) for s in SIDES}
    cal = {s: {"v0": np.array([0.0, -0.5, 0.0]), "Rw0": np.eye(3)} for s in SIDES}

    def neck(s):
        return rest_wrist[s] - st["scale"] * (st["A"][s] @ cal[s]["v0"])

    def Bmat(s):
        return cal[s]["Rw0"].T @ st["A"][s].T @ rest_ee_R[s]

    def rebuild_A():
        for s in SIDES:
            st["A"][s] = _ortho(r_base_from_vr(rig["arms"][s]["base_quat"], tuple(st["tweak"])))

    src = make_source(rig)
    src.start()
    if getattr(src, "viz_url", None):
        _open_and_tile(src.viz_url)

    def op_pose(frame):
        """{side: (v_head2wrist, R_wrist, pinch)} for tracked hands, + head present."""
        if frame is None or frame.head is None:
            return {}
        h = np.asarray(frame.head, float).reshape(4, 4)[:3, 3]
        out = {}
        for s in SIDES:
            hs = frame.hands.get(s)
            if hs is not None and hs.tracked:
                M = np.asarray(hs.wrist, float).reshape(4, 4)
                Rw = _ortho(M[:3, :3])
                if st["oflip"]:
                    Rw = _ortho(FZ @ Rw @ FZ)
                out[s] = (M[:3, 3] - h, Rw, float(getattr(hs, "pinch", 0.0)))
        return out

    # ---- calibration state machine -------------------------------------- #
    cs = {"phase": "idle", "t0": 0.0, "until": 0.0, "ever": False,
          "samp": {s: {"v": [], "R": []} for s in SIDES}, "pinch_t0": 0.0, "ckey": False}

    def start_countdown():
        cs["phase"] = "count"; cs["t0"] = time.monotonic()
        cs["samp"] = {s: {"v": [], "R": []} for s in SIDES}
        print("[head_map] calibration: HOLD hands at your sides...", flush=True)

    def finish(poses):
        ok = []
        for s in SIDES:
            if cs["samp"][s]["v"]:
                cal[s] = {"v0": np.mean(np.stack(cs["samp"][s]["v"]), axis=0),
                          "Rw0": _avg_R(cs["samp"][s]["R"])}
                ok.append(s)
        cs["ever"] = True
        cs["phase"] = "done"; cs["until"] = time.monotonic() + 2.0
        print(f"[head_map] CALIBRATED ({ok}) — current stance = robot rest.", flush=True)

    def tick_cal(poses, now):
        """Drive the calibration FSM; return True if it owns the robot this frame."""
        both = all(s in poses for s in SIDES)
        # triggers (idle only)
        if cs["phase"] == "idle":
            pinch_both = both and all(poses[s][2] > PINCH_ON for s in SIDES)
            cs["pinch_t0"] = cs["pinch_t0"] or (now if pinch_both else 0.0)
            if not pinch_both:
                cs["pinch_t0"] = 0.0
            fire = cs["ckey"] or (not cs["ever"] and both) or \
                   (cs["pinch_t0"] and now - cs["pinch_t0"] > PINCH_HOLD)
            cs["ckey"] = False
            if fire:
                cs["pinch_t0"] = 0.0
                start_countdown()
            else:
                src.overlay = {} if cs["ever"] else {
                    "text": "PINCH BOTH HANDS (or press C) to calibrate", "color": "amber"}
                return False
        if cs["phase"] == "count":
            remaining = CAL_DURATION - (now - cs["t0"])
            if not both:                                  # lost tracking -> restart hold
                src.overlay = {"text": "HOLD BOTH HANDS AT YOUR SIDES", "color": "red"}
                cs["t0"] = now
                return True
            if remaining <= CAL_CAPTURE:
                for s in SIDES:
                    cs["samp"][s]["v"].append(poses[s][0]); cs["samp"][s]["R"].append(poses[s][1])
            src.overlay = {"text": "HOLD AT YOUR SIDES — calibrating",
                           "count": max(0, math.ceil(remaining)), "color": "amber"}
            if remaining <= 0:
                finish(poses)
            return True                                   # robot frozen at rest during count
        if cs["phase"] == "done":
            src.overlay = {"text": "CALIBRATED ✓", "color": "green"}
            if now > cs["until"]:
                cs["phase"] = "idle"; src.overlay = {}
            return True
        return False

    def show():
        print(f"[head_map] scale={st['scale']:.2f} tweak(deg)={np.degrees(st['tweak']).round(1)} "
              f"neck_dz={st['neck_dz']:+.2f} oflip={st['oflip']}", flush=True)

    def on_key(code):
        ch = chr(code) if 0 <= code < 0x110000 else ""
        d = np.radians(5)
        if ch == "C":
            cs["ckey"] = True
        elif ch == "V":
            st["oflip"] = not st["oflip"]; print(f"[head_map] wrist-turn flip = {st['oflip']}", flush=True)
        elif ch == "=":
            st["scale"] *= 1.1; show()
        elif ch == "-":
            st["scale"] /= 1.1; show()
        elif ch == "I":
            st["neck_dz"] += 0.03; show()
        elif ch == "K":
            st["neck_dz"] -= 0.03; show()
        elif ch in "JL":
            st["tweak"][1] += d if ch == "L" else -d; rebuild_A(); show()
        elif ch in "UO":
            st["tweak"][2] += d if ch == "O" else -d; rebuild_A(); show()
        elif ch in "NM":
            st["tweak"][0] += d if ch == "M" else -d; rebuild_A(); show()
        elif ch == "0":
            st["scale"] = float(args.scale); st["tweak"][:] = 0; st["neck_dz"] = 0; rebuild_A(); show()
        elif ch == "P":
            show()

    def q_now(s):
        mm = world.model
        return np.array([world.data.qpos[int(mm.jnt_qposadr[
            mujoco.mj_name2id(mm, mujoco.mjtObj.mjOBJ_JOINT, f"{s}_arm_j{i}")])] for i in range(1, 7)])

    print("\n" + "=" * 72 + "\n  HEAD-ANCHORED TELEOP — position + orientation, with visual calibration\n"
          "  Hold hands AT YOUR SIDES (match the robot) for the countdown in the viz.\n"
          "  C recal · V flip turn · -/= scale · I/K neck · J/L yaw · U/O roll · N/M pitch · 0 reset\n"
          + "=" * 72 + "\n", flush=True)

    with mujoco.viewer.launch_passive(world.model, world.data, key_callback=on_key) as v:
        v.cam.lookat[:] = [-0.1, 0.0, 0.85]; v.cam.distance = 2.6
        v.cam.azimuth = 150; v.cam.elevation = -12
        while v.is_running():
            now = time.monotonic()
            poses = op_pose(src.latest())
            if tick_cal(poses, now):                      # calibration owns the robot -> hold rest
                for s in SIDES:
                    world.set_arm(s, neutral[s])
            else:
                for s in SIDES:
                    if s not in poses:
                        continue
                    v_op, R_op, _ = poses[s]
                    p = neck(s) + st["scale"] * (st["A"][s] @ v_op)
                    p[2] += st["neck_dz"]
                    p = np.clip(p, ws_min, ws_max)
                    pw = base_R[s] @ p + base_pos[s]
                    pw[1] = (min(pw[1], -gap) if s == "left" else max(pw[1], gap))
                    p = base_R[s].T @ (pw - base_pos[s])
                    R_ee = _ortho(st["A"][s] @ R_op @ Bmat(s))
                    target = mink.SE3.from_rotation_and_translation(mink.SO3.from_matrix(R_ee), p)
                    ik[s].seed(q_now(s))
                    world.set_arm(s, ik[s].solve(target))
            world.step(2)
            v.sync()
            time.sleep(1 / 120)
    src.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
