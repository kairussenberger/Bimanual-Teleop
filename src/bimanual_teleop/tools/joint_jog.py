"""Keyboard joint-jog for ONE YAM arm — drive each of the 6 joints directly, no IK,
no VR, no operator. Use it to learn the arm's kinematics, find poses, feel the joint
limits, and sanity-check the model (e.g. that the j6 wrist-roll spins the hand the
way you expect). The other arm just holds its home pose.

HOLD to move (release to stop). The MuJoCo viewer only reports key *presses* (no
release event), so we use the OS key-repeat: while you hold a key it keeps firing
and the joint keeps moving; let go and the repeats stop, so the joint stops ~0.15 s
later. If your OS doesn't auto-repeat into the window, each tap is a small ~1° nudge
instead — so tapping still works for fine adjustments.

Keys (default: RIGHT arm):
  +dir   Q W E R T Y  = j1 j2 j3 j4 j5 j6
  -dir   A S D F G H   (the row directly below)
  X stop all    0 reset to home    P print joint angles

    uv run mjpython -m bimanual_teleop.tools.joint_jog                 # right arm, slow
    uv run mjpython -m bimanual_teleop.tools.joint_jog --rate 0.3      # faster
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from ..config import load_rig
from ..sim.sim_world import SimWorld

INC = "QWERTY"      # hold to move joint j1..j6 in +
DEC = "ASDFGH"      # hold to move joint j1..j6 in −
HOLD_TIMEOUT = 0.15  # s after the last key event a joint keeps moving (bridges key-repeat)


def integrate(q, vel, rate: float, dt: float, lo, hi) -> None:
    """Advance moving joints by rate·dt, clamping to limits and stopping (vel→0) any
    joint that reaches a limit. Mutates q and vel in place."""
    for i in range(len(q)):
        if vel[i]:
            q[i] += vel[i] * rate * dt
            if q[i] <= lo[i]:
                q[i] = lo[i]; vel[i] = 0.0
            elif q[i] >= hi[i]:
                q[i] = hi[i]; vel[i] = 0.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--side", choices=["left", "right"], default="right")
    ap.add_argument("--rate", type=float, default=0.12, help="joint speed while held (rad/s, ~7°/s)")
    args = ap.parse_args()

    import mujoco.viewer
    rig = load_rig()
    world = SimWorld(rig)
    side = args.side
    q = np.asarray(rig["arms"][side]["neutral_q"], dtype=float).copy()
    vel = np.zeros(6)              # current direction per joint: -1 / 0 / +1
    deadline = np.zeros(6)         # monotonic time until which each joint stays "held"
    lo = np.asarray(rig["arms"]["joint_limits"]["lower"], dtype=float)
    hi = np.asarray(rig["arms"]["joint_limits"]["upper"], dtype=float)

    def show():
        print(f"[jog] {side} q(deg) = "
              + "  ".join(f"j{i+1}={np.degrees(q[i]):+6.1f}" for i in range(6)), flush=True)

    def on_key(code):
        ch = chr(code) if 0 <= code < 0x110000 else ""
        now = time.monotonic()
        if ch in INC:
            i = INC.index(ch); vel[i] = +1.0; deadline[i] = now + HOLD_TIMEOUT
        elif ch in DEC:
            i = DEC.index(ch); vel[i] = -1.0; deadline[i] = now + HOLD_TIMEOUT
        elif ch == "X":
            vel[:] = 0.0; show()
        elif ch == "0":
            q[:] = rig["arms"][side]["neutral_q"]; vel[:] = 0.0; show()
        elif ch == "P":
            show()

    print("\n" + "=" * 64 + f"\n  JOINT JOG — {side} arm  (no IK; HOLD a key to move, release to stop)\n"
          "  +dir   Q W E R T Y  = j1 j2 j3 j4 j5 j6\n"
          "  -dir   A S D F G H\n"
          "  X stop all    0 reset home    P print angles\n" + "=" * 64 + "\n", flush=True)
    show()
    with mujoco.viewer.launch_passive(world.model, world.data, key_callback=on_key) as v:
        v.cam.lookat[:] = [-0.1, 0.0, 0.7]; v.cam.distance = 1.8
        v.cam.azimuth = 140 if side == "right" else 40; v.cam.elevation = -12
        last = time.monotonic()
        while v.is_running():
            now = time.monotonic()
            dt = now - last; last = now
            vel[now > deadline] = 0.0          # let go (repeats stopped) → joint stops
            integrate(q, vel, args.rate, dt, lo, hi)
            world.set_arm(side, q)             # position actuators chase q; other arm holds home
            world.step(2)
            v.sync()
            time.sleep(1 / 120)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
