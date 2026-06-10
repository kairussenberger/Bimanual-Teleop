#!/usr/bin/env python
"""Headless Quest ingest diagnostic (spec bring-up step 4: "Add Quest input; log +
visualize the raw hand triads; confirm tracking flags and stale detection").

Starts the live VR source and prints, every tick, the incoming head/hand poses,
per-hand TRACKED/STALE flags, pose age, pinch, and the message sample-rate — so the
operator stream can be verified from the terminal without a renderer. Twist
your wrist and watch the quaternion change: that's the exact signal the orientation
mapping consumes. Exits early once both hands stream a stable tracked pose.

    uv run python scripts/check_quest.py                 # orbit (native app, USB/adb)
    uv run python scripts/check_quest.py --seconds 60
    uv run python scripts/check_quest.py --vr vuer       # WebXR browser source
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from bimanual_teleop.config import SIDES, load_rig          # noqa: E402
from bimanual_teleop.vr.calibrate import body_relative_hand_sample  # noqa: E402
from bimanual_teleop.vr.frames import R_to_quat             # noqa: E402
from bimanual_teleop.vr.ingest import make_source           # noqa: E402


def _head_ok(head) -> bool:
    if head is None:
        return False
    try:
        return not np.allclose(np.asarray(head, float).reshape(4, 4), np.eye(4))
    except (TypeError, ValueError):
        return False


def _fmt_hand(h, head=None, torso_from_head=(0.0, -0.35, 0.0)) -> str:
    if h is None or not h.tracked:
        return "STALE/LOST"
    p = np.asarray(h.wrist, float)[:3, 3]
    q = R_to_quat(np.asarray(h.wrist, float)[:3, :3])
    nlm = 0 if h.landmarks is None else len(h.landmarks)
    body = ""
    if _head_ok(head):
        rel = body_relative_hand_sample(h, np.asarray(head, float).reshape(4, 4), torso_from_head)
        wb = rel.wrist[:3, 3]
        body = f" body=[{wb[0]:+.3f} {wb[1]:+.3f} {wb[2]:+.3f}]"
    else:
        body = " body=NO_HEAD"
    return (f"TRACKED pos=[{p[0]:+.3f} {p[1]:+.3f} {p[2]:+.3f}] "
            f"quat=[{q[0]:+.2f} {q[1]:+.2f} {q[2]:+.2f} {q[3]:+.2f}]{body} lm={nlm} pinch={h.pinch:.2f}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vr", choices=["orbit", "vuer"], default="orbit")
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--rate", type=float, default=4.0, help="print rate (Hz)")
    ap.add_argument("--no-early-exit", action="store_true", help="run the full duration")
    args = ap.parse_args()

    rig = load_rig()
    rig["vr"]["transport"] = args.vr
    rig["vr"]["debug"] = True
    torso_from_head = rig.get("vr", {}).get("torso_from_head", [0.0, -0.35, 0.0])
    src = make_source(rig)
    print(f"[check_quest] starting '{args.vr}' source for {args.seconds:.0f}s "
          f"(Ctrl-C to stop)\n")
    src.start()

    t0 = time.monotonic()
    last_print = 0.0
    stable = 0                       # consecutive prints with BOTH hands tracked
    ever = {s: False for s in SIDES}
    head_seen = False
    try:
        while time.monotonic() - t0 < args.seconds:
            now = time.monotonic()
            if now - last_print < 1.0 / args.rate:
                time.sleep(0.005)
                continue
            last_print = now
            f = src.latest()
            if f is None:
                print(f"  t={now - t0:5.1f}s  (no frame yet)")
                continue
            head_ok = _head_ok(f.head)
            head_seen = head_seen or head_ok
            both = True
            line = []
            for s in SIDES:
                h = f.hands.get(s)
                tracked = bool(h and h.tracked)
                ever[s] = ever[s] or tracked
                both = both and tracked
                line.append(f"{s:>5}: {_fmt_hand(h, f.head, torso_from_head)}")
            counts = getattr(src, "counts", None)
            ctag = f" | msgs {counts}" if counts else ""
            print(f"  t={now - t0:5.1f}s head={'ok' if head_ok else '--'}{ctag}\n"
                  f"        " + "\n        ".join(line))
            stable = stable + 1 if both else 0
            if stable >= 3 and not args.no_early_exit:
                print("\n[check_quest] BOTH hands streaming a stable tracked pose ✓")
                break
    except KeyboardInterrupt:
        print("\n[check_quest] interrupted")
    finally:
        src.stop()

    print("\n  ---- summary ----")
    print(f"  head pose received:    {'YES' if head_seen else 'no'}")
    for s in SIDES:
        print(f"  {s:>5} hand tracked:    {'YES' if ever[s] else 'no'}")
    counts = getattr(src, "counts", None)
    if counts:
        dt = max(time.monotonic() - t0, 1e-6)
        print(f"  msg counts: {counts}  (~{counts.get('wrist', 0) / dt:.0f} wrist msg/s)")
    ok = head_seen and all(ever.values())
    print(f"\n  {'QUEST STREAM OK ✓' if ok else 'NO COMPLETE STREAM — see checklist below'}")
    if not ok:
        print("  checklist: (1) ORBIT app open on the Quest, (2) headset WORN, "
              "(3) BOTH controllers set down (hand-tracking only), (4) hands in view.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
