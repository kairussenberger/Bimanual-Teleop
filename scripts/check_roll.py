#!/usr/bin/env python
"""Live wrist-ROLL signal analyzer (spec bring-up step 5 prep): capture a real wrist
twist from the Quest and measure whether it is a CLEAN single-axis roll — the input
the orientation mapping must turn into a J6 move.

The synthetic harness already proved the IK realises a pure tool-axis roll on j6.
This checks the OTHER half on real hardware in the same body-relative frame used by
arm control: is the operator's pronation/supination arriving as a coherent roll
about one axis (good input), or is it noisy/multi-axis (which would scramble the
mapping regardless of R_align)? It also records the session (replayable) so the
mapping can be debugged offline.

    uv run python scripts/check_roll.py                      # 12 s, right hand
    uv run python scripts/check_roll.py --hand left --seconds 15 --save recordings/roll.npz
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from bimanual_teleop.config import load_rig                      # noqa: E402
from bimanual_teleop.vr.calibrate import head_op_axes            # noqa: E402
from bimanual_teleop.vr.frames import rotvec                     # noqa: E402
from bimanual_teleop.vr.ingest import make_source               # noqa: E402
from bimanual_teleop.vr.replay import SessionRecorder           # noqa: E402


def _head_ok(head) -> bool:
    if head is None:
        return False
    try:
        return not np.allclose(np.asarray(head, float).reshape(4, 4), np.eye(4))
    except (TypeError, ValueError):
        return False


def _body_relative_wrist_rotation(frame, hand: str) -> np.ndarray | None:
    """Wrist orientation expressed in the operator body frame.

    The roll diagnostic must not accept a tracked hand without a valid headset pose,
    because that would silently fall back to raw room/source-frame wrist rotation.
    """
    if frame is None or not _head_ok(frame.head):
        return None
    h = frame.hands.get(hand)
    if h is None or not h.tracked:
        return None
    H = np.asarray(frame.head, float).reshape(4, 4)
    W = np.asarray(h.wrist, float).reshape(4, 4)
    return head_op_axes(H).T @ W[:3, :3]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hand", choices=["left", "right"], default="right")
    ap.add_argument("--seconds", type=float, default=12.0)
    ap.add_argument("--wait", type=float, default=45.0, help="seconds to wait for tracking to start")
    ap.add_argument("--save", default=None, help="also save the raw session (.npz) for replay")
    args = ap.parse_args()

    rig = load_rig()
    rig["vr"]["transport"] = "orbit"
    src = make_source(rig)
    src.start()

    # Wait for the hand to actually start tracking, so donning time doesn't matter.
    print(f"[check_roll] put the headset ON, controllers DOWN, {args.hand} hand in view. "
          f"Waiting up to {args.wait:.0f}s for head + hand tracking...", flush=True)
    w0 = time.monotonic()
    while time.monotonic() - w0 < args.wait:
        f = src.latest()
        if _body_relative_wrist_rotation(f, args.hand) is not None:
            break
        time.sleep(0.05)
    else:
        print(f"  no valid head + {args.hand}-hand tracking within {args.wait:.0f}s — is the "
              f"headset ON your face with controllers down? (body-relative analysis needs it worn)")
        src.stop()
        return 1
    print(f"[check_roll] TRACKING — now TWIST your {args.hand.upper()} wrist back and forth "
          f"for {args.seconds:.0f}s (doorknob)...", flush=True)

    rec = SessionRecorder()
    Rs: list[np.ndarray] = []        # deduped wrist rotation matrices (operator body frame)
    t0 = time.monotonic()
    lastR = None
    n_tracked = 0
    try:
        while time.monotonic() - t0 < args.seconds:
            f = src.latest()
            if f is not None:
                rec.add(f, {"left": True, "right": True}, time.monotonic() - t0)
                R = _body_relative_wrist_rotation(f, args.hand)
                if R is not None:
                    n_tracked += 1
                    if lastR is None or not np.allclose(R, lastR, atol=1e-4):
                        Rs.append(R)
                        lastR = R
            time.sleep(1.0 / 120.0)
    except KeyboardInterrupt:
        pass
    finally:
        src.stop()

    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        rec.save(args.save)
        print(f"[check_roll] saved session -> {args.save} ({len(rec)} frames)\n")

    if len(Rs) < 5:
        print(f"  only {len(Rs)} distinct poses ({n_tracked} tracked reads) — was the "
              f"{args.hand} hand in view and moving? Try again.")
        return 1

    # Incremental body-frame deltas; rotvec(dR) = axis*angle of each step.
    V = np.array([rotvec(Rs[i - 1].T @ Rs[i]) for i in range(1, len(Rs))])
    mags = np.linalg.norm(V, axis=1)
    total_path = float(mags.sum())
    if total_path < 0.2:
        print(f"  barely any rotation detected (path {np.degrees(total_path):.0f}°). "
              f"Twist the wrist more.")
        return 1

    # Dominant rotation axis = principal eigenvector of sum(v vᵀ); fraction of the
    # angular "energy" along it tells us how single-axis (clean) the roll is.
    C = V.T @ V
    w, Q = np.linalg.eigh(C)
    axis = Q[:, -1]
    on_frac = float(w[-1] / w.sum())                 # 1.0 == perfectly single-axis
    # Net roll travelled about that axis (range of the cumulative projection).
    cum = np.cumsum(V @ axis)
    roll_range = float(cum.max() - cum.min())
    # Which wrist-LOCAL axis is this roll about? (forearm/pointing axis => pronation)
    R0 = Rs[len(Rs) // 2]
    local = np.abs(R0.T @ axis)
    local_axis = "xyz"[int(np.argmax(local))]

    print("  ---- wrist-roll analysis ----")
    print(f"  distinct poses analysed:     {len(Rs)} ({n_tracked} tracked reads)")
    print(f"  total angular path:          {np.degrees(total_path):6.0f}°")
    print(f"  single-axis fraction:        {on_frac:6.2f}   (1.00 = perfectly clean roll)")
    print(f"  roll range about that axis:  {np.degrees(roll_range):6.0f}°")
    print(f"  dominant axis (wrist-local): {local_axis} (|comp|={local.max():.2f})")
    clean = on_frac > 0.85 and roll_range > 0.6
    print(f"\n  {'CLEAN SINGLE-AXIS ROLL ✓ — good mapping input' if clean else 'NOISY / MULTI-AXIS — roll input itself is messy'}")
    if clean:
        print("  => the roll SIGNAL is good. If the robot still rolls wrong, replay the saved\n"
              "     session through the mapping scorer + local 3D viewer (no headset needed):\n"
              "       uv run python scripts/analyze_session.py <session.npz>\n"
              "       uv run python -m bimanual_teleop.launch.run_teleop --vr replay <session.npz> --viz")
    return 0 if clean else 1


if __name__ == "__main__":
    raise SystemExit(main())
