#!/usr/bin/env python
"""Offline session analyzer — replay a recorded Quest session (.npz) through the
REAL TeleopEngine and measure, frame by frame, whether the commanded robot motion
matches the operator's hand motion. No headset, no Unity, no robot.

The teleop contract (CLAUDE.md): rotate/translate your hand relative to your BODY
axes (right/up/forward) and the EE must rotate/translate about the corresponding
robot WORLD axes (+Y/+Z/−X) by the same amount. This script quantifies both:

  - ORIENTATION: angle between the world-frame rotation axis your hand actually
    moved about and the world-frame rotation axis the engine commanded the EE
    about, plus the angle-magnitude ratio (1.0 = same amount of rotation).
  - TRANSLATION: direction cosine + magnitude ratio between the body-frame hand
    displacement (mapped to world) and the commanded EE displacement.
  - IK TRACKING: orientation/position gap between the commanded target and the
    pose the IK actually achieved (isolates mapping bugs from solver bugs).

    uv run python scripts/analyze_session.py recordings/roll_right.npz
    uv run python scripts/analyze_session.py session.npz --side left --min-angle 10
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from bimanual_teleop.config import SIDES, load_rig                  # noqa: E402
from bimanual_teleop.engine import TeleopEngine                     # noqa: E402
from bimanual_teleop.vr.calibrate import W_AXES, head_op_axes       # noqa: E402
from bimanual_teleop.vr.frames import quat_to_R, rotvec             # noqa: E402
from bimanual_teleop.vr.replay import ReplaySource                  # noqa: E402


class NullSink:
    def set_arm(self, side, q):
        pass

    def set_hand(self, side, joints):
        pass


def _angle_between(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return float("nan")
    return float(np.degrees(np.arccos(np.clip(a @ b / (na * nb), -1.0, 1.0))))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", help="session .npz from run_teleop --record / check_roll --save")
    ap.add_argument("--side", choices=["left", "right"], default="right")
    ap.add_argument("--min-angle", type=float, default=15.0,
                    help="only score frames where the hand has rotated at least this many deg from anchor")
    ap.add_argument("--calib-seconds", type=float, default=None,
                    help="override vr.calib_seconds (default: rig value, i.e. what a live session does)")
    args = ap.parse_args()

    rig = load_rig()
    if args.calib_seconds is not None:
        rig["vr"]["calib_seconds"] = max(0.0, float(args.calib_seconds))

    src = ReplaySource(args.path)
    t = src.t
    side = args.side
    base_R = quat_to_R(rig["arms"][side]["base_quat"])              # arm base → world
    print(f"session: {args.path}")
    print(f"  frames={len(t)}  duration={src.duration:.1f}s  side={side}")
    head_ok = ~np.isnan(src.head).any(axis=(1, 2))
    lm_ok = ~np.isnan(src._side[side]["landmarks"]).all(axis=(1, 2))
    print(f"  head present {head_ok.mean()*100:.0f}%  | {side} tracked "
          f"{src._side[side]['tracked'].mean()*100:.0f}%  landmarks {lm_ok.mean()*100:.0f}%  "
          f"engaged {src.engaged_arr[:, SIDES.index(side)].mean()*100:.0f}%")

    engine = TeleopEngine(rig, NullSink())
    arm = engine.arm[side]

    # Anchor snapshot, re-captured whenever the mapper (re-)engages.
    anchor = None          # dict(raw_R, op_axes, ee_R, ee_p, q)
    prev_anchor_obj = None
    rows = []              # per-frame mapping scores

    for i in range(len(t)):
        frame = src.frame_at(t[i])
        engaged = src.engaged_at(t[i])
        engine.tick(frame, engaged, float(t[i]))

        m = arm.mapper
        hs = frame.hands.get(side)
        if frame.head is None or hs is None or not hs.tracked:
            continue
        if m.anchor_ctrl is not None and m.anchor_ctrl is not prev_anchor_obj:
            prev_anchor_obj = m.anchor_ctrl
            anchor = {
                "raw_R": np.asarray(hs.wrist, float)[:3, :3].copy(),
                "raw_p": np.asarray(hs.wrist, float)[:3, 3].copy(),
                "op_axes": head_op_axes(frame.head),
                "ee_R": m.anchor_ee.rotation().as_matrix(),
                "ee_p": m.anchor_ee.translation(),
                "t": float(t[i]),
                "q": arm.ik.q,
            }
        if anchor is None or not m.engaged or arm.cmd_R is None:
            continue

        # --- what the OPERATOR did, in robot-world terms ---------------------- #
        W = np.asarray(hs.wrist, float)
        D_xr = W[:3, :3] @ anchor["raw_R"].T                # hand rotation since anchor (WebXR world)
        rv = rotvec(D_xr)
        hand_ang = float(np.degrees(np.linalg.norm(rv)))
        Q = W_AXES @ anchor["op_axes"].T                    # WebXR world → robot world (proper)
        want_axis_w = Q @ (rv / (np.linalg.norm(rv) + 1e-12))

        # --- what the ENGINE commanded ---------------------------------------- #
        D_ee = arm.cmd_R @ anchor["ee_R"].T                 # commanded EE rotation since anchor (base)
        rv_ee = rotvec(D_ee)
        cmd_ang = float(np.degrees(np.linalg.norm(rv_ee)))
        cmd_axis_w = base_R @ (rv_ee / (np.linalg.norm(rv_ee) + 1e-12))

        # --- translation ------------------------------------------------------- #
        # The mapper's anchor_ctrl is the body-frame torso→wrist vector at engage;
        # recompute the current one the same way body_relative_hand_sample does.
        op_axes = head_op_axes(frame.head)
        torso_w = frame.head[:3, 3] + op_axes @ np.asarray(rig["vr"]["torso_from_head"], float)
        ctrl_p = op_axes.T @ (W[:3, 3] - torso_w)
        dp_body = ctrl_p - m.anchor_ctrl.translation()
        want_dp_w = W_AXES @ dp_body
        cmd_dp_w = base_R @ (arm.cmd_pos - anchor["ee_p"])

        # --- IK tracking (achieved vs commanded) ------------------------------- #
        ach_R = arm.ik.fk_ee().rotation().as_matrix()
        ik_ori_err = float(np.degrees(np.linalg.norm(rotvec(ach_R.T @ arm.cmd_R))))

        rows.append({
            "t": float(t[i] - t[0]),
            "hand_ang": hand_ang,
            "cmd_ang": cmd_ang,
            "axis_err": _angle_between(want_axis_w, cmd_axis_w),
            "want_axis": want_axis_w,
            "cmd_axis": cmd_axis_w,
            "dp_dir_err": _angle_between(want_dp_w, cmd_dp_w),
            "dp_ratio": float(np.linalg.norm(cmd_dp_w) / (np.linalg.norm(want_dp_w) + 1e-9)),
            "dp_norm": float(np.linalg.norm(want_dp_w)),
            "ik_ori_err": ik_ori_err,
            "q": arm.ik.q,
        })

    if not rows:
        print("\nno engaged+tracked frames with a command — nothing to score "
              "(did calibration ever finish? try --calib-seconds 0)")
        return 1

    sc = [r for r in rows if r["hand_ang"] >= args.min_angle]
    print(f"\nscored {len(sc)}/{len(rows)} engaged frames with hand rotation ≥ {args.min_angle:.0f}°")
    if sc:
        ax = np.array([r["axis_err"] for r in sc if np.isfinite(r["axis_err"])])
        ratio = np.array([r["cmd_ang"] / max(r["hand_ang"], 1e-9) for r in sc])
        print("\n---- ORIENTATION mapping (hand rotation → commanded EE rotation) ----")
        print(f"  world-axis error:    median {np.median(ax):6.1f}°   p90 {np.percentile(ax, 90):6.1f}°")
        print(f"  angle ratio cmd/hand: median {np.median(ratio):5.2f}    (1.00 = same magnitude)")
        mid = sc[len(sc) // 2]
        print(f"  example @t={mid['t']:.1f}s: hand {mid['hand_ang']:.0f}° about world "
              f"[{mid['want_axis'][0]:+.2f} {mid['want_axis'][1]:+.2f} {mid['want_axis'][2]:+.2f}]  "
              f"→ cmd {mid['cmd_ang']:.0f}° about [{mid['cmd_axis'][0]:+.2f} {mid['cmd_axis'][1]:+.2f} {mid['cmd_axis'][2]:+.2f}]")
        ok_ori = np.median(ax) < 15.0 and 0.8 < np.median(ratio) < 1.25
        print(f"  VERDICT: {'OK — EE rotates about the same world axis as your hand' if ok_ori else 'BROKEN — commanded rotation axis does not match the hand'}")
    else:
        ok_ori = True
        print("  (no frames exceeded --min-angle; orientation unscored)")

    moved = [r for r in rows if r["dp_norm"] > 0.05]
    print("\n---- TRANSLATION mapping (hand displacement → commanded EE displacement) ----")
    if moved:
        de = np.array([r["dp_dir_err"] for r in moved if np.isfinite(r["dp_dir_err"])])
        dr = np.array([r["dp_ratio"] for r in moved])
        print(f"  direction error:     median {np.median(de):6.1f}°   p90 {np.percentile(de, 90):6.1f}°")
        print(f"  magnitude ratio:     median {np.median(dr):5.2f}    (clamps/filters can shrink it)")
        ok_pos = np.median(de) < 20.0
        print(f"  VERDICT: {'OK' if ok_pos else 'BROKEN'}")
    else:
        ok_pos = True
        print("  hand never moved >5cm from anchor; translation unscored")

    ik = np.array([r["ik_ori_err"] for r in rows])
    print("\n---- IK tracking (achieved vs commanded orientation) ----")
    print(f"  median {np.median(ik):5.1f}°   p90 {np.percentile(ik, 90):5.1f}°"
          "   (large = solver/limits, NOT the mapping)")

    q = np.stack([r["q"] for r in rows])
    rng = np.degrees(q.max(axis=0) - q.min(axis=0))
    print("\n---- joint travel while engaged (deg) ----")
    print("  " + "  ".join(f"j{i+1}:{rng[i]:6.1f}" for i in range(6)))

    print(f"\nOVERALL: {'PASS' if (ok_ori and ok_pos) else 'FAIL — mapping does not honor the body↔world contract'}")
    return 0 if (ok_ori and ok_pos) else 1


if __name__ == "__main__":
    raise SystemExit(main())
