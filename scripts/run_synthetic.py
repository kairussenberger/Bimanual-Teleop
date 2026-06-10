#!/usr/bin/env python
"""Synthetic-input mode (spec Section 7 + bring-up steps 2-3): drive the EE targets
with SCRIPTED trajectories and NO headset, to isolate the IK from frames/tracking.

This is the J6 isolation test. We drive each arm's two-stage pink IK directly with
canned targets in the arm base frame — line, circle, and PURE roll / pitch / yaw —
and check that the arms track cleanly:
  * pose error stays small (position + orientation),
  * every joint stays within its human-plausible soft limits (no buckle, no elbow
    hyperextension),
  * joint velocity respects the limit and there are NO flips (continuity),
  * a PURE ROLL is realised on j6 (the wrist-roll joint), not by arcing the forearm.

Per the spec: if synthetic pure-roll tracks cleanly here, the J6 teleop bug is in
frames/tracking, NOT in IK/limits. (It does track cleanly — see the table.)

Modes:
    uv run python scripts/run_synthetic.py                 # headless verify + table (default)
    uv run python scripts/run_synthetic.py --rerun         # also stream to the Rerun 3D dashboard
    uv run python scripts/run_synthetic.py --csv out.csv   # dump per-tick telemetry

Visualization is now the Unity in-headset renderer (docs/UNITY_BRIDGE.md) or the
optional Rerun dashboard — the old MuJoCo offscreen GIF/window was removed with MuJoCo.

Exit code is 0 only if every trajectory passes on both arms.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from bimanual_teleop.vr.frames import SE3, SO3

# Make `src/` importable when run as a bare script (not `python -m`).
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from bimanual_teleop.arms.ik import ArmIK                       # noqa: E402
from bimanual_teleop.config import SIDES, load_rig             # noqa: E402
from bimanual_teleop.logging_utils import RateMeter, TelemetryRing, get_logger  # noqa: E402
from bimanual_teleop.viz.rerun_log import RerunLogger          # noqa: E402
from bimanual_teleop.vr.frames import quat_to_R, rotvec        # noqa: E402

log = get_logger("synthetic")

# --- trajectory + tolerance parameters ------------------------------------- #
FREQ = 0.5            # Hz of each oscillation (2.5 cycles in 5 s)
RAMP = 0.75           # s ease-in so the target never teleports (bounds velocity)
LINE_AMP = 0.07       # m
CIRCLE_R = 0.06       # m
ROLL_AMP = 0.50       # rad (~29°) about the tool axis
PITCH_AMP = 0.40      # rad about the wrist pitch axis
YAW_AMP = 0.40        # rad about the wrist yaw axis

# PASS thresholds (after the ease-in settles).
TOL_POS_CM = 2.0      # max wrist position error
TOL_ORI_DEG = 5.0     # max EE orientation error
# Max single-tick joint jump that still counts as continuous. A real flip is ~pi;
# the legitimate per-tick ceiling is iters*max_vel*dt (= 0.6 rad at the defaults), so
# 1.0 sits safely above legitimate motion and well below a flip.
TOL_FLIP_RAD = 1.0
WARMUP_S = 1.0        # ignore the ease-in transient when scoring error


def smoothstep(t: float, ramp: float = RAMP) -> float:
    if t >= ramp:
        return 1.0
    s = t / ramp
    return s * s * (3.0 - 2.0 * s)


def _axis_angle_R(axis: np.ndarray, angle: float) -> np.ndarray:
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


@dataclass
class Trajectory:
    """A scripted EE motion relative to the arm's home pose, in the arm base frame.
    `pos_fn(t)` is a position offset (m); `ang_fn(t)` is an angle (rad) about
    `axis_key` (one of the home tool/pitch/yaw axes). Ease-in is applied to both."""
    name: str
    pos_fn: Callable[[float], np.ndarray]
    ang_fn: Callable[[float], float] = lambda t: 0.0
    axis_key: str | None = None          # 'roll'|'pitch'|'yaw' or None (hold orientation)


def make_trajectories() -> list[Trajectory]:
    w = 2 * np.pi * FREQ
    return [
        Trajectory("line", lambda t: LINE_AMP * np.sin(w * t) * np.array([1.0, 0.0, 0.0])),
        Trajectory("circle", lambda t: CIRCLE_R * (np.cos(w * t) * np.array([1.0, 0.0, 0.0])
                                                   + np.sin(w * t) * np.array([0.0, 0.0, 1.0]))),
        Trajectory("roll", lambda t: np.zeros(3), lambda t: ROLL_AMP * np.sin(w * t), "roll"),
        Trajectory("pitch", lambda t: np.zeros(3), lambda t: PITCH_AMP * np.sin(w * t), "pitch"),
        Trajectory("yaw", lambda t: np.zeros(3), lambda t: YAW_AMP * np.sin(w * t), "yaw"),
    ]


class SyntheticArm:
    """One arm's IK + its home pose and tool/pitch/yaw axes (base frame)."""

    def __init__(self, rig: dict, side: str):
        self.side = side
        self.ik = ArmIK(rig, side)
        self.ik.reset()
        self.q0 = self.ik.q.copy()
        self.home_p = self.ik.fk_wrist().translation().copy()
        self.home_R = self.ik.fk_ee().rotation().as_matrix().copy()
        roll = self.ik._joint_axis_base(self.ik.joints[5])     # j6 tool/roll axis
        pitch = self.ik._joint_axis_base(self.ik.joints[4])    # j5 pitch axis
        yaw = np.cross(roll, pitch)
        self.axes = {"roll": roll / np.linalg.norm(roll),
                     "pitch": pitch / np.linalg.norm(pitch),
                     "yaw": yaw / np.linalg.norm(yaw)}
        self.base_R = quat_to_R(rig["arms"][side]["base_quat"])
        self.base_pos = np.asarray(rig["arms"][side]["base_pos"], float)

    def reset(self) -> None:
        self.ik.reset()

    def target(self, traj: Trajectory, t: float) -> SE3:
        e = smoothstep(t)
        p = self.home_p + e * traj.pos_fn(t)
        R = self.home_R
        if traj.axis_key is not None:
            R = _axis_angle_R(self.axes[traj.axis_key], e * traj.ang_fn(t)) @ self.home_R
        return SE3.from_rotation_and_translation(SO3.from_matrix(R), p)

    def to_world(self, p_base: np.ndarray, R_base: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.base_R @ p_base + self.base_pos, self.base_R @ R_base


@dataclass
class TrajResult:
    side: str
    name: str
    max_pos_cm: float = 0.0
    max_ori_deg: float = 0.0
    max_vel: float = 0.0
    max_jump: float = 0.0
    min_margin: float = 1e9
    limits_ok: bool = True
    peak_dq: np.ndarray = field(default_factory=lambda: np.zeros(6))

    @property
    def roll_joint(self) -> int:
        return int(np.argmax(np.abs(self.peak_dq)))

    def passed(self, max_vel_limit: float) -> bool:
        ok = (self.max_pos_cm <= TOL_POS_CM and self.max_ori_deg <= TOL_ORI_DEG
              and self.max_jump <= TOL_FLIP_RAD and self.max_vel <= max_vel_limit
              and self.limits_ok)
        if self.name == "roll":                    # pure roll must land on j6
            ok = ok and self.roll_joint == 5
        return ok


def run(args) -> int:
    rig = load_rig()
    # Match the loop rate to the IK's integration dt (rig control.arm_hz) so the
    # measured per-tick joint velocity is comparable to the enforced velocity limit.
    hz = float(args.hz) if args.hz else float(rig["control"]["arm_hz"])
    dt = 1.0 / hz
    n = int(args.seconds * hz)
    max_vel_limit = float(rig["ik"]["max_vel"])
    arms = {s: SyntheticArm(rig, s) for s in SIDES}
    trajectories = [t for t in make_trajectories() if not args.traj or t.name in args.traj]

    rr = RerunLogger(spawn=args.rerun, enabled=args.rerun)
    if args.rerun and not rr.enabled:
        log.warning("--rerun requested but rerun-sdk not installed (uv sync --extra telemetry)")
    telem = TelemetryRing(capacity=n * len(trajectories) * len(SIDES) + 10)
    rate = RateMeter()

    results: list[TrajResult] = []
    sim_t = 0.0
    for traj in trajectories:
        for a in arms.values():
            a.reset()
        res = {s: TrajResult(s, traj.name) for s in SIDES}
        q0 = {s: arms[s].ik.q.copy() for s in SIDES}
        qprev = {s: arms[s].ik.q.copy() for s in SIDES}
        for k in range(n):
            t = k / hz
            sim_t += dt
            rate.update(dt)
            for s, a in arms.items():
                tgt = a.target(traj, t)
                a.ik.solve(tgt)
                q = a.ik.q
                ach_p = a.ik.fk_wrist().translation()
                ach_R = a.ik.fk_ee().rotation().as_matrix()
                pos_err = float(np.linalg.norm(ach_p - tgt.translation()))
                ori_err = float(np.degrees(np.linalg.norm(rotvec(ach_R.T @ tgt.rotation().as_matrix()))))
                dq = q - qprev[s]
                vel = float(np.max(np.abs(dq)) * hz)
                jump = float(np.max(np.abs(dq)))
                margin = float(np.min(a.ik.limit_margins(q)))
                qprev[s] = q.copy()

                r = res[s]
                if t >= WARMUP_S:
                    r.max_pos_cm = max(r.max_pos_cm, pos_err * 100.0)
                    r.max_ori_deg = max(r.max_ori_deg, ori_err)
                r.max_vel = max(r.max_vel, vel)
                r.max_jump = max(r.max_jump, jump)
                r.min_margin = min(r.min_margin, margin)
                r.limits_ok = r.limits_ok and a.ik.within_limits(q)
                r.peak_dq = np.maximum(r.peak_dq, np.abs(q - q0[s]))

                telem.append(t=round(sim_t, 5), traj=traj.name, side=s, loop_hz=round(rate.hz, 1),
                             pos_err_cm=round(pos_err * 100, 4), ori_err_deg=round(ori_err, 4),
                             joint_vel=round(vel, 4), min_margin=round(margin, 4),
                             q=[round(float(x), 4) for x in q])
                if rr.enabled:
                    pw_a, Rw_a = a.to_world(ach_p, ach_R)
                    pw_c, Rw_c = a.to_world(tgt.translation(), tgt.rotation().as_matrix())
                    rr.set_time(sim_t)
                    rr.triad(f"world/{s}/ee_achieved", pw_a, Rw_a)
                    rr.triad(f"world/{s}/ee_commanded", pw_c, Rw_c, length=0.17)
                    rr.scalar(f"err/{s}/pos_cm", pos_err * 100)
                    rr.scalar(f"err/{s}/ori_deg", ori_err)
                    rr.scalar(f"err/{s}/joint_vel", vel)
                    rr.scalar(f"err/{s}/min_margin", margin)

        results.extend(res[s] for s in SIDES)

    ok = _report(results, max_vel_limit)
    if args.csv:
        telem.to_csv(args.csv)
        log.info("wrote telemetry %s (%d rows)", args.csv, len(telem))
    return 0 if ok else 1


def _report(results: list[TrajResult], max_vel_limit: float) -> bool:
    jn = ["j1", "j2", "j3", "j4", "j5", "j6"]
    print("\n  synthetic trajectory tracking (two-stage pink IK, no headset)")
    print("  " + "-" * 92)
    print(f"  {'side':5s} {'traj':7s} {'pos(cm)':>8s} {'ori(deg)':>8s} {'vel(r/s)':>8s} "
          f"{'jump':>6s} {'margin':>7s} {'lim':>4s} {'roll→':>6s}  result")
    print("  " + "-" * 92)
    all_ok = True
    for r in results:
        p = r.passed(max_vel_limit)
        all_ok = all_ok and p
        roll_tag = jn[r.roll_joint] if r.name == "roll" else ""
        print(f"  {r.side:5s} {r.name:7s} {r.max_pos_cm:8.2f} {r.max_ori_deg:8.2f} "
              f"{r.max_vel:8.2f} {r.max_jump:6.3f} {r.min_margin:7.3f} "
              f"{'OK' if r.limits_ok else 'BAD':>4s} {roll_tag:>6s}  {'PASS' if p else 'FAIL'}")
    print("  " + "-" * 92)
    print(f"  velocity limit = {max_vel_limit:.1f} rad/s | pos<{TOL_POS_CM}cm ori<{TOL_ORI_DEG}° "
          f"jump<{TOL_FLIP_RAD} | pure-roll must land on j6")
    print(f"\n  {'ALL TRAJECTORIES PASS ✓' if all_ok else 'SOME TRAJECTORIES FAILED ✗'}\n")
    return all_ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rerun", action="store_true", help="stream telemetry to the Rerun dashboard")
    ap.add_argument("--csv", metavar="PATH", default=None, help="dump per-tick telemetry CSV")
    ap.add_argument("--hz", type=float, default=None,
                    help="control loop rate (default: rig control.arm_hz, matching the IK dt)")
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--traj", nargs="*", default=None,
                    help="subset of: line circle roll pitch yaw (default all)")
    args = ap.parse_args()
    # back-compat fields the smoke test / old callers may set; rendering was removed.
    for k, v in dict(view=False, gif=None, no_gif=True, fps=15.0, width=0, height=0).items():
        if not hasattr(args, k):
            setattr(args, k, v)
    if getattr(args, "view", False):
        log.warning("--view (MuJoCo window) was removed with MuJoCo; use the Unity renderer "
                    "or --rerun. Running headless verify instead.")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
