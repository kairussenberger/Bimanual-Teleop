#!/usr/bin/env python
"""Synthetic-input mode (spec Section 7 + bring-up steps 2-3): drive the EE targets
with SCRIPTED trajectories and NO headset, to isolate the IK from frames/tracking.

This is the J6 isolation test. We drive each arm's two-stage mink IK directly with
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
    uv run python scripts/run_synthetic.py                 # headless verify + GIF + table (default)
    uv run python scripts/run_synthetic.py --no-gif        # fastest: verify + table only
    uv run python scripts/run_synthetic.py --rerun         # also stream to the Rerun dashboard
    uv run mjpython scripts/run_synthetic.py --view        # live MuJoCo window (macOS needs mjpython)

The GIF / live window overlay, at each hand: a SOLID triad = the robot's ACHIEVED
EE frame, a FAINT longer triad = the COMMANDED target frame. When tracking is good
they sit on top of each other; drift between them is the bug, made visible.

Exit code is 0 only if every trajectory passes on both arms.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import mink
import mujoco
import numpy as np

# Make `src/` importable when run as a bare script (not `python -m`).
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from bimanual_teleop.arms.ik import ArmIK                       # noqa: E402
from bimanual_teleop.config import SIDES, load_rig             # noqa: E402
from bimanual_teleop.logging_utils import RateMeter, TelemetryRing, get_logger  # noqa: E402
from bimanual_teleop.sim.sim_world import SimWorld             # noqa: E402
from bimanual_teleop.viz import overlay                        # noqa: E402
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
TOL_FLIP_RAD = 0.5    # max single-tick joint jump (a flip would be ~pi)
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

    def target(self, traj: Trajectory, t: float) -> mink.SE3:
        e = smoothstep(t)
        p = self.home_p + e * traj.pos_fn(t)
        R = self.home_R
        if traj.axis_key is not None:
            R = _axis_angle_R(self.axes[traj.axis_key], e * traj.ang_fn(t)) @ self.home_R
        return mink.SE3.from_rotation_and_translation(mink.SO3.from_matrix(R), p)

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
    hz = args.hz
    dt = 1.0 / hz
    n = int(args.seconds * hz)
    max_vel_limit = float(rig["ik"]["max_vel"])
    arms = {s: SyntheticArm(rig, s) for s in SIDES}
    trajectories = [t for t in make_trajectories() if not args.traj or t.name in args.traj]

    # Rendering / telemetry are best-effort and optional.
    world = renderer = cam = None
    gif_frames: list[np.ndarray] = []
    want_render = (args.gif is not None) and not args.view
    if want_render:
        try:
            world = SimWorld(rig)
            renderer = mujoco.Renderer(world.model, height=args.height, width=args.width)
            cam = mujoco.MjvCamera()
            mujoco.mjv_defaultCamera(cam)
            cam.lookat[:] = (0.0, -0.045, 0.46)         # the two hanging hands
            cam.distance, cam.azimuth, cam.elevation = 1.45, 230.0, -10.0
        except Exception as e:                     # GL not available -> verify only
            log.warning("offscreen rendering unavailable (%s) — verifying without a GIF", e)
            world = renderer = None

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
        render_every = max(1, int(hz / args.fps))
        for k in range(n):
            t = k / hz
            sim_t += dt
            rate.update(dt)
            overlays = []
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

                if renderer is not None:
                    # push the IK solution into the (kinematic) world for display
                    for adr, val in zip(world.info.arm_qadr(s), q):
                        world.data.qpos[adr] = val
                    pw_a, Rw_a = a.to_world(ach_p, ach_R)
                    pw_c, Rw_c = a.to_world(tgt.translation(), tgt.rotation().as_matrix())
                    overlays.append((pw_a, Rw_a, 0.13, 0.009, 1.0))     # achieved (solid)
                    overlays.append((pw_c, Rw_c, 0.20, 0.005, 0.4))     # commanded (faint)

            if renderer is not None and k % render_every == 0:
                mujoco.mj_forward(world.model, world.data)
                renderer.update_scene(world.data, cam)
                for (pos, R, length, width, alpha) in overlays:
                    overlay.triad(renderer.scene, pos, R, length, width, alpha)
                gif_frames.append(renderer.render())

        results.extend(res[s] for s in SIDES)

    ok = _report(results, max_vel_limit)

    if gif_frames and args.gif:
        try:
            import imageio.v3 as iio
            out = Path(args.gif)
            out.parent.mkdir(parents=True, exist_ok=True)
            iio.imwrite(str(out), gif_frames, duration=1000 / args.fps, loop=0)
            log.info("wrote %s (%d frames) — solid triad = achieved, faint = commanded",
                     out, len(gif_frames))
        except Exception as e:
            log.warning("could not write GIF: %s", e)
    if args.csv:
        telem.to_csv(args.csv)
        log.info("wrote telemetry %s (%d rows)", args.csv, len(telem))
    if renderer is not None:
        renderer.close()
    return 0 if ok else 1


def _report(results: list[TrajResult], max_vel_limit: float) -> bool:
    jn = ["j1", "j2", "j3", "j4", "j5", "j6"]
    print("\n  synthetic trajectory tracking (two-stage mink IK, no headset)")
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


def run_view(args) -> int:
    """Live MuJoCo window (needs mjpython on macOS). Loops the trajectories with the
    commanded-vs-achieved triad overlay."""
    import time

    import mujoco.viewer
    rig = load_rig()
    hz = args.hz
    world = SimWorld(rig)
    arms = {s: SyntheticArm(rig, s) for s in SIDES}
    trajectories = [t for t in make_trajectories() if not args.traj or t.name in args.traj]
    log.info("live view: SOLID triad = achieved EE, FAINT = commanded. Ctrl-C to quit.")
    with mujoco.viewer.launch_passive(world.model, world.data) as v:
        t0 = time.time()
        ti = 0
        while v.is_running():
            traj = trajectories[ti % len(trajectories)]
            t = time.time() - t0
            if t > args.seconds:
                t0 = time.time(); ti += 1
                for a in arms.values():
                    a.reset()
                continue
            overlays = []
            for s, a in arms.items():
                tgt = a.target(traj, t)
                a.ik.solve(tgt)
                q = a.ik.q
                for adr, val in zip(world.info.arm_qadr(s), q):
                    world.data.qpos[adr] = val
                pw_a, Rw_a = a.to_world(a.ik.fk_wrist().translation(), a.ik.fk_ee().rotation().as_matrix())
                pw_c, Rw_c = a.to_world(tgt.translation(), tgt.rotation().as_matrix())
                overlays += [(pw_a, Rw_a, 0.12, 0.007, 1.0), (pw_c, Rw_c, 0.17, 0.004, 0.4)]
            mujoco.mj_forward(world.model, world.data)
            if getattr(v, "user_scn", None) is not None:
                v.user_scn.ngeom = 0
                for (pos, R, length, width, alpha) in overlays:
                    overlay.triad(v.user_scn, pos, R, length, width, alpha)
            v.sync()
            time.sleep(1.0 / hz)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--view", action="store_true", help="live MuJoCo window (mjpython)")
    ap.add_argument("--rerun", action="store_true", help="stream telemetry to the Rerun dashboard")
    ap.add_argument("--gif", metavar="PATH", default=str(REPO_ROOT / "out" / "run_synthetic.gif"),
                    help="output GIF path (default out/run_synthetic.gif)")
    ap.add_argument("--no-gif", action="store_true", help="skip rendering (fastest verify)")
    ap.add_argument("--csv", metavar="PATH", default=None, help="dump per-tick telemetry CSV")
    ap.add_argument("--hz", type=float, default=120.0)
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--fps", type=float, default=15.0, help="GIF frame rate")
    ap.add_argument("--width", type=int, default=720)
    ap.add_argument("--height", type=int, default=540)
    ap.add_argument("--traj", nargs="*", default=None,
                    help="subset of: line circle roll pitch yaw (default all)")
    args = ap.parse_args()
    if args.no_gif:
        args.gif = None
    return run_view(args) if args.view else run(args)


if __name__ == "__main__":
    raise SystemExit(main())
