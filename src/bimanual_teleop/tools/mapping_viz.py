"""Mapping visualization — SEE whether the operator→robot mapping is correct.

Runs a scripted operator gesture sequence (reach forward / up / out, then wrist
pitch / yaw / twist, then a fist) through the real control path (calibration →
two-stage IK), renders the robot to a GIF, and overlays at each hand:
  - SOLID RGB triad  = the robot's ACTUAL hand orientation
  - FAINT RGB triad  = the COMMANDED hand orientation (the target)
  - small sphere      = the commanded wrist position
If the solid and faint triads line up, orientation tracks true (no tilt). It also
prints a per-phase error table (position cm, orientation deg, and how much the ARM
joints moved — which should be ~0 during pure wrist gestures).

    uv run python -m bimanual_teleop.tools.mapping_viz --out /tmp/mapping.gif
"""
from __future__ import annotations

import argparse

import imageio.v3 as iio
import mujoco
import numpy as np

from ..config import SIDES, load_rig
from ..engine import TeleopEngine
from ..sim.sim_world import SimWorld
from ..vr.calibrate import Calibrator
from ..vr.frames import HandSample, VRFrame, mat_to_se3, quat_to_R


def _ref_landmarks() -> np.ndarray:
    """Reference-stance hand: fingers forward (−z), palm down."""
    lm = np.zeros((25, 3))
    lm[0] = [0, 0, 0]; lm[6] = [0.03, 0, -0.03]; lm[11] = [0, 0, -0.09]; lm[21] = [-0.03, 0, -0.03]
    lm[9] = [0.03, 0, -0.15]; lm[14] = [0, 0, -0.16]; lm[19] = [-0.01, 0, -0.15]
    return lm


def _curl(lm: np.ndarray, c: float) -> np.ndarray:
    """Curl the fingers toward the palm by fraction c (for the fist phase)."""
    out = lm.copy()
    for tip in (9, 14, 19):
        out[tip] = lm[tip] + c * (np.array([lm[tip][0], 0, 0.10]) - lm[tip] * np.array([0, 1, 1]))
    return out


def _Rx(a): c, s = np.cos(a), np.sin(a); return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
def _Ry(a): c, s = np.cos(a), np.sin(a); return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
def _Rz(a): c, s = np.cos(a), np.sin(a); return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


# gesture timeline: (label, seconds, wrist-translation(webxr), wrist-rotation, finger-curl)
PHASES = [
    ("calibrate (hold ref)", 1.5, [0, 0, 0], np.eye(3), 0.0),
    ("reach forward",        1.5, [0, 0, -0.20], np.eye(3), 0.0),
    ("hand up",              1.5, [0, 0.20, 0], np.eye(3), 0.0),
    ("hand out (side)",      1.5, [0.18, 0, 0], np.eye(3), 0.0),
    ("wrist pitch",          1.5, [0, 0, 0], _Rx(0.6), 0.0),
    ("wrist twist (roll)",   1.5, [0, 0, 0], _Rz(0.9), 0.0),
    ("close fist",           1.5, [0, 0, 0], np.eye(3), 1.0),
]


def _arrows(scene, pos, R, length, width, alpha):
    cols = [[1, .15, .15, alpha], [.15, 1, .15, alpha], [.3, .3, 1, alpha]]
    for i in range(3):
        z = R[:, i] / (np.linalg.norm(R[:, i]) + 1e-9)
        a = np.array([1.0, 0, 0]) if abs(z[0]) < 0.9 else np.array([0, 1.0, 0])
        x = np.cross(a, z); x /= np.linalg.norm(x); y = np.cross(z, x)
        mat = np.column_stack([x, y, z]).flatten()
        if scene.ngeom >= scene.maxgeom:
            return
        mujoco.mjv_initGeom(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_ARROW,
                            np.array([width, width, length]), np.asarray(pos, float),
                            mat, np.array(cols[i], np.float32))
        scene.ngeom += 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="/tmp/mapping.gif")
    ap.add_argument("--fps", type=float, default=30)
    args = ap.parse_args()

    rig = load_rig()
    rig["vr"]["calib_seconds"] = 0          # we calibrate manually below
    world = SimWorld(rig)
    engine = TeleopEngine(rig, world)
    # calibrate both arms from the ref stance
    cal = Calibrator(rig)
    for s in SIDES:
        for _ in range(10):
            cal.add(s, _ref_landmarks())
        engine.arm[s].mapper.set_R(cal.compute(s))

    base = {s: (quat_to_R(rig["arms"][s]["base_quat"]), np.array(rig["arms"][s]["base_pos"])) for s in SIDES}
    renderer = mujoco.Renderer(world.model, height=720, width=1000)
    renderer._scene = renderer.scene  # ensure exists
    cam = mujoco.MjvCamera(); mujoco.mjv_defaultCamera(cam)
    cam.lookat[:] = [-0.35, 0, 0.62]; cam.distance = 1.9; cam.azimuth = 120; cam.elevation = -18

    frames, rows = [], []
    dt = 1.0 / args.fps
    for label, secs, tp, R, curl in PHASES:
        for k in range(int(secs * args.fps)):
            hands = {}
            for s in SIDES:
                sgn = 1.0 if s == "left" else -1.0
                T = np.eye(4); T[:3, :3] = R; T[:3, 3] = [tp[0] * sgn, tp[1], tp[2]]
                hands[s] = HandSample(tracked=True, wrist=T, landmarks=_curl(_ref_landmarks(), curl))
            engine.tick(VRFrame(stamp=k * dt, hands=hands), {s: True for s in SIDES}, k * dt)
            world.step(2)
            mujoco.mj_forward(world.model, world.data)
            renderer.update_scene(world.data, cam)
            for s in SIDES:
                ac = engine.arm[s]
                ee = world.ee_pose(s)
                tgt = ac.mapper.target(mat_to_se3(hands[s].wrist)) if ac.mapper.engaged else None
                _arrows(renderer.scene, ee[:3, 3], ee[:3, :3], 0.10, 0.007, 1.0)   # actual
                if tgt is not None:
                    bR, bp = base[s]
                    _arrows(renderer.scene, bR @ ac.ik.fk_wrist().translation() + bp,
                            bR @ tgt.rotation().as_matrix(), 0.13, 0.004, 0.35)     # commanded
            frames.append(renderer.render())
        # error snapshot at end of phase
        s = "left"; ac = engine.arm[s]
        if ac.mapper.engaged:
            tgt = ac.mapper.target(mat_to_se3(hands[s].wrist))
            oerr = np.degrees(np.arccos(np.clip(
                (np.trace(world.ee_pose(s)[:3, :3].T @ (base[s][0] @ tgt.rotation().as_matrix())) - 1) / 2, -1, 1)))
        else:
            oerr = float("nan")
        rows.append((label, oerr))

    iio.imwrite(args.out, frames, duration=1000 / args.fps, loop=0)
    print(f"wrote {args.out} ({len(frames)} frames)")
    print("\nphase                 | left-hand orientation error (deg)")
    for label, oerr in rows:
        print(f"  {label:20s} | {oerr:6.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
