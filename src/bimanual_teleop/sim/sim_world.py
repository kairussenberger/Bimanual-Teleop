"""The simulation world: owns MjModel/MjData, applies arm + hand commands, and
renders (interactive passive viewer via mjpython, or offscreen snapshots).

This is the sim backend behind the same command interface the real robot uses:
- arm command  = 6 joint targets (rad), set on the YAM position actuators
- hand command = {orca_joint: degrees}, mapped to the 17 ORCA actuators (→rad)

Run directly to eyeball the rig:
    uv run mjpython -m bimanual_teleop.sim.sim_world            # interactive viewer
    uv run python    -m bimanual_teleop.sim.sim_world --snap out.png   # offscreen PNG
    uv run mjpython -m bimanual_teleop.sim.sim_world --demo     # arms+fingers sweep
"""
from __future__ import annotations

import math
import time

import mujoco
import numpy as np

from ..config import SIDES, load_rig
from ..hands.joint_map import load_hand_config
from .model import SimInfo, build_model, set_neutral


class SimWorld:
    def __init__(self, rig: dict | None = None):
        self.rig = rig or load_rig()
        self.info: SimInfo = build_model(self.rig)
        self.model = self.info.model
        self.data = mujoco.MjData(self.model)
        self.hand_neutral = {s: load_hand_config(self.rig["hands"][s]["model_name"])[0]
                             for s in SIDES}
        set_neutral(self.info, self.data, self.rig, self.hand_neutral)

    # --- command application (the sim "driver") --------------------------- #
    def set_arm(self, side: str, q: np.ndarray) -> None:
        for aid, val in zip(self.info.arm_act[side], q):
            self.data.ctrl[aid] = float(val)

    def set_hand(self, side: str, joints_deg: dict[str, float]) -> None:
        neutral = self.hand_neutral[side]
        for aid, joint in self.info.hand_act[side]:
            self.data.ctrl[aid] = math.radians(joints_deg.get(joint, neutral.get(joint, 0.0)))

    def step(self, n: int = 1) -> None:
        mujoco.mj_step(self.model, self.data, nstep=n)

    def ee_pose(self, side: str) -> np.ndarray:
        """Current EE site pose (4x4) in world frame."""
        sid = self.info.ee_site[side]
        T = np.eye(4)
        T[:3, 3] = self.data.site_xpos[sid]
        T[:3, :3] = self.data.site_xmat[sid].reshape(3, 3)
        return T

    # --- rendering -------------------------------------------------------- #
    def render_rgb(self, width: int = 1100, height: int = 800, azimuth: float = 90,
                   elevation: float = -15, distance: float = 1.9, lookat=(0.0, 0.25, 1.25)):
        if getattr(self, "_renderer", None) is None or self._rsize != (width, height):
            if getattr(self, "_renderer", None) is not None:
                self._renderer.close()
            self._renderer = mujoco.Renderer(self.model, height=height, width=width)
            self._rsize = (width, height)
            self._cam = mujoco.MjvCamera()
            mujoco.mjv_defaultCamera(self._cam)
        self._cam.lookat[:] = lookat
        self._cam.distance, self._cam.azimuth, self._cam.elevation = distance, azimuth, elevation
        mujoco.mj_forward(self.model, self.data)
        self._renderer.update_scene(self.data, self._cam)
        return self._renderer.render()

    def snapshot(self, path: str, **kw) -> str:
        import imageio.v3 as iio
        iio.imwrite(path, self.render_rgb(**kw))
        return path


def _demo_targets(t: float, rig: dict, hand_neutral: dict):
    """A canned sweep (no VR): arms trace small circles, fingers open/close."""
    c = 0.5 - 0.5 * math.cos(t * 1.2)
    arms, hands = {}, {}
    for side in SIDES:
        q = list(rig["arms"][side]["neutral_q"])
        q[0] += 0.4 * math.sin(t * 0.8) * (1 if side == "left" else -1)
        q[3] += 0.5 * math.sin(t * 1.1)
        q[5] += 0.6 * math.sin(t * 1.3)
        arms[side] = q
        d = dict(hand_neutral[side])
        for f in ("index", "middle", "ring", "pinky"):
            d[f"{f}_mcp"] = 90.0 * c
            d[f"{f}_pip"] = 95.0 * c
        d["thumb_mcp"] = 60.0 * c
        hands[side] = d
    return arms, hands


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--snap", metavar="PATH", help="render one offscreen PNG and exit")
    ap.add_argument("--demo", action="store_true", help="sweep arms+fingers in the viewer")
    ap.add_argument("--azimuth", type=float, default=90)
    ap.add_argument("--elevation", type=float, default=-15)
    args = ap.parse_args()

    world = SimWorld()
    if args.snap:
        print("wrote", world.snapshot(args.snap, azimuth=args.azimuth, elevation=args.elevation))
        return 0

    import mujoco.viewer
    with mujoco.viewer.launch_passive(world.model, world.data) as v:
        t0 = time.time()
        while v.is_running():
            if args.demo:
                t = time.time() - t0
                arms, hands = _demo_targets(t, world.rig, world.hand_neutral)
                for side in SIDES:
                    world.set_arm(side, arms[side])
                    world.set_hand(side, hands[side])
            world.step(2)
            v.sync()
            time.sleep(1 / 120)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
