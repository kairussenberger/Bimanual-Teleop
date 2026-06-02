"""Frame / axis sanity check — the #1 teleop bug is a sign/axis mismatch.

Commands one arm's EE to its neutral pose ± an offset along each base-frame axis,
applies the IK solution to the combined sim model, and writes snapshots so you can
SEE which world direction each base-frame axis maps to. Use this to set
mapping.r_base_from_vr_euler so "move my hand +X" moves the robot the right way.

    uv run python -m bimanual_teleop.tools.frame_check --side left --out /tmp/fc
"""
from __future__ import annotations

import argparse

import mink
import numpy as np

from ..arms.ik import ArmIK
from ..config import load_rig
from ..sim.sim_world import SimWorld


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--side", choices=["left", "right"], default="left")
    ap.add_argument("--offset", type=float, default=0.12, help="metres along each axis")
    ap.add_argument("--out", default="/tmp/frame_check")
    args = ap.parse_args()

    rig = load_rig()
    world = SimWorld(rig)
    ik = ArmIK(rig, args.side)
    T0 = ik.fk_ee()
    p0, R0 = T0.translation(), T0.rotation()

    cases = {
        "neutral": np.zeros(3),
        "plus_x": np.array([args.offset, 0, 0]),
        "plus_y": np.array([0, args.offset, 0]),
        "plus_z": np.array([0, 0, args.offset]),
    }
    for name, off in cases.items():
        ik.reset()
        target = mink.SE3.from_rotation_and_translation(R0, p0 + off)
        for _ in range(300):
            ik.solve(target)
        world.set_arm(args.side, ik.q)
        world.step(50)
        path = f"{args.out}_{args.side}_{name}.png"
        world.snapshot(path)
        ee = world.ee_pose(args.side)[:3, 3]
        print(f"{name:8s} base-target Δ={off.round(2)}  ->  world EE {ee.round(3)}  [{path}]")
    print("Compare plus_x/y/z snapshots to read off base→world axis mapping.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
